"""Orchestration : un dessin → un verdict, ou une entrée à revoir.

**Lecture seule** — n'écrit jamais de métadonnées.  L'écriture est un second
temps, confirmé par l'utilisateur (voir :mod:`~media_restorer.engines.
signatures.tags`) : même partition que
:mod:`~media_restorer.engines.triage`/:mod:`~media_restorer.engines.
duplicates` — classer est réversible, écrire modifie les fichiers.

Idempotence : un dessin déjà étiqueté « Dessinateur » (y compris la feuille
explicite « sans signature ») est sauté par défaut — sans quoi chaque
relance reposerait la même question.  ``force=True`` l'ignore, utile après un
changement de seuils ou de modèle d'empreinte.

Cache de candidats
-------------------
Localiser (OWL-ViT) et embarquer (SigLIP) chaque dessin coûte cher — plus
qu'un simple passage d'empreinte sur des dessins entiers (deux modèles
``transformers`` par image, voir :mod:`~media_restorer.engines.signatures.
descriptors`).  Le résultat par dessin (boîte, chemin du crop, empreinte) est
donc mis en cache, invalidé par ``(taille, mtime)`` — même motif que
:mod:`~media_restorer.engines.duplicates.embeddings`.  Ce n'est pas qu'un
confort de relance : c'est ce qui permet à la revue de recomparer la file
restante après chaque signature ajoutée sans tout relocaliser/réembarquer à
chaque étape.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from media_restorer.app_settings import app_settings
from media_restorer.engines.signatures import descriptors as _descriptors
from media_restorer.engines.signatures import matching
from media_restorer.engines.signatures.library import LibraryEntry, list_entries
from media_restorer.engines.signatures.location import Box, locate_signature
from media_restorer.engines.signatures.tags import read_artist
from media_restorer.engines.triage import iter_images
from media_restorer.image_io import imread_oriented

ProgressCallback = Callable[[int, int], None]
#: Signature de :func:`scan_corpus` — injectable pour les tests (voir
#: ``extensions/signatures/gui.py``, motif ``scan_fn`` déjà utilisé pour
#: ``doublons``/``pre_classement``).
ScanFn = Callable[..., "list[ScanOutcome]"]

#: Aucune zone de signature localisée sur le dessin — demande un crop manuel.
REASON_NO_LOCATION = "no_location"
#: Une zone a été localisée, mais rien d'assez proche dans la bibliothèque.
REASON_NO_MATCH = "no_match"
#: Plusieurs dessinateurs plausibles — demande un choix.
REASON_AMBIGUOUS = "ambiguous"

_CACHE_FILENAME = "signatures_candidates.json"
_CANDIDATES_DIRNAME = "signatures_candidates"


@dataclass(frozen=True)
class ScanOutcome:
    """Résultat du scan d'UN dessin — soit étiquetable, soit à revoir."""

    path: Path
    artist: str | None = None
    reason: str | None = None
    candidates: tuple[matching.Candidate, ...] = ()
    box: Box | None = None
    crop_path: Path | None = None

    @property
    def needs_review(self) -> bool:
        return self.artist is None


def _stamp(path: Path) -> list[int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return [st.st_size, st.st_mtime_ns]


def _cache_path() -> Path:
    return Path(app_settings().fileName()).with_name(_CACHE_FILENAME)


def _candidates_dir() -> Path:
    directory = Path(app_settings().fileName()).with_name(_CANDIDATES_DIRNAME)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_candidate_cache(path: Path | None = None) -> dict[str, dict]:
    """Cache ``{chemin_dessin: {stamp, box, crop, vector}}``, déjà validé.

    Une entrée dont le fichier a changé (ou disparu) est silencieusement
    écartée, comme :func:`~media_restorer.engines.duplicates.embeddings.load_cache`.
    """
    path = path or _cache_path()
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(blob, dict):
        return {}
    valides: dict[str, dict] = {}
    for chemin, entree in blob.items():
        if not isinstance(entree, dict):
            continue
        if _stamp(Path(chemin)) != entree.get("stamp"):
            continue
        valides[chemin] = entree
    return valides


def save_candidate_cache(cache: dict[str, dict], path: Path | None = None) -> None:
    path = path or _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache), encoding="utf-8")


def _crop_path_for(path: Path) -> Path:
    # Nom déterministe (sha1, pas hash() — randomisé par process en Python) :
    # une même image redonne le même chemin de crop d'un lancement à l'autre.
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()
    return _candidates_dir() / f"{digest}.png"


def scan_corpus(
    root: Path | str,
    *,
    recursive: bool = True,
    detector,
    embedder,
    thresholds: dict | None = None,
    force: bool = False,
    on_progress: ProgressCallback | None = None,
) -> list[ScanOutcome]:
    """Scanne *root* : localise, compare, classe chaque dessin. Lecture seule.

    *thresholds* — ``min_score`` (détection), ``confident_threshold`` et
    ``ambiguous_margin`` (comparaison), voir :mod:`~media_restorer.engines.
    signatures.matching`. Absents : valeurs par défaut de ces modules.
    """
    thresholds = thresholds or {}
    library = list_entries()
    library_vectors = (
        _descriptors.embed_many([e.path for e in library], embedder)
        if library
        else np.zeros((0, 0))
    )

    cache = load_candidate_cache()
    outcomes: list[ScanOutcome] = []
    paths = iter_images(root, recursive=recursive)
    total = len(paths)

    for index, path in enumerate(paths, start=1):
        if not force and read_artist(path) is not None:
            if on_progress is not None:
                on_progress(index, total)
            continue
        outcomes.append(
            _scan_one(
                path,
                detector=detector,
                embedder=embedder,
                library=library,
                library_vectors=library_vectors,
                thresholds=thresholds,
                cache=cache,
            )
        )
        if on_progress is not None:
            on_progress(index, total)

    save_candidate_cache(cache)
    return outcomes


def _scan_one(
    path: Path,
    *,
    detector,
    embedder,
    library: list[LibraryEntry],
    library_vectors: np.ndarray,
    thresholds: dict,
    cache: dict[str, dict],
) -> ScanOutcome:
    stamp = _stamp(path)
    cached = cache.get(str(path))
    box: Box | None
    vector: np.ndarray | None
    crop_path: Path | None

    if cached is not None and cached.get("stamp") == stamp:
        box = Box(**cached["box"]) if cached.get("box") else None
        crop_path = Path(cached["crop"]) if cached.get("crop") else None
        vector = np.array(cached["vector"]) if cached.get("vector") else None
    else:
        image = imread_oriented(path)
        box = (
            locate_signature(
                image, detector=detector, min_score=thresholds.get("min_score", 0.05)
            )
            if image is not None
            else None
        )
        if box is None:
            cache[str(path)] = {"stamp": stamp, "box": None, "crop": None, "vector": None}
            return ScanOutcome(path=path, reason=REASON_NO_LOCATION)

        crop = box.crop(image)
        crop_path = _crop_path_for(path)
        cv2.imwrite(str(crop_path), crop)
        vector = _descriptors.embed_one(crop_path, embedder)
        cache[str(path)] = {
            "stamp": stamp,
            "box": {"xmin": box.xmin, "ymin": box.ymin, "xmax": box.xmax, "ymax": box.ymax},
            "crop": str(crop_path),
            "vector": vector.tolist(),
        }

    if box is None or vector is None:
        return ScanOutcome(path=path, reason=REASON_NO_LOCATION)

    candidates = matching.rank(vector, library, library_vectors) if library else []
    verdict = matching.classify(
        candidates,
        confident_threshold=thresholds.get(
            "confident_threshold", matching.DEFAULT_CONFIDENT_THRESHOLD
        ),
        ambiguous_margin=thresholds.get("ambiguous_margin", matching.DEFAULT_AMBIGUOUS_MARGIN),
    )
    if verdict.regime == matching.CONFIDENT:
        return ScanOutcome(
            path=path,
            artist=verdict.artist,
            candidates=verdict.candidates,
            box=box,
            crop_path=crop_path,
        )
    reason = REASON_AMBIGUOUS if verdict.regime == matching.AMBIGUOUS else REASON_NO_MATCH
    return ScanOutcome(
        path=path, reason=reason, candidates=verdict.candidates, box=box, crop_path=crop_path
    )

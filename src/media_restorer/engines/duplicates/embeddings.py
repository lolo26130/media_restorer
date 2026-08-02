"""Empreintes sémantiques — la seule voie vers les variantes redessinées.

Les régimes R1 et R2 se résolvent géométriquement : une homographie relie les
deux images, et c'est elle qui explique la différence.  **R3 est d'une autre
nature** — un dessin redessiné n'a aucune transformation qui le relie à son
modèle, seulement une ressemblance de contenu.  Il faut donc un descripteur
*appris*, et accepter que le résultat soit une présomption, non une preuve.

L'incertitude assumée
---------------------
La littérature place **DINOv2 devant CLIP** en similarité fine d'images — mais
sur des *photographies*.  Sur des caricatures redessinées, rien ne le garantit.
D'où le catalogue : le modèle est un **paramètre**, et
:mod:`~media_restorer.engines.duplicates.benchmark` tranche par la mesure, sur
les verdicts rendus par l'utilisateur.

SSCD, pourtant état de l'art en détection de copie (jeu DISC21), est
volontairement **absent** : il est entraîné à retrouver des copies
*transformées*, ce que la voie géométrique fait déjà mieux et de façon
explicable.  Il n'a rien à offrir pour un redessin.

Coût mesuré sur cette machine
-----------------------------
=========== ======== ================== ==================
Modèle      Params   Cabrol (8 693 im.) Corpus ×5
=========== ======== ================== ==================
ViT-S/14    22 M     3,5 min CPU        17,4 min CPU
            (1,5 min GPU)               (7,7 min GPU)
ViT-B/14    86 M     11,4 min CPU       57,2 min CPU
            (4,6 min GPU)               (23,0 min GPU)
=========== ======== ================== ==================

Le GPU n'apporte que **2,2 à 2,5×** : c'est un confort, pas un prérequis.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

ProgressCallback = Callable[[int, int], None]

#: Signature d'un extracteur.  Le rappel de progression fait partie du
#: **contrat** : tout extracteur, y compris un substitut de test, doit
#: l'accepter.  Le déduire par un ``except TypeError`` avalerait une vraie
#: erreur de type survenue à l'intérieur de l'extracteur.
Embedder = Callable[..., np.ndarray]      # (paths, *, on_progress=None) -> (n, dim)


@dataclass(frozen=True)
class EmbeddingModel:
    """Un modèle d'empreinte sélectionnable.

    Attributs
    ---------
    key : str
        Identifiant stable — persisté dans les ``QSettings`` **et** dans la clé
        du cache : le renommer invaliderait tous les calculs déjà faits.
    repo : str
        Dépôt Hugging Face.  Les poids sont téléchargés au premier usage ;
        **aucun entraînement** n'est requis.
    notes : str
        Ce que l'on sait — et ce que l'on ignore — de ce modèle sur des dessins.
    """

    key: str
    title: str
    repo: str
    kind: str                 # "dinov2" | "clip" | "siglip"
    cost_hint: str
    notes: str


EMBEDDING_MODELS: tuple[EmbeddingModel, ...] = (
    EmbeddingModel(
        key="dinov2_small", title="DINOv2 ViT-S (rapide)",
        repo="facebook/dinov2-small", kind="dinov2",
        cost_hint="≈ 3,5 min CPU / 1,5 min GPU sur 8 700 images",
        notes=("Auto-supervisé, réputé le meilleur en similarité fine d'images "
               "— mais sur des photographies. À éprouver sur des dessins."),
    ),
    EmbeddingModel(
        key="dinov2_base", title="DINOv2 ViT-B (plus fin, plus lent)",
        repo="facebook/dinov2-base", kind="dinov2",
        cost_hint="≈ 11 min CPU / 4,6 min GPU sur 8 700 images",
        notes="Même famille que ViT-S, quatre fois plus de paramètres.",
    ),
    EmbeddingModel(
        key="clip_base", title="CLIP ViT-B/32",
        repo="openai/clip-vit-base-patch32", kind="clip",
        cost_hint="≈ 4 min CPU sur 8 700 images",
        notes=("Espace partagé avec le texte : ouvre la porte à une recherche "
               "par mots. Donné en retrait de DINOv2 en similarité purement "
               "visuelle."),
    ),
    EmbeddingModel(
        key="siglip_base", title="SigLIP ViT-B/16",
        repo="google/siglip-base-patch16-224", kind="siglip",
        cost_hint="≈ 9 min CPU sur 8 700 images",
        notes="Successeur de CLIP, meilleur en classification à peu d'exemples.",
    ),
)

EMBEDDING_MODELS_BY_KEY = {m.key: m for m in EMBEDDING_MODELS}
DEFAULT_MODEL = "dinov2_small"


def build_embedder(model_key: str = DEFAULT_MODEL, device: str = "cpu",
                   batch_size: int = 8) -> Embedder:
    """Construit l'extracteur d'empreintes du modèle demandé.

    Les poids sont téléchargés au premier usage.  L'import de ``torch`` et de
    ``transformers`` est **différé** jusqu'ici : le lancement de l'application
    ne paie pas le coût d'une bibliothèque dont l'utilisateur ne se servira
    peut-être jamais (convention du projet).
    """
    modele = EMBEDDING_MODELS_BY_KEY.get(model_key)
    if modele is None:
        raise ValueError(f"modèle d'empreinte inconnu : {model_key!r}")

    import torch
    from transformers import AutoImageProcessor, AutoModel

    processeur = AutoImageProcessor.from_pretrained(modele.repo)
    reseau = AutoModel.from_pretrained(modele.repo).to(device).eval()

    def embed(paths: Sequence[Path], *,
              on_progress: ProgressCallback | None = None) -> np.ndarray:
        from PIL import Image

        vecteurs: list[np.ndarray] = []
        for debut in range(0, len(paths), batch_size):
            lot = paths[debut:debut + batch_size]
            images = []
            for chemin in lot:
                try:
                    with Image.open(chemin) as im:
                        im.draft("RGB", (512, 512))
                        images.append(im.convert("RGB").copy())
                except Exception:
                    # Un fichier abîmé ne doit pas interrompre le lot : on lui
                    # substitue une image neutre, son empreinte sera écartée
                    # par le seuil comme n'importe quelle image sans rapport.
                    images.append(Image.new("RGB", (224, 224), (255, 255, 255)))
            entrees = processeur(images=images, return_tensors="pt").to(device)
            with torch.no_grad():
                if modele.kind == "dinov2":
                    # Jeton de classe : le résumé global que DINOv2 apprend.
                    vect = reseau(**entrees).last_hidden_state[:, 0]
                else:
                    # CLIP et SigLIP exposent directement leur tête image.
                    vect = reseau.get_image_features(**entrees)
            vecteurs.append(vect.float().cpu().numpy())
            if on_progress is not None:
                on_progress(min(debut + batch_size, len(paths)), len(paths))
        return _normalise(np.concatenate(vecteurs, axis=0))

    return embed


def _normalise(v: np.ndarray) -> np.ndarray:
    """Normalise chaque ligne : le produit scalaire devient un cosinus."""
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return v / n


# ---------------------------------------------------------------------------
# Cache — recalculer onze minutes d'empreintes à chaque essai serait absurde
# ---------------------------------------------------------------------------

_CACHE_FILENAME = "duplicates_embeddings.json"
_FORMAT_VERSION = 1


def cache_path() -> Path:
    """Emplacement du cache, à côté du ``.ini`` du ``QSettings``."""
    from media_restorer.app_settings import app_settings

    return Path(app_settings().fileName()).with_name(_CACHE_FILENAME)


def _stamp(path: Path) -> list[int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return [st.st_size, st.st_mtime_ns]


def load_cache(model_key: str, path: Path | None = None) -> dict[Path, np.ndarray]:
    """Empreintes en cache pour *model_key*, déjà validées contre les fichiers.

    L'invalidation porte sur **trois** choses : la version du format, la clé du
    modèle, et l'empreinte ``(taille, mtime)`` de chaque fichier.  Oublier la
    clé du modèle ferait comparer des vecteurs DINOv2 à des vecteurs CLIP —
    silencieusement, et avec des résultats absurdes.
    """
    path = path or cache_path()
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(blob, dict) or blob.get("version") != _FORMAT_VERSION:
        return {}
    if blob.get("model") != model_key:
        return {}

    valides: dict[Path, np.ndarray] = {}
    for brut, entree in (blob.get("entries") or {}).items():
        fichier = Path(brut)
        if not isinstance(entree, dict) or _stamp(fichier) != entree.get("stamp"):
            continue
        try:
            valides[fichier] = np.asarray(entree["vector"], dtype=np.float32)
        except (KeyError, TypeError, ValueError):
            continue
    return valides


def save_cache(model_key: str, vectors: dict[Path, np.ndarray],
               path: Path | None = None) -> None:
    """Écrit les empreintes de *model_key*, en remplaçant le contenu."""
    path = path or cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    entrees: dict[str, dict] = {}
    for chemin, vecteur in vectors.items():
        stamp = _stamp(chemin)
        if stamp is None:
            continue
        entrees[str(chemin)] = {"stamp": stamp,
                                "vector": [float(x) for x in np.asarray(vecteur).ravel()]}
    path.write_text(
        json.dumps({"version": _FORMAT_VERSION, "model": model_key, "entries": entrees}),
        encoding="utf-8",
    )


def embed_corpus(
    paths: Sequence[Path],
    embedder: Embedder,
    model_key: str,
    *,
    use_cache: bool = True,
    cache_file: Path | None = None,
    on_progress: ProgressCallback | None = None,
) -> np.ndarray:
    """Empreintes de *paths*, en ne recalculant que ce qui manque.

    Renvoie une matrice ``(len(paths), dim)`` dans l'ordre donné.
    """
    en_cache = load_cache(model_key, cache_file) if use_cache else {}
    manquants = [p for p in paths if p not in en_cache]

    if manquants:
        nouveaux = embedder(manquants, on_progress=on_progress)
        for chemin, vecteur in zip(manquants, nouveaux):
            en_cache[chemin] = np.asarray(vecteur, dtype=np.float32)
        if use_cache:
            save_cache(model_key, en_cache, cache_file)

    return np.stack([en_cache[p] for p in paths])

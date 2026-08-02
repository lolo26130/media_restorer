"""Inventaire des prises de vue : quel JPEG vient de quel RAW ?

Les fichiers RAW (``.NEF``…) sont des **sauvegardes** en vue d'un développement
ultérieur, pas des images à traiter.  Ils restent donc **hors de toutes les
autres chaînes** — :data:`RAW_SUFFIXES` est délibérément **disjoint** de
:data:`~media_restorer.engines.triage.IMAGE_SUFFIXES`, et un test le verrouille.

Ce module ne fait qu'un **inventaire**, en quatre catégories :

===================== ==========================================================
``pairs``             RAW et JPEG appariés de façon fiable
``raw_only``          RAW **sans JPEG** — les JPEG à produire
``jpeg_only``         JPEG sans RAW (scans, exports, autres sources)
``unconfirmed``       appariés par le seul nom de fichier — **à vérifier**
===================== ==========================================================

Sur le corpus de référence (1 877 NEF, 8 422 JPEG) : 1 440 paires sûres (76,7 %),
391 RAW orphelins (20,8 %), 46 à confirmer, 6 759 JPEG sans RAW.  Scan complet en
59 s.

.. warning::

   **Ne jamais passer ``-fast2`` à exiftool.**  Cette option s'arrête avant les
   MakerNotes du constructeur, où vit ``ShutterCount`` — la clé sûre.  Une
   première mesure de ce corpus a ainsi annoncé **0 paire au lieu de 1 440**,
   sans la moindre erreur ni le moindre avertissement.  ``-fast`` (niveau 1)
   conserve les MakerNotes et suffit à écarter les blocs inutiles.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from media_restorer.engines.shots import keys as _keys
from media_restorer.engines.shots.keys import (
    EXIF_FIELDS,
    MATCH_NAME,
    is_reliable,
    keys_of,
    match_method,
)

#: Extensions RAW reconnues.  **Disjoint de** ``IMAGE_SUFFIXES`` : ces fichiers
#: n'entrent dans aucune autre chaîne du projet.
RAW_SUFFIXES = frozenset({".nef", ".cr2", ".cr3", ".arw", ".dng", ".raf", ".orf",
                          ".rw2", ".pef", ".srw"})
#: Extensions dérivées, celles qu'on cherche à rattacher à un RAW.
DERIVED_SUFFIXES = frozenset({".jpg", ".jpeg"})

#: Répertoires jamais parcourus (corbeille DigiKam).
EXCLUDED_DIR_NAMES = frozenset({".dtrash"})

#: Taille des lots passés à ``exiftool`` : au-delà, la ligne de commande devient
#: trop longue ; en deçà, on paie trop de démarrages de processus.
BATCH = 300

ExiftoolRunner = Callable[[Sequence[str]], str]
ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class ShotPair:
    """Un RAW et son dérivé, avec **la façon dont ils ont été appariés**.

    La méthode fait partie du résultat : c'est elle qui permet de présenter à
    part les appariements douteux, au lieu de les noyer parmi les sûrs.
    """

    raw: Path
    derived: Path
    method: str
    meta: dict = field(default_factory=dict)

    @property
    def reliable(self) -> bool:
        return is_reliable(self.method)

    def describe(self) -> str:
        """Phrase justifiant l'appariement — affichée sous l'aperçu."""
        modele = self.meta.get("Model") or "appareil inconnu"
        details = [modele]
        if self.meta.get("ShutterCount"):
            details.append(f"déclenchement {self.meta['ShutterCount']}")
        if self.meta.get("DateTimeOriginal"):
            details.append(str(self.meta["DateTimeOriginal"]))
        rattachement = _keys.MATCH_TITLES.get(self.method, self.method)
        return " · ".join(details) + f"  —  apparié par {rattachement}"


@dataclass(frozen=True)
class ShotInventory:
    """Le résultat complet d'un inventaire.

    Les quatre catégories sont **exhaustives et disjointes** : tout fichier
    rencontré apparaît dans une, et une seule.
    """

    pairs: list[ShotPair]
    raw_only: list[Path]
    jpeg_only: list[Path]
    unconfirmed: list[ShotPair]

    @property
    def n_raw(self) -> int:
        return len(self.pairs) + len(self.raw_only) + len(self.unconfirmed)

    @property
    def n_derived(self) -> int:
        return len(self.pairs) + len(self.jpeg_only) + len(self.unconfirmed)

    def summary(self) -> str:
        """Résumé d'une ligne, pour la barre d'état."""
        apparies = len(self.pairs)
        part = 100 * apparies / self.n_raw if self.n_raw else 0.0
        morceaux = [f"{self.n_raw} RAW", f"{self.n_derived} JPEG",
                    f"{part:.1f} % appariés"]
        if self.raw_only:
            morceaux.append(f"{len(self.raw_only)} RAW sans JPEG")
        if self.unconfirmed:
            morceaux.append(f"{len(self.unconfirmed)} à confirmer")
        return " · ".join(morceaux)


def default_runner(args: Sequence[str]) -> str:
    """Lance ``exiftool`` et renvoie sa sortie standard.

    Runner de production ; les tests en injectent un faux et ne lancent jamais
    le binaire (même motif que :func:`media_restorer.digikam_tags.default_runner`).
    """
    exe = shutil.which("exiftool")
    if exe is None:
        raise RuntimeError(
            "exiftool introuvable — installez-le (paquet « libimage-exiftool-perl ») "
            "pour inventorier les correspondances RAW/JPEG."
        )
    result = subprocess.run([exe, *args], capture_output=True, text=True, timeout=300)
    return result.stdout


def iter_files(root: Path | str, *, recursive: bool = True
               ) -> tuple[list[Path], list[Path]]:
    """``(raws, derives)`` trouvés sous *root*, triés."""
    root = Path(root)
    candidats = root.rglob("*") if recursive else root.glob("*")
    raws: list[Path] = []
    derives: list[Path] = []
    for p in candidats:
        if not p.is_file() or EXCLUDED_DIR_NAMES.intersection(p.parts):
            continue
        suffixe = p.suffix.lower()
        if suffixe in RAW_SUFFIXES:
            raws.append(p)
        elif suffixe in DERIVED_SUFFIXES:
            derives.append(p)
    return sorted(raws), sorted(derives)


def read_metadata(paths: Sequence[Path], runner: ExiftoolRunner | None = None,
                  *, on_progress: ProgressCallback | None = None
                  ) -> dict[Path, dict]:
    """Métadonnées des fichiers, lues par lots.

    Emploie ``-fast`` et **jamais** ``-fast2`` — voir l'avertissement en tête de
    module.  Un lot illisible est ignoré : mieux vaut un inventaire partiel
    qu'aucun inventaire.
    """
    runner = runner or default_runner
    resultat: dict[Path, dict] = {}
    for debut in range(0, len(paths), BATCH):
        lot = paths[debut:debut + BATCH]
        args = ["-j", "-q", "-q", "-fast",
                *(f"-{champ}" for champ in EXIF_FIELDS),
                *(str(p) for p in lot)]
        try:
            for enregistrement in json.loads(runner(args)):
                resultat[Path(enregistrement["SourceFile"])] = enregistrement
        except Exception:
            pass
        if on_progress is not None:
            on_progress(min(debut + BATCH, len(paths)), len(paths))
    return resultat


def scan_shots(
    root: Path | str,
    *,
    recursive: bool = True,
    runner: ExiftoolRunner | None = None,
    on_progress: ProgressCallback | None = None,
    on_stage: Callable[[str], None] | None = None,
) -> ShotInventory:
    """Inventorie les correspondances RAW ↔ JPEG sous *root*.

    Chaque RAW est apparié par sa **meilleure** clé disponible : sûre d'abord,
    horaire ensuite, nom en dernier recours — et dans ce dernier cas le résultat
    part dans ``unconfirmed``, jamais dans ``pairs``.
    """
    raws, derives = iter_files(root, recursive=recursive)
    _annonce(on_stage, f"Lecture des métadonnées de {len(raws) + len(derives)} fichiers…")

    meta = read_metadata([*raws, *derives], runner, on_progress=on_progress)
    _annonce(on_stage, "Appariement…")

    # Un index par clé, tous niveaux de fiabilité confondus.  L'ordre de
    # priorité est appliqué à la lecture, pas à la construction.
    index: dict[tuple, list[Path]] = defaultdict(list)
    for chemin in derives:
        for cle in keys_of(chemin, meta.get(chemin, {})):
            index[cle].append(chemin)

    pairs: list[ShotPair] = []
    unconfirmed: list[ShotPair] = []
    raw_only: list[Path] = []
    utilises: set[Path] = set()

    for brut in raws:
        infos = meta.get(brut, {})
        apparie = None
        for cle in keys_of(brut, infos):          # ordre de fiabilité décroissante
            candidats = [c for c in index.get(cle, [])
                         if not contradicts(infos, meta.get(c, {}))]
            if candidats:
                apparie = ShotPair(raw=brut, derived=candidats[0],
                                   method=match_method(cle), meta=dict(infos))
                break
        if apparie is None:
            raw_only.append(brut)
            continue
        utilises.add(apparie.derived)
        (pairs if apparie.reliable else unconfirmed).append(apparie)

    jpeg_only = [p for p in derives if p not in utilises]
    return ShotInventory(pairs=pairs, raw_only=raw_only,
                         jpeg_only=jpeg_only, unconfirmed=unconfirmed)


def contradicts(a: Mapping[str, object], b: Mapping[str, object]) -> bool:
    """Les métadonnées prouvent-elles que ce sont **deux prises différentes** ?

    Si les deux fichiers portent une clé fiable et que ces clés **diffèrent**,
    on tient une preuve positive : ce ne sont pas le même déclenchement.  Il
    serait alors absurde de retomber sur le nom de fichier — cela reviendrait à
    ignorer la preuve dont on dispose.

    C'est exactement le cas réel ``330ND800/DSC_6769.nef`` face à
    ``329ND800/DSC_6769.jpg`` : mêmes noms, compteurs de déclenchements
    différents, prises sans rapport.  Sans ce garde-fou, l'inventaire
    affirmerait une correspondance fausse.

    Deux fichiers dont l'un au moins n'a pas de clé fiable ne se contredisent
    pas : ils sont simplement muets, et l'appariement par le nom reste un
    candidat légitime — à confirmer.
    """
    for fabrique in (_keys.sure_key, _keys.time_key):
        ka, kb = fabrique(a), fabrique(b)
        if ka is not None and kb is not None and ka != kb:
            return True
    return False


def _annonce(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)

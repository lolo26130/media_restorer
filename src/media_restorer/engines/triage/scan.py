"""Parcours d'un corpus et agrégation des signaux de tri.

Là où :mod:`~media_restorer.engines.triage.signals` mesure **une** image, ce
module en parcourt un répertoire entier et résume les résultats en
distributions — la forme sous laquelle on décide si un axe de tri sépare
réellement le corpus ou non.

Injection de dépendances
------------------------
:func:`scan_directory` prend la fonction de mesure en paramètre (*measure*)
plutôt que de l'importer en dur : les tests y injectent un substitut instantané
et vérifient le parcours, l'échantillonnage et l'agrégation sans jamais décoder
d'image (convention du ``CLAUDE.md`` racine, comme les ``_xxx_factory`` des
workers Qt).

*on_progress* joue le même rôle pour l'avancement : le cœur ne connaît pas Qt,
c'est à l'appelant de transformer ce rappel en barre de progression s'il en veut
une.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Iterable, Sequence

from media_restorer.engines.triage.signals import (
    INK_DENSITIES,
    ORIENTATIONS,
    RESOLUTIONS,
    SUPPORTS,
    ImageSignals,
    TooLarge,
    measure_image,
)

# Mêmes extensions que le reste du projet (voir media_restorer.exif_info).
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}

# Répertoires jamais parcourus : corbeille de DigiKam, et les copies « _original »
# qu'exiftool laisse à côté des fichiers qu'il a modifiés seraient comptées deux fois.
EXCLUDED_DIR_NAMES = {".dtrash"}

Measurer = Callable[[Path], ImageSignals]
ProgressCallback = Callable[[int, int], None]


def iter_images(root: Path | str, *, recursive: bool = True) -> list[Path]:
    """Fichiers image de *root*, corbeilles exclues, dans un ordre stable.

    L'ordre est trié : deux exécutions successives échantillonnent le même
    sous-ensemble pour une graine donnée, ce que l'ordre du système de fichiers
    ne garantirait pas.
    """
    root = Path(root)
    candidates = root.rglob("*") if recursive else root.glob("*")
    return sorted(
        p for p in candidates
        if p.is_file()
        and p.suffix.lower() in IMAGE_SUFFIXES
        and not EXCLUDED_DIR_NAMES.intersection(p.parts)
    )


@dataclass(frozen=True)
class ScanResult:
    """Signaux mesurés, et ce qui n'a pas été mesuré — avec la raison.

    Distinguer les deux écarts n'est pas cosmétique : une image **écartée par
    le plafond** est un choix de l'utilisateur, un fichier **illisible** est un
    incident qu'il voudra peut-être aller regarder.  Les additionner ferait
    annoncer une perte là où il n'y a qu'un filtre.
    """

    signals: list[ImageSignals]
    skipped_large: list[Path]
    unreadable: list[Path]

    @property
    def found(self) -> int:
        """Nombre de fichiers image rencontrés, toutes issues confondues."""
        return len(self.signals) + len(self.skipped_large) + len(self.unreadable)


def scan_directory(
    root: Path | str,
    *,
    recursive: bool = True,
    sample: int | None = None,
    seed: int = 0,
    max_megapixels: float | None = None,
    measure: Measurer | None = None,
    on_progress: ProgressCallback | None = None,
) -> ScanResult:
    """Mesure les images de *root* et renvoie un :class:`ScanResult`.

    *sample* limite le travail à un tirage aléatoire de cette taille (graine
    *seed*, donc reproductible) — indispensable pour calibrer des seuils sans
    parcourir tout un corpus.  ``None`` mesure tout.

    *max_megapixels* écarte les images trop lourdes **sans les décoder** (voir
    :func:`~media_restorer.engines.triage.signals.measure_image`) ; elles sont
    listées à part dans le résultat.

    Les fichiers illisibles sont **ignorés sans interrompre le parcours** : un
    corpus numérisé en contient toujours quelques-uns, et un tri qui
    s'arrêterait au premier fichier tronqué serait inutilisable.  Ils sont
    listés eux aussi, pour que l'appelant puisse signaler une hécatombe plutôt
    que de la passer sous silence.
    """
    paths = iter_images(root, recursive=recursive)
    if sample is not None and sample < len(paths):
        paths = sorted(random.Random(seed).sample(paths, sample))

    # Le plafond est lié à la vraie mesure, qui seule lit l'en-tête du fichier.
    # Un substitut injecté (tests) reste une fonction à un seul paramètre : s'il
    # doit appliquer un plafond, c'est à l'appelant de le lui lier lui-même.
    if measure is None:
        measure = partial(measure_image, max_megapixels=max_megapixels)

    total = len(paths)
    signals: list[ImageSignals] = []
    skipped_large: list[Path] = []
    unreadable: list[Path] = []

    for index, path in enumerate(paths, start=1):
        try:
            signals.append(measure(path))
        except TooLarge:
            skipped_large.append(path)
        except Exception:
            unreadable.append(path)
        if on_progress is not None:
            on_progress(index, total)
    return ScanResult(signals=signals, skipped_large=skipped_large, unreadable=unreadable)


def summarise(signals: Sequence[ImageSignals]) -> dict[str, dict[str, int]]:
    """Distributions par axe : ``{axe: {classe: effectif}}``.

    Toutes les classes possibles apparaissent, **y compris à zéro** : une classe
    absente est une information (« aucun dessin en couleur sur papier neutre »),
    pas une ligne à faire disparaître du tableau.
    """
    axes: dict[str, tuple[Iterable[str], Callable[[ImageSignals], str]]] = {
        "orientation": (ORIENTATIONS, lambda s: s.orientation),
        "densité d'encre": (INK_DENSITIES, lambda s: s.ink_density),
        "résolution": (RESOLUTIONS, lambda s: s.resolution_class),
        "support": (SUPPORTS, lambda s: s.support),
    }
    summary: dict[str, dict[str, int]] = {}
    for axis, (classes, extract) in axes.items():
        counts = Counter(extract(s) for s in signals)
        summary[axis] = {cls: counts.get(cls, 0) for cls in classes}
    return summary


def format_summary(summary: dict[str, dict[str, int]], *, width: int = 40) -> str:
    """Rend :func:`summarise` en texte, avec un histogramme en barres.

    Sans Qt ni dépendance de tracé : sert aussi bien depuis un script que depuis
    une future action d'extension qui l'afficherait dans un widget de texte.
    """
    lines: list[str] = []
    for axis, counts in summary.items():
        total = sum(counts.values())
        lines.append(f"— {axis}")
        for label, count in counts.items():
            share = 100 * count / total if total else 0.0
            bar = "█" * round(width * share / 100)
            lines.append(f"    {label:<30} {share:5.1f} %  ({count:>5})  {bar}")
        lines.append("")
    return "\n".join(lines).rstrip()

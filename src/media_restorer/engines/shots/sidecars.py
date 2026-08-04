"""Un RAW a-t-il déjà été développé ?

Les logiciels de développement RAW ne modifient pas le fichier d'origine : ils
déposent à côté un **fichier annexe** décrivant les réglages appliqués.  Sa
présence est donc la trace qu'un travail a eu lieu.

La distinction compte, et elle est **actionnable** : parmi les RAW sans JPEG,

* **jamais développé** → il reste tout à faire ;
* **déjà développé** → le JPEG a existé puis a été déplacé, renommé ou
  supprimé — c'est un problème de rangement, pas de traitement.

Conventions de nommage
----------------------
Le fichier annexe reprend le **nom complet** du RAW, extension comprise :
``DSC_4149.NEF.xmp`` et non ``DSC_4149.xmp``.  Vérifié sur le corpus de
référence : 1 050 NEF sur 1 877 portent un annexe suivant cette convention,
**aucun** ne suit la convention par radical seul.

============== ===================================
darktable      ``<nom>.NEF.xmp``
RawTherapee    ``<nom>.NEF.pp3``, ``<nom>.NEF.out.pp3``
============== ===================================

La comparaison est **insensible à la casse** : le corpus mêle ``.NEF`` et
``.nef``, ``.xmp`` et ``.XMP``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

#: Extensions d'annexe reconnues, en minuscules.
SIDECAR_SUFFIXES = (".xmp", ".pp3", ".out.pp3")


def sidecar_candidates(raw: Path) -> list[str]:
    """Noms d'annexe possibles pour *raw*, en minuscules.

    Le nom **complet** du RAW sert de base : ``DSC_1.NEF`` donne
    ``dsc_1.nef.xmp``, jamais ``dsc_1.xmp``.
    """
    base = raw.name.lower()
    return [base + suffixe for suffixe in SIDECAR_SUFFIXES]


def index_sidecars(paths: Iterable[Path]) -> set[tuple[Path, str]]:
    """Index ``(répertoire, nom en minuscules)`` des annexes rencontrées.

    Un index plutôt qu'un accès disque par RAW : sur un corpus de 10 000
    fichiers déjà parcouru une fois, interroger le système de fichiers une
    seconde fois pour chaque RAW coûterait inutilement cher.
    """
    index: set[tuple[Path, str]] = set()
    for chemin in paths:
        nom = chemin.name.lower()
        if any(nom.endswith(suffixe) for suffixe in SIDECAR_SUFFIXES):
            index.add((chemin.parent, nom))
    return index


def is_developed(raw: Path, index: set[tuple[Path, str]] | None = None) -> bool:
    """Un fichier annexe accompagne-t-il *raw* ?

    Avec *index*, la réponse ne coûte rien.  Sans lui, on interroge le disque —
    pratique pour un appel isolé, à éviter dans une boucle.
    """
    if index is not None:
        return any((raw.parent, nom) in index for nom in sidecar_candidates(raw))
    for suffixe in SIDECAR_SUFFIXES:
        if (raw.parent / (raw.name + suffixe)).exists():
            return True
        # Casse mêlée dans le corpus : .XMP existe aussi.
        if (raw.parent / (raw.name + suffixe.upper())).exists():
            return True
    return False


def iter_sidecars(root: Path | str, *, recursive: bool = True) -> list[Path]:
    """Tous les fichiers annexes sous *root*."""
    root = Path(root)
    candidats = root.rglob("*") if recursive else root.glob("*")
    return sorted(
        p for p in candidats
        if p.is_file() and any(p.name.lower().endswith(s) for s in SIDECAR_SUFFIXES)
    )

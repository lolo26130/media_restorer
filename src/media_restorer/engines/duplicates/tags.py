"""Enregistrement des doublons dans les étiquettes DigiKam.

Arborescence produite ::

    media_restorer/Doublons/Groupe 042
    media_restorer/Doublons/Représentant
    media_restorer/Doublons/Inclus dans un autre

Un groupe devient donc filtrable d'un clic dans le gestionnaire d'étiquettes, et
le représentant — le membre de plus haute résolution — est désigné
explicitement : c'est celui qu'on garde si l'on décide un jour d'élaguer.

.. warning::

   :data:`owns` ne revendique que la branche ``Doublons``.  Une appartenance
   plus large effacerait en silence les repères
   (:mod:`media_restorer.landmarks`) et le pré-classement
   (:mod:`media_restorer.engines.triage.tags`), puisque l'écriture reconstruit
   chaque champ en entier — voir l'avertissement en tête de
   :mod:`media_restorer.digikam_tags`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

from media_restorer import digikam_tags as _tags
from media_restorer.engines.duplicates.groups import DuplicateGraph, Group

#: Branche réservée à la détection de doublons.
BRANCH = "Doublons"

#: Prédicat d'appartenance — cette branche, et elle seule.
owns = _tags.branch_owner(BRANCH)

LEAF_REPRESENTATIVE = "Représentant"
LEAF_INCLUDED = "Inclus dans un autre"
#: Réservée aux variantes CONFIRMÉES à la revue.  Une présomption non
#: validée n'a rien à faire dans les métadonnées d'un fonds patrimonial.
LEAF_VARIANT = "Variante"

ProgressCallback = Callable[[int, int], None]


def group_label(index: int) -> str:
    """Libellé d'un groupe — numéroté sur trois chiffres, donc trié à l'affichage.

    Auto-suffisant : ``dc:Subject`` et ``IPTC:Keywords`` étant plats, DigiKam n'y
    recopie que la feuille.  « Groupe 042 » y reste compréhensible, « 042 » non.
    """
    return f"Groupe {index:03d}"


def tag_paths_for(label: str, *, representative: bool = False,
                  included: bool = False, variant: bool = False) -> list[list[str]]:
    """Chemins d'étiquettes d'une image, selon son rôle dans le graphe."""
    chemins = [[_tags.ROOT, BRANCH, label]]
    if representative:
        chemins.append([_tags.ROOT, BRANCH, LEAF_REPRESENTATIVE])
    if included:
        chemins.append([_tags.ROOT, BRANCH, LEAF_INCLUDED])
    if variant:
        chemins.append([_tags.ROOT, BRANCH, LEAF_VARIANT])
    return chemins


def write_graph(
    graph: DuplicateGraph,
    *,
    runner: _tags.ExiftoolRunner | None = None,
    on_progress: ProgressCallback | None = None,
    on_image: Callable[[Path], None] | None = None,
) -> tuple[int, list[tuple[Path, str]]]:
    """Écrit tout le graphe dans les métadonnées.  Renvoie ``(écrites, échecs)``.

    Une image verrouillée ou illisible n'interrompt pas la campagne : elle est
    consignée et le parcours continue — sur des milliers de fichiers, s'arrêter
    au premier incident rendrait l'outil inutilisable.
    """
    taches = list(_plan(graph))
    total = len(taches)
    echecs: list[tuple[Path, str]] = []
    ecrites = 0

    for index, (chemin, chemins_tags) in enumerate(taches, start=1):
        if on_image is not None:
            on_image(chemin)
        try:
            _tags.write_tags(chemin, chemins_tags, owns=owns, runner=runner)
            ecrites += 1
        except Exception as exc:
            echecs.append((chemin, str(exc)))
        if on_progress is not None:
            on_progress(index, total)
    return ecrites, echecs


def confirmed_variants(graph: DuplicateGraph) -> set[Path]:
    """Images des variantes **confirmées à la revue**, et d'elles seules.

    Une variante simplement proposée par le modèle n'est pas étiquetée : elle
    n'est qu'une présomption, et l'écrire dans le fichier lui donnerait une
    autorité qu'elle n'a pas.
    """
    from media_restorer.engines.duplicates import verdicts as _verdicts

    confirmees: set[Path] = set()
    for paire in graph.uncertain:
        verdict = _verdicts.verdict_for(paire.a, paire.b)
        if verdict is not None and verdict.confirmed:
            confirmees.update((paire.a, paire.b))
    return confirmees


def _plan(graph: DuplicateGraph) -> Iterable[tuple[Path, list[list[str]]]]:
    """Une entrée par image à étiqueter, étiquettes déjà résolues.

    Les images incluses dans une autre reçoivent leur marque **en plus** de leur
    éventuelle appartenance à un groupe : les deux informations sont
    indépendantes, une image peut être à la fois membre d'un groupe et détail
    d'une troisième.
    """
    incluses: set[Path] = set()
    for paire in graph.inclusions:
        # La plus petite couverture désigne l'image contenue.
        a_dans_b = paire.merit.couverture_a_dans_b or 0.0
        b_dans_a = paire.merit.couverture_b_dans_a or 0.0
        incluses.add(paire.a if a_dans_b < b_dans_a else paire.b)

    par_image: dict[Path, list[list[str]]] = {}
    for numero, groupe in enumerate(graph.groups, start=1):
        label = group_label(numero)
        for membre in groupe.members:
            par_image.setdefault(membre, []).extend(
                tag_paths_for(label, representative=(membre == groupe.representative))
            )
    for chemin in incluses:
        par_image.setdefault(chemin, []).append(
            [_tags.ROOT, BRANCH, LEAF_INCLUDED]
        )
    for chemin in confirmed_variants(graph):
        par_image.setdefault(chemin, []).append(
            [_tags.ROOT, BRANCH, LEAF_VARIANT]
        )
    return par_image.items()


def read_membership(path: Path | str, *,
                    runner: _tags.ExiftoolRunner | None = None) -> list[str]:
    """Étiquettes de doublon déjà portées par *path* (feuilles seules)."""
    runner = runner or _tags.default_runner
    raw = _tags.read_raw(path, runner)
    return [parts[-1] for parts in _tags.read_tag_paths(raw, owns)]


def clear(path: Path | str, *, runner: _tags.ExiftoolRunner | None = None) -> None:
    """Retire toute étiquette de doublon de *path*, sans toucher au reste.

    Utile pour rejouer une campagne : les étiquettes des repères et du
    pré-classement, elles, sont préservées — c'est tout l'intérêt de
    l'appartenance par branche.
    """
    _tags.write_tags(path, [], owns=owns, runner=runner)

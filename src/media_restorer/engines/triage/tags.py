"""Écriture du pré-classement dans les étiquettes DigiKam.

Traduit un :class:`~media_restorer.engines.triage.signals.ImageSignals` en
chemins d'étiquettes, puis délègue l'écriture à
:mod:`media_restorer.digikam_tags` — la logique de fusion non destructive n'est
donc écrite qu'une fois, partagée avec :mod:`media_restorer.landmarks`.

Arborescence produite ::

    media_restorer/Tri/Orientation/Portrait
    media_restorer/Tri/Densité d'encre/Encre moyenne
    media_restorer/Tri/Couleur d'encre/Trait noir
    media_restorer/Tri/Papier/Papier jauni
    media_restorer/Tri/Résolution/6-20 Mpx

La branche :data:`BRANCH` isole ce module de tous les autres écrivains
d'étiquettes.  :data:`owns` ne revendique qu'elle : sans cette restriction, une
écriture de pré-classement effacerait les repères posés par
:mod:`media_restorer.landmarks`, et réciproquement — voir l'avertissement en
tête de :mod:`media_restorer.digikam_tags`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from media_restorer import digikam_tags as _tags
from media_restorer.engines.triage.criteria import CRITERIA, Criterion
from media_restorer.engines.triage.signals import ImageSignals

#: Branche réservée au pré-classement sous la racine de l'application.
BRANCH = "Tri"

#: Prédicat d'appartenance — cette branche, et elle seule.
owns = _tags.branch_owner(BRANCH)


def tag_paths(
    signals: ImageSignals, criteria: Sequence[Criterion] = CRITERIA
) -> list[list[str]]:
    """Chemins d'étiquettes décrivant *signals* selon *criteria*.

    Un chemin par critère retenu.  L'ordre suit celui de *criteria*, lui-même
    stabilisé par :func:`~media_restorer.engines.triage.criteria.select`.
    """
    return [
        [_tags.ROOT, BRANCH, criterion.branch, criterion.extract(signals)]
        for criterion in criteria
    ]


def write_signals(
    path: Path | str,
    signals: ImageSignals,
    criteria: Sequence[Criterion] = CRITERIA,
    *,
    runner: _tags.ExiftoolRunner | None = None,
) -> None:
    """Écrit le pré-classement de *signals* dans les métadonnées de *path*.

    Les étiquettes étrangères — celles de l'utilisateur comme celles des
    repères — sont préservées.  Un second appel remplace le classement
    précédent au lieu de s'y ajouter : reclasser une image avec d'autres seuils
    ne laisse donc pas d'étiquettes périmées derrière lui.
    """
    _tags.write_tags(path, tag_paths(signals, criteria), owns=owns, runner=runner)


def read_classification(
    path: Path | str, *, runner: _tags.ExiftoolRunner | None = None
) -> dict[str, str]:
    """Relit le classement enregistré : ``{branche: classe}``.

    Renvoie un dictionnaire vide si l'image ne porte pas de pré-classement —
    aucune distinction volontaire avec « fichier sans métadonnées » : dans les
    deux cas il n'y a rien à afficher.
    """
    runner = runner or _tags.default_runner
    raw = _tags.read_raw(path, runner)
    return {
        parts[2]: parts[3]
        for parts in _tags.read_tag_paths(raw, owns)
        if len(parts) == 4
    }

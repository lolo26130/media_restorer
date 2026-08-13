"""Enregistrement du dessinateur reconnu dans les étiquettes DigiKam.

Arborescence produite ::

    media_restorer/Dessinateur/<Nom>
    media_restorer/Dessinateur/(sans signature)

Même gabarit que :mod:`~media_restorer.engines.triage.tags` (branche
``Tri``) et :mod:`~media_restorer.engines.duplicates.tags` (branche
``Doublons``) : :data:`owns` ne revendique QUE la branche ``Dessinateur`` —
une appartenance plus large effacerait en silence les autres écrivains
(repères, pré-classement, doublons), l'écriture reconstruisant chaque champ
en entier (voir l'avertissement en tête de
:mod:`media_restorer.digikam_tags`).

La feuille ``(sans signature)`` marque un dessin explicitement passé à la
revue sans signature visible — même logique que « Repère ignoré » dans
:mod:`media_restorer.landmarks` — pour ne jamais reposer la question à
chaque relance (voir l'idempotence de
:mod:`~media_restorer.engines.signatures.pipeline`).
"""
from __future__ import annotations

from pathlib import Path

from media_restorer import digikam_tags as _tags

#: Branche réservée au classement par signature.
BRANCH = "Dessinateur"

#: Prédicat d'appartenance — cette branche, et elle seule.
owns = _tags.branch_owner(BRANCH)

#: Feuille réservée à un dessin explicitement sans signature visible.
NO_SIGNATURE = "(sans signature)"


def write_artist(
    path: Path | str, artist: str, *, runner: _tags.ExiftoolRunner | None = None
) -> None:
    """Étiquette *path* du dessinateur *artist* (remplace une étiquette précédente)."""
    _tags.write_tags(path, [[_tags.ROOT, BRANCH, artist]], owns=owns, runner=runner)


def write_no_signature(
    path: Path | str, *, runner: _tags.ExiftoolRunner | None = None
) -> None:
    """Marque *path* comme explicitement sans signature visible."""
    _tags.write_tags(
        path, [[_tags.ROOT, BRANCH, NO_SIGNATURE]], owns=owns, runner=runner
    )


def read_artist(
    path: Path | str, *, runner: _tags.ExiftoolRunner | None = None
) -> str | None:
    """Dessinateur déjà enregistré pour *path*, ou ``None`` si aucun.

    Renvoie aussi :data:`NO_SIGNATURE` si elle a été explicitement posée — au
    même titre qu'un vrai nom : c'est ce qui rend le scan idempotent (voir
    :mod:`~media_restorer.engines.signatures.pipeline`).
    """
    runner = runner or _tags.default_runner
    raw = _tags.read_raw(path, runner)
    paths = _tags.read_tag_paths(raw, owns)
    return paths[0][-1] if paths else None

"""Structures de données du pipeline Vectorise — indépendantes de GUDHI/Qt."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Stroke:
    """Un trait continu — la trace ininterrompue d'un coup de crayon.

    Paramètres
    ----------
    points : np.ndarray
        ``(N, 2)`` float32, coordonnées image ``(x, y)``, dans l'ordre de
        parcours du trait.
    widths : np.ndarray
        ``(N,)`` float32, demi-épaisseur locale en pixels à chaque point.
    intensity : np.ndarray
        ``(N,)`` float32 dans ``[0, 1]``, noirceur locale à chaque point
        (0 = papier, 1 = noir plein) — sert de proxy à la pression du trait.
    """

    points: np.ndarray
    widths: np.ndarray
    intensity: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.points)
        if len(self.widths) != n or len(self.intensity) != n:
            raise ValueError(
                f"points/widths/intensity de tailles incohérentes : "
                f"{len(self.points)}/{len(self.widths)}/{len(self.intensity)}"
            )


@dataclass
class StrokeSet:
    """Un candidat de retraçage complet — un jeu cohérent de traits.

    Paramètres
    ----------
    strokes : list[Stroke]
        Les traits composant ce candidat.
    pencil_width_px : float
        Largeur de crayon mesurée (médiane des demi-épaisseurs ×2) pour ce
        candidat, en pixels.
    score : float
        Plausibilité topologique dans ``[0, 1]`` — voir
        :mod:`~media_restorer.engines.vectorise.topology` pour son calcul à
        partir de la stabilité de l'homologie persistante.
    label : str
        Description courte destinée à l'utilisateur (liste déroulante de la
        GUI), ex. ``"Ø 0,9 mm — 4 traits (score 0.82)"``.
    """

    strokes: list[Stroke] = field(default_factory=list)
    pencil_width_px: float = 0.0
    score: float = 0.0
    label: str = ""

    @property
    def n_points(self) -> int:
        """Nombre total de points, tous traits confondus."""
        return sum(len(s.points) for s in self.strokes)

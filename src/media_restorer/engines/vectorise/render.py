"""Reconstruction raster — tracés + texture → image ressemblant à l'originale."""
from __future__ import annotations

import cv2
import numpy as np

from media_restorer.engines.vectorise.types import StrokeSet

_DEFAULT_PAPER_COLOR = 245
_DEFAULT_MAX_WIDTH_CHANGES = 4
_DEFAULT_GRAIN_STRENGTH = 0.5


def _simplify_widths(widths: np.ndarray, max_segments: int) -> list[tuple[int, int, float]]:
    """Segmente *widths* en au plus *max_segments* tronçons à largeur constante.

    Un dessinateur ne change pas la pression en continu à chaque pixel — au
    plus quelques fois par trait.  Fusion récursive par division au point de
    plus grand écart local (façon Ramer-Douglas-Peucker, mais appliquée au
    profil de largeur plutôt qu'à la géométrie du trait) : le segment de plus
    forte variance est coupé à son point le plus atypique, jusqu'à respecter
    la borne.  Retourne une liste de ``(start, end, largeur)`` (bornes
    incluses, indices dans *widths*).
    """
    n = len(widths)
    if n == 0:
        return []
    segments = [(0, n - 1)]
    while len(segments) < max_segments:
        worst_idx, worst_var, split_at = -1, -1.0, -1
        for idx, (a, b) in enumerate(segments):
            if b - a < 2:
                continue
            seg = widths[a:b + 1]
            var = float(seg.var())
            if var <= worst_var:
                continue
            local_dev = np.abs(seg - seg.mean())
            split = a + int(np.argmax(local_dev))
            if a < split < b:
                worst_idx, worst_var, split_at = idx, var, split
        if worst_idx < 0:
            break
        a, b = segments.pop(worst_idx)
        segments.insert(worst_idx, (split_at + 1, b))
        segments.insert(worst_idx, (a, split_at))
    segments.sort()
    return [(a, b, float(widths[a:b + 1].mean())) for a, b in segments]


def _tile_texture(texture: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Tuile *texture* (2D) pour couvrir intégralement *shape* = (h, w)."""
    h, w = shape
    th, tw = texture.shape[:2]
    reps_y, reps_x = -(-h // th), -(-w // tw)
    return np.tile(texture, (reps_y, reps_x))[:h, :w]


def vectorise(
    stroke_set: StrokeSet,
    canvas_shape: tuple[int, int],
    texture: np.ndarray,
    *,
    paper_color: int = _DEFAULT_PAPER_COLOR,
    max_width_changes_per_stroke: int = _DEFAULT_MAX_WIDTH_CHANGES,
    grain_strength: float = _DEFAULT_GRAIN_STRENGTH,
) -> np.ndarray:
    """Recompose une image ressemblant à l'originale à partir de *stroke_set*.

    1. Simplifie le profil continu de largeur de chaque trait en au plus
       *max_width_changes_per_stroke* segments à largeur constante — voir
       :func:`_simplify_widths`.
    2. Dessine chaque segment sur un canevas couleur papier (``cv2.polylines``,
       épaisseur = largeur du segment, niveau de gris = intensité locale
       moyenne du segment).
    3. Module le résultat avec *texture* (tuilée sur le canevas, restreinte
       aux pixels de trait) pour un rendu granuleux plutôt que vectoriel plat.

    Retourne une image niveaux de gris ``uint8`` de forme *canvas_shape*.
    """
    h, w = canvas_shape
    canvas = np.full((h, w), float(paper_color), dtype=np.float32)
    stroke_mask = np.zeros((h, w), dtype=np.uint8)

    for stroke in stroke_set.strokes:
        for a, b, width in _simplify_widths(stroke.widths, max_width_changes_per_stroke):
            pts = stroke.points[a:b + 1]
            if len(pts) < 2:
                continue
            intensity = float(stroke.intensity[a:b + 1].mean())
            level = float(paper_color) * (1.0 - intensity)
            thickness = max(1, int(round(width * 2)))  # width = demi-épaisseur
            pts_i = pts.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts_i], isClosed=False, color=level,
                          thickness=thickness, lineType=cv2.LINE_AA)
            cv2.polylines(stroke_mask, [pts_i], isClosed=False, color=255,
                          thickness=thickness, lineType=cv2.LINE_AA)

    tiled = _tile_texture(texture, (h, w)).astype(np.float32)
    grain = tiled - float(tiled.mean())
    modulated = canvas + np.where(stroke_mask.astype(bool), grain * grain_strength, 0.0)
    return np.clip(modulated, 0, 255).astype(np.uint8)

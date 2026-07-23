"""Extraction du grain — résidu haute fréquence des zones sombres d'un dessin."""
from __future__ import annotations

import numpy as np

from media_restorer.imaging import lowpass

_DEFAULT_RADIUS = 15.0
_DEFAULT_PATCH_SIZE = 256
_MIN_MASK_COVERAGE = 0.5


def extract_grain_texture(
    gray: np.ndarray,
    dark_mask: np.ndarray,
    *,
    patch_size: int = _DEFAULT_PATCH_SIZE,
    radius: float = _DEFAULT_RADIUS,
) -> np.ndarray:
    """Extrait un patch de texture de grain, niveaux de gris, tuilable.

    Le résidu haute fréquence (``gray - lowpass(gray, radius)``) isole le
    grain du support (papier, poussière de fusain) indépendamment de la
    forme des traits — même technique que le mode « détail » de
    :class:`~media_restorer.engines.dual_engine.DualExposureEngine`, ici
    appliquée pour caractériser une texture plutôt que fusionner deux
    clichés.  Restreint aux zones sombres (*dark_mask*, typiquement
    :func:`~media_restorer.engines.vectorise.topology.dark_mask`) : le grain
    du papier nu n'est pas ce qu'on veut reproduire sur les traits recréés.

    Le patch retourné est recadré sur la région de *patch_size* × *patch_size*
    de plus forte variance (le grain le plus texturé, donc le plus
    représentatif) parmi une grille de blocs non chevauchants suffisamment
    couverts par *dark_mask* (au moins :data:`_MIN_MASK_COVERAGE`), puis
    normalisé sur toute la plage 0-255 pour rester directement réutilisable
    comme texture par :func:`~media_restorer.engines.vectorise.render.vectorise`.
    """
    residual = gray.astype(np.float32) - lowpass(gray, radius)
    h, w = gray.shape
    patch_size = min(patch_size, h, w)

    best_var, best_pos = -1.0, None
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            block_mask = dark_mask[y:y + patch_size, x:x + patch_size]
            if block_mask.mean() < _MIN_MASK_COVERAGE:
                continue
            var = float(residual[y:y + patch_size, x:x + patch_size].var())
            if var > best_var:
                best_var, best_pos = var, (y, x)

    if best_pos is None:
        # Aucun bloc assez couvert par le masque : repli sur le centre de
        # l'image plutôt que d'échouer — une texture moins représentative
        # vaut mieux qu'aucune texture.
        best_pos = (max(0, (h - patch_size) // 2), max(0, (w - patch_size) // 2))

    y, x = best_pos
    patch = residual[y:y + patch_size, x:x + patch_size]
    lo, hi = float(patch.min()), float(patch.max())
    if hi - lo < 1.0:
        # Moins d'un niveau de gris d'écart : aucune texture réelle à
        # rapporter, seulement du bruit numérique du flou (mesuré : un
        # résidu de région parfaitement plate atteint ~1e-5, jamais
        # exactement 0) — sans ce garde-fou, la normalisation ci-dessous
        # amplifierait ce bruit jusqu'à couvrir toute la plage 0-255.
        return np.full((patch_size, patch_size), 128, dtype=np.uint8)
    normalized = (patch - lo) / (hi - lo) * 255.0
    return normalized.astype(np.uint8)

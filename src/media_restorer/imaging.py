"""Opérations image partagées entre plusieurs moteurs/extensions.

Regroupe les traitements OpenCV/NumPy assez génériques pour être réutilisés
en dehors du module qui les a fait naître — évite qu'une extension ultérieure
ne recopie un algorithme déjà écrit, mesuré et documenté ailleurs.
"""
from __future__ import annotations

import cv2
import numpy as np


def lowpass(img: np.ndarray, radius: float) -> np.ndarray:
    """Passe-bas gaussien de rayon *radius*, calculé en sous-résolution.

    Le résultat étant limité en bande, flouter une version réduite d'un
    facteur ``radius // 4`` puis ré-agrandir est quasi exact (écart max
    mesuré 2,5 niveaux sur 255) et ~13× plus rapide qu'un ``GaussianBlur`` à
    pleine résolution.  Pour les petits rayons le facteur retombe à 1 et le
    flou exact est utilisé.

    Origine : écrit pour :class:`~media_restorer.engines.dual_engine.DualExposureEngine`
    (mode « détail »), puis extrait ici lorsque
    :mod:`~media_restorer.engines.vectorise.texture` en a eu besoin à son
    tour pour isoler le grain d'un dessin — plutôt que de dupliquer un
    algorithme déjà mesuré.
    """
    factor = max(1, int(radius // 4))
    if factor == 1:
        return cv2.GaussianBlur(img.astype(np.float32), (0, 0), radius)
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // factor), max(1, h // factor)),
                       interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small.astype(np.float32), (0, 0), radius / factor)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

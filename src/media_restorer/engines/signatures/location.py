"""Localisation zero-shot de la zone de signature sur un dessin entier.

Réutilise :func:`~media_restorer.engines.face_id.detect.build_detector`
(chargement/cache du pipeline ``transformers``, déjà écrit et déjà **validé
sur des dessins/caricatures** pour les repères de visage) mais pas
:func:`~media_restorer.engines.face_id.detect.detect_landmarks`, qui jette la
boîte et ne renvoie qu'un centre en pourcentage — ici on a besoin de la
boîte elle-même pour découper le crop.

Non testé spécifiquement sur des signatures : risque assumé, qui dégrade
proprement — un dessin non localisé part simplement en revue pour un crop
manuel (voir :mod:`~media_restorer.engines.signatures.pipeline`), jamais une
erreur.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from media_restorer.engines.face_id.detect import Detector, downscale_for_detection

#: Requête texte unique — pas de repères multiples ni de gauche/droite ici.
QUERY = "signature"


@dataclass(frozen=True)
class Box:
    """Boîte englobante en PIXELS de l'image d'ORIGINE (jamais réduite)."""

    xmin: int
    ymin: int
    xmax: int
    ymax: int

    def crop(self, image: np.ndarray) -> np.ndarray:
        return image[self.ymin:self.ymax, self.xmin:self.xmax]


def locate_signature(
    image: np.ndarray,
    *,
    detector: Detector,
    min_score: float = 0.05,
    max_side: int | None = 1536,
) -> Box | None:
    """Localise la meilleure zone « signature » de *image*, ou ``None``.

    Comme :func:`~media_restorer.engines.face_id.detect.detect_landmarks`, la
    détection tourne sur une copie **réduite** pour rester rapide — mais la
    boîte renvoyée ici est en pixels (pas un centre en pourcentage,
    intrinsèquement invariant à l'échelle) : elle DOIT donc être remise à
    l'échelle de l'image d'origine avant d'être renvoyée. L'oublier
    découperait la mauvaise zone, sans la moindre erreur — piège précis à ne
    pas reproduire.
    """
    h, w = image.shape[:2]
    scaled = downscale_for_detection(image, max_side)
    sh, sw = scaled.shape[:2]
    scale_x, scale_y = w / sw, h / sh

    detections = detector(scaled, [QUERY])
    retenues = [d for d in detections if float(d.get("score", 0.0)) >= min_score]
    if not retenues:
        return None
    meilleure = max(retenues, key=lambda d: d["score"])
    box = meilleure["box"]
    return Box(
        xmin=max(0, int(round(box["xmin"] * scale_x))),
        ymin=max(0, int(round(box["ymin"] * scale_y))),
        xmax=min(w, int(round(box["xmax"] * scale_x))),
        ymax=min(h, int(round(box["ymax"] * scale_y))),
    )

"""Cœur de détection de repères (extension Auto Face ID Register), sans Qt.

API publique : :func:`detect_landmarks` (localisation pilotée par la liste de
repères), :func:`build_detector` (détecteur réel via ``transformers``, différé)
et :data:`CANDIDATE_MODELS` (modèles zero-shot proposés au premier usage).  Voir
:mod:`media_restorer.engines.face_id.detect` pour les détails.
"""
from media_restorer.engines.face_id.detect import (
    CANDIDATE_MODELS,
    Detection,
    Detector,
    build_detector,
    detect_landmarks,
)

__all__ = [
    "CANDIDATE_MODELS",
    "Detection",
    "Detector",
    "build_detector",
    "detect_landmarks",
]

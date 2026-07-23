"""Vectorise — cœur de calcul (sans Qt) : analyse topologique et retraçage.

Ne dérive pas de :class:`~media_restorer.engines.base.BaseEngine` : le
contrat « une image en entrée, une image en sortie » ne correspond pas à
cette forme de pipeline (une image en entrée, plusieurs
:class:`~media_restorer.engines.vectorise.types.StrokeSet` candidats en
sortie pour :func:`get_outline` ; :func:`vectorise` redevient « image en
sortie », mais avec des tracés et une texture en entrées supplémentaires).
Départ délibéré de la convention ``engines/`` établie par les moteurs de
restauration.

Voir :mod:`media_restorer.extensions.vectorise` pour la couche GUI qui
enveloppe ces fonctions.

Sous-modules
------------
:mod:`~media_restorer.engines.vectorise.topology`
    Analyse topologique GUDHI — squelette (arbre couvrant minimal via
    l'homologie H0) et candidats de retraçage.
:mod:`~media_restorer.engines.vectorise.tracing`
    Décomposition de l'arbre couvrant en traits individuels (Python pur).
:mod:`~media_restorer.engines.vectorise.texture`
    Extraction du grain des zones sombres.
:mod:`~media_restorer.engines.vectorise.render`
    Reconstruction raster (tracés + texture → image).
:mod:`~media_restorer.engines.vectorise.storage`
    Lecture/écriture HDF5 des tracés.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from media_restorer.engines.vectorise import render as _render
from media_restorer.engines.vectorise import storage as _storage
from media_restorer.engines.vectorise import texture as _texture
from media_restorer.engines.vectorise import topology as _topology
from media_restorer.engines.vectorise import tracing as _tracing
from media_restorer.engines.vectorise.types import Stroke, StrokeSet

__all__ = [
    "Stroke",
    "StrokeSet",
    "get_outline",
    "extract_texture",
    "vectorise",
    "save_strokes",
    "load_strokes",
]


def _to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def get_outline(
    image: np.ndarray,
    *,
    dpi: float = 300.0,
    n_candidates: int = 5,
    max_points: int = 30_000,
    mark_fraction: float = 0.15,
) -> list[StrokeSet]:
    """Imagine des tracés plausibles pour *image* — point d'entrée principal.

    *image* est BGR (convention OpenCV du projet) ou niveaux de gris.
    *dpi* sert uniquement à convertir les largeurs mesurées en millimètres
    pour :attr:`~media_restorer.engines.vectorise.types.StrokeSet.label` ;
    lue depuis l'EXIF de la source si disponible (voir
    :mod:`media_restorer.image_io`), sinon 300 par défaut.

    Voir :func:`~media_restorer.engines.vectorise.topology.estimate_candidates`
    pour l'algorithme et :func:`~media_restorer.engines.vectorise.tracing.strokesets_from_candidates`
    pour la décomposition en traits.
    """
    gray = _to_gray(image)
    analysis, candidates = _topology.estimate_candidates(
        gray, n_candidates=n_candidates, mark_fraction=mark_fraction, max_points=max_points,
    )
    return _tracing.strokesets_from_candidates(candidates, analysis, dpi=dpi)


def extract_texture(
    image: np.ndarray, *, mark_fraction: float = 0.15, patch_size: int = 256
) -> np.ndarray:
    """Extrait un patch de texture de grain depuis les zones sombres de *image*.

    Voir :func:`~media_restorer.engines.vectorise.texture.extract_grain_texture`.
    """
    gray = _to_gray(image)
    mask = _topology.dark_mask(gray, mark_fraction)
    return _texture.extract_grain_texture(gray, mask, patch_size=patch_size)


def vectorise(
    stroke_set: StrokeSet,
    canvas_shape: tuple[int, int],
    texture: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """Recompose une image à partir de *stroke_set* et *texture*.

    Voir :func:`~media_restorer.engines.vectorise.render.vectorise`.
    """
    return _render.vectorise(stroke_set, canvas_shape, texture, **kwargs)


def save_strokes(path: Path, candidates: list[StrokeSet], **kwargs) -> None:
    """Écrit *candidates* en HDF5. Voir :func:`~media_restorer.engines.vectorise.storage.save_strokes`."""
    _storage.save_strokes(path, candidates, **kwargs)


def load_strokes(path: Path) -> list[StrokeSet]:
    """Relit des tracés HDF5. Voir :func:`~media_restorer.engines.vectorise.storage.load_strokes`."""
    return _storage.load_strokes(path)

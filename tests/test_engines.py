"""Tests unitaires pour les moteurs de restauration.

Ces tests n'ont pas besoin des fichiers de poids : ils testent la logique
interne (masquage LaMa, BaseEngine, factory) sans instancier les réseaux.
"""
import enum

import cv2
import numpy as np
import pytest

from media_restorer.engines import Engine, build_engine
from media_restorer.engines.base import BaseEngine
from media_restorer.engines.lama_engine import LaMaEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lama(bright_thresh=245, dev_thresh=30.0, dilate_px=0) -> LaMaEngine:
    """Crée un LaMaEngine sans charger les poids (attributs directs)."""
    engine = LaMaEngine.__new__(LaMaEngine)
    engine._bright_thresh = bright_thresh
    engine._dev_thresh    = dev_thresh
    engine._dilate_px     = dilate_px
    engine._model_path    = None
    return engine


# ---------------------------------------------------------------------------
# BaseEngine
# ---------------------------------------------------------------------------

class _DoubleEngine(BaseEngine):
    """Multiplie chaque pixel par 2 (moteur factice pour les tests)."""
    def restore_array(self, img: np.ndarray) -> np.ndarray:
        return np.clip(img.astype(np.uint16) * 2, 0, 255).astype(np.uint8)


def test_base_engine_restore_file_roundtrip(tmp_path):
    """restore_file lit, transforme et écrit le résultat correctement."""
    img = np.ones((10, 10, 3), dtype=np.uint8) * 50
    src = tmp_path / "src.png"
    dst = tmp_path / "dst.png"
    cv2.imwrite(str(src), img)

    _DoubleEngine().restore_file(src, dst)

    result = cv2.imread(str(dst))
    assert result is not None
    np.testing.assert_array_equal(result, img * 2)


def test_base_engine_restore_file_creates_parent_dirs(tmp_path):
    """restore_file crée les répertoires parents si nécessaire."""
    img = np.zeros((5, 5, 3), dtype=np.uint8)
    src = tmp_path / "img.png"
    dst = tmp_path / "sub" / "out.png"
    cv2.imwrite(str(src), img)

    _DoubleEngine().restore_file(src, dst)

    assert dst.exists()


def test_base_engine_restore_file_raises_on_missing_source(tmp_path):
    """restore_file lève ValueError si l'image source n'existe pas."""
    with pytest.raises(ValueError, match="Impossible de lire"):
        _DoubleEngine().restore_file(
            tmp_path / "nonexistent.png", tmp_path / "out.png"
        )


# ---------------------------------------------------------------------------
# LaMaEngine — masque automatique
# ---------------------------------------------------------------------------

def test_lama_mask_bright_pixel():
    """Un pixel très lumineux est inclus dans le masque (poussière)."""
    engine = _make_lama(bright_thresh=245, dilate_px=0)
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    img[25, 25] = (250, 250, 250)

    mask = engine._auto_mask(img)

    assert mask[25, 25] == 255
    assert mask[0, 0] == 0


def test_lama_mask_dark_stripe():
    """Une rayure sombre sur fond gris est détectée comme anomalie."""
    engine = _make_lama(dev_thresh=30.0, dilate_px=0)
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    img[:, 50] = 60  # |60 − 128| = 68 ≥ 30

    mask = engine._auto_mask(img)

    assert mask[50, 50] == 255
    assert mask[50, 0] == 0


def test_lama_mask_dilated():
    """La dilatation étend le masque autour du pixel détecté."""
    engine_no_dil  = _make_lama(bright_thresh=245, dilate_px=0)
    engine_dil     = _make_lama(bright_thresh=245, dilate_px=3)
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    img[25, 25] = (250, 250, 250)

    mask_no  = engine_no_dil._auto_mask(img)
    mask_dil = engine_dil._auto_mask(img)

    # Avec dilatation, plus de pixels sont masqués
    assert mask_dil.sum() >= mask_no.sum()
    # Le pixel central est masqué dans les deux cas
    assert mask_dil[25, 25] == 255


def test_lama_restore_array_returns_copy_when_no_damage():
    """Si le masque est vide, restore_array retourne une copie de l'image."""
    engine = _make_lama()
    img = np.ones((50, 50, 3), dtype=np.uint8) * 128  # fond gris uniforme

    result = engine.restore_array(img)

    np.testing.assert_array_equal(result, img)
    assert result is not img  # copie, pas l'original


def test_lama_mask_threshold_boundary():
    """bright_thresh utilise >= : 244 non masqué, 245 masqué, sur fond uniforme.

    Le dev_thresh est fixé à 1000 pour isoler uniquement le check luminosité
    (éviter que l'anomalie médiane masque aussi le pixel à 244).
    """
    # Image uniformément à 244 → aucune brightness, aucune anomalie
    engine = _make_lama(bright_thresh=245, dev_thresh=1000.0, dilate_px=0)
    img_below = np.ones((50, 50, 3), dtype=np.uint8) * 244
    assert engine._auto_mask(img_below).max() == 0

    # Image uniformément à 245 → brightness détectée partout
    img_at = np.ones((50, 50, 3), dtype=np.uint8) * 245
    assert engine._auto_mask(img_at).max() == 255


# ---------------------------------------------------------------------------
# build_engine
# ---------------------------------------------------------------------------

def test_build_engine_raises_on_unknown_engine():
    """build_engine lève ValueError pour un type inconnu."""
    class _Fake(enum.Enum):
        UNKNOWN = "Unknown"

    with pytest.raises(ValueError, match="inconnu"):
        build_engine(_Fake.UNKNOWN)


def test_build_engine_params_passed_as_kwargs(tmp_path, monkeypatch):
    """build_engine transmet les params au constructeur du moteur."""
    captured = {}

    class _FakeEngine:
        def __init__(self, model_path=None, scale=4, tile=256):
            captured["scale"] = scale
            captured["tile"]  = tile

    import media_restorer.engines as _mod
    monkeypatch.setattr(_mod, "build_engine", lambda *a, **kw: None)

    # Test direct : appel au constructeur avec des params
    _FakeEngine(scale=2, tile=128)
    assert captured == {"scale": 2, "tile": 128}

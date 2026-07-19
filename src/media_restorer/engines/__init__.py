"""Engine registry: enum, paramètres par défaut et factory function."""
from __future__ import annotations

import enum
from pathlib import Path

from media_restorer.engines.base import BaseEngine
from media_restorer.engines.dual_engine import BASE_IMG1, BASE_IMG2, MODES, MODE_DETAIL


class Engine(enum.Enum):
    REAL_ESRGAN = "Real-ESRGAN"
    SWINIR      = "SwinIR"
    LAMA        = "LaMa"
    GFPGAN      = "GFPGAN"
    DUAL        = "Double-exposition"


# Définitions de paramètres compatibles pyqtgraph ParameterTree.
# Chaque 'name' correspond exactement au kwarg du constructeur du moteur.
ENGINE_PARAMS: dict[Engine, list[dict]] = {
    Engine.REAL_ESRGAN: [
        {
            "name": "scale", "title": "Facteur d'agrandissement",
            "type": "int", "value": 4, "limits": (1, 8), "step": 1,
        },
        {
            "name": "tile", "title": "Taille de tuile (px)",
            "type": "int", "value": 256, "limits": (64, 1024), "step": 32,
        },
    ],
    Engine.SWINIR: [
        {
            "name": "tile", "title": "Taille de tuile (px)",
            "type": "int", "value": 400, "limits": (0, 2000), "step": 32,
        },
        {
            "name": "tile_pad", "title": "Recouvrement de tuile (px)",
            "type": "int", "value": 16, "limits": (0, 128), "step": 8,
        },
    ],
    Engine.LAMA: [
        {
            "name": "bright_thresh", "title": "Seuil luminosité — poussière",
            "type": "int", "value": 245, "limits": (180, 255), "step": 1,
        },
        {
            "name": "dev_thresh", "title": "Seuil écart médiane — rayures",
            "type": "float", "value": 30.0, "limits": (5.0, 100.0), "step": 1.0,
        },
        {
            "name": "dilate_px", "title": "Dilatation du masque (px)",
            "type": "int", "value": 4, "limits": (0, 20), "step": 1,
        },
        {
            "name": "tile", "title": "Taille de tuile (px)",
            "type": "int", "value": 1024, "limits": (0, 4096), "step": 32,
        },
        {
            "name": "tile_pad", "title": "Recouvrement de tuile (px)",
            "type": "int", "value": 100, "limits": (0, 512), "step": 8,
        },
    ],
    Engine.GFPGAN: [
        {
            "name": "upscale", "title": "Facteur d'agrandissement",
            "type": "int", "value": 2, "limits": (1, 4), "step": 1,
        },
    ],
    Engine.DUAL: [
        {
            "name": "second_path", "title": "2ᵉ image (autre éclairage)",
            "type": "file", "value": "",
            "nameFilter": "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)",
        },
        {
            "name": "mode", "title": "Mode de fusion",
            "type": "list", "value": MODE_DETAIL, "limits": list(MODES),
        },
        {
            "name": "alpha", "title": "Fondu  image 1 ↔ image 2",
            "type": "slider", "value": 0.5, "limits": (0.0, 1.0), "step": 0.01,
        },
        {
            "name": "align", "title": "Recaler (translation)",
            "type": "bool", "value": True,
        },
        {
            "name": "match_levels", "title": "Harmoniser les niveaux",
            "type": "bool", "value": False,
        },
        {
            "name": "detail_base", "title": "Détail — base basse fréquence",
            "type": "list", "value": BASE_IMG2, "limits": [BASE_IMG2, BASE_IMG1],
        },
        {
            "name": "detail_radius", "title": "Détail — rayon (px)",
            "type": "int", "value": 15, "limits": (1, 200), "step": 1,
        },
        {
            "name": "detail_gain", "title": "Détail — gain",
            "type": "float", "value": 0.8, "limits": (0.0, 2.0), "step": 0.1,
        },
        {
            "name": "w_contrast", "title": "Mertens — poids contraste",
            "type": "float", "value": 1.0, "limits": (0.0, 2.0), "step": 0.1,
        },
        {
            "name": "w_exposure", "title": "Mertens — poids exposition",
            "type": "float", "value": 0.0, "limits": (0.0, 2.0), "step": 0.1,
        },
    ],
}


def build_engine(
    engine: Engine,
    model_path: Path | None = None,
    params: dict | None = None,
) -> BaseEngine:
    """Instancier le moteur demandé avec ses paramètres.

    *params* est un dict ``{kwarg: valeur}`` issu du ParameterTree.
    Les clés correspondent aux noms définis dans ENGINE_PARAMS.
    """
    kw = params or {}
    if engine is Engine.REAL_ESRGAN:
        from media_restorer.engines.realesrgan_engine import RealESRGANEngine
        return RealESRGANEngine(model_path=model_path, **kw)
    if engine is Engine.SWINIR:
        from media_restorer.engines.swinir_engine import SwinIREngine
        return SwinIREngine(model_path=model_path)
    if engine is Engine.LAMA:
        from media_restorer.engines.lama_engine import LaMaEngine
        return LaMaEngine(model_path=model_path, **kw)
    if engine is Engine.GFPGAN:
        from media_restorer.engines.gfpgan_engine import GFPGANEngine
        return GFPGANEngine(model_path=model_path, **kw)
    if engine is Engine.DUAL:
        from media_restorer.engines.dual_engine import DualExposureEngine
        return DualExposureEngine(model_path=model_path, **kw)
    raise ValueError(f"Moteur inconnu : {engine}")

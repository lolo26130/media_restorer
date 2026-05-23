"""Engine registry: enum, paramètres par défaut et factory function."""
from __future__ import annotations

import enum
from pathlib import Path

from media_restorer.engines.base import BaseEngine


class Engine(enum.Enum):
    REAL_ESRGAN = "Real-ESRGAN"
    SWINIR      = "SwinIR"
    LAMA        = "LaMa"
    GFPGAN      = "GFPGAN"


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
        # Modèle fixe (débruiteur noise=25) — aucun paramètre utilisateur
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
    ],
    Engine.GFPGAN: [
        {
            "name": "upscale", "title": "Facteur d'agrandissement",
            "type": "int", "value": 2, "limits": (1, 4), "step": 1,
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
    raise ValueError(f"Moteur inconnu : {engine}")

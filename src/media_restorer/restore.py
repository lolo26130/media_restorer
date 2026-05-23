"""Fonctions de restauration photo — couche de compatibilité CLI.

L'implémentation réelle est dans engines/realesrgan_engine.py.
Ce module expose restore_image() et restore_image_array() pour
ne pas casser le CLI (cli.py) ni les scripts existants.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from media_restorer.power import performance_mode


def restore_image(
    input_path: Path | str,
    output_path: Path | str,
    model_path: Path | str | None = None,
    scale: int = 4,
) -> Path:
    """Restaure une image fichier et écrit le résultat dans *output_path*."""
    from media_restorer.engines.realesrgan_engine import RealESRGANEngine
    eng = RealESRGANEngine(
        model_path=Path(model_path) if model_path else None,
        scale=scale,
    )
    with performance_mode():
        return eng.restore_file(Path(input_path), Path(output_path))


def restore_image_array(
    img: np.ndarray,
    model_path: Path | str | None = None,
    scale: int = 4,
) -> np.ndarray:
    """Restaure un tableau NumPy/OpenCV BGR et retourne le résultat."""
    from media_restorer.engines.realesrgan_engine import RealESRGANEngine
    eng = RealESRGANEngine(
        model_path=Path(model_path) if model_path else None,
        scale=scale,
    )
    with performance_mode():
        return eng.restore_array(img)

"""Moteur Real-ESRGAN — montée en résolution généraliste."""
from __future__ import annotations

from pathlib import Path

import numpy as np

import media_restorer._compat  # noqa: F401 — doit précéder basicsr
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

from media_restorer.engines.base import BaseEngine

_DEFAULT_MODEL = "RealESRGAN_x4plus.pth"
_DEFAULT_SCALE = 4


class RealESRGANEngine(BaseEngine):
    """Moteur Real-ESRGAN — montée en résolution généraliste par réseau GAN.

    Real-ESRGAN (Xinntao Wang, ICCV 2021) est un réseau antagoniste génératif
    entraîné pour la restauration aveugle d'images réelles (photos numérisées,
    captures d'écran dégradées, vieilles photographies).  Il combine une
    architecture RRDB (Residual-in-Residual Dense Block) avec un discriminateur
    U-Net pour produire des textures fines et réalistes.

    Caractéristiques du modèle utilisé (RealESRGAN_x4plus.pth) :
      - Facteur d'agrandissement configurable (1 × à 8 ×, défaut 4 ×)
      - Traitement par tuiles (tile) pour contrôler l'empreinte mémoire
      - Mode CPU uniquement (half=False, device="cpu")

    Cas d'usage :
      - Agrandissement de petites photos numérisées
      - Amélioration du piqué de portraits et paysages
      - Prétraitement avant impression grand format

    Poids à télécharger : ``RealESRGAN_x4plus.pth``
      https://github.com/xinntao/Real-ESRGAN/releases
    À placer dans : ``models/RealESRGAN_x4plus.pth``

    Paramètres
    ----------
    model_path : Path | None
        Chemin vers le fichier ``.pth``.  Si None, cherche dans ``models/``.
    scale : int
        Facteur d'agrandissement (défaut 4).
    tile : int
        Taille de la tuile en pixels (défaut 256).  Réduire si OOM.
    """
    def __init__(self, model_path: Path | None = None, scale: int = _DEFAULT_SCALE, tile: int = 256) -> None:
        if model_path is None:
            model_path = Path(__file__).parents[3] / "models" / _DEFAULT_MODEL
        self._model_path = Path(model_path)
        self._scale = scale
        self._tile  = tile
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Poids introuvables : {self._model_path}\n"
                "Télécharger RealESRGAN_x4plus.pth depuis "
                "https://github.com/xinntao/Real-ESRGAN/releases "
                "et le placer dans models/"
            )

    def _build_upsampler(self) -> RealESRGANer:
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=self._scale,
        )
        return RealESRGANer(
            scale=self._scale,
            model_path=str(self._model_path),
            model=model,
            tile=self._tile, tile_pad=10, pre_pad=0,
            half=False, device="cpu",
        )

    def restore_array(self, img: np.ndarray) -> np.ndarray:
        upsampler = self._build_upsampler()
        output, _ = upsampler.enhance(img, outscale=self._scale)
        return output

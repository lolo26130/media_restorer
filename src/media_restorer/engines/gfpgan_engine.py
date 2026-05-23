"""Moteur GFPGAN — restauration spécialisée pour les visages.

GFPGAN (Tencent, 2021) intègre la détection faciale et une GAN conditionnée
par des priors de génération de visages.  Il surpasse nettement Real-ESRGAN
sur les portraits dégradés.

Poids à télécharger : GFPGANv1.4.pth
  https://github.com/TencentARC/GFPGAN/releases
À placer dans : models/GFPGANv1.4.pth
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import media_restorer._compat  # noqa: F401 — doit précéder gfpgan/basicsr
from media_restorer.engines.base import BaseEngine

_DEFAULT_MODEL = "GFPGANv1.4.pth"
_DEFAULT_UPSCALE = 2


class GFPGANEngine(BaseEngine):
    """Moteur GFPGAN — restauration faciale par GAN conditionné.

    GFPGAN (Generative Facial Prior GAN, Tencent ARC, CVPR 2021) intègre
    un détecteur facial (RetinaFace) et une GAN StyleGAN2 pré-entraînée
    comme prior de génération pour halluciner des détails fins sur les visages :
    yeux, dents, peau, sourcils.  Il surpasse nettement Real-ESRGAN sur les
    portraits dégradés.

    Architecture :
      - **Encoder** U-Net extrait les features de l'image dégradée
      - **Prior facial** (StyleGAN2 gelé) fournit la prior de texture
      - **SFT layers** (Spatial Feature Transform) fusionnent les deux
      - **Paste-back** : les visages restaurés sont réinsérés dans l'image
        originale (zones non-faciales non modifiées)

    Cas d'usage :
      - Portraits de famille dégradés ou très flous
      - Photos argentiques de personnes scannées à faible résolution
      - Cartes postales et photos d'identité anciennes

    Limitation : inefficace sur paysages, architecture ou objets sans visage —
    préférer Real-ESRGAN ou SwinIR dans ce cas.

    Poids à télécharger : ``GFPGANv1.4.pth``
      https://github.com/TencentARC/GFPGAN/releases
    À placer dans : ``models/GFPGANv1.4.pth``

    Paramètres
    ----------
    model_path : Path | None
        Chemin vers ``GFPGANv1.4.pth``.  Si None, cherche dans ``models/``.
    upscale : int
        Facteur d'agrandissement global (défaut 2).
    """
    def __init__(self, model_path: Path | None = None, upscale: int = _DEFAULT_UPSCALE) -> None:
        if model_path is None:
            model_path = Path(__file__).parents[3] / "models" / _DEFAULT_MODEL
        self._model_path = Path(model_path)
        self._upscale = upscale
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Poids GFPGAN introuvables : {self._model_path}\n"
                "Télécharger GFPGANv1.4.pth depuis "
                "https://github.com/TencentARC/GFPGAN/releases "
                "et le placer dans models/"
            )

    def restore_array(self, img: np.ndarray) -> np.ndarray:
        from gfpgan import GFPGANer
        restorer = GFPGANer(
            model_path=str(self._model_path),
            upscale=self._upscale,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
        )
        _, _, restored = restorer.enhance(
            img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
        )
        return restored

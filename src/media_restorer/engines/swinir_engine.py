"""Moteur SwinIR — débruitage couleur pour vieilles photos.

SwinIR (Liang et al., ICCV 2021) est un réseau de restauration à base de
Swin Transformer.  Ce moteur utilise le modèle de débruitage couleur entraîné
sur DFWB (noise level 25), qui supprime efficacement poussières, grain
argentique et rayures fines sans recourir à un traitement face-spécifique :
l'image restaurée conserve le naturel de l'original.

Caractéristiques :

- Scale 1 (pas de zoom) : aucune distorsion de proportion
- Non spécialisé visages : résultats naturels sur portraits et paysages
- Léger (≈ 38 Mo) vs GFPGAN (333 Mo)

Poids : ``005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth`` —
https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth
(à placer dans ``models/005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth``)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import media_restorer._compat  # noqa: F401 — doit précéder basicsr
from basicsr.archs.swinir_arch import SwinIR

from media_restorer.engines.base import BaseEngine

_DEFAULT_MODEL = "005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth"
_WINDOW_SIZE = 8


class SwinIREngine(BaseEngine):
    """Moteur SwinIR — débruitage couleur par Swin Transformer.

    SwinIR (Liang et al., ICCV 2021) remplace les convolutions classiques par
    des Self-Attention fenêtrées (Shifted Window Multi-Head Self-Attention).
    Ce moteur utilise le modèle de débruitage couleur entraîné sur DFWB
    (noise level σ=25), particulièrement efficace sur le grain argentique,
    les poussières fines et les rayures légères.

    Caractéristiques :

    - Scale × 1 (pas d'agrandissement) : proportions exactes conservées
    - Non spécialisé visages : résultat naturel sur portraits *et* paysages
    - Léger (≈ 38 Mo) par rapport à GFPGAN (333 Mo)
    - Padding réfléchissant pour gérer les images de taille quelconque

    Cas d'usage :

    - Réduction du grain argentique sur pellicule numérisée
    - Nettoyage doux sans perte de détail sur paysages et architectures
    - Premier passage avant un agrandissement Real-ESRGAN

    Poids : ``005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth`` —
    https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth
    (à placer dans ``models/005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth``)

    Paramètres
    ----------
    model_path : Path | None
        Chemin vers le fichier ``.pth``.  Si None, cherche dans ``models/``.
    """
    def __init__(self, model_path: Path | None = None) -> None:
        if model_path is None:
            model_path = Path(__file__).parents[3] / "models" / _DEFAULT_MODEL
        self._model_path = Path(model_path)
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Poids SwinIR introuvables : {self._model_path}\n"
                "Télécharger depuis :\n"
                "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
                "005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth\n"
                "et le placer dans models/"
            )

    def _build_model(self) -> SwinIR:
        model = SwinIR(
            upscale=1, in_chans=3, img_size=128, window_size=_WINDOW_SIZE,
            img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2,
            upsampler='', resi_connection='1conv',
        )
        weights = torch.load(str(self._model_path), map_location="cpu")
        key = next((k for k in ("params_ema", "params") if k in weights), None)
        model.load_state_dict(weights[key] if key else weights, strict=True)
        model.eval()
        return model

    def restore_array(self, img: np.ndarray) -> np.ndarray:
        model = self._build_model()

        # BGR uint8 → RGB float32 [0,1] tensor NCHW
        img_f = img.astype(np.float32) / 255.0
        img_rgb = np.ascontiguousarray(img_f[:, :, ::-1])       # BGR→RGB
        t = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0)

        # Padding réfléchissant pour que H,W soient multiples de window_size
        _, _, h, w = t.shape
        h_pad = (h + _WINDOW_SIZE - 1) // _WINDOW_SIZE * _WINDOW_SIZE - h
        w_pad = (w + _WINDOW_SIZE - 1) // _WINDOW_SIZE * _WINDOW_SIZE - w
        t = F.pad(t, (0, w_pad, 0, h_pad), mode="reflect")

        with torch.no_grad():
            out = model(t)

        # Rognage + RGB float → BGR uint8
        out = out[:, :, :h, :w].squeeze(0).clamp(0, 1).numpy()
        out_bgr = (out.transpose(1, 2, 0)[:, :, ::-1] * 255.0).round().astype(np.uint8)
        return out_bgr

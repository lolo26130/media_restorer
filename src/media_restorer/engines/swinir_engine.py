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
- Traitement par tuiles (comme Real-ESRGAN) pour contrôler l'empreinte
  mémoire sur les grandes images

Poids : ``005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth`` —
https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth
(à placer dans ``models/005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth``)
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import media_restorer._compat  # noqa: F401 — doit précéder basicsr
from basicsr.archs.swinir_arch import SwinIR

from media_restorer.engines.base import BaseEngine

_DEFAULT_MODEL = "005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth"
_WINDOW_SIZE = 8
_DEFAULT_TILE = 400
_DEFAULT_TILE_PAD = 16


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
    - Traitement par tuiles (``tile``/``tile_pad``) : sans cela, l'attention
      Swin Transformer tourne sur l'image entière en un seul forward pass —
      sur un scan haute résolution (ex. 3200×4800), les activations
      intermédiaires peuvent saturer la mémoire partagée d'un iGPU et geler
      la session graphique. Le découpage en tuiles ramène chaque forward
      pass à une empreinte mémoire bornée, comme pour
      :class:`~media_restorer.engines.realesrgan_engine.RealESRGANEngine`.

    Cas d'usage :

    - Réduction du grain argentique sur pellicule numérisée
    - Nettoyage doux sans perte de détail sur paysages et architectures
    - Premier passage avant un agrandissement Real-ESRGAN

    Stratégie GPU
    -------------
    Le modèle est construit une seule fois au premier appel de
    ``restore_array`` via la propriété ``_get_model``, puis conservé dans
    ``self._model`` pour toute la durée de vie de l'instance (même stratégie
    que :class:`~media_restorer.engines.realesrgan_engine.RealESRGANEngine`).

    Le device est choisi automatiquement (GPU si ``torch.cuda.is_available()``,
    CPU sinon), mais le modèle reste **toujours en FP32**, contrairement à
    :class:`~media_restorer.engines.realesrgan_engine.RealESRGANEngine`.
    En FP16, le masque d'attention fenêtrée décalée de basicsr
    (``calculate_mask``) reste en float32 pour toute image dont la taille
    diffère de ``img_size=128`` et n'est jamais casté vers le dtype du
    modèle : l'addition promeut le tenseur d'attention en float32, et le
    ``attn @ v`` suivant plante (``expected scalar type Half but found
    Float``) — en pratique sur toute taille réelle. FP32 contourne le bug ;
    le modèle étant léger (38 Mo), le surcoût mémoire/temps reste faible.

    Le tenseur d'entrée est transféré sur le même device que le modèle ;
    la sortie est ramenée sur CPU avant conversion NumPy.

    Poids : ``005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth`` —
    https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth
    (à placer dans ``models/005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth``)

    Paramètres
    ----------
    model_path : Path | None
        Chemin vers le fichier ``.pth``.  Si None, cherche dans ``models/``.
    tile : int
        Taille de la tuile en pixels (défaut 400). 0 désactive le
        découpage et traite l'image entière en un seul forward pass
        (déconseillé au-delà de quelques mégapixels — voir docstring
        de classe). Réduire si OOM.
    tile_pad : int
        Recouvrement ajouté autour de chaque tuile, en pixels (défaut 16),
        pour éviter les artefacts de bord visibles à la jointure des tuiles.
    """
    def __init__(
        self,
        model_path: Path | None = None,
        tile: int = _DEFAULT_TILE,
        tile_pad: int = _DEFAULT_TILE_PAD,
    ) -> None:
        if model_path is None:
            model_path = Path(__file__).parents[3] / "models" / _DEFAULT_MODEL
        self._model_path = Path(model_path)
        self._tile     = tile
        self._tile_pad = tile_pad
        self._model: SwinIR | None = None
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Poids SwinIR introuvables : {self._model_path}\n"
                "Télécharger depuis :\n"
                "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
                "005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth\n"
                "et le placer dans models/"
            )

    @property
    def _get_model(self) -> SwinIR:
        """Construit le modèle au premier accès et le met en cache."""
        if self._model is None:
            gpu = torch.cuda.is_available()
            device = torch.device("cuda" if gpu else "cpu")
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
            # Pas de .half() : le masque d'attention fenêtrée décalée
            # (calculate_mask, basicsr/archs/swinir_arch.py) est recalculé en
            # float32 pour toute taille d'image différente de img_size=128 et
            # n'est jamais casté — l'additionner à une activation half fait
            # remonter tout le tenseur en float32, puis `attn @ v` plante
            # (« expected scalar type Half but found Float ») dès que l'entrée
            # n'est pas exactement 128×128. Rester en FP32 évite le bug ;
            # SwinIR est assez léger (38 Mo) pour que le surcoût soit négligeable.
            self._model = model.to(device)
        return self._model

    def _forward(self, model: SwinIR, t: torch.Tensor) -> torch.Tensor:
        """Un forward pass, avec padding réfléchissant au multiple de window_size.

        *t* est NCHW, de taille quelconque ; le résultat est rogné pour
        retrouver exactement les dimensions d'entrée.
        """
        _, _, h, w = t.shape
        h_pad = (h + _WINDOW_SIZE - 1) // _WINDOW_SIZE * _WINDOW_SIZE - h
        w_pad = (w + _WINDOW_SIZE - 1) // _WINDOW_SIZE * _WINDOW_SIZE - w
        t_padded = F.pad(t, (0, w_pad, 0, h_pad), mode="reflect")
        with torch.no_grad():
            out = model(t_padded)
        return out[:, :, :h, :w]

    def _forward_tiled(self, model: SwinIR, t: torch.Tensor) -> torch.Tensor:
        """Découpe *t* en tuiles chevauchantes et les traite indépendamment.

        Même stratégie que ``RealESRGANer.tile_process`` (recouvrement
        ``tile_pad`` pour éviter les artefacts de bord), simplifiée ici
        puisque SwinIR ne change pas la résolution (scale ×1) : chaque
        tuile de sortie a exactement la taille de sa tuile d'entrée.
        """
        _, c, h, w = t.shape
        tile, pad = self._tile, self._tile_pad
        output = t.new_zeros((1, c, h, w))
        tiles_x = math.ceil(w / tile)
        tiles_y = math.ceil(h / tile)

        for y in range(tiles_y):
            for x in range(tiles_x):
                in_x0, in_x1 = x * tile, min(x * tile + tile, w)
                in_y0, in_y1 = y * tile, min(y * tile + tile, h)
                pad_x0, pad_x1 = max(in_x0 - pad, 0), min(in_x1 + pad, w)
                pad_y0, pad_y1 = max(in_y0 - pad, 0), min(in_y1 + pad, h)

                tile_in  = t[:, :, pad_y0:pad_y1, pad_x0:pad_x1]
                tile_out = self._forward(model, tile_in)

                # Zone utile de la tuile de sortie (recouvrement exclu)
                out_x0 = in_x0 - pad_x0
                out_y0 = in_y0 - pad_y0
                output[:, :, in_y0:in_y1, in_x0:in_x1] = tile_out[
                    :, :, out_y0:out_y0 + (in_y1 - in_y0), out_x0:out_x0 + (in_x1 - in_x0)
                ]
        return output

    def restore_array(self, img: np.ndarray) -> np.ndarray:
        model = self._get_model
        device = next(model.parameters()).device

        # BGR uint8 → RGB float32 [0,1] tensor NCHW (modèle toujours en FP32)
        img_f = img.astype(np.float32) / 255.0
        img_rgb = np.ascontiguousarray(img_f[:, :, ::-1])       # BGR→RGB
        t = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)

        _, _, h, w = t.shape
        if self._tile <= 0 or (h <= self._tile and w <= self._tile):
            out = self._forward(model, t)
        else:
            out = self._forward_tiled(model, t)

        # RGB float → BGR uint8
        out = out.squeeze(0).clamp(0, 1).float().cpu().numpy()
        out_bgr = (out.transpose(1, 2, 0)[:, :, ::-1] * 255.0).round().astype(np.uint8)
        return out_bgr

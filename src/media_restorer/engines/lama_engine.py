"""Moteur LaMa — inpainting pour vieilles photos.

LaMa (Resolution-robust Large Mask inpainting, Samsung Research 2021) utilise
des convolutions de Fourier pour reconstruire les zones endommagées (déchirures,
rayures profondes, taches de poussière) avec une cohérence globale remarquable,
même sur de grandes surfaces manquantes.

Détection automatique des dégâts :
  - pixels très lumineux (≥ bright_thresh) : poussière, traits blancs
  - pixels qui s'écartent fortement de leur voisinage médian (≥ dev_thresh) :
    rayures claires ou sombres, déchirures

Modèle : big-lama.pt (~200 Mo)
  Placé dans models/big-lama.pt → utilisé directement (pas de téléchargement).
  Absent → téléchargé automatiquement dans le cache torch hub au premier lancement.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from media_restorer.engines.base import BaseEngine

if TYPE_CHECKING:
    from simple_lama_inpainting import SimpleLama

_DEFAULT_MODEL = "big-lama.pt"


class LaMaEngine(BaseEngine):
    """Moteur LaMa — inpainting automatique pour photos endommagées.

    LaMa (Large Mask inpainting, Samsung Research 2021) utilise des
    convolutions de Fourier (Fast Fourier Convolution) pour reconstruire
    les zones endommagées avec une cohérence globale remarquable, même sur
    de grandes surfaces manquantes ou des rayures traversant l'image entière.

    Détection automatique des dégâts :
      - **Poussière / bords surexposés** : pixels dont la luminosité dépasse
        ``bright_thresh`` (défaut 245 / 255)
      - **Rayures claires ou sombres** : pixels dont l'écart à la médiane
        locale (fenêtre 21 px) dépasse ``dev_thresh`` (défaut 30 niveaux)
      - **Dilatation** : le masque est élargi de ``dilate_px`` pixels pour
        couvrir les bordures semi-transparentes

    Si aucun pixel n'est masqué, l'image originale est retournée telle quelle
    (aucun traitement réseau, gain de temps).

    Cas d'usage :
      - Réparation de déchirures et grandes lacunes
      - Suppression de taches de moisissures ou d'eau
      - Elimination de rayures profondes traversant le cliché

    Modèle : ``big-lama.pt`` (≈ 200 Mo)
      Cherché d'abord dans ``models/big-lama.pt`` puis téléchargé via
      torch hub si absent.

    Paramètres
    ----------
    model_path : Path | None
        Chemin vers ``big-lama.pt``.  Si None et si le fichier local est
        absent, téléchargement automatique au premier lancement.
    bright_thresh : int
        Seuil de luminosité pour la détection de poussières (défaut 245).
    dev_thresh : float
        Écart minimal à la médiane pour la détection de rayures (défaut 30.0).
    dilate_px : int
        Rayon de dilatation du masque en pixels (défaut 4).
    """
    def __init__(
        self,
        model_path: Path | None = None,
        bright_thresh: int  = 245,
        dev_thresh:    float = 30.0,
        dilate_px:     int  = 4,
    ) -> None:
        candidate = Path(model_path) if model_path else Path(__file__).parents[3] / "models" / _DEFAULT_MODEL
        self._model_path    = candidate if candidate.exists() else None
        self._bright_thresh = bright_thresh
        self._dev_thresh    = dev_thresh
        self._dilate_px     = dilate_px
        self._lama: SimpleLama | None = None

    @property
    def _get_lama(self) -> SimpleLama:
        """Construit SimpleLama au premier accès et le met en cache.

        Sans ce cache, ``torch.jit.load`` (poids ~200 Mo) et le transfert
        vers le device seraient répétés à chaque image — coûteux en lot
        (même stratégie que RealESRGANEngine/SwinIREngine).

        Device : GPU si disponible (comme le défaut upstream de
        ``simple_lama_inpainting.SimpleLama``), CPU sinon. Sur cette machine
        (Radeon 780M), le forward pass FFC de big-lama tourne ~4× plus vite
        sur GPU que sur CPU à pleine résolution (mesuré : 20.9 s vs 84.6 s
        sur un scan 3200×4800), sans dépassement mémoire malgré l'absence
        de tuilage dans ce modèle.
        """
        if self._lama is None:
            import torch
            from simple_lama_inpainting import SimpleLama
            if self._model_path is not None:
                os.environ["LAMA_MODEL"] = str(self._model_path)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._lama = SimpleLama(device=device)
        return self._lama

    def _auto_mask(self, img_bgr: np.ndarray) -> np.ndarray:
        """Calculer automatiquement le masque des zones endommagées.

        Retourne un tableau uint8 H×W : 255 = zone à reconstruire, 0 = conserver.
        """
        gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        # Taches très lumineuses (poussière, bords de déchirures surexposés)
        bright  = (gray >= self._bright_thresh).astype(np.uint8)
        # Rayures : écart important à la médiane locale (robuste aux outliers)
        blurred = cv2.medianBlur(gray, 21)
        diff    = gray.astype(np.int16) - blurred.astype(np.int16)
        anomaly = (np.abs(diff) >= self._dev_thresh).astype(np.uint8)
        mask    = np.clip(bright + anomaly, 0, 1).astype(np.uint8) * 255
        # Dilater légèrement pour couvrir les pixels de bordure
        if self._dilate_px > 0:
            r = self._dilate_px
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            mask = cv2.dilate(mask, k)
        return mask

    def restore_array(self, img: np.ndarray) -> np.ndarray:
        mask = self._auto_mask(img)
        if mask.max() == 0:
            return img.copy()  # aucun dégât détecté, image intacte
        lama    = self._get_lama
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result  = lama(img_rgb, mask)           # retourne PIL Image RGB
        return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

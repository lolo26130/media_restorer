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

import math
import os
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from media_restorer.engines.base import BaseEngine

if TYPE_CHECKING:
    from simple_lama_inpainting import SimpleLama

_DEFAULT_MODEL = "big-lama.pt"
_DEFAULT_TILE = 1024
_DEFAULT_TILE_PAD = 100


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

    Traitement par tuiles (``tile``/``tile_pad``)
      ``simple_lama_inpainting`` ne tuile pas en interne : il passe l'image
      entière (paddée au multiple de 8) dans le réseau FFC de big-lama en un
      seul forward pass. Les convolutions de Fourier de ce réseau opèrent sur
      la carte d'activation entière, dont l'empreinte mémoire croît avec
      H×W — sur un iGPU (mémoire partagée avec le système, ex. Radeon 780M),
      un forward pass à pleine résolution sur une très grande image peut
      saturer cette mémoire partagée et planter le driver (voire geler toute
      la session, pas seulement le processus Python).

      Mesuré sur cette machine : un scan 3200×4800 (~15,4 Mpx) passe sans
      problème en un seul forward pass GPU. Un scan 7370×4916 (~36,2 Mpx,
      plaque de verre haute résolution) — soit 2,4× plus de pixels — a fait
      planter la machine. La limite sûre en un seul passage se situe donc
      quelque part entre ces deux valeurs ; par prudence, toute image dont
      une dimension dépasse ``tile`` est désormais découpée en tuiles
      chevauchantes (recouvrement ``tile_pad`` pour donner au réseau assez
      de contexte autour de chaque tuile), chaque tuile étant traitée dans
      un forward pass indépendant dont l'empreinte mémoire est bornée par
      ``(tile + 2×tile_pad)²`` — très en-deçà des ~15 Mpx validés, quelle
      que soit la taille totale de l'image. Les tuiles ne contenant aucun
      pixel de masque sont ignorées (pas d'appel réseau inutile), comme pour
      l'image entière ci-dessus.

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
    tile : int
        Taille de tuile en pixels (défaut 1024). Au-delà de cette taille
        sur un axe, l'image est découpée en tuiles pour borner la mémoire
        GPU utilisée par forward pass. 0 désactive le découpage et traite
        l'image entière en un seul passage (déconseillé au-delà de
        quelques mégapixels sur iGPU — voir docstring de classe).
    tile_pad : int
        Recouvrement ajouté autour de chaque tuile, en pixels (défaut 100),
        pour donner au réseau du contexte au-delà de la zone utile et
        éviter les artefacts de bord.
    """
    def __init__(
        self,
        model_path: Path | None = None,
        bright_thresh: int  = 245,
        dev_thresh:    float = 30.0,
        dilate_px:     int  = 4,
        tile:          int  = _DEFAULT_TILE,
        tile_pad:      int  = _DEFAULT_TILE_PAD,
    ) -> None:
        candidate = Path(model_path) if model_path else Path(__file__).parents[3] / "models" / _DEFAULT_MODEL
        self._model_path    = candidate if candidate.exists() else None
        self._bright_thresh = bright_thresh
        self._dev_thresh    = dev_thresh
        self._dilate_px     = dilate_px
        self._tile          = tile
        self._tile_pad      = tile_pad
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
        sur un scan 3200×4800). ``simple_lama_inpainting`` lui-même ne tuile
        jamais — c'est ``LaMaEngine._restore_tiled`` qui borne la mémoire
        pour les images dépassant ``tile`` (voir docstring de classe).
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

    def _restore_tiled(self, lama: SimpleLama, img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Découpe *img_rgb*/*mask* en tuiles chevauchantes traitées indépendamment.

        Même stratégie que ``SwinIREngine._forward_tiled`` (recouvrement
        ``tile_pad`` pour donner du contexte au réseau), adaptée à l'API
        PIL/NumPy de ``SimpleLama`` plutôt qu'à des tenseurs. Les tuiles
        sans pixel masqué sont recopiées telles quelles, sans appel réseau.
        Seuls les pixels effectivement masqués sont remplacés par le résultat
        du réseau (composition avec l'original), pour éviter toute jointure
        visible entre tuiles dans les zones non endommagées.
        """
        h, w = mask.shape[:2]
        tile, pad = self._tile, self._tile_pad
        output = img_rgb.copy()
        tiles_x = math.ceil(w / tile)
        tiles_y = math.ceil(h / tile)

        for y in range(tiles_y):
            for x in range(tiles_x):
                in_x0, in_x1 = x * tile, min(x * tile + tile, w)
                in_y0, in_y1 = y * tile, min(y * tile + tile, h)
                mask_tile = mask[in_y0:in_y1, in_x0:in_x1]
                if mask_tile.max() == 0:
                    continue  # rien à reconstruire dans cette tuile

                pad_x0, pad_x1 = max(in_x0 - pad, 0), min(in_x1 + pad, w)
                pad_y0, pad_y1 = max(in_y0 - pad, 0), min(in_y1 + pad, h)
                img_crop  = img_rgb[pad_y0:pad_y1, pad_x0:pad_x1]
                mask_crop = mask[pad_y0:pad_y1, pad_x0:pad_x1]
                result_crop = np.array(lama(img_crop, mask_crop))  # PIL → NumPy RGB

                out_x0, out_y0 = in_x0 - pad_x0, in_y0 - pad_y0
                out_slice  = result_crop[out_y0:out_y0 + (in_y1 - in_y0), out_x0:out_x0 + (in_x1 - in_x0)]
                keep       = mask_tile.astype(bool)
                output[in_y0:in_y1, in_x0:in_x1][keep] = out_slice[keep]
        return output

    def restore_array(self, img: np.ndarray) -> np.ndarray:
        mask = self._auto_mask(img)
        if mask.max() == 0:
            return img.copy()  # aucun dégât détecté, image intacte
        lama    = self._get_lama
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w    = mask.shape[:2]

        if self._tile <= 0 or (h <= self._tile and w <= self._tile):
            result_rgb = np.array(lama(img_rgb, mask))  # PIL → NumPy RGB
        else:
            result_rgb = self._restore_tiled(lama, img_rgb, mask)
        return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

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

    Cas d'usage :
      - Agrandissement de petites photos numérisées
      - Amélioration du piqué de portraits et paysages
      - Prétraitement avant impression grand format

    Poids : ``RealESRGAN_x4plus.pth`` —
    https://github.com/xinntao/Real-ESRGAN/releases
    (à placer dans ``models/RealESRGAN_x4plus.pth``)

    Stratégie GPU (générale)
    ------------------------
    L'upsampler (RealESRGANer + poids) est construit une seule fois au premier
    appel de ``restore_array`` puis conservé dans ``self._upsampler`` pour toute
    la durée de vie de l'instance.  Sans ce cache, le chargement des poids depuis
    le disque et leur transfert vers le GPU dominent le temps total et annulent
    le gain d'accélération.

    Le device est choisi automatiquement à la construction :

    - ``torch.cuda.is_available()`` → True  : GPU (ROCm ou CUDA), ``half=True``
    - ``torch.cuda.is_available()`` → False : CPU, ``half=False``

    ``half=True`` active la précision FP16 sur GPU, ce qui divise l'empreinte
    mémoire par deux et accélère les convolutions sur les architectures modernes
    sans perte visible de qualité pour la super-résolution.

    ``tile=256`` est le découpage spatial appliqué aux grandes images : chaque
    tuile est traitée indépendamment pour éviter les erreurs OOM.  Une valeur
    trop grande sature la bande passante mémoire (surtout sur GPU intégré) ;
    256 px est le meilleur compromis vitesse/mémoire.

    Note spécifique — AMD Radeon 780M (gfx1103, iGPU Ryzen 8845HS)
    ---------------------------------------------------------------
    Cette puce n'est pas dans la liste de support officielle de ROCm 5.7
    (qui s'arrête à gfx1102).  Le flag d'environnement
    ``HSA_OVERRIDE_GFX_VERSION=11.0.0`` (positionné dans ``cli.py``) force
    le runtime HSA à utiliser les noyaux compilés pour gfx1100, compatibles
    avec gfx1103 en pratique.

    Étant une iGPU, la 780M partage la RAM système comme VRAM (~47 Go visibles
    par PyTorch).  Les transferts CPU↔GPU sont des copies en mémoire partagée,
    moins coûteuses que sur une carte discrète mais non négligeables.

    Performances mesurées sur cette machine (photo 602×596 px, 9 tuiles 256 px,
    modèle déjà en cache) :

    - GPU Radeon 780M : ~3.8 s
    - CPU Ryzen 8845HS : ~38.6 s
    - Gain effectif    : ×10

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
        self._scale      = scale
        self._tile       = tile
        self._upsampler: RealESRGANer | None = None
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Poids introuvables : {self._model_path}\n"
                "Télécharger RealESRGAN_x4plus.pth depuis "
                "https://github.com/xinntao/Real-ESRGAN/releases "
                "et le placer dans models/"
            )

    @property
    def _get_upsampler(self) -> RealESRGANer:
        """Construit l'upsampler au premier accès et le met en cache.

        Le chargement des poids + transfert GPU prend ~1 s (iGPU partagée) ;
        le mettre en cache ramène les appels suivants à la durée de calcul pure.
        """
        if self._upsampler is None:
            import torch
            gpu = torch.cuda.is_available()
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=self._scale,
            )
            self._upsampler = RealESRGANer(
                scale=self._scale,
                model_path=str(self._model_path),
                model=model,
                tile=self._tile, tile_pad=10, pre_pad=0,
                half=gpu,
                device=torch.device("cuda" if gpu else "cpu"),
            )
        return self._upsampler

    def restore_array(self, img: np.ndarray) -> np.ndarray:
        output, _ = self._get_upsampler.enhance(img, outscale=self._scale)
        return output

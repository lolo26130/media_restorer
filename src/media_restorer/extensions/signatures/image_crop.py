"""Sélection interactive d'une zone rectangulaire sur une image agrandie.

Widget propre à cette extension (pas au cœur) : la règle du projet est
d'extraire un widget partagé seulement quand un **second** outil en a
besoin — ``image_click.py`` a été déplacé au cœur précisément à ce
moment-là, pas avant.  À reconsidérer si une future extension en a l'usage.

Aucun widget de sélection rectangulaire n'existait dans ce projet avant
celui-ci : chaque ``pg.ImageView`` du projet masque explicitement le bouton
ROI intégré de pyqtgraph (``ui.roiBtn.hide()``).  Celui-ci construit
justement sur ce ``pg.RectROI`` jusqu'ici inutilisé, dans le même esprit que
:class:`~media_restorer.image_click.ImageClick` (``pg.ImageView`` +
survol/clavier) mais pour désigner une **zone**, pas un point.

Expose :meth:`set_region`/:meth:`get_region` **programmatiques**, pas
seulement pilotables à la souris : la revue peut ainsi pré-positionner la
zone sur une boîte déjà localisée par OWL-ViT (voir
:mod:`~media_restorer.engines.signatures.location`), et les tests peuvent
piloter le widget sans simuler d'événements souris bruts — même esprit que
:meth:`~media_restorer.image_click.ImageClick.get_tags`.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg

#: Taille par défaut (en pixels image) de la zone proposée quand aucune boîte
#: n'a été localisée automatiquement — assez petite pour tenir sur une
#: signature typique, l'utilisateur la redimensionne au besoin.
_DEFAULT_REGION_SIZE = 150


class ImageCrop(pg.ImageView):
    """Vue d'image pyqtgraph avec une zone rectangulaire déplaçable/redimensionnable.

    Deux étapes d'initialisation (constructeur sans argument, puis
    :meth:`setup`), même contrainte qu':class:`~media_restorer.image_click.ImageClick` :
    un ``QWidget`` instanciable par Qt ne doit jamais exiger d'argument.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._initialized = False
        self._roi: pg.RectROI | None = None
        self._img_w: int | None = None
        self._img_h: int | None = None

    def setup(self) -> None:
        """Prépare le widget (une seule fois) : masque les boutons superflus,
        installe la zone de sélection.  Appels ultérieurs ignorés.
        """
        if self._initialized:
            return
        self._initialized = True
        self.ui.roiBtn.hide()
        self.ui.menuBtn.hide()
        self.ui.histogram.hide()

    def set_image(self, img: np.ndarray) -> None:
        """Affiche *img* « à l'endroit » et pose une zone par défaut centrée.

        Même transposition que :meth:`~media_restorer.image_click.ImageClick.set_image` —
        pyqtgraph est col-major, NumPy row-major.
        """
        self._img_h, self._img_w = int(img.shape[0]), int(img.shape[1])
        display = img.T if img.ndim == 2 else np.transpose(img, (1, 0, 2))
        self.setImage(display, autoLevels=False, levels=(0, 255))
        self._reset_roi()

    def _reset_roi(self) -> None:
        if self._roi is not None:
            self.getView().removeItem(self._roi)
        w = self._img_w or _DEFAULT_REGION_SIZE
        h = self._img_h or _DEFAULT_REGION_SIZE
        size = min(_DEFAULT_REGION_SIZE, w, h) or 1
        x0 = max(0, (w - size) // 2)
        y0 = max(0, (h - size) // 2)
        roi = pg.RectROI(
            [x0, y0], [size, size], pen=pg.mkPen("r", width=2), sideScalers=True
        )
        roi.addScaleHandle([1, 1], [0, 0])
        roi.addScaleHandle([0, 0], [1, 1])
        self.getView().addItem(roi)
        self._roi = roi

    def set_region(self, xmin: int, ymin: int, xmax: int, ymax: int) -> None:
        """Positionne la zone en PIXELS de l'image (ex. une boîte localisée automatiquement)."""
        if self._roi is None:
            self._reset_roi()
        assert self._roi is not None
        self._roi.setPos([xmin, ymin])
        self._roi.setSize([max(1, xmax - xmin), max(1, ymax - ymin)])

    def get_region(self) -> tuple[int, int, int, int]:
        """Zone actuelle en PIXELS de l'image, bornée à ses dimensions : ``(xmin, ymin, xmax, ymax)``."""
        assert self._roi is not None
        pos = self._roi.pos()
        size = self._roi.size()
        w = self._img_w or 0
        h = self._img_h or 0
        xmin = int(round(max(0, min(pos.x(), w))))
        ymin = int(round(max(0, min(pos.y(), h))))
        xmax = int(round(max(0, min(pos.x() + size.x(), w))))
        ymax = int(round(max(0, min(pos.y() + size.y(), h))))
        return xmin, ymin, xmax, ymax

    def get_crop(self, image: np.ndarray) -> np.ndarray:
        """Recadre *image* (tableau NumPy d'origine, pas transposé) selon :meth:`get_region`."""
        xmin, ymin, xmax, ymax = self.get_region()
        return image[ymin:ymax, xmin:xmax]

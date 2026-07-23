"""Widgets Qt partagés entre plusieurs extensions.

Regroupe ce qui ne dépend d'aucune extension en particulier — évite qu'une
extension ultérieure ne recopie un widget déjà écrit et documenté ailleurs
(voir :mod:`media_restorer.imaging` pour le même principe côté calcul).
"""
from __future__ import annotations

import cv2
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QMainWindow


class ResultWindow(QMainWindow):
    """Affiche une image résultat dans une fenêtre indépendante.

    Née dans l'extension Media Restorer (un onglet moteur = une fenêtre de
    résultat), extraite ici lorsque l'extension Vectorise en a eu besoin à
    son tour pour afficher/animer ses tracés — plutôt que de dupliquer ce
    widget.
    """

    def __init__(self, title: str, preserve_zoom: bool = False) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self.resize(900, 700)
        self._view = pg.ImageView()
        self._view.ui.roiBtn.hide()
        self._view.ui.menuBtn.hide()
        self.setCentralWidget(self._view)
        # Certains usages (ex. onglet Double-exposition de Media Restorer) y
        # relancent un traitement pour comparer des réglages sur un même
        # détail zoomé — perdre le zoom à chaque clic obligerait à re-zoomer
        # à chaque comparaison.  Par défaut, chaque résultat est cadré
        # automatiquement comme pour les autres moteurs.
        self._preserve_zoom = preserve_zoom
        self._has_content   = False

    def show_image(self, img_bgr: np.ndarray) -> None:
        """Affiche un nouveau résultat pleine résolution.

        Cadre automatiquement la vue au premier affichage — rien à
        préserver.  Aux affichages suivants, ne recadre que si
        *preserve_zoom* est désactivé.
        """
        auto_range = not (self._preserve_zoom and self._has_content)
        self._set_image(img_bgr, auto_range=auto_range)
        self._has_content = True
        self.show()
        self.raise_()
        self.activateWindow()

    def update_image(
        self, img_bgr: np.ndarray, scale: tuple[float, float] | None = None
    ) -> None:
        """Remplace l'image ; réaffiche la fenêtre si elle avait été fermée.

        Utilisé pour un aperçu interactif (ex. slider) : pendant que
        l'utilisateur agit sur une fenêtre déjà visible, ne pas reprendre le
        focus ni recadrer la vue à chaque cran.

        Mais si la fenêtre a été fermée entre-temps, la rouvrir plutôt que de
        laisser l'interaction paraître sans effet : un ``update_image`` muet
        sur une fenêtre invisible ne laisse à l'utilisateur aucun moyen de
        comprendre pourquoi l'image ne change pas.

        *scale* étire l'image sur l'aire qu'elle doit occuper dans la vue.
        Indispensable pour un aperçu en résolution réduite : sans lui, une
        image plus petite se dessine dans un coin du cadrage précédent — elle
        change bien, mais hors du champ regardé, ce qui donne l'impression
        que l'interaction n'agit pas.  Sans objet à la réouverture — il n'y a
        pas de cadrage précédent à respecter, ``autoRange`` suffit.
        """
        was_visible = self.isVisible()
        self._set_image(
            img_bgr, auto_range=not was_visible, scale=scale if was_visible else None
        )
        if not was_visible:
            self.show()
            self.raise_()
            self.activateWindow()

    def _set_image(
        self,
        img_bgr: np.ndarray,
        auto_range: bool,
        scale: tuple[float, float] | None = None,
    ) -> None:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if img_bgr.ndim == 3 else img_bgr
        # scale=None → pyqtgraph applique une transformation identité, ce qui
        # annule l'étirement d'un aperçu précédent.
        self._view.setImage(
            rgb, autoLevels=False, levels=(0, 255), autoRange=auto_range, scale=scale
        )

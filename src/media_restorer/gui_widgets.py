"""Widgets Qt partagés entre plusieurs extensions.

Regroupe ce qui ne dépend d'aucune extension en particulier — évite qu'une
extension ultérieure ne recopie un widget déjà écrit et documenté ailleurs
(voir :mod:`media_restorer.imaging` pour le même principe côté calcul).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QMainWindow,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


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


class InfoExifPanel(QWidget):
    """Panneau lecture seule des métadonnées d'une image ou d'un répertoire.

    Utilisé dans le dock « Infos, Exif » de la fenêtre racine
    (:class:`~media_restorer.gui_root.ImageTreatmentWindow`).  Reçoit un chemin,
    affiche les infos lues par :mod:`media_restorer.exif_info` (cœur sans Qt)
    dans un ``QTableWidget`` clé/valeur.

    Les fonctions de lecture sont injectables (``read_fn`` / ``summary_fn``) pour
    les tests — même motif que le ``exiftool_runner`` de
    :class:`~media_restorer.landmarks.LandmarkSet` : aucun test n'a besoin du
    vrai binaire ``exiftool``.
    """

    def __init__(
        self,
        *,
        read_fn: Callable[[Path], dict] | None = None,
        summary_fn: Callable[..., dict] | None = None,
    ) -> None:
        super().__init__()
        from media_restorer import exif_info

        self._read_fn = read_fn or exif_info.read_image_info
        self._summary_fn = summary_fn or exif_info.read_directory_summary

        self._table = QTableWidget(0, 2, self)
        self._table.setHorizontalHeaderLabels(["Propriété", "Valeur"])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._table)

    def show_for_path(self, path: Path) -> None:
        """Affiche les métadonnées de l'image *path*."""
        self._fill(self._read_fn(path))

    def show_directory(self, path: Path, *, recursive: bool = False) -> None:
        """Affiche le résumé du répertoire *path*."""
        self._fill(self._summary_fn(path, recursive=recursive))

    def clear(self) -> None:
        """Vide le tableau (aucune cible)."""
        self._table.setRowCount(0)

    def _fill(self, info: dict) -> None:
        self._table.setRowCount(len(info))
        for row, (key, value) in enumerate(info.items()):
            self._table.setItem(row, 0, QTableWidgetItem(str(key)))
            self._table.setItem(row, 1, QTableWidgetItem(str(value)))

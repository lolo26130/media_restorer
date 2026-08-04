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
from PyQt6.QtCore import QByteArray, QSize, Qt

from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDockWidget,
    QHeaderView,
    QLabel,
    QMainWindow,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from media_restorer.app_settings import app_settings

#: Largeur par défaut, en pixels, d'un dock qui porte un aperçu d'image.
#: Sert à la fois de taille conseillée au widget (:attr:`ImagePreview.SIZE_HINT`)
#: et de largeur imposée au dock au premier lancement — les deux sont
#: nécessaires, voir :func:`apply_default_dock_width`.
PREVIEW_DOCK_WIDTH = 460


def restore_layout(window: QMainWindow, prefix: str) -> bool:
    """Rétablit géométrie et disposition des docks ; dit si elle a trouvé quoi que ce soit.

    À appeler **en fin de construction**, une fois tous les docks créés, empilés
    (``tabifyDockWidget``) et l'onglet initial choisi : ``restoreState`` remplace
    ces choix par ceux de la dernière session, il doit donc parler en dernier.

    Renvoie ``False`` au tout premier lancement — il n'y a alors rien à
    rétablir, et c'est à l'appelant de poser une disposition par défaut
    utilisable (voir :func:`apply_default_dock_width`).

    .. warning::

       Un ``QDockWidget`` **sans ``objectName``** est purement et simplement
       ignoré par ``saveState``/``restoreState`` — Qt écrit un avertissement sur
       la sortie d'erreur puis continue.  La mémorisation semble alors
       fonctionner sans rien mémoriser.  Tout dock concerné doit porter un nom
       d'objet **stable d'une version à l'autre** : c'est la clé qui relie un
       état enregistré à son dock.  Verrouillé par
       ``test_gui_layout.py::test_every_dock_has_an_object_name``.
    """
    settings = app_settings()
    trouve = False
    geometrie = settings.value(f"{prefix}/geometry")
    if isinstance(geometrie, QByteArray) and not geometrie.isEmpty():
        trouve = window.restoreGeometry(geometrie) or trouve
    etat = settings.value(f"{prefix}/dock_state")
    if isinstance(etat, QByteArray) and not etat.isEmpty():
        trouve = window.restoreState(etat) or trouve
    return trouve


def save_layout(window: QMainWindow, prefix: str) -> None:
    """Mémorise géométrie et disposition — à appeler depuis ``closeEvent``.

    ``closeEvent`` est le seul endroit par lequel passe *toute* fermeture de la
    fenêtre ; y placer l'enregistrement garantit qu'aucun chemin de sortie ne
    l'oublie.
    """
    settings = app_settings()
    settings.setValue(f"{prefix}/geometry", window.saveGeometry())
    settings.setValue(f"{prefix}/dock_state", window.saveState())


def apply_default_dock_width(
    window: QMainWindow, dock: QDockWidget, width: int = PREVIEW_DOCK_WIDTH
) -> None:
    """Donne à *dock* une largeur utilisable au tout premier lancement.

    Deux gestes, dans cet ordre, parce que chacun seul est insuffisant :

    1. **élargir la fenêtre du manque** — ``resizeDocks`` *répartit* la largeur
       existante, il n'en crée pas.  La fenêtre racine naît à 769 px (fixés par
       ``root.ui``, mesuré) : les 460 px demandés pour l'aperçu y étaient
       ramenés à 180 px, faute de place à prendre.  On ajoute donc *width* à la
       largeur de la fenêtre, dans la limite de l'écran disponible ;
    2. **imposer la largeur du dock** — car ``sizeHint()`` ne suffit pas pour un
       dock : ``QMainWindow`` arbitre entre les docks et le widget central, et
       le vœu d'un enfant y perd régulièrement.  Seul ``resizeDocks`` tranche.

    Sur un écran trop étroit pour tout accorder, la fenêtre est simplement
    plafonnée : rien ne déborde, et l'aperçu obtient ce qui reste.

    .. important::

       À appeler depuis ``showEvent``, **pas** depuis ``__init__``.  Avant le
       premier affichage, ``QMainWindow`` n'a pas encore arbitré ses zones : le
       même appel donnait alors 241 px là où il en donne 460 après ``show()``
       (mesuré, fenêtre de 1229 px).

    À n'appeler **que** lorsque :func:`restore_layout` a renvoyé ``False`` :
    sinon on écraserait la taille que l'utilisateur avait choisie, ce qui
    ramènerait précisément le défaut qu'on corrige.
    """
    ecran = QApplication.primaryScreen()
    if ecran is not None:
        disponible = ecran.availableGeometry()
        # Jamais rétrécissante : ``max(window.…)`` garde la taille du .ui si
        # celle-ci dépasse déjà ce qu'on demanderait (cas de PreClassementGUI,
        # large de 1100 px, à qui l'aperçu revient sans rien changer).
        window.resize(
            max(window.width(), min(window.width() + width, disponible.width())),
            max(window.height(), min(int(disponible.height() * 0.75),
                                     disponible.height())),
        )
    window.resizeDocks([dock], [width], Qt.Orientation.Horizontal)


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


class ImagePreview(QWidget):
    """Aperçu d'une image dans un ``pg.ImageView``, chargé en résolution réduite.

    Conçu pour un **dock d'aperçu** que l'on parcourt au clavier : la sélection
    peut changer plusieurs fois par seconde, il est donc exclu de décoder à
    pleine résolution.  Le chargement utilise ``Image.draft()`` — le décodeur
    JPEG ne produit alors que la taille utile — puis réduit à *max_side*.  Un
    scan de 50 Mpx s'affiche ainsi en quelques dizaines de millisecondes au
    lieu de plusieurs secondes.

    L'orientation EXIF est appliquée via
    :func:`~media_restorer.image_io.apply_exif_orientation`, la même fonction
    que le reste du projet : un portrait pris à l'horizontale s'affiche droit,
    et la logique d'orientation n'est écrite qu'une fois.

    Ne lève jamais : un fichier illisible affiche un message dans la vue plutôt
    que d'interrompre le parcours — on navigue souvent dans des corpus qui
    contiennent quelques fichiers abîmés.
    """

    #: Côté maximal de l'aperçu.  1600 px suffit largement à un dock, et
    #: garde le chargement sous la barre des ~50 ms même sur un gros TIFF.
    PREVIEW_MAX_SIDE = 1600

    #: Taille CONSEILLÉE au gestionnaire de disposition.  Sans elle, un
    #: pg.ImageView placé dans un dock naît à quelques dizaines de pixels : Qt
    #: n'a aucune raison de lui donner plus, faute d'indication.  C'est ce qui
    #: obligeait à redimensionner le dock à chaque lancement.
    SIZE_HINT = QSize(PREVIEW_DOCK_WIDTH, 380)
    MINIMUM_HINT = QSize(180, 150)

    def __init__(self, *, max_side: int = PREVIEW_MAX_SIDE) -> None:
        super().__init__()
        self._max_side = max_side
        self._current: Path | None = None

        self._view = pg.ImageView()
        self._view.ui.roiBtn.hide()
        self._view.ui.menuBtn.hide()
        self._view.ui.histogram.hide()      # inutile pour un simple aperçu

        self._message = QLabel("")
        self._message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._message.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)
        layout.addWidget(self._message)
        self._set_message("Aucune image sélectionnée.")

    def sizeHint(self) -> QSize:            # noqa: N802 — API Qt
        """Taille souhaitée : sans elle, l'aperçu naît minuscule dans un dock."""
        return QSize(self.SIZE_HINT)

    def minimumSizeHint(self) -> QSize:     # noqa: N802 — API Qt
        """Plancher : en deçà, l'aperçu ne montre plus rien d'utile."""
        return QSize(self.MINIMUM_HINT)

    @property
    def current_path(self) -> Path | None:
        """Image actuellement affichée, ou ``None``."""
        return self._current

    def show_path(self, path: Path | str) -> bool:
        """Affiche *path* en aperçu.  Renvoie ``False`` si la lecture a échoué.

        Recharger la même image est un no-op : parcourir la liste avec les
        flèches redéclenche la sélection sans que le fichier change, et
        recadrer la vue à chaque fois donnerait un clignotement inutile.
        """
        path = Path(path)
        if path == self._current:
            return True
        rgb = self._load(path)
        if rgb is None:
            self._current = None
            self._view.clear()
            self._set_message(f"« {path.name} » : image illisible.")
            return False
        # Transposition (x, y) : pyqtgraph est col-major, NumPy row-major —
        # même convention que media_restorer.image_click.
        self._view.setImage(
            np.transpose(rgb, (1, 0, 2)), autoLevels=False, levels=(0, 255),
            autoRange=True,
        )
        self._current = path
        self._set_message("")
        return True

    def clear(self) -> None:
        """Vide l'aperçu (plus aucune sélection)."""
        self._current = None
        self._view.clear()
        self._set_message("Aucune image sélectionnée.")

    def _set_message(self, text: str) -> None:
        self._message.setText(text)
        self._message.setVisible(bool(text))

    def _load(self, path: Path) -> np.ndarray | None:
        """Charge *path* réduit et redressé, ou ``None`` s'il est illisible."""
        from PIL import Image

        from media_restorer.image_io import apply_exif_orientation, exif_orientation

        try:
            with Image.open(path) as im:
                im.draft("RGB", (self._max_side, self._max_side))
                thumb = im.convert("RGB")
                thumb.thumbnail((self._max_side, self._max_side))
                rgb = np.asarray(thumb, dtype=np.uint8)
        except Exception:
            return None
        try:
            return apply_exif_orientation(rgb, exif_orientation(path))
        except Exception:
            return rgb           # orientation illisible : l'image reste utile


class InfoExifPanel(QWidget):
    """Panneau lecture seule des métadonnées d'une image ou d'un répertoire.

    Utilisé dans le dock « Infos, Exif » de la fenêtre racine
    (:class:`~media_restorer.gui_root.ImageTreatmentWindow`).  Affiche les infos
    lues par :mod:`media_restorer.exif_info` (cœur sans Qt) sous forme
    **hiérarchique** : un ``QTreeWidget`` où chaque groupe de provenance (``File``,
    ``EXIF``, ``XMP``, ``MakerNotes``…) est un nœud parent repliable, ses tags en
    enfants clé/valeur.  Cliquer sur la flèche d'un groupe le replie.

    ``QTreeWidget`` plutôt que ``pyqtgraph.DataTreeWidget`` : deux colonnes
    propres (Propriété/Valeur) sans la colonne « type » superflue de ce dernier,
    et maîtrise totale de l'ordre des groupes.

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

        self._tree = QTreeWidget(self)
        self._tree.setColumnCount(2)
        self._tree.setHeaderLabels(["Propriété", "Valeur"])
        self._tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._tree.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        header = self._tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._tree)

    def show_for_path(self, path: Path) -> None:
        """Affiche les métadonnées de l'image *path* (arbre par provenance)."""
        self._fill(self._read_fn(path))

    def show_directory(self, path: Path, *, recursive: bool = False) -> None:
        """Affiche le résumé du répertoire *path*."""
        self._fill(self._summary_fn(path, recursive=recursive))

    def clear(self) -> None:
        """Vide l'arbre (aucune cible)."""
        self._tree.clear()

    def _fill(self, grouped: dict) -> None:
        """Peuple l'arbre depuis ``{groupe: {tag: valeur}}`` — groupes dépliés."""
        self._tree.clear()
        for group, entries in grouped.items():
            parent = QTreeWidgetItem(self._tree, [str(group), ""])
            for key, value in entries.items():
                QTreeWidgetItem(parent, [str(key), str(value)])
            parent.setExpanded(True)

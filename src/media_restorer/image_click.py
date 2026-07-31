"""Désignation interactive de points nommés à la souris sur une image.

Widget partagé (cœur, hors extension) : utilisé par l'extension
:mod:`~media_restorer.extensions.manual_mouse_points` (désignation manuelle) et
par :mod:`~media_restorer.extensions.auto_face_id_register` (revue/correction de
repères détectés automatiquement).  Placé ici plutôt que dans une extension
pour que les deux le réutilisent sans dépendre l'une de l'autre — même principe
que :class:`~media_restorer.gui_widgets.ResultWindow`.

Copié puis adapté — sans importation — de
``TraiteImages.classes.data_classes.ImageClick`` : même interaction clavier
(survol souris, ``Entrée`` = marquer, ``Espace`` = passer, ``Q`` = terminer),
débarrassée des mixins propres au projet d'origine (``CommonQtMethods``,
``CommonQtMethods_pyqtgraph``) dont la logique de pointage n'a pas besoin.

Améliorations par rapport à la source
-------------------------------------
- **Signaux Qt** (``point_marked``, ``tagging_finished``, ``instruction_changed``)
  au lieu d'un attribut ``self.show`` tantôt ``print`` tantôt ``QLabel`` : la
  fenêtre hôte réagit à la fin du marquage sans sonder l'état du widget.
- **Coordonnées en pourcentage** : l'image est affichée transposée sur ses
  deux premiers axes (exigence d'affichage « à l'endroit » de pyqtgraph en
  ``col-major``, l'ordre par défaut du reste de l'application), ce qui fait
  que ``mapSceneToView`` renvoie ``(x = colonne, y = ligne)`` en pixels.  Ces
  pixels sont convertis en **pourcentage (0–100) de la largeur/hauteur** avant
  d'être émis/stockés : les repères deviennent indépendants de la résolution
  (aucune conversion lors d'un changement de résolution — voir
  :mod:`media_restorer.landmarks`).  La conversion inverse (pourcentage →
  pixel) sert uniquement à dessiner les repères sur la vue.
- Focus clavier activé explicitement (``StrongFocus``), sans quoi ``Entrée``/
  ``Espace``/``Q`` n'atteindraient jamais ``keyPressEvent``.
"""
from __future__ import annotations

from enum import IntEnum, auto

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from PyQt6.QtCore import pyqtSignal

DEFAULT_LABELS = ["Left Eye", "Right Eye", "Nose", "Left Ear", "Right Ear"]


class ImageClickMode(IntEnum):
    """État de la désignation de repères."""

    READY_TO_BEGIN = auto()
    SELECT_IN_PROGRESS = auto()
    DONE = auto()


class ImageClick(pg.ImageView):
    """Vue d'image pyqtgraph permettant de désigner des repères au clavier/souris.

    Deux étapes d'initialisation (constructeur sans argument custom, puis
    :meth:`setup`) — un ``QWidget`` susceptible d'être instancié par Qt ne doit
    jamais exiger d'argument dans son constructeur (contrainte reprise de la
    source, ici respectée par prudence même si le widget est créé à la main).

    Signaux
    -------
    point_marked(str, object)
        Émis à chaque repère traité : ``(label, (x, y))`` s'il est marqué,
        ``(label, None)`` s'il est passé.
    tagging_finished(dict)
        Émis quand l'utilisateur termine (``Q``) : ``{label: (x, y) | None}``.
    instruction_changed(str)
        Émis à chaque changement de consigne — la fenêtre hôte l'affiche
        (barre d'état) en plus de la superposition sur l'image.
    """

    point_marked = pyqtSignal(str, object)
    tagging_finished = pyqtSignal(dict)
    instruction_changed = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._initialized = False
        self._labels: list[str] = list(DEFAULT_LABELS)
        self._points: list[tuple[str, tuple[float, float] | None]] = []
        self._current_index = 0
        # Dernière position souris, en **pixels** de l'image (pour dessiner) ;
        # convertie en pourcentage au moment de marquer.
        self._last_mouse_pos: tuple[float, float] | None = None
        # Dimensions de l'image affichée (largeur, hauteur) en pixels, pour les
        # conversions pixel ↔ pourcentage.  ``None`` tant qu'aucune image.
        self._img_w: int | None = None
        self._img_h: int | None = None
        self._annotation_items: list = []
        # Repères déjà présents dans les métadonnées, superposés dans une autre
        # couleur — gardés à part des annotations de pointage pour qu'un
        # nouveau cycle de désignation (data_setup) ne les efface pas.
        self._existing_items: list = []
        self.tags: dict[str, tuple[float, float] | None] = {}
        self.mode = ImageClickMode.READY_TO_BEGIN

    def setup(self, labels: list[str] | None = None) -> None:
        """Prépare le widget (une seule fois) : masque les boutons pyqtgraph superflus,
        active le focus clavier, installe la superposition de consigne et le
        suivi de la souris.  Appels ultérieurs ignorés.
        """
        if self._initialized:
            return
        self._initialized = True

        self.ui.roiBtn.hide()
        self.ui.menuBtn.hide()
        self.ui.histogram.hide()
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

        view = self.getView()
        view.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        view.scene().sigMouseMoved.connect(self._on_mouse_move)

        self._status = pg.TextItem(color="b", anchor=(0, 0))
        view.addItem(self._status)

        self.data_setup(labels)

    def data_setup(self, labels: list[str] | None = None) -> None:
        """Réinitialise l'état de désignation (repères à zéro), sans toucher à l'image.

        Appelée par :meth:`setup` puis à chaque nouveau marquage (bouton
        « Commencer » de la fenêtre hôte) et après un cycle terminé par ``Q``.
        """
        if labels is not None:
            self._labels = list(labels)
        self.mode = ImageClickMode.READY_TO_BEGIN
        self._points = []
        self._current_index = 0
        self._last_mouse_pos = None
        self._clear_annotations()
        self.tags = {}
        self._update_instruction_text()

    # ------------------------------------------------------------------
    # Affichage de l'image
    # ------------------------------------------------------------------

    def set_image(self, img: np.ndarray) -> None:
        """Affiche *img* « à l'endroit » et réinitialise la désignation.

        Transpose les deux premiers axes : pyqtgraph en ``col-major`` (défaut
        du reste de l'application) attend ``(x, y[, canal])``, alors qu'une
        image NumPy est ``(ligne, colonne[, canal])``.  Après transposition,
        ``mapSceneToView`` renvoie ``(x = colonne, y = ligne)``.
        """
        self._img_h, self._img_w = int(img.shape[0]), int(img.shape[1])
        if img.ndim == 2:
            display = img.T
        else:
            display = np.transpose(img, (1, 0, 2))
        self.setImage(display, autoLevels=False, levels=(0, 255))
        self.clear_existing_points()  # nouvelle image → repères superposés obsolètes
        self.data_setup(self._labels)

    def _px_to_pct(self, x: float, y: float) -> tuple[float, float]:
        """Pixel ``(x, y)`` → pourcentage ``(0–100)`` de la largeur/hauteur.

        Multiplication avant division (``x*100/w``) : reste exact pour un pixel
        entier quand la dimension divise ``x*100`` (évite ``7/100*100`` = 7.0000…1).
        """
        w = self._img_w or 1
        h = self._img_h or 1
        return (x * 100.0 / w, y * 100.0 / h)

    def _pct_to_px(self, px: float, py: float) -> tuple[float, float]:
        """Pourcentage ``(0–100)`` → pixel ``(x, y)`` pour le dessin sur la vue."""
        w = self._img_w or 1
        h = self._img_h or 1
        return (px / 100.0 * w, py / 100.0 * h)

    def show_existing_points(self, points: dict[str, tuple[float, float] | None], color="g") -> None:
        """Superpose des repères déjà connus (ex. lus dans les métadonnées).

        *points* est en **pourcentage** (0–100) ; converti en pixels pour le
        dessin.  Rendus dans une couleur distincte (*color*, vert par défaut) et
        avec un symbole différent (croix) de ceux désignés à la souris (cercles
        rouges), pour qu'on les distingue au premier coup d'œil.  Les points
        passés (``None``) ne sont pas dessinés.  Remplace tout affichage
        précédent de repères existants.
        """
        self.clear_existing_points()
        view = self.getView()
        for label, pt in points.items():
            if pt is None:
                continue
            x, y = self._pct_to_px(*pt)
            scatter = pg.ScatterPlotItem([x], [y], size=12, brush=color, symbol="x")
            view.addItem(scatter)
            text = pg.TextItem(label, color=color)
            text.setPos(x + 5, y - 10)
            view.addItem(text)
            self._existing_items.extend((scatter, text))

    def clear_existing_points(self) -> None:
        """Retire les repères superposés par :meth:`show_existing_points`."""
        view = self.getView()
        for item in self._existing_items:
            view.removeItem(item)
        self._existing_items.clear()

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def _on_mouse_move(self, pos) -> None:
        view = self.getView()
        if view.sceneBoundingRect().contains(pos):
            p = view.mapSceneToView(pos)
            self._last_mouse_pos = (p.x(), p.y())  # pixels (float)

    def _update_instruction_text(self) -> None:
        if self._current_index < len(self._labels):
            text = (
                f"Suivant : {self._labels[self._current_index]}\n"
                "Entrée = marquer | Espace = passer | Q = terminer"
            )
        else:
            text = "Tous les repères sont traités — appuyez sur Q pour terminer."
        self._status.setText(text)
        self.instruction_changed.emit(text.replace("\n", " · "))

    def keyPressEvent(self, ev) -> None:
        view = self.getView()
        key = ev.key()

        if key in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            self._mark_current(view)
        elif key == QtCore.Qt.Key.Key_Space:
            self._skip_current()
        elif key == QtCore.Qt.Key.Key_Q:
            self._finish()
        else:
            super().keyPressEvent(ev)

        self._update_instruction_text()

    def _mark_current(self, view) -> None:
        if self._current_index >= len(self._labels) or self._last_mouse_pos is None:
            return
        self.mode = ImageClickMode.SELECT_IN_PROGRESS
        x, y = self._last_mouse_pos            # pixels (pour dessiner)
        pct = self._px_to_pct(x, y)            # pourcentage (pour stocker/émettre)
        label = self._labels[self._current_index]
        self._points.append((label, pct))

        scatter = pg.ScatterPlotItem([x], [y], size=8, brush="r")
        view.addItem(scatter)
        text = pg.TextItem(label, color="r")
        text.setPos(x + 5, y - 10)
        view.addItem(text)
        self._annotation_items.extend((scatter, text))

        self._current_index += 1
        self.point_marked.emit(label, pct)

    def _skip_current(self) -> None:
        if self._current_index >= len(self._labels):
            self.mode = ImageClickMode.DONE
            return
        self.mode = ImageClickMode.SELECT_IN_PROGRESS
        label = self._labels[self._current_index]
        self._points.append((label, None))
        self._current_index += 1
        self.point_marked.emit(label, None)

    def _finish(self) -> None:
        self.mode = ImageClickMode.DONE
        self.tags = dict(self._points)
        self._clear_annotations()
        self._current_index = 0  # permet de recommencer
        self.tagging_finished.emit(self.tags)

    def _clear_annotations(self) -> None:
        view = self.getView()
        for item in self._annotation_items:
            view.removeItem(item)
        self._annotation_items.clear()

    def get_tags(self) -> dict[str, tuple[float, float] | None]:
        """Repères désignés, ``{label: (x, y) | None}`` en pourcentage (0–100)."""
        return self.tags

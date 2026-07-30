"""Désignation interactive de points nommés à la souris sur une image.

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
- **Coordonnées pixel correctes** : l'image est affichée transposée sur ses
  deux premiers axes (exigence d'affichage « à l'endroit » de pyqtgraph en
  ``col-major``, l'ordre par défaut du reste de l'application), ce qui fait
  que ``mapSceneToView`` renvoie directement ``(x = colonne, y = ligne)`` —
  les coordonnées pixel standard, prêtes à être stockées.
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
        self._points: list[tuple[str, tuple[int, int] | None]] = []
        self._current_index = 0
        self._last_mouse_pos: tuple[int, int] | None = None
        self._annotation_items: list = []
        self.tags: dict[str, tuple[int, int] | None] = {}
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
        if img.ndim == 2:
            display = img.T
        else:
            display = np.transpose(img, (1, 0, 2))
        self.setImage(display, autoLevels=False, levels=(0, 255))
        self.data_setup(self._labels)

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def _on_mouse_move(self, pos) -> None:
        view = self.getView()
        if view.sceneBoundingRect().contains(pos):
            p = view.mapSceneToView(pos)
            self._last_mouse_pos = (int(p.x()), int(p.y()))

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
        x, y = self._last_mouse_pos
        label = self._labels[self._current_index]
        self._points.append((label, (x, y)))

        scatter = pg.ScatterPlotItem([x], [y], size=8, brush="r")
        view.addItem(scatter)
        text = pg.TextItem(label, color="r")
        text.setPos(x + 5, y - 10)
        view.addItem(text)
        self._annotation_items.extend((scatter, text))

        self._current_index += 1
        self.point_marked.emit(label, (x, y))

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

    def get_tags(self) -> dict[str, tuple[int, int] | None]:
        """Repères désignés, sous la forme ``{label: (x, y) | None}``."""
        return self.tags

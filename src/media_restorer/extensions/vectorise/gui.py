"""Fenêtre Qt de l'extension Vectorise — analyse topologique et retraçage.

Mise en page définie dans ``views/main.ui`` (compilée automatiquement, même
mécanisme que :mod:`media_restorer.extensions.media_restorer.gui`).  Le cœur
de calcul (:mod:`media_restorer.engines.vectorise`) ne dépend d'aucun widget
Qt ; cette fenêtre se contente de l'orchestrer : charger l'image, lancer
``get_outline`` dans un thread de fond, peupler la liste des candidats,
animer une sélection, extraire/choisir une texture, vectoriser et enregistrer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pyqtgraph as pg
from pyqtgraph.parametertree import Parameter, ParameterTree
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

import media_restorer.engines.vectorise as vec
from media_restorer.app_settings import TOOLTIP_MODE_KEY, app_settings
from media_restorer.engines.vectorise.topology import px_to_mm
from media_restorer.engines.vectorise.types import StrokeSet
from media_restorer.gui_widgets import ResultWindow
from media_restorer.image_io import exif_dpi, imread_oriented
from media_restorer.power import performance_mode
from OutilsQt.Utils_Qt import compile_ui, compile_qrc, tooltips_from_code

_UI_SRC  = Path(__file__).parent / "views" / "main.ui"
_UI_PY   = Path(__file__).parent / "views" / "ui_main.py"
# resources/ est partagé au niveau racine du paquet, comme pour l'extension
# Media Restorer — voir sa docstring équivalente pour la même remontée à 2 niveaux.
_QRC_SRC = Path(__file__).parents[2] / "resources" / "icons" / "media_restorer.qrc"
_QRC_PY  = Path(__file__).parents[2] / "resources" / "icons" / "media_restorer_rc.py"

_TOOLTIP_TYPES: list[tuple] = [
    (QAction, "action", "triggered"),
]

compile_ui(_UI_SRC, _UI_PY)
compile_qrc(_QRC_SRC, _QRC_PY)

from media_restorer.resources.icons import media_restorer_rc as _rc  # noqa: F401, E402
from media_restorer.extensions.vectorise.views.ui_main import Ui_MainWindow  # noqa: E402

# Papier — couleur de fond de l'animation et de la vectorisation (voir
# render._DEFAULT_PAPER_COLOR, dupliqué ici pour l'aperçu incrémental
# de l'animation, qui ne passe pas par render.vectorise).
_PAPER_COLOR = 245
_ANIMATION_FRAME_BUDGET = 150

_PARAM_DEFS: list[dict] = [
    {
        "name": "n_candidates", "title": "Nombre de candidats",
        "type": "int", "value": 5, "limits": (1, 12), "step": 1,
    },
    {
        "name": "max_points", "title": "Points échantillonnés (max)",
        "type": "int", "value": 30_000, "limits": (1_000, 100_000), "step": 1_000,
    },
    {
        "name": "mark_fraction", "title": "Fraction sombre considérée",
        "type": "float", "value": 0.15, "limits": (0.01, 0.5), "step": 0.01,
    },
    {
        "name": "width_override", "title": "Forcer l'intervalle de largeur",
        "type": "bool", "value": False,
    },
    {
        "name": "width_min_mm", "title": "Largeur min (mm)",
        "type": "float", "value": 0.3, "limits": (0.05, 5.0), "step": 0.05,
    },
    {
        "name": "width_max_mm", "title": "Largeur max (mm)",
        "type": "float", "value": 1.5, "limits": (0.05, 5.0), "step": 0.05,
    },
    {
        "name": "max_width_changes_per_stroke", "title": "Changements de largeur / trait",
        "type": "int", "value": 4, "limits": (1, 10), "step": 1,
    },
    {
        "name": "animation_speed", "title": "Vitesse d'animation (images/s)",
        "type": "int", "value": 30, "limits": (5, 60), "step": 5,
    },
]


# ---------------------------------------------------------------------------
# Worker (thread d'arrière-plan)
# ---------------------------------------------------------------------------

class _GetOutlineWorker(QThread):
    """Lance ``get_outline`` (potentiellement plusieurs secondes) hors du thread UI.

    Signal nommé ``result_ready`` — pas ``finished``, qui shadowe
    ``QThread.finished`` (convention du projet, voir ``CLAUDE.md``).
    """

    result_ready = pyqtSignal(list)  # list[StrokeSet]
    error        = pyqtSignal(str)

    def __init__(
        self,
        image: np.ndarray,
        params: dict,
        get_outline_fn: Callable[..., list[StrokeSet]] | None = None,
    ) -> None:
        super().__init__()
        self._image = image
        self._params = params
        # Injectable pour les tests (voir CLAUDE.md : « passer une
        # _stream_factory/_engine_factory en paramètre des workers au lieu
        # d'importer directement ») — évite qu'un test doive attendre le
        # vrai calcul topologique pour vérifier le câblage des signaux.
        self._get_outline_fn = get_outline_fn or vec.get_outline

    def run(self) -> None:
        try:
            with performance_mode():
                result = self._get_outline_fn(self._image, **self._params)
            self.result_ready.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Fenêtre principale
# ---------------------------------------------------------------------------

class VectoriseGUI(QMainWindow):
    """Fenêtre « Vectorise » — analyse topologique et retraçage de dessins.

    Comme :class:`~media_restorer.extensions.media_restorer.gui.PhotoRestorationGUI`,
    ne choisit plus elle-même sa cible : le fichier à traiter lui est transmis
    à la construction par la fenêtre racine
    :class:`~media_restorer.gui_root.ImageTreatmentWindow`, via
    :class:`~media_restorer.extensions.vectorise.VectoriseExtension`.  Outil à
    image unique — un répertoire choisi dans la racine est signalé mais non
    traité (voir :meth:`_apply_target`).

    Paramètres
    ----------
    target_path : Path | None
        Fichier image à charger dès la construction.  ``None`` construit la
        fenêtre sans rien charger — utilisé par les tests et par tout usage
        direct de cette classe hors de la fenêtre racine.
    get_outline_fn : Callable | None
        Substitut de :func:`~media_restorer.engines.vectorise.get_outline`
        pour les tests (voir :class:`_GetOutlineWorker`).
    """

    def __init__(
        self,
        target_path: Path | None = None,
        get_outline_fn: Callable[..., list[StrokeSet]] | None = None,
    ) -> None:
        super().__init__()
        self._get_outline_fn = get_outline_fn

        self._target_path: Path | None = None
        self._original:    np.ndarray | None = None
        self._dpi:          float = 300.0
        self._strokesets:   list[StrokeSet] = []
        self._texture:      np.ndarray | None = None
        self._vectorised:   np.ndarray | None = None
        self._get_outline_worker: _GetOutlineWorker | None = None
        self._animation_state: dict | None = None

        # ── UI de base (.ui compilé) ────────────────────────────────────
        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)

        # Expose les QActions sur self pour tooltips_from_code (même
        # convention que PhotoRestorationGUI — voir sa docstring équivalente).
        self.actionGetOutline           = self._ui.actionGetOutline
        self.actionShow                 = self._ui.actionShow
        self.actionSaveStrokes          = self._ui.actionSaveStrokes
        self.actionSaveTextureFromImage = self._ui.actionSaveTextureFromImage
        self.actionSelectTexture        = self._ui.actionSelectTexture
        self.actionVectorise            = self._ui.actionVectorise
        self.actionSave                 = self._ui.actionSave

        self._image_view = pg.ImageView()
        self._image_view.ui.roiBtn.hide()
        self._image_view.ui.menuBtn.hide()
        self._ui.imageLayout.addWidget(self._image_view)

        # ── Dock Paramètres + liste des candidats ───────────────────────
        self._param_root = Parameter.create(name="params", type="group", children=_PARAM_DEFS)
        tree = ParameterTree(showHeader=False)
        tree.setParameters(self._param_root)

        self._combo_candidates = QComboBox()
        self._combo_candidates.setObjectName("comboCandidates")
        candidate_row = QWidget()
        row_layout = QHBoxLayout(candidate_row)
        row_layout.setContentsMargins(6, 2, 6, 2)
        row_layout.addWidget(QLabel("Candidat :"))
        row_layout.addWidget(self._combo_candidates)

        container = QWidget()
        vlayout = QVBoxLayout(container)
        vlayout.setContentsMargins(0, 0, 0, 0)
        vlayout.setSpacing(2)
        vlayout.addWidget(tree)
        vlayout.addWidget(candidate_row)

        dock = QDockWidget("Paramètres", self)
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        dock.setWidget(container)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        # ── Fenêtre de résultat (animation + vectorisation) ─────────────
        # preserve_zoom=True : comparer plusieurs candidats ou réglages sur
        # un même détail zoomé, comme l'onglet Double-exposition de Media
        # Restorer — perdre le zoom à chaque animation obligerait à
        # re-zoomer à chaque essai.
        self._result_window = ResultWindow("Vectorise", preserve_zoom=True)

        # ── Animation ────────────────────────────────────────────────────
        self._animation_timer = QTimer(self)
        self._animation_timer.timeout.connect(self._advance_animation)

        # ── Tooltips ─────────────────────────────────────────────────────
        tooltips_from_code(
            self, mode=app_settings().value(TOOLTIP_MODE_KEY, "docstrings"),
            liste_types_actions=_TOOLTIP_TYPES,
        )

        # ── Cible pré-chargée par la fenêtre racine ─────────────────────
        self._apply_target(target_path)

    # ------------------------------------------------------------------
    # Cible
    # ------------------------------------------------------------------

    def _apply_target(self, target_path: Path | None) -> None:
        if target_path is None:
            return
        if target_path.is_dir():
            self.statusBar().showMessage(
                "Vectorise ne traite qu'un fichier à la fois — "
                "choisissez une image depuis Image Treatment."
            )
            return
        img = imread_oriented(target_path)
        if img is None:
            QMessageBox.critical(self, "Erreur", f"Impossible de lire : {target_path}")
            return
        self._target_path = target_path
        self._original = img
        self._dpi = exif_dpi(target_path)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img
        self._image_view.setImage(rgb, autoLevels=False, levels=(0, 255))
        self._ui.actionGetOutline.setEnabled(True)
        self.statusBar().showMessage(f"Image chargée : {target_path} ({self._dpi:.0f} dpi)")

    def _read_params(self) -> dict:
        return {child.name(): child.value() for child in self._param_root.children()}

    # ------------------------------------------------------------------
    # get_outline
    # ------------------------------------------------------------------

    @pyqtSlot()
    def on_actionGetOutline_triggered(self) -> None:
        """Analyse la topologie de l'image et propose des tracés plausibles.

        Lance ``get_outline`` dans un thread d'arrière-plan (l'analyse peut
        prendre plusieurs secondes) : l'action est désactivée pendant le
        calcul puis réactivée à la fin. Peuple la liste des candidats et
        active « Animer »/« Enregistrer les tracés ».
        """
        self._start_get_outline()

    def _start_get_outline(self) -> None:
        if self._original is None:
            return
        params = self._read_params()
        kwargs = dict(
            dpi=self._dpi,
            n_candidates=params["n_candidates"],
            max_points=params["max_points"],
            mark_fraction=params["mark_fraction"],
        )
        self._ui.actionGetOutline.setEnabled(False)
        self.statusBar().showMessage("Analyse topologique en cours…")
        self._get_outline_worker = _GetOutlineWorker(self._original, kwargs, self._get_outline_fn)
        self._get_outline_worker.result_ready.connect(self._on_get_outline_done)
        self._get_outline_worker.error.connect(self._on_get_outline_error)
        self._get_outline_worker.start()

    def _on_get_outline_done(self, strokesets: list[StrokeSet]) -> None:
        self._ui.actionGetOutline.setEnabled(True)
        params = self._read_params()
        if params["width_override"] and strokesets:
            lo, hi = params["width_min_mm"], params["width_max_mm"]
            filtered = [
                s for s in strokesets if lo <= px_to_mm(s.pencil_width_px, self._dpi) <= hi
            ]
            # Un filtre qui ne laisse plus rien serait pire que pas de
            # filtre du tout : l'utilisateur perdrait toute la liste sans
            # recours autre que relancer le calcul.
            strokesets = filtered or strokesets

        self._strokesets = strokesets
        self._combo_candidates.clear()
        self._combo_candidates.addItems([s.label for s in strokesets])

        has_candidates = bool(strokesets)
        self._ui.actionShow.setEnabled(has_candidates)
        self._ui.actionSaveStrokes.setEnabled(has_candidates)
        self._update_vectorise_enabled()
        self.statusBar().showMessage(f"{len(strokesets)} candidat(s) trouvé(s).")

    def _on_get_outline_error(self, msg: str) -> None:
        self._ui.actionGetOutline.setEnabled(True)
        QMessageBox.critical(self, "Erreur — analyse topologique", msg)
        self.statusBar().showMessage("Échec de l'analyse topologique.")

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------

    @pyqtSlot()
    def on_actionShow_triggered(self) -> None:
        """Anime le tracé sélectionné dans la liste des candidats.

        Dessine le trait progressivement sur un canevas couleur papier, par
        lots de points calculés pour tenir en :data:`_ANIMATION_FRAME_BUDGET`
        images quel que soit le nombre de points du tracé — l'animation dure
        toujours quelques secondes, ni instantanée ni interminable.
        """
        self._start_animation()

    def _start_animation(self) -> None:
        if not self._strokesets or self._original is None:
            return
        index = self._combo_candidates.currentIndex()
        if index < 0:
            return
        stroke_set = self._strokesets[index]

        all_points = [
            (s_idx, p_idx)
            for s_idx, stroke in enumerate(stroke_set.strokes)
            for p_idx in range(len(stroke.points))
        ]
        if not all_points:
            self.statusBar().showMessage("Candidat sans points à animer.")
            return

        h, w = self._original.shape[:2]
        canvas = np.full((h, w), _PAPER_COLOR, dtype=np.uint8)
        step = max(1, -(-len(all_points) // _ANIMATION_FRAME_BUDGET))
        speed = max(1, self._read_params()["animation_speed"])

        self._animation_timer.stop()
        self._animation_state = {
            "stroke_set": stroke_set, "canvas": canvas, "points": all_points,
            "step": step, "pos": 0,
        }
        self._result_window.show_image(canvas)
        self._animation_timer.start(max(1, int(1000 / speed)))

    def _advance_animation(self) -> None:
        state = self._animation_state
        if state is None:
            self._animation_timer.stop()
            return
        points, pos, step = state["points"], state["pos"], state["step"]
        canvas, stroke_set = state["canvas"], state["stroke_set"]

        end = min(pos + step, len(points))
        for i in range(pos, end):
            s_idx, p_idx = points[i]
            stroke = stroke_set.strokes[s_idx]
            x, y = stroke.points[p_idx]
            radius = max(1, int(round(stroke.widths[p_idx])))
            level = int(_PAPER_COLOR * (1.0 - float(stroke.intensity[p_idx])))
            cv2.circle(canvas, (int(x), int(y)), radius, level, -1, lineType=cv2.LINE_AA)
        state["pos"] = end

        self._result_window.update_image(canvas)
        if end >= len(points):
            self._animation_timer.stop()
            self._animation_state = None

    # ------------------------------------------------------------------
    # Tracés — enregistrement
    # ------------------------------------------------------------------

    @pyqtSlot()
    def on_actionSaveStrokes_triggered(self) -> None:
        """Enregistre tous les candidats trouvés dans un fichier ``.strokes.h5``.

        Nommé d'après l'image source, suffixe ``.strokes.h5`` — voir
        :func:`~media_restorer.engines.vectorise.save_strokes`.
        """
        self._save_strokes()

    def _save_strokes(self) -> None:
        if not self._strokesets or self._target_path is None or self._original is None:
            return
        out_path = self._target_path.with_name(self._target_path.stem + ".strokes.h5")
        vec.save_strokes(
            out_path, self._strokesets,
            source_path=self._target_path, dpi=self._dpi,
            image_shape=self._original.shape[:2],
        )
        self.statusBar().showMessage(f"Tracés enregistrés : {out_path}")

    # ------------------------------------------------------------------
    # Texture
    # ------------------------------------------------------------------

    @pyqtSlot()
    def on_actionSaveTextureFromImage_triggered(self) -> None:
        """Extrait le grain des zones sombres de l'image et l'enregistre.

        Écrit ``<image>.texture.png`` et l'utilise immédiatement comme
        texture active pour « Vectoriser » — inutile de la sélectionner à
        nouveau après l'avoir extraite.
        """
        self._save_texture_from_image()

    def _save_texture_from_image(self) -> None:
        if self._original is None or self._target_path is None:
            return
        texture = vec.extract_texture(self._original)
        out_path = self._target_path.with_name(self._target_path.stem + ".texture.png")
        cv2.imwrite(str(out_path), texture)
        self._texture = texture
        self._update_vectorise_enabled()
        self.statusBar().showMessage(f"Texture enregistrée : {out_path}")

    @pyqtSlot()
    def on_actionSelectTexture_triggered(self) -> None:
        """Utilise un fichier de texture existant plutôt que celle extraite automatiquement.

        La texture choisie remplace immédiatement la texture active pour
        « Vectoriser », sans modifier quoi que ce soit sur le disque.
        """
        self._select_texture()

    def _select_texture(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choisir une texture", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if not path:
            return
        img = imread_oriented(path)
        if img is None:
            QMessageBox.critical(self, "Erreur", f"Impossible de lire : {path}")
            return
        self._texture = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        self._update_vectorise_enabled()
        self.statusBar().showMessage(f"Texture active : {path}")

    def _update_vectorise_enabled(self) -> None:
        self._ui.actionVectorise.setEnabled(bool(self._strokesets) and self._texture is not None)

    # ------------------------------------------------------------------
    # Vectorisation
    # ------------------------------------------------------------------

    @pyqtSlot()
    def on_actionVectorise_triggered(self) -> None:
        """Recompose une image à partir du tracé sélectionné et de la texture active.

        Rapide (moins d'une seconde même sur un scan de plusieurs dizaines
        de mégapixels — mesuré) : exécuté directement sur le thread UI, sans
        worker séparé.  Affiche le résultat et active « Enregistrer ».
        """
        self._start_vectorise()

    def _start_vectorise(self) -> None:
        if not self._strokesets or self._texture is None or self._original is None:
            return
        index = self._combo_candidates.currentIndex()
        if index < 0:
            return
        stroke_set = self._strokesets[index]
        params = self._read_params()
        result = vec.vectorise(
            stroke_set, self._original.shape[:2], self._texture,
            max_width_changes_per_stroke=params["max_width_changes_per_stroke"],
        )
        self._vectorised = result
        self._result_window.show_image(result)
        self._ui.actionSave.setEnabled(True)
        self.statusBar().showMessage("Vectorisation terminée.")

    @pyqtSlot()
    def on_actionSave_triggered(self) -> None:
        """Enregistre le résultat de la vectorisation dans un fichier choisi.

        N'est disponible qu'après une vectorisation réussie — même
        sémantique que ``actionSave`` de Media Restorer.
        """
        self._save_vectorised()

    def _save_vectorised(self) -> None:
        if self._vectorised is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Enregistrer", "", "Images (*.png *.jpg *.bmp)"
        )
        if path:
            cv2.imwrite(path, self._vectorised)
            self.statusBar().showMessage(f"Enregistré : {path}")

    # ------------------------------------------------------------------
    # Fermeture
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        """Arrête proprement le thread de calcul et l'animation en cours.

        Sans cela, fermer la fenêtre pendant un ``get_outline`` en cours
        produirait « QThread: Destroyed while thread is still running »
        (même raison que ``PhotoRestorationGUI.closeEvent``).
        """
        if self._get_outline_worker is not None and self._get_outline_worker.isRunning():
            self._get_outline_worker.terminate()
            self._get_outline_worker.wait()
        self._animation_timer.stop()
        self._result_window.close()
        super().closeEvent(event)

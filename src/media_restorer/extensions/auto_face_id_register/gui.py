"""Fenêtre Qt de l'extension Auto Face ID Register — détection puis revue.

Détecte les repères (yeux, nez…) sur un dessin/caricature via un modèle de
vision zero-shot (cœur :mod:`media_restorer.engines.face_id`, sans Qt), les
pré-affiche pour **revue/correction** à la souris (widget partagé
:class:`~media_restorer.image_click.ImageClick`), puis les écrit dans les
métadonnées (:class:`~media_restorer.landmarks.LandmarkSet`) après confirmation.

Les repères détectés sont les mêmes que ceux de la désignation manuelle : la
liste vient de :mod:`media_restorer.landmark_config` (partagée).  Le modèle de
détection, lui, est choisi au premier usage et mémorisé dans le TOML propre à
cette extension (:mod:`~media_restorer.extensions.auto_face_id_register.config`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from pyqtgraph.parametertree import Parameter, ParameterTree
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QDockWidget,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

import numpy as np

from media_restorer.app_settings import TOOLTIP_MODE_KEY, app_settings
from media_restorer.engines.face_id import CANDIDATE_MODELS
from media_restorer.extensions.auto_face_id_register import config as _model_config
from media_restorer.image_click import DEFAULT_LABELS, ImageClick
from media_restorer.image_io import imread_oriented
from media_restorer import landmark_config as _labels_config
from media_restorer.landmarks import ExiftoolRunner, LandmarkSet
from media_restorer.power import performance_mode
from OutilsQt.Utils_Qt import compile_ui, compile_qrc, tooltips_from_code

_UI_SRC  = Path(__file__).parent / "views" / "main.ui"
_UI_PY   = Path(__file__).parent / "views" / "ui_main.py"
_QRC_SRC = Path(__file__).parents[2] / "resources" / "icons" / "media_restorer.qrc"
_QRC_PY  = Path(__file__).parents[2] / "resources" / "icons" / "media_restorer_rc.py"

_TOOLTIP_TYPES: list[tuple] = [
    (QAction, "action", "triggered"),
]

# Couleur (cyan) des repères détectés superposés — distincte du rouge des
# repères re-désignés à la souris pendant la correction.
_DETECTED_COLOR = "c"

compile_ui(_UI_SRC, _UI_PY)
compile_qrc(_QRC_SRC, _QRC_PY)

from media_restorer.resources.icons import media_restorer_rc as _rc  # noqa: F401, E402
from media_restorer.extensions.auto_face_id_register.views.ui_main import Ui_MainWindow  # noqa: E402

# Fonction de détection : ``(image, labels) -> {label: (x, y) | None}``.
DetectFn = Callable[[np.ndarray, "list[str]"], "dict[str, tuple[int, int] | None]"]


# ---------------------------------------------------------------------------
# Worker (thread d'arrière-plan)
# ---------------------------------------------------------------------------

class _DetectWorker(QThread):
    """Lance la détection (téléchargement/inférence lents) hors du thread UI.

    Signal ``result_ready`` (pas ``finished``, qui shadowe ``QThread.finished``
    — convention du projet).
    """

    result_ready = pyqtSignal(dict)   # {label: (x, y) | None}
    error        = pyqtSignal(str)

    def __init__(self, image: np.ndarray, labels: list[str], detect_fn: DetectFn) -> None:
        super().__init__()
        self._image = image
        self._labels = labels
        self._detect_fn = detect_fn

    def run(self) -> None:
        try:
            with performance_mode():
                result = self._detect_fn(self._image, self._labels)
            self.result_ready.emit(result)
        except Exception as exc:  # transformers absent, modèle introuvable…
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Fenêtre principale
# ---------------------------------------------------------------------------

class AutoFaceIdRegisterGUI(QMainWindow):
    """Fenêtre « Auto Face ID Register » — détecte, fait revoir, puis enregistre.

    Paramètres
    ----------
    target_path : Path | None
        Image à charger dès la construction (``None`` pour les tests / usage
        direct).
    exiftool_runner : ExiftoolRunner | None
        Substitut du lanceur ``exiftool`` pour les tests (voir
        :mod:`media_restorer.landmarks`).
    detect_fn : DetectFn | None
        Substitut de la détection pour les tests — ``(image, labels) -> {label:
        (x,y)|None}``.  ``None`` (production) construit le détecteur réel à
        partir du modèle choisi (voir :meth:`_production_detect`).
    labels_config_path : Path | None
        TOML de la liste **partagée** de repères (défaut : emplacement standard).
    model_config_path : Path | None
        TOML **propre** à cette extension, mémorisant le modèle choisi.
    """

    _EMPTY_POINTS = "Aucun repère détecté."

    def __init__(
        self,
        target_path: Path | None = None,
        exiftool_runner: ExiftoolRunner | None = None,
        detect_fn: DetectFn | None = None,
        labels_config_path: Path | None = None,
        model_config_path: Path | None = None,
    ) -> None:
        super().__init__()
        self._exiftool_runner = exiftool_runner
        self._detect_fn = detect_fn
        self._labels_config_path = labels_config_path or _labels_config.config_path()
        self._model_config_path = model_config_path or _model_config.config_path()

        self._target_path: Path | None = None
        self._original: np.ndarray | None = None
        self._tags: dict[str, tuple[int, int] | None] = {}
        self._detect_worker: _DetectWorker | None = None
        # Détecteur réel mis en cache entre deux détections (chargement du modèle
        # coûteux) ; reconstruit si le modèle choisi change.
        self._detector = None
        self._detector_model: str | None = None
        self._detector_device: str | None = None
        self._resolved_model: str | None = None

        # ── UI de base ──────────────────────────────────────────────────
        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)
        self.actionDetect         = self._ui.actionDetect
        self.actionReview         = self._ui.actionReview
        self.actionSaveToMetadata = self._ui.actionSaveToMetadata
        self.actionChooseModel    = self._ui.actionChooseModel

        initial_labels = _labels_config.load_labels(DEFAULT_LABELS, path=self._labels_config_path)

        # ── Vue de pointage (partagée) ──────────────────────────────────
        self._image_click = ImageClick()
        self._image_click.setup(labels=list(initial_labels))
        self._image_click.tagging_finished.connect(self._on_review_finished)
        self._image_click.point_marked.connect(self._on_point_marked)
        self._image_click.instruction_changed.connect(self.statusBar().showMessage)
        self._ui.imageLayout.addWidget(self._image_click)

        # ── Dock Paramètres ─────────────────────────────────────────────
        self._param_root = Parameter.create(
            name="params", type="group",
            children=[{
                "name": "labels", "title": "Repères à détecter (un par ligne)",
                "type": "text", "value": "\n".join(initial_labels),
            }],
        )
        # Édition persistée dans la MÊME liste partagée que la désignation
        # manuelle (connexion après create() pour ne pas écrire la valeur initiale).
        self._param_root.child("labels").sigValueChanged.connect(self._save_labels)
        tree = ParameterTree(showHeader=False)
        tree.setParameters(self._param_root)

        self._points_label = QLabel(self._EMPTY_POINTS)
        self._points_label.setWordWrap(True)
        self._points_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._model_label = QLabel()
        self._model_label.setWordWrap(True)

        container = QWidget()
        vlayout = QVBoxLayout(container)
        vlayout.setContentsMargins(0, 0, 0, 0)
        vlayout.addWidget(tree)
        vlayout.addWidget(QLabel("<b>Repères détectés / corrigés :</b>"))
        vlayout.addWidget(self._points_label)
        vlayout.addWidget(self._model_label)
        vlayout.addStretch(1)
        dock = QDockWidget("Paramètres", self)
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        dock.setWidget(container)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        tooltips_from_code(
            self, mode=app_settings().value(TOOLTIP_MODE_KEY, "docstrings"),
            liste_types_actions=_TOOLTIP_TYPES,
        )
        self._refresh_model_label()
        self._apply_target(target_path)

    # ------------------------------------------------------------------
    # Cible / labels
    # ------------------------------------------------------------------

    def _apply_target(self, target_path: Path | None) -> None:
        if target_path is None:
            return
        if target_path.is_dir():
            self.statusBar().showMessage(
                "Auto Face ID Register ne traite qu'un fichier à la fois — "
                "choisissez une image depuis Image Treatment."
            )
            return
        img = imread_oriented(target_path)
        if img is None:
            QMessageBox.critical(self, "Erreur", f"Impossible de lire : {target_path}")
            return
        self._target_path = target_path
        self._original = img
        self._image_click.set_image(img)
        self._ui.actionDetect.setEnabled(True)
        self.statusBar().showMessage(f"Image chargée : {target_path}")

    def _read_labels(self) -> list[str]:
        raw = self._param_root.child("labels").value()
        labels = [line.strip() for line in raw.splitlines() if line.strip()]
        return labels or list(DEFAULT_LABELS)

    def _save_labels(self, *_args) -> None:
        _labels_config.save_labels(self._read_labels(), path=self._labels_config_path)

    @staticmethod
    def _format_points(points: dict[str, tuple[int, int] | None], empty_text: str) -> str:
        if not points:
            return empty_text
        return "\n".join(
            f"{label} : {'(non trouvé)' if pt is None else f'({pt[0]}, {pt[1]})'}"
            for label, pt in points.items()
        )

    def _render_points(self) -> None:
        self._points_label.setText(self._format_points(self._tags, self._EMPTY_POINTS))

    # ------------------------------------------------------------------
    # Choix du modèle
    # ------------------------------------------------------------------

    def _refresh_model_label(self) -> None:
        model = _model_config.load_model(path=self._model_config_path)
        device = _model_config.load_device(path=self._model_config_path)
        shown = model if model else "non choisi"
        self._model_label.setText(f"<i>Modèle : {shown} · appareil : {device.upper()}</i>")

    @pyqtSlot()
    def on_actionChooseModel_triggered(self) -> None:
        """Choisit (ou change) le modèle de détection, mémorisé pour la suite.

        Propose les modèles zero-shot connus ; un nom personnalisé (autre
        modèle Hugging Face) reste saisissable.  Le choix est enregistré dans
        le fichier de configuration de l'extension.
        """
        self._choose_model()

    def _choose_model(self) -> str | None:
        current = _model_config.load_model(path=self._model_config_path)
        items = list(CANDIDATE_MODELS)
        start = items.index(current) if current in items else 0
        name, ok = QInputDialog.getItem(
            self, "Modèle de détection",
            "Modèle zero-shot (téléchargé au premier usage, éditable) :",
            items, start, True,
        )
        if not ok or not name.strip():
            return None
        name = name.strip()

        # Appareil : CPU (fiable) ou GPU (peut se figer sur gfx1103).
        devices = list(_model_config.DEVICES)
        cur_dev = _model_config.load_device(path=self._model_config_path)
        dstart = devices.index(cur_dev) if cur_dev in devices else 0
        device, ok_dev = QInputDialog.getItem(
            self, "Appareil de détection",
            "CPU (fiable, recommandé) ou GPU (plus rapide, peut se figer sur cette carte) :",
            devices, dstart, False,
        )
        if not ok_dev:
            device = cur_dev

        _model_config.save_model(name, path=self._model_config_path)
        _model_config.save_device(device, path=self._model_config_path)
        # Invalide le cache : modèle ou appareil a pu changer.
        self._detector = None
        self._detector_model = None
        self._detector_device = None
        self._refresh_model_label()
        self.statusBar().showMessage(f"Modèle : {name} · appareil : {device.upper()}")
        return name

    # ------------------------------------------------------------------
    # Détection
    # ------------------------------------------------------------------

    @pyqtSlot()
    def on_actionDetect_triggered(self) -> None:
        """Détecte automatiquement les repères sur l'image.

        Choisit le modèle au premier usage si nécessaire, puis lance la
        détection dans un thread d'arrière-plan (téléchargement/inférence
        potentiellement longs).  Les repères trouvés sont superposés (croix
        cyan) et listés ; « Corriger » et « Enregistrer » deviennent
        disponibles.
        """
        self._start_detect()

    def _start_detect(self) -> None:
        if self._original is None:
            return
        if self._detect_fn is not None:
            detect_fn = self._detect_fn           # test : détection injectée
        else:
            model = _model_config.load_model(path=self._model_config_path) or self._choose_model()
            if model is None:
                self.statusBar().showMessage("Détection annulée — aucun modèle choisi.")
                return
            self._resolved_model = model
            detect_fn = self._production_detect

        self._ui.actionDetect.setEnabled(False)
        self.statusBar().showMessage("Détection en cours…")
        self._detect_worker = _DetectWorker(self._original, self._read_labels(), detect_fn)
        self._detect_worker.result_ready.connect(self._on_detect_done)
        self._detect_worker.error.connect(self._on_detect_error)
        self._detect_worker.start()

    def _production_detect(
        self, image: np.ndarray, labels: list[str]
    ) -> dict[str, tuple[int, int] | None]:
        """Détection réelle (dans le thread worker) : construit/réutilise le détecteur."""
        from media_restorer.engines import face_id

        device = _model_config.load_device(path=self._model_config_path)
        if (
            self._detector is None
            or self._detector_model != self._resolved_model
            or self._detector_device != device
        ):
            self._detector = face_id.build_detector(self._resolved_model, device=device)
            self._detector_model = self._resolved_model
            self._detector_device = device
        return face_id.detect_landmarks(image, labels, detector=self._detector)

    def _on_detect_done(self, points: dict) -> None:
        self._ui.actionDetect.setEnabled(True)
        self._tags = dict(points)
        self._image_click.show_existing_points(
            {k: v for k, v in points.items() if v is not None}, color=_DETECTED_COLOR
        )
        self._render_points()
        found = sum(1 for v in points.values() if v is not None)
        self._ui.actionReview.setEnabled(bool(points))
        self._ui.actionSaveToMetadata.setEnabled(bool(points))
        self.statusBar().showMessage(
            f"{found}/{len(points)} repère(s) détecté(s). "
            "Corrigez à la souris si besoin, puis enregistrez."
        )

    def _on_detect_error(self, msg: str) -> None:
        self._ui.actionDetect.setEnabled(True)
        QMessageBox.critical(self, "Erreur — détection", msg)
        self.statusBar().showMessage("Échec de la détection.")

    # ------------------------------------------------------------------
    # Revue / correction (réutilise le pointage manuel)
    # ------------------------------------------------------------------

    @pyqtSlot()
    def on_actionReview_triggered(self) -> None:
        """Revoit et corrige les repères détectés en les re-désignant à la souris.

        Démarre un cycle de désignation (survol + Entrée/Espace/Q).  Un repère
        **passé** (Espace) conserve sa valeur détectée ; seuls les repères
        re-marqués écrasent la détection.  Les croix cyan restent visibles comme
        référence.
        """
        self._start_review()

    def _start_review(self) -> None:
        if self._original is None or not self._tags:
            return
        self._image_click.data_setup(labels=self._read_labels())
        self._image_click.setFocus()
        self.statusBar().showMessage(
            "Correction : survol + Entrée (marquer) / Espace (garder la détection) / Q (terminer)."
        )

    def _on_point_marked(self, label: str, point: object) -> None:
        # Ne rien écraser sur un « passer » (Espace) : la détection est conservée.
        if point is not None:
            self._tags[label] = point
            self._render_points()
            self.statusBar().showMessage(f"{label} : ({point[0]}, {point[1]})")

    def _on_review_finished(self, tags: dict) -> None:
        # Fusionne : les repères re-marqués écrasent, les repères passés gardent
        # la valeur détectée (déjà dans self._tags via _on_point_marked).
        self._render_points()
        self._ui.actionSaveToMetadata.setEnabled(bool(self._tags))
        marked = sum(1 for v in tags.values() if v is not None)
        self.statusBar().showMessage(
            f"Correction terminée — {marked} repère(s) re-marqué(s). Enregistrez si c'est bon."
        )

    # ------------------------------------------------------------------
    # Écriture dans les métadonnées
    # ------------------------------------------------------------------

    @pyqtSlot()
    def on_actionSaveToMetadata_triggered(self) -> None:
        """Écrit les repères (détectés/corrigés) dans les métadonnées, après confirmation.

        Modifie le fichier image (tag ``UserComment`` via ``exiftool``, copie
        ``_original`` conservée) : demande donc confirmation d'abord.
        """
        self._save_to_metadata()

    def _save_to_metadata(self) -> None:
        if not self._tags or self._target_path is None:
            return
        found = sum(1 for v in self._tags.values() if v is not None)
        reply = QMessageBox.question(
            self, "Écrire dans les métadonnées ?",
            f"Écrire {found} repère(s) dans les métadonnées de :\n"
            f"{self._target_path.name} ?\n\n"
            "Le fichier image sera modifié (une copie « _original » est conservée).",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            self.statusBar().showMessage("Écriture annulée.")
            return
        try:
            LandmarkSet(points=dict(self._tags)).write_to_metadata(
                self._target_path, runner=self._exiftool_runner
            )
        except Exception as exc:
            QMessageBox.critical(self, "Erreur — écriture des métadonnées", str(exc))
            self.statusBar().showMessage("Échec de l'écriture des métadonnées.")
            return
        self.statusBar().showMessage(
            f"Repères enregistrés dans les métadonnées : {self._target_path.name}"
        )

    # ------------------------------------------------------------------
    # Fermeture
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        """Arrête proprement un thread de détection en cours avant de fermer."""
        if self._detect_worker is not None and self._detect_worker.isRunning():
            self._detect_worker.terminate()
            self._detect_worker.wait()
        super().closeEvent(event)

"""PyQt6 GUI for interactive photo restoration.

Mise en page et icônes définis dans views/main.ui (compilé automatiquement).
Ressources compilées depuis resources/icons/media_restorer.qrc via rcc Qt6.
Les tooltips des actions et des onglets sont générés automatiquement depuis
les docstrings et le code source (OutilsQt.Utils_Qt.tooltips_from_code).
"""

from __future__ import annotations

import inspect
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pyqtgraph as pg
from pyqtgraph.parametertree import Parameter, ParameterTree
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from media_restorer.colab_calc import ColabCalc
from media_restorer.engines import ENGINE_PARAMS, Engine, build_engine
from media_restorer.download_models import MODEL_REGISTRY
from media_restorer.power import performance_mode
from OutilsQt.Utils_Qt import compile_ui, compile_qrc, tooltips_from_code

pg.setConfigOption("imageAxisOrder", "row-major")

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

# Noms des sous-répertoires de sortie à exclure lors d'un parcours récursif.
_ENGINE_NAMES: frozenset[str] = frozenset(e.value for e in Engine)


def _write_params_toml(out_dir: Path, params: dict) -> None:
    """Écrit *out_dir/used_parameters.toml* avec les paramètres du batch.

    Si *params* est vide, le fichier contient uniquement un commentaire
    ``# no parameters``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    toml_path = out_dir / "used_parameters.toml"
    if not params:
        toml_path.write_text("# no parameters\n", encoding="utf-8")
        return
    lines: list[str] = []
    for k, v in params.items():
        if isinstance(v, bool):
            lines.append(f"{k} = {str(v).lower()}")
        elif isinstance(v, str):
            escaped = v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            lines.append(f'"{k}" = "{escaped}"')
        else:
            lines.append(f"{k} = {v}")
    toml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _image_files(directory: Path, *, recursive: bool) -> list[Path]:
    """Liste triée des images dans *directory*.

    En mode récursif, exclut les sous-répertoires de premier niveau dont le
    nom correspond à un moteur (sorties de traitements précédents).
    """
    candidates = directory.rglob("*") if recursive else directory.iterdir()
    result = []
    for p in candidates:
        if not p.is_file() or p.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        rel = p.relative_to(directory)
        if recursive and len(rel.parts) > 1 and rel.parts[0] in _ENGINE_NAMES:
            continue
        result.append(p)
    return sorted(result)

_UI_SRC  = Path(__file__).parent / "views" / "main.ui"
_UI_PY   = Path(__file__).parent / "views" / "ui_main.py"
_QRC_SRC = Path(__file__).parent / "resources" / "icons" / "media_restorer.qrc"
_QRC_PY  = Path(__file__).parent / "resources" / "icons" / "media_restorer_rc.py"

# Types de widgets inspectés par tooltips_from_code.
# Chaque entrée : (type_Qt, préfixe_du_nom, signal).
# Les onglets du QTabWidget sont gérés séparément par _set_tab_tooltips().
_TOOLTIP_TYPES: list[tuple] = [
    (QAction, "action", "triggered"),
]

compile_ui(_UI_SRC, _UI_PY)
compile_qrc(_QRC_SRC, _QRC_PY)

from media_restorer.resources.icons import media_restorer_rc as _rc  # noqa: F401, E402
from media_restorer.views.ui_main import Ui_MainWindow               # noqa: E402


# ---------------------------------------------------------------------------
# Fenêtre de résultat indépendante (une par moteur)
# ---------------------------------------------------------------------------

class ResultWindow(QMainWindow):
    """Affiche l'image restaurée dans une fenêtre portant le nom du moteur."""

    def __init__(self, engine_name: str) -> None:
        super().__init__()
        self.setWindowTitle(engine_name)
        self.resize(900, 700)
        self._view = pg.ImageView()
        self._view.ui.roiBtn.hide()
        self._view.ui.menuBtn.hide()
        self.setCentralWidget(self._view)

    def show_image(self, img_bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if img_bgr.ndim == 3 else img_bgr
        self._view.setImage(rgb, autoLevels=False, levels=(0, 255))
        self.show()
        self.raise_()
        self.activateWindow()


# ---------------------------------------------------------------------------
# Workers (threads d'arrière-plan)
# ---------------------------------------------------------------------------

class _RestoreWorker(QThread):
    result_ready = pyqtSignal(np.ndarray)
    error        = pyqtSignal(str)

    def __init__(
        self,
        img: np.ndarray,
        engine: Engine,
        model_path: Path | None,
        params: dict,
    ) -> None:
        super().__init__()
        self._img        = img
        self._engine     = engine
        self._model_path = model_path
        self._params     = params

    def run(self) -> None:
        try:
            eng = build_engine(self._engine, self._model_path, self._params)
            with performance_mode():
                result = eng.restore_array(self._img)
            self.result_ready.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class _BatchRestoreWorker(QThread):
    image_done = pyqtSignal(str, float)  # filename, wall_seconds
    all_done   = pyqtSignal(int, int)    # n_success, n_total
    file_error = pyqtSignal(str, str)    # filename, message

    def __init__(
        self,
        directory: Path,
        engine: Engine,
        model_path: Path | None,
        params: dict,
        recursive: bool = False,
    ) -> None:
        super().__init__()
        self._dir        = directory
        self._engine     = engine
        self._model_path = model_path
        self._params     = params
        self._recursive  = recursive

    def run(self) -> None:
        images    = _image_files(self._dir, recursive=self._recursive)
        out_dir   = self._dir / self._engine.value
        n_success = 0
        try:
            eng = build_engine(self._engine, self._model_path, self._params)
        except Exception as exc:
            self.file_error.emit("(init moteur)", str(exc))
            self.all_done.emit(0, len(images))
            return

        _write_params_toml(out_dir, self._params)

        import torch
        _gpu = torch.cuda.is_available()

        with performance_mode():
            for img_path in images:
                out_path = out_dir / img_path.relative_to(self._dir)
                t0       = time.monotonic()
                try:
                    eng.restore_file(img_path, out_path)
                    wall_elapsed = time.monotonic() - t0
                    n_success  += 1
                    self.image_done.emit(img_path.name, wall_elapsed)
                except Exception as exc:
                    self.file_error.emit(img_path.name, str(exc))
                if _gpu:
                    # Attend la fin des kernels GPU puis laisse 50 ms au
                    # compositor KDE pour ses flips vsync — évite le timeout
                    # DRM (flip_done timedout) sur iGPU partagé (780M).
                    torch.cuda.synchronize()
                    time.sleep(0.05)
        self.all_done.emit(n_success, len(images))


# ---------------------------------------------------------------------------
# Téléchargement des modèles
# ---------------------------------------------------------------------------

class _DownloadWorker(QThread):
    """Thread de téléchargement des modèles manquants."""

    file_start    = pyqtSignal(str, int)   # name, total_bytes
    file_progress = pyqtSignal(int, int)   # done_bytes, total_bytes
    file_done     = pyqtSignal(str, bool)  # name, was_skipped
    file_error    = pyqtSignal(str, str)   # name, error_msg
    all_done      = pyqtSignal(int, int)   # n_downloaded, n_total

    def run(self) -> None:
        n_downloaded = 0
        for name, dest, url in MODEL_REGISTRY:
            if dest.exists():
                self.file_done.emit(name, True)
                continue
            self.file_start.emit(name, 0)
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)

                def _hook(block_num: int, block_size: int, total: int) -> None:
                    done = min(block_num * block_size,
                               total if total > 0 else block_num * block_size)
                    self.file_progress.emit(done, max(total, 1))

                urllib.request.urlretrieve(url, tmp, reporthook=_hook)
                tmp.rename(dest)
                n_downloaded += 1
                self.file_done.emit(name, False)
            except Exception as exc:
                tmp.unlink(missing_ok=True)
                self.file_error.emit(name, str(exc))
        self.all_done.emit(n_downloaded, len(MODEL_REGISTRY))


class _DownloadDialog(QDialog):
    """Boîte de dialogue de progression du téléchargement des modèles."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Téléchargement des modèles")
        self.setMinimumWidth(540)
        self.setModal(True)

        layout = QVBoxLayout(self)

        self._lbl_file = QLabel("Prêt.")
        layout.addWidget(self._lbl_file)

        self._bar_file = QProgressBar()
        self._bar_file.setRange(0, 100)
        layout.addWidget(self._bar_file)

        self._lbl_overall = QLabel()
        layout.addWidget(self._lbl_overall)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(300)
        layout.addWidget(self._log)

        self._btn_close = QPushButton("Fermer")
        self._btn_close.setEnabled(False)
        self._btn_close.clicked.connect(self.accept)
        layout.addWidget(
            self._btn_close, 0, Qt.AlignmentFlag.AlignRight
        )

        self._n_done  = 0
        self._n_total = len(MODEL_REGISTRY)
        self._refresh_overall()

    def _refresh_overall(self) -> None:
        self._lbl_overall.setText(
            f"Progression : {self._n_done} / {self._n_total} modèles"
        )

    def on_file_start(self, name: str, total: int) -> None:
        self._lbl_file.setText(f"Téléchargement : {name}")
        self._bar_file.setValue(0)
        self._log.appendPlainText(f"↓ {name}…")

    def on_file_progress(self, done: int, total: int) -> None:
        if total > 0:
            self._bar_file.setValue(done * 100 // total)

    def on_file_done(self, name: str, skipped: bool) -> None:
        self._n_done += 1
        self._refresh_overall()
        self._bar_file.setValue(100)
        msg = (
            f"✓ {name} (déjà présent)"
            if skipped
            else f"✓ {name} téléchargé"
        )
        self._log.appendPlainText(msg)

    def on_file_error(self, name: str, msg: str) -> None:
        self._n_done += 1
        self._refresh_overall()
        self._log.appendPlainText(f"✗ {name} : {msg}")

    def on_all_done(self, n_downloaded: int, n_total: int) -> None:
        self._lbl_file.setText("Terminé.")
        self._btn_close.setEnabled(True)
        self._log.appendPlainText(
            f"\nFini — {n_downloaded} modèle(s) téléchargé(s) sur {n_total}."
        )


# ---------------------------------------------------------------------------
# Fenêtre principale
# ---------------------------------------------------------------------------

class PhotoRestorationGUI(ColabCalc, QMainWindow):
    """Fenêtre principale de l'application de restauration photo.

    Charge une image (ou un répertoire), choisit un moteur de restauration
    parmi les onglets du panneau de paramètres, lance la restauration dans
    un thread de fond, puis affiche le résultat dans une fenêtre dédiée.

    Les tooltips des actions et des onglets sont générés automatiquement à
    partir des docstrings (ou du code source) via
    :func:`OutilsQt.Utils_Qt.tooltips_from_code`.  Le mode est sélectionnable
    via le comboBox du dock « Tooltips » en bas de la fenêtre.
    """

    def __init__(self, model_path: Path | None = None) -> None:
        super().__init__()
        self._model_path      = model_path
        self._original:       np.ndarray | None = None
        self._restored:       np.ndarray | None = None
        self._worker:         _RestoreWorker      | None = None
        self._batch_worker:   _BatchRestoreWorker | None = None
        self._pending_engine: Engine | None = None
        self._pending_batch:  Engine | None = None
        self._open_windows:   list[QWidget]   = []

        # ── UI de base (.ui compilé — icônes chargées via ressources Qt) ──
        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)

        # Expose les QActions directement sur self pour tooltips_from_code.
        # setupUi(self) les crée sur self._ui ; on les copie pour que
        # inspect.getmembers(self) les trouve sous leur nom "action*".
        self.actionOpen          = self._ui.actionOpen
        self.actionRestore       = self._ui.actionRestore
        self.actionSave          = self._ui.actionSave
        self.actionBatch         = self._ui.actionBatch
        self.actionColabConnect  = self._ui.actionColabConnect
        self.actionColabRestore  = self._ui.actionColabRestore

        # ── pg.ImageView — image originale, ajouté dans imageContainer ───
        self._image_view = pg.ImageView()
        self._image_view.ui.roiBtn.hide()
        self._image_view.ui.menuBtn.hide()
        self._ui.imageLayout.addWidget(self._image_view)

        # ── DockWidget Paramètres ─────────────────────────────────────────
        self._tab_engines:    list[Engine]               = list(Engine)
        self._param_roots:    dict[Engine, Parameter]    = {}
        self._result_windows: dict[Engine, ResultWindow] = {}

        tab_widget = QTabWidget()
        self._tab_widget = tab_widget

        for engine in self._tab_engines:
            param_defs = ENGINE_PARAMS.get(engine, [])
            root = Parameter.create(
                name="params", type="group", children=param_defs
            )
            self._param_roots[engine] = root
            tree = ParameterTree(showHeader=False)
            tree.setParameters(root)
            tab_widget.addTab(tree, engine.value)
            self._result_windows[engine] = ResultWindow(engine.value)

        # Combo récursif + container
        self._combo_recursive = QComboBox()
        self._combo_recursive.setObjectName("comboRecursive")
        self._combo_recursive.addItems(["non récursif", "récursif"])
        bottom_bar = QWidget()
        hl = QHBoxLayout(bottom_bar)
        hl.setContentsMargins(6, 2, 6, 2)
        hl.addWidget(QLabel("Répertoire :"))
        hl.addWidget(self._combo_recursive)
        hl.addStretch()

        container = QWidget()
        vl = QVBoxLayout(container)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(2)
        vl.addWidget(tab_widget)
        vl.addWidget(bottom_bar)

        dock = QDockWidget("Paramètres des moteurs", self)
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
        )
        dock.setWidget(container)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        # ── Tooltips — mode initial depuis comboBox ────────────────────
        self._setup_tooltips(self._ui.comboTooltipMode.currentText())

        # ── Badge de calcul (droite de la statusBar) ──────────────────
        self._compute_badge = QLabel()
        self._compute_badge.setContentsMargins(0, 0, 4, 0)
        self.statusBar().addPermanentWidget(self._compute_badge)
        self._refresh_compute_badge()

    # ------------------------------------------------------------------
    # Badge de calcul
    # ------------------------------------------------------------------

    def _refresh_compute_badge(self) -> None:
        """Met à jour le badge GPU/CPU/Colab à droite de la barre de statut."""
        import torch
        if self._colab__get_url() is not None:
            text  = "Colab ☁"
            style = "color:#0055aa; background:#ddeeff; border:1px solid #0055aa;"
        elif torch.cuda.is_available():
            name  = (torch.cuda.get_device_name(0)
                     .replace("AMD Radeon ", "")
                     .replace(" Graphics", "")
                     .strip())
            text  = f"GPU · {name}"
            style = "color:#1a6b1a; background:#e0f5e0; border:1px solid #4caf50;"
        else:
            text  = "CPU"
            style = "color:#555555; background:#f0f0f0; border:1px solid #aaaaaa;"
        self._compute_badge.setText(text)
        self._compute_badge.setStyleSheet(
            f"QLabel {{ {style} border-radius:3px; padding:1px 6px;"
            f" font-size:11px; font-weight:bold; }}"
        )

    # ------------------------------------------------------------------
    # Propriétés
    # ------------------------------------------------------------------

    @property
    def _current_engine(self) -> Engine:
        return self._tab_engines[self._tab_widget.currentIndex()]

    def _read_params(self, engine: Engine) -> dict:
        """Lire les valeurs courantes du ParameterTree d'un moteur."""
        return {
            child.name(): child.value()
            for child in self._param_roots[engine].children()
        }

    # ------------------------------------------------------------------
    # Handlers d'actions — nommés on_<widget>_<signal> pour tooltips_from_code
    # ------------------------------------------------------------------

    @pyqtSlot()
    def on_actionOpen_triggered(self) -> None:
        """Ouvre un fichier image (PNG, JPEG, BMP, TIFF) et l'affiche dans la vue principale.

        Déclenche une boîte de dialogue de sélection de fichier. L'image est
        lue en BGR via OpenCV puis affichée en RGB dans pg.ImageView. L'action
        Restaurer est activée et l'action Enregistrer désactivée jusqu'à la
        prochaine restauration réussie.
        """
        self._load_image()

    @pyqtSlot()
    def on_actionRestore_triggered(self) -> None:
        """Lance la restauration de l'image chargée avec le moteur de l'onglet actif.

        La restauration s'exécute dans un thread d'arrière-plan (_RestoreWorker)
        afin de ne pas bloquer l'interface. Le bouton est désactivé pendant le
        traitement puis réactivé à la fin. Le résultat s'affiche dans une fenêtre
        indépendante (ResultWindow) portant le nom du moteur.
        """
        self._start_restore()

    @pyqtSlot()
    def on_actionSave_triggered(self) -> None:
        """Enregistre l'image restaurée dans un fichier choisi par l'utilisateur.

        Formats supportés : PNG, JPEG, BMP. N'est disponible qu'après une
        restauration réussie. Utilise cv2.imwrite en BGR natif (pas de
        conversion de couleur).
        """
        self._save_image()

    @pyqtSlot()
    def on_actionBatch_triggered(self) -> None:
        """Traite en lot toutes les images d'un répertoire choisi.

        L'utilisateur sélectionne un répertoire ; toutes les images trouvées
        sont restaurées séquentiellement dans un thread dédié
        (_BatchRestoreWorker). Les résultats sont sauvegardés dans un
        sous-répertoire nommé d'après le moteur. La barre d'état affiche
        le nom du dernier fichier traité et le temps CPU consommé.
        """
        self._start_batch()

    # ------------------------------------------------------------------
    # Colab
    # ------------------------------------------------------------------

    @pyqtSlot()
    def on_actionColabConnect_triggered(self) -> None:
        """Configure l'URL du tunnel cloudflared pointant vers le serveur Colab.

        Colle l'URL affichée par la cellule « tunnel » du notebook après
        démarrage.  Vérifie la connexion via GET /health et active l'action
        « Restaurer (Colab) » si une image est déjà chargée.
        """
        url, ok = QInputDialog.getText(
            self, "Connexion Colab", "URL du tunnel cloudflared :"
        )
        if not ok or not url.strip():
            return
        self._colab__set_url(url.strip())
        if self._colab__is_connected():
            self.statusBar().showMessage("Colab : connecté ✓")
            self._ui.actionColabRestore.setEnabled(self._original is not None)
            self._refresh_compute_badge()
        else:
            self._colab__set_url("")          # URL invalide — on efface
            self._refresh_compute_badge()
            self.statusBar().showMessage("Colab : URL inaccessible ✗")
            QMessageBox.warning(
                self,
                "Colab",
                f"Impossible de joindre :\n{url.strip()}\n\n(GET /health a échoué)",
            )

    @pyqtSlot()
    def on_actionColabRestore_triggered(self) -> None:
        """Lance la restauration via le serveur Google Colab (GPU).

        Même comportement que la restauration locale : thread d'arrière-plan,
        résultat affiché dans la ResultWindow du moteur sélectionné, action
        Enregistrer activée en cas de succès.
        """
        if self._original is None:
            return
        self._pending_engine = self._current_engine
        self._ui.actionColabRestore.setEnabled(False)
        self.statusBar().showMessage(
            f"Restauration Colab en cours ({self._pending_engine.value})…"
        )
        self._colab__start_restore(
            self._original,
            self._pending_engine.value,
            self._on_colab_restore_done,
            self._on_colab_restore_error,
        )

    def _on_colab_restore_done(self, result: np.ndarray) -> None:
        self._restored = result
        win = self._result_windows[self._pending_engine]
        self._register_window(win)
        win.show_image(result)
        self._ui.actionSave.setEnabled(True)
        self._ui.actionColabRestore.setEnabled(True)
        self.statusBar().showMessage("Restauration Colab terminée.")

    def _on_colab_restore_error(self, msg: str) -> None:
        self._ui.actionColabRestore.setEnabled(True)
        QMessageBox.critical(self, f"Erreur Colab — {self._pending_engine.value}", msg)
        self.statusBar().showMessage("Échec de la restauration Colab.")

    # ------------------------------------------------------------------
    # Tooltips
    # ------------------------------------------------------------------

    def _setup_tooltips(self, mode: str) -> None:
        """Met à jour les tooltips actions (via tooltips_from_code) et onglets."""
        tooltips_from_code(self, mode=mode, liste_types_actions=_TOOLTIP_TYPES)
        self._set_tab_tooltips()

    def _set_tab_tooltips(self) -> None:
        """Applique la docstring de chaque classe moteur comme tooltip de son onglet."""
        from media_restorer.engines.realesrgan_engine import RealESRGANEngine
        from media_restorer.engines.swinir_engine import SwinIREngine
        from media_restorer.engines.lama_engine import LaMaEngine
        from media_restorer.engines.gfpgan_engine import GFPGANEngine
        _cls: dict[Engine, type] = {
            Engine.REAL_ESRGAN: RealESRGANEngine,
            Engine.SWINIR:      SwinIREngine,
            Engine.LAMA:        LaMaEngine,
            Engine.GFPGAN:      GFPGANEngine,
        }
        for i, engine in enumerate(self._tab_engines):
            doc = inspect.cleandoc(_cls[engine].__doc__ or engine.value)
            self._tab_widget.setTabToolTip(i, doc)

    @pyqtSlot(str)
    def on_comboTooltipMode_currentTextChanged(self, mode: str) -> None:
        self._setup_tooltips(mode)

    # ------------------------------------------------------------------
    # Suivi des fenêtres enfants
    # ------------------------------------------------------------------

    def _register_window(self, w: QWidget) -> QWidget:
        """Enregistre *w* pour fermeture automatique à la sortie.

        Appelé chaque fois qu'une fenêtre secondaire est affichée pour la
        première fois.  ``closeEvent`` itère sur cette liste et appelle
        ``close()`` sur chaque entrée avant de fermer la fenêtre principale.
        """
        if w not in self._open_windows:
            self._open_windows.append(w)
        return w

    # ------------------------------------------------------------------
    # Chargement / affichage de l'image originale
    # ------------------------------------------------------------------

    def _load_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Ouvrir une image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            QMessageBox.critical(self, "Erreur", f"Impossible de lire : {path}")
            return
        self._original = img
        self._restored = None
        self._ui.actionSave.setEnabled(False)
        self._ui.actionRestore.setEnabled(True)
        self._ui.actionColabRestore.setEnabled(self._colab__get_url() is not None)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img
        self._image_view.setImage(rgb, autoLevels=False, levels=(0, 255))
        self.statusBar().showMessage(f"Image chargée : {path}")

    # ------------------------------------------------------------------
    # Restauration image simple
    # ------------------------------------------------------------------

    def _start_restore(self) -> None:
        if self._original is None:
            return
        self._pending_engine = self._current_engine
        params = self._read_params(self._pending_engine)
        self._ui.actionRestore.setEnabled(False)
        self.statusBar().showMessage(
            f"Restauration en cours ({self._pending_engine.value})…"
        )
        self._worker = _RestoreWorker(
            self._original, self._pending_engine, self._model_path, params
        )
        self._worker.result_ready.connect(self._on_restore_done)
        self._worker.error.connect(self._on_restore_error)
        self._worker.start()

    def _on_restore_done(self, result: np.ndarray) -> None:
        self._restored = result
        win = self._result_windows[self._pending_engine]
        self._register_window(win)
        win.show_image(result)
        self._ui.actionSave.setEnabled(True)
        self._ui.actionRestore.setEnabled(True)
        self.statusBar().showMessage("Restauration terminée.")

    def _on_restore_error(self, msg: str) -> None:
        self._ui.actionRestore.setEnabled(True)
        QMessageBox.critical(self, f"Erreur {self._pending_engine.value}", msg)
        self.statusBar().showMessage("Échec de la restauration.")

    # ------------------------------------------------------------------
    # Enregistrement
    # ------------------------------------------------------------------

    def _save_image(self) -> None:
        if self._restored is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Enregistrer", "", "Images (*.png *.jpg *.bmp)"
        )
        if path:
            cv2.imwrite(path, self._restored)
            self.statusBar().showMessage(f"Enregistré : {path}")

    # ------------------------------------------------------------------
    # Traitement par répertoire
    # ------------------------------------------------------------------

    def _start_batch(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Choisir un répertoire")
        if not directory:
            return
        dir_path  = Path(directory)
        recursive = self._combo_recursive.currentIndex() == 1
        images    = _image_files(dir_path, recursive=recursive)
        if not images:
            QMessageBox.information(self, "Répertoire vide", "Aucune image trouvée.")
            return
        self._pending_batch    = self._current_engine
        self._batch_done       = 0
        self._batch_total      = len(images)
        self._batch_elapsed    = 0.0        # somme des durées observées
        params = self._read_params(self._pending_batch)
        self._ui.actionBatch.setEnabled(False)
        self.statusBar().showMessage(
            f"0 / {self._batch_total} — {self._pending_batch.value} en cours…"
        )
        self._batch_worker = _BatchRestoreWorker(
            dir_path, self._pending_batch, self._model_path, params, recursive
        )
        self._batch_worker.image_done.connect(self._on_batch_image_done)
        self._batch_worker.file_error.connect(self._on_batch_file_error)
        self._batch_worker.all_done.connect(self._on_batch_all_done)
        self._batch_worker.start()

    def _on_batch_image_done(self, filename: str, wall_seconds: float) -> None:
        self._batch_done    += 1
        self._batch_elapsed += wall_seconds
        avg = self._batch_elapsed / self._batch_done
        self.statusBar().showMessage(
            f"{self._batch_done} / {self._batch_total} images"
            f"  —  moy. {avg:.1f} s/img"
            f"  —  {filename}"
        )

    def _on_batch_file_error(self, filename: str, msg: str) -> None:
        self.statusBar().showMessage(f"Erreur sur {filename} : {msg}")

    def _on_batch_all_done(self, n_success: int, n_total: int) -> None:
        self._ui.actionBatch.setEnabled(True)
        self.statusBar().showMessage(
            f"Terminé — {n_success}/{n_total} image(s)"
            f" → sous-répertoire {self._pending_batch.value}"
        )


    @pyqtSlot()
    def on_btnDownloadModels_clicked(self) -> None:
        """Ouvre le dialogue de téléchargement des poids des moteurs.

        Lance _DownloadWorker dans un thread séparé et affiche la progression
        dans _DownloadDialog.  Les fichiers déjà présents sont signalés et
        ignorés (skip_existing=True dans download_models.download_all).
        """
        dlg    = _DownloadDialog(self)
        worker = _DownloadWorker()
        worker.file_start.connect(dlg.on_file_start)
        worker.file_progress.connect(dlg.on_file_progress)
        worker.file_done.connect(dlg.on_file_done)
        worker.file_error.connect(dlg.on_file_error)
        worker.all_done.connect(dlg.on_all_done)
        worker.start()
        dlg.exec()
        if worker.isRunning():
            worker.terminate()
            worker.wait()

    def closeEvent(self, event) -> None:
        """Termine proprement les threads actifs avant de fermer la fenêtre.

        Sans cela, détruire la fenêtre pendant qu'un thread tourne produit
        "QThread: Destroyed while thread is still running" (le GC Python
        libère le worker avant que le thread C++ ait fini).
        """
        for worker in (self._worker, self._batch_worker, self._colab_worker):
            if worker is not None and worker.isRunning():
                worker.terminate()
                worker.wait()
        for window in list(self._open_windows):
            window.close()
        super().closeEvent(event)


def run_gui(model_path: Path | None = None) -> None:
    """Lancer l'application Qt."""
    import sys
    app    = QApplication(sys.argv)
    app.aboutToQuit.connect(app.closeAllWindows)
    window = PhotoRestorationGUI(model_path=model_path)
    window.show()
    sys.exit(app.exec())

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
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
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

from media_restorer.app_settings import TOOLTIP_MODE_KEY, app_settings
from media_restorer.extensions.media_restorer.colab_calc import ColabCalc
from media_restorer.engines import ENGINE_PARAMS, Engine, build_engine
from media_restorer.download_models import MODEL_REGISTRY
from media_restorer.gui_widgets import ResultWindow
from media_restorer.image_io import imread_oriented
from media_restorer.power import performance_mode
from OutilsQt.Utils_Qt import compile_ui, compile_qrc, tooltips_from_code

pg.setConfigOption("imageAxisOrder", "row-major")

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

# Onglet Double-exposition : paramètres qui déclenchent un recalcul de l'aperçu.
# « second_path » et « align » en sont exclus — ils invalident le recalage
# lui-même, qui n'est refait que par « Restaurer ».
_DUAL_LIVE_PARAMS = (
    "mode", "alpha", "match_levels",
    "detail_base", "detail_radius", "detail_gain",
    "w_contrast", "w_exposure",
)

# Paramètres n'ayant d'effet que dans certains modes → grisés ailleurs.
# Rempli à l'import depuis dual_engine pour éviter de dupliquer les libellés.
def _dual_param_modes() -> dict[str, str]:
    from media_restorer.engines.dual_engine import (
        MODE_DETAIL, MODE_FONDU, MODE_FUSION,
    )
    return {
        "alpha":         MODE_FONDU,
        "detail_base":   MODE_DETAIL,
        "detail_radius": MODE_DETAIL,
        "detail_gain":   MODE_DETAIL,
        "w_contrast":    MODE_FUSION,
        "w_exposure":    MODE_FUSION,
    }

# Côté max de l'aperçu interactif, en pixels (cf. mesures dans __init__).
_DUAL_PREVIEW_MAX = 1600

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

    Exclut tout fichier dont l'un des répertoires parents (entre *directory*
    et le fichier) porte le nom d'un moteur — quelle que soit la profondeur.
    Cela évite de retraiter les sorties d'une session précédente.
    """
    candidates = directory.rglob("*") if recursive else directory.iterdir()
    result = []
    for p in candidates:
        if not p.is_file() or p.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        rel = p.relative_to(directory)
        if _ENGINE_NAMES.intersection(rel.parts[:-1]):
            continue
        result.append(p)
    return sorted(result)

_UI_SRC  = Path(__file__).parent / "views" / "main.ui"
_UI_PY   = Path(__file__).parent / "views" / "ui_main.py"
# resources/ reste au niveau racine du paquet (partagé entre la fenêtre
# racine et toutes les extensions) — remonter de extensions/media_restorer/
# jusqu'à media_restorer/ nécessite 2 niveaux, pas 1.
_QRC_SRC = Path(__file__).parents[2] / "resources" / "icons" / "media_restorer.qrc"
_QRC_PY  = Path(__file__).parents[2] / "resources" / "icons" / "media_restorer_rc.py"

# Types de widgets inspectés par tooltips_from_code.
# Chaque entrée : (type_Qt, préfixe_du_nom, signal).
# Les onglets du QTabWidget sont gérés séparément par _set_tab_tooltips().
_TOOLTIP_TYPES: list[tuple] = [
    (QAction, "action", "triggered"),
]

compile_ui(_UI_SRC, _UI_PY)
compile_qrc(_QRC_SRC, _QRC_PY)

from media_restorer.resources.icons import media_restorer_rc as _rc                        # noqa: F401, E402
from media_restorer.extensions.media_restorer.views.ui_main import Ui_MainWindow            # noqa: E402


# ---------------------------------------------------------------------------
# Workers (threads d'arrière-plan)
# ---------------------------------------------------------------------------

class _RestoreWorker(QThread):
    result_ready = pyqtSignal(np.ndarray)
    error        = pyqtSignal(str)
    # Couple d'images recalé, si le moteur en expose un (DualExposureEngine).
    # Permet à la GUI de refaire un fondu sans relancer tout le traitement.
    pair_ready   = pyqtSignal(object)

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
            pair = getattr(eng, "aligned_pair", None)
            if pair is not None:
                self.pair_ready.emit(pair)
            self.result_ready.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class _BatchRestoreWorker(QThread):
    """Worker de traitement par lot d'un répertoire d'images.

    Signaux émis
    ------------
    image_done(filename, wall_seconds)
        Après chaque image réussie — durée réelle (horloge murale, GPU inclus).
    file_error(filename, message)
        En cas d'échec sur une image individuelle (le lot continue).
    all_done(n_success, n_total)
        À la fin du lot, quel que soit le nombre d'erreurs.

    Prévention du crash DRM (iGPU AMD)
    ------------------------------------
    Sur un iGPU partagé (ex. Radeon 780M), PyTorch et le compositor KDE se
    disputent le même GPU.  Un traitement en rafale sans pause sature la file
    de commandes GPU et provoque des timeouts DRM (``flip_done timedout``),
    pouvant crasher la session graphique.

    Après chaque image, ``torch.cuda.synchronize()`` vide la file GPU, puis
    une pause de 50 ms laisse au compositor le temps d'effectuer ses flips
    vsync (~3 cycles à 60 Hz) avant l'image suivante.
    """

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
    """Fenêtre « Media Restorer » — restauration interactive de photos anciennes.

    Choisit un moteur de restauration parmi les onglets du panneau de
    paramètres, règle ses paramètres, lance la restauration dans un thread de
    fond, puis affiche le résultat dans une fenêtre dédiée.

    Cette fenêtre ne choisit plus elle-même sa cible (fichier ou répertoire) :
    c'est le rôle de la fenêtre racine
    :class:`~media_restorer.gui_root.ImageTreatmentWindow` (« Image
    Treatment »), qui lance Media Restorer via
    :class:`~media_restorer.extensions.media_restorer_ext.MediaRestorerExtension`
    en lui passant *target_path*/*recursive* — voir ces paramètres.  Les
    préférences globales d'application (mode des tooltips, apparence) sont
    elles aussi réglées dans la fenêtre racine, et lues ici via
    :func:`~media_restorer.app_settings.app_settings`.

    Les tooltips des actions et des onglets sont générés automatiquement à
    partir des docstrings (ou du code source) via
    :func:`OutilsQt.Utils_Qt.tooltips_from_code`, dans le mode choisi dans la
    fenêtre racine et persisté entre les sessions.

    Paramètres
    ----------
    model_path : Path | None
        Chemin de poids à forcer pour le moteur actif ; ``None`` laisse
        chaque moteur chercher ses poids par défaut (voir
        :mod:`media_restorer.download_models`).
    target_path : Path | None
        Fichier image (mode restauration interactive, un seul cliché) ou
        répertoire (mode traitement par lot) à charger dès la construction.
        ``None`` construit la fenêtre sans rien charger — utilisé par les
        tests et par tout usage direct de cette classe hors de la fenêtre
        racine.
    recursive : bool
        Parcourt les sous-répertoires de *target_path* si c'est un
        répertoire.  Sans effet si *target_path* est un fichier ou ``None``.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        target_path: Path | None = None,
        recursive: bool = False,
    ) -> None:
        super().__init__()
        self._model_path      = model_path
        self._batch_dir:      Path | None = None
        self._batch_recursive = False
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
        self.actionRestore         = self._ui.actionRestore
        self.actionSave            = self._ui.actionSave
        self.actionBatch           = self._ui.actionBatch
        self.actionColabConnect    = self._ui.actionColabConnect
        self.actionColabRestore    = self._ui.actionColabRestore
        self.actionDownloadModels  = self._ui.actionDownloadModels

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
            self._result_windows[engine] = ResultWindow(
                engine.value, preserve_zoom=(engine is Engine.DUAL)
            )

        # ── Aperçu interactif (onglet Double-exposition) ──────────────────
        # Le recalage n'est fait qu'au « Restaurer » ; ensuite tout changement
        # de paramètre rejoue la seule fusion, sur une réduction du couple
        # mémorisé.  Mesuré sur un couple 36 Mpx : à pleine résolution le mode
        # « détail » demande 1,8 s et « fusion » 5,4 s — inutilisable au
        # slider ; réduit à _DUAL_PREVIEW_MAX px, aucun mode ne dépasse 0,15 s.
        # Un timer anti-rebond évite d'empiler un recalcul par cran de slider.
        self._dual_pair:    tuple[np.ndarray, np.ndarray] | None = None
        self._dual_preview: tuple[np.ndarray, np.ndarray] | None = None
        self._dual_timer = QTimer(self)
        self._dual_timer.setSingleShot(True)
        self._dual_timer.setInterval(150)
        self._dual_timer.timeout.connect(self._refresh_dual_preview)

        dual_root = self._param_roots[Engine.DUAL]
        for name in _DUAL_LIVE_PARAMS:
            dual_root.child(name).sigValueChanged.connect(
                lambda *_: self._dual_timer.start()
            )
        # Griser les paramètres sans effet dans le mode courant : sans cela le
        # slider « fondu » paraît actif alors qu'aucun mode sauf « fondu » ne
        # le consomme — c'est exactement ce qui prête à confusion.
        dual_root.child("mode").sigValueChanged.connect(
            lambda *_: self._sync_dual_param_states()
        )
        self._sync_dual_param_states()

        # Statut de la cible répertoire — en lecture seule : le choix du
        # répertoire et du mode récursif se fait désormais dans la fenêtre
        # racine (ImageTreatmentWindow), avant même la construction de cette
        # fenêtre.  N'affiche rien en mode fichier unique.
        self._batch_status_label = QLabel()
        bottom_bar = QWidget()
        hl = QHBoxLayout(bottom_bar)
        hl.setContentsMargins(6, 2, 6, 2)
        hl.addWidget(self._batch_status_label)
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

        # ── Tooltips — mode persisté, réglé dans la fenêtre racine ─────
        # L'apparence (skin) est déjà appliquée au niveau de la QApplication
        # par la fenêtre racine avant que Media Restorer ne soit construite
        # (ImageTreatmentWindow.__init__) : aucune action nécessaire ici, une
        # QPalette d'application s'applique automatiquement à toute fenêtre
        # créée par la suite.
        self._setup_tooltips(
            app_settings().value(TOOLTIP_MODE_KEY, "docstrings")
        )

        # ── Badge de calcul (droite de la statusBar) ──────────────────
        self._compute_badge = QLabel()
        self._compute_badge.setContentsMargins(0, 0, 4, 0)
        self.statusBar().addPermanentWidget(self._compute_badge)
        self._refresh_compute_badge()

        # ── Cible pré-chargée par la fenêtre racine ────────────────────
        self._apply_target(target_path, recursive)

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
        if self._current_engine is Engine.DUAL:
            QMessageBox.information(
                self, "Colab",
                "Le moteur Double-exposition combine deux images locales et ne "
                "consomme pas de GPU : il s'exécute toujours en local.\n\n"
                "Le protocole Colab n'envoie qu'une seule image par requête.",
            )
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
        from media_restorer.engines.dual_engine import DualExposureEngine
        _cls: dict[Engine, type] = {
            Engine.REAL_ESRGAN: RealESRGANEngine,
            Engine.SWINIR:      SwinIREngine,
            Engine.LAMA:        LaMaEngine,
            Engine.GFPGAN:      GFPGANEngine,
            Engine.DUAL:        DualExposureEngine,
        }
        for i, engine in enumerate(self._tab_engines):
            doc = inspect.cleandoc(_cls[engine].__doc__ or engine.value)
            self._tab_widget.setTabToolTip(i, doc)

    @pyqtSlot()
    def on_actionDownloadModels_triggered(self) -> None:
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
    # Cible pré-chargée par la fenêtre racine
    # ------------------------------------------------------------------

    def _apply_target(self, target_path: Path | None, recursive: bool) -> None:
        """Configure la fenêtre selon la cible transmise à la construction.

        Un fichier active le mode restauration interactive (``_load_path``,
        ``actionRestore``) ; un répertoire active le mode traitement par lot
        (``actionBatch``, sans dialogue puisque la cible est déjà connue).
        ``target_path=None`` laisse la fenêtre dans son état par défaut — un
        seul cas d'usage réel en dehors des tests : instancier cette classe
        directement sans passer par la fenêtre racine.
        """
        self._ui.actionBatch.setEnabled(False)
        if target_path is None:
            return
        if target_path.is_dir():
            self._batch_dir       = target_path
            self._batch_recursive = recursive
            self._ui.actionBatch.setEnabled(True)
            mode = "récursif" if recursive else "non récursif"
            self._batch_status_label.setText(f"Répertoire : {target_path} ({mode})")
            self.statusBar().showMessage(
                f"Répertoire prêt — cliquez « Traiter un répertoire » ({mode})."
            )
        else:
            self._load_path(target_path)

    def _load_path(self, path: Path) -> None:
        """Charge et affiche l'image de *path* dans la vue principale.

        Contrairement à l'ancienne ``_load_image``, ne déclenche aucun
        dialogue : *path* est déjà connu (fourni par la fenêtre racine à la
        construction).
        """
        img = imread_oriented(path)
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
        self._dual_pair = self._dual_preview = None
        self._worker = _RestoreWorker(
            self._original, self._pending_engine, self._model_path, params
        )
        self._worker.result_ready.connect(self._on_restore_done)
        self._worker.pair_ready.connect(self._on_pair_ready)
        self._worker.error.connect(self._on_restore_error)
        self._worker.start()

    def _on_pair_ready(self, pair: object) -> None:
        """Mémorise le couple recalé émis par le moteur Double-exposition.

        En conserve aussi une réduction, base de l'aperçu interactif : elle
        n'est calculée qu'ici (≈ 0,02 s), pas à chaque mouvement de slider.
        """
        img_a, img_b = pair  # type: ignore[misc]
        self._dual_pair = (img_a, img_b)
        scale = _DUAL_PREVIEW_MAX / max(img_a.shape[:2])
        if scale >= 1.0:
            self._dual_preview = (img_a, img_b)
        else:
            self._dual_preview = tuple(
                cv2.resize(i, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                for i in (img_a, img_b)
            )

    # Options que SliderParameterItem.optsChanged relit pour reconstruire son
    # échelle, et qu'il faut donc lui redonner à chaque changement d'option.
    _SLIDER_SCALE_OPTS = ("step", "limits", "precision")

    @classmethod
    def _set_param_enabled(cls, param: Parameter, enabled: bool) -> None:
        """Active/désactive *param* sans casser l'échelle d'un slider.

        ``Parameter.setOpts`` ne propage que les options **dont la valeur
        change**, et ``SliderParameterItem.optsChanged`` reconstruit le
        « span » du slider à chaque notification en relisant le pas dans le
        dictionnaire *partiel* reçu, avec un défaut de 1.  Un
        ``setOpts(enabled=…)`` réduisait donc le slider 0–1 de 101 crans à
        deux (``arange(0, 2, 1)``) : il sautait de 0 à 1, et la valeur
        courante était écrasée au passage.  Relayer ``step`` à ``setOpts`` ne
        suffit pas — inchangé, il est justement filtré avant d'être transmis.

        On pose donc l'option puis on émet soi-même la notification, en y
        réinjectant les options d'échelle pour que la reconstruction soit
        fidèle.
        """
        if param.opts.get("enabled", True) == enabled:
            return
        param.opts["enabled"] = enabled
        changed = {"enabled": enabled}
        changed.update(
            (k, param.opts[k]) for k in cls._SLIDER_SCALE_OPTS if k in param.opts
        )
        param.sigOptionsChanged.emit(param, changed)

    def _sync_dual_param_states(self) -> None:
        """Grise les paramètres sans effet dans le mode de fusion courant."""
        root = self._param_roots[Engine.DUAL]
        mode = root.child("mode").value()
        for name, required_mode in _dual_param_modes().items():
            self._set_param_enabled(root.child(name), mode == required_mode)

    def _refresh_dual_preview(self) -> None:
        """Recalcule l'aperçu après un changement de paramètre de fusion.

        Rejoue ``DualExposureEngine.fuse`` — donc exactement le calcul de
        production — sur la **réduction** du couple recalé : tous les modes
        répondent alors en moins de 0,15 s, là où « fusion (Mertens) »
        demanderait 5,4 s en pleine résolution.

        L'aperçu étant en résolution réduite, il ne remplace pas
        ``self._restored`` et « Enregistrer » est désactivé : c'est
        « Restaurer » qui produit le résultat pleine résolution enregistrable.
        Le titre de la fenêtre de résultat le signale.
        """
        if self._dual_preview is None:
            # Pas de couple recalé en mémoire — un mouvement de slider avant
            # tout « Restaurer » n'a rien à réutiliser.  Sans ce message,
            # bouger le slider ne fait strictement rien d'observable : c'est
            # le symptôme qu'on veut éviter à tout prix ici.
            self.statusBar().showMessage(
                "Aperçu Double-exposition : cliquez d'abord sur « Restaurer » "
                "pour calculer un premier résultat."
            )
            return
        params = self._read_params(Engine.DUAL)
        engine = build_engine(Engine.DUAL, self._model_path, params)
        try:
            result = engine.fuse(*self._dual_preview)  # type: ignore[attr-defined]
        except Exception as exc:
            self.statusBar().showMessage(f"Aperçu impossible : {exc}")
            return

        # L'aperçu est plus petit que le résultat pleine résolution : on
        # l'étire sur la même aire, sinon il se dessine dans un coin du
        # cadrage courant et paraît ne pas réagir.
        full_h, full_w = self._dual_pair[0].shape[:2]
        prev_h, prev_w = result.shape[:2]
        window = self._result_windows[Engine.DUAL]
        window.setWindowTitle(
            f"{Engine.DUAL.value} — aperçu {prev_w}×{prev_h}"
        )
        window.update_image(result, scale=(full_w / prev_w, full_h / prev_h))
        self._restored = None
        self._ui.actionSave.setEnabled(False)
        self.statusBar().showMessage(
            f"Aperçu « {params['mode']} » en résolution réduite "
            "— « Restaurer » pour la pleine résolution."
        )

    def _on_restore_done(self, result: np.ndarray) -> None:
        self._restored = result
        win = self._result_windows[self._pending_engine]
        # Efface un éventuel suffixe « aperçu » laissé par _refresh_dual_preview :
        # ce résultat-ci est bien en pleine résolution.
        win.setWindowTitle(self._pending_engine.value)
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
        if self._batch_dir is None:
            # Ne devrait pas arriver via l'UI (actionBatch n'est activée que
            # si _apply_target a reçu un répertoire) — reste un garde-fou
            # explicite plutôt qu'un plantage silencieux si jamais déclenché
            # autrement (test, appel direct).
            self.statusBar().showMessage(
                "Aucun répertoire à traiter — relancez depuis Image Treatment."
            )
            return
        if self._current_engine is Engine.DUAL:
            QMessageBox.information(
                self, "Traitement par lot",
                "Le moteur Double-exposition travaille sur un couple d'images "
                "choisi explicitement.\n\nEn lot, la même 2ᵉ image serait "
                "appliquée à tout le répertoire — utilisez « Restaurer » "
                "couple par couple.",
            )
            return
        dir_path  = self._batch_dir
        recursive = self._batch_recursive
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

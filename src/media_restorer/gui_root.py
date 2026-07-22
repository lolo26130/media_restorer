"""Fenêtre racine « Image Treatment » — point d'entrée de l'application.

Mise en page définie dans ``views/root.ui`` (compilé automatiquement, même
mécanisme que :mod:`media_restorer.gui` — voir :func:`OutilsQt.Utils_Qt.compile_ui`).
Choisit une cible (fichier ou répertoire), règle les préférences globales de
l'application (mode des tooltips, apparence), puis lance l'un des outils
enregistrés dans :mod:`media_restorer.extensions` — Media Restorer aujourd'hui,
d'autres outils demain sans modification de cette fenêtre (voir
:mod:`media_restorer.extensions` pour la marche à suivre).
"""
from __future__ import annotations

from functools import partial
from pathlib import Path

from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox
from PyQt6.QtCore import pyqtSlot

from media_restorer.app_settings import SKIN_KEY, TOOLTIP_MODE_KEY, app_settings
from media_restorer.extensions import Extension, ExtensionContext, all_extensions
from media_restorer.theme import THEME_SYSTEM, apply_theme
from OutilsQt.Utils_Qt import compile_ui, compile_qrc, tooltips_from_code

_UI_SRC  = Path(__file__).parent / "views" / "root.ui"
_UI_PY   = Path(__file__).parent / "views" / "ui_root.py"
_QRC_SRC = Path(__file__).parent / "resources" / "icons" / "media_restorer.qrc"
_QRC_PY  = Path(__file__).parent / "resources" / "icons" / "media_restorer_rc.py"

_TOOLTIP_TYPES: list[tuple] = [
    (QAction, "action", "triggered"),
]

compile_ui(_UI_SRC, _UI_PY)
compile_qrc(_QRC_SRC, _QRC_PY)

from media_restorer.resources.icons import media_restorer_rc as _rc  # noqa: E402, F401, I001
from media_restorer.views.ui_root import Ui_RootWindow                # noqa: E402


class ImageTreatmentWindow(QMainWindow):
    """Fenêtre racine — sélection de cible, préférences globales, lancement d'un outil.

    Point d'entrée unique de l'application (voir
    :func:`media_restorer.gui.run_gui`) : choisit un fichier ou un répertoire
    à traiter, règle les préférences globales (mode des tooltips, apparence),
    puis lance l'un des outils du registre :mod:`media_restorer.extensions` —
    Media Restorer aujourd'hui, d'autres outils demain sans modification de
    cette classe : le menu et la toolbar « Extensions » se peuplent
    dynamiquement depuis :func:`~media_restorer.extensions.all_extensions`.

    La cible choisie (fichier ou répertoire) et le mode récursif sont
    transmis à l'outil lancé via un
    :class:`~media_restorer.extensions.ExtensionContext` ; cette fenêtre ne
    connaît rien de plus sur ce que fait l'outil de la cible.
    """

    def __init__(self) -> None:
        super().__init__()
        self._target:      Path | None = None
        self._is_directory = False
        self._open_windows: list[QMainWindow] = []

        # ── UI de base (.ui compilé) ────────────────────────────────────
        self._ui = Ui_RootWindow()
        self._ui.setupUi(self)

        # Expose les QActions statiques sur self pour tooltips_from_code
        # (même convention que PhotoRestorationGUI — voir gui.py).
        self.actionChooseFile      = self._ui.actionChooseFile
        self.actionChooseDirectory = self._ui.actionChooseDirectory

        # ── Registre d'extensions — peuplement dynamique du menu/toolbar ──
        # Importer le module d'une extension l'enregistre (voir
        # media_restorer.extensions.media_restorer_ext) ; en ajouter une
        # nouvelle à l'avenir n'exige qu'une ligne d'import supplémentaire
        # ici, aucune autre modification de cette fenêtre.
        import media_restorer.extensions.media_restorer_ext  # noqa: F401

        self._launch_actions: list[QAction] = []
        for extension in all_extensions():
            action = QAction(QIcon(extension.icon), extension.name, self)
            action.setToolTip(extension.description)
            action.setEnabled(False)  # activée dès qu'une cible est choisie
            action.triggered.connect(partial(self._launch, extension))
            self._ui.menuExtensions.addAction(action)
            self._ui.toolBar.addAction(action)
            self._launch_actions.append(action)

        # ── Préférences persistées (voir media_restorer.app_settings) ───
        saved_skin = app_settings().value(SKIN_KEY, THEME_SYSTEM)
        apply_theme(QApplication.instance(), saved_skin)
        self._ui.comboSkin.setCurrentText(saved_skin)
        self._ui.comboSkin.currentTextChanged.connect(self._on_skin_changed)

        self._ui.comboTooltipMode.setCurrentText(
            app_settings().value(TOOLTIP_MODE_KEY, "docstrings")
        )
        self._ui.comboTooltipMode.currentTextChanged.connect(
            self._on_tooltip_mode_changed
        )
        self._setup_tooltips(self._ui.comboTooltipMode.currentText())

    # ------------------------------------------------------------------
    # Tooltips
    # ------------------------------------------------------------------

    def _setup_tooltips(self, mode: str) -> None:
        """Applique le mode de tooltips aux actions statiques de cette fenêtre.

        N'agit que sur ``actionChooseFile``/``actionChooseDirectory`` : les
        actions de lancement d'extension, créées dynamiquement, portent déjà
        leur propre tooltip (la description de l'extension) et ne passent pas
        par ce mécanisme fondé sur les docstrings de méthodes ``on_*``.
        """
        tooltips_from_code(self, mode=mode, liste_types_actions=_TOOLTIP_TYPES)

    def _on_tooltip_mode_changed(self, mode: str) -> None:
        self._setup_tooltips(mode)
        app_settings().setValue(TOOLTIP_MODE_KEY, mode)

    def _on_skin_changed(self, skin: str) -> None:
        apply_theme(QApplication.instance(), skin)
        app_settings().setValue(SKIN_KEY, skin)

    # ------------------------------------------------------------------
    # Choix de la cible
    # ------------------------------------------------------------------

    @pyqtSlot()
    def on_actionChooseFile_triggered(self) -> None:
        """Choisit un fichier image à traiter.

        Ouvre un dialogue de sélection de fichier.  Le chemin retenu devient
        la cible transmise à l'extension lancée ensuite, en mode fichier
        unique — le combo « Mode répertoire » est désactivé, sans objet ici.
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "Choisir un fichier", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if not path:
            return
        self._set_target(Path(path), is_directory=False)

    @pyqtSlot()
    def on_actionChooseDirectory_triggered(self) -> None:
        """Choisit un répertoire à traiter en lot.

        Ouvre un dialogue de sélection de répertoire.  Le mode récursif du
        combo « Mode répertoire » (activé dès qu'un répertoire est choisi)
        s'appliquera à cette cible lors du lancement de l'extension.
        """
        directory = QFileDialog.getExistingDirectory(self, "Choisir un répertoire")
        if not directory:
            return
        self._set_target(Path(directory), is_directory=True)

    def _set_target(self, path: Path, is_directory: bool) -> None:
        self._target      = path
        self._is_directory = is_directory
        self._ui.comboRecursive.setEnabled(is_directory)
        kind = "Répertoire" if is_directory else "Fichier"
        self._ui.labelTarget.setText(f"{kind} sélectionné : {path}")
        self.statusBar().showMessage(f"Cible prête : {path}")
        for action in self._launch_actions:
            action.setEnabled(True)

    # ------------------------------------------------------------------
    # Lancement d'une extension
    # ------------------------------------------------------------------

    def _launch(self, extension: Extension) -> None:
        """Lance *extension* sur la cible choisie, sans fermer la racine.

        La fenêtre ouverte est mémorisée dans ``self._open_windows`` pour
        éviter sa libération prématurée par le ramasse-miettes Python — même
        précaution que
        :meth:`~media_restorer.gui.PhotoRestorationGUI._register_window`
        pour les fenêtres de résultat de Media Restorer.
        """
        if self._target is None:
            QMessageBox.information(
                self, "Aucune cible",
                "Choisissez d'abord un fichier ou un répertoire à traiter.",
            )
            return
        recursive = self._is_directory and self._ui.comboRecursive.currentIndex() == 1
        context = ExtensionContext(path=self._target, recursive=recursive)
        window = extension.launch(context)
        self._open_windows.append(window)

    def closeEvent(self, event) -> None:
        """Ferme aussi les fenêtres d'extension ouvertes depuis la racine.

        Sans cela, fermer la fenêtre racine laisserait les outils lancés
        orphelins — potentiellement surprenant si l'utilisateur perçoit la
        racine comme le point d'entrée de toute la session de travail.
        """
        for window in list(self._open_windows):
            window.close()
        super().closeEvent(event)

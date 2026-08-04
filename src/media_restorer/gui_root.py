"""Fenêtre racine « Image Treatment » — point d'entrée de l'application.

Mise en page définie dans ``views/root.ui`` (compilé automatiquement, même
mécanisme que :mod:`media_restorer.extensions.media_restorer.gui` — voir
:func:`OutilsQt.Utils_Qt.compile_ui`).
Choisit une cible (fichier ou répertoire), règle les préférences globales de
l'application (mode des tooltips, apparence), puis lance l'un des outils
enregistrés dans :mod:`media_restorer.extensions` — Media Restorer aujourd'hui,
d'autres outils demain sans modification de cette fenêtre (voir
:mod:`media_restorer.extensions` pour la marche à suivre).
"""
from __future__ import annotations

import subprocess
from functools import partial
from pathlib import Path

from PyQt6.QtGui import QAction, QDesktopServices, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QMainWindow,
    QMessageBox,
)
from PyQt6.QtCore import Qt, QUrl, pyqtSlot

from media_restorer.app_settings import SKIN_KEY, TOOLTIP_MODE_KEY, app_settings
from media_restorer.gui_shots import ShotInventoryPanel
from media_restorer.gui_widgets import (
    ImagePreview,
    InfoExifPanel,
    apply_default_dock_width,
    restore_layout,
    save_layout,
)
from media_restorer.extensions import Extension, ExtensionContext, all_extensions
from media_restorer.theme import THEME_SYSTEM, apply_theme
from OutilsQt.Utils_Qt import compile_ui, compile_qrc, tooltips_from_code

_UI_SRC  = Path(__file__).parent / "views" / "root.ui"
_UI_PY   = Path(__file__).parent / "views" / "ui_root.py"
_QRC_SRC = Path(__file__).parent / "resources" / "icons" / "media_restorer.qrc"
_QRC_PY  = Path(__file__).parent / "resources" / "icons" / "media_restorer_rc.py"

# src/media_restorer/gui_root.py → racine du dépôt (même remontée que
# download_models._ROOT, à la même profondeur dans l'arborescence).
_REPO_ROOT = Path(__file__).parents[2]
_DOCS_SOURCE = _REPO_ROOT / "docs" / "source"
_DOCS_HTML   = _REPO_ROOT / "docs" / "build" / "html"

# Préfixe des clés QSettings de la disposition.  saveState() sérialise taille,
# position, empilement et onglet actif de TOUS les docks d'un coup : c'est le
# mécanisme prévu par Qt, préférable à retenir chaque dock à la main
# (voir media_restorer.gui_widgets.save_layout).
_LAYOUT_PREFIX = "root"

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
    :func:`~media_restorer.gui_root.run_gui`) : choisit un fichier ou un répertoire
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
        self.actionHelp            = self._ui.actionHelp

        # ── Registre d'extensions — peuplement dynamique du menu/toolbar ──
        # Importer le module d'une extension l'enregistre (voir
        # media_restorer.extensions.media_restorer) ; en ajouter une
        # nouvelle à l'avenir n'exige qu'une ligne d'import supplémentaire
        # ici, aucune autre modification de cette fenêtre.
        import media_restorer.extensions.media_restorer  # noqa: F401
        import media_restorer.extensions.vectorise  # noqa: F401
        import media_restorer.extensions.manual_mouse_points  # noqa: F401
        import media_restorer.extensions.auto_face_id_register  # noqa: F401
        import media_restorer.extensions.pre_classement  # noqa: F401
        import media_restorer.extensions.doublons  # noqa: F401

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

        # ── Dock « Infos, Exif » ────────────────────────────────────────
        # Métadonnées de la cible en temps réel (image → EXIF ; répertoire →
        # résumé).  Pendant un traitement par lot lancé par une extension, il
        # se met à jour sur l'image en cours si l'extension expose le signal
        # optionnel ``current_image_changed`` — voir _launch et le contrat
        # documenté dans media_restorer.extensions.
        self._info_panel = InfoExifPanel()
        self._info_dock = QDockWidget("Infos, Exif", self)
        # ⚠ Un dock SANS objectName est PUREMENT ET SIMPLEMENT IGNORÉ par
        # saveState/restoreState (Qt le signale sur la sortie d'erreur, puis
        # continue) : sans ces trois lignes, la mémorisation de la disposition
        # ne mémoriserait rien.  Le nom doit rester STABLE d'une version à
        # l'autre — c'est la clé qui relie un état enregistré à son dock.
        self._info_dock.setObjectName("dockInfoExif")
        self._info_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._info_dock.setWidget(self._info_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._info_dock)

        # ── Dock « Aperçu » ─────────────────────────────────────────────
        # L'image de la cible, chargée en résolution réduite (voir
        # media_restorer.gui_widgets.ImagePreview).  Suit exactement la même
        # règle que le dock « Infos, Exif » : cible fichier → cette image,
        # cible répertoire → rien à montrer, et pendant un traitement par lot
        # il suit l'image en cours via ``current_image_changed``.
        self._preview_panel = ImagePreview()
        self._preview_dock = QDockWidget("Aperçu", self)
        self._preview_dock.setObjectName("dockApercu")      # voir _info_dock
        self._preview_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._preview_dock.setWidget(self._preview_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._preview_dock)

        # ── Dock « Correspondances RAW » ────────────────────────────────
        # Inventaire des paires RAW/JPEG : quel JPEG vient de quel NEF, et
        # surtout quels RAW n'ont pas encore de dérivé.  Les RAW restent hors
        # de toutes les autres chaînes (voir engines/shots).
        # TABIFIÉ avec « Infos, Exif » pour ne pas encombrer : on bascule d'un
        # onglet à l'autre plutôt que d'empiler trois docks.
        # Le scan ne part JAMAIS tout seul — il coûte une minute sur 10 000
        # fichiers, et se déclenche par un bouton du panneau.
        self._shots_panel = ShotInventoryPanel()
        self._shots_dock = QDockWidget("Correspondances RAW", self)
        self._shots_dock.setObjectName("dockCorrespondancesRaw")   # voir _info_dock
        self._shots_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._shots_dock.setWidget(self._shots_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._shots_dock)
        self.tabifyDockWidget(self._info_dock, self._shots_dock)
        self._info_dock.raise_()          # « Infos, Exif » reste l'onglet visible
        self._shots_panel.current_image_changed.connect(self._on_extension_image)

        # restoreState en dernier : il remplace l'empilement et l'onglet
        # choisis juste au-dessus par ceux de la dernière session.
        # Rien de mémorisé ⇒ disposition par défaut, posée au premier
        # affichage (voir showEvent) et non ici.
        self._needs_default_layout = not restore_layout(self, _LAYOUT_PREFIX)

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
        self._refresh_info_dock()

    def _refresh_info_dock(self) -> None:
        """Réaffiche les docks « Infos, Exif » et « Aperçu » selon la cible courante.

        Image → ses métadonnées et son aperçu ; répertoire → le résumé du
        dossier (nombre d'images, taille) et aucun aperçu, puisqu'il n'y a pas
        une image à montrer mais des milliers.  C'est aussi l'état vers lequel
        les deux docks **reviennent** à la fin d'un traitement par lot (voir
        :meth:`_on_extension_image`).
        """
        self._shots_panel.set_target(self._target, is_directory=self._is_directory)
        if self._target is None:
            self._info_panel.clear()
            self._preview_panel.clear()
        elif self._is_directory:
            recursive = self._ui.comboRecursive.currentIndex() == 1
            self._info_panel.show_directory(self._target, recursive=recursive)
            self._preview_panel.clear()
        else:
            self._info_panel.show_for_path(self._target)
            self._preview_panel.show_path(self._target)

    # ------------------------------------------------------------------
    # Lancement d'une extension
    # ------------------------------------------------------------------

    def _launch(self, extension: Extension) -> None:
        """Lance *extension* sur la cible choisie, sans fermer la racine.

        La fenêtre ouverte est mémorisée dans ``self._open_windows`` pour
        éviter sa libération prématurée par le ramasse-miettes Python — même
        précaution que
        :meth:`~media_restorer.extensions.media_restorer.gui.PhotoRestorationGUI._register_window`
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

        # Contrat optionnel : une extension qui traite plusieurs images peut
        # exposer ``current_image_changed`` (voir media_restorer.extensions).
        # On s'y branche par duck-typing — les extensions à image unique ne
        # l'exposent pas.  La dernière fenêtre lancée pilote le dock (chaque
        # _launch rebranche) — comportement voulu.
        signal = getattr(window, "current_image_changed", None)
        if signal is not None:
            signal.connect(self._on_extension_image)

    def _on_extension_image(self, path: Path | None) -> None:
        """Met à jour les deux docks sur l'image en cours dans une extension.

        Reçu du signal optionnel ``current_image_changed`` : un ``Path`` affiche
        ses métadonnées **et** son aperçu ; ``None`` (fin de lot) les fait
        revenir à la cible courante.  Une extension qui expose ce signal — par
        exemple Pré-classement, sur la ligne sélectionnée dans son tableau —
        pilote donc les deux docks d'un coup, sans rien savoir de leur existence.
        """
        if path is None:
            self._refresh_info_dock()
        else:
            self._info_panel.show_for_path(path)
            self._preview_panel.show_path(path)

    # ------------------------------------------------------------------
    # Aide
    # ------------------------------------------------------------------

    @pyqtSlot()
    def on_actionHelp_triggered(self) -> None:
        """Ouvre la documentation Sphinx dans le navigateur par défaut.

        Reconstruit la documentation (``uv run python -m sphinx``) si elle
        n'a jamais été générée — même logique que ``open-docs.sh`` à la
        racine du dépôt, pour ne jamais ouvrir un lien mort.  Le cas courant
        (docs déjà construites) est instantané ; seule cette reconstruction
        rare bloque brièvement l'interface (jusqu'à 2 min, au-delà desquelles
        l'opération est abandonnée) — proportionné pour une action aussi
        occasionnelle.
        """
        self._open_help()

    def _open_help(self) -> None:
        index = _DOCS_HTML / "index.html"
        if not index.exists():
            self.statusBar().showMessage("Documentation absente — construction en cours…")
            QApplication.processEvents()
            try:
                # « uv run sphinx-build » échoue sur cette installation
                # (« Failed to spawn: sphinx-build » — même symptôme que
                # « uv run pytest », déjà rencontré dans ce projet) ;
                # « uv run python -m sphinx » fonctionne de manière fiable.
                subprocess.run(
                    ["uv", "run", "python", "-m", "sphinx", "-b", "html",
                     str(_DOCS_SOURCE), str(_DOCS_HTML), "-q"],
                    check=True, cwd=_REPO_ROOT, timeout=120,
                )
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
                QMessageBox.critical(
                    self, "Documentation",
                    f"Impossible de construire la documentation :\n{exc}",
                )
                self.statusBar().showMessage("Échec de la construction de la documentation.")
                return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(index)))
        self.statusBar().showMessage(f"Documentation ouverte : {index}")

    def showEvent(self, event) -> None:      # noqa: N802 — API Qt
        """Pose la disposition par défaut au tout premier affichage.

        Et non dans ``__init__`` : avant ``show()``, ``QMainWindow`` n'a pas
        encore arbitré ses zones, et la largeur demandée pour l'aperçu y était
        rabotée (241 px au lieu de 460, mesuré).  Le drapeau garantit qu'un
        simple masquer/réafficher ne réécrase pas un réglage de l'utilisateur.
        """
        super().showEvent(event)
        if self._needs_default_layout:
            self._needs_default_layout = False
            apply_default_dock_width(self, self._preview_dock)

    def closeEvent(self, event) -> None:
        """Mémorise la disposition, puis ferme les fenêtres d'extension ouvertes.

        Sans la fermeture en cascade, quitter la racine laisserait les outils
        lancés orphelins — surprenant si l'utilisateur perçoit la racine comme
        le point d'entrée de toute la session.

        La disposition est enregistrée **ici et nulle part ailleurs** : c'est le
        seul endroit par lequel passe toute fermeture de la fenêtre.
        """
        save_layout(self, _LAYOUT_PREFIX)
        for window in list(self._open_windows):
            window.close()
        super().closeEvent(event)


def run_gui() -> None:
    """Lancer l'application Qt — ouvre la fenêtre racine « Image Treatment ».

    Point d'entrée de l'application : ``cli.py`` n'importe que cette
    fonction et n'a donc jamais besoin de connaître les extensions
    (:mod:`media_restorer.extensions`) — c'est :class:`ImageTreatmentWindow`
    qui choisit une cible (fichier ou répertoire) et lance Media Restorer, ou
    tout autre outil enregistré, sur cette cible.
    """
    import sys

    app    = QApplication(sys.argv)
    app.aboutToQuit.connect(app.closeAllWindows)
    window = ImageTreatmentWindow()
    window.show()
    sys.exit(app.exec())

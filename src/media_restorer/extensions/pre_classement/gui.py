"""Fenêtre Qt de l'extension Pré-classement — tri grossier d'un corpus.

Orchestre le cœur sans Qt :func:`~media_restorer.engines.triage.scan_directory`
(mesure) et :mod:`media_restorer.engines.triage.tags` (écriture des étiquettes
DigiKam).  Deux étapes volontairement séparées : **classer est réversible,
écrire modifie les fichiers** — d'où deux actions, deux workers, et une
confirmation unique pour tout le lot avant la seconde.

Le panneau de paramètres est construit **depuis**
:data:`~media_restorer.engines.triage.criteria.CRITERIA` : ajouter un critère au
catalogue le fait apparaître ici sans qu'une ligne de ce module change.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from pyqtgraph.parametertree import Parameter, ParameterTree
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from media_restorer import tabular
from media_restorer.app_settings import TOOLTIP_MODE_KEY, app_settings
from media_restorer.digikam_tags import ExiftoolRunner
from media_restorer.engines.triage import (
    CRITERIA,
    METHODS,
    METHOD_SIGNALS,
    Criterion,
    ImageSignals,
    ScanResult,
    format_summary,
    scan_directory,
    select,
    summarise,
)
from media_restorer.engines.triage import tags as triage_tags
from media_restorer.gui_widgets import (
    ImagePreview,
    apply_default_dock_width,
    restore_layout,
    save_layout,
)
from OutilsQt.Utils_Qt import compile_qrc, compile_ui, tooltips_from_code

_UI_SRC = Path(__file__).parent / "views" / "main.ui"
_UI_PY = Path(__file__).parent / "views" / "ui_main.py"
_QRC_SRC = Path(__file__).parents[2] / "resources" / "icons" / "media_restorer.qrc"
_QRC_PY = Path(__file__).parents[2] / "resources" / "icons" / "media_restorer_rc.py"

_TOOLTIP_TYPES: list[tuple] = [(QAction, "action", "triggered")]

#: Préfixe QSettings de la disposition — l'aperçu retrouve la taille qu'on
#: lui a donnée, au lieu de renaître minuscule à chaque ouverture.
_LAYOUT_PREFIX = "pre_classement"

# Clés QSettings — préfixées pour ne pas se mêler à celles des autres outils.
_SETTINGS_PREFIX = "pre_classement/"
_KEY_MAX_MPX = _SETTINGS_PREFIX + "max_megapixels"
_KEY_CRITERIA = _SETTINGS_PREFIX + "criteria"

#: Plafond par défaut : au-delà, une image coûte cher à décoder pour un intérêt
#: nul dans un tri grossier.  Réglable, et lu depuis QSettings au lancement.
DEFAULT_MAX_MEGAPIXELS = 50

#: Délai avant de charger l'aperçu, en millisecondes.  Assez court pour
#: paraître instantané, assez long pour qu'un défilement maintenu ne charge
#: que l'image sur laquelle on s'arrête.
PREVIEW_DEBOUNCE_MS = 120

compile_ui(_UI_SRC, _UI_PY)
compile_qrc(_QRC_SRC, _QRC_PY)

from media_restorer.resources.icons import media_restorer_rc as _rc  # noqa: F401, E402
from media_restorer.extensions.pre_classement.views.ui_main import Ui_MainWindow  # noqa: E402

#: Signature de la fonction de mesure d'un corpus (injectable pour les tests).
ScanFn = Callable[..., ScanResult]


class _ClassifyWorker(QThread):
    """Mesure un corpus dans un fil dédié.

    Signal de résultat nommé ``result_ready`` et **jamais** ``finished`` : ce
    dernier shadowe :attr:`QThread.finished` et provoque des connexions
    silencieusement rompues (convention du ``CLAUDE.md``).
    """

    result_ready = pyqtSignal(object)      # ScanResult
    progress = pyqtSignal(int, int)        # (courante, total)
    image_changed = pyqtSignal(str)        # chemin de l'image en cours
    error = pyqtSignal(str)

    def __init__(
        self,
        root: Path,
        *,
        recursive: bool,
        max_megapixels: float | None,
        scan_fn: ScanFn | None = None,
    ) -> None:
        super().__init__()
        self._root = root
        self._recursive = recursive
        self._max_megapixels = max_megapixels
        # Injection de dépendance : les tests substituent un scan instantané et
        # vérifient le câblage Qt sans jamais décoder d'image.
        self._scan_fn = scan_fn or scan_directory

    def run(self) -> None:
        try:
            result = self._scan_fn(
                self._root,
                recursive=self._recursive,
                max_megapixels=self._max_megapixels,
                on_progress=self._report,
            )
        except Exception as exc:                       # corpus illisible, droits…
            self.error.emit(str(exc))
            return
        self.result_ready.emit(result)

    def _report(self, index: int, total: int) -> None:
        self.progress.emit(index, total)


class _WriteWorker(QThread):
    """Écrit les étiquettes de tout un lot, un appel ``exiftool`` par image.

    Dans un fil séparé : sur plusieurs milliers d'images, écrire depuis le fil
    d'interface figerait la fenêtre pendant des minutes.
    """

    result_ready = pyqtSignal(int, list)   # (écrites, [(chemin, message)] en échec)
    progress = pyqtSignal(int, int)
    image_changed = pyqtSignal(str)

    def __init__(
        self,
        signals: Sequence[ImageSignals],
        criteria: Sequence[Criterion],
        *,
        runner: ExiftoolRunner | None = None,
    ) -> None:
        super().__init__()
        self._signals = list(signals)
        self._criteria = list(criteria)
        self._runner = runner

    def run(self) -> None:
        total = len(self._signals)
        echecs: list[tuple[str, str]] = []
        ecrites = 0
        for index, signals in enumerate(self._signals, start=1):
            self.image_changed.emit(str(signals.path))
            try:
                triage_tags.write_signals(
                    signals.path, signals, self._criteria, runner=self._runner
                )
                ecrites += 1
            except Exception as exc:
                # Une image verrouillée ne doit pas interrompre les 8 000 autres.
                echecs.append((str(signals.path), str(exc)))
            self.progress.emit(index, total)
        self.result_ready.emit(ecrites, echecs)


class PreClassementGUI(QMainWindow):
    """Fenêtre « Pré-classement » — classe un corpus, puis étiquette les fichiers.

    Paramètres
    ----------
    target_path : Path | None
        Répertoire (ou fichier) choisi dans la fenêtre racine.
    recursive : bool
        Mode de parcours transmis par :class:`~media_restorer.extensions.ExtensionContext`.
    scan_fn : ScanFn | None
        Substitut de la fonction de mesure, pour les tests.
    exiftool_runner : ExiftoolRunner | None
        Substitut du lanceur ``exiftool``, pour les tests.

    Signaux
    -------
    current_image_changed : pyqtSignal(object)
        ``Path`` de l'image en cours pendant un lot, ``None`` à la fin.  Contrat
        **optionnel** documenté dans :mod:`media_restorer.extensions` : la
        fenêtre racine s'y branche pour tenir à jour son dock « Infos, Exif ».
    """

    current_image_changed = pyqtSignal(object)          # Path | None

    _EMPTY_SUMMARY = "Aucun classement effectué."

    def __init__(
        self,
        target_path: Path | None = None,
        recursive: bool = False,
        scan_fn: ScanFn | None = None,
        exiftool_runner: ExiftoolRunner | None = None,
    ) -> None:
        super().__init__()
        self._scan_fn = scan_fn
        self._exiftool_runner = exiftool_runner
        self._recursive = recursive
        self._target: Path | None = None
        self._result: ScanResult | None = None
        self._criteria: tuple[Criterion, ...] = ()
        # Index par chemin : le tableau est triable, l'ordre des lignes ne
        # correspond donc plus à celui du résultat.
        self._signals_by_path: dict[str, ImageSignals] = {}
        self._worker: QThread | None = None

        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)

        # Exposées sur self pour tooltips_from_code (convention commune).
        self.actionClassify = self._ui.actionClassify
        self.actionWriteTags = self._ui.actionWriteTags
        self.actionExportCsv = self._ui.actionExportCsv

        self._build_parameters()
        self._build_preview_dock()
        self._build_summary_dock()
        # Rien de mémorisé ⇒ disposition par défaut, posée au premier
        # affichage (voir showEvent) et non ici.
        self._needs_default_layout = not restore_layout(self, _LAYOUT_PREFIX)

        tooltips_from_code(
            self, mode=app_settings().value(TOOLTIP_MODE_KEY, "docstrings"),
            liste_types_actions=_TOOLTIP_TYPES,
        )
        self._apply_target(target_path)

    # ------------------------------------------------------------------
    # Construction de l'interface
    # ------------------------------------------------------------------

    def _build_parameters(self) -> None:
        """Panneau de paramètres, **engendré depuis le catalogue de critères**.

        Aucun widget câblé à la main : ajouter un ``Criterion`` au catalogue
        suffit à le faire apparaître, cochable et persisté.
        """
        settings = app_settings()
        actifs = settings.value(_KEY_CRITERIA)
        actifs = set(actifs) if actifs else {c.key for c in CRITERIA}

        self._param_root = Parameter.create(
            name="params", type="group",
            children=[
                {
                    "name": "method", "title": "Méthode",
                    "type": "list", "value": METHOD_SIGNALS,
                    "limits": {label: key for key, label in METHODS.items()},
                },
                {
                    "name": "max_mpx", "title": "Plafond de résolution (Mpx)",
                    "type": "int",
                    "value": int(settings.value(_KEY_MAX_MPX, DEFAULT_MAX_MEGAPIXELS)),
                    "limits": (1, 1000), "step": 5,
                },
                {
                    "name": "recursive", "title": "Parcours récursif",
                    "type": "bool", "value": self._recursive,
                },
                {
                    "name": "criteria", "title": "Critères", "type": "group",
                    "children": [
                        {"name": c.key, "title": c.title, "type": "bool",
                         "value": c.key in actifs}
                        for c in CRITERIA
                    ],
                },
            ],
        )
        self._param_root.child("max_mpx").sigValueChanged.connect(self._save_settings)
        for c in CRITERIA:
            self._param_root.child("criteria", c.key).sigValueChanged.connect(
                self._save_settings
            )

        tree = ParameterTree(showHeader=False)
        tree.setParameters(self._param_root)
        self._param_tree = tree

    def _build_preview_dock(self) -> None:
        """Dock « Aperçu » : l'image de la ligne sélectionnée, en résolution réduite.

        Branché sur ``itemSelectionChanged`` et non sur un clic : ce signal est
        aussi émis quand la sélection change **au clavier** (flèches, Page haut/
        bas), ce qui est le mode de parcours naturel d'un tableau de plusieurs
        milliers de lignes.

        Le chargement est **différé** par un minuteur : maintenir une flèche
        enfoncée traverse des dizaines de lignes par seconde, et charger chacune
        d'elles saturerait le fil d'interface pour n'afficher que la dernière.
        """
        self._preview = ImagePreview()
        dock = QDockWidget("Aperçu", self)
        dock.setObjectName("dockApercu")  # sans objectName, restoreState ignore le dock
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        dock.setWidget(self._preview)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        self._preview_dock = dock

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(PREVIEW_DEBOUNCE_MS)
        self._preview_timer.timeout.connect(self._show_selected_preview)
        self._ui.tableResults.itemSelectionChanged.connect(self._preview_timer.start)

    def _selected_signals(self) -> ImageSignals | None:
        """Signaux de la ligne sélectionnée, ou ``None``.

        Le tableau étant triable, la position d'une ligne ne correspond plus à
        son rang dans le résultat : le chemin est donc stocké sur la cellule
        (``Qt.ItemDataRole.UserRole``) plutôt que déduit de l'index.
        """
        if self._result is None:
            return None
        items = self._ui.tableResults.selectedItems()
        if not items:
            return None
        chemin = self._ui.tableResults.item(items[0].row(), 0).data(
            Qt.ItemDataRole.UserRole
        )
        return self._signals_by_path.get(chemin)

    @pyqtSlot()
    def _show_selected_preview(self) -> None:
        """Affiche l'image sélectionnée et en informe le dock « Infos, Exif »."""
        signals = self._selected_signals()
        if signals is None:
            self._preview.clear()
            return
        self._preview.show_path(signals.path)
        # La racine suit la sélection : métadonnées et aperçu restent cohérents.
        self.current_image_changed.emit(signals.path)

    def _build_summary_dock(self) -> None:
        """Dock « Paramètres et synthèse » : réglages en haut, distributions dessous."""
        self._summary_label = QLabel(self._EMPTY_SUMMARY)
        self._summary_label.setWordWrap(False)
        self._summary_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        # Chasse fixe : les barres de l'histogramme de format_summary ne
        # s'alignent qu'avec une police à largeur constante.
        self._summary_label.setStyleSheet("font-family: monospace;")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._param_tree)
        layout.addWidget(QLabel("<b>Synthèse :</b>"))
        layout.addWidget(self._summary_label)
        layout.addStretch(1)

        dock = QDockWidget("Paramètres et synthèse", self)
        dock.setObjectName("dockParametres")  # sans objectName, restoreState ignore le dock
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        dock.setWidget(container)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _save_settings(self) -> None:
        """Persiste plafond et critères cochés (relus au prochain lancement)."""
        settings = app_settings()
        settings.setValue(_KEY_MAX_MPX, self._param_root["max_mpx"])
        settings.setValue(_KEY_CRITERIA, [c.key for c in self._selected_criteria()])

    def _selected_criteria(self) -> tuple[Criterion, ...]:
        """Critères cochés, dans l'ordre du catalogue (voir ``criteria.select``)."""
        return select([c.key for c in CRITERIA
                       if self._param_root["criteria", c.key]])

    def _apply_target(self, path: Path | None) -> None:
        """Charge la cible reçue de la fenêtre racine."""
        self._target = path
        if path is None:
            self.statusBar().showMessage("Aucune cible.")
            return
        if not path.is_dir():
            # Outil de lot : une image seule n'a rien à classer statistiquement.
            self.statusBar().showMessage(
                f"« {path.name} » est un fichier — choisissez un répertoire."
            )
            return
        self.actionClassify.setEnabled(True)
        self.statusBar().showMessage(f"Cible : {path}")

    # ------------------------------------------------------------------
    # Classement
    # ------------------------------------------------------------------

    @pyqtSlot()
    def on_actionClassify_triggered(self) -> None:
        """Mesure toutes les images de la cible et les classe selon les critères cochés.

        Ne modifie aucun fichier : le classement est entièrement réversible,
        seule l'action « Écrire les étiquettes » touche aux images.
        """
        self._start_classify()

    def _start_classify(self) -> None:
        if self._target is None or self._worker is not None:
            return
        self._criteria = self._selected_criteria()
        if not self._criteria:
            QMessageBox.information(
                self, "Aucun critère",
                "Cochez au moins un critère dans le panneau de paramètres.",
            )
            return

        self._set_busy(True, "Classement en cours…")
        worker = _ClassifyWorker(
            self._target,
            recursive=bool(self._param_root["recursive"]),
            max_megapixels=float(self._param_root["max_mpx"]),
            scan_fn=self._scan_fn,
        )
        worker.progress.connect(self._on_progress)
        worker.result_ready.connect(self._on_classified)
        worker.error.connect(self._on_error)
        self._worker = worker
        worker.start()

    @pyqtSlot(object)
    def _on_classified(self, result: ScanResult) -> None:
        self._worker = None
        self._result = result
        self._fill_table(result)
        self._summary_label.setText(format_summary(summarise(result.signals)))
        self.current_image_changed.emit(None)          # le dock revient au résumé

        has_rows = bool(result.signals)
        self.actionWriteTags.setEnabled(has_rows)
        self.actionExportCsv.setEnabled(has_rows)
        self._set_busy(False, self._describe(result))

    def _describe(self, result: ScanResult) -> str:
        """Résume le parcours en distinguant le filtre de l'incident."""
        parts = [f"{len(result.signals)} image(s) classée(s)"]
        if result.skipped_large:
            parts.append(f"{len(result.skipped_large)} écartée(s) par le plafond")
        if result.unreadable:
            parts.append(f"{len(result.unreadable)} illisible(s)")
        return " — ".join(parts)

    def _fill_table(self, result: ScanResult) -> None:
        """Une ligne par image, une colonne par critère retenu."""
        table = self._ui.tableResults
        table.setSortingEnabled(False)                 # tri pendant le remplissage = lignes mélangées
        table.clear()
        table.setColumnCount(1 + len(self._criteria))
        table.setHorizontalHeaderLabels(
            ["Fichier", *(c.title for c in self._criteria)]
        )
        table.setRowCount(len(result.signals))
        self._signals_by_path = {str(s.path): s for s in result.signals}
        for row, signals in enumerate(result.signals):
            cellule = QTableWidgetItem(signals.path.name)
            # Le chemin complet voyage avec la cellule : le nom affiché ne suffit
            # pas (deux sous-dossiers peuvent contenir le même nom de fichier).
            cellule.setData(Qt.ItemDataRole.UserRole, str(signals.path))
            table.setItem(row, 0, cellule)
            for col, criterion in enumerate(self._criteria, start=1):
                table.setItem(row, col, QTableWidgetItem(criterion.extract(signals)))
        table.resizeColumnsToContents()
        table.setSortingEnabled(True)
        self._preview.clear()          # nouvelle campagne : l'aperçu est obsolète

    # ------------------------------------------------------------------
    # Écriture des étiquettes
    # ------------------------------------------------------------------

    @pyqtSlot()
    def on_actionWriteTags_triggered(self) -> None:
        """Écrit le classement dans les métadonnées de chaque image, au format DigiKam.

        Modifie les fichiers (``exiftool`` conserve une copie ``_original``) :
        une confirmation unique est demandée pour tout le lot.  Les étiquettes
        étrangères — celles de l'utilisateur comme les repères — sont préservées.
        """
        self._write_tags()

    def _write_tags(self) -> None:
        if self._result is None or self._worker is not None:
            return
        signals = self._result.signals
        reply = QMessageBox.question(
            self, "Écrire les étiquettes ?",
            f"Écrire le classement de {len(signals)} image(s) dans leurs "
            f"métadonnées ?\n\n"
            f"Critères : {', '.join(c.title for c in self._criteria)}\n\n"
            "Les fichiers seront modifiés (une copie « _original » est conservée "
            "par exiftool). Les étiquettes déjà présentes sont préservées.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            self.statusBar().showMessage("Écriture annulée.")
            return

        self._set_busy(True, "Écriture des étiquettes…")
        worker = _WriteWorker(signals, self._criteria, runner=self._exiftool_runner)
        worker.progress.connect(self._on_progress)
        worker.image_changed.connect(self._on_image_changed)
        worker.result_ready.connect(self._on_written)
        self._worker = worker
        worker.start()

    @pyqtSlot(int, list)
    def _on_written(self, written: int, failures: list) -> None:
        self._worker = None
        self.current_image_changed.emit(None)
        self._set_busy(False, f"{written} image(s) étiquetée(s).")
        if failures:
            apercu = "\n".join(f"• {Path(p).name} : {msg}" for p, msg in failures[:10])
            reste = f"\n… et {len(failures) - 10} autre(s)." if len(failures) > 10 else ""
            QMessageBox.warning(
                self, "Écritures en échec",
                f"{len(failures)} image(s) n'ont pas pu être étiquetées :\n\n{apercu}{reste}",
            )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    @pyqtSlot()
    def on_actionExportCsv_triggered(self) -> None:
        """Enregistre le tableau de classement dans un fichier CSV, sans modifier les images.

        Utile pour vérifier un classement dans un tableur avant de l'écrire dans
        les métadonnées, ou pour le conserver hors des fichiers.
        """
        self._export_csv()

    def _export_csv(self) -> None:
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Exporter le classement", "classement.csv", tabular.FILE_FILTER
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = tabular.writer(f)
                writer.writerow(["fichier", *(c.title for c in self._criteria)])
                for signals in self._result.signals:
                    writer.writerow([
                        str(signals.path),
                        *(c.extract(signals) for c in self._criteria),
                    ])
        except OSError as exc:
            QMessageBox.critical(self, "Erreur — export CSV", str(exc))
            return
        self.statusBar().showMessage(f"Classement exporté : {path}")

    # ------------------------------------------------------------------
    # État et progression
    # ------------------------------------------------------------------

    def _set_busy(self, busy: bool, message: str) -> None:
        self._ui.progressBar.setVisible(busy)
        for action in (self.actionClassify, self.actionWriteTags, self.actionExportCsv):
            action.setEnabled(not busy and action.isEnabled())
        if busy:
            self.actionClassify.setEnabled(False)
        else:
            self.actionClassify.setEnabled(self._target is not None)
        self.statusBar().showMessage(message)

    @pyqtSlot(int, int)
    def _on_progress(self, index: int, total: int) -> None:
        self._ui.progressBar.setMaximum(max(total, 1))
        self._ui.progressBar.setValue(index)

    @pyqtSlot(str)
    def _on_image_changed(self, path: str) -> None:
        """Relaie l'image en cours vers le dock « Infos, Exif » de la racine."""
        self.current_image_changed.emit(Path(path))

    @pyqtSlot(str)
    def _on_error(self, message: str) -> None:
        self._worker = None
        self._set_busy(False, "Échec du classement.")
        QMessageBox.critical(self, "Erreur — classement", message)

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

    def closeEvent(self, event) -> None:      # noqa: N802 — API Qt
        """Mémorise la disposition des docks avant de fermer.

        Seul chemin par lequel passe toute fermeture de la fenêtre : y placer
        l'enregistrement garantit qu'aucune sortie ne l'oublie.
        """
        save_layout(self, _LAYOUT_PREFIX)
        super().closeEvent(event)

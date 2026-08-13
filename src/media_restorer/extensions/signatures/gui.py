"""Fenêtre Qt de l'extension Signatures — classement des dessins par dessinateur.

Orchestre le cœur sans Qt :func:`~media_restorer.engines.signatures.pipeline.scan_corpus`
(scan **silencieux**, lecture seule) puis, séparément, la revue des cas
difficiles et l'écriture des métadonnées.  Trois temps, jamais mélangés :

1. **Scanner** — ``_ScanWorker`` (fil séparé) localise, compare, classe
   chaque dessin.  Ne modifie aucun fichier.
2. **Revue** — sur le fil PRINCIPAL, un dessin à la fois, via des
   ``QDialog.exec()`` modaux.  Le ``_ScanWorker`` a déjà terminé et s'est
   arrêté avant que ceci ne commence : aucune analogie avec le piège
   documenté dans ``CLAUDE.md`` (thread vivant + boucle d'événements
   imbriquée).  Chaque décision de revue écrit IMMÉDIATEMENT (un seul
   fichier, coût négligeable), puis recompare la file restante à la
   bibliothèque mise à jour — l'effort manuel se réduit ainsi à peu près à
   « une fois par dessinateur distinct », pas « une fois par dessin ».
3. **Écrire les étiquettes** — ``_WriteWorker`` (fil séparé), pour le lot de
   verdicts confiants du scan initial, confirmation unique.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable

from pyqtgraph.parametertree import Parameter, ParameterTree
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QDialog,
    QDockWidget,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QTableWidgetItem,
)

from media_restorer import tabular
from media_restorer.app_settings import TOOLTIP_MODE_KEY, app_settings
from media_restorer.digikam_tags import ExiftoolRunner
from media_restorer.engines.duplicates.device import DEVICE_ORDER, DEVICE_TITLES
from media_restorer.engines.duplicates.device import resolve as resolve_device
from media_restorer.engines.duplicates.embeddings import (
    EMBEDDING_MODELS,
    EMBEDDING_MODELS_BY_KEY,
    build_embedder,
)
from media_restorer.engines.face_id.detect import CANDIDATE_MODELS, build_detector
from media_restorer.engines.signatures import descriptors, library, matching, tags
from media_restorer.engines.signatures.pipeline import (
    REASON_AMBIGUOUS,
    REASON_NO_LOCATION,
    REASON_NO_MATCH,
    ScanFn,
    ScanOutcome,
    scan_corpus,
)
from media_restorer.extensions.signatures import config as _config
from media_restorer.extensions.signatures.review_dialog import SignatureReviewDialog
from media_restorer.gui_widgets import (
    ImagePreview,
    apply_default_dock_width,
    restore_layout,
    save_layout,
)
from media_restorer.image_io import imread_oriented
from OutilsQt.Utils_Qt import compile_qrc, compile_ui, tooltips_from_code

_UI_SRC = Path(__file__).parent / "views" / "main.ui"
_UI_PY = Path(__file__).parent / "views" / "ui_main.py"
_QRC_SRC = Path(__file__).parents[2] / "resources" / "icons" / "media_restorer.qrc"
_QRC_PY = Path(__file__).parents[2] / "resources" / "icons" / "media_restorer_rc.py"

_TOOLTIP_TYPES: list[tuple] = [(QAction, "action", "triggered")]

#: Préfixe QSettings de la disposition mémorisée.
_LAYOUT_PREFIX = "signatures"

_SETTINGS_PREFIX = "signatures/"
_KEY_CONFIDENT_THRESHOLD = _SETTINGS_PREFIX + "confident_threshold"
_KEY_AMBIGUOUS_MARGIN = _SETTINGS_PREFIX + "ambiguous_margin"
_KEY_MIN_SCORE = _SETTINGS_PREFIX + "min_score"
_KEY_RECURSIVE = _SETTINGS_PREFIX + "recursive"

PREVIEW_DEBOUNCE_MS = 120

compile_ui(_UI_SRC, _UI_PY)
compile_qrc(_QRC_SRC, _QRC_PY)

from media_restorer.resources.icons import media_restorer_rc as _rc  # noqa: F401, E402
from media_restorer.extensions.signatures.views.ui_main import Ui_MainWindow  # noqa: E402


class _ScanWorker(QThread):
    """Scan silencieux d'un corpus dans un fil dédié — jamais ``finished``
    comme nom de signal (shadowe ``QThread.finished``, convention du projet).
    """

    result_ready = pyqtSignal(object)      # list[ScanOutcome]
    progress = pyqtSignal(int, int)
    error = pyqtSignal(str)

    def __init__(
        self,
        root: Path,
        *,
        recursive: bool,
        thresholds: dict,
        force: bool,
        scan_fn: ScanFn,
    ) -> None:
        super().__init__()
        self._root = root
        self._recursive = recursive
        self._thresholds = thresholds
        self._force = force
        self._scan_fn = scan_fn

    def run(self) -> None:
        try:
            outcomes = self._scan_fn(
                self._root,
                recursive=self._recursive,
                thresholds=self._thresholds,
                force=self._force,
                on_progress=self._report,
            )
        except Exception as exc:
            self.error.emit(str(exc))
            return
        self.result_ready.emit(outcomes)

    def _report(self, index: int, total: int) -> None:
        self.progress.emit(index, total)


class _WriteWorker(QThread):
    """Écrit les étiquettes des verdicts confiants du scan initial."""

    result_ready = pyqtSignal(int, list)   # (écrites, [(chemin, message)] en échec)
    progress = pyqtSignal(int, int)

    def __init__(self, outcomes: list[ScanOutcome], *, runner: ExiftoolRunner | None = None) -> None:
        super().__init__()
        self._outcomes = outcomes
        self._runner = runner

    def run(self) -> None:
        total = len(self._outcomes)
        echecs: list[tuple[str, str]] = []
        ecrites = 0
        for index, outcome in enumerate(self._outcomes, start=1):
            try:
                tags.write_artist(outcome.path, outcome.artist, runner=self._runner)
                ecrites += 1
            except Exception as exc:
                echecs.append((str(outcome.path), str(exc)))
            self.progress.emit(index, total)
        self.result_ready.emit(ecrites, echecs)


class SignaturesGUI(QMainWindow):
    """Fenêtre « Signatures » — classe un corpus de dessins par dessinateur.

    Signaux
    -------
    current_image_changed : pyqtSignal(object)
        ``Path`` de l'image sélectionnée, ``None`` sinon.  Contrat optionnel
        documenté dans :mod:`media_restorer.extensions` : la fenêtre racine
        s'y branche pour le dock « Infos, Exif » et l'aperçu.
    """

    current_image_changed = pyqtSignal(object)

    def __init__(
        self,
        target_path: Path | None = None,
        recursive: bool = True,
        *,
        scan_fn: ScanFn | None = None,
        embedder_factory: Callable[[], object] | None = None,
        detector_factory: Callable[[], object] | None = None,
        exiftool_runner: ExiftoolRunner | None = None,
    ) -> None:
        super().__init__()
        self._scan_fn = scan_fn
        self._embedder_factory = embedder_factory
        self._detector_factory = detector_factory
        self._exiftool_runner = exiftool_runner
        self._recursive = recursive
        self._target: Path | None = None
        self._worker: QThread | None = None
        self._outcomes: list[ScanOutcome] = []
        self._confident: list[ScanOutcome] = []
        self._pending_review: list[ScanOutcome] = []
        self._embedder_cache: tuple[object, tuple] | None = None
        self._detector_cache: tuple[object, tuple] | None = None

        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)

        self.actionScan = self._ui.actionScan
        self.actionReview = self._ui.actionReview
        self.actionWriteTags = self._ui.actionWriteTags
        self.actionExportCsv = self._ui.actionExportCsv

        self._build_parameters()
        self._build_preview_dock()
        self._build_parameters_dock()
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
        settings = app_settings()
        self._param_root = Parameter.create(
            name="params", type="group",
            children=[
                {
                    "name": "recursive", "title": "Parcours récursif",
                    "type": "bool",
                    "value": bool(settings.value(_KEY_RECURSIVE, self._recursive)),
                },
                {
                    "name": "force", "title": "Réanalyser tout",
                    "type": "bool", "value": False,
                },
                {
                    "name": "confident_threshold", "title": "Seuil de confiance",
                    "type": "float",
                    "value": float(settings.value(
                        _KEY_CONFIDENT_THRESHOLD, matching.DEFAULT_CONFIDENT_THRESHOLD
                    )),
                    "limits": (0.0, 1.0), "step": 0.01,
                },
                {
                    "name": "ambiguous_margin", "title": "Marge d'ambiguïté",
                    "type": "float",
                    "value": float(settings.value(
                        _KEY_AMBIGUOUS_MARGIN, matching.DEFAULT_AMBIGUOUS_MARGIN
                    )),
                    "limits": (0.0, 1.0), "step": 0.01,
                },
                {
                    "name": "min_score", "title": "Score minimal de détection",
                    "type": "float",
                    "value": float(settings.value(_KEY_MIN_SCORE, 0.05)),
                    "limits": (0.0, 1.0), "step": 0.01,
                },
                {
                    "name": "embedding_model", "title": "Modèle de comparaison",
                    "type": "list", "value": _config.load_embedding_model(),
                    "limits": {m.title: m.key for m in EMBEDDING_MODELS},
                },
                {
                    "name": "embedding_device", "title": "Appareil (comparaison)",
                    "type": "list", "value": _config.load_embedding_device(),
                    "limits": {DEVICE_TITLES[d]: d for d in DEVICE_ORDER},
                },
                {
                    "name": "detection_model", "title": "Modèle de localisation",
                    "type": "list", "value": _config.load_detection_model(),
                    "limits": {m: m for m in CANDIDATE_MODELS},
                },
                {
                    "name": "detection_device", "title": "Appareil (localisation)",
                    "type": "list", "value": _config.load_detection_device(),
                    "limits": {DEVICE_TITLES[d]: d for d in DEVICE_ORDER},
                },
            ],
        )
        for key in ("confident_threshold", "ambiguous_margin", "min_score", "recursive"):
            self._param_root.child(key).sigValueChanged.connect(self._save_settings)
        self._param_root.child("embedding_model").sigValueChanged.connect(
            lambda p, v: _config.save_embedding_model(v)
        )
        self._param_root.child("embedding_device").sigValueChanged.connect(
            lambda p, v: _config.save_embedding_device(v)
        )
        self._param_root.child("detection_model").sigValueChanged.connect(
            lambda p, v: _config.save_detection_model(v)
        )
        self._param_root.child("detection_device").sigValueChanged.connect(
            lambda p, v: _config.save_detection_device(v)
        )

        tree = ParameterTree(showHeader=False)
        tree.setParameters(self._param_root)
        self._param_tree = tree

    def _save_settings(self) -> None:
        settings = app_settings()
        settings.setValue(_KEY_CONFIDENT_THRESHOLD, self._param_root["confident_threshold"])
        settings.setValue(_KEY_AMBIGUOUS_MARGIN, self._param_root["ambiguous_margin"])
        settings.setValue(_KEY_MIN_SCORE, self._param_root["min_score"])
        settings.setValue(_KEY_RECURSIVE, self._param_root["recursive"])

    def _build_preview_dock(self) -> None:
        self._preview = ImagePreview()
        dock = QDockWidget("Aperçu", self)
        dock.setObjectName("dockApercu")
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

    def _build_parameters_dock(self) -> None:
        dock = QDockWidget("Paramètres", self)
        dock.setObjectName("dockParametres")
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        dock.setWidget(self._param_tree)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _selected_path(self) -> Path | None:
        items = self._ui.tableResults.selectedItems()
        if not items:
            return None
        raw = self._ui.tableResults.item(items[0].row(), 0).data(Qt.ItemDataRole.UserRole)
        return Path(raw) if raw else None

    @pyqtSlot()
    def _show_selected_preview(self) -> None:
        path = self._selected_path()
        if path is None:
            self._preview.clear()
            return
        self._preview.show_path(path)
        self.current_image_changed.emit(path)

    def _apply_target(self, path: Path | None) -> None:
        self._target = path
        if path is None:
            self.statusBar().showMessage("Aucune cible.")
            return
        if not path.is_dir():
            self.statusBar().showMessage(
                f"« {path.name} » est un fichier — choisissez un répertoire."
            )
            return
        self.actionScan.setEnabled(True)
        self.statusBar().showMessage(f"Cible : {path}")

    # ------------------------------------------------------------------
    # Fabriques de production (modèles transformers, mises en cache)
    # ------------------------------------------------------------------

    def _get_embedder(self):
        if self._embedder_factory is not None:
            return self._embedder_factory()
        model = self._param_root["embedding_model"] if hasattr(self, "_param_root") else _config.load_embedding_model()
        device_mode = self._param_root["embedding_device"] if hasattr(self, "_param_root") else _config.load_embedding_device()
        resolved_device, _explanation = resolve_device(device_mode)
        key = (model, resolved_device)
        if self._embedder_cache is None or self._embedder_cache[1] != key:
            self._embedder_cache = (build_embedder(model, device=resolved_device), key)
        return self._embedder_cache[0]

    def _get_detector(self):
        if self._detector_factory is not None:
            return self._detector_factory()
        model = self._param_root["detection_model"]
        device_mode = self._param_root["detection_device"]
        key = (model, device_mode)
        if self._detector_cache is None or self._detector_cache[1] != key:
            self._detector_cache = (build_detector(model, device=device_mode), key)
        return self._detector_cache[0]

    def _production_scan(self, root, **kwargs):
        return scan_corpus(root, detector=self._get_detector(), embedder=self._get_embedder(), **kwargs)

    def _thresholds(self) -> dict:
        return {
            "confident_threshold": float(self._param_root["confident_threshold"]),
            "ambiguous_margin": float(self._param_root["ambiguous_margin"]),
            "min_score": float(self._param_root["min_score"]),
        }

    # ------------------------------------------------------------------
    # Scan
    # ------------------------------------------------------------------

    @pyqtSlot()
    def on_actionScan_triggered(self) -> None:
        """Localise et compare la signature de chaque dessin de la cible. Lecture seule."""
        self._start_scan()

    def _start_scan(self) -> None:
        if self._target is None or self._worker is not None:
            return
        self._set_busy(True, "Scan en cours…")
        worker = _ScanWorker(
            self._target,
            recursive=bool(self._param_root["recursive"]),
            thresholds=self._thresholds(),
            force=bool(self._param_root["force"]),
            scan_fn=self._scan_fn or self._production_scan,
        )
        worker.progress.connect(self._on_progress)
        worker.result_ready.connect(self._on_scanned)
        worker.error.connect(self._on_error)
        self._worker = worker
        worker.start()

    @pyqtSlot(object)
    def _on_scanned(self, outcomes: list[ScanOutcome]) -> None:
        self._worker = None
        self._outcomes = outcomes
        self._confident = [o for o in outcomes if not o.needs_review]
        self._pending_review = [o for o in outcomes if o.needs_review]
        self._fill_table(outcomes)
        self.current_image_changed.emit(None)

        self.actionReview.setEnabled(bool(self._pending_review))
        self.actionWriteTags.setEnabled(bool(self._confident))
        self.actionExportCsv.setEnabled(bool(outcomes))
        self._set_busy(
            False,
            f"{len(self._confident)} reconnu(s) automatiquement, "
            f"{len(self._pending_review)} à revoir.",
        )

    def _fill_table(self, outcomes: list[ScanOutcome]) -> None:
        table = self._ui.tableResults
        table.setSortingEnabled(False)
        table.clear()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["Fichier", "Statut", "Dessinateur", "Score"])
        table.setRowCount(len(outcomes))
        for row, outcome in enumerate(outcomes):
            cellule = QTableWidgetItem(outcome.path.name)
            cellule.setData(Qt.ItemDataRole.UserRole, str(outcome.path))
            table.setItem(row, 0, cellule)
            self._set_row_cells(row, outcome)
        table.resizeColumnsToContents()
        table.setSortingEnabled(True)
        self._preview.clear()

    def _set_row_cells(self, row: int, outcome: ScanOutcome) -> None:
        table = self._ui.tableResults
        table.setItem(row, 1, QTableWidgetItem(_status_label(outcome)))
        table.setItem(row, 2, QTableWidgetItem(outcome.artist or ""))
        score = f"{outcome.candidates[0].score:.2f}" if outcome.candidates else ""
        table.setItem(row, 3, QTableWidgetItem(score))

    def _find_row(self, path: Path) -> int | None:
        """Ligne portant *path*, cherchée en direct — jamais un index mis en cache.

        Le tableau est TRIABLE (``sortingEnabled``) : un index retenu à
        l'ajout ne désigne plus la bonne ligne dès que l'utilisateur trie une
        colonne. Coût négligeable ici (une recherche par décision de revue,
        pas une boucle chaude).
        """
        table = self._ui.tableResults
        target = str(path)
        for row in range(table.rowCount()):
            item = table.item(row, 0)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == target:
                return row
        return None

    def _apply_outcome_update(self, updated: ScanOutcome) -> None:
        """Remplace l'outcome de ``updated.path`` PARTOUT : table ET ``self._outcomes``.

        Corrige un bug réel : ``_update_row`` ne rafraîchissait que la
        cellule affichée, jamais ``self._outcomes`` — l'export CSV relisait
        alors l'état d'avant la revue (tout en « À revoir »), quel que soit
        ce qui avait été décidé depuis. Les deux doivent changer ENSEMBLE, un
        seul point d'entrée pour ne plus jamais les faire diverger.
        """
        for index, outcome in enumerate(self._outcomes):
            if outcome.path == updated.path:
                self._outcomes[index] = updated
                break
        row = self._find_row(updated.path)
        if row is not None:
            self._set_row_cells(row, updated)

    # ------------------------------------------------------------------
    # Revue
    # ------------------------------------------------------------------

    @pyqtSlot()
    def on_actionReview_triggered(self) -> None:
        """Revoit un par un les dessins non reconnus automatiquement."""
        self._start_review()

    def _start_review(self) -> None:
        if not self._pending_review:
            return
        dialog = SignatureReviewDialog(known_artists=library.known_artists(), parent=self)
        while self._pending_review:
            outcome = self._pending_review[0]
            image = imread_oriented(outcome.path)
            if image is None:
                self._pending_review.pop(0)
                self._apply_outcome_update(replace(outcome, reason="illisible"))
                continue
            # Relu à CHAQUE item : un nom confirmé à l'instant (déjà écrit sur
            # disque par library.add_entry, voir _apply_decision) doit
            # apparaître dans la liste dès l'item suivant.
            dialog.set_known_artists(library.known_artists())
            dialog.set_outcome(outcome, image)
            result = dialog.exec()
            if result != QDialog.DialogCode.Accepted:
                break  # fenêtre fermée : la revue s'arrête, l'item reste en tête de file
            self._pending_review.pop(0)
            decision = dialog.decision()
            if decision is not None and not decision.skipped:
                self._apply_decision(outcome, decision)

        self.actionReview.setEnabled(bool(self._pending_review))
        self.actionWriteTags.setEnabled(bool(self._confident))
        self.statusBar().showMessage(
            f"{len(self._pending_review)} dessin(s) restant(s) à revoir."
        )

    def _apply_decision(self, outcome: ScanOutcome, decision) -> None:
        if decision.no_signature:
            try:
                tags.write_no_signature(outcome.path, runner=self._exiftool_runner)
            except Exception as exc:
                QMessageBox.warning(self, "Erreur — écriture", str(exc))
                return
            self._apply_outcome_update(replace(outcome, artist=None, reason="sans signature"))
            return

        if decision.artist is None or decision.region is None:
            return
        image = imread_oriented(outcome.path)
        if image is None:
            return
        xmin, ymin, xmax, ymax = decision.region
        crop = image[ymin:ymax, xmin:xmax]
        embedder = self._get_embedder()
        entry, warning = library.add_entry(decision.artist, crop, embedder=embedder)
        if warning is not None:
            QMessageBox.information(
                self, "Signature proche d'un autre auteur",
                f"Cette signature ressemble fortement (score {warning.score:.2f}) à une "
                f"signature déjà enregistrée pour « {warning.existing.artist} ». "
                f"Elle a quand même été ajoutée sous « {entry.artist} », comme demandé — "
                "vérifiez qu'il ne s'agit pas d'une confusion.",
            )
        try:
            tags.write_artist(outcome.path, entry.artist, runner=self._exiftool_runner)
        except Exception as exc:
            QMessageBox.warning(self, "Erreur — écriture", str(exc))
            return
        self._apply_outcome_update(replace(outcome, artist=entry.artist))
        self._recheck_pending()

    def _recheck_pending(self) -> None:
        """Recompare la file restante à la bibliothèque mise à jour.

        C'est ce qui réduit l'effort manuel à peu près à « une fois par
        dessinateur distinct » plutôt que « une fois par dessin » — voir la
        docstring de module.
        """
        if not self._pending_review:
            return
        entries = library.list_entries()
        if not entries:
            return
        embedder = self._get_embedder()
        vectors = descriptors.embed_many([e.path for e in entries], embedder)
        thresholds = self._thresholds()
        still_pending: list[ScanOutcome] = []
        for outcome in self._pending_review:
            if outcome.crop_path is None or not outcome.crop_path.exists():
                still_pending.append(outcome)
                continue
            vector = descriptors.embed_one(outcome.crop_path, embedder)
            candidates = matching.rank(vector, entries, vectors)
            verdict = matching.classify(
                candidates,
                confident_threshold=thresholds["confident_threshold"],
                ambiguous_margin=thresholds["ambiguous_margin"],
            )
            if verdict.regime == matching.CONFIDENT:
                try:
                    tags.write_artist(outcome.path, verdict.artist, runner=self._exiftool_runner)
                except Exception:
                    still_pending.append(replace(outcome, candidates=verdict.candidates))
                    continue
                resolved = replace(outcome, artist=verdict.artist, candidates=verdict.candidates)
                self._apply_outcome_update(resolved)
            else:
                reason = REASON_AMBIGUOUS if verdict.regime == matching.AMBIGUOUS else REASON_NO_MATCH
                still_pending.append(replace(outcome, reason=reason, candidates=verdict.candidates))
        self._pending_review = still_pending

    # ------------------------------------------------------------------
    # Écriture des étiquettes (lot confiant du scan initial)
    # ------------------------------------------------------------------

    @pyqtSlot()
    def on_actionWriteTags_triggered(self) -> None:
        """Écrit le dessinateur reconnu dans les métadonnées, pour tout le lot confiant."""
        self._write_tags()

    def _write_tags(self) -> None:
        if not self._confident or self._worker is not None:
            return
        reply = QMessageBox.question(
            self, "Écrire les étiquettes ?",
            f"Écrire le dessinateur reconnu de {len(self._confident)} image(s) dans "
            "leurs métadonnées ?\n\nLes fichiers seront modifiés (une copie "
            "« _original » est conservée par exiftool). Les étiquettes déjà "
            "présentes d'autres branches sont préservées.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            self.statusBar().showMessage("Écriture annulée.")
            return

        self._set_busy(True, "Écriture des étiquettes…")
        worker = _WriteWorker(self._confident, runner=self._exiftool_runner)
        worker.progress.connect(self._on_progress)
        worker.result_ready.connect(self._on_written)
        self._worker = worker
        worker.start()

    @pyqtSlot(int, list)
    def _on_written(self, written: int, failures: list) -> None:
        self._worker = None
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
        """Enregistre le tableau de résultats en CSV, sans modifier les images."""
        self._export_csv()

    def _export_csv(self) -> None:
        if not self._outcomes:
            return
        # À côté du dossier scanné (son PARENT), pas dedans : le CSV décrit
        # ce dossier de l'extérieur — nommé d'après lui (« 1949 » →
        # « signatures_1949.csv ») pour rester identifiable une fois sorti
        # de son contexte.
        if self._target is not None:
            suggested = self._target.parent / f"signatures_{self._target.name}.csv"
        else:
            suggested = Path("signatures.csv")
        path, _ = QFileDialog.getSaveFileName(
            self, "Exporter les résultats", str(suggested), tabular.FILE_FILTER
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = tabular.writer(f)
                writer.writerow(["fichier", "statut", "dessinateur", "score"])
                for outcome in self._outcomes:
                    score = f"{outcome.candidates[0].score:.2f}" if outcome.candidates else ""
                    writer.writerow(
                        [str(outcome.path), _status_label(outcome), outcome.artist or "", score]
                    )
        except OSError as exc:
            QMessageBox.critical(self, "Erreur — export CSV", str(exc))
            return
        self.statusBar().showMessage(f"Résultats exportés : {path}")

    # ------------------------------------------------------------------
    # État, progression, disposition
    # ------------------------------------------------------------------

    def _set_busy(self, busy: bool, message: str) -> None:
        self._ui.progressBar.setVisible(busy)
        self.actionScan.setEnabled(not busy and self._target is not None)
        self.statusBar().showMessage(message)

    @pyqtSlot(int, int)
    def _on_progress(self, index: int, total: int) -> None:
        self._ui.progressBar.setMaximum(max(total, 1))
        self._ui.progressBar.setValue(index)

    @pyqtSlot(str)
    def _on_error(self, message: str) -> None:
        self._worker = None
        self._set_busy(False, "Échec du scan.")
        QMessageBox.critical(self, "Erreur — scan", message)

    def showEvent(self, event) -> None:      # noqa: N802 — API Qt
        """Pose la disposition par défaut au tout premier affichage (voir ``gui_widgets``)."""
        super().showEvent(event)
        if self._needs_default_layout:
            self._needs_default_layout = False
            apply_default_dock_width(self, self._preview_dock)

    def closeEvent(self, event) -> None:      # noqa: N802 — API Qt
        """Mémorise la disposition des docks avant de fermer — seul chemin de sortie."""
        save_layout(self, _LAYOUT_PREFIX)
        super().closeEvent(event)


def _status_label(outcome: ScanOutcome) -> str:
    if outcome.artist is not None:
        return "Auto"
    labels = {
        REASON_NO_LOCATION: "À revoir (non localisée)",
        REASON_NO_MATCH: "À revoir (aucune correspondance)",
        REASON_AMBIGUOUS: "À revoir (ambigu)",
    }
    return labels.get(outcome.reason or "", "À revoir")

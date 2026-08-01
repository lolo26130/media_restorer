"""Tests de la fenêtre Pré-classement (classement d'un corpus + étiquetage).

Qt offscreen.  La mesure du corpus est toujours **injectée** (aucune image
décodée) et ``exiftool`` aussi (aucun fichier modifié).  Les workers sont rendus
synchrones — ``start`` → ``run``, jamais de vrai thread OS : voir le piège de
segfault documenté dans ``.claude/CLAUDE.md``.
"""
from pathlib import Path

import pytest
from PyQt6.QtWidgets import QMessageBox

import media_restorer.extensions.pre_classement  # noqa: F401 — enregistre l'extension
from media_restorer.engines.triage import CRITERIA, ImageSignals, ScanResult
from media_restorer.extensions.pre_classement.gui import (
    PreClassementGUI,
    _ClassifyWorker,
    _WriteWorker,
)


@pytest.fixture(autouse=True)
def _flush_qt_deletions():
    """Purge les suppressions Qt différées entre deux tests.

    Convention du ``CLAUDE.md`` pour toute fenêtre accumulant des widgets
    lourds : sans elle, la suite s'approche du seuil de segfault observé sur les
    fenêtres à ``pg.ImageView``.
    """
    yield
    import gc

    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is not None:
        app.processEvents()
        gc.collect()
        app.processEvents()


@pytest.fixture(autouse=True)
def _sync_workers(monkeypatch):
    """Rend les deux workers synchrones : ``run()`` sans thread OS."""
    monkeypatch.setattr(_ClassifyWorker, "start", _ClassifyWorker.run)
    monkeypatch.setattr(_WriteWorker, "start", _WriteWorker.run)


def _signals(name: str, w=200, h=100, cov=25.0, ink=0.0, paper=100.0) -> ImageSignals:
    """Paysage, encre moyenne, trait noir, papier jauni — sauf paramètres contraires."""
    return ImageSignals(Path(name), w, h, cov, ink, paper)


def _fake_scan(*_args, **kwargs):
    """Substitut instantané de ``scan_directory`` : deux images, une écartée."""
    on_progress = kwargs.get("on_progress")
    if on_progress is not None:
        on_progress(1, 2)
        on_progress(2, 2)
    return ScanResult(
        signals=[_signals("/corpus/a.jpg"), _signals("/corpus/b.jpg", w=100, h=200)],
        skipped_large=[Path("/corpus/enorme.tif")],
        unreadable=[],
    )


def _make_window(qtbot, tmp_path, *, scan_fn=_fake_scan, runner=None, target=None):
    target = target if target is not None else tmp_path
    win = PreClassementGUI(
        target_path=target, recursive=False, scan_fn=scan_fn,
        exiftool_runner=runner or (lambda args: "[]"),
    )
    qtbot.addWidget(win)
    return win


# ---------------------------------------------------------------------------
# Panneau de paramètres — engendré depuis le catalogue
# ---------------------------------------------------------------------------

def test_parameter_tree_is_built_from_the_criteria_catalogue(qtbot, tmp_path):
    """Ajouter un critère au catalogue doit suffire à le faire apparaître ici."""
    win = _make_window(qtbot, tmp_path)

    groupe = win._param_root.child("criteria")
    assert [c.name() for c in groupe.children()] == [c.key for c in CRITERIA]
    assert all(win._param_root["criteria", c.key] for c in CRITERIA)  # cochés par défaut


def test_default_resolution_cap_is_fifty_megapixels(qtbot, tmp_path):
    win = _make_window(qtbot, tmp_path)

    assert win._param_root["max_mpx"] == 50


def test_unchecking_a_criterion_removes_it_from_the_selection(qtbot, tmp_path):
    win = _make_window(qtbot, tmp_path)
    win._param_root.child("criteria", "resolution").setValue(False)

    assert "resolution" not in [c.key for c in win._selected_criteria()]


def test_selection_keeps_the_catalogue_order(qtbot, tmp_path):
    """L'ordre des colonnes ne doit pas dépendre de l'ordre de décochage."""
    win = _make_window(qtbot, tmp_path)
    for key in ("orientation", "ink_density", "ink_colour"):
        win._param_root.child("criteria", key).setValue(False)

    assert [c.key for c in win._selected_criteria()] == ["paper", "resolution"]


# ---------------------------------------------------------------------------
# Activation des actions
# ---------------------------------------------------------------------------

def test_actions_start_disabled_until_a_classification_exists(qtbot, tmp_path):
    win = _make_window(qtbot, tmp_path)

    assert win.actionClassify.isEnabled()          # une cible est fournie
    assert not win.actionWriteTags.isEnabled()
    assert not win.actionExportCsv.isEnabled()


def test_a_single_file_target_cannot_be_classified(qtbot, tmp_path):
    """Outil de lot : une image seule n'a rien à classer statistiquement."""
    fichier = tmp_path / "une.jpg"
    fichier.write_bytes(b"x")

    win = _make_window(qtbot, tmp_path, target=fichier)

    assert not win.actionClassify.isEnabled()
    assert "fichier" in win.statusBar().currentMessage()


def test_classifying_enables_writing_and_export(qtbot, tmp_path):
    win = _make_window(qtbot, tmp_path)
    win.on_actionClassify_triggered()

    assert win.actionWriteTags.isEnabled()
    assert win.actionExportCsv.isEnabled()


def test_classifying_without_any_criterion_is_refused(qtbot, tmp_path, monkeypatch):
    win = _make_window(qtbot, tmp_path)
    for c in CRITERIA:
        win._param_root.child("criteria", c.key).setValue(False)
    vus = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **kw: vus.append(a))

    win.on_actionClassify_triggered()

    assert vus                                     # prévenu, pas planté
    assert not win.actionWriteTags.isEnabled()


# ---------------------------------------------------------------------------
# Tableau et synthèse
# ---------------------------------------------------------------------------

def test_table_has_one_column_per_selected_criterion(qtbot, tmp_path):
    win = _make_window(qtbot, tmp_path)
    win._param_root.child("criteria", "resolution").setValue(False)
    win.on_actionClassify_triggered()

    table = win._ui.tableResults
    assert table.rowCount() == 2
    assert table.columnCount() == 1 + 4            # fichier + 4 critères restants
    entetes = [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())]
    assert entetes[0] == "Fichier"
    assert "Résolution" not in entetes


def test_table_shows_the_classified_values(qtbot, tmp_path):
    win = _make_window(qtbot, tmp_path)
    win.on_actionClassify_triggered()

    table = win._ui.tableResults
    ligne = {table.horizontalHeaderItem(i).text(): table.item(0, i).text()
             for i in range(table.columnCount())}
    assert ligne["Fichier"] == "a.jpg"
    assert ligne["Orientation"] == "Paysage"
    # En-tête = le *titre* du critère ; la *branche* (« Papier ») ne sert qu'aux
    # étiquettes écrites, elle n'apparaît pas dans l'interface.
    assert ligne["Teinte du papier"] == "Papier jauni"


def test_status_distinguishes_a_filtered_image_from_an_incident(qtbot, tmp_path):
    """Écartée par le plafond n'est pas illisible — l'utilisateur doit voir la nuance."""
    win = _make_window(qtbot, tmp_path)
    win.on_actionClassify_triggered()

    message = win.statusBar().currentMessage()
    assert "2 image(s) classée(s)" in message
    assert "1 écartée(s) par le plafond" in message
    assert "illisible" not in message              # aucune, donc non mentionné


def test_summary_is_filled_after_classification(qtbot, tmp_path):
    win = _make_window(qtbot, tmp_path)
    assert win._summary_label.text() == win._EMPTY_SUMMARY

    win.on_actionClassify_triggered()

    assert "orientation" in win._summary_label.text()


# ---------------------------------------------------------------------------
# Écriture des étiquettes
# ---------------------------------------------------------------------------

def _write_capturing_runner(calls):
    def runner(args):
        if any("=" in a for a in args if a.startswith("-")):
            calls.append(args)
            return "1 image files updated"
        return "[]"
    return runner


def test_writing_asks_once_for_the_whole_batch(qtbot, tmp_path, monkeypatch):
    """Une seule confirmation, pas une par image : 8 000 boîtes seraient absurdes."""
    questions = []
    monkeypatch.setattr(
        QMessageBox, "question",
        lambda *a, **kw: questions.append(a) or QMessageBox.StandardButton.Yes,
    )
    calls = []
    win = _make_window(qtbot, tmp_path, runner=_write_capturing_runner(calls))
    win.on_actionClassify_triggered()

    win.on_actionWriteTags_triggered()

    assert len(questions) == 1
    assert len(calls) == 2                         # une écriture par image


def test_declining_the_confirmation_writes_nothing(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **kw: QMessageBox.StandardButton.No)
    calls = []
    win = _make_window(qtbot, tmp_path, runner=_write_capturing_runner(calls))
    win.on_actionClassify_triggered()

    win.on_actionWriteTags_triggered()

    assert calls == []
    assert "annulée" in win.statusBar().currentMessage()


def test_written_tags_sit_under_the_triage_branch(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **kw: QMessageBox.StandardButton.Yes)
    calls = []
    win = _make_window(qtbot, tmp_path, runner=_write_capturing_runner(calls))
    win.on_actionClassify_triggered()
    win.on_actionWriteTags_triggered()

    prefixe = "-XMP-digiKam:TagsList="
    ecrits = [a[len(prefixe):] for a in calls[0] if a.startswith(prefixe)]
    assert "media_restorer/Tri/Orientation/Paysage" in ecrits
    assert all(e.startswith("media_restorer/Tri/") for e in ecrits)


def test_a_failing_image_does_not_stop_the_batch(qtbot, tmp_path, monkeypatch):
    """Une image verrouillée ne doit pas interrompre les milliers d'autres."""
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **kw: QMessageBox.StandardButton.Yes)
    avertissements = []
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **kw: avertissements.append(a))

    def runner(args):
        if any("=" in a for a in args if a.startswith("-")):
            if "/corpus/a.jpg" in args:
                raise RuntimeError("fichier verrouillé")
            return ""
        return "[]"

    win = _make_window(qtbot, tmp_path, runner=runner)
    win.on_actionClassify_triggered()
    win.on_actionWriteTags_triggered()

    assert avertissements                          # l'échec est signalé…
    assert "1 image(s) étiquetée(s)" in win.statusBar().currentMessage()  # …l'autre est passée


# ---------------------------------------------------------------------------
# Contrat optionnel avec le dock « Infos, Exif » de la racine
# ---------------------------------------------------------------------------

def test_current_image_changed_follows_the_batch_then_reverts(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **kw: QMessageBox.StandardButton.Yes)
    vus = []
    win = _make_window(qtbot, tmp_path, runner=_write_capturing_runner([]))
    win.current_image_changed.connect(vus.append)

    win.on_actionClassify_triggered()
    win.on_actionWriteTags_triggered()

    assert Path("/corpus/a.jpg") in vus            # suit l'image en cours…
    assert vus[-1] is None                         # …puis rend la main au résumé


# ---------------------------------------------------------------------------
# Export CSV
# ---------------------------------------------------------------------------

def test_csv_export_writes_a_row_per_image(qtbot, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog

    cible = tmp_path / "classement.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        lambda *a, **kw: (str(cible), "CSV (*.csv)"))
    win = _make_window(qtbot, tmp_path)
    win.on_actionClassify_triggered()

    win.on_actionExportCsv_triggered()

    lignes = cible.read_text(encoding="utf-8").strip().splitlines()
    assert lignes[0].startswith("fichier,Orientation")
    assert len(lignes) == 3                        # en-tête + deux images
    assert "Paysage" in lignes[1]


def test_cancelling_the_save_dialog_writes_no_file(qtbot, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **kw: ("", ""))
    win = _make_window(qtbot, tmp_path)
    win.on_actionClassify_triggered()

    win.on_actionExportCsv_triggered()

    assert list(tmp_path.glob("*.csv")) == []


# ---------------------------------------------------------------------------
# Dock « Aperçu » — sélection au clavier comprise
# ---------------------------------------------------------------------------

def _real_image(path: Path, size=(120, 80), colour=(200, 180, 150)) -> Path:
    """Écrit un vrai PNG : l'aperçu charge réellement le fichier."""
    import numpy as np
    from PIL import Image

    arr = np.zeros((size[1], size[0], 3), np.uint8)
    arr[:, :] = colour
    Image.fromarray(arr).save(path)
    return path


def _scan_of(*paths):
    """Substitut de scan renvoyant un signal par chemin réel donné."""
    def scan(*_args, **_kwargs):
        return ScanResult(
            signals=[_signals(str(p)) for p in paths],
            skipped_large=[], unreadable=[],
        )
    return scan


def test_preview_dock_exists_and_starts_empty(qtbot, tmp_path):
    win = _make_window(qtbot, tmp_path)

    assert win._preview_dock.windowTitle() == "Aperçu"
    assert win._preview.current_path is None


def test_selection_is_debounced_not_loaded_immediately(qtbot, tmp_path):
    """Maintenir une flèche traverse des dizaines de lignes : on ne charge que la dernière."""
    img = _real_image(tmp_path / "a.png")
    win = _make_window(qtbot, tmp_path, scan_fn=_scan_of(img))
    win.on_actionClassify_triggered()

    win._ui.tableResults.selectRow(0)

    assert win._preview_timer.isActive()           # différé…
    assert win._preview.current_path is None       # …donc rien n'est encore chargé
    assert win._preview_timer.isSingleShot()


def test_selection_change_loads_the_image(qtbot, tmp_path):
    img = _real_image(tmp_path / "a.png")
    win = _make_window(qtbot, tmp_path, scan_fn=_scan_of(img))
    win.on_actionClassify_triggered()

    win._ui.tableResults.selectRow(0)
    win._show_selected_preview()                   # ce que fera le minuteur

    assert win._preview.current_path == img


def test_keyboard_navigation_triggers_the_same_signal(qtbot, tmp_path):
    """« Sans cliquer » : itemSelectionChanged est émis aussi au clavier."""
    from PyQt6.QtCore import Qt as QtNS
    from PyQt6.QtGui import QKeyEvent
    from PyQt6.QtCore import QEvent

    a = _real_image(tmp_path / "a.png")
    b = _real_image(tmp_path / "b.png", colour=(50, 50, 50))
    win = _make_window(qtbot, tmp_path, scan_fn=_scan_of(a, b))
    win.on_actionClassify_triggered()
    table = win._ui.tableResults
    table.selectRow(0)
    win._show_selected_preview()

    table.setCurrentCell(1, 0)                     # équivaut à la flèche « bas »
    win._show_selected_preview()

    assert win._preview.current_path == b


def test_the_path_travels_with_the_cell_so_sorting_cannot_break_it(qtbot, tmp_path):
    """Le tableau est triable : l'index de ligne ne désigne plus le bon fichier."""
    a = _real_image(tmp_path / "zzz.png")
    b = _real_image(tmp_path / "aaa.png", colour=(50, 50, 50))
    win = _make_window(qtbot, tmp_path, scan_fn=_scan_of(a, b))
    win.on_actionClassify_triggered()
    table = win._ui.tableResults

    table.sortItems(0)                             # « aaa » passe en tête
    table.selectRow(0)
    win._show_selected_preview()

    assert win._preview.current_path == b          # et non l'image de rang 0 initial


def test_selection_also_updates_the_root_metadata_dock(qtbot, tmp_path):
    img = _real_image(tmp_path / "a.png")
    win = _make_window(qtbot, tmp_path, scan_fn=_scan_of(img))
    win.on_actionClassify_triggered()
    vus = []
    win.current_image_changed.connect(vus.append)

    win._ui.tableResults.selectRow(0)
    win._show_selected_preview()

    assert vus == [img]                            # aperçu et métadonnées cohérents


def test_an_unreadable_file_shows_a_message_instead_of_crashing(qtbot, tmp_path):
    casse = tmp_path / "casse.png"
    casse.write_bytes(b"ceci n'est pas une image")
    win = _make_window(qtbot, tmp_path, scan_fn=_scan_of(casse))
    win.on_actionClassify_triggered()

    win._ui.tableResults.selectRow(0)
    win._show_selected_preview()

    assert win._preview.current_path is None
    assert "illisible" in win._preview._message.text()


def test_reclassifying_clears_the_stale_preview(qtbot, tmp_path):
    img = _real_image(tmp_path / "a.png")
    win = _make_window(qtbot, tmp_path, scan_fn=_scan_of(img))
    win.on_actionClassify_triggered()
    win._ui.tableResults.selectRow(0)
    win._show_selected_preview()
    assert win._preview.current_path is not None

    win.on_actionClassify_triggered()              # nouvelle campagne

    assert win._preview.current_path is None

"""Tests de la fenêtre Signatures — scan, revue, écriture, export.

Qt offscreen.  Ni OWL-ViT ni SigLIP ne sont jamais chargés : détecteur et
empreinteur sont toujours des fonctions injectées.  ``exiftool`` est injecté.
Les workers sont rendus synchrones (``start`` → ``run``, jamais de vrai
thread OS) — même motif que ``test_gui_auto_face_id_register.py``.
``ImageCrop`` est un ``pg.ImageView`` : la fixture ``_flush_qt_deletions``
purge les suppressions différées entre les tests (piège documenté dans
``CLAUDE.md``).
"""
from __future__ import annotations

import gc
import json

import cv2
import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication, QMessageBox

import media_restorer.extensions.signatures  # noqa: F401 — enregistre l'extension
from media_restorer.engines.signatures import library, matching
from media_restorer.engines.signatures.pipeline import ScanOutcome
from media_restorer.extensions.signatures.gui import SignaturesGUI, _ScanWorker, _WriteWorker
from media_restorer.extensions.signatures.image_crop import ImageCrop
from media_restorer.extensions.signatures.review_dialog import (
    ReviewDecision,
    SignatureReviewDialog,
)


@pytest.fixture(autouse=True)
def _flush_qt_deletions():
    yield
    app = QApplication.instance()
    if app is not None:
        app.processEvents()
        gc.collect()
        app.processEvents()


@pytest.fixture(autouse=True)
def _sync_workers(monkeypatch):
    """Rend les workers synchrones : exécute ``run()`` sans thread OS."""
    monkeypatch.setattr(_ScanWorker, "start", _ScanWorker.run)
    monkeypatch.setattr(_WriteWorker, "start", _WriteWorker.run)


def _empty_read(args):
    return json.dumps([{"SourceFile": "x.jpg"}])


def _capturing_runner(calls):
    def runner(args):
        if any("=" in a for a in args if a.startswith("-")):
            calls.append(args)
            return "1 image files updated"
        return json.dumps([{"SourceFile": "x.jpg"}])
    return runner


def _fake_embedder(paths, on_progress=None):
    return np.zeros((len(paths), 2))


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    for name in ("a.png", "b.png"):
        cv2.imwrite(str(root / name), np.full((30, 30, 3), 120, dtype="uint8"))
    return root


def _make_window(qtbot, *, target=None, runner=None, scan_fn=None, embedder_factory=None):
    win = SignaturesGUI(
        target_path=target,
        exiftool_runner=runner or _empty_read,
        scan_fn=scan_fn,
        embedder_factory=embedder_factory or (lambda: _fake_embedder),
    )
    qtbot.addWidget(win)
    return win


# ---------------------------------------------------------------------------
# État initial / cible
# ---------------------------------------------------------------------------


def test_no_target_leaves_scan_disabled(qtbot):
    win = _make_window(qtbot)
    assert not win.actionScan.isEnabled()
    assert not win.actionReview.isEnabled()
    assert not win.actionWriteTags.isEnabled()


def test_a_file_target_is_refused(qtbot, corpus):
    win = _make_window(qtbot, target=corpus / "a.png")
    assert not win.actionScan.isEnabled()
    assert "répertoire" in win.statusBar().currentMessage().lower()


def test_a_directory_target_enables_scan(qtbot, corpus):
    win = _make_window(qtbot, target=corpus)
    assert win.actionScan.isEnabled()


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------


def _fake_scan_fn(outcomes):
    def scan_fn(root, *, recursive, thresholds, force, on_progress=None):
        return outcomes
    return scan_fn


def test_scan_fills_the_table_and_splits_confident_from_review(qtbot, corpus):
    a, b = corpus / "a.png", corpus / "b.png"
    entry = library.LibraryEntry("Cabrol", a)
    outcomes = [
        ScanOutcome(path=a, artist="Cabrol", candidates=(matching.Candidate(entry, 0.9),)),
        ScanOutcome(path=b, reason="no_match", candidates=()),
    ]
    win = _make_window(qtbot, target=corpus, scan_fn=_fake_scan_fn(outcomes))

    win.on_actionScan_triggered()

    assert win._ui.tableResults.rowCount() == 2
    assert len(win._confident) == 1
    assert len(win._pending_review) == 1
    assert win.actionWriteTags.isEnabled()
    assert win.actionReview.isEnabled()
    assert win.actionExportCsv.isEnabled()


def test_scan_error_is_reported_without_crashing(qtbot, corpus, monkeypatch):
    def boom(root, **kwargs):
        raise RuntimeError("modèle indisponible")

    win = _make_window(qtbot, target=corpus, scan_fn=boom)
    shown = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **kw: shown.append(a))

    win.on_actionScan_triggered()

    assert shown
    assert win.actionScan.isEnabled()  # réactivée pour réessayer


# ---------------------------------------------------------------------------
# Écriture des étiquettes (lot confiant)
# ---------------------------------------------------------------------------


def test_write_tags_writes_only_confident_outcomes_after_confirmation(qtbot, corpus, monkeypatch):
    a, b = corpus / "a.png", corpus / "b.png"
    entry = library.LibraryEntry("Cabrol", a)
    outcomes = [
        ScanOutcome(path=a, artist="Cabrol", candidates=(matching.Candidate(entry, 0.9),)),
        ScanOutcome(path=b, reason="no_match", candidates=()),
    ]
    calls = []
    win = _make_window(qtbot, target=corpus, scan_fn=_fake_scan_fn(outcomes),
                        runner=_capturing_runner(calls))
    win.on_actionScan_triggered()

    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **kw: QMessageBox.StandardButton.Yes
    )
    win.on_actionWriteTags_triggered()

    assert len(calls) == 1  # une seule image confiante à écrire
    assert any("Dessinateur" in a and "Cabrol" in a for a in calls[0])


def test_write_tags_cancelled_writes_nothing(qtbot, corpus, monkeypatch):
    a = corpus / "a.png"
    entry = library.LibraryEntry("Cabrol", a)
    outcomes = [ScanOutcome(path=a, artist="Cabrol", candidates=(matching.Candidate(entry, 0.9),))]
    calls = []
    win = _make_window(qtbot, target=corpus, scan_fn=_fake_scan_fn(outcomes),
                        runner=_capturing_runner(calls))
    win.on_actionScan_triggered()

    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **kw: QMessageBox.StandardButton.No
    )
    win.on_actionWriteTags_triggered()

    assert calls == []


# ---------------------------------------------------------------------------
# Revue — appliquée directement (pas de simulation souris/QDialog.exec)
# ---------------------------------------------------------------------------


def test_apply_decision_with_a_name_adds_to_the_library_and_writes(qtbot, corpus):
    a = corpus / "a.png"
    crop_path = corpus / "crop.png"
    cv2.imwrite(str(crop_path), np.full((10, 10, 3), 50, dtype="uint8"))
    outcome = ScanOutcome(path=a, reason="no_match", candidates=(), crop_path=crop_path)

    calls = []
    win = _make_window(qtbot, target=corpus, runner=_capturing_runner(calls))
    win._pending_review = [outcome]

    win._apply_decision(outcome, ReviewDecision(artist="Sennep", region=(0, 0, 20, 20)))

    assert library.known_artists() == ["Sennep"]
    assert any("Sennep" in a for call in calls for a in call)


def test_apply_decision_no_signature_writes_the_dedicated_leaf(qtbot, corpus):
    from media_restorer.engines.signatures import tags as _tags

    a = corpus / "a.png"
    outcome = ScanOutcome(path=a, reason="no_location", candidates=())
    calls = []
    win = _make_window(qtbot, target=corpus, runner=_capturing_runner(calls))
    win._pending_review = [outcome]

    win._apply_decision(outcome, ReviewDecision(no_signature=True))

    assert any(_tags.NO_SIGNATURE in a for call in calls for a in call)


def test_recheck_pending_auto_resolves_a_newly_matching_outcome(qtbot, corpus, tmp_path):
    """Le raffinement demandé : ajouter une signature résout les autres en attente."""
    a, b = corpus / "a.png", corpus / "b.png"
    crop_b = corpus / "crop_b.png"
    cv2.imwrite(str(crop_b), np.full((10, 10, 3), 77, dtype="uint8"))

    def matching_embedder(paths, on_progress=None):
        # Toutes les empreintes sont identiques : tout ce qui est comparé à
        # la bibliothèque devient un verdict confiant.
        return np.stack([np.array([1.0, 0.0]) for _ in paths])

    calls = []
    win = _make_window(qtbot, target=corpus, runner=_capturing_runner(calls),
                        embedder_factory=lambda: matching_embedder)
    outcome_b = ScanOutcome(path=b, reason="no_match", candidates=(), crop_path=crop_b)
    win._pending_review = [outcome_b]

    # Ajoute une première signature (déclenche _recheck_pending en interne).
    outcome_a = ScanOutcome(path=a, reason="no_match", candidates=())
    win._apply_decision(
        outcome_a, ReviewDecision(artist="Cabrol", region=(0, 0, 20, 20))
    )

    assert win._pending_review == []  # « b » a été résolu automatiquement
    assert any("Cabrol" in val for call in calls for val in call)


# ---------------------------------------------------------------------------
# ImageCrop — pilotage programmatique, pas de souris simulée
# ---------------------------------------------------------------------------


def test_image_crop_set_and_get_region_round_trip(qtbot):
    widget = ImageCrop()
    qtbot.addWidget(widget)
    widget.setup()
    widget.set_image(np.zeros((100, 200, 3), dtype="uint8"))  # h=100, w=200

    widget.set_region(10, 20, 60, 80)
    xmin, ymin, xmax, ymax = widget.get_region()

    assert (xmin, ymin, xmax, ymax) == (10, 20, 60, 80)


def test_image_crop_region_is_clamped_to_image_bounds(qtbot):
    widget = ImageCrop()
    qtbot.addWidget(widget)
    widget.setup()
    widget.set_image(np.zeros((50, 50, 3), dtype="uint8"))

    widget.set_region(-10, -10, 1000, 1000)
    xmin, ymin, xmax, ymax = widget.get_region()

    assert xmin == 0 and ymin == 0
    assert xmax == 50 and ymax == 50


def test_review_dialog_starts_with_the_given_known_artists(qtbot):
    dialog = SignatureReviewDialog(known_artists=["Cabrol", "Sennep"])
    qtbot.addWidget(dialog)

    items = [dialog._name_combo.itemText(i) for i in range(dialog._name_combo.count())]
    assert items == ["Cabrol", "Sennep"]


def test_review_dialog_set_known_artists_refreshes_the_combo(qtbot):
    """Bug corrigé : la liste déroulante restait vide, jamais rechargée."""
    dialog = SignatureReviewDialog(known_artists=[])
    qtbot.addWidget(dialog)
    assert dialog._name_combo.count() == 0

    dialog.set_known_artists(["Cabrol"])
    assert [dialog._name_combo.itemText(0)] == ["Cabrol"]

    # Un nom confirmé PENDANT la revue (donc déjà sur disque) doit réapparaître
    # dès l'appel suivant, sans reconstruire la boîte de dialogue.
    dialog.set_known_artists(["Cabrol", "Sennep"])
    items = [dialog._name_combo.itemText(i) for i in range(dialog._name_combo.count())]
    assert items == ["Cabrol", "Sennep"]


def test_start_review_refreshes_known_artists_between_items(qtbot, corpus, monkeypatch):
    """La revue relit la bibliothèque avant CHAQUE item, pas seulement à l'ouverture."""
    a, b = corpus / "a.png", corpus / "b.png"
    crop_a = corpus / "crop_a.png"
    cv2.imwrite(str(crop_a), np.full((10, 10, 3), 30, dtype="uint8"))
    outcome_a = ScanOutcome(path=a, reason="no_match", candidates=(), crop_path=crop_a)
    outcome_b = ScanOutcome(path=b, reason="no_match", candidates=())

    win = _make_window(qtbot, target=corpus, runner=_empty_read)
    win._pending_review = [outcome_a, outcome_b]

    seen_known_artists = []
    real_set_known_artists = SignatureReviewDialog.set_known_artists

    def spy_set_known_artists(self, artists):
        seen_known_artists.append(list(artists))
        real_set_known_artists(self, artists)

    monkeypatch.setattr(SignatureReviewDialog, "set_known_artists", spy_set_known_artists)

    # Le premier item se confirme sous « Cabrol », puis la boîte se ferme —
    # on n'a pas besoin d'aller jusqu'au bout pour vérifier le rechargement.
    calls = {"n": 0}

    def fake_exec(self):
        calls["n"] += 1
        if calls["n"] == 1:
            self._decision = ReviewDecision(artist="Cabrol", region=(0, 0, 5, 5))
            from PyQt6.QtWidgets import QDialog
            return QDialog.DialogCode.Accepted
        from PyQt6.QtWidgets import QDialog
        return QDialog.DialogCode.Rejected  # arrête la revue au second item

    monkeypatch.setattr(SignatureReviewDialog, "exec", fake_exec)

    win._start_review()

    assert seen_known_artists[0] == []          # rien avant le premier item
    assert seen_known_artists[1] == ["Cabrol"]   # Cabrol vient d'être confirmé


def test_image_crop_get_crop_extracts_the_right_region(qtbot):
    widget = ImageCrop()
    qtbot.addWidget(widget)
    widget.setup()
    image = np.arange(100).reshape(10, 10).astype("uint8")
    image_rgb = np.stack([image] * 3, axis=-1)
    widget.set_image(image_rgb)
    widget.set_region(2, 3, 5, 6)

    crop = widget.get_crop(image_rgb)

    assert crop.shape == (3, 3, 3)
    assert np.array_equal(crop, image_rgb[3:6, 2:5])


# ---------------------------------------------------------------------------
# Export CSV — nom et dossier suggérés
# ---------------------------------------------------------------------------


def test_export_csv_suggests_the_targets_parent_and_a_derived_name(qtbot, corpus, monkeypatch):
    """Bug corrigé : la boîte de dialogue s'ouvrait sans dossier ni nom pertinents.

    Le CSV est proposé à côté du dossier scanné (son PARENT), nommé d'après
    lui — un dossier « 1949 » suggère « signatures_1949.csv ».
    """
    a = corpus / "a.png"
    entry = library.LibraryEntry("Cabrol", a)
    outcomes = [ScanOutcome(path=a, artist="Cabrol", candidates=(matching.Candidate(entry, 0.9),))]
    win = _make_window(qtbot, target=corpus, scan_fn=_fake_scan_fn(outcomes))
    win.on_actionScan_triggered()

    from PyQt6.QtWidgets import QFileDialog

    seen = {}

    def fake_get_save_file_name(*args, **kwargs):
        seen["suggested"] = args[2] if len(args) > 2 else kwargs.get("directory", "")
        return "", ""

    monkeypatch.setattr(QFileDialog, "getSaveFileName", fake_get_save_file_name)
    win.on_actionExportCsv_triggered()

    assert seen["suggested"] == str(corpus.parent / f"signatures_{corpus.name}.csv")

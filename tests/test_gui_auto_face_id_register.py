"""Tests de la fenêtre Auto Face ID Register (détection + revue).

Qt offscreen.  La détection est toujours injectée (aucun modèle chargé) et le
worker est rendu synchrone (``start`` → ``run``, pas de vrai thread OS — voir
le piège documenté dans ``.claude/CLAUDE.md``).  ``exiftool`` est injecté.
"""
import json

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QInputDialog, QMessageBox

import media_restorer.extensions.auto_face_id_register  # noqa: F401 — enregistre l'extension
from media_restorer.extensions.auto_face_id_register.gui import (
    AutoFaceIdRegisterGUI,
    _DetectWorker,
)


@pytest.fixture(autouse=True)
def _flush_qt_deletions():
    """Purge les suppressions Qt différées (ImageClick = pg.ImageView) — voir
    la fixture homonyme de ``test_gui_manual_mouse_points.py``."""
    yield
    import gc

    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is not None:
        app.processEvents()
        gc.collect()
        app.processEvents()


@pytest.fixture(autouse=True)
def _sync_worker(monkeypatch):
    """Rend ``_DetectWorker.start()`` synchrone : exécute ``run()`` sans thread OS."""
    monkeypatch.setattr(_DetectWorker, "start", _DetectWorker.run)


@pytest.fixture
def image_file(tmp_path):
    # 100×100 : un pixel (x, y) vaut exactement (x %, y %) — repères en pourcentage.
    path = tmp_path / "caricature.jpg"
    cv2.imwrite(str(path), np.full((100, 100, 3), 180, np.uint8))
    return path


# Coordonnées de détection factices, en pourcentage (0–100).
_DETECTED = {"Left Eye": (10.0, 20.0), "Right Eye": (30.0, 20.0), "Nose": (20.0, 35.0)}


def _fake_detect(image, labels):
    """Détection factice : coordonnées connues pour les libellés connus."""
    return {label: _DETECTED.get(label) for label in labels}


def _empty_read(args):
    return json.dumps([{"SourceFile": "x.jpg"}])


def _is_write(args):
    return any(a.startswith("-UserComment=") for a in args)


def _make_window(qtbot, target=None, runner=None, detect_fn=_fake_detect, tmp_path=None):
    kwargs = dict(
        target_path=target,
        exiftool_runner=runner or _empty_read,
        detect_fn=detect_fn,
    )
    if tmp_path is not None:
        kwargs["labels_config_path"] = tmp_path / "labels.toml"
        kwargs["model_config_path"] = tmp_path / "model.toml"
    win = AutoFaceIdRegisterGUI(**kwargs)
    qtbot.addWidget(win)
    return win


def _send_key(win, key):
    ev = QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier)
    win._image_click.keyPressEvent(ev)


# ---------------------------------------------------------------------------
# État initial
# ---------------------------------------------------------------------------

def test_no_target_leaves_actions_disabled(qtbot):
    win = _make_window(qtbot)

    assert not win._ui.actionDetect.isEnabled()
    assert not win._ui.actionReview.isEnabled()
    assert not win._ui.actionSaveToMetadata.isEnabled()


def test_loading_a_file_enables_detect_only(qtbot, image_file):
    win = _make_window(qtbot, target=image_file)

    assert win._ui.actionDetect.isEnabled()
    assert not win._ui.actionReview.isEnabled()
    assert not win._ui.actionSaveToMetadata.isEnabled()


def test_directory_target_is_reported(qtbot, tmp_path):
    win = _make_window(qtbot, target=tmp_path)

    assert not win._ui.actionDetect.isEnabled()
    assert "fichier" in win.statusBar().currentMessage().lower()


# ---------------------------------------------------------------------------
# Détection
# ---------------------------------------------------------------------------

def test_detect_fills_points_and_enables_review_and_save(qtbot, image_file):
    win = _make_window(qtbot, target=image_file)

    win.on_actionDetect_triggered()

    assert win._tags["Left Eye"] == (10.0, 20.0)
    assert win._tags["Right Eye"] == (30.0, 20.0)
    assert win._tags["Nose"] == (20.0, 35.0)
    assert win._ui.actionReview.isEnabled()
    assert win._ui.actionSaveToMetadata.isEnabled()
    # Superposition : 3 points trouvés → 3 × (scatter + texte) = 6 items.
    assert len(win._image_click._existing_items) == 6
    assert "Left Eye : (10.0%, 20.0%)" in win._points_label.text()


def test_detect_reports_unfound_labels_as_none(qtbot, image_file):
    def detect_none(image, labels):
        return {label: None for label in labels}

    win = _make_window(qtbot, target=image_file, detect_fn=detect_none)
    win.on_actionDetect_triggered()

    assert all(v is None for v in win._tags.values())
    assert "(non trouvé)" in win._points_label.text()


def test_detect_error_is_reported_without_crashing(qtbot, image_file, monkeypatch):
    def boom(image, labels):
        raise RuntimeError("transformers absent")

    win = _make_window(qtbot, target=image_file, detect_fn=boom)
    shown = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **kw: shown.append(a))
    win.on_actionDetect_triggered()  # ne doit pas lever

    assert shown
    assert win._ui.actionDetect.isEnabled()  # réactivée pour réessayer
    assert "chec" in win.statusBar().currentMessage().lower()


# ---------------------------------------------------------------------------
# Revue / correction
# ---------------------------------------------------------------------------

def test_review_skip_keeps_detected_value(qtbot, image_file):
    win = _make_window(qtbot, target=image_file)
    win.on_actionDetect_triggered()
    win.on_actionReview_triggered()

    # Tout passer (Espace) → aucune correction → valeurs détectées conservées.
    for _ in win._read_labels():
        _send_key(win, Qt.Key.Key_Space)
    _send_key(win, Qt.Key.Key_Q)

    assert win._tags["Left Eye"] == (10, 20)
    assert win._tags["Nose"] == (20, 35)


def test_review_remark_overrides_detected_value(qtbot, image_file):
    win = _make_window(qtbot, target=image_file)
    win.on_actionDetect_triggered()
    win.on_actionReview_triggered()

    # Re-marquer le 1er repère (Left Eye) ailleurs, passer le reste.
    win._image_click._last_mouse_pos = (99, 88)
    _send_key(win, Qt.Key.Key_Return)
    for _ in win._read_labels()[1:]:
        _send_key(win, Qt.Key.Key_Space)
    _send_key(win, Qt.Key.Key_Q)

    assert win._tags["Left Eye"] == (99, 88)   # corrigé
    assert win._tags["Right Eye"] == (30, 20)  # détecté conservé


# ---------------------------------------------------------------------------
# Enregistrement
# ---------------------------------------------------------------------------

def test_save_writes_after_confirmation(qtbot, image_file, monkeypatch):
    writes = []

    def runner(args):
        if _is_write(args):
            writes.append(args)
            return ""
        return _empty_read(args)

    win = _make_window(qtbot, target=image_file, runner=runner)
    win.on_actionDetect_triggered()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **kw: QMessageBox.StandardButton.Yes)
    win.on_actionSaveToMetadata_triggered()

    assert len(writes) == 1
    tag = next(a for a in writes[0] if a.startswith("-UserComment="))
    payload = json.loads(tag[len("-UserComment="):])["media_restorer_landmarks"]
    assert payload["Left Eye"] == [10, 20]
    assert payload["Nose"] == [20, 35]


def test_save_cancelled_writes_nothing(qtbot, image_file, monkeypatch):
    writes = []

    def runner(args):
        if _is_write(args):
            writes.append(args)
            return ""
        return _empty_read(args)

    win = _make_window(qtbot, target=image_file, runner=runner)
    win.on_actionDetect_triggered()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **kw: QMessageBox.StandardButton.No)
    win.on_actionSaveToMetadata_triggered()

    assert writes == []
    assert "annul" in win.statusBar().currentMessage().lower()


# ---------------------------------------------------------------------------
# Choix du modèle (persisté)
# ---------------------------------------------------------------------------

def _fake_getitem(model, device):
    """Faux QInputDialog.getItem : renvoie *device* pour le 2e dialogue, *model* pour le 1er."""
    def getitem(*a, **kw):
        title = a[1] if len(a) > 1 else kw.get("title", "")
        if "Appareil" in title:
            return (device, True)
        return (model, True)
    return getitem


def test_choose_model_persists_model_and_device(qtbot, tmp_path, monkeypatch):
    from media_restorer.extensions.auto_face_id_register import config as cfg
    win = _make_window(qtbot, tmp_path=tmp_path)

    monkeypatch.setattr(
        QInputDialog, "getItem",
        _fake_getitem("google/owlv2-base-patch16-ensemble", "gpu"),
    )
    win.on_actionChooseModel_triggered()

    assert cfg.load_model(path=tmp_path / "model.toml") == "google/owlv2-base-patch16-ensemble"
    assert cfg.load_device(path=tmp_path / "model.toml") == "gpu"
    assert "owlv2" in win._model_label.text()
    assert "GPU" in win._model_label.text()


def test_device_defaults_to_cpu_in_the_label(qtbot, tmp_path):
    win = _make_window(qtbot, tmp_path=tmp_path)

    assert "CPU" in win._model_label.text()


def test_choose_model_cancelled_saves_nothing(qtbot, tmp_path, monkeypatch):
    win = _make_window(qtbot, tmp_path=tmp_path)

    monkeypatch.setattr(QInputDialog, "getItem", lambda *a, **kw: ("", False))
    win.on_actionChooseModel_triggered()

    assert not (tmp_path / "model.toml").exists()


# ---------------------------------------------------------------------------
# Icône
# ---------------------------------------------------------------------------

def test_extension_icon_is_registered_in_the_shared_qrc():
    import re
    from pathlib import Path

    import media_restorer
    from media_restorer.extensions.auto_face_id_register import AutoFaceIdRegisterExtension

    qrc = Path(media_restorer.__file__).parent / "resources" / "icons" / "media_restorer.qrc"
    registered = set(re.findall(r"<file>([^<]+)</file>", qrc.read_text()))

    assert AutoFaceIdRegisterExtension.icon.removeprefix(":/icons/") in registered

"""Tests de la fenêtre Manual Mouse Points (extension de désignation de repères).

Qt (offscreen via conftest).  Le lanceur ``exiftool`` est toujours injecté :
aucun test ne lance le vrai binaire ni n'écrit dans un fichier image.  Le
pointage clavier est piloté directement par des ``QKeyEvent`` synthétiques —
aucun vrai thread ni boucle d'événements imbriquée (voir le piège documenté
dans ``.claude/CLAUDE.md``), sans objet ici de toute façon puisque cette
extension n'a pas de worker.
"""
import json

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QMessageBox

# Importer le paquet enregistre l'extension (voir extensions/__init__.py) —
# nécessaire pour le test d'icône ci-dessous, indépendamment de l'ordre des tests.
import media_restorer.extensions.manual_mouse_points  # noqa: F401
from media_restorer.extensions.manual_mouse_points.gui import ManualMousePointsGUI


@pytest.fixture
def image_file(tmp_path):
    path = tmp_path / "portrait.jpg"
    cv2.imwrite(str(path), np.full((80, 60, 3), 180, np.uint8))
    return path


def _make_window(qtbot, target=None, runner=None):
    win = ManualMousePointsGUI(target_path=target, exiftool_runner=runner)
    qtbot.addWidget(win)
    return win


def _send_key(win, key):
    ev = QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier)
    win._image_click.keyPressEvent(ev)


def _mark_sequence(win, points):
    """Simule une désignation : *points* = liste de ``(x, y)`` (marquer) ou ``None`` (passer)."""
    for pt in points:
        if pt is None:
            _send_key(win, Qt.Key.Key_Space)
        else:
            win._image_click._last_mouse_pos = pt
            _send_key(win, Qt.Key.Key_Return)
    _send_key(win, Qt.Key.Key_Q)  # terminer


# ---------------------------------------------------------------------------
# Cible et état initial des actions
# ---------------------------------------------------------------------------

def test_no_target_leaves_actions_disabled(qtbot):
    win = _make_window(qtbot)

    assert not win._ui.actionStartTagging.isEnabled()
    assert not win._ui.actionSaveToMetadata.isEnabled()
    assert not win._ui.actionShowMetadata.isEnabled()


def test_loading_a_file_enables_tagging_and_show_but_not_save(qtbot, image_file):
    win = _make_window(qtbot, target=image_file)

    assert win._ui.actionStartTagging.isEnabled()
    assert win._ui.actionShowMetadata.isEnabled()
    assert not win._ui.actionSaveToMetadata.isEnabled()  # rien à enregistrer encore


def test_directory_target_is_reported_and_leaves_actions_disabled(qtbot, tmp_path):
    win = _make_window(qtbot, target=tmp_path)

    assert not win._ui.actionStartTagging.isEnabled()
    assert "fichier" in win.statusBar().currentMessage().lower()


# ---------------------------------------------------------------------------
# Libellés modifiables
# ---------------------------------------------------------------------------

def test_editing_labels_drives_the_designation(qtbot, image_file):
    win = _make_window(qtbot, target=image_file)
    win._param_root.child("labels").setValue("Mouth\nChin")

    win.on_actionStartTagging_triggered()
    _mark_sequence(win, [(5, 6), (7, 8)])

    assert list(win._tags) == ["Mouth", "Chin"]
    assert win._tags == {"Mouth": (5, 6), "Chin": (7, 8)}


def test_blank_labels_fall_back_to_defaults(qtbot, image_file):
    win = _make_window(qtbot, target=image_file)
    win._param_root.child("labels").setValue("   \n  \n")

    assert win._read_labels() == ["Left Eye", "Right Eye", "Nose", "Left Ear", "Right Ear"]


# ---------------------------------------------------------------------------
# Désignation → activation de l'enregistrement
# ---------------------------------------------------------------------------

def test_finishing_a_designation_enables_save(qtbot, image_file):
    win = _make_window(qtbot, target=image_file)
    win.on_actionStartTagging_triggered()
    assert not win._ui.actionSaveToMetadata.isEnabled()

    _mark_sequence(win, [(10, 20), None, (30, 40), None, None])

    assert win._ui.actionSaveToMetadata.isEnabled()
    assert win._tags["Left Eye"] == (10, 20)
    assert win._tags["Right Eye"] is None
    assert win._tags["Nose"] == (30, 40)


# ---------------------------------------------------------------------------
# Écriture dans les métadonnées (avec confirmation)
# ---------------------------------------------------------------------------

def test_saving_asks_for_confirmation_and_writes_when_accepted(qtbot, image_file, monkeypatch):
    calls = []
    win = _make_window(qtbot, target=image_file, runner=lambda args: calls.append(args) or "")
    win.on_actionStartTagging_triggered()
    _mark_sequence(win, [(10, 20), (30, 20), None, None, None])

    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **kw: QMessageBox.StandardButton.Yes)
    win.on_actionSaveToMetadata_triggered()

    assert len(calls) == 1
    tag = next(a for a in calls[0] if a.startswith("-UserComment="))
    payload = json.loads(tag[len("-UserComment="):])["media_restorer_landmarks"]
    assert payload == {"Left Eye": [10, 20], "Right Eye": [30, 20],
                       "Nose": None, "Left Ear": None, "Right Ear": None}


def test_saving_is_cancelled_when_confirmation_declined(qtbot, image_file, monkeypatch):
    calls = []
    win = _make_window(qtbot, target=image_file, runner=lambda args: calls.append(args) or "")
    win.on_actionStartTagging_triggered()
    _mark_sequence(win, [(10, 20), (30, 20), None, None, None])

    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **kw: QMessageBox.StandardButton.No)
    win.on_actionSaveToMetadata_triggered()

    assert calls == []  # rien écrit
    assert "annul" in win.statusBar().currentMessage().lower()


def test_save_failure_is_reported_without_crashing(qtbot, image_file, monkeypatch):
    def _boom(args):
        raise RuntimeError("exiftool introuvable")

    win = _make_window(qtbot, target=image_file, runner=_boom)
    win.on_actionStartTagging_triggered()
    _mark_sequence(win, [(10, 20), (30, 20), None, None, None])

    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **kw: QMessageBox.StandardButton.Yes)
    shown = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **kw: shown.append(a))
    win.on_actionSaveToMetadata_triggered()  # ne doit pas lever

    assert shown
    assert "chec" in win.statusBar().currentMessage().lower()


# ---------------------------------------------------------------------------
# Lecture des métadonnées
# ---------------------------------------------------------------------------

def test_show_metadata_reads_and_displays_stored_points(qtbot, image_file, monkeypatch):
    from media_restorer.landmarks import LandmarkSet
    stored = LandmarkSet(points={"Left Eye": (1, 2), "Nose": None}).to_json()
    runner = lambda args: json.dumps([{"SourceFile": "x", "UserComment": stored}])

    win = _make_window(qtbot, target=image_file, runner=runner)
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **kw: shown.append(a))
    win.on_actionShowMetadata_triggered()

    assert shown
    body = shown[0][2]
    assert "Left Eye" in body and "(1, 2)" in body and "Nose" in body


def test_show_metadata_reports_when_none_stored(qtbot, image_file, monkeypatch):
    runner = lambda args: json.dumps([{"SourceFile": "x"}])  # pas de UserComment

    win = _make_window(qtbot, target=image_file, runner=runner)
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **kw: shown.append(a))
    win.on_actionShowMetadata_triggered()

    assert shown
    assert "aucun" in shown[0][2].lower()


# ---------------------------------------------------------------------------
# Enregistrement de l'icône
# ---------------------------------------------------------------------------

def test_extension_icon_is_registered_in_the_shared_qrc():
    """Même garde-fou que test_extensions, ciblé sur cette extension."""
    import re
    from pathlib import Path

    import media_restorer
    from media_restorer.extensions.manual_mouse_points import ManualMousePointsExtension

    qrc = Path(media_restorer.__file__).parent / "resources" / "icons" / "media_restorer.qrc"
    registered = set(re.findall(r"<file>([^<]+)</file>", qrc.read_text()))

    assert ManualMousePointsExtension.icon.removeprefix(":/icons/") in registered

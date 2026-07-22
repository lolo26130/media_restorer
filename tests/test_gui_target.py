"""Tests de PhotoRestorationGUI(target_path=..., recursive=...).

Depuis l'introduction de la fenêtre racine (ImageTreatmentWindow), Media
Restorer ne choisit plus lui-même sa cible : elle lui est transmise à la
construction.  Voir media_restorer.gui_root et
media_restorer.extensions.media_restorer.
"""
import cv2
import numpy as np
import pytest

from media_restorer.engines import Engine
from media_restorer.extensions.media_restorer.gui import PhotoRestorationGUI


@pytest.fixture
def make_window(qtbot):
    created = []

    def _make(**kwargs) -> PhotoRestorationGUI:
        win = PhotoRestorationGUI(**kwargs)
        qtbot.addWidget(win)
        created.append(win)
        return win

    return _make


def test_no_target_leaves_the_window_in_its_neutral_state(make_window):
    """target_path=None (défaut) — comportement inchangé pour les autres tests."""
    win = make_window()

    assert win._original is None
    assert win._batch_dir is None
    assert not win._ui.actionBatch.isEnabled()
    assert not win._ui.actionRestore.isEnabled()


def test_file_target_loads_the_image_and_enables_restore(make_window, tmp_path):
    img = np.full((20, 30, 3), 128, dtype=np.uint8)
    path = tmp_path / "photo.png"
    cv2.imwrite(str(path), img)

    win = make_window(target_path=path)

    assert win._original is not None
    assert win._original.shape[:2] == (20, 30)
    assert win._ui.actionRestore.isEnabled()
    assert not win._ui.actionBatch.isEnabled()


def test_directory_target_enables_batch_not_restore(make_window, tmp_path):
    win = make_window(target_path=tmp_path, recursive=True)

    assert win._batch_dir == tmp_path
    assert win._batch_recursive is True
    assert win._ui.actionBatch.isEnabled()
    assert not win._ui.actionRestore.isEnabled()
    assert "récursif" in win._batch_status_label.text()


def test_directory_target_non_recursive_label_says_so(make_window, tmp_path):
    win = make_window(target_path=tmp_path, recursive=False)

    assert "non récursif" in win._batch_status_label.text()


def test_start_batch_without_a_directory_shows_a_status_message_not_a_crash(make_window):
    """Garde-fou : actionBatch ne devrait être activable qu'avec un répertoire,
    mais un appel direct à _start_batch() ne doit pas planter pour autant.
    """
    win = make_window()

    win._start_batch()  # ne doit pas lever

    assert "répertoire" in win.statusBar().currentMessage().lower()


def test_start_batch_uses_the_preset_directory_without_a_dialog(make_window, tmp_path, monkeypatch):
    """La cible étant déjà connue, aucun QFileDialog ne doit s'ouvrir."""
    from PyQt6.QtWidgets import QFileDialog

    img = np.zeros((10, 10, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "a.png"), img)

    def _boom(*a, **kw):
        raise AssertionError("QFileDialog ne doit pas être sollicité")

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", _boom)

    win = make_window(target_path=tmp_path, recursive=False)
    win._tab_widget.setCurrentIndex(win._tab_engines.index(Engine.REAL_ESRGAN))

    win._start_batch()  # ne doit pas lever ni ouvrir de dialogue

    assert win._batch_worker is not None

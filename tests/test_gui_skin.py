"""Tests du sélecteur de skin dans le dock « Tooltips » de la fenêtre principale."""
import pytest
from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import QApplication

from media_restorer.gui import PhotoRestorationGUI
from media_restorer.theme import THEME_DARK, THEME_LIGHT, THEME_SYSTEM, THEMES


def _skin_settings() -> QSettings:
    """QSettings("media_restorer","media_restorer") retombe sur NativeFormat,
    distinct d'IniFormat pour QSettings.setPath — l'isolation posée par
    tests/conftest.py (setPath(IniFormat, …)) ne s'appliquerait pas sans ce
    format explicite, et un test finirait par écrire dans la vraie config.
    """
    return QSettings(
        QSettings.Format.IniFormat,
        QSettings.Scope.UserScope,
        "media_restorer",
        "media_restorer",
    )


@pytest.fixture
def window(qtbot):
    win = PhotoRestorationGUI()
    qtbot.addWidget(win)
    return win


def test_skin_combo_offers_exactly_the_three_themes(window):
    combo = window._ui.comboSkin
    items = [combo.itemText(i) for i in range(combo.count())]

    assert items == list(THEMES)


def test_skin_combo_defaults_to_system(window):
    assert window._ui.comboSkin.currentText() == THEME_SYSTEM


@pytest.mark.parametrize("skin", [THEME_LIGHT, THEME_DARK])
def test_choosing_a_skin_changes_the_application_palette(window, skin):
    window._ui.comboSkin.setCurrentText(skin)

    base_lightness = QApplication.instance().palette().color(QPalette.ColorRole.Base).lightness()
    if skin == THEME_DARK:
        assert base_lightness < 60
    else:
        assert base_lightness > 200


def test_choosing_a_skin_persists_it_across_windows(window):
    """Le choix survit à la fermeture de la fenêtre — c'est tout l'intérêt.

    Une nouvelle fenêtre doit rouvrir avec le dernier skin choisi, pas
    retomber sur « Système » à chaque lancement.
    """
    window._ui.comboSkin.setCurrentText(THEME_DARK)

    assert _skin_settings().value("skin") == THEME_DARK


def test_new_window_restores_the_persisted_skin(window, qtbot):
    window._ui.comboSkin.setCurrentText(THEME_DARK)

    second = PhotoRestorationGUI()
    qtbot.addWidget(second)

    assert second._ui.comboSkin.currentText() == THEME_DARK


def test_first_launch_without_saved_preference_defaults_to_system(window):
    """Aucune préférence enregistrée (premier lancement) → « Système »."""
    _skin_settings().clear()

    fresh = PhotoRestorationGUI()

    assert fresh._ui.comboSkin.currentText() == THEME_SYSTEM
    fresh.close()

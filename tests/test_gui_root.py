"""Tests de la fenêtre racine ImageTreatmentWindow (media_restorer.gui_root)."""
import pytest
from PyQt6.QtGui import QPalette

from media_restorer.app_settings import SKIN_KEY, TOOLTIP_MODE_KEY, app_settings
from media_restorer.extensions import ExtensionContext
from media_restorer.gui_root import ImageTreatmentWindow
from media_restorer.theme import THEME_DARK, THEME_LIGHT, THEME_SYSTEM, THEMES


@pytest.fixture
def window(qtbot):
    win = ImageTreatmentWindow()
    qtbot.addWidget(win)
    return win


def test_window_title_is_image_treatment(window):
    assert window.windowTitle() == "Image Treatment"


# ---------------------------------------------------------------------------
# Menu Extensions peuplé dynamiquement
# ---------------------------------------------------------------------------

def test_media_restorer_is_registered_as_an_extension_action(window):
    """Media Restorer apparaît dans le menu Extensions sans câblage en dur."""
    names = [a.text() for a in window._launch_actions]

    assert "Media Restorer" in names


def test_extension_actions_start_disabled_without_a_target(window):
    """Impossible de lancer un outil avant d'avoir choisi une cible."""
    assert all(not a.isEnabled() for a in window._launch_actions)


def test_extension_actions_are_also_in_the_menu_and_toolbar(window):
    menu_texts = [a.text() for a in window._ui.menuExtensions.actions()]
    toolbar_actions = window._ui.toolBar.actions()

    for action in window._launch_actions:
        assert action.text() in menu_texts
        assert action in toolbar_actions


# ---------------------------------------------------------------------------
# Choix de cible
# ---------------------------------------------------------------------------

def test_choosing_a_file_enables_launch_but_not_recursive_mode(window, tmp_path):
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"\xff\xd8\xff")  # contenu sans importance ici

    window._set_target(img, is_directory=False)

    assert all(a.isEnabled() for a in window._launch_actions)
    assert not window._ui.comboRecursive.isEnabled()
    assert str(img) in window._ui.labelTarget.text()


def test_choosing_a_directory_enables_recursive_mode(window, tmp_path):
    window._set_target(tmp_path, is_directory=True)

    assert window._ui.comboRecursive.isEnabled()
    assert all(a.isEnabled() for a in window._launch_actions)


# ---------------------------------------------------------------------------
# Lancement d'une extension
# ---------------------------------------------------------------------------

def test_launch_without_target_shows_a_message_instead_of_crashing(window, monkeypatch):
    shown = []
    monkeypatch.setattr(
        "media_restorer.gui_root.QMessageBox.information",
        lambda *a, **kw: shown.append(a),
    )
    from media_restorer.extensions import all_extensions

    window._launch(all_extensions()[0])

    assert shown, "un message doit prévenir qu'aucune cible n'est choisie"


def test_launch_forwards_the_chosen_target_as_context(window, tmp_path, qtbot):
    captured = {}

    class _Recorder:
        name = "Recorder"
        description = "test"
        icon = ":/icons/gear--plus.png"

        def launch(self, context: ExtensionContext):
            captured["context"] = context
            from PyQt6.QtWidgets import QMainWindow
            win = QMainWindow()
            qtbot.addWidget(win)
            return win

    window._set_target(tmp_path, is_directory=True)
    window._ui.comboRecursive.setCurrentIndex(1)  # récursif

    window._launch(_Recorder())

    assert captured["context"].path == tmp_path
    assert captured["context"].recursive is True


def test_launch_keeps_a_reference_to_the_opened_window(window, tmp_path, qtbot):
    from PyQt6.QtWidgets import QMainWindow

    class _Recorder:
        name, description, icon = "Recorder", "test", ":/icons/gear--plus.png"

        def launch(self, context):
            win = QMainWindow()
            qtbot.addWidget(win)
            return win

    window._set_target(tmp_path, is_directory=False)
    before = len(window._open_windows)

    window._launch(_Recorder())

    assert len(window._open_windows) == before + 1


# ---------------------------------------------------------------------------
# Préférences persistées — apparence et mode tooltips
# ---------------------------------------------------------------------------

def test_skin_combo_offers_exactly_the_three_themes(window):
    combo = window._ui.comboSkin
    items = [combo.itemText(i) for i in range(combo.count())]

    assert items == list(THEMES)


def test_skin_combo_defaults_to_system(window):
    assert window._ui.comboSkin.currentText() == THEME_SYSTEM


@pytest.mark.parametrize("skin", [THEME_LIGHT, THEME_DARK])
def test_choosing_a_skin_changes_the_application_palette(window, skin):
    from PyQt6.QtWidgets import QApplication

    window._ui.comboSkin.setCurrentText(skin)

    base_lightness = QApplication.instance().palette().color(QPalette.ColorRole.Base).lightness()
    if skin == THEME_DARK:
        assert base_lightness < 60
    else:
        assert base_lightness > 200


def test_choosing_a_skin_persists_it(window):
    window._ui.comboSkin.setCurrentText(THEME_DARK)

    assert app_settings().value(SKIN_KEY) == THEME_DARK


def test_new_window_restores_the_persisted_skin(window, qtbot):
    window._ui.comboSkin.setCurrentText(THEME_DARK)

    second = ImageTreatmentWindow()
    qtbot.addWidget(second)

    assert second._ui.comboSkin.currentText() == THEME_DARK


def test_choosing_a_tooltip_mode_persists_it(window):
    window._ui.comboTooltipMode.setCurrentText("code")

    assert app_settings().value(TOOLTIP_MODE_KEY) == "code"


def test_first_launch_without_saved_preferences_defaults_to_docstrings_and_system(window):
    app_settings().clear()

    fresh = ImageTreatmentWindow()

    assert fresh._ui.comboTooltipMode.currentText() == "docstrings"
    assert fresh._ui.comboSkin.currentText() == THEME_SYSTEM
    fresh.close()


# ---------------------------------------------------------------------------
# Fermeture
# ---------------------------------------------------------------------------

def test_close_event_closes_windows_opened_from_the_root(window, tmp_path, qtbot):
    from PyQt6.QtWidgets import QMainWindow

    class _Recorder:
        name, description, icon = "Recorder", "test", ":/icons/gear--plus.png"

        def launch(self, context):
            win = QMainWindow()
            qtbot.addWidget(win)
            win.show()
            return win

    window._set_target(tmp_path, is_directory=False)
    window._launch(_Recorder())
    opened = window._open_windows[-1]
    assert opened.isVisible()

    window.close()

    assert not opened.isVisible()

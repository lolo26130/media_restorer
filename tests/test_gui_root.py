"""Tests de la fenêtre racine ImageTreatmentWindow (media_restorer.gui_root)."""
from pathlib import Path

import pytest
from PyQt6.QtGui import QPalette

from media_restorer.app_settings import SKIN_KEY, TOOLTIP_MODE_KEY, app_settings
from media_restorer.extensions import ExtensionContext
from media_restorer.gui_root import ImageTreatmentWindow
from media_restorer.theme import THEME_DARK, THEME_LIGHT, THEME_SYSTEM, THEMES
from media_restorer import tabular


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
# Dock « Infos, Exif »
# ---------------------------------------------------------------------------

def test_info_dock_is_present(window):
    assert window._info_dock.windowTitle() == "Infos, Exif"


def test_choosing_a_file_populates_the_info_dock(window, tmp_path):
    reads = []
    # read_fn/summary_fn injectés (comme exiftool_runner) — pas de vrai exiftool.
    # Forme hiérarchisée : {groupe: {tag: valeur}}.
    window._info_panel._read_fn = (
        lambda p: reads.append(p) or {"File": {"FileName": Path(p).name}}
    )
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"\xff\xd8\xff")

    window._set_target(img, is_directory=False)

    assert reads == [img]
    # un groupe racine « File » déplié, avec un tag enfant
    tree = window._info_panel._tree
    assert tree.topLevelItemCount() == 1
    assert tree.topLevelItem(0).text(0) == "File"
    assert tree.topLevelItem(0).childCount() == 1


def test_choosing_a_directory_shows_the_summary_not_a_file(window, tmp_path):
    summaries = []
    window._info_panel._summary_fn = (
        lambda p, recursive=False: summaries.append((p, recursive))
        or {"Répertoire": {"Images": "3"}}
    )
    window._info_panel._read_fn = lambda p: {"NE DOIT PAS": {"ÊTRE": "APPELÉ"}}

    window._set_target(tmp_path, is_directory=True)

    assert summaries == [(tmp_path, False)]
    tree = window._info_panel._tree
    assert tree.topLevelItem(0).text(0) == "Répertoire"
    assert tree.topLevelItem(0).child(0).text(1) == "3"


def test_extension_current_image_changed_updates_then_reverts_dock(window, tmp_path, qtbot):
    from PyQt6.QtWidgets import QMainWindow
    from PyQt6.QtCore import pyqtSignal

    reads, summaries = [], []
    window._info_panel._read_fn = (
        lambda p: reads.append(Path(p)) or {"File": {"FileName": Path(p).name}}
    )
    window._info_panel._summary_fn = (
        lambda p, recursive=False: summaries.append(p) or {"Répertoire": {"Images": "0"}}
    )

    class _BatchWindow(QMainWindow):
        current_image_changed = pyqtSignal(object)

    class _BatchExtension:
        name, description, icon = "Batch", "test", ":/icons/gear--plus.png"

        def launch(self, context):
            return _BatchWindow()

    window._set_target(tmp_path, is_directory=True)   # dock = résumé
    assert len(summaries) == 1

    window._launch(_BatchExtension())
    launched = window._open_windows[-1]
    qtbot.addWidget(launched)

    # image en cours → métadonnées de cette image
    img = tmp_path / "DSC_1006.JPG"
    launched.current_image_changed.emit(img)
    assert reads[-1] == img

    # fin de lot (None) → retour au résumé du répertoire
    launched.current_image_changed.emit(None)
    assert len(summaries) == 2  # résumé réaffiché


def test_single_image_extension_without_the_signal_is_fine(window, tmp_path, qtbot):
    """Une extension à image unique n'expose pas le signal — aucun câblage, aucune erreur."""
    from PyQt6.QtWidgets import QMainWindow

    class _PlainExtension:
        name, description, icon = "Plain", "test", ":/icons/gear--plus.png"

        def launch(self, context):
            win = QMainWindow()
            qtbot.addWidget(win)
            return win

    window._set_target(tmp_path, is_directory=False)
    window._launch(_PlainExtension())  # ne doit pas lever


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

# ---------------------------------------------------------------------------
# Aide
# ---------------------------------------------------------------------------

def test_open_help_uses_existing_docs_without_rebuilding(window, monkeypatch, tmp_path):
    index = tmp_path / "index.html"
    index.write_text("<html></html>")
    monkeypatch.setattr("media_restorer.gui_root._DOCS_HTML", tmp_path)
    run_calls = []
    monkeypatch.setattr("media_restorer.gui_root.subprocess.run", lambda *a, **kw: run_calls.append(a))
    opened = []
    monkeypatch.setattr(
        "media_restorer.gui_root.QDesktopServices.openUrl", lambda url: opened.append(url)
    )

    window._open_help()

    assert not run_calls  # aucune reconstruction : l'index existait déjà
    assert len(opened) == 1
    assert str(index) in opened[0].toLocalFile()
    assert str(index) in window.statusBar().currentMessage()


def test_open_help_rebuilds_when_docs_are_missing(window, monkeypatch, tmp_path):
    missing_html = tmp_path / "html"  # n'existe pas encore
    monkeypatch.setattr("media_restorer.gui_root._DOCS_HTML", missing_html)

    def _fake_build(cmd, **kwargs):
        # simule sphinx-build en créant l'index attendu
        missing_html.mkdir(parents=True, exist_ok=True)
        (missing_html / "index.html").write_text("<html></html>")

    monkeypatch.setattr("media_restorer.gui_root.subprocess.run", _fake_build)
    opened = []
    monkeypatch.setattr(
        "media_restorer.gui_root.QDesktopServices.openUrl", lambda url: opened.append(url)
    )

    window._open_help()

    assert len(opened) == 1  # la reconstruction a bien eu lieu puis ouvert le résultat


def test_open_help_reports_a_failed_rebuild_without_crashing(window, monkeypatch, tmp_path):
    import subprocess

    missing_html = tmp_path / "html"
    monkeypatch.setattr("media_restorer.gui_root._DOCS_HTML", missing_html)

    def _boom(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr("media_restorer.gui_root.subprocess.run", _boom)
    shown = []
    monkeypatch.setattr(
        "media_restorer.gui_root.QMessageBox.critical", lambda *a, **kw: shown.append(a)
    )
    opened = []
    monkeypatch.setattr(
        "media_restorer.gui_root.QDesktopServices.openUrl", lambda url: opened.append(url)
    )

    window._open_help()  # ne doit pas lever

    assert shown
    assert not opened  # rien à ouvrir, la reconstruction a échoué
    assert "chec" in window.statusBar().currentMessage().lower()


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


# ---------------------------------------------------------------------------
# Dock « Aperçu »
# ---------------------------------------------------------------------------

def _png(path: Path, colour=(200, 180, 150)) -> Path:
    import numpy as np
    from PIL import Image

    arr = np.zeros((60, 90, 3), np.uint8)
    arr[:, :] = colour
    Image.fromarray(arr).save(path)
    return path


def test_preview_dock_is_present_and_starts_empty(window):
    win = window

    assert win._preview_dock.windowTitle() == "Aperçu"
    assert win._preview_panel.current_path is None


def test_choosing_a_file_target_shows_it_in_the_preview(window, tmp_path):
    img = _png(tmp_path / "cible.png")
    win = window

    win._set_target(img, is_directory=False)

    assert win._preview_panel.current_path == img


def test_choosing_a_directory_leaves_the_preview_empty(window, tmp_path):
    """Un répertoire n'a pas une image à montrer mais des milliers."""
    img = _png(tmp_path / "cible.png")
    win = window
    win._set_target(img, is_directory=False)

    win._set_target(tmp_path, is_directory=True)

    assert win._preview_panel.current_path is None


def test_an_extension_image_drives_both_docks_then_reverts(window, tmp_path):
    """Le signal optionnel pilote métadonnées ET aperçu, sans rien en savoir."""
    cible = _png(tmp_path / "cible.png")
    autre = _png(tmp_path / "autre.png", colour=(20, 20, 20))
    win = window
    win._set_target(cible, is_directory=False)

    win._on_extension_image(autre)
    assert win._preview_panel.current_path == autre

    win._on_extension_image(None)                  # fin de lot
    assert win._preview_panel.current_path == cible


# ---------------------------------------------------------------------------
# Dock « Correspondances RAW »
# ---------------------------------------------------------------------------

def _inventaire(tmp_path):
    """Un inventaire factice couvrant les quatre catégories."""
    from media_restorer.engines.shots import MATCH_NAME, MATCH_SURE, ShotInventory, ShotPair

    nef, jpg = tmp_path / "a.nef", tmp_path / "a.jpg"
    return ShotInventory(
        pairs=[ShotPair(nef, jpg, MATCH_SURE,
                        meta={"Model": "NIKON D800", "ShutterCount": "14427"})],
        raw_only=[tmp_path / "orphelin.nef"],
        jpeg_only=[tmp_path / "scan.jpg"],
        unconfirmed=[ShotPair(tmp_path / "d.nef", tmp_path / "d.jpg", MATCH_NAME)],
    )


def test_shots_dock_is_present_and_tabified_with_the_exif_dock(window):
    """Tabifié pour ne pas empiler un troisième dock à droite."""
    assert window._shots_dock.windowTitle() == "Correspondances RAW"
    assert window._info_dock in window.tabifiedDockWidgets(window._shots_dock)


def test_shots_scan_never_starts_on_its_own(window, tmp_path):
    """Le scan coûte une minute : il ne part que sur clic explicite."""
    window._set_target(tmp_path, is_directory=True)

    assert window._shots_panel._button.isEnabled()      # prêt…
    assert window._shots_panel._inventory is None       # …mais rien n'a démarré
    assert "Scanner" in window._shots_panel._status.text()


def test_shots_scan_button_is_disabled_for_a_file_target(window, tmp_path):
    fichier = _png(tmp_path / "une.png")

    window._set_target(fichier, is_directory=False)

    assert not window._shots_panel._button.isEnabled()


def test_each_tab_carries_its_count(qtbot, tmp_path):
    from media_restorer.gui_shots import ShotInventoryPanel

    panneau = ShotInventoryPanel()
    qtbot.addWidget(panneau)

    panneau._on_scanned(_inventaire(tmp_path))

    titres = [panneau._tabs.tabText(i) for i in range(panneau._tabs.count())]
    assert titres == ["Paires (1)", "RAW sans JPEG (1)",
                      "JPEG sans RAW (1)", "⚠ À confirmer (1)"]
    assert "RAW sans JPEG" in panneau._status.text()


def test_selecting_a_pair_explains_why_it_was_matched(qtbot, tmp_path):
    """L'utilisateur doit pouvoir juger l'appariement, pas seulement le lire."""
    from media_restorer.gui_shots import ShotInventoryPanel

    vus = []
    panneau = ShotInventoryPanel(preview_fn=lambda p: vus.append(p) or None)
    qtbot.addWidget(panneau)
    panneau._on_scanned(_inventaire(tmp_path))

    arbre = panneau._trees["pairs"]
    arbre.setCurrentItem(arbre.topLevelItem(0))

    assert "NIKON D800" in panneau._details.text()
    assert "14427" in panneau._details.text()
    # Un RAW passe par l'extraction d'aperçu, JAMAIS par Pillow directement.
    assert vus == [tmp_path / "a.nef"]


def test_a_raw_without_extractable_preview_says_so(qtbot, tmp_path):
    from media_restorer.gui_shots import ShotInventoryPanel

    panneau = ShotInventoryPanel(preview_fn=lambda p: None)
    qtbot.addWidget(panneau)
    panneau._on_scanned(_inventaire(tmp_path))

    arbre = panneau._trees["raw_only"]
    arbre.setCurrentItem(arbre.topLevelItem(0))

    assert "aucun aperçu" in panneau._details.text()


def test_shots_export_writes_every_category(qtbot, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog

    from media_restorer.gui_shots import ShotInventoryPanel

    cible = tmp_path / "correspondances.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        lambda *a, **kw: (str(cible), "CSV (*.csv)"))
    panneau = ShotInventoryPanel()
    qtbot.addWidget(panneau)
    panneau._on_scanned(_inventaire(tmp_path))

    panneau._export_csv()

    lignes = cible.read_text(encoding="utf-8").strip().splitlines()
    # Séparateur TABULÉ (media_restorer.tabular) : un chemin peut contenir
    # une virgule, jamais une tabulation.
    assert lignes[0].split(tabular.DELIMITER) == [
        "categorie", "raw", "jpeg", "methode", "appareil", "developpe"
    ]
    assert len(lignes) == 5                       # en-tête + les quatre catégories

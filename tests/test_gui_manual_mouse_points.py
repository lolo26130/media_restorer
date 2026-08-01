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
from PyQt6.QtWidgets import QInputDialog, QMessageBox

# Importer le paquet enregistre l'extension (voir extensions/__init__.py) —
# nécessaire pour le test d'icône ci-dessous, indépendamment de l'ordre des tests.
import media_restorer.extensions.manual_mouse_points  # noqa: F401
from media_restorer.extensions.manual_mouse_points.gui import ManualMousePointsGUI


@pytest.fixture(autouse=True)
def _flush_qt_deletions():
    """Vide les suppressions Qt différées après chaque test de ce fichier.

    ``ImageClick`` étant un ``pg.ImageView``, chaque fenêtre construite ici
    ajoute une vue graphique de plus ; sans forcer le traitement des
    ``deleteLater`` entre les tests, ces objets C++ s'accumulent sur toute la
    session et rapprochent la suite du segfault d'accumulation déjà documenté
    (voir ``.claude/CLAUDE.md`` et ``tests/conftest.py``).  Autouse → instancié
    tôt, donc finalisé en dernier : s'exécute *après* que ``qtbot`` a fermé les
    widgets du test, au bon moment pour purger ce qu'il vient de programmer.
    """
    yield
    import gc

    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is not None:
        app.processEvents()
        gc.collect()
        app.processEvents()


@pytest.fixture
def image_file(tmp_path):
    # 100×100 : un pixel (x, y) vaut exactement (x %, y %) — les repères sont
    # stockés en pourcentage (0–100), voir media_restorer.landmarks.
    path = tmp_path / "portrait.jpg"
    cv2.imwrite(str(path), np.full((100, 100, 3), 180, np.uint8))
    return path


def _empty_read(args):
    """Réponse d'``exiftool -j`` pour une image sans nos repères."""
    return json.dumps([{"SourceFile": "x.jpg"}])


def _is_write(args):
    """Vrai si *args* est un appel d'écriture (une affectation « -TAG=… »).

    Critère volontairement indépendant du format de stockage : l'écriture des
    repères passe par plusieurs champs (étiquettes DigiKam + régions MWG, voir
    :mod:`media_restorer.landmarks`), la lecture n'affecte jamais rien.
    """
    return any("=" in a for a in args if a.startswith("-"))


def _tag_values(args, tag):
    """Valeurs affectées à *tag* dans une ligne de commande exiftool."""
    prefix = f"-{tag}="
    return [a[len(prefix):] for a in args if a.startswith(prefix)]


def _make_window(qtbot, target=None, runner=None, config_path=None):
    # Par défaut, un runner « aucune métadonnée » : aucun test ne lance le vrai
    # binaire exiftool à l'ouverture de l'image (qui lit désormais les repères).
    win = ManualMousePointsGUI(
        target_path=target, exiftool_runner=runner or _empty_read, config_path=config_path
    )
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
# Persistance des libellés dans le TOML de l'extension
# ---------------------------------------------------------------------------

def test_labels_are_loaded_from_config_on_startup(qtbot, tmp_path):
    from media_restorer import landmark_config as cfg
    cfg_file = tmp_path / "cfg.toml"
    cfg.save_labels(["Mouth", "Chin", "Left Brow"], path=cfg_file)

    win = _make_window(qtbot, config_path=cfg_file)

    assert win._read_labels() == ["Mouth", "Chin", "Left Brow"]


def test_add_label_extends_the_list_and_persists_to_config(qtbot, tmp_path, monkeypatch):
    from media_restorer import landmark_config as cfg
    cfg_file = tmp_path / "cfg.toml"
    win = _make_window(qtbot, config_path=cfg_file)
    before = win._read_labels()

    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **kw: ("Mouth", True))
    win._on_add_label()

    assert win._read_labels() == before + ["Mouth"]
    assert "Mouth" in cfg.load_labels([], path=cfg_file)  # écrit sur disque


def test_add_label_cancelled_changes_nothing(qtbot, tmp_path, monkeypatch):
    cfg_file = tmp_path / "cfg.toml"
    win = _make_window(qtbot, config_path=cfg_file)
    before = win._read_labels()

    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **kw: ("", False))  # annulé
    win._on_add_label()

    assert win._read_labels() == before
    assert not cfg_file.exists()  # rien écrit


def test_duplicate_label_is_not_added(qtbot, tmp_path, monkeypatch):
    cfg_file = tmp_path / "cfg.toml"
    win = _make_window(qtbot, config_path=cfg_file)

    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **kw: ("Nose", True))  # déjà présent
    win._on_add_label()

    assert win._read_labels().count("Nose") == 1


def test_editing_labels_persists_to_config(qtbot, tmp_path):
    from media_restorer import landmark_config as cfg
    cfg_file = tmp_path / "cfg.toml"
    win = _make_window(qtbot, config_path=cfg_file)

    win._param_root.child("labels").setValue("Alpha\nBeta")

    assert cfg.load_labels([], path=cfg_file) == ["Alpha", "Beta"]


# ---------------------------------------------------------------------------
# Désignation → activation de l'enregistrement
# ---------------------------------------------------------------------------

def test_marked_points_appear_in_the_dock(qtbot, image_file):
    win = _make_window(qtbot, target=image_file)
    assert win._results_label.text() == win._EMPTY_RESULTS

    win.on_actionStartTagging_triggered()
    _mark_sequence(win, [(10, 20), (30, 20), None, None, None])

    text = win._results_label.text()
    assert "Left Eye : (10.0%, 20.0%)" in text
    assert "Right Eye : (30.0%, 20.0%)" in text
    assert "Nose : (passé)" in text


def test_dock_updates_live_as_each_point_is_marked(qtbot, image_file):
    win = _make_window(qtbot, target=image_file)
    win.on_actionStartTagging_triggered()

    win._image_click._last_mouse_pos = (10, 20)
    _send_key(win, Qt.Key.Key_Return)  # premier point, avant même le Q final

    assert "Left Eye : (10.0%, 20.0%)" in win._results_label.text()


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
    writes = []

    def runner(args):
        # La lecture à l'ouverture passe aussi par ce runner : ne capturer que
        # les écritures pour ne pas compter le read initial.
        if _is_write(args):
            writes.append(args)
            return ""
        return _empty_read(args)

    win = _make_window(qtbot, target=image_file, runner=runner)
    win.on_actionStartTagging_triggered()
    _mark_sequence(win, [(10, 20), (30, 20), None, None, None])

    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **kw: QMessageBox.StandardButton.Yes)
    win.on_actionSaveToMetadata_triggered()

    assert len(writes) == 1
    # Étiquettes DigiKam : repères marqués, repères passés, et la provenance
    # « manuelle » propre à cette extension.
    assert _tag_values(writes[0], "XMP-digiKam:TagsList") == [
        "media_restorer/Repère/Left Eye",
        "media_restorer/Repère/Right Eye",
        "media_restorer/Repère ignoré/Nose",
        "media_restorer/Repère ignoré/Left Ear",
        "media_restorer/Repère ignoré/Right Ear",
        "media_restorer/Repérage manuel",
    ]
    # Coordonnées : régions MWG normalisées (10 % → 0.1), les repères passés
    # n'en produisent aucune.
    regions = _tag_values(writes[0], "XMP-mwg-rs:RegionInfo")[0]
    assert "Name=Left Eye" in regions and "X=0.1,Y=0.2" in regions
    assert "Nose" not in regions


def test_saving_is_cancelled_when_confirmation_declined(qtbot, image_file, monkeypatch):
    writes = []

    def runner(args):
        if _is_write(args):
            writes.append(args)
            return ""
        return _empty_read(args)

    win = _make_window(qtbot, target=image_file, runner=runner)
    win.on_actionStartTagging_triggered()
    _mark_sequence(win, [(10, 20), (30, 20), None, None, None])

    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **kw: QMessageBox.StandardButton.No)
    win.on_actionSaveToMetadata_triggered()

    assert writes == []  # rien écrit
    assert "annul" in win.statusBar().currentMessage().lower()


def test_save_failure_is_reported_without_crashing(qtbot, image_file, monkeypatch):
    def runner(args):
        # Échoue seulement à l'écriture ; la lecture d'ouverture doit réussir.
        if _is_write(args):
            raise RuntimeError("exiftool introuvable")
        return _empty_read(args)

    win = _make_window(qtbot, target=image_file, runner=runner)
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
    assert "Left Eye" in body and "(1.0%, 2.0%)" in body and "Nose" in body


def test_show_metadata_reports_when_none_stored(qtbot, image_file, monkeypatch):
    runner = lambda args: json.dumps([{"SourceFile": "x"}])  # pas de UserComment

    win = _make_window(qtbot, target=image_file, runner=runner)
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **kw: shown.append(a))
    win.on_actionShowMetadata_triggered()

    assert shown
    assert "aucun" in shown[0][2].lower()


def test_stored_points_appear_on_open_in_metadata_list_and_overlay(qtbot, image_file):
    from media_restorer.landmarks import LandmarkSet
    stored = LandmarkSet(points={"Left Eye": (1, 2), "Nose": None}).to_json()
    runner = lambda args: json.dumps([{"SourceFile": "x", "UserComment": stored}])

    # L'ouverture (construction avec cible) déclenche la lecture automatique.
    win = _make_window(qtbot, target=image_file, runner=runner)

    # Liste « métadonnées » du dock, indépendante de la liste « souris ».
    assert "Left Eye : (1.0%, 2.0%)" in win._metadata_label.text()
    assert "Nose : (passé)" in win._metadata_label.text()
    assert win._results_label.text() == win._EMPTY_RESULTS
    # Superposition sur l'image : un point marqué (Nose passé non dessiné)
    # → scatter + texte = 2 items existants.
    assert len(win._image_click._existing_items) == 2


def test_metadata_and_mouse_lists_stay_independent(qtbot, image_file):
    from media_restorer.landmarks import LandmarkSet
    stored = LandmarkSet(points={"Left Eye": (1, 2)}).to_json()

    def runner(args):
        if _is_write(args):
            return ""
        return json.dumps([{"SourceFile": "x", "UserComment": stored}])

    win = _make_window(qtbot, target=image_file, runner=runner)
    win.on_actionStartTagging_triggered()
    _mark_sequence(win, [(50, 60), (10, 10), None, None, None])

    # La liste souris reflète le pointage ; la liste métadonnées reste celle
    # lue à l'ouverture — les deux ne se mélangent pas.
    assert "Left Eye : (50.0%, 60.0%)" in win._results_label.text()
    assert "Left Eye : (1.0%, 2.0%)" in win._metadata_label.text()


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

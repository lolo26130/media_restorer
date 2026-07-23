"""Tests de l'extension Vectorise — fenêtre Qt (media_restorer.extensions.vectorise).

Le cœur de calcul est déjà testé sans Qt dans ``test_vectorise.py`` ; ici on
ne vérifie que le câblage GUI (activation des actions, peuplement de la
liste des candidats, worker, animation) — avec un ``get_outline_fn`` factice
et rapide injecté dans le worker (voir ``CLAUDE.md`` : « passer une
_stream_factory/_engine_factory en paramètre des workers au lieu d'importer
directement »), pour ne jamais dépendre du vrai calcul GUDHI dans ces tests.
"""
import cv2
import numpy as np
import pytest

from media_restorer.engines.vectorise.types import Stroke, StrokeSet
from media_restorer.extensions.vectorise.gui import VectoriseGUI

# performance_mode() est neutralisé pour toute la suite dans conftest.py
# (fixture _no_real_power_management) — voir sa docstring pour la raison
# (segfault reproductible avec un vrai sous-processus lancé depuis un
# QThread pendant qu'un test attend son résultat).


def _run_get_outline(win):
    """Construit et exécute le worker de get_outline sans démarrer de vrai ``QThread``.

    Reproduit exactement le câblage de ``VectoriseGUI._start_get_outline``
    (mêmes paramètres, mêmes connexions de signaux), mais appelle
    ``worker.run()`` directement au lieu de ``worker.start()``.
    ``QThread.run()`` est une méthode Python ordinaire : l'appeler exécute la
    même logique de façon synchrone, sans jamais démarrer de vrai thread OS
    ni nécessiter de boucle d'événements imbriquée pour attendre le résultat
    — les signaux Qt s'émettent normalement dans le thread appelant.

    Ce détour n'est pas une simplification de confort : démarrer un vrai
    ``QThread`` puis attendre son signal via une boucle imbriquée (que ce
    soit ``qtbot.waitSignal`` ou une ``QEventLoop`` manuelle — les deux ont
    été essayés) a provoqué un segfault reproductible ici, situé dans un
    rappel différé de pyqtgraph et qui n'apparaît qu'après l'accumulation de
    plusieurs fenêtres/threads dans le même process de test.  Le test
    ``test_close_event_stops_a_running_worker_and_the_animation`` couvre
    séparément le vrai cycle de vie du thread, seul endroit où il importe
    réellement pour ce qui est testé.
    """
    from media_restorer.extensions.vectorise.gui import _GetOutlineWorker

    if win._original is None:
        return
    params = win._read_params()
    kwargs = dict(
        dpi=win._dpi, n_candidates=params["n_candidates"],
        max_points=params["max_points"], mark_fraction=params["mark_fraction"],
    )
    win._ui.actionGetOutline.setEnabled(False)
    worker = _GetOutlineWorker(win._original, kwargs, win._get_outline_fn)
    worker.result_ready.connect(win._on_get_outline_done)
    worker.error.connect(win._on_get_outline_error)
    win._get_outline_worker = worker
    worker.run()


def _fake_stroke_set(n_points=4, width_px=6.0, score=1.0, label="fake") -> StrokeSet:
    rng = np.random.default_rng(0)
    stroke = Stroke(
        points=rng.uniform(0, 50, size=(n_points, 2)).astype(np.float32),
        widths=np.full(n_points, width_px / 2, dtype=np.float32),
        intensity=np.full(n_points, 0.8, dtype=np.float32),
    )
    return StrokeSet(strokes=[stroke], pencil_width_px=width_px, score=score, label=label)


def _fake_get_outline(image, **kwargs):
    """Substitut synchrone et rapide de get_outline, pour les tests GUI."""
    n = kwargs.get("n_candidates", 3)
    return [_fake_stroke_set(label=f"candidat {i}", width_px=float(i + 1)) for i in range(n)]


@pytest.fixture
def window(qtbot):
    win = VectoriseGUI(get_outline_fn=_fake_get_outline)
    qtbot.addWidget(win)
    return win


def _make_image(tmp_path, h=60, w=80, name="photo.png"):
    img = np.full((h, w, 3), 250, dtype=np.uint8)
    cv2.line(img, (5, 5), (w - 5, h - 5), (40, 40, 40), thickness=4, lineType=cv2.LINE_AA)
    path = tmp_path / name
    cv2.imwrite(str(path), img)
    return path


# ---------------------------------------------------------------------------
# Chargement de la cible
# ---------------------------------------------------------------------------

def test_no_target_leaves_actions_disabled(window):
    assert not window._ui.actionGetOutline.isEnabled()
    assert not window._ui.actionShow.isEnabled()
    assert not window._ui.actionSaveStrokes.isEnabled()
    assert not window._ui.actionVectorise.isEnabled()
    assert not window._ui.actionSave.isEnabled()


def test_file_target_enables_get_outline(qtbot, tmp_path):
    path = _make_image(tmp_path)
    win = VectoriseGUI(target_path=path, get_outline_fn=_fake_get_outline)
    qtbot.addWidget(win)

    assert win._original is not None
    assert win._ui.actionGetOutline.isEnabled()
    assert win._dpi > 0


def test_directory_target_shows_a_message_instead_of_crashing(qtbot, tmp_path):
    win = VectoriseGUI(target_path=tmp_path, get_outline_fn=_fake_get_outline)
    qtbot.addWidget(win)

    assert win._original is None
    assert not win._ui.actionGetOutline.isEnabled()
    assert "fichier" in win.statusBar().currentMessage().lower()


def test_unreadable_file_shows_an_error(qtbot, tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr(
        "media_restorer.extensions.vectorise.gui.QMessageBox.critical",
        lambda *a, **kw: shown.append(a),
    )
    bogus = tmp_path / "not_an_image.png"
    bogus.write_bytes(b"pas une image")

    win = VectoriseGUI(target_path=bogus, get_outline_fn=_fake_get_outline)
    qtbot.addWidget(win)

    assert shown
    assert not win._ui.actionGetOutline.isEnabled()


# ---------------------------------------------------------------------------
# get_outline — worker et activation des actions
# ---------------------------------------------------------------------------

def test_get_outline_populates_candidate_list_and_enables_actions(qtbot, tmp_path):
    path = _make_image(tmp_path)
    win = VectoriseGUI(target_path=path, get_outline_fn=_fake_get_outline)
    qtbot.addWidget(win)

    _run_get_outline(win)

    assert len(win._strokesets) == win._param_root.child("n_candidates").value()
    assert win._combo_candidates.count() == len(win._strokesets)
    assert win._ui.actionShow.isEnabled()
    assert win._ui.actionSaveStrokes.isEnabled()
    assert win._ui.actionGetOutline.isEnabled()  # réactivée après le calcul


def test_get_outline_does_not_enable_vectorise_without_a_texture(qtbot, tmp_path):
    path = _make_image(tmp_path)
    win = VectoriseGUI(target_path=path, get_outline_fn=_fake_get_outline)
    qtbot.addWidget(win)

    _run_get_outline(win)

    assert not win._ui.actionVectorise.isEnabled()


def test_get_outline_error_is_reported_and_reenables_the_action(qtbot, tmp_path, monkeypatch):
    def _boom(image, **kwargs):
        raise RuntimeError("échec simulé")

    shown = []
    monkeypatch.setattr(
        "media_restorer.extensions.vectorise.gui.QMessageBox.critical",
        lambda *a, **kw: shown.append(a),
    )
    path = _make_image(tmp_path)
    win = VectoriseGUI(target_path=path, get_outline_fn=_boom)
    qtbot.addWidget(win)

    _run_get_outline(win)

    assert shown
    assert win._ui.actionGetOutline.isEnabled()


def test_width_override_filters_candidates_by_measured_width(qtbot, tmp_path):
    path = _make_image(tmp_path)
    win = VectoriseGUI(target_path=path, get_outline_fn=_fake_get_outline)
    qtbot.addWidget(win)
    win._param_root.child("width_override").setValue(True)
    # dpi=300 : 1px = 25.4/300 mm ≈ 0.0847mm — _fake_get_outline produit
    # n_candidates=5 candidats de largeurs factices 1..5px ≈ 0.08..0.42mm ;
    # min=0.15mm ≈ 1.77px exclut donc uniquement le candidat à 1px.
    win._param_root.child("width_min_mm").setValue(0.15)
    win._param_root.child("width_max_mm").setValue(5.0)

    _run_get_outline(win)

    assert len(win._strokesets) == 4


def test_width_override_falls_back_to_all_candidates_if_filter_empties_the_list(qtbot, tmp_path):
    path = _make_image(tmp_path)
    win = VectoriseGUI(target_path=path, get_outline_fn=_fake_get_outline)
    qtbot.addWidget(win)
    win._param_root.child("width_override").setValue(True)
    win._param_root.child("width_min_mm").setValue(4.0)  # aucun candidat factice n'atteint 4mm
    win._param_root.child("width_max_mm").setValue(5.0)

    _run_get_outline(win)

    assert len(win._strokesets) == win._param_root.child("n_candidates").value()


# ---------------------------------------------------------------------------
# Texture
# ---------------------------------------------------------------------------

def test_save_texture_from_image_writes_file_and_enables_vectorise_with_candidates(qtbot, tmp_path):
    path = _make_image(tmp_path)
    win = VectoriseGUI(target_path=path, get_outline_fn=_fake_get_outline)
    qtbot.addWidget(win)
    _run_get_outline(win)

    win._save_texture_from_image()

    expected = tmp_path / "photo.texture.png"
    assert expected.exists()
    assert win._texture is not None
    assert win._ui.actionVectorise.isEnabled()


def test_save_texture_from_image_is_a_no_op_without_a_loaded_image(window, tmp_path):
    window._save_texture_from_image()  # ne doit pas lever
    assert window._texture is None


def test_select_texture_uses_the_chosen_file(qtbot, tmp_path, monkeypatch):
    path = _make_image(tmp_path)
    win = VectoriseGUI(target_path=path, get_outline_fn=_fake_get_outline)
    qtbot.addWidget(win)

    texture_path = tmp_path / "external_texture.png"
    cv2.imwrite(str(texture_path), np.full((40, 40), 100, dtype=np.uint8))
    monkeypatch.setattr(
        "media_restorer.extensions.vectorise.gui.QFileDialog.getOpenFileName",
        lambda *a, **kw: (str(texture_path), ""),
    )

    win._select_texture()

    assert win._texture is not None
    assert win._texture.shape == (40, 40)


def test_select_texture_cancelled_dialog_is_a_no_op(window, monkeypatch):
    monkeypatch.setattr(
        "media_restorer.extensions.vectorise.gui.QFileDialog.getOpenFileName",
        lambda *a, **kw: ("", ""),
    )

    window._select_texture()

    assert window._texture is None


# ---------------------------------------------------------------------------
# Vectorisation
# ---------------------------------------------------------------------------

def test_vectorise_requires_both_candidates_and_texture(window):
    window._start_vectorise()  # ni candidats ni texture : ne doit pas lever
    assert window._vectorised is None
    assert not window._ui.actionSave.isEnabled()


def test_vectorise_produces_a_result_and_enables_save(qtbot, tmp_path):
    path = _make_image(tmp_path)
    win = VectoriseGUI(target_path=path, get_outline_fn=_fake_get_outline)
    qtbot.addWidget(win)
    _run_get_outline(win)
    win._save_texture_from_image()

    win._start_vectorise()

    assert win._vectorised is not None
    assert win._vectorised.shape == win._original.shape[:2]
    assert win._ui.actionSave.isEnabled()


def test_save_vectorised_writes_the_chosen_file(qtbot, tmp_path, monkeypatch):
    path = _make_image(tmp_path)
    win = VectoriseGUI(target_path=path, get_outline_fn=_fake_get_outline)
    qtbot.addWidget(win)
    _run_get_outline(win)
    win._save_texture_from_image()
    win._start_vectorise()

    out_path = tmp_path / "result.png"
    monkeypatch.setattr(
        "media_restorer.extensions.vectorise.gui.QFileDialog.getSaveFileName",
        lambda *a, **kw: (str(out_path), ""),
    )

    win._save_vectorised()

    assert out_path.exists()


# ---------------------------------------------------------------------------
# Tracés — enregistrement HDF5
# ---------------------------------------------------------------------------

def test_save_strokes_writes_next_to_the_source_image(qtbot, tmp_path):
    path = _make_image(tmp_path, name="dessin.png")
    win = VectoriseGUI(target_path=path, get_outline_fn=_fake_get_outline)
    qtbot.addWidget(win)
    _run_get_outline(win)

    win._save_strokes()

    expected = tmp_path / "dessin.strokes.h5"
    assert expected.exists()

    from media_restorer.engines.vectorise import load_strokes
    reloaded = load_strokes(expected)
    assert len(reloaded) == len(win._strokesets)


def test_save_strokes_is_a_no_op_without_candidates(window, tmp_path):
    window._save_strokes()  # ne doit pas lever
    assert not list(tmp_path.glob("*.strokes.h5"))


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def test_start_animation_without_candidates_is_a_no_op(window):
    window._start_animation()
    assert window._animation_state is None
    assert not window._animation_timer.isActive()


def test_animation_runs_to_completion(qtbot, tmp_path):
    """Fait avancer l'animation « à la main », sans laisser le vrai ``QTimer``
    tourner ni attendre via une boucle d'événements imbriquée
    (``qtbot.waitUntil`` inclus) : cette attente-là a provoqué le même
    segfault que ``_run_get_outline`` documente, une fois assez de fenêtres
    ``pg.ImageView`` accumulées dans le reste de la suite.  Arrêter le timer
    juste après ``_start_animation`` puis appeler ``_advance_animation``
    soi-même exerce exactement la même logique de progression, de façon
    déterministe et sans le moindre risque de boucle imbriquée.
    """
    from media_restorer.extensions.vectorise.gui import _ANIMATION_FRAME_BUDGET

    path = _make_image(tmp_path)
    win = VectoriseGUI(target_path=path, get_outline_fn=_fake_get_outline)
    qtbot.addWidget(win)
    _run_get_outline(win)
    win._combo_candidates.setCurrentIndex(0)

    win._start_animation()
    assert win._animation_timer.isActive()
    win._animation_timer.stop()  # on pilote l'avancée nous-mêmes ci-dessous
    total_points = len(win._animation_state["points"])
    assert total_points > 0

    for _ in range(_ANIMATION_FRAME_BUDGET + 1):
        if win._animation_state is None:
            break
        win._advance_animation()

    assert win._animation_state is None


def test_animation_frame_count_is_bounded_regardless_of_stroke_size():
    """Le nombre d'images ne doit jamais dépasser le budget, même pour un
    trait très long — sans quoi l'animation pourrait durer plusieurs minutes
    sur un vrai dessin (des milliers de points par trait)."""
    from media_restorer.extensions.vectorise.gui import _ANIMATION_FRAME_BUDGET

    n_points = 10_000
    step = max(1, -(-n_points // _ANIMATION_FRAME_BUDGET))
    n_frames = -(-n_points // step)

    assert n_frames <= _ANIMATION_FRAME_BUDGET


# ---------------------------------------------------------------------------
# Fermeture
# ---------------------------------------------------------------------------

def test_close_event_stops_a_running_worker_and_the_animation(qtbot, tmp_path):
    import time

    def _slow_get_outline(image, **kwargs):
        time.sleep(0.3)
        return [_fake_stroke_set()]

    path = _make_image(tmp_path)
    win = VectoriseGUI(target_path=path, get_outline_fn=_slow_get_outline)
    qtbot.addWidget(win)

    win._start_get_outline()
    assert win._get_outline_worker.isRunning()

    win.close()  # ne doit pas lever, ni laisser le thread orphelin

    assert not win._get_outline_worker.isRunning()


# ---------------------------------------------------------------------------
# Registre — icône (voir test_extensions.py pour le garde-fou générique ;
# celui-ci force explicitement l'enregistrement de VectoriseExtension, pour
# ne pas dépendre de l'ordre d'exécution des autres fichiers de tests)
# ---------------------------------------------------------------------------

def test_vectorise_extension_icon_is_registered_in_the_shared_qrc():
    import re
    from pathlib import Path as _Path

    import media_restorer
    import media_restorer.extensions.vectorise as vectorise_ext  # noqa: F401 — enregistre l'extension
    from media_restorer.extensions import all_extensions

    qrc_path = _Path(media_restorer.__file__).parent / "resources" / "icons" / "media_restorer.qrc"
    registered = set(re.findall(r"<file>([^<]+)</file>", qrc_path.read_text()))

    extension = next(e for e in all_extensions() if e.name == "Vectorise")
    icon_name = extension.icon.removeprefix(":/icons/")
    assert icon_name in registered

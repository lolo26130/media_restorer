"""Tests de l'aperçu interactif de l'onglet Double-exposition.

Le point sensible : le slider « fondu » n'agit que dans le mode ``fondu``, et
les paramètres de ``détail`` / ``fusion`` n'agissent que dans leur mode.  Sans
retour visuel ni grisage, l'onglet donne l'impression que les réglages sont
sans effet.  Ces tests verrouillent les deux mécanismes qui l'évitent.
"""
import cv2
import numpy as np
import pytest

from media_restorer.engines import Engine
from media_restorer.engines.dual_engine import (
    MODE_DETAIL,
    MODE_FONDU,
    MODE_FUSION,
    MODES,
)
from media_restorer.gui import (
    _DUAL_LIVE_PARAMS,
    _DUAL_PREVIEW_MAX,
    PhotoRestorationGUI,
)


@pytest.fixture
def window(qtbot):
    win = PhotoRestorationGUI()
    qtbot.addWidget(win)
    return win


@pytest.fixture
def dual_root(window):
    return window._param_roots[Engine.DUAL]


def _pair(h=2400, w=1600):
    """Couple recalé factice, tel que le worker l'émettrait."""
    rng = np.random.default_rng(0)
    img_a = cv2.GaussianBlur(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), (0, 0), 2)
    img_b = np.clip(img_a.astype(int) + 30, 0, 255).astype(np.uint8)
    return img_a, img_b


# ---------------------------------------------------------------------------
# Grisage des paramètres selon le mode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode, enabled, disabled", [
    (MODE_FONDU,  ["alpha"],                          ["detail_radius", "w_contrast"]),
    (MODE_DETAIL, ["detail_base", "detail_radius", "detail_gain"], ["alpha", "w_contrast"]),
    (MODE_FUSION, ["w_contrast", "w_exposure"],       ["alpha", "detail_radius"]),
])
def test_only_the_params_of_the_current_mode_stay_enabled(
    window, dual_root, mode, enabled, disabled
):
    """Changer de mode grise les paramètres que ce mode n'utilise pas."""
    dual_root.child("mode").setValue(mode)

    for name in enabled:
        assert dual_root.child(name).opts.get("enabled", True) is True, name
    for name in disabled:
        assert dual_root.child(name).opts.get("enabled", True) is False, name


def _slider_span(param):
    """Valeurs atteignables par le widget slider de *param*."""
    return next(iter(param.items)).span


def test_greying_the_slider_preserves_its_resolution(window, dual_root):
    """Griser puis réactiver « alpha » ne doit pas dégrader son échelle.

    Régression : ``SliderParameterItem.optsChanged`` reconstruit le span à
    chaque ``setOpts`` en lisant le pas dans le dictionnaire d'options
    *partiel*, avec un défaut de 1.  Un ``setOpts(enabled=…)`` seul ramenait
    le slider 0–1 de 101 crans à deux — il sautait de 0 à 1 sans valeur
    intermédiaire, et ce dès le lancement puisque le mode par défaut
    (« détail ») grise « alpha ».
    """
    alpha = dual_root.child("alpha")
    expected = round((1.0 - 0.0) / 0.01) + 1          # 101 crans pour 0→1 au pas 0,01

    assert len(_slider_span(alpha)) == expected, "cassé dès la construction"

    dual_root.child("mode").setValue(MODE_DETAIL)     # grise alpha
    assert len(_slider_span(alpha)) == expected, "cassé par le grisage"

    dual_root.child("mode").setValue(MODE_FONDU)      # le réactive
    assert len(_slider_span(alpha)) == expected, "cassé par la réactivation"


def test_greyed_slider_still_reaches_intermediate_values(window, dual_root):
    """Le slider expose bien des valeurs intermédiaires, pas seulement 0 et 1."""
    dual_root.child("mode").setValue(MODE_DETAIL)     # grise alpha
    dual_root.child("mode").setValue(MODE_FONDU)      # le réactive
    span = _slider_span(dual_root.child("alpha"))

    assert span[0] == pytest.approx(0.0)
    assert span[-1] == pytest.approx(1.0)
    assert any(0.4 < v < 0.6 for v in span)


def test_greying_preserves_the_current_value(window, dual_root):
    """Passer d'un mode à l'autre ne modifie pas la valeur réglée."""
    dual_root.child("mode").setValue(MODE_FONDU)
    dual_root.child("alpha").setValue(0.25)

    dual_root.child("mode").setValue(MODE_DETAIL)
    dual_root.child("mode").setValue(MODE_FONDU)

    assert dual_root.child("alpha").value() == pytest.approx(0.25)


def test_params_common_to_every_mode_are_never_greyed(window, dual_root):
    """« align », « match_levels » et « 2ᵉ image » servent dans tous les modes."""
    for mode in MODES:
        dual_root.child("mode").setValue(mode)
        for name in ("second_path", "align", "match_levels"):
            assert dual_root.child(name).opts.get("enabled", True) is True


# ---------------------------------------------------------------------------
# Déclenchement de l'aperçu
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", _DUAL_LIVE_PARAMS)
def test_every_live_param_schedules_a_preview(window, dual_root, name):
    """Chaque paramètre de fusion relance le timer anti-rebond de l'aperçu."""
    window._dual_timer.stop()
    param = dual_root.child(name)
    value = param.value()
    if name == "mode":
        new_value = MODE_FONDU if value != MODE_FONDU else MODE_DETAIL
    elif isinstance(value, bool):
        new_value = not value
    elif isinstance(value, str):
        new_value = next(v for v in param.opts["limits"] if v != value)
    else:
        new_value = value + 1

    param.setValue(new_value)

    assert window._dual_timer.isActive()


@pytest.mark.parametrize("name", ["second_path", "align"])
def test_params_invalidating_the_alignment_do_not_trigger_a_preview(
    window, dual_root, name
):
    """« 2ᵉ image » et « recaler » exigent un vrai « Restaurer ».

    L'aperçu rejoue la fusion sur un couple *déjà recalé* : ces deux
    paramètres-là changent le recalage lui-même, qu'un aperçu ne peut pas
    refaire à partir du couple mémorisé.
    """
    window._dual_timer.stop()
    param = dual_root.child(name)

    param.setValue(not param.value() if name == "align" else "/tmp/autre.png")

    assert not window._dual_timer.isActive()


# ---------------------------------------------------------------------------
# Calcul de l'aperçu
# ---------------------------------------------------------------------------

def test_pair_ready_caches_a_downscaled_preview(window):
    """Le couple d'aperçu est réduit une seule fois, au plus à _DUAL_PREVIEW_MAX."""
    window._on_pair_ready(_pair(2400, 1600))

    assert window._dual_pair[0].shape[:2] == (2400, 1600)     # plein cadre conservé
    preview_a, preview_b = window._dual_preview
    assert max(preview_a.shape[:2]) == _DUAL_PREVIEW_MAX
    assert preview_a.shape == preview_b.shape


def test_pair_ready_does_not_upscale_small_images(window):
    """Une image déjà plus petite que la limite n'est pas ré-agrandie."""
    window._on_pair_ready(_pair(300, 200))

    assert window._dual_preview[0].shape[:2] == (300, 200)


@pytest.mark.parametrize("mode", MODES)
def test_preview_refreshes_the_window_in_every_mode(window, dual_root, mode):
    """Tous les modes produisent un aperçu — pas seulement « fondu ».

    Régression : l'aperçu ne réagissait qu'au slider ``alpha`` et donc qu'au
    mode « fondu », les quatre autres modes restant sans retour visuel.
    """
    window._on_pair_ready(_pair(600, 400))
    result_window = window._result_windows[Engine.DUAL]
    result_window.show_image(np.zeros((10, 10, 3), dtype=np.uint8))
    dual_root.child("mode").setValue(mode)

    window._refresh_dual_preview()

    assert "aperçu" in result_window.windowTitle()


def test_preview_does_not_pass_itself_off_as_a_saveable_result(window):
    """L'aperçu étant réduit, il n'alimente pas « Enregistrer »."""
    window._on_pair_ready(_pair(600, 400))
    window._restored = np.zeros((10, 10, 3), dtype=np.uint8)
    window._ui.actionSave.setEnabled(True)

    window._refresh_dual_preview()

    assert window._restored is None
    assert not window._ui.actionSave.isEnabled()


def test_preview_does_not_reopen_a_closed_result_window(window):
    """Si la fenêtre de résultat est fermée, l'aperçu ne la rouvre pas.

    ``update_image`` ne peint que dans une fenêtre déjà visible : la rouvrir
    volerait le focus à l'utilisateur en plein réglage d'un slider.
    """
    window._on_pair_ready(_pair(600, 400))
    result_window = window._result_windows[Engine.DUAL]
    result_window.hide()

    window._refresh_dual_preview()          # ne doit pas lever

    assert not result_window.isVisible()


def test_preview_is_a_no_op_before_any_restore(window):
    """Sans couple mémorisé, l'aperçu ne fait rien plutôt que de planter."""
    assert window._dual_preview is None

    window._refresh_dual_preview()          # ne doit pas lever

    assert window._restored is None


def test_full_restore_clears_the_preview_marker_from_the_title(window):
    """Après un « Restaurer » complet, le titre ne dit plus « aperçu »."""
    window._on_pair_ready(_pair(600, 400))
    window._pending_engine = Engine.DUAL
    window._refresh_dual_preview()
    assert "aperçu" in window._result_windows[Engine.DUAL].windowTitle()

    window._on_restore_done(np.zeros((20, 20, 3), dtype=np.uint8))

    assert window._result_windows[Engine.DUAL].windowTitle() == Engine.DUAL.value
    assert window._ui.actionSave.isEnabled()


def test_preview_uses_the_same_computation_as_production(window, dual_root):
    """L'aperçu rejoue DualExposureEngine.fuse — pas une approximation.

    Garantit qu'un réglage jugé bon sur l'aperçu donnera le même rendu en
    pleine résolution, au facteur d'échelle près.
    """
    from media_restorer.engines import build_engine

    window._on_pair_ready(_pair(600, 400))
    result_window = window._result_windows[Engine.DUAL]
    result_window.show_image(np.zeros((10, 10, 3), dtype=np.uint8))
    dual_root.child("mode").setValue(MODE_DETAIL)
    params = window._read_params(Engine.DUAL)

    window._refresh_dual_preview()
    expected = build_engine(Engine.DUAL, None, params).fuse(*window._dual_preview)

    np.testing.assert_array_equal(
        window._result_windows[Engine.DUAL]._view.image,
        cv2.cvtColor(expected, cv2.COLOR_BGR2RGB),
    )

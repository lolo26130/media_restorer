"""Tests de ResultWindow — conservation du zoom entre deux résultats.

Relancer « Restaurer » recalcule un nouveau résultat pleine résolution, affiché
via ``show_image``.  Par défaut cette méthode recadre systématiquement la vue
(``autoRange``), ce qui est le bon comportement pour un premier affichage mais
ferait perdre le zoom choisi si l'utilisateur relance « Restaurer » pour
comparer un réglage sur un détail précis — cas fréquent pour l'onglet
Double-exposition, où l'on ajuste les paramètres puis reconfirme en pleine
résolution.
"""
import numpy as np
import pytest

from media_restorer.extensions.media_restorer.gui import ResultWindow


@pytest.fixture
def window(qtbot):
    win = ResultWindow("test", preserve_zoom=True)
    qtbot.addWidget(win)
    return win


def _view_range(win):
    """Plage de vue actuelle, aplatie pour comparaison avec pytest.approx."""
    (x0, x1), (y0, y1) = win._view.getView().viewRange()
    return (x0, x1, y0, y1)


def _zoom_in(win):
    """Simule un zoom manuel de l'utilisateur sur un détail."""
    win._view.getView().setRange(xRange=(10, 40), yRange=(10, 40), padding=0)


def test_first_show_always_autoranges(window):
    """Le tout premier affichage cadre la vue — rien à préserver encore."""
    img = np.zeros((100, 200, 3), dtype=np.uint8)

    window.show_image(img)

    x0, x1, _y0, _y1 = _view_range(window)
    # La vue s'est adaptée à l'image (200×100), pas restée à sa plage par défaut.
    assert x1 - x0 == pytest.approx(200, rel=0.2)


def test_preserve_zoom_keeps_view_range_across_repeated_restores(window):
    """Un « Restaurer » relancé ne recadre plus si preserve_zoom est actif."""
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    window.show_image(img)
    _zoom_in(window)
    zoomed_range = _view_range(window)

    window.show_image(np.full((100, 200, 3), 128, dtype=np.uint8))

    assert _view_range(window) == pytest.approx(zoomed_range, rel=1e-6)


def test_preserve_zoom_off_by_default_still_autoranges_every_time(qtbot):
    """Les fenêtres des autres moteurs gardent l'ancien comportement.

    Sans ``preserve_zoom=True`` (le défaut, utilisé par tous les moteurs sauf
    Double-exposition), chaque résultat continue de recadrer la vue.
    """
    win = ResultWindow("autre moteur")
    qtbot.addWidget(win)
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    win.show_image(img)
    _zoom_in(win)
    zoomed_range = _view_range(win)

    win.show_image(np.full((100, 200, 3), 128, dtype=np.uint8))

    assert _view_range(win) != pytest.approx(zoomed_range, rel=1e-6)


def test_preserve_zoom_survives_several_consecutive_restores(window):
    """Le zoom tient sur plusieurs relances successives, pas une seule."""
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    window.show_image(img)
    _zoom_in(window)
    zoomed_range = _view_range(window)

    for value in (64, 128, 200):
        window.show_image(np.full((100, 200, 3), value, dtype=np.uint8))
        assert _view_range(window) == pytest.approx(zoomed_range, rel=1e-6)

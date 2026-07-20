"""Tests du module theme.py — contraste garanti des palettes claire/sombre.

Contexte : les SpinBox de pyqtgraph (onglets de paramètres) étaient peu ou
pas lisibles, faute de toute palette explicite appliquée par l'application.
Ces tests verrouillent le seul invariant qui compte ici : un écart de
luminosité large et sans ambiguïté entre le fond des champs d'édition
(``Base``) et leur texte (``Text``), y compris à l'état désactivé.
"""
import pytest
from PyQt6.QtGui import QColor, QPalette

from media_restorer.theme import (
    THEME_DARK,
    THEME_LIGHT,
    THEME_SYSTEM,
    THEMES,
    _prefers_dark_from_palette,
    apply_theme,
    build_palette,
    system_prefers_dark,
)


def _lightness(palette: QPalette, role: QPalette.ColorRole, group=QPalette.ColorGroup.Active) -> int:
    return palette.color(group, role).lightness()


# ---------------------------------------------------------------------------
# build_palette — le cœur du correctif : contraste Base/Text garanti
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dark", [True, False])
def test_base_and_text_have_strong_contrast(dark):
    """Le fond et le texte des champs d'édition (SpinBox) sont bien séparés.

    C'est exactement ce que lit un QAbstractSpinBox pour se dessiner — un
    écart trop faible ici reproduirait le bug d'origine.
    """
    palette = build_palette(dark)

    base_l = _lightness(palette, QPalette.ColorRole.Base)
    text_l = _lightness(palette, QPalette.ColorRole.Text)

    assert abs(base_l - text_l) > 100


def test_dark_palette_has_light_text_on_dark_base():
    palette = build_palette(dark=True)

    assert _lightness(palette, QPalette.ColorRole.Base) < 60
    assert _lightness(palette, QPalette.ColorRole.Text) > 180


def test_light_palette_has_dark_text_on_light_base():
    palette = build_palette(dark=False)

    assert _lightness(palette, QPalette.ColorRole.Base) > 200
    assert _lightness(palette, QPalette.ColorRole.Text) < 60


@pytest.mark.parametrize("dark", [True, False])
def test_disabled_text_still_contrasts_with_base(dark):
    """Un champ désactivé (paramètre grisé) reste lisible, pas invisible."""
    palette = build_palette(dark)

    base_l = _lightness(palette, QPalette.ColorRole.Base)
    disabled_text_l = _lightness(
        palette, QPalette.ColorRole.Text, QPalette.ColorGroup.Disabled
    )

    assert abs(base_l - disabled_text_l) > 40


@pytest.mark.parametrize("dark", [True, False])
def test_alternate_base_differs_from_base_for_row_banding(dark):
    """Les bandes alternées du ParameterTree restent visibles."""
    palette = build_palette(dark)

    assert palette.color(QPalette.ColorRole.Base) != palette.color(
        QPalette.ColorRole.AlternateBase
    )


# ---------------------------------------------------------------------------
# Détection du thème système
# ---------------------------------------------------------------------------

def test_prefers_dark_from_palette_reads_text_vs_window_lightness():
    """La même heuristique que pyqtgraph.ParameterTree._updatePalette."""
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
    dark_palette.setColor(QPalette.ColorRole.WindowText, QColor(230, 230, 230))
    assert _prefers_dark_from_palette(dark_palette) is True

    light_palette = QPalette()
    light_palette.setColor(QPalette.ColorRole.Window, QColor(240, 240, 240))
    light_palette.setColor(QPalette.ColorRole.WindowText, QColor(20, 20, 20))
    assert _prefers_dark_from_palette(light_palette) is False


def test_system_prefers_dark_returns_a_bool():
    """Ne lève pas et rend un booléen, quelle que soit la plateforme de test."""
    assert isinstance(system_prefers_dark(), bool)


# ---------------------------------------------------------------------------
# apply_theme
# ---------------------------------------------------------------------------

def test_apply_theme_dark_sets_a_high_contrast_dark_palette(qapp):
    apply_theme(qapp, THEME_DARK)

    assert _lightness(qapp.palette(), QPalette.ColorRole.Base) < 60


def test_apply_theme_light_sets_a_high_contrast_light_palette(qapp):
    apply_theme(qapp, THEME_LIGHT)

    assert _lightness(qapp.palette(), QPalette.ColorRole.Base) > 200


def test_apply_theme_system_matches_detected_preference(qapp, monkeypatch):
    """« Système » n'est pas un no-op : il applique la palette détectée."""
    import media_restorer.theme as theme_mod

    monkeypatch.setattr(theme_mod, "system_prefers_dark", lambda: True)
    apply_theme(qapp, THEME_SYSTEM)
    assert _lightness(qapp.palette(), QPalette.ColorRole.Base) < 60

    monkeypatch.setattr(theme_mod, "system_prefers_dark", lambda: False)
    apply_theme(qapp, THEME_SYSTEM)
    assert _lightness(qapp.palette(), QPalette.ColorRole.Base) > 200


def test_apply_theme_forces_fusion_style(qapp):
    """Fusion est le seul style qui respecte intégralement une QPalette explicite."""
    apply_theme(qapp, THEME_DARK)

    assert qapp.style().objectName().lower() == "fusion"


def test_apply_theme_rejects_unknown_skin(qapp):
    with pytest.raises(ValueError, match="Skin inconnu"):
        apply_theme(qapp, "Néon")


def test_themes_tuple_matches_what_apply_theme_accepts(qapp):
    """Verrou anti-dérive : THEMES doit lister exactement ce qu'apply_theme gère."""
    for theme in THEMES:
        apply_theme(qapp, theme)          # ne doit lever pour aucune valeur listée

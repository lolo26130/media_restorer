"""Apparence (skin) de l'application — clair, sombre, ou système.

Origine : les valeurs affichées dans les SpinBox de pyqtgraph
(``ParameterTree``, onglets de paramètres des moteurs) étaient peu ou pas
lisibles.  pyqtgraph n'expose aucune option de configuration pour ça —
``pg.setConfigOption`` ne concerne que le rendu des graphiques (axes,
courbes), pas les widgets d'édition Qt natifs.  Le rendu d'un ``SpinBox``
dépend uniquement de la ``QPalette`` ambiante ; ``ParameterTree`` réagit déjà
aux changements de palette (recalcule ses couleurs de bandes alternées) mais
ne peut évidemment pas corriger une palette elle-même mal contrastée.

Cette application ne fixait jusqu'ici aucune palette explicite : elle héritait
de celle que Qt construit par défaut.  Sur cette machine, PyQt6 est installé
depuis PyPI et embarque son propre Qt6, sans accès aux greffons
d'intégration de thème système (KDE/Breeze) — la palette par défaut qui en
résulte peut être mal contrastée pour les widgets d'édition même sur un
bureau en thème sombre.  D'où ce module : des palettes explicites,
construites et testées pour un contraste garanti, plutôt qu'une dépendance à
une intégration système qui peut être absente ou partielle.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QGuiApplication, QPalette
from PyQt6.QtWidgets import QApplication

THEME_SYSTEM = "Système"
THEME_LIGHT = "Clair"
THEME_DARK = "Sombre"
THEMES = (THEME_SYSTEM, THEME_LIGHT, THEME_DARK)


def _prefers_dark_from_palette(palette: QPalette) -> bool:
    """Heuristique de secours : le texte est plus clair que le fond en thème sombre.

    Même méthode que ``pyqtgraph.parametertree.ParameterTree._updatePalette``
    (dont le commentaire source explique que ``styleHints().colorScheme()``
    a pu retourner l'inverse de la réalité sur certaines plateformes) —
    utilisée uniquement quand cette API ne tranche pas.
    """
    text = palette.color(QPalette.ColorRole.WindowText).lightness()
    window = palette.color(QPalette.ColorRole.Window).lightness()
    return text > window


def system_prefers_dark() -> bool:
    """Meilleure estimation du thème du système d'exploitation.

    Priorité à ``QStyleHints.colorScheme()`` (Qt ≥ 6.5), qui interroge
    directement le système (portail freedesktop sous Linux) plutôt que la
    palette de l'application — donc fiable même si celle-ci n'a pas encore
    été corrigée par ce module.  Repli sur :func:`_prefers_dark_from_palette`
    si la plateforme ne sait pas répondre (``Unknown``).
    """
    scheme = QGuiApplication.styleHints().colorScheme()
    if scheme == Qt.ColorScheme.Dark:
        return True
    if scheme == Qt.ColorScheme.Light:
        return False
    return _prefers_dark_from_palette(QGuiApplication.palette())


def build_palette(dark: bool) -> QPalette:
    """Palette à fort contraste, notamment sur Base/Text — ce que lit un SpinBox.

    Les couleurs sont choisies pour un écart de luminosité large et sans
    ambiguïté entre le fond des champs d'édition (``Base``) et leur texte
    (``Text``), y compris à l'état désactivé (``Disabled``) : c'est
    précisément ce contraste qui manquait dans la palette par défaut.
    """
    palette = QPalette()
    if dark:
        window, base, text = QColor(45, 45, 47), QColor(30, 30, 32), QColor(230, 230, 230)
        alternate, disabled = QColor(53, 53, 56), QColor(115, 115, 118)
    else:
        window, base, text = QColor(239, 239, 239), QColor(255, 255, 255), QColor(20, 20, 20)
        alternate, disabled = QColor(228, 228, 228), QColor(150, 150, 150)
    highlight, highlighted_text = QColor(61, 132, 199), QColor(255, 255, 255)

    for role, color in (
        (QPalette.ColorRole.Window, window),
        (QPalette.ColorRole.WindowText, text),
        (QPalette.ColorRole.Base, base),
        (QPalette.ColorRole.AlternateBase, alternate),
        (QPalette.ColorRole.ToolTipBase, base),
        (QPalette.ColorRole.ToolTipText, text),
        (QPalette.ColorRole.Text, text),
        (QPalette.ColorRole.Button, window),
        (QPalette.ColorRole.ButtonText, text),
        (QPalette.ColorRole.BrightText, QColor(255, 90, 90)),
        (QPalette.ColorRole.Link, highlight),
        (QPalette.ColorRole.Highlight, highlight),
        (QPalette.ColorRole.HighlightedText, highlighted_text),
    ):
        palette.setColor(role, color)

    for role in (
        QPalette.ColorRole.Text,
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(QPalette.ColorGroup.Disabled, role, disabled)

    return palette


def apply_theme(app: QApplication, theme: str) -> None:
    """Applique *theme* (l'un de :data:`THEMES`) à *app*.

    ``THEME_SYSTEM`` détecte le thème du système (voir
    :func:`system_prefers_dark`) et applique la palette correspondante,
    plutôt que de ne rien faire — un no-op laisserait la palette par défaut
    peu contrastée qui a motivé ce module (voir docstring de module).

    ``Fusion`` est imposé comme style : c'est le seul style Qt qui respecte
    intégralement une ``QPalette`` explicite sur toutes les plateformes —
    les styles natifs ignorent parfois certaines couleurs (c'est d'ailleurs
    une cause plausible du problème de lisibilité d'origine).
    """
    if theme not in THEMES:
        raise ValueError(f"Skin inconnu : {theme!r} (attendu : {THEMES})")
    dark = theme == THEME_DARK or (theme == THEME_SYSTEM and system_prefers_dark())
    app.setStyle("Fusion")
    app.setPalette(build_palette(dark))

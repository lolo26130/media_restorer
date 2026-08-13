"""Disposition des fenêtres : taille utile au premier lancement, puis mémorisée.

Deux défauts distincts sont couverts ici, tous deux signalés par l'utilisateur
sous la même phrase (« les aperçus sont trop petits par défaut, […] ne pas avoir
à redimensionner le dock à chaque fois ») :

1. **le premier lancement** — un ``pg.ImageView`` dans un dock naissait à sa
   largeur minimale ; mesuré à 180 px alors que ``ImagePreview.SIZE_HINT`` en
   demande 460.  Un ``sizeHint`` ne suffit pas pour un dock, seul
   ``resizeDocks`` tranche (voir ``gui_widgets.apply_default_dock_width``) ;
2. **les lancements suivants** — la taille choisie par l'utilisateur doit
   primer, donc être enregistrée à la fermeture et relue à l'ouverture.

Le garde-fou le plus important du fichier est
:func:`test_every_dock_has_an_object_name` : un ``QDockWidget`` sans
``objectName`` est ignoré par ``saveState``/``restoreState``, Qt se contentant
d'un avertissement sur la sortie d'erreur.  La mémorisation semblerait alors
fonctionner sans rien mémoriser.  Même forme de panne silencieuse que l'icône
manquante d'une extension — et même parade : l'énumérer plutôt que l'espérer.

⚠ ``QT_QPA_PLATFORM=offscreen`` n'honore pas ``restoreGeometry`` (la taille de
la *fenêtre* ne bouge pas), mais applique bien ``restoreState`` (la répartition
entre docks).  Les tests fixent donc explicitement la taille de la fenêtre et
n'observent que les docks.
"""
from __future__ import annotations

import gc

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QDockWidget

from media_restorer.gui_widgets import (
    PREVIEW_DOCK_WIDTH,
    ImagePreview,
    restore_layout,
)

TAILLE_FENETRE = (1600, 900)


@pytest.fixture(autouse=True)
def _flush_qt_deletions():
    """Purge les ``deleteLater`` entre les tests.

    Chaque fenêtre ouverte ici crée des ``pg.ImageView`` ; les accumuler dans un
    même processus rapproche du seuil de segfault documenté dans le
    ``CLAUDE.md``.
    """
    yield
    QApplication.processEvents()
    gc.collect()


def _fenetres(qtbot):
    """Les quatre fenêtres à docks du projet, prêtes et affichées."""
    from media_restorer.extensions.doublons.gui import DoublonsGUI
    from media_restorer.extensions.pre_classement.gui import PreClassementGUI
    from media_restorer.extensions.signatures.gui import SignaturesGUI
    from media_restorer.gui_root import ImageTreatmentWindow

    for classe in (ImageTreatmentWindow, PreClassementGUI, DoublonsGUI, SignaturesGUI):
        fenetre = classe()
        qtbot.addWidget(fenetre)
        fenetre.resize(*TAILLE_FENETRE)
        fenetre.show()
        QApplication.processEvents()
        yield fenetre


# ---------------------------------------------------------------------------
# Garde-fou : sans objectName, la mémorisation ne mémorise rien
# ---------------------------------------------------------------------------

def test_every_dock_has_an_object_name(qtbot):
    """Un dock anonyme est ignoré par saveState — panne silencieuse.

    Qt écrit « 'objectName' not set for QDockWidget » sur la sortie d'erreur
    puis continue : sans ce test, un dock ajouté plus tard sans nom d'objet
    ferait perdre la disposition de TOUTE la fenêtre sans que rien n'échoue.
    """
    for fenetre in _fenetres(qtbot):
        for dock in fenetre.findChildren(QDockWidget):
            assert dock.objectName(), (
                f"dock « {dock.windowTitle()} » de {type(fenetre).__name__} "
                "sans objectName — saveState l'ignorera silencieusement"
            )


def test_object_names_are_unique_within_a_window(qtbot):
    """Deux docks homonymes se disputeraient le même état enregistré."""
    for fenetre in _fenetres(qtbot):
        noms = [d.objectName() for d in fenetre.findChildren(QDockWidget)]
        assert len(noms) == len(set(noms)), f"{type(fenetre).__name__} : {noms}"


# ---------------------------------------------------------------------------
# Premier lancement : une taille d'emblée utilisable
# ---------------------------------------------------------------------------

def test_the_preview_advertises_a_usable_size():
    """La taille conseillée doit dépasser franchement le plancher."""
    apercu = ImagePreview()
    assert apercu.sizeHint().width() == PREVIEW_DOCK_WIDTH
    assert apercu.sizeHint().width() > apercu.minimumSizeHint().width()


def test_a_fresh_preview_dock_is_not_born_at_its_minimum(qtbot):
    """Le défaut signalé : 180 px (le minimum) au lieu d'une largeur utile.

    Aucun état n'est enregistré (QSettings est redirigé vers un répertoire
    jetable par ``conftest``), on est donc bien dans le cas « premier
    lancement ».  La fenêtre est élargie ici pour que l'assertion ne dépende
    pas de la taille d'écran de la machine de test — l'élargissement
    automatique, lui, est vérifié par le test suivant.
    """
    from media_restorer.extensions.pre_classement.gui import PreClassementGUI
    from media_restorer.gui_root import ImageTreatmentWindow

    for classe in (ImageTreatmentWindow, PreClassementGUI):
        fenetre = classe()
        qtbot.addWidget(fenetre)
        fenetre.resize(*TAILLE_FENETRE)
        fenetre.show()
        QApplication.processEvents()

        dock = fenetre._preview_dock
        assert dock.width() > dock.widget().minimumSizeHint().width(), (
            f"{classe.__name__} : l'aperçu naît à son minimum"
        )
        assert dock.width() >= PREVIEW_DOCK_WIDTH


def test_an_unforced_first_launch_widens_the_root_window(qtbot):
    """Sans redimensionnement manuel, la fenêtre doit se faire de la place.

    ``root.ui`` fixe 769 px, où l'aperçu ne recevait que son minimum :
    ``resizeDocks`` *répartit* la largeur, il n'en crée pas.  La disposition par
    défaut élargit donc la fenêtre au premier affichage — dans la limite de
    l'écran, quel qu'il soit sur la machine de test.
    """
    from media_restorer.gui_root import ImageTreatmentWindow

    fenetre = ImageTreatmentWindow()
    qtbot.addWidget(fenetre)
    largeur_ui = fenetre.width()          # avant tout affichage : celle du .ui
    fenetre.show()
    QApplication.processEvents()

    ecran = QApplication.primaryScreen().availableGeometry().width()
    attendue = min(largeur_ui + PREVIEW_DOCK_WIDTH, ecran)
    assert fenetre.width() >= min(attendue, ecran)
    assert fenetre.width() > largeur_ui or largeur_ui >= ecran


def test_the_default_layout_is_applied_only_once(qtbot):
    """Masquer puis réafficher ne doit pas réécraser un réglage manuel.

    ``showEvent`` se déclenche à chaque réapparition de la fenêtre ; sans le
    drapeau, la largeur par défaut reviendrait balayer le choix de
    l'utilisateur au moindre aller-retour.
    """
    from media_restorer.gui_root import ImageTreatmentWindow

    fenetre = ImageTreatmentWindow()
    qtbot.addWidget(fenetre)
    fenetre.resize(*TAILLE_FENETRE)
    fenetre.show()
    QApplication.processEvents()

    fenetre.resizeDocks([fenetre._preview_dock], [220], Qt.Orientation.Horizontal)
    QApplication.processEvents()
    fenetre.hide()
    fenetre.show()
    QApplication.processEvents()

    assert fenetre._preview_dock.width() == 220


# ---------------------------------------------------------------------------
# Lancements suivants : le réglage de l'utilisateur prime
# ---------------------------------------------------------------------------

def test_nothing_to_restore_on_a_first_launch(qtbot):
    """``restore_layout`` doit *dire* qu'elle n'a rien trouvé.

    C'est ce booléen qui décide d'appliquer — ou non — la largeur par défaut ;
    l'appliquer à tort écraserait le réglage de l'utilisateur.
    """
    from media_restorer.gui_root import ImageTreatmentWindow

    fenetre = ImageTreatmentWindow()
    qtbot.addWidget(fenetre)
    assert restore_layout(fenetre, "prefixe_jamais_ecrit") is False


def test_a_resized_dock_comes_back_the_same_size(qtbot):
    """Le cœur de la demande : ne plus redimensionner à chaque lancement."""
    from media_restorer.gui_root import ImageTreatmentWindow

    premiere = ImageTreatmentWindow()
    qtbot.addWidget(premiere)
    premiere.resize(*TAILLE_FENETRE)
    premiere.show()
    QApplication.processEvents()
    premiere.resizeDocks(
        [premiere._preview_dock], [300], Qt.Orientation.Horizontal
    )
    QApplication.processEvents()
    assert premiere._preview_dock.width() == 300
    premiere.close()          # closeEvent → save_layout

    seconde = ImageTreatmentWindow()
    qtbot.addWidget(seconde)
    seconde.resize(*TAILLE_FENETRE)
    seconde.show()
    QApplication.processEvents()

    assert seconde._preview_dock.width() == 300, (
        "la largeur réglée par l'utilisateur doit primer sur la valeur par défaut"
    )


def test_closing_the_root_window_records_the_layout(qtbot):
    """La fermeture est le seul chemin de sortie : c'est là qu'on enregistre."""
    from media_restorer.app_settings import app_settings
    from media_restorer.gui_root import ImageTreatmentWindow

    assert not [c for c in app_settings().allKeys() if c.startswith("root/")]

    fenetre = ImageTreatmentWindow()
    qtbot.addWidget(fenetre)
    fenetre.show()
    QApplication.processEvents()
    fenetre.close()

    cles = {c for c in app_settings().allKeys() if c.startswith("root/")}
    assert cles == {"root/geometry", "root/dock_state"}


def test_each_window_keeps_its_own_layout(qtbot):
    """Préfixes distincts : régler le pré-classement ne touche pas la racine."""
    from media_restorer.app_settings import app_settings
    from media_restorer.extensions.pre_classement.gui import PreClassementGUI

    fenetre = PreClassementGUI()
    qtbot.addWidget(fenetre)
    fenetre.show()
    QApplication.processEvents()
    fenetre.close()

    cles = set(app_settings().allKeys())
    assert "pre_classement/dock_state" in cles
    assert "root/dock_state" not in cles


def test_the_shot_panel_remembers_its_splitter(qtbot):
    """Le séparateur vit DANS un dock : ``saveState`` ne le couvre pas.

    Il se mémorise donc lui-même, sans quoi l'aperçu des correspondances RAW
    reprendrait sa part par défaut à chaque ouverture.
    """
    from media_restorer.gui_shots import ShotInventoryPanel

    premier = ShotInventoryPanel()
    qtbot.addWidget(premier)
    premier.resize(1000, 500)
    premier.show()
    QApplication.processEvents()
    premier._splitter.setSizes([250, 750])
    premier._splitter.splitterMoved.emit(250, 1)   # comme le ferait la poignée

    second = ShotInventoryPanel()
    qtbot.addWidget(second)
    second.resize(1000, 500)
    second.show()
    QApplication.processEvents()

    gauche, droite = second._splitter.sizes()
    assert droite > gauche, "la part donnée à l'aperçu doit avoir été retenue"


def test_saving_a_layout_never_touches_the_real_configuration():
    """Verrou d'hygiène : les tests écrivent dans un fichier jetable.

    ``conftest`` redirige ``QSettings`` ; le vérifier ici évite qu'un test de
    disposition ne finisse par régler la vraie fenêtre de l'utilisateur — ce
    qui est arrivé pendant l'écriture de cette fonctionnalité.
    """
    from media_restorer.app_settings import app_settings

    assert "/.config/media_restorer/" not in app_settings().fileName()

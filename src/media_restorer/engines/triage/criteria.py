"""Catalogue déclaratif des critères de tri.

Chaque critère décrit **un axe** de classement : son libellé, la branche
d'étiquettes qu'il alimente, ses classes possibles, et la façon d'extraire sa
valeur d'un :class:`~media_restorer.engines.triage.signals.ImageSignals`.

Pourquoi un catalogue plutôt que du code
----------------------------------------
C'est ce qui rend l'interface générique : le panneau de paramètres de
l'extension se construit **depuis** :data:`CRITERIA` (une case à cocher par
critère), et le tableau de résultats en tire ses colonnes.  Ajouter un critère
demain — y compris servi par un modèle, via :attr:`Criterion.method` — ne
demandera pas de toucher une ligne de code Qt.  Même principe que
:data:`~media_restorer.engines.ENGINE_PARAMS`, qui pilote déjà le
``ParameterTree`` des moteurs de restauration.

Contrainte de nommage des classes
---------------------------------
Deux règles, héritées du travail sur les étiquettes de repères, et que
:func:`check_labels` vérifie :

1. **Jamais de ``/`` dans un libellé de classe.**  C'est le séparateur de
   ``XMP-digiKam:TagsList`` : un libellé qui en contient casserait la hiérarchie
   et créerait un niveau fantôme dans l'arbre de DigiKam.
2. **Un libellé doit rester compréhensible seul.**  ``XMP-dc:Subject`` et
   ``IPTC:Keywords`` sont plats : DigiKam n'y recopie que la feuille.  D'où
   « Encre moyenne » plutôt que « Moyen », qui ne voudrait rien dire hors de sa
   branche.

C'est cette seconde règle qui a fait **scinder en deux axes** l'ancien libellé
combiné ``"trait noir / papier neutre"`` de
:data:`~media_restorer.engines.triage.signals.SUPPORTS` — et le résultat colle
mieux à l'analyse : la couleur de l'encre est un axe de **contenu**, la teinte
du papier un axe de **condition de numérisation**.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from media_restorer.engines.triage.signals import (
    INK_DENSITIES,
    ORIENTATIONS,
    RESOLUTIONS,
    ImageSignals,
)

# Méthode de calcul d'un critère.  Une seule est implémentée ; les autres sont
# le crochet prévu pour les niveaux supérieurs (voir le rapport
# ``docs/rapport-tri-grossier-corpus.tex``).
METHOD_SIGNALS = "signals"
METHODS = {METHOD_SIGNALS: "Signaux gratuits (sans modèle)"}


@dataclass(frozen=True)
class Criterion:
    """Un axe de classement, et de quoi le calculer comme l'afficher.

    Attributs
    ---------
    key : str
        Identifiant stable, utilisé comme clé de paramètre et de colonne.
        Ne jamais le renommer : il est persisté dans les ``QSettings``.
    title : str
        Libellé affiché dans le panneau de paramètres et en en-tête de colonne.
    branch : str
        Branche d'étiquettes alimentée : ``media_restorer/Tri/<branch>/<classe>``.
    classes : tuple[str, ...]
        Toutes les valeurs possibles, dans l'ordre d'affichage.
    extract : Callable
        Extrait la classe d'un :class:`ImageSignals`.
    method : str
        Comment le critère est calculé.  ``"signals"`` n'exige aucun modèle.
    """

    key: str
    title: str
    branch: str
    classes: tuple[str, ...]
    extract: Callable[[ImageSignals], str]
    method: str = METHOD_SIGNALS


# Libellés auto-suffisants (voir la docstring de module) : ils doivent rester
# lisibles seuls, une fois recopiés dans les champs plats de DigiKam.
_INK_DENSITY_LABELS = ("Encre très claire", "Encre claire", "Encre moyenne", "Encre dense")
_DENSITY_LABEL_BY_CLASS = dict(zip(INK_DENSITIES, _INK_DENSITY_LABELS))

_ORIENTATION_LABELS = ("Portrait", "Paysage", "Carré")
_ORIENTATION_LABEL_BY_CLASS = dict(zip(ORIENTATIONS, _ORIENTATION_LABELS))


CRITERIA: tuple[Criterion, ...] = (
    Criterion(
        key="orientation",
        title="Orientation",
        branch="Orientation",
        classes=_ORIENTATION_LABELS,
        extract=lambda s: _ORIENTATION_LABEL_BY_CLASS[s.orientation],
    ),
    Criterion(
        key="ink_density",
        title="Densité d'encre",
        branch="Densité d'encre",
        classes=_INK_DENSITY_LABELS,
        extract=lambda s: _DENSITY_LABEL_BY_CLASS[s.ink_density],
    ),
    Criterion(
        key="ink_colour",
        title="Couleur d'encre",
        branch="Couleur d'encre",
        classes=("Trait noir", "Encre colorée"),
        extract=lambda s: "Encre colorée" if s.coloured_ink else "Trait noir",
    ),
    Criterion(
        key="paper",
        title="Teinte du papier",
        branch="Papier",
        classes=("Papier neutre", "Papier jauni"),
        extract=lambda s: "Papier jauni" if s.tinted_paper else "Papier neutre",
    ),
    Criterion(
        key="resolution",
        title="Résolution",
        branch="Résolution",
        classes=RESOLUTIONS,
        extract=lambda s: s.resolution_class,
    ),
)

CRITERIA_BY_KEY = {c.key: c for c in CRITERIA}


def criteria_for(method: str = METHOD_SIGNALS) -> tuple[Criterion, ...]:
    """Critères calculables par *method*.

    Permet à l'interface de n'offrir que ce que la méthode choisie sait
    produire, sans qu'elle ait à connaître les méthodes elle-même.
    """
    return tuple(c for c in CRITERIA if c.method == method)


def select(keys: Sequence[str]) -> tuple[Criterion, ...]:
    """Critères désignés par *keys*, dans l'ordre du catalogue.

    L'ordre du catalogue prime sur celui de *keys* : les colonnes du tableau et
    les étiquettes écrites restent ainsi dans un ordre stable, quel que soit
    l'ordre dans lequel l'utilisateur a coché les cases.  Une clé inconnue est
    ignorée — une configuration persistée peut mentionner un critère retiré
    depuis.
    """
    wanted = set(keys)
    return tuple(c for c in CRITERIA if c.key in wanted)


def check_labels() -> list[str]:
    """Anomalies de nommage dans le catalogue — vide si tout va bien.

    Vérifie les deux règles de la docstring de module.  Appelée par la suite de
    tests plutôt qu'à l'exécution : un libellé fautif doit faire échouer la CI,
    pas surgir au moment d'écrire dans les fichiers de l'utilisateur.
    """
    problemes: list[str] = []
    for criterion in CRITERIA:
        if "/" in criterion.branch:
            problemes.append(f"branche « {criterion.branch} » : contient « / »")
        for label in criterion.classes:
            if "/" in label:
                problemes.append(f"classe « {label} » : contient « / »")
            if not label.strip():
                problemes.append(f"critère {criterion.key} : classe vide")
    return problemes

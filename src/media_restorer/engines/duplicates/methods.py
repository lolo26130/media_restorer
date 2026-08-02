"""Catalogue déclaratif des méthodes de détection de doublons.

Chaque méthode décrit **une brique** de la chaîne : à quel étage elle
intervient, ce qu'elle coûte, et si elle est active par défaut.  L'interface
construit ses cases à cocher **depuis** :data:`METHODS` — le code Qt ne nomme
aucune méthode, si bien qu'en ajouter une plus tard (les descripteurs appris du
lot suivant) ne touchera pas une ligne d'interface.

Même patron que :data:`~media_restorer.engines.triage.criteria.CRITERIA`, qui a
fait ses preuves, et que :data:`~media_restorer.engines.ENGINE_PARAMS`.

Les trois étages
----------------
La chaîne est ordonnée du gratuit au coûteux ; c'est ce qui la rend praticable
(voir ``docs/rapport-doublons-dessins.tex``) :

``PREFILTER``
    Élague des paires sans rien calculer de neuf — réutilise les mesures déjà
    en cache dans :mod:`media_restorer.engines.triage`.
``CANDIDATE``
    Propose, pour chaque image, ses plus proches voisines selon un descripteur
    **global** (donc rapide).  Ne conclut rien.
``VERIFY``
    Vérifie géométriquement une paire candidate et produit le dictionnaire de
    mérite.  Coûteux : c'est pourquoi il ne voit qu'une fraction des paires.

Une méthode désactivée disparaît de la chaîne sans effet de bord : les étages
sont indépendants, et chaque étage sait fonctionner avec un sous-ensemble de ses
méthodes — voire aucune, auquel cas il laisse passer tout ce qu'il reçoit.
"""
from __future__ import annotations

from dataclasses import dataclass

# --- Étages ---------------------------------------------------------------
STAGE_PREFILTER = "prefilter"
STAGE_CANDIDATE = "candidate"
STAGE_VERIFY = "verify"

STAGE_TITLES = {
    STAGE_PREFILTER: "Pré-filtrage (gratuit)",
    STAGE_CANDIDATE: "Recherche de candidats",
    STAGE_VERIFY: "Vérification géométrique",
}
STAGE_ORDER = (STAGE_PREFILTER, STAGE_CANDIDATE, STAGE_VERIFY)


@dataclass(frozen=True)
class Method:
    """Une brique sélectionnable de la chaîne de détection.

    Attributs
    ---------
    key : str
        Identifiant stable — persisté dans les ``QSettings``, à ne jamais
        renommer.
    title : str
        Libellé affiché dans le panneau de paramètres.
    stage : str
        Étage auquel la méthode intervient (voir la docstring de module).
    description : str
        Une phrase, affichée en info-bulle : ce que la méthode apporte, et ce
        qu'elle ne sait pas faire.
    default : bool
        Cochée au premier lancement.
    cost : str
        Ordre de grandeur indicatif, affiché à l'utilisateur pour qu'il sache
        ce qu'il engage en cochant.
    """

    key: str
    title: str
    stage: str
    description: str
    default: bool = True
    cost: str = ""


METHODS: tuple[Method, ...] = (
    Method(
        key="triage_prefilter",
        title="Écarter les images trop dissemblables",
        stage=STAGE_PREFILTER,
        description=(
            "Utilise les mesures déjà calculées par le pré-classement "
            "(orientation, densité d'encre, teinte) pour éliminer d'emblée les "
            "paires sans espoir. Ne relit aucun fichier."
        ),
        cost="gratuit",
    ),
    Method(
        key="fourier_mellin",
        title="Descripteur Fourier-Mellin",
        stage=STAGE_CANDIDATE,
        description=(
            "Descripteur global invariant par rotation, homothétie et "
            "translation. Rapide, mais global : il propose mal les inclusions "
            "partielles (un dessin dans une page)."
        ),
        cost="~1 ms/image",
    ),
    Method(
        key="ink_profile",
        title="Profil d'encre (histogramme d'orientations)",
        stage=STAGE_CANDIDATE,
        description=(
            "Descripteur global fondé sur la distribution des orientations du "
            "trait, peu sensible à l'épaisseur. Complète Fourier-Mellin sur les "
            "dessins retracés ou ré-encrés."
        ),
        cost="~2 ms/image",
    ),
    Method(
        key="orb_magsac",
        title="Traits locaux ORB + MAGSAC",
        stage=STAGE_VERIFY,
        description=(
            "Vérification géométrique de référence. Mesuré 8× plus rapide que "
            "SIFT et aussi invariant sur ce corpus. Traite nativement "
            "l'inclusion partielle."
        ),
        cost="~15 ms/paire",
    ),
    Method(
        key="sift_magsac",
        title="Traits locaux SIFT + MAGSAC (plus lent, plus fin)",
        stage=STAGE_VERIFY,
        description=(
            "Même principe qu'ORB, environ 8× plus coûteux. À réserver aux cas "
            "ambigus : mesuré meilleur qu'ORB sur les forts agrandissements, "
            "moins bon sur les changements d'épaisseur de trait."
        ),
        default=False,
        cost="~150 ms/paire",
    ),
)

METHODS_BY_KEY = {m.key: m for m in METHODS}


def methods_for(stage: str, keys: tuple[str, ...] | None = None) -> tuple[Method, ...]:
    """Méthodes actives d'un étage, dans l'ordre du catalogue.

    *keys* restreint aux méthodes sélectionnées ; ``None`` renvoie celles
    activées par défaut.  L'ordre du catalogue prime toujours sur celui de
    *keys*, afin que le résultat ne dépende pas de l'ordre dans lequel
    l'utilisateur a coché ses cases.
    """
    if keys is None:
        return tuple(m for m in METHODS if m.stage == stage and m.default)
    voulu = set(keys)
    return tuple(m for m in METHODS if m.stage == stage and m.key in voulu)


def default_keys() -> tuple[str, ...]:
    """Clés des méthodes cochées au premier lancement."""
    return tuple(m.key for m in METHODS if m.default)


def check_catalogue() -> list[str]:
    """Anomalies du catalogue — liste vide si tout va bien.

    Vérifiée par les tests plutôt qu'à l'exécution : un catalogue incohérent
    doit faire échouer la suite, pas surprendre l'utilisateur en cours de
    campagne.
    """
    problemes: list[str] = []
    vus: set[str] = set()
    for m in METHODS:
        if m.stage not in STAGE_ORDER:
            problemes.append(f"{m.key} : étage inconnu « {m.stage} »")
        if m.key in vus:
            problemes.append(f"{m.key} : clé en double")
        vus.add(m.key)
        if not m.description.strip():
            problemes.append(f"{m.key} : description vide")
    # Sans méthode de vérification, aucun dictionnaire de mérite ne peut être
    # produit : la chaîne perdrait sa raison d'être.
    if not any(m.stage == STAGE_VERIFY and m.default for m in METHODS):
        problemes.append("aucune méthode de vérification active par défaut")
    return problemes

"""Écriture et lecture d'étiquettes hiérarchiques au format DigiKam, sans Qt.

Moteur partagé par tout ce qui veut poser des étiquettes sur une image :
:mod:`media_restorer.landmarks` (repères faciaux) et
:mod:`media_restorer.engines.triage.tags` (pré-classement).  Extrait de
``landmarks.py``, où il avait été écrit d'abord — la logique de **fusion non
destructive** est trop subtile pour être dupliquée d'un appelant à l'autre.

Les six champs
--------------
DigiKam synchronise le même jeu d'étiquettes entre six champs, chacun avec sa
convention.  La liste et les séparateurs sont ceux qu'il déclare lui-même dans
``~/.config/digikamrc``, section ``[DMetadata Settings][readTagsNamespaces]``
(clés ``separator`` et ``tagPaths``), et qu'on retrouve tels quels sur une image
réellement étiquetée par lui :

============================== ========== ===========================================
Champ                          Séparateur Contenu
============================== ========== ===========================================
``XMP-digiKam:TagsList``       ``/``      chemin complet — **source de vérité** en lecture
``XMP-lr:HierarchicalSubject`` ``|``      même chemin, convention Lightroom
``XMP-microsoft:LastKeywordXMP`` ``/``    même chemin, convention Windows
``XMP-mediapro:CatalogSets``   ``|``      même chemin, convention MediaPro
``XMP-dc:Subject``             (aucun)    **feuille** seule (``tagPaths=0``)
``IPTC:Keywords``              (aucun)    feuille seule (``tagPaths=0``, hérité)
============================== ========== ===========================================

S'y ajoute ``IPTC:CodedCharacterSet=UTF8`` : IPTC n'est pas UTF-8 par défaut et
nos libellés sont accentués.  DigiKam écrit la même déclaration.

Le septième champ que DigiKam écrit, ``XMP-acdsee:Categories``, est du XML
imbriqué : seul format de la liste à ne pas être une simple liste de chemins, il
est laissé de côté — aucun des champs relus par DigiKam n'en dépend.

Fusion non destructive, et le piège d'appartenance
--------------------------------------------------
Ces champs sont **curés par l'utilisateur**.  Une écriture nue les écraserait :
``exiftool -TagsList=…`` **remplace la liste entière**, il ne la complète pas
(vérifié : un second appel efface les valeurs du premier).  :func:`write_tags`
procède donc par lecture → fusion → écriture, et reconstruit chaque champ en
entier.

D'où le paramètre :data:`Owns` — un prédicat qui répond « ce chemin
d'étiquette est-il à moi ? ».  Ce qui lui appartient est remplacé, tout le reste
est relu puis réécrit tel quel.

.. warning::

   **Un prédicat trop large détruit les étiquettes d'un autre appelant, en
   silence.**  Si les repères revendiquaient tout ``media_restorer/…``, la
   première écriture de pré-classement effacerait les repères, et
   réciproquement — sans erreur ni exception, juste des étiquettes disparues.
   Chaque appelant doit donc revendiquer **ses branches et elles seules**
   (voir :func:`branch_owner`).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

# Lance ``exiftool`` avec *args* (sans l'exécutable) et renvoie sa sortie
# standard.  Injectable pour les tests — voir :func:`default_runner`.
ExiftoolRunner = Callable[[list[str]], str]

# « Ce chemin d'étiquette, découpé en composants, m'appartient-il ? »
Owns = Callable[[Sequence[str]], bool]

# Racine commune à toutes les étiquettes posées par l'application : regroupe le
# tout sous une seule branche repliable dans le gestionnaire de DigiKam.
ROOT = "media_restorer"

TAG_HIERARCHICAL = {               # champ exiftool → séparateur de chemin
    "XMP-digiKam:TagsList": "/",
    "XMP-lr:HierarchicalSubject": "|",
    "XMP-microsoft:LastKeywordXMP": "/",
    "XMP-mediapro:CatalogSets": "|",
}
TAG_FLAT = ("XMP-dc:Subject", "IPTC:Keywords")
TAG_CHARSET = "IPTC:CodedCharacterSet"

# Champ le plus fidèle, propre à DigiKam : celui qu'on lit en priorité.
PRIMARY_TAG = "XMP-digiKam:TagsList"


def default_runner(args: list[str]) -> str:
    """Lance le vrai binaire ``exiftool`` et renvoie sa sortie standard.

    Runner de production (les tests en injectent un faux).  Lève un
    :class:`RuntimeError` explicite si ``exiftool`` n'est pas installé, plutôt
    que le ``FileNotFoundError`` brut de :func:`subprocess.run`.
    """
    exe = shutil.which("exiftool")
    if exe is None:
        raise RuntimeError(
            "exiftool introuvable — installez-le (paquet « libimage-exiftool-perl » "
            "sous Debian/Ubuntu) pour lire/écrire les étiquettes dans les métadonnées."
        )
    result = subprocess.run(
        [exe, *args], capture_output=True, text=True, check=True, timeout=60
    )
    return result.stdout


def branch_owner(*branches: str, root: str = ROOT) -> Owns:
    """Prédicat revendiquant ``<root>/<branche>/…`` pour les *branches* données.

    C'est la façon recommandée de construire un :data:`Owns` : revendiquer des
    branches nommées, et jamais la racine entière — voir l'avertissement en tête
    de module.
    """
    owned = frozenset(branches)

    def owns(parts: Sequence[str]) -> bool:
        return len(parts) > 1 and parts[0] == root and parts[1] in owned

    return owns


# ---------------------------------------------------------------------------
# Syntaxe « structure » d'exiftool (XMP-mwg-rs:RegionInfo et consorts)
# ---------------------------------------------------------------------------
# exiftool accepte une structure imbriquée sous la forme
# ``{Clé=valeur,Liste=[{...},{...}]}``.  Le caractère ``|`` y sert d'échappement
# pour les caractères de structure — indispensable dès qu'une valeur contient
# une virgule ou un signe égal.


def escape_struct(value: str) -> str:
    """Échappe les caractères de structure d'exiftool (``|`` en préfixe)."""
    out = value.replace("|", "||")
    for char in "{}[],=":
        out = out.replace(char, f"|{char}")
    return out


def to_struct(value: Any) -> str:
    """Sérialise *value* (dict / list / scalaire) en syntaxe structure exiftool."""
    if isinstance(value, dict):
        return "{" + ",".join(f"{k}={to_struct(v)}" for k, v in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(to_struct(v) for v in value) + "]"
    return escape_struct(str(value))


def as_list(value: Any) -> list:
    """Normalise une valeur exiftool : un champ à une seule entrée n'est pas une liste."""
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


# ---------------------------------------------------------------------------
# Lecture
# ---------------------------------------------------------------------------

def read_raw(
    path: Path | str,
    runner: ExiftoolRunner,
    *,
    extra_tags: Iterable[str] = (),
) -> dict:
    """Lit en un seul appel les six champs d'étiquettes, plus *extra_tags*.

    ``-struct`` conserve la structure imbriquée des champs qui en ont (par
    exemple ``RegionInfo``), indispensable pour réécrire à l'identique ce qui ne
    nous appartient pas.

    Ne lève jamais : un fichier illisible ou sans métadonnées se comporte comme
    un fichier vide, et l'écriture repart alors d'une base vierge.
    """
    args = [
        "-j", "-struct",
        *(f"-{tag}" for tag in TAG_HIERARCHICAL),
        *(f"-{tag}" for tag in TAG_FLAT),
        *(f"-{tag}" for tag in extra_tags),
        str(path),
    ]
    try:
        records = json.loads(runner(args))
    except Exception:
        return {}
    if not isinstance(records, list) or not records or not isinstance(records[0], dict):
        return {}
    return records[0]


def split_paths(raw: dict, tag: str, separator: str) -> list[list[str]]:
    """Chemins lus dans *tag*, découpés selon **son propre** séparateur.

    Chaque champ est découpé avec sa convention plutôt que déduit de
    ``TagsList`` : une image étiquetée par Lightroom seul ne porte que
    ``HierarchicalSubject``, et déduire ses entrées d'un ``TagsList`` absent
    reviendrait à les effacer.
    """
    entries = as_list(raw.get(tag.split(":")[-1]))
    return [str(entry).split(separator) for entry in entries]


def read_tag_paths(raw: dict, owns: Owns) -> list[list[str]]:
    """Chemins nous appartenant, lus dans le premier champ qui en porte.

    Ordre de préférence : celui de :data:`TAG_HIERARCHICAL`, donc
    ``TagsList`` (le champ propre à DigiKam) d'abord.  Un seul champ suffit :
    les quatre sont des copies du même jeu.
    """
    for tag, separator in TAG_HIERARCHICAL.items():
        paths = [parts for parts in split_paths(raw, tag, separator) if owns(parts)]
        if paths:
            return paths
    return []


def own_leaves(raw: dict, owns: Owns) -> set[str]:
    """Feuilles nous appartenant, **tous** champs hiérarchiques confondus.

    Sert à nettoyer les champs plats (``dc:Subject``, ``IPTC:Keywords``), où
    rien ne distingue une de nos feuilles d'une étiquette de l'utilisateur.
    L'union — et non le premier champ qui répond, comme pour la lecture des
    chemins — est nécessaire au cas où les champs seraient désynchronisés : une
    feuille périmée qu'un seul d'entre eux mentionne encore doit quand même être
    reconnue comme nôtre, sinon elle survivrait indéfiniment comme « étrangère »
    et le champ plat accumulerait des entrées orphelines.
    """
    return {
        parts[-1]
        for tag, separator in TAG_HIERARCHICAL.items()
        for parts in split_paths(raw, tag, separator)
        if owns(parts)
    }


# ---------------------------------------------------------------------------
# Écriture
# ---------------------------------------------------------------------------

def tag_args(raw: dict, new_paths: Sequence[Sequence[str]], *, owns: Owns) -> list[str]:
    """Arguments d'écriture des six champs, nos entrées fusionnées aux autres.

    Chaque champ est réécrit **en entier** (une affectation ``-TAG=`` remplace
    la liste, elle ne la complète pas), d'où la reconstruction complète :
    entrées étrangères conservées dans leur ordre, puis les nôtres.
    """
    owned_leaves = own_leaves(raw, owns)
    new_leaves = [parts[-1] for parts in new_paths]
    args: list[str] = []

    for tag, separator in TAG_HIERARCHICAL.items():
        foreign = [
            separator.join(parts)
            for parts in split_paths(raw, tag, separator)
            if not owns(parts)
        ]
        args += [f"-{tag}=" + value for value in foreign]
        args += [f"-{tag}=" + separator.join(parts) for parts in new_paths]

    for tag in TAG_FLAT:
        foreign = [
            str(value) for value in as_list(raw.get(tag.split(":")[-1]))
            if str(value) not in owned_leaves and str(value) not in new_leaves
        ]
        args += [f"-{tag}=" + value for value in [*foreign, *new_leaves]]

    # Aucune entrée du tout : effacer explicitement, sinon l'ancienne liste
    # resterait en place (une affectation absente ne touche pas au champ).
    for tag in (*TAG_HIERARCHICAL, *TAG_FLAT):
        if not any(arg.startswith(f"-{tag}=") for arg in args):
            args.append(f"-{tag}=")

    # IPTC n'est pas UTF-8 par défaut : sans cette déclaration, des libellés
    # accentués se reliraient en Latin-1 par un logiciel respectant la norme.
    args.append(f"-{TAG_CHARSET}=UTF8")
    return args


def write_tags(
    path: Path | str,
    new_paths: Sequence[Sequence[str]],
    *,
    owns: Owns,
    runner: ExiftoolRunner | None = None,
    extra_tags: Iterable[str] = (),
    extra_args: Sequence[str] = (),
    raw: dict | None = None,
) -> None:
    """Écrit *new_paths* dans les six champs de *path*, en préservant le reste.

    *extra_args* est ajouté à l'appel ``exiftool`` — c'est ainsi que
    :mod:`media_restorer.landmarks` joint ses régions MWG **dans la même
    écriture** : un seul passage, une seule copie ``<image>_original``.

    *raw* permet de réutiliser une lecture déjà faite (l'appelant qui a besoin
    de champs supplémentaires les lit une fois avec :func:`read_raw` puis passe
    le résultat ici) plutôt que d'interroger ``exiftool`` deux fois.
    """
    runner = runner or default_runner
    if raw is None:
        raw = read_raw(path, runner, extra_tags=extra_tags)
    runner([*tag_args(raw, new_paths, owns=owns), *extra_args, str(path)])

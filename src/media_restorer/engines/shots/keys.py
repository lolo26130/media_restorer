"""Clés d'identification d'une prise de vue, par ordre de fiabilité.

Apparier un RAW et son JPEG n'est pas un problème d'analyse d'image : les deux
fichiers sont la **même prise de vue**, et leurs métadonnées le disent
exactement.  Encore faut-il choisir la bonne clé.

.. warning::

   **Le nom de fichier n'est pas une clé.**  Le compteur des boîtiers Nikon
   reboucle : sur le corpus de référence, 62 noms de NEF apparaissent plusieurs
   fois, et ``330ND800/DSC_6769.nef`` n'a rien à voir avec
   ``329ND800/DSC_6769.jpg`` — ce sont deux prises différentes.  Apparier par le
   nom produit donc de **faux appariements silencieux**.

Les trois clés, de la plus sûre à la plus douteuse
--------------------------------------------------

``SURE`` — ``(SerialNumber, ShutterCount)``
    Le compteur de déclenchements est monotone par boîtier : le couple identifie
    une prise de vue de façon unique.  **100 % des NEF du corpus le portent**, et
    il apparie 76,7 % d'entre eux.

``TIME`` — ``(Model, DateTimeOriginal, SubSecTimeOriginal)``
    Repli pour les JPEG ré-exportés qui auraient perdu leurs MakerNotes.
    N'apporte rien sur le corpus de référence, mais coûte peu et couvrira
    d'autres cas.  La sous-seconde évite d'apparier deux vues d'une rafale.

``NAME`` — le nom de fichier, en dernier recours
    Conservé **uniquement** pour signaler des candidats à vérifier à l'œil, dans
    une catégorie à part.  Jamais présenté comme un appariement acquis.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

#: Méthodes d'appariement, de la plus fiable à la moins fiable.
MATCH_SURE = "sure"
MATCH_TIME = "horaire"
MATCH_NAME = "nom"

MATCH_ORDER = (MATCH_SURE, MATCH_TIME, MATCH_NAME)
MATCH_TITLES = {
    MATCH_SURE: "n° de série + déclenchements",
    MATCH_TIME: "date et sous-seconde",
    MATCH_NAME: "nom de fichier — à confirmer",
}

#: Champs à demander à ``exiftool``.  Voir l'avertissement de
#: :mod:`~media_restorer.engines.shots.pairing` sur l'option ``-fast2``.
EXIF_FIELDS = (
    "SerialNumber", "ShutterCount",
    "DateTimeOriginal", "SubSecTimeOriginal", "Model",
)


def sure_key(meta: Mapping[str, Any]) -> tuple | None:
    """``(SerialNumber, ShutterCount)`` — identifiant unique d'une prise."""
    serie, compteur = meta.get("SerialNumber"), meta.get("ShutterCount")
    if serie in (None, "") or compteur in (None, ""):
        return None
    return (MATCH_SURE, str(serie), str(compteur))


def time_key(meta: Mapping[str, Any]) -> tuple | None:
    """``(Model, DateTimeOriginal, SubSecTimeOriginal)`` — repli horaire.

    Le modèle de boîtier fait partie de la clé : deux appareils différents
    peuvent déclencher à la même seconde.
    """
    instant = meta.get("DateTimeOriginal")
    if instant in (None, ""):
        return None
    return (MATCH_TIME, str(meta.get("Model", "")), str(instant),
            str(meta.get("SubSecTimeOriginal", "")))


def name_key(path: Path) -> tuple:
    """Le nom sans extension, en minuscules — **candidat, pas appariement**."""
    return (MATCH_NAME, path.stem.lower())


def keys_of(path: Path, meta: Mapping[str, Any]) -> list[tuple]:
    """Toutes les clés d'un fichier, **dans l'ordre de fiabilité décroissante**.

    C'est cet ordre qui garantit qu'un appariement sûr l'emporte toujours sur un
    appariement par le nom, quelle que soit la façon dont l'appelant parcourt
    les fichiers.
    """
    cles = []
    for fabrique in (sure_key, time_key):
        if (cle := fabrique(meta)) is not None:
            cles.append(cle)
    cles.append(name_key(path))
    return cles


def match_method(key: tuple) -> str:
    """Méthode correspondant à une clé (son premier élément)."""
    return str(key[0])


def is_reliable(method: str) -> bool:
    """Un appariement par le nom seul n'est **pas** fiable — voir l'avertissement."""
    return method in (MATCH_SURE, MATCH_TIME)

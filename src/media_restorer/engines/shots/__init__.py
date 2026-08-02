"""Inventaire des correspondances RAW ↔ JPEG, sans Qt.

Répond à une seule question : **quel JPEG vient de quel RAW ?**  Ce n'est pas
une analyse d'image mais une **jointure par métadonnées** — exacte, sans seuil,
sans présomption.

Les fichiers RAW sont des **sauvegardes** en vue d'un développement ultérieur :
ils restent délibérément **hors de toutes les autres chaînes** du projet.
:data:`~media_restorer.engines.shots.pairing.RAW_SUFFIXES` est disjoint de
``IMAGE_SUFFIXES``, et un test le verrouille — ni le pré-classement ni la
détection de doublons ne doivent jamais voir un ``.NEF``.

Quatre catégories, exhaustives et disjointes : paires fiables, RAW sans JPEG
(*ceux dont il faut produire le dérivé*), JPEG sans RAW, et appariements par le
seul nom de fichier — présentés à part parce qu'ils ne sont pas fiables.

Deux pièges, tous deux **silencieux**, documentés là où ils comptent :

* ``exiftool -fast2`` supprime ``ShutterCount``
  (:mod:`~media_restorer.engines.shots.pairing`) ;
* Pillow « ouvre » un RAW et n'en lit que la vignette 160×120
  (:mod:`~media_restorer.engines.shots.preview`).
"""
from media_restorer.engines.shots.keys import (
    EXIF_FIELDS,
    MATCH_NAME,
    MATCH_ORDER,
    MATCH_SURE,
    MATCH_TIME,
    MATCH_TITLES,
    is_reliable,
    keys_of,
    name_key,
    sure_key,
    time_key,
)
from media_restorer.engines.shots.pairing import (
    DERIVED_SUFFIXES,
    RAW_SUFFIXES,
    ShotInventory,
    ShotPair,
    iter_files,
    read_metadata,
    scan_shots,
)
from media_restorer.engines.shots.preview import extract_preview, clear_cache

__all__ = [
    "DERIVED_SUFFIXES",
    "EXIF_FIELDS",
    "MATCH_NAME",
    "MATCH_ORDER",
    "MATCH_SURE",
    "MATCH_TIME",
    "MATCH_TITLES",
    "RAW_SUFFIXES",
    "ShotInventory",
    "ShotPair",
    "clear_cache",
    "extract_preview",
    "is_reliable",
    "iter_files",
    "keys_of",
    "name_key",
    "read_metadata",
    "scan_shots",
    "sure_key",
    "time_key",
]

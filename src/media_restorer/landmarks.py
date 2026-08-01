"""Lecture/écriture de points nommés (repères) dans les métadonnées d'une image.

Duplique — par copie, sans importation — puis recentre la gestion de
métadonnées de ``TraiteImages.classes.data_classes.DataImages`` (son
``read_tags`` via ExifTool, son ``exif_string_to_nested_dict``).  La classe
d'origine était un couteau suisse (chargement d'image, rotation, mise à
l'échelle, alignement de visage *et* EXIF) ; on n'en garde ici que la brique
métadonnées, réduite à un seul rôle : **stocker et relire un jeu de points
nommés** (« Left Eye », « Nose »…) désignés à la souris (voir
:class:`~media_restorer.image_click.ImageClick`).

Format de stockage — compatible DigiKam
---------------------------------------
Les repères ne sont plus enregistrés dans une charge utile JSON privée
(``UserComment``), mais dans les **champs standard que DigiKam lit et écrit**,
afin que les images soient filtrables depuis son gestionnaire d'étiquettes.
Deux natures d'information, deux supports :

**1. Ce qui est filtrable → étiquettes hiérarchiques**

Le même jeu d'étiquettes est écrit dans les six champs que DigiKam synchronise
entre eux, chacun avec sa propre convention.  La liste et les séparateurs sont
ceux que DigiKam déclare lui-même dans ``~/.config/digikamrc``, section
``[DMetadata Settings][readTagsNamespaces]`` (``separator`` et ``tagPaths``) —
et qu'on retrouve tels quels sur une image réellement étiquetée par lui :

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

DigiKam écrit un septième champ, ``XMP-acdsee:Categories``, sous forme de XML
imbriqué (``<Categories><Category…>``) : seul format de la liste à ne pas être
une simple liste de chemins, il est laissé de côté — aucun des champs relus
par DigiKam n'en dépend.

Arborescence écrite sous la racine :data:`_ROOT` :

.. code-block:: text

    media_restorer/Repère/<Libellé>          un repère marqué
    media_restorer/Repère ignoré/<Libellé>   un repère explicitement passé
    media_restorer/Repérage manuel           provenance (extension utilisée)
    media_restorer/Repérage automatique
    media_restorer/Repérage complet          aucun repère passé

Les branches sont nommées de façon que leur **feuille reste compréhensible
seule** : ``dc:Subject`` et ``IPTC:Keywords`` étant plats, DigiKam n'y recopie
que le dernier segment.  « Repérage complet » est donc d'un seul tenant plutôt
que ``Repérage/Complet``, dont la feuille « Complet » ne voudrait rien dire
hors contexte.

**2. Les coordonnées → régions MWG**

Elles ne peuvent pas être des étiquettes : un chemin ``.../Left Eye/34,2;51,8``
créerait une feuille unique par image et rendrait l'arbre d'étiquettes
inexploitable.  Elles vont dans ``XMP-mwg-rs:RegionInfo`` (standard *Metadata
Working Group*, lu nativement par DigiKam), dont les aires sont **déjà
normalisées entre 0 et 1** — nos pourcentages 0–100 s'y transposent sans
perte ni dépendance à la résolution.  Le type de région est
:data:`_REGION_TYPE` (« Focus ») et non « Face », pour ne pas peupler l'arbre
« Personnes » de DigiKam avec des noms de repères.

Fusion non destructive
----------------------
Contrairement à ``UserComment``, ces champs sont **curés par l'utilisateur**.
Une écriture nue les écraserait : ``exiftool -TagsList=…`` remplace la liste
entière, il ne l'complète pas.  :meth:`LandmarkSet.write_to_metadata` fait donc
systématiquement une **lecture-fusion-écriture** : les étiquettes étrangères
(hors racine ``media_restorer``) et les régions étrangères (type ≠ « Focus »,
typiquement les visages nommés par DigiKam) sont relues puis réécrites telles
quelles à côté des nôtres.

Lecture des images antérieures
------------------------------
Une image écrite par la version précédente ne porte que l'ancien
``UserComment`` JSON.  :meth:`LandmarkSet.read_from_metadata` retombe dessus
quand aucune étiquette ``media_restorer`` n'est présente — le travail déjà
enregistré reste lisible (voir :meth:`LandmarkSet.from_json`).

Autres améliorations par rapport à la source
--------------------------------------------
- Aucune dépendance à ``PyExifTool`` (non installé ici) : appel direct du
  binaire ``exiftool`` en sous-processus — le *runner* est injectable, si bien
  que les tests n'ont jamais besoin de lancer le vrai binaire (convention
  d'injection de dépendances du ``CLAUDE.md`` racine).
- Sans état d'image : ``LandmarkSet`` ne transporte que les points, pas les
  pixels — la lecture/écriture des pixels reste à :mod:`media_restorer.image_io`.
- ``exiftool`` conserve par défaut une copie ``<image>_original`` : l'écriture
  reste réversible.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from media_restorer import digikam_tags as _tags
from media_restorer.digikam_tags import ExiftoolRunner

# Un point : coordonnées en **pourcentage** (0–100) de la largeur (x) et de la
# hauteur (y) de l'image, ou ``None`` si le repère a été passé.  Les
# pourcentages sont indépendants de la résolution : aucune conversion n'est
# nécessaire si l'image est redimensionnée (c'est tout l'intérêt de ce choix).
Point = tuple[float, float] | None

# -- Arborescence d'étiquettes ---------------------------------------------
# Racine commune à toute l'application (voir :mod:`media_restorer.digikam_tags`).
_ROOT = _tags.ROOT
_BRANCH_MARKED = "Repère"
_BRANCH_SKIPPED = "Repère ignoré"
_LEAF_COMPLETE = "Repérage complet"
# Provenance : clé interne → feuille affichée.  Les feuilles sont d'un seul
# tenant pour rester lisibles dans les champs plats (dc:Subject, IPTC:Keywords).
_LEAF_BY_SOURCE = {"manual": "Repérage manuel", "auto": "Repérage automatique"}
_SOURCE_BY_LEAF = {leaf: source for source, leaf in _LEAF_BY_SOURCE.items()}

# -- Champs porteurs --------------------------------------------------------
# Les six champs d'étiquettes DigiKam et la syntaxe « structure » d'exiftool
# sont dans :mod:`media_restorer.digikam_tags`, partagés avec le pré-classement.
_TAG_REGIONS = "XMP-mwg-rs:RegionInfo"
# Type MWG de nos régions.  « Focus » (point d'intérêt) plutôt que « Face » :
# DigiKam n'importe que les régions « Face » dans son arbre « Personnes ».
_REGION_TYPE = "Focus"

# Format hérité (lecture seule) : ancienne charge utile JSON privée.
_LEGACY_TAG = "UserComment"
_LEGACY_SCHEMA_KEY = "media_restorer_landmarks"


# Branches revendiquées par ce module — et **elles seules**.  Revendiquer toute
# la racine ``media_restorer`` effacerait en silence les étiquettes posées par
# le pré-classement (:mod:`media_restorer.engines.triage.tags`), et
# réciproquement : l'écriture reconstruit chaque champ en entier.  Voir
# l'avertissement en tête de :mod:`media_restorer.digikam_tags`.
_owns = _tags.branch_owner(
    _BRANCH_MARKED, _BRANCH_SKIPPED, _LEAF_COMPLETE, *_LEAF_BY_SOURCE.values()
)


@dataclass
class LandmarkSet:
    """Jeu de points nommés désignés sur une image, sérialisable dans ses métadonnées.

    Version recentrée de ``DataImages`` : ne porte que les repères, pas les
    pixels ni les transformations d'image.

    Attributs
    ---------
    points : dict[str, Point]
        Repère → coordonnées ``(x, y)`` en **pourcentage** (0–100) de la
        largeur/hauteur, ou ``None`` si passé.  L'ordre d'insertion (garanti
        par ``dict``) reflète l'ordre de désignation.
    source : str | None
        Provenance du repérage : ``"manual"`` (désignation à la souris),
        ``"auto"`` (détection), ou ``None`` (non renseignée — aucune étiquette
        de provenance n'est alors écrite).  Sert uniquement à produire une
        étiquette filtrable dans DigiKam.
    """

    points: dict[str, Point] = field(default_factory=dict)
    source: str | None = None

    # -- Arborescence d'étiquettes ------------------------------------------

    def tag_paths(self) -> list[list[str]]:
        """Chemins d'étiquettes décrivant ce jeu de repères (racine incluse).

        Un chemin par repère (marqué ou ignoré), plus les étiquettes de
        synthèse (provenance, repérage complet).  Ne dépend que de
        :attr:`points` et :attr:`source` — les coordonnées, elles, partent dans
        les régions MWG.
        """
        paths = [
            [_ROOT, _BRANCH_MARKED if pt is not None else _BRANCH_SKIPPED, label]
            for label, pt in self.points.items()
        ]
        leaf = _LEAF_BY_SOURCE.get(self.source or "")
        if leaf:
            paths.append([_ROOT, leaf])
        if self.points and all(pt is not None for pt in self.points.values()):
            paths.append([_ROOT, _LEAF_COMPLETE])
        return paths

    # -- Sérialisation JSON héritée (lecture des images antérieures) --------

    def to_json(self) -> str:
        """Charge utile JSON du **format hérité** (``UserComment``).

        N'est plus écrite : conservée pour documenter la forme relue par
        :meth:`from_json` et permettre aux tests de fabriquer une image
        « ancienne version ».
        """
        serialisable = {
            label: ([float(pt[0]), float(pt[1])] if pt is not None else None)
            for label, pt in self.points.items()
        }
        return json.dumps({_LEGACY_SCHEMA_KEY: serialisable}, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "LandmarkSet | None":
        """Reconstruit un :class:`LandmarkSet` depuis le format hérité, ou ``None``.

        Renvoie ``None`` (et non un jeu vide) si *raw* n'est pas notre schéma :
        c'est ce qui permet au repli hérité de
        :meth:`read_from_metadata` de distinguer « pas de repères enregistrés »
        d'un ``UserComment`` sans rapport laissé par un autre logiciel.
        """
        try:
            blob = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(blob, dict) or _LEGACY_SCHEMA_KEY not in blob:
            return None
        payload = blob[_LEGACY_SCHEMA_KEY]
        if not isinstance(payload, dict):
            return None
        points: dict[str, Point] = {}
        for label, value in payload.items():
            if value is None:
                points[label] = None
            elif isinstance(value, (list, tuple)) and len(value) == 2:
                points[label] = (float(value[0]), float(value[1]))
            # tout autre forme est ignorée (donnée corrompue) plutôt que fatale
        return cls(points=points)

    # -- Support métadonnées (exiftool en sous-processus) -------------------

    def write_to_metadata(self, path: Path | str, *, runner: ExiftoolRunner | None = None) -> None:
        """Écrit les repères de *path* au format DigiKam (étiquettes + régions MWG).

        Procède par **lecture-fusion-écriture** : les étiquettes et régions
        déjà présentes qui ne sont pas les nôtres sont préservées (voir la
        section « Fusion non destructive » de ce module).  ``exiftool``
        conserve par défaut une copie ``<image>_original``.  *runner* est
        injectable pour les tests (défaut : le vrai binaire, voir
        :func:`_default_runner`).
        """
        runner = runner or _tags.default_runner
        existing = _read_raw(path, runner)
        _tags.write_tags(
            path, self.tag_paths(),
            owns=_owns, runner=runner, raw=existing,
            extra_args=_region_args(existing, self.points),
        )

    @classmethod
    def read_from_metadata(cls, path: Path | str, *, runner: ExiftoolRunner | None = None) -> "LandmarkSet":
        """Relit les repères depuis les métadonnées de *path*.

        Lit les étiquettes ``media_restorer`` de ``TagsList`` (quels repères,
        marqués ou ignorés, et la provenance) et les coordonnées dans les
        régions MWG.  À défaut, retombe sur l'ancien ``UserComment`` JSON — une
        image écrite par la version précédente reste lisible.

        Renvoie un jeu vide si le fichier ne porte aucun repère à nous (aucune
        métadonnée, ou étiquettes d'un autre logiciel).  Ne distingue pas
        volontairement l'un de l'autre côté appelant : dans les deux cas, il
        n'y a rien à afficher.
        """
        raw = _read_raw(path, runner or _tags.default_runner)
        paths = _tags.read_tag_paths(raw, _owns)
        if not paths:
            return cls._read_legacy(raw)

        coords = _region_points(raw)
        points: dict[str, Point] = {}
        source: str | None = None
        for parts in paths:
            if len(parts) == 3 and parts[1] == _BRANCH_MARKED:
                # Étiquette « marqué » sans région correspondante (régions
                # effacées par un autre outil) : le repère est **omis**, jamais
                # rendu comme passé.  Le rendre passé serait affirmer une
                # information fausse — et la prochaine écriture graverait cette
                # rétrogradation dans le fichier.  Absent = « on ne sait rien ».
                if parts[2] in coords:
                    points[parts[2]] = coords[parts[2]]
            elif len(parts) == 3 and parts[1] == _BRANCH_SKIPPED:
                points[parts[2]] = None
            elif len(parts) == 2:
                source = _SOURCE_BY_LEAF.get(parts[1], source)
        return cls(points=points, source=source)

    @classmethod
    def _read_legacy(cls, raw: dict) -> "LandmarkSet":
        """Repli sur l'ancien ``UserComment`` JSON (images d'avant ce format)."""
        payload = raw.get(_LEGACY_TAG)
        if not isinstance(payload, str):
            return cls()
        return cls.from_json(payload) or cls()


# ---------------------------------------------------------------------------
# Lecture brute et construction des arguments exiftool
# ---------------------------------------------------------------------------

def _read_raw(path: Path | str, runner: ExiftoolRunner) -> dict:
    """Lecture unique couvrant étiquettes, régions, dimensions et format hérité.

    Étend :func:`media_restorer.digikam_tags.read_raw` des champs propres aux
    repères : les régions MWG (les coordonnées), les dimensions natives
    (``AppliedToDimensions``, requis par MWG pour interpréter une aire
    normalisée) et l'ancien ``UserComment`` (repli hérité).  Un seul appel
    ``exiftool`` sert donc à la fois la lecture et la fusion d'écriture.
    """
    return _tags.read_raw(
        path, runner,
        extra_tags=("ImageWidth", "ImageHeight", _LEGACY_TAG, _TAG_REGIONS),
    )


def _region_args(raw: dict, points: dict[str, Point]) -> list[str]:
    """Argument d'écriture de ``RegionInfo``, régions étrangères préservées.

    Les repères passés (``None``) n'ont pas de coordonnées : ils ne produisent
    aucune région et ne survivent que par leur étiquette « Repère ignoré ».
    """
    info = raw.get("RegionInfo")
    info = info if isinstance(info, dict) else {}
    foreign = [
        region for region in _tags.as_list(info.get("RegionList"))
        if isinstance(region, dict) and region.get("Type") != _REGION_TYPE
    ]
    ours = [
        {
            "Name": label,
            "Type": _REGION_TYPE,
            # 0–100 % → 0–1 normalisé : la seule conversion du format.
            "Area": {"X": pt[0] / 100.0, "Y": pt[1] / 100.0, "Unit": "normalized"},
        }
        for label, pt in points.items() if pt is not None
    ]
    regions = [*foreign, *ours]
    if not regions:
        return [f"-{_TAG_REGIONS}="]

    struct: dict[str, Any] = {}
    # AppliedToDimensions : requis par MWG pour interpréter une aire normalisée.
    # Repris tel quel s'il existe, sinon reconstruit depuis la taille de l'image.
    applied = info.get("AppliedToDimensions")
    if isinstance(applied, dict):
        struct["AppliedToDimensions"] = applied
    elif raw.get("ImageWidth") and raw.get("ImageHeight"):
        struct["AppliedToDimensions"] = {
            "W": raw["ImageWidth"], "H": raw["ImageHeight"], "Unit": "pixel",
        }
    struct["RegionList"] = regions
    return [f"-{_TAG_REGIONS}=" + _tags.to_struct(struct)]


def _region_points(raw: dict) -> dict[str, tuple[float, float]]:
    """Coordonnées (en %) de nos régions MWG, indexées par nom de repère."""
    info = raw.get("RegionInfo")
    if not isinstance(info, dict):
        return {}
    coords: dict[str, tuple[float, float]] = {}
    for region in _tags.as_list(info.get("RegionList")):
        if not isinstance(region, dict) or region.get("Type") != _REGION_TYPE:
            continue
        area = region.get("Area")
        name = region.get("Name")
        if not isinstance(area, dict) or not isinstance(name, str):
            continue
        try:
            coords[name] = (float(area["X"]) * 100.0, float(area["Y"]) * 100.0)
        except (KeyError, TypeError, ValueError):
            continue
    return coords

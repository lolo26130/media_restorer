"""Tests du cœur de stockage des repères (media_restorer.landmarks).

Sans Qt.  Le lanceur ``exiftool`` est injecté (un faux) dans tous les tests
unitaires : ils vérifient la **forme des arguments** produits et le **décodage**
de sorties fabriquées à la main, sans jamais lancer le binaire.

Un unique test d'intégration en fin de fichier lance le vrai ``exiftool`` sur
une image jetable (il se saute tout seul si le binaire est absent) : c'est le
seul qui puisse prouver que la syntaxe « structure » des régions MWG est
acceptée et que la fusion préserve réellement les étiquettes d'un autre
logiciel — un faux runner qui réinterpréterait cette syntaxe pourrait se
tromper exactement comme le code testé.
"""
import json
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from media_restorer.landmarks import LandmarkSet

_TAGS_LIST = "XMP-digiKam:TagsList"
_HIERARCHICAL = "XMP-lr:HierarchicalSubject"
_SUBJECT = "XMP-dc:Subject"
_KEYWORDS = "IPTC:Keywords"
_REGIONS = "XMP-mwg-rs:RegionInfo"


def _exiftool_json(**fields):
    """Reproduit la sortie ``exiftool -j -struct …`` (un enregistrement)."""
    return json.dumps([{"SourceFile": "x.jpg", **fields}])


def _values(args, tag):
    """Valeurs affectées à *tag* dans une ligne de commande exiftool."""
    prefix = f"-{tag}="
    return [a[len(prefix):] for a in args if a.startswith(prefix)]


def _region(name, x, y, kind="Focus"):
    """Une région MWG telle que la relit ``exiftool -struct`` (aire normalisée)."""
    return {"Name": name, "Type": kind, "Area": {"X": x, "Y": y, "Unit": "normalized"}}


def _capturing_runner(calls, **existing):
    """Runner qui sert *existing* en lecture et mémorise l'appel d'écriture."""
    def runner(args):
        if any("=" in a for a in args if a.startswith("-")):
            calls.append(args)
            return "1 image files updated"
        return _exiftool_json(**existing)
    return runner


# ---------------------------------------------------------------------------
# Arborescence d'étiquettes
# ---------------------------------------------------------------------------

def test_tag_paths_distinguish_marked_from_skipped_landmarks():
    landmarks = LandmarkSet(points={"Left Eye": (10.0, 20.0), "Nose": None})

    assert landmarks.tag_paths() == [
        ["media_restorer", "Repère", "Left Eye"],
        ["media_restorer", "Repère ignoré", "Nose"],
    ]


def test_tag_paths_add_source_and_completeness_leaves():
    landmarks = LandmarkSet(points={"Left Eye": (10.0, 20.0)}, source="auto")

    leaves = [parts[-1] for parts in landmarks.tag_paths()]

    assert "Repérage automatique" in leaves
    assert "Repérage complet" in leaves  # aucun repère passé


def test_tag_paths_omit_completeness_when_a_landmark_is_skipped():
    landmarks = LandmarkSet(points={"Left Eye": (10.0, 20.0), "Nose": None}, source="manual")

    leaves = [parts[-1] for parts in landmarks.tag_paths()]

    assert "Repérage manuel" in leaves
    assert "Repérage complet" not in leaves


def test_tag_paths_omit_source_leaf_when_unknown():
    assert LandmarkSet(points={"A": (1.0, 2.0)}).tag_paths() == [
        ["media_restorer", "Repère", "A"],
        ["media_restorer", "Repérage complet"],
    ]


# ---------------------------------------------------------------------------
# Écriture : forme des arguments
# ---------------------------------------------------------------------------

def test_write_fills_the_six_digikam_fields_with_their_own_separator(tmp_path):
    """Les six champs, séparateurs compris, tels que DigiKam les déclare.

    Liste et séparateurs relevés dans ``~/.config/digikamrc``
    (``[DMetadata Settings][readTagsNamespaces]``) et confirmés sur une image
    réellement étiquetée par DigiKam : ``tagPaths=1`` → chemin complet,
    ``tagPaths=0`` (``dc:Subject``, ``IPTC:Keywords``) → feuille seule.
    """
    calls = []
    LandmarkSet(points={"Left Eye": (10.0, 20.0)}, source="manual").write_to_metadata(
        tmp_path / "p.jpg", runner=_capturing_runner(calls)
    )

    args = calls[0]
    for tag in (_TAGS_LIST, "XMP-microsoft:LastKeywordXMP"):
        assert "media_restorer/Repère/Left Eye" in _values(args, tag)
    for tag in (_HIERARCHICAL, "XMP-mediapro:CatalogSets"):
        assert "media_restorer|Repère|Left Eye" in _values(args, tag)
    for tag in (_SUBJECT, _KEYWORDS):
        assert "Left Eye" in _values(args, tag)
        assert "Repère" not in _values(args, tag)   # jamais le chemin, la feuille
    assert "Repérage manuel" in _values(args, _SUBJECT)


def test_write_converts_percentages_to_normalised_mwg_areas(tmp_path):
    calls = []
    LandmarkSet(points={"Left Eye": (34.2, 51.8)}).write_to_metadata(
        tmp_path / "p.jpg", runner=_capturing_runner(calls, ImageWidth=400, ImageHeight=300)
    )

    struct = _values(calls[0], _REGIONS)[0]
    assert "Name=Left Eye" in struct
    assert "Type=Focus" in struct          # jamais « Face » : pas d'arbre Personnes
    assert "X=0.342" in struct
    assert "Y=0.518" in struct
    assert "W=400" in struct and "H=300" in struct  # AppliedToDimensions requis par MWG


def test_write_emits_no_region_for_a_skipped_landmark(tmp_path):
    calls = []
    LandmarkSet(points={"Nose": None}).write_to_metadata(
        tmp_path / "p.jpg", runner=_capturing_runner(calls)
    )

    # Le repère passé ne survit que par son étiquette « Repère ignoré ».
    assert _values(calls[0], _REGIONS) == [""]
    assert "media_restorer/Repère ignoré/Nose" in _values(calls[0], _TAGS_LIST)


def test_write_preserves_foreign_tags(tmp_path):
    """Les étiquettes curées par l'utilisateur ne doivent jamais être écrasées."""
    calls = []
    runner = _capturing_runner(
        calls,
        TagsList=["Vacances/Bretagne", "media_restorer/Repère/Ancien"],
        HierarchicalSubject=["Vacances|Bretagne", "media_restorer|Repère|Ancien"],
        Subject=["Bretagne", "Ancien"],
        Keywords=["Bretagne", "Ancien"],
    )

    LandmarkSet(points={"Left Eye": (10.0, 20.0)}).write_to_metadata(
        tmp_path / "p.jpg", runner=runner
    )

    args = calls[0]
    tags = _values(args, _TAGS_LIST)
    assert "Vacances/Bretagne" in tags                    # étiquette étrangère gardée
    assert "media_restorer/Repère/Ancien" not in tags     # la nôtre, remplacée
    assert "Bretagne" in _values(args, _SUBJECT)
    assert "Ancien" not in _values(args, _SUBJECT)


def test_write_cleans_stale_leaves_from_a_desynchronised_flat_field(tmp_path):
    """Un champ plat désynchronisé ne doit pas accumuler des repères orphelins.

    ``Subject`` mentionne encore « Ancien », que seul ``CatalogSets`` rattache
    à notre racine (``TagsList`` ne le connaît plus).  Reconnaître nos feuilles
    sur le seul premier champ qui répond laisserait « Ancien » passer pour une
    étiquette de l'utilisateur — et survivre à chaque écriture.
    """
    calls = []
    runner = _capturing_runner(
        calls,
        TagsList=["media_restorer/Repère/Récent"],
        CatalogSets=["media_restorer|Repère|Ancien"],
        Subject=["Bretagne", "Récent", "Ancien"],
    )

    LandmarkSet(points={"Left Eye": (10.0, 20.0)}).write_to_metadata(
        tmp_path / "p.jpg", runner=runner
    )

    subject = _values(calls[0], _SUBJECT)
    assert "Bretagne" in subject       # étiquette de l'utilisateur, conservée
    assert "Ancien" not in subject     # feuille orpheline, nettoyée
    assert "Récent" not in subject


def test_write_declares_the_iptc_charset(tmp_path):
    """Sans déclaration, les libellés accentués se reliraient en Latin-1."""
    calls = []
    LandmarkSet(points={"Left Eye": (10.0, 20.0)}, source="manual").write_to_metadata(
        tmp_path / "p.jpg", runner=_capturing_runner(calls)
    )

    assert _values(calls[0], "IPTC:CodedCharacterSet") == ["UTF8"]


def test_write_preserves_foreign_face_regions(tmp_path):
    """Une région « Face » (visage nommé dans DigiKam) survit à notre écriture."""
    calls = []
    runner = _capturing_runner(
        calls,
        RegionInfo={
            "AppliedToDimensions": {"W": 400, "H": 300, "Unit": "pixel"},
            "RegionList": [_region("Alice", 0.5, 0.4, kind="Face"),
                           _region("Vieux Repère", 0.1, 0.1)],
        },
    )

    LandmarkSet(points={"Left Eye": (10.0, 20.0)}).write_to_metadata(
        tmp_path / "p.jpg", runner=runner
    )

    struct = _values(calls[0], _REGIONS)[0]
    assert "Name=Alice" in struct            # visage étranger conservé
    assert "Vieux Repère" not in struct      # notre région précédente, remplacée
    assert "Name=Left Eye" in struct


def test_write_escapes_structure_characters_in_a_landmark_name(tmp_path):
    calls = []
    LandmarkSet(points={"Oeil, gauche": (10.0, 20.0)}).write_to_metadata(
        tmp_path / "p.jpg", runner=_capturing_runner(calls)
    )

    # La virgule sépare les champs d'une structure exiftool : elle doit être
    # échappée par « | », sinon le nom casse la structure entière.
    assert "Name=Oeil|, gauche" in _values(calls[0], _REGIONS)[0]


# ---------------------------------------------------------------------------
# Lecture
# ---------------------------------------------------------------------------

def test_read_recombines_tags_and_regions(tmp_path):
    runner = lambda args: _exiftool_json(
        TagsList=["media_restorer/Repère/Left Eye",
                  "media_restorer/Repère ignoré/Nose",
                  "media_restorer/Repérage automatique"],
        RegionInfo={"RegionList": [_region("Left Eye", 0.342, 0.518)]},
    )

    result = LandmarkSet.read_from_metadata(tmp_path / "p.jpg", runner=runner)

    # approx : l'aller-retour % → normalisé → % passe par une division et une
    # multiplication flottantes (écart ~1e-14, sans effet à l'échelle du pixel).
    assert result.points == {"Left Eye": pytest.approx((34.2, 51.8)), "Nose": None}
    assert result.source == "auto"


def test_read_preserves_tag_order(tmp_path):
    runner = lambda args: _exiftool_json(
        TagsList=[f"media_restorer/Repère ignoré/{label}" for label in ("C", "A", "B")]
    )

    result = LandmarkSet.read_from_metadata(tmp_path / "p.jpg", runner=runner)

    assert list(result.points) == ["C", "A", "B"]


def test_read_ignores_tags_outside_our_root(tmp_path):
    """Une étiquette « Left Eye » de l'utilisateur n'est pas prise pour un repère."""
    runner = lambda args: _exiftool_json(
        TagsList=["Portraits/Left Eye"],
        RegionInfo={"RegionList": [_region("Left Eye", 0.1, 0.2)]},
    )

    assert LandmarkSet.read_from_metadata(tmp_path / "p.jpg", runner=runner).points == {}


def test_read_ignores_foreign_face_regions(tmp_path):
    runner = lambda args: _exiftool_json(
        TagsList=["media_restorer/Repère/Alice"],
        RegionInfo={"RegionList": [_region("Alice", 0.5, 0.4, kind="Face")]},
    )

    # Une région « Face » n'est pas une de nos coordonnées : le repère n'emprunte
    # pas la position d'un visage nommé dans DigiKam.
    assert LandmarkSet.read_from_metadata(tmp_path / "p.jpg", runner=runner).points == {}


def test_a_marked_landmark_without_region_is_omitted_not_demoted(tmp_path):
    """Ne jamais rendre « passé » un repère dont la région a disparu.

    Le rendre passé affirmerait une information fausse — et la réécriture
    suivante graverait cette rétrogradation dans le fichier, transformant une
    incohérence passagère en perte définitive.  Absent = « on ne sait rien ».
    """
    runner = lambda args: _exiftool_json(
        TagsList=["media_restorer/Repère/Left Eye", "media_restorer/Repère ignoré/Nose"],
    )  # aucune région : elles ont été effacées par un autre outil

    result = LandmarkSet.read_from_metadata(tmp_path / "p.jpg", runner=runner)

    assert result.points == {"Nose": None}          # le « passé » reste passé
    assert "Left Eye" not in result.points          # le marqué disparaît, pas rétrogradé


def test_read_returns_empty_without_any_metadata(tmp_path):
    assert LandmarkSet.read_from_metadata(
        tmp_path / "p.jpg", runner=lambda args: _exiftool_json()
    ).points == {}


def test_read_returns_empty_on_unusable_exiftool_output(tmp_path):
    assert LandmarkSet.read_from_metadata(tmp_path / "p.jpg", runner=lambda args: "").points == {}


# ---------------------------------------------------------------------------
# Format hérité (images écrites par la version précédente)
# ---------------------------------------------------------------------------

def test_read_falls_back_to_the_legacy_usercomment(tmp_path):
    stored = LandmarkSet(points={"Left Eye": (10.0, 20.0), "Nose": None}).to_json()
    runner = lambda args: _exiftool_json(UserComment=stored)

    result = LandmarkSet.read_from_metadata(tmp_path / "p.jpg", runner=runner)

    assert result.points == {"Left Eye": (10.0, 20.0), "Nose": None}


def test_digikam_tags_take_precedence_over_a_legacy_usercomment(tmp_path):
    runner = lambda args: _exiftool_json(
        UserComment=LandmarkSet(points={"Ancien": (1.0, 2.0)}).to_json(),
        TagsList=["media_restorer/Repère ignoré/Nouveau"],
    )

    assert list(LandmarkSet.read_from_metadata(tmp_path / "p.jpg", runner=runner).points) == ["Nouveau"]


def test_legacy_payload_from_another_program_is_not_mistaken_for_landmarks():
    assert LandmarkSet.from_json("just a plain caption") is None
    assert LandmarkSet.from_json(json.dumps({"something_else": 1})) is None


def test_legacy_payload_ignores_malformed_coordinates_without_crashing():
    raw = json.dumps({"media_restorer_landmarks": {"Ok": [1, 2], "Bad": [1, 2, 3], "Weird": "x"}})

    restored = LandmarkSet.from_json(raw)

    assert restored is not None
    assert restored.points == {"Ok": (1, 2)}


# ---------------------------------------------------------------------------
# Intégration — vrai exiftool sur une image jetable
# ---------------------------------------------------------------------------

@pytest.mark.skipif(shutil.which("exiftool") is None, reason="exiftool non installé")
def test_real_exiftool_round_trip_preserves_foreign_metadata(tmp_path):
    img = tmp_path / "photo.jpg"
    cv2.imwrite(str(img), np.full((300, 400, 3), 180, np.uint8))
    subprocess.run(
        ["exiftool", "-overwrite_original",
         "-XMP-digiKam:TagsList=Vacances/Bretagne",
         "-XMP-dc:Subject=Bretagne",
         "-XMP-mwg-rs:RegionInfo={AppliedToDimensions={W=400,H=300,Unit=pixel},"
         "RegionList=[{Name=Alice,Type=Face,Area={X=0.5,Y=0.4,W=0.2,H=0.2,Unit=normalized}}]}",
         str(img)],
        check=True, capture_output=True,
    )

    LandmarkSet(
        points={"Left Eye": (34.2, 51.8), "Nose": None}, source="auto"
    ).write_to_metadata(img)
    result = LandmarkSet.read_from_metadata(img)

    assert result.points == {"Left Eye": pytest.approx((34.2, 51.8)), "Nose": None}
    assert result.source == "auto"

    out = subprocess.run(
        ["exiftool", "-j", "-XMP-digiKam:TagsList", "-XMP-dc:Subject",
         "-IPTC:Keywords", "-IPTC:CodedCharacterSet", "-XMP-mwg-rs:RegionName", str(img)],
        check=True, capture_output=True, text=True,
    )
    record = json.loads(out.stdout)[0]
    assert "Vacances/Bretagne" in record["TagsList"]   # étiquette de l'utilisateur intacte
    assert "Bretagne" in record["Subject"]
    assert "Alice" in record["RegionName"]             # visage DigiKam intact
    # IPTC : accents réellement relus, et encodage déclaré comme le fait DigiKam.
    assert "Repérage automatique" in record["Keywords"]
    assert record["CodedCharacterSet"] == "UTF8"

"""Tests du moteur d'étiquettes DigiKam partagé (media_restorer.digikam_tags).

Sans Qt.  Le lanceur ``exiftool`` est injecté partout sauf dans le dernier test
d'intégration, qui se saute tout seul si le binaire est absent.

Le test le plus important du fichier est
:func:`test_two_owners_never_erase_each_other` : c'est lui qui verrouille la
raison d'être du module — deux écrivains d'étiquettes (repères et
pré-classement) partagent les mêmes six champs, et une écriture reconstruit
chaque champ **en entier**.  Sans prédicat d'appartenance restreint, chacun
effacerait l'autre sans lever la moindre erreur.
"""
import json
import shutil

import numpy as np
import pytest

try:                                   # cv2 sert seulement au test d'intégration
    import cv2
except ImportError:                    # pragma: no cover
    cv2 = None

from media_restorer import digikam_tags as T

_TAGS_LIST = "XMP-digiKam:TagsList"
_HIERARCHICAL = "XMP-lr:HierarchicalSubject"
_SUBJECT = "XMP-dc:Subject"
_KEYWORDS = "IPTC:Keywords"


def _exiftool_json(**fields):
    """Reproduit la sortie ``exiftool -j -struct …`` (un enregistrement)."""
    return json.dumps([{"SourceFile": "x.jpg", **fields}])


def _values(args, tag):
    prefix = f"-{tag}="
    return [a[len(prefix):] for a in args if a.startswith(prefix)]


def _capturing_runner(calls, **existing):
    """Sert *existing* en lecture, mémorise l'appel d'écriture."""
    def runner(args):
        if any("=" in a for a in args if a.startswith("-")):
            calls.append(args)
            return "1 image files updated"
        return _exiftool_json(**existing)
    return runner


# ---------------------------------------------------------------------------
# Le prédicat d'appartenance
# ---------------------------------------------------------------------------

def test_branch_owner_claims_only_its_own_branches():
    owns = T.branch_owner("Tri")

    assert owns(["media_restorer", "Tri", "Orientation", "Portrait"])
    assert not owns(["media_restorer", "Repère", "Left Eye"])   # branche voisine
    assert not owns(["Vacances", "Tri"])                        # autre racine
    assert not owns(["media_restorer"])                         # racine nue


def test_branch_owner_accepts_several_branches():
    owns = T.branch_owner("Repère", "Repère ignoré")

    assert owns(["media_restorer", "Repère", "A"])
    assert owns(["media_restorer", "Repère ignoré", "B"])
    assert not owns(["media_restorer", "Tri", "C"])


# ---------------------------------------------------------------------------
# Lecture
# ---------------------------------------------------------------------------

def test_read_raw_requests_the_six_fields_and_the_extras():
    seen = []
    T.read_raw("/p.jpg", lambda args: seen.append(args) or _exiftool_json(),
               extra_tags=("ImageWidth",))

    args = seen[0]
    assert "-j" in args and "-struct" in args and "/p.jpg" in args
    for tag in (*T.TAG_HIERARCHICAL, *T.TAG_FLAT):
        assert f"-{tag}" in args
    assert "-ImageWidth" in args


def test_read_raw_never_raises_on_unusable_output():
    assert T.read_raw("/p.jpg", lambda args: "pas du json") == {}
    assert T.read_raw("/p.jpg", lambda args: "[]") == {}
    def boom(args):
        raise RuntimeError("exiftool absent")
    assert T.read_raw("/p.jpg", boom) == {}


def test_each_field_is_split_with_its_own_separator():
    raw = {"TagsList": ["a/b"], "HierarchicalSubject": ["a|b"]}
    assert T.split_paths(raw, _TAGS_LIST, "/") == [["a", "b"]]
    assert T.split_paths(raw, _HIERARCHICAL, "|") == [["a", "b"]]


def test_read_tag_paths_prefers_tagslist():
    raw = {"TagsList": ["media_restorer/Tri/A"],
           "HierarchicalSubject": ["media_restorer|Tri|PERIME"]}

    assert T.read_tag_paths(raw, T.branch_owner("Tri")) == [["media_restorer", "Tri", "A"]]


def test_read_tag_paths_falls_back_to_another_field():
    """Une image étiquetée par Lightroom seul ne porte pas de TagsList."""
    raw = {"HierarchicalSubject": ["media_restorer|Tri|A"]}

    assert T.read_tag_paths(raw, T.branch_owner("Tri")) == [["media_restorer", "Tri", "A"]]


def test_own_leaves_unions_every_hierarchical_field():
    """Une feuille périmée que seul un champ désynchronisé mentionne reste nôtre."""
    raw = {"TagsList": ["media_restorer/Tri/Recent"],
           "CatalogSets": ["media_restorer|Tri|Ancien"]}

    assert T.own_leaves(raw, T.branch_owner("Tri")) == {"Recent", "Ancien"}


# ---------------------------------------------------------------------------
# Écriture
# ---------------------------------------------------------------------------

def test_write_fills_the_six_fields_and_declares_the_charset():
    calls = []
    T.write_tags("/p.jpg", [["media_restorer", "Tri", "Orientation", "Portrait"]],
                 owns=T.branch_owner("Tri"), runner=_capturing_runner(calls))

    args = calls[0]
    for tag in (_TAGS_LIST, "XMP-microsoft:LastKeywordXMP"):
        assert "media_restorer/Tri/Orientation/Portrait" in _values(args, tag)
    for tag in (_HIERARCHICAL, "XMP-mediapro:CatalogSets"):
        assert "media_restorer|Tri|Orientation|Portrait" in _values(args, tag)
    for tag in (_SUBJECT, _KEYWORDS):
        assert _values(args, tag) == ["Portrait"]        # feuille seule
    assert _values(args, T.TAG_CHARSET) == ["UTF8"]      # IPTC n'est pas UTF-8


def test_write_preserves_foreign_tags():
    calls = []
    runner = _capturing_runner(
        calls,
        TagsList=["Vacances/Bretagne", "media_restorer/Tri/Ancien"],
        Subject=["Bretagne", "Ancien"],
    )
    T.write_tags("/p.jpg", [["media_restorer", "Tri", "Neuf"]],
                 owns=T.branch_owner("Tri"), runner=runner)

    tags = _values(calls[0], _TAGS_LIST)
    assert "Vacances/Bretagne" in tags        # étiquette de l'utilisateur gardée
    assert "media_restorer/Tri/Ancien" not in tags   # la nôtre, remplacée
    assert "Bretagne" in _values(calls[0], _SUBJECT)
    assert "Ancien" not in _values(calls[0], _SUBJECT)


def test_writing_nothing_clears_our_fields_explicitly():
    """Une affectation absente ne toucherait pas au champ : il faut l'effacer."""
    calls = []
    T.write_tags("/p.jpg", [], owns=T.branch_owner("Tri"),
                 runner=_capturing_runner(calls))

    for tag in (*T.TAG_HIERARCHICAL, *T.TAG_FLAT):
        assert _values(calls[0], tag) == [""]


def test_extra_args_ride_along_in_the_same_exiftool_call():
    """Une seule écriture, donc une seule copie « _original » conservée."""
    calls = []
    T.write_tags("/p.jpg", [["media_restorer", "Tri", "A"]],
                 owns=T.branch_owner("Tri"), runner=_capturing_runner(calls),
                 extra_args=["-XMP-mwg-rs:RegionInfo={}"])

    assert "-XMP-mwg-rs:RegionInfo={}" in calls[0]
    assert calls[0][-1] == "/p.jpg"          # le chemin reste en dernier


def test_a_supplied_raw_avoids_a_second_read():
    calls = []
    def runner(args):
        calls.append(args)
        return "1 image files updated"

    T.write_tags("/p.jpg", [["media_restorer", "Tri", "A"]],
                 owns=T.branch_owner("Tri"), runner=runner, raw={})

    assert len(calls) == 1                   # l'écriture seule, pas de lecture


# ---------------------------------------------------------------------------
# Syntaxe « structure » d'exiftool
# ---------------------------------------------------------------------------

def test_struct_serialisation_nests_and_escapes():
    struct = {"Liste": [{"Nom": "Oeil, gauche"}], "N": 2}

    assert T.to_struct(struct) == "{Liste=[{Nom=Oeil|, gauche}],N=2}"


@pytest.mark.parametrize("brut,attendu", [
    ("a,b", "a|,b"), ("a=b", "a|=b"), ("a|b", "a||b"), ("{x}", "|{x|}"),
])
def test_every_structure_character_is_escaped(brut, attendu):
    assert T.escape_struct(brut) == attendu


# ---------------------------------------------------------------------------
# LE test : deux écrivains cohabitent
# ---------------------------------------------------------------------------

def test_two_owners_never_erase_each_other():
    """Repères et pré-classement partagent les six champs sans se détruire.

    Chaque écriture reconstruit les champs **en entier** : c'est le prédicat
    d'appartenance restreint, et lui seul, qui empêche le second écrivain
    d'effacer le premier.  Régression silencieuse s'il venait à s'élargir.
    """
    store: dict[str, list[str]] = {}

    def runner(args):
        assignations = [a for a in args if a.startswith("-") and "=" in a]
        if not assignations:                                   # lecture
            return _exiftool_json(**{k: v for k, v in store.items()})
        nouveau: dict[str, list[str]] = {}
        for a in assignations:
            tag, _, valeur = a[1:].partition("=")
            nouveau.setdefault(tag.split(":")[-1], []).append(valeur)
        store.update({k: [v for v in vs if v] for k, vs in nouveau.items()})
        return "1 image files updated"

    repere = T.branch_owner("Repère")
    tri = T.branch_owner("Tri")

    T.write_tags("/p.jpg", [["media_restorer", "Repère", "Left Eye"]],
                 owns=repere, runner=runner)
    T.write_tags("/p.jpg", [["media_restorer", "Tri", "Orientation", "Portrait"]],
                 owns=tri, runner=runner)

    raw = T.read_raw("/p.jpg", runner)
    assert T.read_tag_paths(raw, repere) == [["media_restorer", "Repère", "Left Eye"]]
    assert T.read_tag_paths(raw, tri) == [["media_restorer", "Tri", "Orientation", "Portrait"]]


# ---------------------------------------------------------------------------
# Intégration — vrai exiftool
# ---------------------------------------------------------------------------

@pytest.mark.skipif(shutil.which("exiftool") is None or cv2 is None,
                    reason="exiftool ou cv2 absent")
def test_real_exiftool_keeps_both_writers_and_the_user_tags(tmp_path):
    from media_restorer.landmarks import LandmarkSet

    img = tmp_path / "photo.jpg"
    cv2.imwrite(str(img), np.full((300, 400, 3), 190, np.uint8))
    tri = T.branch_owner("Tri")
    tri_paths = [["media_restorer", "Tri", "Orientation", "Paysage"]]

    T.write_tags(img, [["Vacances", "Bretagne"]], owns=T.branch_owner("Vacances"))
    T.write_tags(img, tri_paths, owns=tri)
    LandmarkSet(points={"Left Eye": (34.2, 51.8), "Nose": None},
                source="auto").write_to_metadata(img)

    # Les repères ont écrit en dernier : le tri et l'utilisateur survivent.
    raw = T.read_raw(img, T.default_runner)
    assert T.read_tag_paths(raw, tri) == tri_paths
    assert "Vacances/Bretagne" in T.as_list(raw.get("TagsList"))
    assert LandmarkSet.read_from_metadata(img).points["Nose"] is None

    # Et réciproquement : le tri réécrit sans effacer les repères.
    T.write_tags(img, tri_paths, owns=tri)
    releve = LandmarkSet.read_from_metadata(img)
    assert releve.points["Left Eye"] == pytest.approx((34.2, 51.8))
    assert releve.source == "auto"

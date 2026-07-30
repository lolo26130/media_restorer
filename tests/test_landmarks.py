"""Tests du cœur de stockage des repères (media_restorer.landmarks).

Sans Qt.  Le lanceur ``exiftool`` est toujours injecté (un faux) : aucun test
ne dépend du vrai binaire ni n'écrit sur disque.
"""
import json

from media_restorer.landmarks import LandmarkSet


# ---------------------------------------------------------------------------
# Sérialisation JSON (aller-retour)
# ---------------------------------------------------------------------------

def test_to_json_then_from_json_round_trips_points_and_skips():
    original = LandmarkSet(points={"Left Eye": (10, 20), "Nose": None, "Right Eye": (30, 25)})

    restored = LandmarkSet.from_json(original.to_json())

    assert restored is not None
    assert restored.points == {"Left Eye": (10, 20), "Nose": None, "Right Eye": (30, 25)}


def test_from_json_preserves_insertion_order():
    original = LandmarkSet(points={"C": (1, 1), "A": (2, 2), "B": (3, 3)})

    restored = LandmarkSet.from_json(original.to_json())

    assert list(restored.points) == ["C", "A", "B"]


def test_from_json_returns_none_for_unrelated_comment():
    """Un UserComment sans rapport n'est jamais pris pour nos repères."""
    assert LandmarkSet.from_json("just a plain caption") is None
    assert LandmarkSet.from_json(json.dumps({"something_else": 1})) is None


def test_from_json_ignores_malformed_coordinates_without_crashing():
    raw = json.dumps({"media_restorer_landmarks": {"Ok": [1, 2], "Bad": [1, 2, 3], "Weird": "x"}})

    restored = LandmarkSet.from_json(raw)

    assert restored is not None
    assert restored.points == {"Ok": (1, 2)}


# ---------------------------------------------------------------------------
# Écriture via un runner injecté
# ---------------------------------------------------------------------------

def test_write_to_metadata_calls_runner_with_usercomment_tag(tmp_path):
    calls = []
    landmarks = LandmarkSet(points={"Left Eye": (10, 20), "Nose": None})
    img = tmp_path / "photo.jpg"

    landmarks.write_to_metadata(img, runner=lambda args: calls.append(args) or "")

    assert len(calls) == 1
    args = calls[0]
    assert str(img) in args
    tag_arg = next(a for a in args if a.startswith("-UserComment="))
    payload = json.loads(tag_arg[len("-UserComment="):])
    assert payload == {"media_restorer_landmarks": {"Left Eye": [10, 20], "Nose": None}}


# ---------------------------------------------------------------------------
# Lecture via un runner injecté
# ---------------------------------------------------------------------------

def _exiftool_json(user_comment):
    """Reproduit la forme de sortie de ``exiftool -UserComment -j``."""
    record = {"SourceFile": "x.jpg"}
    if user_comment is not None:
        record["UserComment"] = user_comment
    return json.dumps([record])


def test_read_from_metadata_parses_our_schema(tmp_path):
    stored = LandmarkSet(points={"Left Eye": (10, 20), "Nose": None}).to_json()
    runner = lambda args: _exiftool_json(stored)

    result = LandmarkSet.read_from_metadata(tmp_path / "p.jpg", runner=runner)

    assert result.points == {"Left Eye": (10, 20), "Nose": None}


def test_read_from_metadata_returns_empty_when_no_usercomment(tmp_path):
    runner = lambda args: _exiftool_json(None)

    result = LandmarkSet.read_from_metadata(tmp_path / "p.jpg", runner=runner)

    assert result.points == {}


def test_read_from_metadata_returns_empty_for_unrelated_comment(tmp_path):
    runner = lambda args: _exiftool_json("a caption from another program")

    result = LandmarkSet.read_from_metadata(tmp_path / "p.jpg", runner=runner)

    assert result.points == {}


def test_read_from_metadata_returns_empty_on_empty_exiftool_output(tmp_path):
    runner = lambda args: ""

    result = LandmarkSet.read_from_metadata(tmp_path / "p.jpg", runner=runner)

    assert result.points == {}


def test_write_then_read_round_trip_through_a_fake_metadata_store(tmp_path):
    """Aller-retour complet écriture→lecture sans toucher au disque réel."""
    store: dict[str, str] = {}

    def runner(args):
        # écriture : « -UserComment=... <path> »
        tag = next((a for a in args if a.startswith("-UserComment=")), None)
        if tag is not None:
            store["UserComment"] = tag[len("-UserComment="):]
            return "1 image files updated"
        # lecture : « -UserComment -j <path> »
        return _exiftool_json(store.get("UserComment"))

    LandmarkSet(points={"Left Eye": (5, 6), "Right Eye": (7, 8), "Mouth": None}).write_to_metadata(
        tmp_path / "p.jpg", runner=runner
    )
    result = LandmarkSet.read_from_metadata(tmp_path / "p.jpg", runner=runner)

    assert result.points == {"Left Eye": (5, 6), "Right Eye": (7, 8), "Mouth": None}

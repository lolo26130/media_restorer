"""Tests de la config persistante des repères (extension Manual Mouse Points).

Sans Qt : les fonctions reçoivent un chemin explicite (``path=``), donc ne
touchent ni à ``QSettings`` ni à la vraie configuration de l'utilisateur.
"""
import tomllib

from media_restorer import landmark_config as cfg

_DEFAULT = ["Left Eye", "Right Eye", "Nose"]


def test_save_then_load_round_trips(tmp_path):
    path = tmp_path / "cfg.toml"
    cfg.save_labels(["Left Eye", "Right Eye", "Mouth", "Chin"], path=path)

    assert cfg.load_labels(_DEFAULT, path=path) == ["Left Eye", "Right Eye", "Mouth", "Chin"]


def test_saved_file_is_valid_toml(tmp_path):
    path = tmp_path / "cfg.toml"
    cfg.save_labels(['A label with "quotes"', "Nez"], path=path)

    with open(path, "rb") as f:
        data = tomllib.load(f)  # ne doit pas lever
    assert data["labels"] == ['A label with "quotes"', "Nez"]


def test_load_returns_default_when_file_absent(tmp_path):
    assert cfg.load_labels(_DEFAULT, path=tmp_path / "does-not-exist.toml") == _DEFAULT


def test_load_returns_default_on_malformed_toml(tmp_path):
    path = tmp_path / "cfg.toml"
    path.write_text("labels = [this is not valid toml", encoding="utf-8")

    assert cfg.load_labels(_DEFAULT, path=path) == _DEFAULT


def test_load_returns_default_when_labels_wrong_type(tmp_path):
    path = tmp_path / "cfg.toml"
    path.write_text('labels = 42\n', encoding="utf-8")

    assert cfg.load_labels(_DEFAULT, path=path) == _DEFAULT


def test_load_ignores_blank_and_non_string_entries(tmp_path):
    path = tmp_path / "cfg.toml"
    path.write_text('labels = ["Nez", "   ", "Bouche"]\n', encoding="utf-8")

    assert cfg.load_labels(_DEFAULT, path=path) == ["Nez", "Bouche"]


def test_save_strips_and_deduplicates(tmp_path):
    path = tmp_path / "cfg.toml"
    cfg.save_labels(["  Nez  ", "Nez", "", "Bouche"], path=path)

    assert cfg.load_labels(_DEFAULT, path=path) == ["Nez", "Bouche"]


def test_save_creates_missing_parent_directory(tmp_path):
    path = tmp_path / "sub" / "dir" / "cfg.toml"
    cfg.save_labels(["Nez"], path=path)

    assert path.exists()
    assert cfg.load_labels(_DEFAULT, path=path) == ["Nez"]

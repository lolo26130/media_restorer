"""Tests de la config du modèle (extension Auto Face ID Register).

Sans Qt : chemin explicite (``path=``), aucune écriture dans la vraie config.
"""
import tomllib

from media_restorer.extensions.auto_face_id_register import config as cfg


def test_save_then_load_model_round_trips(tmp_path):
    path = tmp_path / "cfg.toml"
    cfg.save_model("google/owlv2-base-patch16-ensemble", path=path)

    assert cfg.load_model(path=path) == "google/owlv2-base-patch16-ensemble"


def test_saved_model_file_is_valid_toml(tmp_path):
    path = tmp_path / "cfg.toml"
    cfg.save_model("org/model-name", path=path)

    with open(path, "rb") as f:
        data = tomllib.load(f)  # ne doit pas lever
    assert data["detection_model"] == "org/model-name"


def test_load_model_returns_none_when_absent(tmp_path):
    assert cfg.load_model(path=tmp_path / "nope.toml") is None


def test_load_model_returns_none_on_malformed_toml(tmp_path):
    path = tmp_path / "cfg.toml"
    path.write_text("detection_model = [broken", encoding="utf-8")

    assert cfg.load_model(path=path) is None


def test_load_model_returns_none_when_key_wrong_type(tmp_path):
    path = tmp_path / "cfg.toml"
    path.write_text("detection_model = 123\n", encoding="utf-8")

    assert cfg.load_model(path=path) is None


# ---------------------------------------------------------------------------
# Appareil de détection (cpu / gpu)
# ---------------------------------------------------------------------------

def test_device_defaults_to_cpu_when_absent(tmp_path):
    assert cfg.load_device(path=tmp_path / "nope.toml") == "cpu"


def test_save_then_load_device_round_trips(tmp_path):
    path = tmp_path / "cfg.toml"
    cfg.save_device("gpu", path=path)

    assert cfg.load_device(path=path) == "gpu"


def test_invalid_device_falls_back_to_cpu_on_save(tmp_path):
    path = tmp_path / "cfg.toml"
    cfg.save_device("quantum", path=path)

    assert cfg.load_device(path=path) == "cpu"


def test_invalid_device_falls_back_to_cpu_on_load(tmp_path):
    path = tmp_path / "cfg.toml"
    path.write_text('detection_device = "tpu"\n', encoding="utf-8")

    assert cfg.load_device(path=path) == "cpu"


def test_saving_model_preserves_device_and_vice_versa(tmp_path):
    path = tmp_path / "cfg.toml"
    cfg.save_model("org/m", path=path)
    cfg.save_device("gpu", path=path)          # ne doit pas effacer le modèle

    assert cfg.load_model(path=path) == "org/m"
    assert cfg.load_device(path=path) == "gpu"

    cfg.save_model("org/m2", path=path)        # ne doit pas effacer l'appareil
    assert cfg.load_device(path=path) == "gpu"
    assert cfg.load_model(path=path) == "org/m2"

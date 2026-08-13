"""Configuration persistante propre à l'extension Signatures.

Un **unique** fichier TOML — ``signatures.toml`` — dans le même répertoire de
configuration que le reste de l'application (voir
:func:`~media_restorer.app_settings.app_settings`), séparé de celui de
:mod:`~media_restorer.extensions.auto_face_id_register.config` bien que les
deux persistent un couple ``(modèle, device)`` — chaque extension reste
responsable de sa propre configuration, jamais couplée à une autre.

Deux modèles ``transformers`` distincts, deux couples de réglages
--------------------------------------------------------------------
- **localisation** (où est la signature sur le dessin) : un détecteur
  zero-shot OWL-ViT, comme
  :mod:`~media_restorer.extensions.auto_face_id_register` — mêmes clés,
  même prudence ROCm (``cpu`` par défaut) ;
- **comparaison** (à quel dessinateur elle ressemble) : un modèle
  d'empreinte de :mod:`~media_restorer.engines.duplicates.embeddings`,
  ``siglip_base`` par défaut — voir
  :mod:`~media_restorer.engines.signatures.descriptors` pour la mesure qui
  justifie ce choix, différent de celui de ``doublons``.

Comme :mod:`media_restorer.landmark_config` et
:mod:`~media_restorer.extensions.auto_face_id_register.config` : lecture via
:mod:`tomllib` (standard), écriture à la main, chemin dérivé de
``QSettings`` — donc isolé en test par la même redirection.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

from media_restorer.app_settings import app_settings
from media_restorer.engines.duplicates.device import DEVICE_CPU, DEVICE_ORDER
from media_restorer.engines.face_id.detect import CANDIDATE_MODELS
from media_restorer.engines.signatures.descriptors import DEFAULT_MODEL as _DEFAULT_EMBEDDING_MODEL

_CONFIG_FILENAME = "signatures.toml"

_DETECTION_MODEL_KEY = "detection_model"
_DETECTION_DEVICE_KEY = "detection_device"
_EMBEDDING_MODEL_KEY = "embedding_model"
_EMBEDDING_DEVICE_KEY = "embedding_device"

_DEFAULT_DETECTION_MODEL = CANDIDATE_MODELS[0]


def config_path() -> Path:
    """Chemin du fichier de configuration unique de l'extension."""
    return Path(app_settings().fileName()).with_name(_CONFIG_FILENAME)


def _read_all(path: Path) -> dict:
    """Contenu brut du TOML, ou ``{}`` si absent/corrompu (ne lève jamais)."""
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_key(path: Path, key: str, value: str) -> None:
    """Réécrit *key* dans le TOML, en préservant les autres clés déjà présentes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _read_all(path)
    data[key] = value
    lines = ["# Configuration de l'extension Signatures (media_restorer)."]
    for k, v in data.items():
        if isinstance(v, str):
            escaped = v.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'{k} = "{escaped}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_detection_model(*, path: Path | None = None) -> str:
    value = _read_all(path or config_path()).get(_DETECTION_MODEL_KEY)
    return value if isinstance(value, str) and value.strip() else _DEFAULT_DETECTION_MODEL


def save_detection_model(model_name: str, *, path: Path | None = None) -> None:
    _write_key(path or config_path(), _DETECTION_MODEL_KEY, model_name)


def load_detection_device(*, path: Path | None = None) -> str:
    value = _read_all(path or config_path()).get(_DETECTION_DEVICE_KEY)
    return value if value in DEVICE_ORDER else DEVICE_CPU


def save_detection_device(device: str, *, path: Path | None = None) -> None:
    if device not in DEVICE_ORDER:
        device = DEVICE_CPU
    _write_key(path or config_path(), _DETECTION_DEVICE_KEY, device)


def load_embedding_model(*, path: Path | None = None) -> str:
    value = _read_all(path or config_path()).get(_EMBEDDING_MODEL_KEY)
    return value if isinstance(value, str) and value.strip() else _DEFAULT_EMBEDDING_MODEL


def save_embedding_model(model_key: str, *, path: Path | None = None) -> None:
    _write_key(path or config_path(), _EMBEDDING_MODEL_KEY, model_key)


def load_embedding_device(*, path: Path | None = None) -> str:
    value = _read_all(path or config_path()).get(_EMBEDDING_DEVICE_KEY)
    return value if value in DEVICE_ORDER else DEVICE_CPU


def save_embedding_device(device: str, *, path: Path | None = None) -> None:
    if device not in DEVICE_ORDER:
        device = DEVICE_CPU
    _write_key(path or config_path(), _EMBEDDING_DEVICE_KEY, device)

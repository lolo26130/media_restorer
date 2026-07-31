"""Configuration persistante propre à l'extension Auto Face ID Register.

Un **unique** fichier TOML — ``auto_face_id_register.toml`` — dans le même
répertoire de configuration que le reste de l'application (voir
:func:`~media_restorer.app_settings.app_settings`).  N'y sont stockés que les
réglages propres à *cette* extension :

- ``detection_model`` : le modèle de détection choisi au premier usage ;
- ``detection_device`` : ``"cpu"`` (défaut) ou ``"gpu"``.

Pourquoi ``cpu`` par défaut
---------------------------
Le chargement du modèle **vers le GPU ROCm** de la Radeon 780M (gfx1103) s'est
révélé pouvoir **se figer** (initialisation HSA bloquée dans un process frais).
Comme la détection est une action ponctuelle tournant sur une image **réduite**
(``max_side`` de :func:`~media_restorer.engines.face_id.detect.detect_landmarks`),
le CPU est assez rapide et surtout fiable ; le GPU reste proposé en option pour
qui veut l'essayer.

La liste des **repères**, elle, est **partagée** et vit dans
:mod:`media_restorer.landmark_config` (mêmes repères que la désignation
manuelle).

Comme :mod:`media_restorer.landmark_config` : lecture via :mod:`tomllib`
(standard), écriture à la main (aucun écrivain TOML présent), chemin dérivé de
``QSettings`` — donc isolé en test par la même redirection.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

from media_restorer.app_settings import app_settings

_CONFIG_FILENAME = "auto_face_id_register.toml"
_MODEL_KEY = "detection_model"
_DEVICE_KEY = "detection_device"

DEVICES = ("cpu", "gpu")
_DEFAULT_DEVICE = "cpu"


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


def _write_all(path: Path, model: str | None, device: str | None) -> None:
    """Réécrit le fichier avec les clés fournies (les ``None`` sont omis)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Configuration de l'extension Auto Face ID Register (media_restorer)."]
    for key, value in ((_MODEL_KEY, model), (_DEVICE_KEY, device)):
        if value is not None:
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'{key} = "{escaped}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_model(*, path: Path | None = None) -> str | None:
    """Nom du modèle enregistré, ou ``None`` si aucun (l'extension proposera d'en choisir un)."""
    value = _read_all(path or config_path()).get(_MODEL_KEY)
    return value if isinstance(value, str) and value.strip() else None


def save_model(model_name: str, *, path: Path | None = None) -> None:
    """Enregistre *model_name*, en conservant le choix d'appareil existant."""
    path = path or config_path()
    _write_all(path, model_name, _read_all(path).get(_DEVICE_KEY))


def load_device(*, path: Path | None = None) -> str:
    """Appareil de détection enregistré (``"cpu"``/``"gpu"``), ``"cpu"`` par défaut."""
    value = _read_all(path or config_path()).get(_DEVICE_KEY)
    return value if value in DEVICES else _DEFAULT_DEVICE


def save_device(device: str, *, path: Path | None = None) -> None:
    """Enregistre *device*, en conservant le modèle existant."""
    if device not in DEVICES:
        device = _DEFAULT_DEVICE
    path = path or config_path()
    _write_all(path, _read_all(path).get(_MODEL_KEY), device)

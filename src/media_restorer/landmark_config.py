"""Configuration persistante partagée de la liste des repères.

Un **unique** fichier TOML — ``manual_mouse_points.toml`` — dans le répertoire
de configuration de l'application (le même que celui du ``QSettings`` partagé,
voir :func:`~media_restorer.app_settings.app_settings`) : la liste des repères
proposés (« Left Eye », « Nose »… plus ceux que l'utilisateur ajoute).

Module de cœur (hors extension) : la liste est **partagée** entre l'extension
:mod:`~media_restorer.extensions.manual_mouse_points` (qui la propose au
pointage manuel) et :mod:`~media_restorer.extensions.auto_face_id_register`
(qui l'utilise comme requêtes de détection) — d'où sa place ici plutôt que dans
l'une des deux, pour qu'elles ne dépendent pas l'une de l'autre.  Le nom de
fichier reste ``manual_mouse_points.toml`` (l'extension historique) pour ne pas
casser une configuration déjà écrite.

Pourquoi un TOML écrit à la main
--------------------------------
La lecture se fait avec :mod:`tomllib` (bibliothèque standard depuis Python
3.11).  Aucun écrivain TOML n'est présent dans l'environnement (ni ``tomli_w``
ni ``toml``) : comme le contenu se réduit à une liste de chaînes, on l'écrit à
la main, chaque libellé étant échappé via :func:`json.dumps` (une chaîne JSON
valide est aussi une chaîne « basic » TOML valide).  Cela évite d'ajouter une
dépendance pour un fichier aussi trivial.

Chemin et isolation des tests
-----------------------------
Le chemin dérive de :func:`~media_restorer.app_settings.app_settings` :
``…/media_restorer/manual_mouse_points.toml``, à côté du ``.ini`` du
``QSettings``.  Les tests redirigent déjà ``QSettings`` vers un dossier jetable
(fixture ``_isolated_qsettings`` de ``tests/conftest.py``), si bien que ce
fichier de configuration y est automatiquement isolé lui aussi — aucun test
n'écrit dans la vraie configuration de l'utilisateur.
"""
from __future__ import annotations

import json
import tomllib
from pathlib import Path

from media_restorer.app_settings import app_settings

_CONFIG_FILENAME = "manual_mouse_points.toml"


def config_path() -> Path:
    """Chemin du fichier de configuration unique de l'extension.

    Placé dans le répertoire de configuration de l'application (celui du
    ``QSettings`` partagé), donc isolé en test via la même redirection.
    """
    return Path(app_settings().fileName()).with_name(_CONFIG_FILENAME)


def load_labels(default: list[str], *, path: Path | None = None) -> list[str]:
    """Liste des repères enregistrée, ou *default* si le fichier est absent/illisible.

    Ne lève jamais : une configuration absente, corrompue ou d'un type
    inattendu retombe silencieusement sur *default* — l'extension doit rester
    utilisable même si le fichier a été édité à la main de travers.
    """
    path = path or config_path()
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return list(default)
    labels = data.get("labels")
    if not isinstance(labels, list):
        return list(default)
    cleaned = [item.strip() for item in labels if isinstance(item, str) and item.strip()]
    return cleaned or list(default)


def save_labels(labels: list[str], *, path: Path | None = None) -> None:
    """Écrit *labels* dans le fichier de configuration (créé au besoin).

    Les libellés vides/espaces sont ignorés ; les doublons sont dédupliqués en
    conservant le premier ordre d'apparition.
    """
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    cleaned: list[str] = []
    for item in labels:
        item = item.strip()
        if item and item not in seen:
            seen.add(item)
            cleaned.append(item)

    lines = [
        "# Repères de l'extension Manual Mouse Points (media_restorer).",
        "# Modifiable à la main ou via le panneau de paramètres de l'extension.",
        "labels = [",
    ]
    lines += [f"    {json.dumps(label, ensure_ascii=False)}," for label in cleaned]
    lines.append("]")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

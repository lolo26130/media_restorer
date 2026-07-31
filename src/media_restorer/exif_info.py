"""Lecture des métadonnées d'image pour le dock « Infos, Exif ».

Cœur sans Qt (comme :mod:`media_restorer.landmarks` / :mod:`media_restorer.image_io`),
alimentant :class:`~media_restorer.gui_widgets.InfoExifPanel`.

Un unique lecteur : ``exiftool -j`` via un *runner* injectable (même motif que
:mod:`media_restorer.landmarks`, donc testable sans le binaire), avec **repli
gracieux sur Pillow** si ``exiftool`` est absent ou échoue.  Consulter des infos
ne doit jamais planter — contrairement à l'écriture de repères, une lecture
d'affichage retombe toujours sur un résultat minimal plutôt que de lever.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Callable

# Lance ``exiftool`` avec *args* (sans l'exécutable) et renvoie sa sortie
# standard.  Injectable pour les tests — voir :func:`_default_runner`.
ExiftoolRunner = Callable[[list[str]], str]

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}

# Tags exiftool affichés en tête, dans cet ordre, s'ils existent ; libellé
# lisible en valeur.  Le reste des tags exiftool suit, tel quel.
_PRIORITY_TAGS: dict[str, str] = {
    "FileName": "Fichier",
    "ImageSize": "Dimensions",
    "FileType": "Format",
    "FileSize": "Taille",
    "XResolution": "Résolution",
    "Orientation": "Orientation",
    "DateTimeOriginal": "Date de prise de vue",
    "Model": "Appareil",
    "LensModel": "Objectif",
}


def _default_runner(args: list[str]) -> str:
    """Lance le vrai binaire ``exiftool`` (les tests injectent un faux).

    Lève :class:`FileNotFoundError` si ``exiftool`` est absent — capté par
    :func:`read_image_info`, qui bascule alors sur le repli Pillow.
    """
    exe = shutil.which("exiftool")
    if exe is None:
        raise FileNotFoundError("exiftool introuvable")
    result = subprocess.run(
        [exe, *args], capture_output=True, text=True, check=True, timeout=30
    )
    return result.stdout


def read_image_info(path: Path | str, *, runner: ExiftoolRunner | None = None) -> dict[str, str]:
    """Métadonnées de *path* prêtes à afficher : ``{libellé: valeur}`` ordonné.

    Lit via ``exiftool -j`` ; en cas d'échec (binaire absent, fichier illisible,
    JSON inattendu) retombe sur Pillow.  Ne lève jamais.
    """
    runner = runner or _default_runner
    try:
        records = json.loads(runner(["-j", str(path)]))
        if isinstance(records, list) and records and isinstance(records[0], dict):
            return _format_tags(records[0])
    except Exception:
        pass
    return _pillow_fallback(Path(path))


def _format_tags(tags: dict) -> dict[str, str]:
    """Ordonne les tags exiftool : champs prioritaires d'abord, puis le reste."""
    info: dict[str, str] = {}
    for tag, label in _PRIORITY_TAGS.items():
        if tag in tags:
            info[label] = str(tags[tag])
    for tag, value in tags.items():
        if tag in _PRIORITY_TAGS or tag == "SourceFile":
            continue
        info[tag] = str(value)
    return info


def _pillow_fallback(path: Path) -> dict[str, str]:
    """Champs de base via Pillow quand exiftool n'est pas disponible."""
    info: dict[str, str] = {"Fichier": path.name}
    try:
        from PIL import Image
        with Image.open(path) as im:
            info["Dimensions"] = f"{im.width}×{im.height}"
            info["Format"] = im.format or "?"
            info["Mode"] = im.mode
    except Exception:
        info["Métadonnées"] = "illisibles (exiftool absent, Pillow a échoué)"
    try:
        info["Taille"] = f"{path.stat().st_size / 1e6:.2f} Mo"
    except OSError:
        pass
    return info


def read_directory_summary(path: Path | str, *, recursive: bool = False) -> dict[str, str]:
    """Résumé d'un répertoire cible : chemin, mode, nombre d'images, taille totale.

    Ne lève jamais : un fichier illisible est simplement ignoré du décompte de
    taille.
    """
    directory = Path(path)
    candidates = directory.rglob("*") if recursive else directory.iterdir()
    images = [p for p in candidates if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES]
    total = 0
    for p in images:
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return {
        "Répertoire": str(directory),
        "Mode": "récursif" if recursive else "non récursif",
        "Images": str(len(images)),
        "Taille totale": f"{total / 1e6:.1f} Mo",
    }

"""Lecture des métadonnées d'image pour le dock « Infos, Exif ».

Cœur sans Qt (comme :mod:`media_restorer.landmarks` / :mod:`media_restorer.image_io`),
alimentant :class:`~media_restorer.gui_widgets.InfoExifPanel`.

Les métadonnées sont renvoyées **hiérarchisées par provenance** :
``{groupe: {tag: valeur}}`` — les groupes sont les familles exiftool (``File``,
``EXIF``, ``XMP``, ``MakerNotes``, ``Composite``…), obtenues via ``exiftool -g``.
Le dock les affiche en arbre repliable plutôt qu'en longue liste plate.

Un unique lecteur : ``exiftool -g -j`` via un *runner* injectable (même motif que
:mod:`media_restorer.landmarks`, donc testable sans le binaire), avec **repli
gracieux sur Pillow** si ``exiftool`` est absent ou échoue.  Consulter des infos
ne doit jamais planter — une lecture d'affichage retombe toujours sur un
résultat minimal plutôt que de lever.
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

# Un jeu de métadonnées hiérarchisé : groupe de provenance → {tag: valeur}.
GroupedInfo = dict[str, dict[str, str]]

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}

# Ordre d'affichage des groupes de provenance : les plus utiles d'abord, puis
# le reste dans l'ordre d'exiftool.  ``ExifTool`` (version de l'outil) est
# renvoyé en dernier — accessoire.
_PRIORITY_GROUPS = ("File", "EXIF", "Composite", "XMP", "IPTC", "MakerNotes")
_DEPRIORITISED_GROUPS = ("ExifTool",)


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


def read_image_info(path: Path | str, *, runner: ExiftoolRunner | None = None) -> GroupedInfo:
    """Métadonnées de *path*, hiérarchisées par provenance : ``{groupe: {tag: valeur}}``.

    Lit via ``exiftool -g -j`` (regroupement par famille) ; en cas d'échec
    (binaire absent, fichier illisible, JSON inattendu) retombe sur Pillow.
    Ne lève jamais.
    """
    runner = runner or _default_runner
    try:
        records = json.loads(runner(["-g", "-j", str(path)]))
        if isinstance(records, list) and records and isinstance(records[0], dict):
            return _format_grouped(records[0])
    except Exception:
        pass
    return _pillow_fallback(Path(path))


def _format_grouped(record: dict) -> GroupedInfo:
    """Ordonne les groupes exiftool (prioritaires d'abord) et stringifie les valeurs.

    *record* est la sortie ``exiftool -g -j`` : ``{"SourceFile": str, "File":
    {...}, "EXIF": {...}, ...}``.  ``SourceFile`` (chaîne, pas un groupe) est
    ignoré ; chaque autre entrée dict est un groupe de provenance.
    """
    groups = {k: v for k, v in record.items() if isinstance(v, dict)}
    ordered: GroupedInfo = {}
    for name in _PRIORITY_GROUPS:
        if name in groups:
            ordered[name] = _stringify(groups.pop(name))
    deferred = {name: groups.pop(name) for name in _DEPRIORITISED_GROUPS if name in groups}
    for name, tags in groups.items():          # groupes restants, ordre exiftool
        ordered[name] = _stringify(tags)
    for name, tags in deferred.items():         # accessoires en dernier
        ordered[name] = _stringify(tags)
    return ordered


def _stringify(tags: dict) -> dict[str, str]:
    return {str(tag): str(value) for tag, value in tags.items()}


def _pillow_fallback(path: Path) -> GroupedInfo:
    """Champs de base via Pillow quand exiftool n'est pas disponible.

    Un seul groupe « Image » — même forme hiérarchisée que le chemin exiftool.
    """
    fields: dict[str, str] = {"Fichier": path.name}
    try:
        from PIL import Image
        with Image.open(path) as im:
            fields["Dimensions"] = f"{im.width}×{im.height}"
            fields["Format"] = im.format or "?"
            fields["Mode"] = im.mode
    except Exception:
        fields["Métadonnées"] = "illisibles (exiftool absent, Pillow a échoué)"
    try:
        fields["Taille"] = f"{path.stat().st_size / 1e6:.2f} Mo"
    except OSError:
        pass
    return {"Image": fields}


def read_directory_summary(path: Path | str, *, recursive: bool = False) -> GroupedInfo:
    """Résumé d'un répertoire cible, sous la même forme hiérarchisée (un groupe).

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
        "Répertoire": {
            "Chemin": str(directory),
            "Mode": "récursif" if recursive else "non récursif",
            "Images": str(len(images)),
            "Taille totale": f"{total / 1e6:.1f} Mo",
        }
    }

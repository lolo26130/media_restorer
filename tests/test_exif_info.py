"""Tests du lecteur de métadonnées (media_restorer.exif_info), sans Qt.

Le runner ``exiftool`` est toujours injecté : aucun test ne lance le vrai
binaire.  Le repli Pillow est exercé en faisant échouer le runner.  Les
métadonnées sont hiérarchisées par provenance : ``{groupe: {tag: valeur}}``.
"""
import json

import cv2
import numpy as np

from media_restorer import exif_info


def _exiftool_g_json(**groups):
    """Reproduit la sortie ``exiftool -g -j`` : SourceFile + groupes imbriqués."""
    record = {"SourceFile": "x.jpg", **groups}
    return json.dumps([record])


# ---------------------------------------------------------------------------
# Lecture via exiftool -g (runner injecté) — hiérarchie par provenance
# ---------------------------------------------------------------------------

def test_groups_are_nested_by_provenance():
    runner = lambda args: _exiftool_g_json(
        File={"FileName": "DSC.JPG"}, EXIF={"Model": "NIKON D800"}
    )

    info = exif_info.read_image_info("x.jpg", runner=runner)

    assert info["File"] == {"FileName": "DSC.JPG"}
    assert info["EXIF"] == {"Model": "NIKON D800"}


def test_priority_groups_come_first():
    # exiftool renvoie ExifTool/JFIF avant File/EXIF ; on doit réordonner.
    runner = lambda args: _exiftool_g_json(
        ExifTool={"ExifToolVersion": "12.76"},
        JFIF={"JFIFVersion": "1.01"},
        File={"FileName": "a.jpg"},
        EXIF={"Model": "X"},
    )

    info = exif_info.read_image_info("x.jpg", runner=runner)

    keys = list(info)
    assert keys[0] == "File"
    assert keys[1] == "EXIF"
    assert keys[-1] == "ExifTool"  # accessoire, rejeté en fin


def test_sourcefile_is_not_a_group():
    runner = lambda args: _exiftool_g_json(File={"FileName": "a.jpg"})

    info = exif_info.read_image_info("x.jpg", runner=runner)

    assert "SourceFile" not in info


def test_values_are_stringified():
    runner = lambda args: _exiftool_g_json(Composite={"Megapixels": 0.002})

    info = exif_info.read_image_info("x.jpg", runner=runner)

    assert info["Composite"]["Megapixels"] == "0.002"


def test_runner_receives_group_and_json_flags():
    seen = []
    runner = lambda args: seen.append(args) or _exiftool_g_json(File={"FileName": "a.jpg"})

    exif_info.read_image_info("/some/photo.jpg", runner=runner)

    assert "-g" in seen[0]
    assert "-j" in seen[0]
    assert "/some/photo.jpg" in seen[0]


# ---------------------------------------------------------------------------
# Repli Pillow (exiftool absent / en échec) — un seul groupe « Image »
# ---------------------------------------------------------------------------

def test_falls_back_to_pillow_when_runner_raises(tmp_path):
    img = tmp_path / "photo.png"
    cv2.imwrite(str(img), np.full((30, 40, 3), 128, np.uint8))

    def boom(args):
        raise FileNotFoundError("exiftool introuvable")

    info = exif_info.read_image_info(img, runner=boom)

    assert set(info) == {"Image"}
    assert info["Image"]["Fichier"] == "photo.png"
    assert info["Image"]["Dimensions"] == "40×30"  # largeur×hauteur
    assert info["Image"]["Format"] == "PNG"


def test_fallback_never_raises_on_unreadable_file(tmp_path):
    missing = tmp_path / "nope.jpg"

    def boom(args):
        raise FileNotFoundError

    info = exif_info.read_image_info(missing, runner=boom)

    assert info["Image"]["Fichier"] == "nope.jpg"  # au minimum le nom, sans lever


def test_malformed_exiftool_output_falls_back(tmp_path):
    img = tmp_path / "photo.png"
    cv2.imwrite(str(img), np.full((10, 10, 3), 200, np.uint8))
    runner = lambda args: "not json at all"

    info = exif_info.read_image_info(img, runner=runner)

    assert info["Image"]["Fichier"] == "photo.png"  # repli déclenché


# ---------------------------------------------------------------------------
# Résumé de répertoire — même forme hiérarchisée (groupe « Répertoire »)
# ---------------------------------------------------------------------------

def test_directory_summary_counts_images_only(tmp_path):
    for name in ("a.jpg", "b.png", "c.txt", "notes.md"):
        (tmp_path / name).write_bytes(b"x")

    summary = exif_info.read_directory_summary(tmp_path)

    assert set(summary) == {"Répertoire"}
    fields = summary["Répertoire"]
    assert fields["Images"] == "2"  # a.jpg + b.png, pas les .txt/.md
    assert fields["Mode"] == "non récursif"
    assert str(tmp_path) in fields["Chemin"]


def test_directory_summary_recursive_descends(tmp_path):
    (tmp_path / "top.jpg").write_bytes(b"x")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "deep.jpg").write_bytes(b"x")

    flat = exif_info.read_directory_summary(tmp_path, recursive=False)
    deep = exif_info.read_directory_summary(tmp_path, recursive=True)

    assert flat["Répertoire"]["Images"] == "1"
    assert deep["Répertoire"]["Images"] == "2"
    assert deep["Répertoire"]["Mode"] == "récursif"

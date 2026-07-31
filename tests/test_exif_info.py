"""Tests du lecteur de métadonnées (media_restorer.exif_info), sans Qt.

Le runner ``exiftool`` est toujours injecté : aucun test ne lance le vrai
binaire.  Le repli Pillow est exercé en faisant échouer le runner.
"""
import json

import cv2
import numpy as np

from media_restorer import exif_info


def _exiftool_json(**tags):
    record = {"SourceFile": "x.jpg", **tags}
    return json.dumps([record])


# ---------------------------------------------------------------------------
# Lecture via exiftool (runner injecté)
# ---------------------------------------------------------------------------

def test_priority_tags_come_first_and_are_relabelled():
    runner = lambda args: _exiftool_json(
        FileName="DSC.JPG", ImageSize="4912x7360", Model="NIKON D800"
    )

    info = exif_info.read_image_info("x.jpg", runner=runner)

    assert info["Fichier"] == "DSC.JPG"
    assert info["Dimensions"] == "4912x7360"
    assert info["Appareil"] == "NIKON D800"
    # les libellés prioritaires connus précèdent les tags bruts
    assert list(info)[0] == "Fichier"


def test_non_priority_tags_are_kept_verbatim():
    runner = lambda args: _exiftool_json(FileName="a.jpg", ColorSpace="sRGB")

    info = exif_info.read_image_info("x.jpg", runner=runner)

    assert info["ColorSpace"] == "sRGB"


def test_sourcefile_is_dropped():
    runner = lambda args: _exiftool_json(FileName="a.jpg")

    info = exif_info.read_image_info("x.jpg", runner=runner)

    assert "SourceFile" not in info.values()
    assert "SourceFile" not in info


def test_runner_receives_json_flag_and_path():
    seen = []
    runner = lambda args: seen.append(args) or _exiftool_json(FileName="a.jpg")

    exif_info.read_image_info("/some/photo.jpg", runner=runner)

    assert "-j" in seen[0]
    assert "/some/photo.jpg" in seen[0]


# ---------------------------------------------------------------------------
# Repli Pillow (exiftool absent / en échec)
# ---------------------------------------------------------------------------

def test_falls_back_to_pillow_when_runner_raises(tmp_path):
    img = tmp_path / "photo.png"
    cv2.imwrite(str(img), np.full((30, 40, 3), 128, np.uint8))

    def boom(args):
        raise FileNotFoundError("exiftool introuvable")

    info = exif_info.read_image_info(img, runner=boom)

    assert info["Fichier"] == "photo.png"
    assert info["Dimensions"] == "40×30"  # largeur×hauteur
    assert info["Format"] == "PNG"


def test_fallback_never_raises_on_unreadable_file(tmp_path):
    missing = tmp_path / "nope.jpg"

    def boom(args):
        raise FileNotFoundError

    info = exif_info.read_image_info(missing, runner=boom)

    assert info["Fichier"] == "nope.jpg"  # au minimum le nom, sans lever


def test_malformed_exiftool_output_falls_back(tmp_path):
    img = tmp_path / "photo.png"
    cv2.imwrite(str(img), np.full((10, 10, 3), 200, np.uint8))
    runner = lambda args: "not json at all"

    info = exif_info.read_image_info(img, runner=runner)

    assert info["Fichier"] == "photo.png"  # repli déclenché


# ---------------------------------------------------------------------------
# Résumé de répertoire
# ---------------------------------------------------------------------------

def test_directory_summary_counts_images_only(tmp_path):
    for name in ("a.jpg", "b.png", "c.txt", "notes.md"):
        (tmp_path / name).write_bytes(b"x")

    summary = exif_info.read_directory_summary(tmp_path)

    assert summary["Images"] == "2"  # a.jpg + b.png, pas les .txt/.md
    assert summary["Mode"] == "non récursif"
    assert str(tmp_path) in summary["Répertoire"]


def test_directory_summary_recursive_descends(tmp_path):
    (tmp_path / "top.jpg").write_bytes(b"x")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "deep.jpg").write_bytes(b"x")

    flat = exif_info.read_directory_summary(tmp_path, recursive=False)
    deep = exif_info.read_directory_summary(tmp_path, recursive=True)

    assert flat["Images"] == "1"
    assert deep["Images"] == "2"
    assert deep["Mode"] == "récursif"

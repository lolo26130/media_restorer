"""Tests du redressement EXIF (media_restorer.image_io).

La table ``_EXIF_TRANSFORMS`` est validée contre ``PIL.ImageOps.exif_transpose``,
implémentation de référence, pour les 8 valeurs du tag Orientation.
"""
import cv2
import numpy as np
import pytest

from media_restorer.image_io import (
    apply_exif_orientation,
    exif_orientation,
    imread_oriented,
)


def _asymmetric_bgr(h=60, w=100) -> np.ndarray:
    """Image BGR asymétrique en x, en y et entre canaux.

    Indispensable ici : une image symétrique ne distinguerait pas une rotation
    d'un miroir, ni la transposée de la transversale.

    Volontairement *lisse* (dégradés) et non bruitée : la même image sert aux
    comparaisons exactes et à un aller-retour JPEG, or du bruit aléatoire est
    le pire cas pour le DCT et le sous-échantillonnage de chrominance — les
    pertes du codec masqueraient ce qu'on veut mesurer.
    """
    ys, xs = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = (xs * 255 // max(w - 1, 1)).astype(np.uint8)   # dégradé en x
    img[:, :, 1] = (ys * 255 // max(h - 1, 1)).astype(np.uint8)   # dégradé en y
    img[:, :, 2] = 128
    img[: h // 6, :, :] = 0                                       # bande en haut
    img[:, : w // 6, :] = 255                                     # bande à gauche
    return img


def _pillow_reference(img_bgr: np.ndarray, orientation: int) -> np.ndarray:
    """Résultat de PIL.ImageOps.exif_transpose, ramené en BGR NumPy."""
    from PIL import Image, ImageOps
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    pil.getexif()[274] = orientation
    return cv2.cvtColor(np.array(ImageOps.exif_transpose(pil)), cv2.COLOR_RGB2BGR)


@pytest.mark.parametrize("orientation", range(1, 9))
def test_apply_exif_orientation_matches_pillow(orientation):
    """Chaque transformation coïncide exactement avec celle de Pillow."""
    img = _asymmetric_bgr()

    result = apply_exif_orientation(img, orientation)

    np.testing.assert_array_equal(result, _pillow_reference(img, orientation))


@pytest.mark.parametrize("orientation, swaps", [
    (1, False), (2, False), (3, False), (4, False),
    (5, True), (6, True), (7, True), (8, True),
])
def test_apply_exif_orientation_swaps_axes_only_when_expected(orientation, swaps):
    """Seules les orientations 5–8 échangent hauteur et largeur."""
    img = _asymmetric_bgr(60, 100)

    h, w = apply_exif_orientation(img, orientation).shape[:2]

    assert (h, w) == ((100, 60) if swaps else (60, 100))


def test_apply_exif_orientation_1_is_identity():
    """L'orientation 1 (cas le plus courant) ne recopie ni ne modifie rien."""
    img = _asymmetric_bgr()

    assert apply_exif_orientation(img, 1) is img


def test_imread_oriented_rotates_a_real_jpeg(tmp_path):
    """Bout en bout : un JPEG Orientation=6 est relu redressé.

    Reproduit le cas des fichiers Nikon de la collection (Orientation = 6,
    rotation de 90°), qui s'affichaient couchés dans l'application.
    """
    from PIL import Image
    img = _asymmetric_bgr(60, 100)                  # paysage sur le disque
    path = tmp_path / "photo.jpg"
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    exif = pil.getexif()
    exif[274] = 6
    pil.save(str(path), "JPEG", exif=exif, quality=100)

    assert exif_orientation(path) == 6
    raw   = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)   # ancien comportement
    fixed = imread_oriented(path)

    assert raw.shape[:2] == (60, 100)               # couché : le bug d'origine
    assert fixed.shape[:2] == (100, 60)             # redressé
    # Aux pertes JPEG près, c'est bien la rotation de l'image d'origine
    expected = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    assert np.abs(fixed.astype(int) - expected.astype(int)).mean() < 3.0
    # Et dans le bon sens : la bande claire était à gauche, elle passe en haut
    # (une rotation anti-horaire l'aurait envoyée en bas).
    assert fixed[:5, :, :].mean() > 250
    assert fixed[-5:, :, :].mean() < 250


def test_imread_oriented_leaves_files_without_exif_untouched(tmp_path):
    """Un PNG sans EXIF est rendu tel quel."""
    img = _asymmetric_bgr()
    path = tmp_path / "sans_exif.png"
    cv2.imwrite(str(path), img)

    assert exif_orientation(path) == 1
    np.testing.assert_array_equal(imread_oriented(path), img)


def test_imread_oriented_preserves_alpha_and_bit_depth(tmp_path):
    """IMREAD_UNCHANGED reste le défaut : alpha et 16 bits sont conservés."""
    rgba = cv2.cvtColor(_asymmetric_bgr(), cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = 128
    path_rgba = tmp_path / "alpha.png"
    cv2.imwrite(str(path_rgba), rgba)

    path_16 = tmp_path / "seize_bits.png"
    cv2.imwrite(str(path_16), (_asymmetric_bgr().astype(np.uint16) * 257))

    assert imread_oriented(path_rgba).shape[2] == 4
    assert imread_oriented(path_16).dtype == np.uint16


def test_imread_oriented_returns_none_on_unreadable_file(tmp_path):
    """Même contrat que cv2.imread : None si le fichier n'est pas une image."""
    path = tmp_path / "pas_une_image.jpg"
    path.write_text("ceci n'est pas une image")

    assert imread_oriented(path) is None


def test_exif_orientation_never_raises(tmp_path):
    """Un fichier illisible ou un tag aberrant retombent sur 1, sans lever."""
    broken = tmp_path / "casse.jpg"
    broken.write_bytes(b"\xff\xd8\xff\xe1 donnees tronquees")

    assert exif_orientation(broken) == 1
    assert exif_orientation(tmp_path / "inexistant.jpg") == 1

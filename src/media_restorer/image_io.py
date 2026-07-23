"""Lecture d'images avec prise en compte de l'orientation EXIF.

Point d'entrée unique : :func:`imread_oriented`.  Tous les chargements
d'images de l'application passent par lui (fenêtre principale, ``restore_file``
des moteurs, 2ᵉ cliché du moteur Double-exposition) afin que les tableaux
NumPy manipulés soient toujours **dans le sens où le boîtier a cadré**.

Pourquoi ce module existe
-------------------------
``cv2.imread`` traite l'orientation EXIF différemment selon le drapeau :

- ``IMREAD_UNCHANGED`` (celui qu'utilisait l'application) l'**ignore** ;
- ``IMREAD_COLOR`` l'**applique**.

Les boîtiers Nikon utilisés pour cette collection écrivent ``Orientation = 6``
(rotation de 90°) : les images s'affichaient donc couchées dans l'application,
et deux chemins de lecture différents pouvaient livrer la même photo dans deux
orientations incompatibles.

Stratégie retenue : lire les pixels bruts (``IMREAD_IGNORE_ORIENTATION``, ce
qui préserve le comportement historique quant à la profondeur de bits et au
canal alpha), lire le tag EXIF séparément via Pillow, puis appliquer la
transformation avec OpenCV.  On ne dépend donc pas du drapeau passé à
``imread``.

Écriture
--------
Les images étant redressées dès la lecture, les tableaux traités puis écrits
par ``cv2.imwrite`` contiennent déjà les pixels dans le bon sens et ne portent
aucun tag EXIF — ce qui est cohérent : un tag ``Orientation`` résiduel
ferait tourner l'image une seconde fois à l'affichage.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# Transformations à appliquer pour chaque valeur du tag EXIF Orientation.
# Références croisées avec ``PIL.ImageOps.exif_transpose`` (cf. tests) :
#   1 rien · 2 miroir H · 3 rotation 180° · 4 miroir V
#   5 transposée · 6 rotation 90° horaire · 7 transversale · 8 rotation 90° anti-horaire
_EXIF_TRANSFORMS = {
    2: lambda a: cv2.flip(a, 1),
    3: lambda a: cv2.rotate(a, cv2.ROTATE_180),
    4: lambda a: cv2.flip(a, 0),
    5: lambda a: cv2.transpose(a),
    6: lambda a: cv2.rotate(a, cv2.ROTATE_90_CLOCKWISE),
    7: lambda a: cv2.flip(cv2.transpose(a), -1),
    8: lambda a: cv2.rotate(a, cv2.ROTATE_90_COUNTERCLOCKWISE),
}

_EXIF_ORIENTATION_TAG = 274


def exif_orientation(path: Path | str) -> int:
    """Valeur du tag EXIF ``Orientation`` de *path*, ou 1 si absent.

    Retourne 1 (« pas de transformation ») pour tout ce qui n'est pas
    exploitable : format sans EXIF (PNG, BMP), fichier illisible par Pillow,
    tag absent ou hors de la plage 1–8.  Ne lève jamais : l'orientation est
    une information accessoire, son absence ne doit pas empêcher le
    chargement de l'image.
    """
    try:
        from PIL import Image
        with Image.open(str(path)) as pil:      # lazy : n'décode que l'en-tête
            value = pil.getexif().get(_EXIF_ORIENTATION_TAG)
    except Exception:
        return 1
    return value if isinstance(value, int) and 1 <= value <= 8 else 1


_EXIF_XRESOLUTION_TAG = 282
_DEFAULT_DPI = 300.0


def exif_dpi(path: Path | str) -> float:
    """Résolution EXIF de *path* en points par pouce, ou 300 si absente.

    Lit uniquement ``XResolution`` (tag 282) — les scans de ce projet sont
    numérisés à résolution identique en X et Y, et une légère différence
    n'aurait de toute façon aucun effet visible sur les usages qui en
    dépendent (ex. :func:`~media_restorer.engines.vectorise.topology.px_to_mm`
    pour l'affichage d'une largeur de crayon approximative).  Ne lève
    jamais, par le même principe que :func:`exif_orientation`.
    """
    try:
        from PIL import Image
        with Image.open(str(path)) as pil:
            value = pil.getexif().get(_EXIF_XRESOLUTION_TAG)
    except Exception:
        return _DEFAULT_DPI
    if value is None:
        return _DEFAULT_DPI
    try:
        dpi = float(value)
    except (TypeError, ValueError):
        return _DEFAULT_DPI
    return dpi if dpi > 0 else _DEFAULT_DPI


def apply_exif_orientation(img: np.ndarray, orientation: int) -> np.ndarray:
    """Applique la transformation EXIF *orientation* à *img*."""
    transform = _EXIF_TRANSFORMS.get(orientation)
    return img if transform is None else transform(img)


def imread_oriented(
    path: Path | str,
    flags: int = cv2.IMREAD_UNCHANGED,
) -> np.ndarray | None:
    """Lit *path* et le redresse selon son orientation EXIF.

    Remplaçant direct de ``cv2.imread`` — même valeur de retour, ``None`` si
    le fichier n'est pas lisible.  *flags* est complété par
    ``IMREAD_IGNORE_ORIENTATION`` pour que la rotation soit appliquée ici, et
    ici seulement, quel que soit le drapeau demandé.
    """
    img = cv2.imread(str(path), flags | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        return None
    return apply_exif_orientation(img, exif_orientation(path))

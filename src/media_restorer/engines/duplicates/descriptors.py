"""Descripteurs globaux invariants, pour l'étage « candidats ».

Un descripteur global résume une image en un vecteur court, comparable par
simple produit scalaire.  Il ne conclut rien — il ne fait que **proposer** les
paires que la vérification géométrique examinera (voir
:mod:`~media_restorer.engines.duplicates.candidates`).

Deux descripteurs complémentaires, tous deux calculés sur une vignette :

``fourier_mellin``
    Invariant par translation, rotation et homothétie.  Le principe tient en
    deux transformées : le **module** de la transformée de Fourier est
    invariant par translation ; passé en coordonnées **log-polaires**, une
    rotation et une homothétie deviennent deux translations, qu'un second
    module de transformée de Fourier absorbe à son tour.  C'est la construction
    de Reddy et Chatterji, employée ici comme descripteur et non comme méthode
    de recalage.

``ink_profile``
    Histogramme des orientations du trait, rendu invariant par rotation de la
    même façon (module de sa transformée de Fourier circulaire).  Il ne regarde
    que la **direction** des gradients, jamais leur intensité : il est donc peu
    sensible à l'épaisseur du trait et au contraste — les deux axes sur lesquels
    les traits locaux se sont montrés les plus fragiles.

Les deux se concatènent sans précaution : chacun est normalisé séparément avant
d'être mis bout à bout, si bien qu'aucun ne domine l'autre par son échelle.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

#: Côté de la vignette d'analyse.  Assez grand pour que le spectre soit
#: informatif, assez petit pour que le coût reste dominé par la lecture.
TILE = 256
#: Taille de la grille log-polaire (angles × rayons).
POLAR_ANGLES = 128
POLAR_RADII = 128
#: Nombre de basses fréquences conservées dans chaque direction.
KEEP = 16
#: Nombre de secteurs d'orientation du profil d'encre.
ORIENTATION_BINS = 72


def load_tile(path: Path | str, side: int = TILE) -> np.ndarray:
    """Vignette en niveaux de gris, redressée et de taille fixe.

    Taille **fixe** et non proportionnelle : le descripteur doit être comparable
    d'une image à l'autre, ce qui suppose une grille commune.  L'invariance à
    l'homothétie est apportée par la construction, pas par la mise à l'échelle.
    """
    import cv2
    from PIL import Image

    from media_restorer.image_io import apply_exif_orientation, exif_orientation

    with Image.open(path) as im:
        im.draft("L", (side * 2, side * 2))
        g = im.convert("L")
        g.thumbnail((side * 2, side * 2))
        arr = np.asarray(g, np.uint8)
    try:
        arr = apply_exif_orientation(arr, exif_orientation(path))
    except Exception:
        pass
    return cv2.resize(arr, (side, side), interpolation=cv2.INTER_AREA)


def fourier_mellin(tile: np.ndarray) -> np.ndarray:
    """Descripteur invariant par translation, rotation et homothétie."""
    import cv2

    f = tile.astype(np.float32)
    # Fenêtre de Hann : sans elle, les bords francs de la vignette dominent le
    # spectre par leurs discontinuités, et le descripteur décrit le cadre.
    fenetre = np.outer(np.hanning(f.shape[0]), np.hanning(f.shape[1])).astype(np.float32)
    spectre = np.abs(np.fft.fftshift(np.fft.fft2(f * fenetre)))
    spectre = np.log1p(spectre)                      # comprime la dynamique

    centre = (spectre.shape[1] / 2.0, spectre.shape[0] / 2.0)
    rayon_max = min(centre)
    polaire = cv2.warpPolar(
        spectre, (POLAR_RADII, POLAR_ANGLES), centre, rayon_max,
        cv2.INTER_LINEAR + cv2.WARP_POLAR_LOG,
    )
    # Rotation et homothétie sont maintenant des translations : un second module
    # de transformée de Fourier les absorbe.
    invariant = np.abs(np.fft.fft2(polaire))[:KEEP, :KEEP]
    return _unit(np.log1p(invariant).ravel())


def ink_profile(tile: np.ndarray) -> np.ndarray:
    """Histogramme d'orientations du trait, invariant par rotation.

    Ne dépend que de la direction des gradients — les magnitudes ne servent
    qu'à pondérer, et le vecteur est normalisé : ni le contraste ni l'épaisseur
    du trait n'y interviennent au premier ordre.
    """
    import cv2

    f = tile.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    # Modulo π : un trait n'a pas de sens, seule son orientation compte.
    angle = np.mod(np.arctan2(gy, gx), np.pi)

    idx = np.clip((angle / np.pi * ORIENTATION_BINS).astype(int), 0, ORIENTATION_BINS - 1)
    hist = np.bincount(idx.ravel(), weights=magnitude.ravel(), minlength=ORIENTATION_BINS)
    # Une rotation décale circulairement l'histogramme ; le module de sa
    # transformée de Fourier est donc invariant par rotation.
    return _unit(np.abs(np.fft.rfft(hist)))


def _unit(v: np.ndarray) -> np.ndarray:
    """Normalise en norme 1, en tolérant le vecteur nul."""
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


#: Fonctions de descripteur, indexées par la clé de méthode du catalogue.
DESCRIPTORS = {
    "fourier_mellin": fourier_mellin,
    "ink_profile": ink_profile,
}


def describe(path: Path | str, keys: tuple[str, ...]) -> np.ndarray:
    """Descripteur concaténé de *path* pour les méthodes *keys*.

    Chaque partie est normalisée avant concaténation : aucune ne domine l'autre
    par son échelle, quelle que soit sa longueur.  L'ordre suit
    :data:`DESCRIPTORS`, non celui de *keys*, pour que deux appels avec les
    mêmes méthodes produisent des vecteurs comparables.
    """
    tile = load_tile(path)
    parties = [fn(tile) for nom, fn in DESCRIPTORS.items() if nom in set(keys)]
    if not parties:
        raise ValueError("aucune méthode de descripteur sélectionnée")
    return np.concatenate(parties).astype(np.float32)

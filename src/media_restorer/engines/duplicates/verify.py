"""Vérification géométrique d'une paire candidate.

Détecte des points d'intérêt dans les deux images, les apparie, puis estime une
homographie robuste qui rejette les appariements aberrants.  Les appariements
survivants — les *inliers* — mesurent la similarité, et l'homographie fournit
l'explication (voir :mod:`~media_restorer.engines.duplicates.merit`).

Le choix du détecteur, tranché par la mesure
--------------------------------------------
Mesuré sur douze dessins du corpus, quatorze transformations contrôlées (voir
``docs/rapport-doublons-dessins.tex``) :

* **ORB** — 18 ms, aussi invariant que SIFT en rotation, contraste, recadrage et
  compression.  C'est le défaut, et c'est **contraire à sa réputation**, établie
  sur des photographies : un trait noir sur fond clair produit des coins très
  contrastés, terrain idéal pour un détecteur FAST.
* **SIFT** — 148 ms, meilleur sur les forts agrandissements, mais **moins bon**
  sur les changements d'épaisseur de trait (0,76 contre 0,90).
* **AKAZE** — 71 ms, intermédiaire, avec quelques effondrements individuels.

L'estimateur est **MAGSAC++** (``cv2.USAC_MAGSAC``), qui évite le réglage
délicat du seuil d'inlier de RANSAC.

Réduire plutôt qu'agrandir
--------------------------
ORB tient mieux la réduction (0,98 à ×0,5) que l'agrandissement (0,85 à ×2).
:func:`verify_pair` ramène donc les deux images à une taille de travail commune
en **réduisant la plus grande** — ce qui place systématiquement le détecteur
dans son bon régime, et coûte moins cher au passage.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from media_restorer.engines.duplicates import merit as _merit
from media_restorer.engines.duplicates.merit import Merit

#: Côté de travail.  Les mesures du rapport ont été faites à cette taille.
WORK_SIDE = 1000
#: Seuil du test de ratio de Lowe : un appariement n'est retenu que s'il est
#: nettement meilleur que le second candidat.
LOWE_RATIO = 0.75
#: Tolérance de reprojection de MAGSAC, en pixels.
REPROJ_THRESHOLD = 3.0

_DETECTORS = {
    "orb_magsac": ("ORB", 2000),
    "sift_magsac": ("SIFT", 2000),
    "akaze_magsac": ("AKAZE", 0),
}


def make_detector(key: str = "orb_magsac"):
    """Instancie le détecteur et la norme de distance associée.

    Import d'OpenCV différé : le cœur ne le paie qu'à l'usage réel.
    """
    import cv2

    nom, n = _DETECTORS.get(key, _DETECTORS["orb_magsac"])
    if nom == "SIFT":
        return cv2.SIFT_create(nfeatures=n), cv2.NORM_L2
    if nom == "AKAZE":
        return cv2.AKAZE_create(), cv2.NORM_HAMMING
    return cv2.ORB_create(nfeatures=n), cv2.NORM_HAMMING


def load_work(path: Path | str, side: int = WORK_SIDE) -> np.ndarray:
    """Image de travail en niveaux de gris, redressée, réduite au besoin."""
    from PIL import Image

    from media_restorer.image_io import apply_exif_orientation, exif_orientation

    with Image.open(path) as im:
        im.draft("L", (side, side))
        g = im.convert("L")
        g.thumbnail((side, side))
        arr = np.asarray(g, np.uint8)
    try:
        return apply_exif_orientation(arr, exif_orientation(path))
    except Exception:
        return arr


def verify_pair(
    a: np.ndarray,
    b: np.ndarray,
    *,
    method: str = "orb_magsac",
    cosinus: float | None = None,
    with_photometry: bool = True,
) -> Merit:
    """Vérifie une paire d'images de travail et renvoie son mérite.

    Ne lève pas sur une paire pathologique : une image sans point d'intérêt
    exploitable produit un mérite en régime sémantique, ce qui est l'information
    juste — « aucune transformation trouvée » — plutôt qu'une exception à
    rattraper pour chacune des dizaines de milliers de paires.
    """
    import cv2

    det, norm = make_detector(method)
    ka, da = det.detectAndCompute(a, None)
    kb, db = det.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 4 or len(kb) < 4:
        return _merit.build(None, 0, 0, a.shape, b.shape, cosinus=cosinus)

    bons = []
    for paire in cv2.BFMatcher(norm).knnMatch(da, db, k=2):
        if len(paire) == 2 and paire[0].distance < LOWE_RATIO * paire[1].distance:
            bons.append(paire[0])
    if len(bons) < 4:
        return _merit.build(None, 0, len(bons), a.shape, b.shape, cosinus=cosinus)

    src = np.float32([ka[m.queryIdx].pt for m in bons]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in bons]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, REPROJ_THRESHOLD,
                                 maxIters=5000, confidence=0.999)
    n_inliers = int(mask.sum()) if mask is not None else 0
    if H is None or n_inliers < _merit.INLIERS_MINIMUM:
        return _merit.build(None, n_inliers, len(bons), a.shape, b.shape, cosinus=cosinus)

    photo = epaisseurs = None
    if with_photometry:
        # Photométrie et épaisseur se mesurent APRÈS recalage : sans cela, on
        # comparerait des pixels qui ne se correspondent pas.
        recale = cv2.warpPerspective(b, np.linalg.inv(H), (a.shape[1], a.shape[0]),
                                     borderValue=255)
        valide = recale < 255
        photo = _merit.photometry(a, recale, valide)
        epaisseurs = (_merit.stroke_width(a), _merit.stroke_width(recale))

    return _merit.build(H, n_inliers, len(bons), a.shape, b.shape,
                        cosinus=cosinus, photo=photo, epaisseurs=epaisseurs)


def verify_paths(
    path_a: Path | str,
    path_b: Path | str,
    *,
    method: str = "orb_magsac",
    cosinus: float | None = None,
) -> Merit:
    """Charge deux fichiers et vérifie la paire.  Ne lève jamais."""
    try:
        a, b = load_work(path_a), load_work(path_b)
    except Exception:
        return _merit.build(None, 0, 0, (1, 1), (1, 1), cosinus=cosinus)
    return verify_pair(a, b, method=method, cosinus=cosinus)

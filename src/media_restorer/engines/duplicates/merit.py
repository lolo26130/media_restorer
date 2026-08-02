"""Le dictionnaire de mérite : un score gradué, et l'origine de la différence.

Idée directrice — **la transformation estimée \\emph{est} l'explication.**  On ne
fabrique pas un score explicable après coup : on lit l'explication dans
l'homographie que la vérification géométrique a déjà calculée.

Mesuré sur le corpus (voir ``docs/rapport-doublons-dessins.tex``) : la
décomposition retrouve une transformation connue à **moins de 0,02° et 0,001 en
échelle**, et la couverture asymétrique vaut **exactement** la fraction d'aire
occupée.

Pourquoi deux couvertures et non une
------------------------------------
C'est le mécanisme qui distingue « A et B sont le même dessin » de « B est un
détail de A » :

* **deux couvertures proches de 1** → même dessin (régime R1) ;
* **une proche de 1, l'autre faible** → inclusion (R2), et la valeur faible
  \\emph{dit} quelle proportion l'un occupe dans l'autre.

Un scalaire unique de similarité ne pourrait pas faire cette distinction — c'est
la raison d'être du dictionnaire.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

# Régimes (voir §2 du rapport).
REGIME_GEOMETRIQUE = "geometrique"   # R1 — les deux images se recouvrent
REGIME_PARTIEL = "partielle"         # R2 — l'une est contenue dans l'autre
REGIME_SEMANTIQUE = "semantique"     # R3 — aucune transformation trouvée

#: En deçà, la couverture est jugée « partielle » plutôt que « totale ».
COUVERTURE_TOTALE = 0.80
#: Nombre d'inliers en deçà duquel on ne conclut rien de géométrique.
INLIERS_MINIMUM = 12


@dataclass
class Merit:
    """Score gradué d'une paire, et d'où vient la différence.

    Les champs géométriques restent à ``None`` en régime sémantique : un
    dictionnaire partiellement vide **est une information** — il signifie
    qu'aucune transformation n'a été trouvée, ce que l'utilisateur doit savoir
    avant d'accorder du crédit au score.
    """

    regime: str
    merite: float

    n_inliers: int = 0
    ratio_inliers: float = 0.0
    rotation_deg: float | None = None
    echelle: float | None = None
    anisotropie: float | None = None
    translation_px: tuple[float, float] | None = None
    couverture_a_dans_b: float | None = None
    couverture_b_dans_a: float | None = None
    gain_photometrique: float | None = None
    offset_photometrique: float | None = None
    ratio_epaisseur_trait: float | None = None
    cosinus_semantique: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Forme sérialisable (export CSV/JSON)."""
        return asdict(self)

    # -- Restitution en français -------------------------------------------

    def explain(self, nom_a: str = "A", nom_b: str = "B") -> str:
        """Phrase lisible décrivant la relation — le vrai livrable pour l'humain.

        « B est un détail de A — rotation 15°, échelle 0,25, couverture 25 %… »
        se vérifie d'un coup d'œil ; « score 0,83 » ne se vérifie pas.  C'est
        tout l'intérêt d'un mérite explicable.
        """
        if self.regime == REGIME_SEMANTIQUE:
            cos = self.cosinus_semantique
            détail = f" (similarité {cos:.2f})" if cos is not None else ""
            return (f"Aucune transformation géométrique entre {nom_a} et {nom_b} : "
                    f"ressemblance de contenu seulement{détail} — à vérifier à l'œil.")

        morceaux: list[str] = []
        if self.rotation_deg is not None and abs(self.rotation_deg) >= 0.5:
            morceaux.append(f"rotation {self.rotation_deg:+.0f}°")
        if self.echelle is not None and abs(self.echelle - 1) >= 0.02:
            morceaux.append(f"échelle {self.echelle:.2f}")
        if self.gain_photometrique is not None and abs(self.gain_photometrique - 1) >= 0.05:
            morceaux.append(f"contraste {100 * (self.gain_photometrique - 1):+.0f} %")
        if self.ratio_epaisseur_trait is not None and abs(self.ratio_epaisseur_trait - 1) >= 0.1:
            morceaux.append(f"trait {self.ratio_epaisseur_trait:.1f}× plus épais")

        if self.regime == REGIME_PARTIEL:
            # On nomme le contenant et le contenu d'après la plus petite couverture.
            a_dans_b = self.couverture_a_dans_b or 0.0
            b_dans_a = self.couverture_b_dans_a or 0.0
            if a_dans_b < b_dans_a:
                tete = f"{nom_a} est un détail de {nom_b}"
                part = a_dans_b
            else:
                tete = f"{nom_b} est un détail de {nom_a}"
                part = b_dans_a
            morceaux.insert(0, f"couverture {100 * part:.0f} %")
        else:
            tete = f"{nom_a} et {nom_b} sont le même dessin"

        suite = " — " + ", ".join(morceaux) if morceaux else ""
        return f"{tete}{suite}. {self.n_inliers} points concordants."


# ---------------------------------------------------------------------------
# Décomposition de l'homographie
# ---------------------------------------------------------------------------

def decompose(H: np.ndarray) -> dict[str, Any]:
    """Rotation, échelle, anisotropie et translation, lues dans *H*.

    La partie linéaire $2\\times2$ se décompose en valeurs singulières
    $A = U\\Sigma V^{\\mathsf T}$ : l'angle de $UV^{\\mathsf T}$ donne la
    rotation, $\\sqrt{\\sigma_1\\sigma_2}$ l'échelle, et $\\sigma_1/\\sigma_2$
    l'anisotropie (1 pour une similitude pure, davantage s'il y a cisaillement —
    signe d'une photographie prise de biais plutôt que d'un scan).

    Le signe de l'angle suit la convention des coordonnées image (axe $y$ vers
    le bas) : c'est une convention, pas une erreur.
    """
    A = np.asarray(H, dtype=np.float64)[:2, :2]
    U, S, Vt = np.linalg.svd(A)
    R = U @ Vt
    return {
        "rotation_deg": float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))),
        "echelle": float(np.sqrt(max(S[0] * S[1], 1e-12))),
        "anisotropie": float(S[0] / S[1]) if S[1] > 1e-12 else float("inf"),
        "translation_px": (float(H[0, 2]), float(H[1, 2])),
    }


def coverage(H: np.ndarray, forme_src: tuple[int, int], forme_dst: tuple[int, int]) -> float:
    """Part de *dst* couverte par le rectangle de *src* projeté par *H*.

    Valeur dans [0, 1].  C'est la grandeur qui détecte l'inclusion partielle :
    projeter les quatre coins puis mesurer l'aire de l'intersection avec le
    cadre de destination.
    """
    import cv2

    h, w = forme_src[:2]
    coins = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    try:
        proj = cv2.perspectiveTransform(coins, np.asarray(H, np.float64)).reshape(-1, 2)
    except cv2.error:
        return 0.0
    if not np.isfinite(proj).all():
        return 0.0
    hd, wd = forme_dst[:2]
    cadre = np.array([[0, 0], [wd, 0], [wd, hd], [0, hd]], np.float32)
    try:
        aire, _ = cv2.intersectConvexConvex(proj.astype(np.float32), cadre)
    except cv2.error:
        return 0.0
    return float(np.clip(aire / float(wd * hd), 0.0, 1.0))


def photometry(a: np.ndarray, b_recale: np.ndarray, masque: np.ndarray | None = None
               ) -> tuple[float, float]:
    """Ajuste $I_a \\approx g\\,I_b + o$ : *g* est le gain, *o* le décalage.

    Mesuré **après recalage**, donc sur des pixels qui se correspondent
    réellement.  *g* traduit l'écart de contraste, *o* celui de luminosité —
    exactement les deux axes que l'utilisateur voulait voir séparés.
    """
    x = b_recale.astype(np.float64).ravel()
    y = a.astype(np.float64).ravel()
    if masque is not None:
        m = masque.ravel().astype(bool)
        x, y = x[m], y[m]
    if x.size < 16 or float(np.std(x)) < 1e-6:
        return 1.0, 0.0
    gain, offset = np.polyfit(x, y, 1)
    return float(gain), float(offset)


def stroke_width(gray: np.ndarray) -> float:
    """Épaisseur médiane du trait, en pixels.

    Transformée de distance sur le masque d'encre : en chaque pixel d'encre,
    la distance au fond le plus proche vaut la demi-épaisseur locale.  La
    médiane sur les pixels d'encre est robuste aux aplats et aux salissures.
    """
    import cv2

    g = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    seuil = 0.75 * float(g.mean())
    encre = (g < seuil).astype(np.uint8)
    if not encre.any():
        return 0.0
    dist = cv2.distanceTransform(encre, cv2.DIST_L2, 3)
    valeurs = dist[encre.astype(bool)]
    return float(2.0 * np.median(valeurs))


# ---------------------------------------------------------------------------
# Assemblage
# ---------------------------------------------------------------------------

def build(
    H: np.ndarray | None,
    n_inliers: int,
    n_matches: int,
    forme_a: tuple[int, int],
    forme_b: tuple[int, int],
    *,
    cosinus: float | None = None,
    photo: tuple[float, float] | None = None,
    epaisseurs: tuple[float, float] | None = None,
) -> Merit:
    """Construit le dictionnaire de mérite d'une paire.

    Sans homographie exploitable, on retombe sur le régime sémantique : le
    mérite se réduit alors au cosinus, et **aucun champ géométrique n'est
    inventé**.
    """
    if H is None or n_inliers < INLIERS_MINIMUM:
        return Merit(
            regime=REGIME_SEMANTIQUE,
            merite=float(cosinus) if cosinus is not None else 0.0,
            n_inliers=int(n_inliers),
            ratio_inliers=float(n_inliers / n_matches) if n_matches else 0.0,
            cosinus_semantique=cosinus,
        )

    geo = decompose(H)
    couv_ab = coverage(H, forme_a, forme_b)
    couv_ba = coverage(np.linalg.inv(np.asarray(H, np.float64)), forme_b, forme_a)
    ratio = float(n_inliers / n_matches) if n_matches else 0.0

    partiel = min(couv_ab, couv_ba) < COUVERTURE_TOTALE
    # Le mérite combine la qualité de l'appariement et l'étendue du recouvrement :
    # une superposition parfaite sur 4 % de la page n'est pas le même événement
    # qu'une superposition parfaite sur toute l'image.
    merite = float(np.clip(ratio * max(couv_ab, couv_ba), 0.0, 1.0))

    ratio_ep = None
    if epaisseurs and epaisseurs[1] > 1e-6:
        ratio_ep = float(epaisseurs[0] / epaisseurs[1])

    return Merit(
        regime=REGIME_PARTIEL if partiel else REGIME_GEOMETRIQUE,
        merite=merite,
        n_inliers=int(n_inliers),
        ratio_inliers=ratio,
        rotation_deg=geo["rotation_deg"],
        echelle=geo["echelle"],
        anisotropie=geo["anisotropie"],
        translation_px=geo["translation_px"],
        couverture_a_dans_b=couv_ab,
        couverture_b_dans_a=couv_ba,
        gain_photometrique=photo[0] if photo else None,
        offset_photometrique=photo[1] if photo else None,
        ratio_epaisseur_trait=ratio_ep,
        cosinus_semantique=cosinus,
    )

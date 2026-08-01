"""Signaux de tri « gratuits » mesurés sur une image, sans aucun modèle.

Quatre grandeurs suffisent à un premier tri grossier d'un corpus de dessins
numérisés : orientation, résolution, densité d'encre, et chromatisme.  Toutes se
calculent en lisant l'image à résolution réduite — aucun réseau de neurones,
aucun téléchargement de poids, aucun GPU.

Pourquoi deux mesures de chromatisme et non une
-----------------------------------------------
La saturation **moyenne** d'une image confond deux situations sans rapport : un
dessin réellement colorié, et un dessin au trait noir sur un papier jauni par le
temps.  Sur le corpus de référence, la mesure naïve rangeait 47\\ % des images en
« teintées » — chiffre inexploitable, car **36 % d'entre elles étaient du trait
noir sur papier vieilli**.

D'où deux mesures séparées :

* :attr:`ImageSignals.ink_saturation` — saturation des pixels **d'encre**
  (les plus sombres) : élevée, le dessin est réellement en couleur ;
* :attr:`ImageSignals.paper_saturation` — saturation des pixels **de papier**
  (les plus clairs) : élevée, le support est jauni.

Le premier est un axe de **contenu**, le second un axe de **condition de
numérisation** (candidat naturel à un traitement de restauration par lot).  Les
confondre reviendrait à trier sur l'état des scans en croyant trier sur le sujet.

Seuils
------
Les seuils sont **empiriques**, calés sur le corpus de référence, et exposés en
constantes de module pour rester ajustables sans toucher au code.  Ils décrivent
un compromis, pas une vérité : un corpus de nature différente demandera de les
revoir, et les catégories dérivées sont recalculées à la volée depuis les
mesures brutes, qui elles ne changent jamais.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# -- Orientation ------------------------------------------------------------
# Bande morte autour du carré : une page A4 scannée de travers reste « portrait ».
SQUARE_RATIO_RANGE = (0.90, 1.15)

# -- Densité d'encre --------------------------------------------------------
# Un pixel est « encre » s'il est nettement plus sombre que la luminance moyenne.
# Seuil RELATIF (et non absolu) : un scan sous-exposé ne bascule pas d'un coup
# toute l'image en « dense ».
INK_THRESHOLD_FACTOR = 0.75
# Bornes des classes de densité, en pourcentage de pixels d'encre.
INK_DENSITY_BINS = (5.0, 15.0, 35.0)

# -- Chromatisme ------------------------------------------------------------
# Quantiles délimitant l'encre (les plus sombres) et le papier (les plus clairs).
INK_QUANTILE = 0.20
PAPER_QUANTILE = 0.60
# Au-delà de ces saturations moyennes (écart max-min des canaux RVB, sur 255) :
COLOURED_INK_SATURATION = 25.0
TINTED_PAPER_SATURATION = 12.0

# -- Résolution -------------------------------------------------------------
RESOLUTION_BINS_MPX = (1.0, 6.0, 20.0)

# Côté de la vignette d'analyse.  Assez grand pour que les quantiles soient
# stables, assez petit pour que le coût reste dominé par la lecture du fichier.
THUMBNAIL_SIDE = 160

ORIENTATIONS = ("portrait", "paysage", "carré")
INK_DENSITIES = ("très clair", "clair", "moyen", "dense")
RESOLUTIONS = ("< 1 Mpx", "1-6 Mpx", "6-20 Mpx", "> 20 Mpx")
SUPPORTS = (
    "trait noir / papier neutre",
    "trait noir / papier jauni",
    "encre colorée / papier neutre",
    "encre colorée / papier jauni",
)


@dataclass(frozen=True)
class ImageSignals:
    """Mesures brutes d'une image, et catégories qui s'en déduisent.

    Les attributs sont les **mesures** ; les propriétés sont les **catégories**,
    recalculées à la volée depuis les seuils du module.  Cette séparation permet
    de rejouer un tri avec d'autres seuils sans relire les 11\\ 000 fichiers :
    seules les mesures méritent d'être mises en cache.

    Attributs
    ---------
    path : Path
        Fichier mesuré.
    width, height : int
        Dimensions **natives**, lues dans l'en-tête (pas celles de la vignette).
    ink_coverage : float
        Part de pixels d'encre, en pourcentage (0–100).
    ink_saturation : float
        Saturation moyenne des pixels d'encre (0–255).
    paper_saturation : float
        Saturation moyenne des pixels de papier (0–255).
    """

    path: Path
    width: int
    height: int
    ink_coverage: float
    ink_saturation: float
    paper_saturation: float

    # -- Catégories dérivées ------------------------------------------------

    @property
    def megapixels(self) -> float:
        return self.width * self.height / 1e6

    @property
    def orientation(self) -> str:
        """``"portrait"``, ``"paysage"`` ou ``"carré"``."""
        low, high = SQUARE_RATIO_RANGE
        ratio = self.width / self.height if self.height else 1.0
        if ratio < low:
            return "portrait"
        return "paysage" if ratio > high else "carré"

    @property
    def ink_density(self) -> str:
        """Classe de densité d'encre (voir :data:`INK_DENSITY_BINS`)."""
        return _classify(self.ink_coverage, INK_DENSITY_BINS, INK_DENSITIES)

    @property
    def resolution_class(self) -> str:
        """Classe de résolution (voir :data:`RESOLUTION_BINS_MPX`)."""
        return _classify(self.megapixels, RESOLUTION_BINS_MPX, RESOLUTIONS)

    @property
    def coloured_ink(self) -> bool:
        """Vrai si le dessin lui-même est en couleur — axe de **contenu**."""
        return self.ink_saturation > COLOURED_INK_SATURATION

    @property
    def tinted_paper(self) -> bool:
        """Vrai si le support est jauni — axe de **condition**, pas de contenu."""
        return self.paper_saturation > TINTED_PAPER_SATURATION

    @property
    def support(self) -> str:
        """Combinaison encre/papier, l'un des quatre libellés de :data:`SUPPORTS`."""
        return SUPPORTS[2 * int(self.coloured_ink) + int(self.tinted_paper)]


def _classify(value: float, bins: tuple[float, ...], labels: tuple[str, ...]) -> str:
    """Range *value* dans la classe correspondante (bornes exclusives, croissantes)."""
    for bound, label in zip(bins, labels):
        if value < bound:
            return label
    return labels[-1]


class TooLarge(Exception):
    """L'image dépasse le plafond de résolution demandé.

    Exception distincte d'une erreur de lecture : une image écartée par le
    plafond est un **choix**, un fichier corrompu est un **incident**.  Les
    confondre ferait annoncer « 54 fichiers illisibles » là où il n'y en a que
    trois, voir :class:`~media_restorer.engines.triage.scan.ScanResult`.
    """


def measure_image(
    path: Path | str,
    *,
    thumbnail_side: int = THUMBNAIL_SIDE,
    max_megapixels: float | None = None,
) -> ImageSignals:
    """Mesure les signaux gratuits de *path*.

    Lève ``OSError`` (ou ``PIL.UnidentifiedImageError``) si le fichier n'est pas
    une image lisible — c'est à l'appelant de décider quoi faire d'un fichier
    corrompu, :func:`~media_restorer.engines.triage.scan.scan_directory` les
    ignore par exemple.

    *max_megapixels* écarte les images trop lourdes en levant :class:`TooLarge`.
    Le contrôle a lieu **après la lecture de l'en-tête et avant tout décodage**
    de pixels : ``Image.open()`` ne lit que l'en-tête, si bien qu'une image
    écartée ne coûte que quelques microsecondes — c'est ce qui rend le plafond
    gratuit plutôt qu'une simple protection.

    Optimisation de lecture : ``Image.draft()`` demande au décodeur JPEG de ne
    produire qu'une version réduite, ce qui évite de décompresser entièrement un
    scan de plusieurs centaines de mégaoctets.  **Cela ne s'applique qu'au
    JPEG** : un PNG ou un TIFF de même taille sera décodé en pleine résolution,
    et coûtera sensiblement plus cher.
    """
    from PIL import Image  # import différé : le cœur ne le paie qu'à l'usage

    with Image.open(path) as im:
        width, height = im.size          # en-tête seul — aucun pixel décodé
        if max_megapixels is not None and width * height / 1e6 > max_megapixels:
            raise TooLarge(f"{width}×{height} ({width * height / 1e6:.1f} Mpx)")
        im.draft("RGB", (thumbnail_side, thumbnail_side))
        thumb = im.convert("RGB")
        thumb.thumbnail((thumbnail_side, thumbnail_side))
        rgb = np.asarray(thumb, dtype=np.float32)

    return _signals_from_array(Path(path), width, height, rgb)


def _signals_from_array(
    path: Path, width: int, height: int, rgb: np.ndarray
) -> ImageSignals:
    """Calcule les trois mesures depuis un tableau RVB (H, W, 3) en flottants."""
    flat = rgb.reshape(-1, 3)
    # Luminance perceptuelle approchée (mêmes poids que la vignette d'origine) :
    # le vert domine la perception, le bleu compte peu.
    lum = (3 * flat[:, 0] + 6 * flat[:, 1] + flat[:, 2]) / 10
    saturation = flat.max(axis=1) - flat.min(axis=1)

    coverage = 100.0 * float(np.mean(lum < INK_THRESHOLD_FACTOR * lum.mean()))

    # Quantiles plutôt qu'un tri complet : même résultat, coût linéaire.
    ink_cut = float(np.quantile(lum, INK_QUANTILE))
    paper_cut = float(np.quantile(lum, PAPER_QUANTILE))
    ink_mask = lum <= ink_cut
    paper_mask = lum >= paper_cut

    def _mean_saturation(mask: np.ndarray) -> float:
        # Une image parfaitement uniforme peut vider un masque : on retombe alors
        # sur la saturation globale plutôt que de renvoyer NaN.
        return float(saturation[mask].mean()) if mask.any() else float(saturation.mean())

    return ImageSignals(
        path=path,
        width=width,
        height=height,
        ink_coverage=coverage,
        ink_saturation=_mean_saturation(ink_mask),
        paper_saturation=_mean_saturation(paper_mask),
    )

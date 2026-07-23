"""Analyse topologique (GUDHI) du champ de noirceur d'un dessin.

Une seule construction de complexe simplicial (:class:`gudhi.AlphaComplex`,
Delaunay 2D) sur un nuage de points échantillonné dans les zones sombres de
l'image suffit à produire :

- le **squelette du dessin** : les paires de mort de l'homologie H0 de
  l'homologie persistante sont, par construction, exactement les arêtes de
  l'**arbre couvrant minimal** du nuage de points (fait standard : dans une
  filtration croissante, les arêtes qui font mourir une composante connexe
  sont précisément celles retenues par l'algorithme de Kruskal). Aucun outil
  de skeletonisation externe n'est utilisé — le squelette *est* un
  sous-produit direct de l'analyse de persistance ;
- un **intervalle de largeurs de crayon plausible** et plusieurs **candidats
  de retraçage** : la distribution des longueurs de ces arêtes présente des
  sauts (grandes longueurs séparant les fusions « courtes », continuation
  naturelle d'un même trait, des fusions « longues », saut entre deux traits
  distincts) — ce sont ces sauts qui définissent les coupures candidates,
  comme on coupe un dendrogramme de classification hiérarchique à
  différentes hauteurs ;
- un **signal de stabilité pour les boucles** (traits fermés) via les paires
  H1, calculées par le même appel à ``persistence()``.

Ce module ne dépend que de GUDHI, OpenCV et NumPy — délibérément pas de
scikit-image ni de networkx : le traçage sort du simplexe simplicial
lui-même (voir :mod:`~media_restorer.engines.vectorise.tracing` pour la
décomposition de cet arbre couvrant en traits individuels, elle aussi en
Python pur).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import gudhi
import numpy as np


@dataclass(frozen=True)
class Edge:
    """Une arête de l'arbre couvrant minimal — une paire de mort H0.

    *i*/*j* indexent :attr:`TopologyAnalysis.points` (et les tableaux
    parallèles ``widths``/``intensities``).
    """

    i: int
    j: int
    length_px: float


@dataclass
class TopologyAnalysis:
    """Résultat de l'analyse topologique d'un nuage de points sombres.

    Paramètres
    ----------
    points : np.ndarray
        ``(N, 2)`` float64, coordonnées ``(x, y)`` des points échantillonnés.
    widths : np.ndarray
        ``(N,)`` float32, demi-épaisseur locale (px) — valeur de la
        transformée de distance du masque binaire à chaque point.
    intensities : np.ndarray
        ``(N,)`` float32 dans ``[0, 1]``, noirceur locale à chaque point.
    mst_edges : list[Edge]
        Arêtes de l'arbre couvrant minimal, triées par longueur croissante.
    loop_persistences : list[float]
        Durée de vie (mort − naissance, en unité de filtration α) de chaque
        boucle H1, triée décroissante (les plus stables d'abord).
    """

    points: np.ndarray
    widths: np.ndarray
    intensities: np.ndarray
    mst_edges: list[Edge]
    loop_persistences: list[float]


@dataclass
class RawCandidate:
    """Sous-forêt de l'arbre couvrant minimal retenue pour une coupure donnée.

    Paramètres
    ----------
    edges : list[Edge]
        Arêtes retenues (longueur ≤ *cutoff_px*).
    cutoff_px : float
        Seuil de longueur utilisé pour cette coupure.
    pencil_width_px : float
        Largeur de crayon mesurée sur les points couverts par ce candidat
        (médiane des demi-épaisseurs locales, ×2).
    score : float
        Plausibilité topologique dans ``[0, 1]`` — voir :func:`_score_and_cutoffs`.
    """

    edges: list[Edge]
    cutoff_px: float
    pencil_width_px: float
    score: float


# ---------------------------------------------------------------------------
# Échantillonnage du nuage de points
# ---------------------------------------------------------------------------

def _decimate_darkest(gray: np.ndarray, mask: np.ndarray, block: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices ``(rows, cols)`` du pixel le plus sombre de chaque bloc *block×block*.

    Un bloc sans aucun pixel masqué ne contribue aucun point.  Entièrement
    vectorisé (pas de boucle Python sur les blocs) : reste rapide même sur
    une image de plusieurs dizaines de mégapixels.
    """
    h, w = gray.shape
    h2, w2 = (h // block) * block, (w // block) * block
    if h2 == 0 or w2 == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    g = gray[:h2, :w2].astype(np.float32)
    m = mask[:h2, :w2]
    darkness = np.where(m, 255.0 - g, -1.0)
    bh, bw = h2 // block, w2 // block
    blocks = darkness.reshape(bh, block, bw, block).transpose(0, 2, 1, 3).reshape(bh, bw, block * block)
    best_val = blocks.max(axis=2)
    flat_idx = blocks.argmax(axis=2)
    valid_bi, valid_bj = np.nonzero(best_val > 0.0)
    r_in = flat_idx[valid_bi, valid_bj] // block
    c_in = flat_idx[valid_bi, valid_bj] % block
    rows = valid_bi * block + r_in
    cols = valid_bj * block + c_in
    return rows, cols


def dark_mask(gray: np.ndarray, fraction: float) -> np.ndarray:
    """Masque des pixels plus sombres que le ``fraction``-ième rang, par rang plutôt que par seuil.

    Immunisé au plateau de fond (voir docstring de :func:`sample_dark_points`) :
    ``argpartition``/``partition`` déterminent la valeur-limite de rang
    ``round(fraction * N)``, mais de nombreux pixels peuvent être à égalité
    sur cette valeur exacte — deux situations très différentes à distinguer :

    - la valeur-limite est celle du **plateau de fond** (papier) — c'est le
      cas quand *fraction* dépasse la proportion réelle de contenu sombre
      (un simple trait fin sur grand fond, par exemple) : une comparaison
      large (``<=``) inclurait alors la quasi-totalité de l'image (mesuré :
      un pixel de papier à 250, identique au reste du fond, se retrouvait
      marqué « sombre »), et sur une image totalement plate, l'image
      entière — plus aucun pixel de fond disponible pour
      ``cv2.distanceTransform``, qui renvoie alors son sentinel ``FLT_MAX``,
      le bug d'origine) ;
    - la valeur-limite est celle d'un **contenu réel de couleur uniforme**
      (un trait dessiné d'un seul ton, par exemple) — y exclure les égalités
      couperait alors du contenu légitime (mesuré : *fraction* choisie pour
      correspondre exactement au contenu réel, comparaison stricte
      retournant un masque vide).

    Le signal qui distingue les deux : la taille de l'ensemble à égalité par
    rapport à *fraction*.  Une comparaison large qui ne dépasse pas
    largement *fraction* est fiable (peu ou pas d'égalités à la limite) ; une
    comparaison large qui explose très au-delà de *fraction* signale un
    plateau — on retombe alors sur la comparaison stricte.
    """
    flat = gray.ravel()
    k = max(1, int(round(flat.size * fraction)))
    boundary = np.partition(flat, k - 1)[k - 1]
    mask = gray <= boundary
    if mask.mean() > fraction * 1.5:
        mask = gray < boundary
    return mask


def sample_dark_points(
    gray: np.ndarray,
    *,
    mark_fraction: float = 0.15,
    max_points: int = 30_000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Échantillonne un nuage de points dans les zones sombres de *gray*.

    Retourne ``(points, widths, intensities)`` — voir
    :class:`TopologyAnalysis` pour leur forme/sens.

    *mark_fraction* est un seuillage adaptatif (fraction de la distribution
    de luminosité, pas un seuil fixe en niveaux de gris) : les
    ``mark_fraction`` pixels les plus sombres de l'image sont considérés
    comme des candidats « trait », sélectionnés par rang
    (:func:`numpy.argpartition`) plutôt que par percentile — un scan de
    dessin a typiquement un fond de papier occupant un large plateau de
    valeurs quasi constantes, sur lequel un percentile dégénère souvent
    exactement à la valeur du plateau (mesuré : ``np.percentile`` retombait
    sur la couleur du papier, incluant alors la quasi-totalité de l'image
    dans le masque).  S'adapte ainsi à l'exposition propre à chaque scan
    sans supposer un fond blanc à 255, et sans cet écueil.

    La décimation par blocs (:func:`_decimate_darkest`) ramène le nuage à
    environ *max_points*, indépendamment de la résolution native — une image
    de plusieurs dizaines de mégapixels n'est donc jamais transmise telle
    quelle à GUDHI (voir docstring de module).  Le bloc de décimation double
    jusqu'à ce que le nombre de points passe sous ``1.5 × max_points`` ; un
    sous-tirage aléatoire final ramène exactement à *max_points* si besoin.
    """
    mask = dark_mask(gray, mark_fraction)

    block = 2
    rows = cols = np.empty(0, dtype=np.int64)
    while True:
        rows, cols = _decimate_darkest(gray, mask, block)
        if len(rows) <= max_points * 1.5 or block > min(gray.shape):
            break
        block *= 2

    if len(rows) > max_points:
        rng = rng or np.random.default_rng()
        keep = rng.choice(len(rows), size=max_points, replace=False)
        rows, cols = rows[keep], cols[keep]

    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    widths = dist[rows, cols].astype(np.float32)
    intensities = ((255.0 - gray[rows, cols].astype(np.float32)) / 255.0).astype(np.float32)
    points = np.stack([cols, rows], axis=1).astype(np.float64)  # (x, y)
    return points, widths, intensities


# ---------------------------------------------------------------------------
# Complexe simplicial et extraction de la persistance
# ---------------------------------------------------------------------------

def _build_simplex_tree(points: np.ndarray) -> gudhi.SimplexTree:
    """Construit le complexe simplicial sur *points*.

    ``AlphaComplex`` (triangulation de Delaunay 2D) est la construction la
    plus fidèle géométriquement pour un nuage plan, et confirmée fonctionner
    sur cette machine (wheel manylinux avec CGAL embarqué).  Repli sur
    ``RipsComplex`` si la construction échoue sur une autre plateforme —
    même API ``SimplexTree`` en aval, donc sans impact sur le reste du
    pipeline.
    """
    try:
        return gudhi.AlphaComplex(points=points).create_simplex_tree()
    except Exception:
        bbox_diag = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
        max_edge = max(bbox_diag / 50.0, 1.0)
        return gudhi.RipsComplex(points=points, max_edge_length=max_edge).create_simplex_tree(max_dimension=2)


def analyze(
    gray: np.ndarray,
    *,
    mark_fraction: float = 0.15,
    max_points: int = 30_000,
    rng: np.random.Generator | None = None,
) -> TopologyAnalysis:
    """Échantillonne, construit le complexe simplicial et calcule la persistance.

    Appelle ``SimplexTree.persistence()`` une seule fois — tous les candidats
    de :func:`candidates_from_analysis` réutilisent ce même résultat, coupé à
    différents seuils, plutôt que de reconstruire le complexe par hypothèse
    de largeur.
    """
    points, widths, intensities = sample_dark_points(
        gray, mark_fraction=mark_fraction, max_points=max_points, rng=rng
    )
    if len(points) < 3:
        return TopologyAnalysis(points, widths, intensities, [], [])

    st = _build_simplex_tree(points)
    st.persistence(homology_coeff_field=2, persistence_dim_max=False)
    pairs = st.persistence_pairs()

    mst_edges: list[Edge] = []
    loop_persistences: list[float] = []
    for birth, death in pairs:
        if len(birth) == 1 and len(death) == 2:
            i, j = death
            length = float(np.linalg.norm(points[i] - points[j]))
            mst_edges.append(Edge(i=i, j=j, length_px=length))
        elif len(birth) == 2 and len(death) == 3:
            loop_persistences.append(float(st.filtration(death) - st.filtration(birth)))

    mst_edges.sort(key=lambda e: e.length_px)
    loop_persistences.sort(reverse=True)
    return TopologyAnalysis(points, widths, intensities, mst_edges, loop_persistences)


# ---------------------------------------------------------------------------
# Candidats : coupure de l'arbre couvrant à différents seuils
# ---------------------------------------------------------------------------

def _score_and_cutoffs(edges: list[Edge], n_candidates: int) -> list[tuple[float, float]]:
    """Coupures candidates et leur score de stabilité topologique.

    La distribution *triée* des longueurs d'arêtes de l'arbre couvrant
    minimal est découpée en *n_candidates* tranches de rang égal ; dans
    chaque tranche, la coupure retenue est celle du plus grand saut de
    longueur *local à cette tranche*. Combine ainsi deux exigences qui se
    sont révélées incompatibles séparément (mesuré sur des cas réels) :

    - **diversité garantie** — une tranche par candidat assure des coupures
      réparties sur toute la distribution, y compris sur un contenu à tons
      continus (dégradés de fusain/ombrage) où il n'existe pas de rupture
      nette globale : chercher uniquement les plus grands sauts *absolus*
      y concentre alors tous les candidats presque au même endroit (la
      quasi-totalité de l'arbre couvrant survit à toute coupure
      raisonnable) ;
    - **justification topologique de chaque coupure** — dans un dessin à
      traits nettement séparés, les sauts réellement structurants
      (transition entre continuation d'un même trait et saut vers un trait
      distinct) sont très concentrés en fin de distribution (rang > 99,9 %
      dans un cas mesuré) : des quantiles fixes les manquent complètement.
      Chercher le plus grand saut *dans chaque tranche* les retrouve quel
      que soit leur rang exact.

    Le score (dans ``[0, 1]``) est la taille du saut retenu, relative au
    plus grand saut de toute la distribution.

    Retourne au plus *n_candidates* couples ``(cutoff_px, score)``, du plus
    fragmenté au plus continu.
    """
    if not edges:
        return []
    lengths = np.array([e.length_px for e in edges])
    if len(lengths) < 2:
        return [(float(lengths[0]), 1.0)]

    gaps = np.diff(lengths)
    max_gap = float(gaps.max()) if gaps.max() > 0 else 1.0

    n_bins = max(1, n_candidates)
    bin_edges = np.linspace(0, len(gaps), n_bins + 1).astype(int)

    results = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        if hi <= lo:
            continue
        local_idx = lo + int(np.argmax(gaps[lo:hi]))
        results.append((float(lengths[local_idx]), float(gaps[local_idx]) / max_gap))
    return results


def candidates_from_analysis(analysis: TopologyAnalysis, n_candidates: int = 5) -> list[RawCandidate]:
    """Dérive jusqu'à *n_candidates* candidats de l'analyse topologique déjà calculée.

    Triés par score décroissant (le plus topologiquement stable en premier).
    """
    cutoffs_scores = _score_and_cutoffs(analysis.mst_edges, n_candidates)
    out: list[RawCandidate] = []
    for cutoff, score in cutoffs_scores:
        kept = [e for e in analysis.mst_edges if e.length_px <= cutoff]
        if not kept:
            continue
        touched = sorted({e.i for e in kept} | {e.j for e in kept})
        pencil_width_px = float(2.0 * np.median(analysis.widths[touched]))
        out.append(RawCandidate(edges=kept, cutoff_px=cutoff, pencil_width_px=pencil_width_px, score=score))
    out.sort(key=lambda c: c.score, reverse=True)
    return out


def estimate_candidates(
    gray: np.ndarray,
    *,
    n_candidates: int = 5,
    mark_fraction: float = 0.15,
    max_points: int = 30_000,
    rng: np.random.Generator | None = None,
) -> tuple[TopologyAnalysis, list[RawCandidate]]:
    """Point d'entrée principal : analyse puis dérivation des candidats.

    Voir :func:`analyze` et :func:`candidates_from_analysis`.
    """
    analysis = analyze(gray, mark_fraction=mark_fraction, max_points=max_points, rng=rng)
    return analysis, candidates_from_analysis(analysis, n_candidates)


def px_to_mm(px: float, dpi: float) -> float:
    """Convertit une longueur en pixels vers des millimètres, via *dpi*."""
    return px / dpi * 25.4


def plausible_width_range_mm(candidates: list[RawCandidate], dpi: float) -> tuple[float, float]:
    """Intervalle ``(min, max)`` en mm des largeurs mesurées sur *candidates*."""
    if not candidates:
        return (0.0, 0.0)
    widths_px = [c.pencil_width_px for c in candidates]
    return (px_to_mm(min(widths_px), dpi), px_to_mm(max(widths_px), dpi))

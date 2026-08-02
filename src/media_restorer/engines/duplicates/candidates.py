"""Étage « candidats » : proposer les paires à vérifier, sans jamais tout comparer.

Vérifier géométriquement toutes les paires est hors de portée — mesuré 7,9 jours
avec ORB sur 8 693 images, et le nombre de paires croît en $n^2$.  Cet étage
réduit le problème : pour chaque image, ses *k* plus proches voisines selon un
descripteur **global**, donc rapide.  On passe alors de 37,8 millions de paires
à environ 87 000, soit un rapport **434×**.

Pourquoi par blocs — la décision qui rend le passage à l'échelle gratuit
------------------------------------------------------------------------
La similarité complète est une matrice $n\\times n$ :

======== ============== =====================
Corpus   Paires         Matrice ``float32``
======== ============== =====================
8 693    37,8 M         0,3 Go
43 465   945 M          **7,6 Go**
100 000  5,0 G          40 Go
======== ============== =====================

Le corpus doit croître d'un facteur cinq.  Matérialiser la matrice entière
passerait donc de « négligeable » à « 7,6 Go d'un seul tenant ».
:func:`top_k` calcule donc la similarité **bloc de lignes par bloc de lignes**
et ne conserve que le top-*k* de chaque bloc : l'empreinte mémoire reste
proportionnelle à ``block × n``, quelle que soit la taille du corpus.

Le résultat est **exact**, identique à celui d'un calcul en une fois — ce n'est
pas une approximation, contrairement à ce qu'apporterait un index approché
(``faiss``, ``hnswlib``), inutile en deçà de $10^5$ images environ.
"""
from __future__ import annotations

from typing import Callable, Iterator, Sequence

import numpy as np

#: Nombre de lignes traitées d'un coup.  512 × 100 000 float32 = 205 Mo au pire,
#: tout en gardant des produits matriciels assez gros pour saturer BLAS.
DEFAULT_BLOCK = 512

ProgressCallback = Callable[[int, int], None]


def normalise(descripteurs: np.ndarray) -> np.ndarray:
    """Normalise chaque ligne, de sorte que le produit scalaire soit un cosinus.

    Une ligne nulle (descripteur non calculable) est laissée nulle : sa
    similarité à tout le reste vaut alors 0, ce qui l'écarte naturellement sans
    cas particulier ni division par zéro.
    """
    d = np.asarray(descripteurs, dtype=np.float32)
    if d.ndim != 2:
        raise ValueError("descripteurs doit être un tableau (n, dim)")
    normes = np.linalg.norm(d, axis=1, keepdims=True)
    normes[normes < 1e-12] = 1.0
    return d / normes


def top_k(
    descripteurs: np.ndarray,
    k: int = 20,
    *,
    block: int = DEFAULT_BLOCK,
    seuil: float = 0.0,
    on_progress: ProgressCallback | None = None,
) -> list[tuple[int, int, float]]:
    """Paires candidates ``(i, j, similarité)`` avec ``i < j``, triées décroissant.

    Chaque image propose ses *k* plus proches voisines ; l'union est
    dédoublonnée, si bien qu'une paire retenue des deux côtés n'apparaît qu'une
    fois.  *seuil* écarte les similarités trop faibles avant même de proposer.

    La mémoire consommée ne dépend que de *block*, jamais de la taille du corpus
    — voir la docstring de module.
    """
    d = normalise(descripteurs)
    n = d.shape[0]
    if n < 2:
        return []
    k = min(k, n - 1)
    paires: dict[tuple[int, int], float] = {}

    for debut in range(0, n, block):
        fin = min(debut + block, n)
        sims = d[debut:fin] @ d.T                     # (bloc, n) — jamais (n, n)
        # Une image est toujours sa propre voisine la plus proche : on l'exclut.
        for local, globale in enumerate(range(debut, fin)):
            sims[local, globale] = -np.inf
        voisins = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k] if k > 0 else None
        if voisins is None:
            continue
        for local, globale in enumerate(range(debut, fin)):
            for j in voisins[local]:
                s = float(sims[local, j])
                if s < seuil or not np.isfinite(s):
                    continue
                cle = (globale, int(j)) if globale < j else (int(j), globale)
                # Une paire vue des deux côtés garde sa meilleure similarité.
                if s > paires.get(cle, -np.inf):
                    paires[cle] = s
        if on_progress is not None:
            on_progress(fin, n)

    return sorted(((i, j, s) for (i, j), s in paires.items()),
                  key=lambda t: t[2], reverse=True)


def iter_blocks(n: int, block: int = DEFAULT_BLOCK) -> Iterator[tuple[int, int]]:
    """Bornes ``(début, fin)`` des blocs de lignes — exposé pour les tests."""
    for debut in range(0, n, block):
        yield debut, min(debut + block, n)


def apply_prefilter(
    paires: Sequence[tuple[int, int, float]],
    compatible: Callable[[int, int], bool] | None,
) -> list[tuple[int, int, float]]:
    """Écarte les paires que *compatible* juge sans espoir.

    Sert l'étage de pré-filtrage : *compatible* s'appuie sur les mesures déjà en
    cache (densité d'encre, orientation…) et ne relit aucun fichier.  ``None``
    laisse tout passer, ce qui est le comportement attendu quand l'utilisateur
    a décoché le pré-filtrage.
    """
    if compatible is None:
        return list(paires)
    return [(i, j, s) for i, j, s in paires if compatible(i, j)]

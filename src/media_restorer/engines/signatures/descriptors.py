"""Empreinte de signature — réutilise ``engines.duplicates.embeddings`` tel quel.

Un descripteur fait main (Fourier-Mellin, histogramme d'orientation, comme
:mod:`~media_restorer.engines.duplicates.descriptors`) ne couvrirait qu'une
transformation géométrique entre deux crops — or deux signatures de la même
main sur deux dessins différents n'ont **aucune** transformation qui les
relie, c'est le cas R3 (redessin) qu'``engines/duplicates/embeddings.py``
identifie déjà comme exigeant des empreintes *apprises*. D'où la réutilisation
directe de ce module plutôt que l'invention d'un nouveau descripteur.

Modèle par défaut — mesuré, pas supposé
----------------------------------------
Validé avant tout code GUI sur 9 signatures réelles de 5 dessinateurs (crops
serrés à l'encre seule, issus de vrais numéros du Canard) : test du
plus-proche-voisin en laissant une signature de côté et vérifiant que sa plus
proche voisine parmi les 8 restantes est bien du même auteur.

=============== ==========================================
Modèle          Plus-proche-voisin correct
=============== ==========================================
DINOv2-small    3/9
DINOv2-base     4/9
CLIP ViT-B/32   2/9
**SigLIP**      **7/9** (7/8 hors le seul auteur sans pair)
=============== ==========================================

DINOv2 et CLIP — la référence usuelle d'``embeddings.py`` sur des dessins
entiers — ne séparent pas les auteurs sur un crop de signature, à peine
au-dessus du hasard : sur un simple trait d'encre épars, il n'y a quasiment
aucun contenu de « scène » à saisir, ce sur quoi ces modèles sont entraînés.
SigLIP s'en sort nettement mieux (perte par paires plutôt qu'auto-
distillation photographique). D'où :data:`DEFAULT_MODEL` distinct de celui
d'``embeddings.py``. Réserve honnête : échantillon petit (n=9) — à
revalider informellement en observant le taux de verdicts CONFIDENT/AMBIGUOUS
réels à mesure que la bibliothèque grossit.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from media_restorer.engines.duplicates.embeddings import Embedder

#: Contrairement à ``DEFAULT_MODEL`` d'``embeddings.py`` (DINOv2, référence
#: pour des dessins entiers) — voir la mesure dans la docstring de module.
DEFAULT_MODEL = "siglip_base"


def embed_one(path: Path, embedder: Embedder) -> np.ndarray:
    """Empreinte d'un seul crop — confort au-dessus d'un :data:`Embedder` par lots."""
    return embed_many([path], embedder)[0]


def embed_many(paths: Sequence[Path], embedder: Embedder) -> np.ndarray:
    """Empreintes de plusieurs crops, dans l'ordre donné."""
    return embedder(list(paths))

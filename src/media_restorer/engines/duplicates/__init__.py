"""Détection de dessins doublons non triviaux, sans Qt.

Trouve les doublons qu'aucune comparaison de fichiers ne verrait : mêmes dessins
re-numérisés, republiés sur un autre support, recadrés — indépendamment d'une
translation, d'une rotation, d'une homothétie, d'un écart de contraste ou d'une
épaisseur de trait différente.  La méthode, sa bibliographie et ses mesures sur
le corpus sont dans ``docs/rapport-doublons-dessins.tex``.

Comme :mod:`~media_restorer.engines.triage`,
:mod:`~media_restorer.engines.vectorise` et
:mod:`~media_restorer.engines.face_id`, ce paquet **ne dérive pas de**
:class:`~media_restorer.engines.base.BaseEngine` : son contrat va d'un *corpus*
vers un *graphe*, non d'une image vers une image.  Il n'apparaît donc pas non
plus dans l'énumération ``Engine``.

Ce que ce lot couvre — et ce qu'il ne couvre pas
------------------------------------------------
✅ **R1** — même original, deux numérisations (une homographie relie les images) ;
✅ **R2** — inclusion partielle : même dessin sur un autre support, détail,
recadrage (l'homographie envoie vers une **sous-région**).

❌ **R3** — variante *redessinée* par l'artiste.  Aucune transformation
géométrique ne relie deux tracés différents : ce cas exige des descripteurs
appris, prévus dans un lot ultérieur.  Le crochet existe
(:data:`~media_restorer.engines.duplicates.methods.STAGE_CANDIDATE`), la méthode
non — et l'interface doit le dire plutôt que de laisser croire à une couverture
complète.

Chaîne en trois étages
----------------------
Vérifier toutes les paires est hors de portée — mesuré 7,9 jours avec ORB sur
8 693 images.  D'où :

1. **pré-filtrage** (gratuit) — réutilise les mesures déjà en cache du
   pré-classement ;
2. **candidats** — descripteurs globaux, similarité **par blocs** (l'empreinte
   mémoire ne dépend pas de la taille du corpus, voir
   :mod:`~media_restorer.engines.duplicates.candidates`) ;
3. **vérification** — ORB + MAGSAC sur les seules paires retenues, produisant le
   dictionnaire de mérite.

Soit environ **26 minutes** au lieu de 7,9 jours, pour un résultat identique.
"""
from media_restorer.engines.duplicates.benchmark import compare_models, format_comparison
from media_restorer.engines.duplicates.candidates import (
    DEFAULT_BLOCK,
    apply_prefilter,
    normalise,
    top_k,
)
from media_restorer.engines.duplicates.descriptors import DESCRIPTORS, describe, load_tile
from media_restorer.engines.duplicates.device import (
    DEVICE_AUTO, DEVICE_CPU, DEVICE_GPU, DEVICE_ORDER, DEVICE_TITLES,
    probe_gpu, resolve,
)
from media_restorer.engines.duplicates.embeddings import (
    DEFAULT_MODEL, EMBEDDING_MODELS, EMBEDDING_MODELS_BY_KEY,
    EmbeddingModel, build_embedder, embed_corpus,
)
from media_restorer.engines.duplicates.groups import (
    DuplicateGraph,
    Group,
    Pair,
    build_graph,
    group_of,
)
from media_restorer.engines.duplicates.merit import (
    REGIME_GEOMETRIQUE,
    REGIME_PARTIEL,
    REGIME_SEMANTIQUE,
    Merit,
    coverage,
    decompose,
    photometry,
    stroke_width,
)
from media_restorer.engines.duplicates.methods import (
    METHODS,
    METHODS_BY_KEY,
    STAGE_CANDIDATE,
    STAGE_ORDER,
    STAGE_PREFILTER,
    STAGE_TITLES,
    STAGE_VERIFY,
    Method,
    default_keys,
    methods_for,
)
from media_restorer.engines.duplicates.verdicts import (
    MIN_VERDICTS, Verdict, calibrate, pair_key, record, verdict_for,
)
from media_restorer.engines.duplicates.verify import load_work, verify_pair, verify_paths

__all__ = [
    "DEFAULT_BLOCK",
    "DEFAULT_MODEL",
    "DEVICE_AUTO",
    "DEVICE_CPU",
    "DEVICE_GPU",
    "DEVICE_ORDER",
    "DEVICE_TITLES",
    "EMBEDDING_MODELS",
    "EMBEDDING_MODELS_BY_KEY",
    "MIN_VERDICTS",
    "DESCRIPTORS",
    "METHODS",
    "METHODS_BY_KEY",
    "REGIME_GEOMETRIQUE",
    "REGIME_PARTIEL",
    "REGIME_SEMANTIQUE",
    "STAGE_CANDIDATE",
    "STAGE_ORDER",
    "STAGE_PREFILTER",
    "STAGE_TITLES",
    "STAGE_VERIFY",
    "DuplicateGraph",
    "Group",
    "Merit",
    "EmbeddingModel",
    "Method",
    "Pair",
    "Verdict",
    "build_embedder",
    "calibrate",
    "compare_models",
    "embed_corpus",
    "format_comparison",
    "pair_key",
    "probe_gpu",
    "record",
    "resolve",
    "verdict_for",
    "apply_prefilter",
    "build_graph",
    "coverage",
    "decompose",
    "default_keys",
    "describe",
    "group_of",
    "load_tile",
    "load_work",
    "methods_for",
    "normalise",
    "photometry",
    "stroke_width",
    "top_k",
    "verify_pair",
    "verify_paths",
]

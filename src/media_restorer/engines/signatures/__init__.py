"""Cœur de classement des dessins par signature d'auteur, sans Qt.

Comme :mod:`~media_restorer.engines.triage`,
:mod:`~media_restorer.engines.duplicates`,
:mod:`~media_restorer.engines.vectorise` et
:mod:`~media_restorer.engines.face_id`, ce paquet **ne dérive pas de**
:class:`~media_restorer.engines.base.BaseEngine` : son contrat va d'un
*dessin* vers un *dessinateur* (ou une entrée à revoir), pas d'une image vers
une image. Il n'apparaît donc pas non plus dans l'énumération ``Engine``.

Voir :mod:`~media_restorer.engines.signatures.pipeline` pour l'orchestration
complète, et :mod:`~media_restorer.engines.signatures.descriptors` pour le
choix — mesuré, pas supposé — du modèle d'empreinte.
"""
from media_restorer.engines.signatures.descriptors import DEFAULT_MODEL, embed_many, embed_one
from media_restorer.engines.signatures.library import (
    DuplicateWarning,
    InvalidArtistName,
    LibraryEntry,
    add_entry,
    known_artists,
    library_dir,
    list_entries,
    validate_artist_name,
)
from media_restorer.engines.signatures.location import Box, QUERY, locate_signature
from media_restorer.engines.signatures.matching import (
    AMBIGUOUS,
    CONFIDENT,
    DEFAULT_AMBIGUOUS_MARGIN,
    DEFAULT_CONFIDENT_THRESHOLD,
    NO_MATCH,
    Candidate,
    Verdict,
    classify,
    rank,
)
from media_restorer.engines.signatures.pipeline import (
    REASON_AMBIGUOUS,
    REASON_NO_LOCATION,
    REASON_NO_MATCH,
    ScanOutcome,
    load_candidate_cache,
    save_candidate_cache,
    scan_corpus,
)
from media_restorer.engines.signatures.tags import (
    BRANCH,
    NO_SIGNATURE,
    owns,
    read_artist,
    write_artist,
    write_no_signature,
)

__all__ = [
    "AMBIGUOUS",
    "BRANCH",
    "CONFIDENT",
    "DEFAULT_AMBIGUOUS_MARGIN",
    "DEFAULT_CONFIDENT_THRESHOLD",
    "DEFAULT_MODEL",
    "NO_MATCH",
    "NO_SIGNATURE",
    "QUERY",
    "REASON_AMBIGUOUS",
    "REASON_NO_LOCATION",
    "REASON_NO_MATCH",
    "Box",
    "Candidate",
    "DuplicateWarning",
    "InvalidArtistName",
    "LibraryEntry",
    "ScanOutcome",
    "Verdict",
    "add_entry",
    "classify",
    "embed_many",
    "embed_one",
    "known_artists",
    "library_dir",
    "list_entries",
    "load_candidate_cache",
    "locate_signature",
    "owns",
    "rank",
    "read_artist",
    "save_candidate_cache",
    "scan_corpus",
    "validate_artist_name",
    "write_artist",
    "write_no_signature",
]

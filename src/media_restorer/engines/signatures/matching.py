"""Comparaison d'une empreinte de crop à la bibliothèque de signatures.

Une requête contre N entrées : pas besoin du traitement **par blocs**
d'``engines/duplicates/candidates.py`` (pensé pour n² paires sur un corpus de
10⁵ images) — un simple produit scalaire suffit, les empreintes issues de
:func:`~media_restorer.engines.duplicates.embeddings.build_embedder` étant
déjà normalisées (cosinus = produit scalaire).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from media_restorer.engines.signatures.library import LibraryEntry

#: Un seul candidat, franchement au-dessus du seuil et sans rival proche.
CONFIDENT = "confident"
#: Plusieurs dessinateurs plausibles — la marge ne permet pas de trancher seul.
AMBIGUOUS = "ambiguous"
#: Rien d'assez proche dans la bibliothèque (ou bibliothèque vide).
NO_MATCH = "no_match"

#: Réglages par défaut, exposés au ParameterTree de l'extension (persistés
#: QSettings, comme les seuils de ``doublons``/``pre_classement``).
DEFAULT_CONFIDENT_THRESHOLD = 0.85
DEFAULT_AMBIGUOUS_MARGIN = 0.05


@dataclass(frozen=True)
class Candidate:
    """Une entrée de bibliothèque et son score de similarité à la requête."""

    entry: LibraryEntry
    score: float


@dataclass(frozen=True)
class Verdict:
    """Résultat de :func:`classify` — un des trois régimes ci-dessus."""

    regime: str
    candidates: tuple[Candidate, ...]

    @property
    def artist(self) -> str | None:
        """Auteur retenu si :attr:`regime` vaut :data:`CONFIDENT`, sinon ``None``."""
        return self.candidates[0].entry.artist if self.regime == CONFIDENT else None


def rank(
    query: np.ndarray, library: list[LibraryEntry], vectors: np.ndarray
) -> list[Candidate]:
    """Classe les entrées de *library* par similarité décroissante à *query*.

    *vectors* — empreintes de *library*, dans le même ordre, déjà calculées
    (voir :mod:`~media_restorer.engines.signatures.pipeline`, qui les tient en
    cache : les recalculer à chaque comparaison serait absurde).
    """
    if not library:
        return []
    scores = vectors @ query
    order = np.argsort(scores)[::-1]
    return [Candidate(entry=library[i], score=float(scores[i])) for i in order]


def classify(
    candidates: list[Candidate],
    *,
    confident_threshold: float = DEFAULT_CONFIDENT_THRESHOLD,
    ambiguous_margin: float = DEFAULT_AMBIGUOUS_MARGIN,
) -> Verdict:
    """Classe une liste déjà triée de candidats (voir :func:`rank`).

    - Aucun candidat, ou le meilleur sous le seuil : :data:`NO_MATCH`.
    - Le meilleur au-dessus du seuil, sans rival d'un AUTRE auteur assez
      proche (ou aucun rival — plusieurs entrées du même auteur en tête ne
      comptent pas comme une hésitation) : :data:`CONFIDENT`.
    - Sinon (un autre auteur à moins de *ambiguous_margin* du meilleur) :
      :data:`AMBIGUOUS`.
    """
    if not candidates or candidates[0].score < confident_threshold:
        return Verdict(regime=NO_MATCH, candidates=tuple(candidates))

    best = candidates[0]
    rivals = [c for c in candidates[1:] if c.entry.artist != best.entry.artist]
    if not rivals or best.score - rivals[0].score >= ambiguous_margin:
        return Verdict(regime=CONFIDENT, candidates=tuple(candidates))
    return Verdict(regime=AMBIGUOUS, candidates=tuple(candidates))

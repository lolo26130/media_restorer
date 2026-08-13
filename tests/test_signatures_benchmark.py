"""Banc d'essai de comparaison de modèles — sans Qt, sans modèle réel.

La vérité terrain est la bibliothèque elle-même (chaque entrée porte déjà
son auteur) : aucun jeu de verdicts à préparer, contrairement à
``engines/duplicates/benchmark.py``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from media_restorer.engines.signatures import benchmark
from media_restorer.engines.signatures.library import LibraryEntry


def _entries(spec: dict[str, int]) -> list[LibraryEntry]:
    """*spec* : ``{auteur: nombre d'entrées}`` — chemins factices, jamais lus."""
    entries = []
    for artist, n in spec.items():
        for i in range(n):
            entries.append(LibraryEntry(artist, Path(f"/lib/{artist}/{i:04d}.png")))
    return entries


def _embedder_from(vector_by_path: dict[Path, np.ndarray]):
    def embedder(paths, on_progress=None):
        return np.stack([vector_by_path[p] for p in paths])
    return embedder


def test_evaluate_model_scores_a_perfectly_separating_embedder():
    entries = _entries({"Cabrol": 2, "Sennep": 2})
    # Cabrol proche de [1, 0], Sennep proche de [0, 1] : séparation parfaite.
    vectors = {
        entries[0].path: np.array([1.0, 0.0]), entries[1].path: np.array([0.9, 0.1]),
        entries[2].path: np.array([0.0, 1.0]), entries[3].path: np.array([0.1, 0.9]),
    }
    score = benchmark.evaluate_model(entries, _embedder_from(vectors))
    assert score.correct == 4
    assert score.evaluable == 4
    assert score.accuracy == 1.0


def test_evaluate_model_excludes_singleton_artists_from_accuracy():
    """Un auteur sans second exemplaire ne peut structurellement pas être « retrouvé »."""
    entries = _entries({"Cabrol": 2, "Solo": 1})
    vectors = {
        entries[0].path: np.array([1.0, 0.0]), entries[1].path: np.array([0.9, 0.1]),
        entries[2].path: np.array([0.5, 0.5]),
    }
    score = benchmark.evaluate_model(entries, _embedder_from(vectors))
    assert score.evaluable == 2          # « Solo » exclu du calcul
    assert "Solo" not in score.per_artist
    assert score.correct == 2


def test_evaluate_model_still_uses_singletons_as_distractors():
    """Un auteur sans pair compte quand même comme rival possible pour les autres."""
    entries = _entries({"Cabrol": 2, "Solo": 1})
    # Le second Cabrol est plus proche de Solo que du premier Cabrol.
    vectors = {
        entries[0].path: np.array([1.0, 0.0]),
        entries[1].path: np.array([0.0, 1.0]),
        entries[2].path: np.array([0.0, 0.99]),   # "Solo", très proche du 2e Cabrol
    }
    score = benchmark.evaluate_model(entries, _embedder_from(vectors))
    assert score.per_artist["Cabrol"] == (1, 2)   # un seul des deux Cabrol se retrouve


def test_compare_models_ranks_by_accuracy_and_reports_the_winner():
    entries = _entries({"Cabrol": 2, "Sennep": 2})

    def perfect(paths, on_progress=None):
        return np.stack([
            np.array([1.0, 0.0]) if "Cabrol" in str(p) else np.array([0.0, 1.0])
            for p in paths
        ])

    def useless(paths, on_progress=None):
        return np.stack([np.array([1.0, 0.0]) for _ in paths])   # tout se ressemble

    result = benchmark.compare_models(
        {"parfait": perfect, "inutile": useless}, entries=entries
    )
    assert result["meilleur"] == "parfait"
    assert result["classement"][0].model_key == "parfait"
    assert result["classement"][0].accuracy == 1.0
    assert "parfait" in result["message"]


def test_compare_models_reports_nothing_comparable_below_two_samples():
    entries = _entries({"Cabrol": 1, "Sennep": 1})
    result = benchmark.compare_models({"x": lambda paths, on_progress=None: np.zeros((len(paths), 2))},
                                       entries=entries)
    assert result["classement"] == []
    assert result["meilleur"] is None


def test_format_comparison_lists_every_model():
    entries = _entries({"Cabrol": 2})
    embedder = _embedder_from({
        entries[0].path: np.array([1.0, 0.0]), entries[1].path: np.array([0.9, 0.1]),
    })
    result = benchmark.compare_models({"m1": embedder, "m2": embedder}, entries=entries)
    text = benchmark.format_comparison(result)
    assert "m1" in text and "m2" in text

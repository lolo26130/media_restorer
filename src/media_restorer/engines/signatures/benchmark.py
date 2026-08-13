"""Banc d'essai : quel modèle reconnaît vraiment un dessinateur à sa signature ?

Même esprit que :mod:`~media_restorer.engines.duplicates.benchmark`, en plus
simple : là où les doublons rejouent des *verdicts* explicites sur des
paires, la bibliothèque de signatures porte déjà sa vérité terrain — chaque
entrée est rangée dans le dossier de SON auteur (voir
:mod:`~media_restorer.engines.signatures.library`).  Aucun verdict à
collecter séparément.

``descriptors.DEFAULT_MODEL`` a été fixé par une mesure sur 9 signatures
préparées à la main, AVANT tout code GUI (voir la docstring de
``descriptors.py``).  Ce module permet de REJOUER cette mesure sur la VRAIE
bibliothèque, à mesure qu'elle grossit avec l'usage — confirmer le choix
initial avec plus de confiance, ou le remettre en cause si un autre modèle
prend l'avantage sur un corpus plus large.

Mesuré une première fois sur la bibliothèque réelle (22 signatures, 8
dessinateurs, 5 avec au moins deux exemplaires) : SigLIP 16/19 (84 %) contre
DINOv2-base 10/19, DINOv2-small 9/19 et CLIP 9/19 — confirme largement le
choix initial, avec un écart plus net que sur l'échantillon synthétique de
départ.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace

import numpy as np

from media_restorer.engines.duplicates.embeddings import Embedder
from media_restorer.engines.signatures.library import LibraryEntry, list_entries


@dataclass(frozen=True)
class ModelScore:
    """Résultat d'UN modèle sur la bibliothèque, plus-proche-voisin en laisse-un-de-côté."""

    model_key: str
    correct: int
    evaluable: int
    #: ``{auteur: (corrects, total)}`` — détail par dessinateur, pour
    #: comprendre OÙ un modèle se trompe, pas seulement de combien.
    per_artist: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.correct / self.evaluable if self.evaluable else 0.0


def evaluate_model(entries: list[LibraryEntry], embedder: Embedder) -> ModelScore:
    """Précision du plus-proche-voisin en laisse-un-de-côté, pour un embedder déjà construit.

    Pour chaque entrée, sa plus proche voisine (cosinus) parmi les AUTRES
    doit être du même auteur.  Un auteur sans second exemplaire ne peut
    STRUCTURELLEMENT pas avoir de bonne réponse (rien à retrouver) : exclu du
    calcul de précision, mais conservé comme distracteur pour les autres —
    l'exclure complètement fausserait la mesure en facilitant les autres
    recherches.
    """
    paths = [e.path for e in entries]
    artists = [e.artist for e in entries]
    vectors = embedder(paths)
    sims = vectors @ vectors.T
    np.fill_diagonal(sims, -1.0)

    counts = Counter(artists)
    correct = 0
    evaluable = 0
    per_artist: dict[str, list[bool]] = {}
    for i, artist in enumerate(artists):
        if counts[artist] < 2:
            continue
        evaluable += 1
        j = int(np.argmax(sims[i]))
        ok = artists[j] == artist
        correct += ok
        per_artist.setdefault(artist, []).append(ok)

    return ModelScore(
        model_key="", correct=correct, evaluable=evaluable,
        per_artist={a: (sum(oks), len(oks)) for a, oks in per_artist.items()},
    )


def compare_models(
    embedder_by_model: dict[str, Embedder],
    *,
    entries: list[LibraryEntry] | None = None,
) -> dict:
    """Compare plusieurs modèles sur LA MÊME bibliothèque de signatures.

    *entries* par défaut : la bibliothèque réelle
    (:func:`~media_restorer.engines.signatures.library.list_entries`) — passer
    une liste explicite sert surtout aux tests.

    Renvoie ``{"classement": [ModelScore, ...], "meilleur": clé|None, "message": str}``.
    """
    entries = entries if entries is not None else list_entries()
    counts = Counter(e.artist for e in entries)
    if sum(1 for n in counts.values() if n >= 2) == 0:
        return {
            "classement": [], "meilleur": None,
            "message": ("Aucun dessinateur n'a encore deux signatures ou plus dans "
                        "la bibliothèque — rien à comparer pour l'instant."),
        }

    classement = []
    for cle, embedder in embedder_by_model.items():
        score = evaluate_model(entries, embedder)
        classement.append(replace(score, model_key=cle))
    classement.sort(key=lambda s: s.accuracy, reverse=True)
    meilleur = classement[0]
    return {
        "classement": classement,
        "meilleur": meilleur.model_key,
        "message": (
            f"{meilleur.model_key} l'emporte ({meilleur.correct}/{meilleur.evaluable} "
            f"plus-proche-voisin correct, {100 * meilleur.accuracy:.0f} %). Valable pour "
            f"cette bibliothèque — à rejouer à mesure qu'elle grossit."
        ),
    }


def format_comparison(resultat: dict) -> str:
    """Rend la comparaison en texte, pour une console ou un futur widget."""
    lignes = [resultat.get("message", "")]
    if resultat.get("classement"):
        lignes.append("")
        lignes.append(f"{'modèle':<15}{'correct':>10}{'précision':>12}")
        for s in resultat["classement"]:
            lignes.append(f"{s.model_key:<15}{s.correct:>4}/{s.evaluable:<5}{100 * s.accuracy:>10.0f} %")
    return "\n".join(lignes)

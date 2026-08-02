"""Orchestration : d'un répertoire au graphe de doublons.

Enchaîne les trois étages — pré-filtrage, candidats, vérification — en
n'activant que les méthodes demandées.  Une seule fonction publique,
:func:`find_duplicates`, appelable depuis une interface, un script ou un test.

Les étages sont **indépendants** : décocher toutes les méthodes d'un étage le
neutralise, il laisse alors passer ce qu'il reçoit.  Décocher la vérification
n'a en revanche pas de sens — sans elle, aucun mérite ne peut être calculé —
et c'est l'appelant qui doit le refuser (l'interface le fait).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from media_restorer.engines.duplicates import candidates as _cand
from media_restorer.engines.duplicates import descriptors as _desc
from media_restorer.engines.duplicates import verify as _verify
from media_restorer.engines.duplicates.groups import DuplicateGraph, Pair, build_graph
from media_restorer.engines.duplicates.merit import REGIME_SEMANTIQUE
from media_restorer.engines.duplicates.methods import (
    STAGE_CANDIDATE,
    STAGE_PREFILTER,
    STAGE_VERIFY,
    methods_for,
)

ProgressCallback = Callable[[int, int], None]
StageCallback = Callable[[str], None]

#: Écart de densité d'encre au-delà duquel deux dessins ne peuvent pas être
#: doublons.  Volontairement large : le pré-filtre doit être *sûr*, quitte à
#: laisser passer des paires que la vérification écartera.
INK_TOLERANCE = 3.0


def find_duplicates(
    root: Path | str,
    *,
    methods: Sequence[str] | None = None,
    recursive: bool = True,
    top_k: int = 20,
    threshold: float = 0.5,
    max_megapixels: float | None = 50.0,
    on_progress: ProgressCallback | None = None,
    on_stage: StageCallback | None = None,
) -> DuplicateGraph:
    """Cherche les doublons de *root* et renvoie le graphe.

    Ne lève pas sur un fichier isolé illisible : il est écarté et le parcours
    continue.  Sur un corpus numérisé, s'arrêter au premier incident rendrait
    l'outil inutilisable.
    """
    from media_restorer.engines.triage import iter_images, measure_image
    from media_restorer.engines.triage.signals import TooLarge

    cles = tuple(methods) if methods is not None else None
    chemins = iter_images(root, recursive=recursive)

    # --- Étage 1 : descripteurs globaux -----------------------------------
    _stage(on_stage, f"Description de {len(chemins)} images…")
    cles_desc = tuple(m.key for m in methods_for(STAGE_CANDIDATE, cles))
    if not cles_desc:
        cles_desc = ("fourier_mellin",)     # sans candidat, rien à proposer

    retenus: list[Path] = []
    vecteurs: list[np.ndarray] = []
    mesures: dict[Path, object] = {}
    prefiltre_actif = bool(methods_for(STAGE_PREFILTER, cles))

    for index, chemin in enumerate(chemins, start=1):
        try:
            if max_megapixels is not None:
                # Écarté sur l'en-tête, sans décoder : le plafond est gratuit.
                mesures[chemin] = measure_image(chemin, max_megapixels=max_megapixels)
            elif prefiltre_actif:
                mesures[chemin] = measure_image(chemin)
            vecteurs.append(_desc.describe(chemin, cles_desc))
            retenus.append(chemin)
        except TooLarge:
            pass                                   # choix de l'utilisateur
        except Exception:
            pass                                   # fichier illisible
        if on_progress is not None:
            on_progress(index, len(chemins))

    if len(retenus) < 2:
        return DuplicateGraph(groups=[], inclusions=[], uncertain=[])

    # --- Étage 2 : candidats, par blocs -----------------------------------
    _stage(on_stage, "Recherche des candidats…")
    paires = _cand.top_k(np.stack(vecteurs), k=top_k)

    # --- Étage 0 : pré-filtrage (gratuit, sur les mesures déjà faites) ----
    if prefiltre_actif and mesures:
        paires = _cand.apply_prefilter(
            paires, _make_compatible(retenus, mesures)
        )

    # --- Étage 3 : vérification géométrique --------------------------------
    verifs = methods_for(STAGE_VERIFY, cles)
    if not verifs:
        return DuplicateGraph(groups=[], inclusions=[], uncertain=[])
    methode = verifs[0].key

    _stage(on_stage, f"Vérification de {len(paires)} paires candidates…")
    verifiees: list[Pair] = []
    for index, (i, j, cos) in enumerate(paires, start=1):
        m = _verify.verify_paths(retenus[i], retenus[j], method=methode, cosinus=cos)
        # Une paire en régime SÉMANTIQUE est un candidat que la vérification
        # géométrique a rejeté — rien de plus.  Tant que la famille D (R3) n'est
        # pas implémentée, la remonter comme « incertaine » laisserait croire à
        # une analyse de contenu qui n'a pas eu lieu : sur un corpus de dessins
        # au trait, les descripteurs globaux se ressemblent tous, et la quasi-
        # totalité des candidats retombe ici.  On les écarte donc.
        if m.regime != REGIME_SEMANTIQUE and m.merite >= threshold:
            verifiees.append(Pair(a=retenus[i], b=retenus[j], merit=m))
        if on_progress is not None:
            on_progress(index, len(paires))

    resolutions = {p: s.width * s.height for p, s in mesures.items()
                   if hasattr(s, "width")}
    return build_graph(verifiees, seuil=threshold, resolutions=resolutions)


def _make_compatible(chemins: Sequence[Path], mesures: dict) -> Callable[[int, int], bool]:
    """Prédicat de pré-filtrage fondé sur les mesures du pré-classement.

    Deux dessins dont la densité d'encre diffère d'un facteur trois ne sont pas
    des doublons.  Aucun fichier n'est relu : ces mesures existent déjà.
    """
    def compatible(i: int, j: int) -> bool:
        a = mesures.get(chemins[i])
        b = mesures.get(chemins[j])
        if a is None or b is None:
            return True                       # au doute, on laisse passer
        ca, cb = getattr(a, "ink_coverage", 0.0), getattr(b, "ink_coverage", 0.0)
        if min(ca, cb) < 1e-6:
            return True
        rapport = max(ca, cb) / min(ca, cb)
        return rapport <= INK_TOLERANCE

    return compatible


def _stage(callback: StageCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)

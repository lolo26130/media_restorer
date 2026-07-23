"""Décomposition de l'arbre couvrant minimal en tracés — Python pur.

Chaque :class:`~media_restorer.engines.vectorise.topology.RawCandidate` est
une **forêt** (sous-ensemble d'arêtes de l'arbre couvrant minimal, donc sans
cycle par construction). Ce module la décompose en un nombre raisonnable de
:class:`~media_restorer.engines.vectorise.types.Stroke` par « épluchage du
diamètre » : dans chaque composante, on extrait le chemin le plus long
(diamètre de l'arbre, trouvé par deux parcours en largeur), on le retire, et
on recommence sur ce qui reste jusqu'à une borne de traits par composante ou
jusqu'à épuisement.

Aucune bibliothèque de graphes externe — l'algorithme est court sur un arbre
et s'implémente directement avec un dict d'adjacence et
``collections.deque``.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from media_restorer.engines.vectorise.topology import RawCandidate, TopologyAnalysis, px_to_mm
from media_restorer.engines.vectorise.types import Stroke, StrokeSet

_DEFAULT_MIN_STROKE_POINTS = 3
_DEFAULT_MAX_STROKES_PER_COMPONENT = 12


def _build_adjacency(edges) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = {}
    for e in edges:
        adjacency.setdefault(e.i, set()).add(e.j)
        adjacency.setdefault(e.j, set()).add(e.i)
    return adjacency


def _connected_components(adjacency: dict[int, set[int]]) -> list[set[int]]:
    """Composantes connexes de *adjacency*, par union-find (rapide, O((V+E) α(V)))."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, neighbors in adjacency.items():
        for b in neighbors:
            union(a, b)

    groups: dict[int, set[int]] = {}
    for node in adjacency:
        groups.setdefault(find(node), set()).add(node)
    return list(groups.values())


def _bfs_farthest(adjacency: dict[int, set[int]], start: int) -> tuple[int, dict[int, int | None]]:
    """Nœud le plus loin de *start* par BFS, et les prédécesseurs du parcours."""
    predecessors: dict[int, int | None] = {start: None}
    queue: deque[int] = deque([start])
    farthest = start
    while queue:
        node = queue.popleft()
        farthest = node
        for neighbor in adjacency.get(node, ()):
            if neighbor not in predecessors:
                predecessors[neighbor] = node
                queue.append(neighbor)
    return farthest, predecessors


def _diameter_path(adjacency: dict[int, set[int]], start: int) -> list[int]:
    """Chemin le plus long (diamètre) de la composante contenant *start*.

    Algorithme standard sur un arbre : un premier BFS depuis un nœud
    quelconque trouve une extrémité du diamètre, un second BFS depuis cette
    extrémité trouve l'autre — le chemin entre les deux est le diamètre.
    """
    one_end, _ = _bfs_farthest(adjacency, start)
    other_end, predecessors = _bfs_farthest(adjacency, one_end)
    path = [other_end]
    pred = predecessors[path[-1]]
    while pred is not None:
        path.append(pred)
        pred = predecessors[pred]
    path.reverse()
    return path


def _peel_component(
    adjacency: dict[int, set[int]],
    component: set[int],
    min_stroke_points: int,
    max_strokes: int,
) -> list[list[int]]:
    """Épluche *component* en au plus *max_strokes* chemins (diamètres successifs)."""
    remaining = {n: set(adjacency.get(n, ())) & component for n in component}
    paths: list[list[int]] = []
    for _ in range(max_strokes):
        active = next((n for n, nbrs in remaining.items() if nbrs), None)
        if active is None:
            break
        path = _diameter_path(remaining, active)
        if len(path) < min_stroke_points:
            break
        paths.append(path)
        for a, b in zip(path, path[1:]):
            remaining[a].discard(b)
            remaining[b].discard(a)
    return paths


def _stroke_from_path(path: list[int], analysis: TopologyAnalysis) -> Stroke:
    idx = np.array(path, dtype=np.int64)
    return Stroke(
        points=analysis.points[idx].astype(np.float32),
        widths=analysis.widths[idx].astype(np.float32),
        intensity=analysis.intensities[idx].astype(np.float32),
    )


def _make_label(candidate: RawCandidate, n_strokes: int, dpi: float) -> str:
    width_mm = px_to_mm(candidate.pencil_width_px, dpi)
    plural = "s" if n_strokes != 1 else ""
    return f"Ø {width_mm:.2g} mm — {n_strokes} trait{plural} (score {candidate.score:.2f})"


def strokes_from_candidate(
    candidate: RawCandidate,
    analysis: TopologyAnalysis,
    *,
    dpi: float = 300.0,
    min_stroke_points: int = _DEFAULT_MIN_STROKE_POINTS,
    max_strokes_per_component: int = _DEFAULT_MAX_STROKES_PER_COMPONENT,
) -> StrokeSet:
    """Décompose *candidate* (une forêt d'arêtes MST) en :class:`StrokeSet`.

    Les composantes de moins de *min_stroke_points* points (souvent
    majoritaires aux coupures serrées — voir docstring de module) sont
    ignorées : trop courtes pour constituer un trait plausible.  Chaque
    composante retenue fournit au plus *max_strokes_per_component* traits —
    « un nombre raisonnable de tracés », pas une reconstruction exhaustive
    de chaque point isolé.
    """
    adjacency = _build_adjacency(candidate.edges)
    strokes: list[Stroke] = []
    for component in _connected_components(adjacency):
        if len(component) < min_stroke_points:
            continue
        for path in _peel_component(adjacency, component, min_stroke_points, max_strokes_per_component):
            strokes.append(_stroke_from_path(path, analysis))

    return StrokeSet(
        strokes=strokes,
        pencil_width_px=candidate.pencil_width_px,
        score=candidate.score,
        label=_make_label(candidate, len(strokes), dpi),
    )


def strokesets_from_candidates(
    candidates: list[RawCandidate],
    analysis: TopologyAnalysis,
    *,
    dpi: float = 300.0,
    min_stroke_points: int = _DEFAULT_MIN_STROKE_POINTS,
    max_strokes_per_component: int = _DEFAULT_MAX_STROKES_PER_COMPONENT,
) -> list[StrokeSet]:
    """Applique :func:`strokes_from_candidate` à chaque candidat."""
    return [
        strokes_from_candidate(
            c, analysis, dpi=dpi,
            min_stroke_points=min_stroke_points,
            max_strokes_per_component=max_strokes_per_component,
        )
        for c in candidates
    ]

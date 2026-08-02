"""Du semis de paires au graphe de doublons.

Le résultat d'une campagne n'est pas une liste de paires mais **un graphe** :
sommets = images, arêtes = paires retenues.  Deux natures d'arête, qu'il serait
faux de confondre :

**Arête symétrique** — les deux couvertures sont proches de 1 : les images se
recouvrent mutuellement, c'est *le même dessin*.  Ces arêtes se composent : si
A ≈ B et B ≈ C, alors les trois appartiennent au même groupe.

**Arête dirigée** — une couverture est faible, l'autre proche de 1 : l'un est
*contenu* dans l'autre (un dessin dans une page, un détail agrandi).  Ces
arêtes **ne se composent pas**.

.. warning::

   **La transitivité est le piège de ce module.**  Si A contient B et B contient
   C, il ne s'ensuit **pas** que A ≈ C — ni même, en toute rigueur, que A
   contient C au sens où on l'entend.  Fusionner les groupes par les arêtes
   d'inclusion agglomérerait progressivement tout un fonds d'affiches en un
   seul groupe géant.  Les composantes connexes ne sont donc calculées **que
   sur les arêtes symétriques** ; les inclusions restent des relations entre
   groupes, jamais des fusions.

Le représentant d'un groupe est son membre de plus haute résolution : c'est
celui qu'on veut conserver, et il est désigné explicitement plutôt que laissé au
hasard de l'ordre de parcours.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from media_restorer.engines.duplicates.merit import (
    REGIME_PARTIEL,
    REGIME_SEMANTIQUE,
    Merit,
)


@dataclass(frozen=True)
class Pair:
    """Une paire vérifiée, avec son mérite."""

    a: Path
    b: Path
    merit: Merit

    @property
    def symmetric(self) -> bool:
        """Vrai si les deux images se recouvrent mutuellement (même dessin)."""
        return self.merit.regime not in (REGIME_PARTIEL, REGIME_SEMANTIQUE)


@dataclass
class Group:
    """Un ensemble d'images jugées identiques, et son représentant.

    Attributs
    ---------
    members : list[Path]
        Images du groupe, ordre stable (tri par chemin).
    representative : Path
        Membre à conserver — celui de plus haute résolution.
    pairs : list[Pair]
        Les paires qui ont motivé le regroupement : c'est la **preuve**, que la
        revue humaine doit pouvoir consulter plutôt que de faire confiance à
        l'agrégat.
    """

    members: list[Path]
    representative: Path
    pairs: list[Pair] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.members)


@dataclass
class DuplicateGraph:
    """Le résultat complet d'une campagne."""

    groups: list[Group]
    inclusions: list[Pair]          # arêtes dirigées, hors groupes
    uncertain: list[Pair]           # régime sémantique : à trancher à l'œil

    def summary(self) -> str:
        """Résumé en une phrase, pour la barre d'état."""
        n_img = sum(g.size for g in self.groups)
        return (f"{len(self.groups)} groupe(s) de doublons "
                f"({n_img} image(s)), {len(self.inclusions)} inclusion(s), "
                f"{len(self.uncertain)} paire(s) incertaine(s)")


def _resolution(path: Path, cache: dict[Path, int]) -> int:
    """Nombre de pixels de *path*, mémorisé.  0 si illisible."""
    if path not in cache:
        try:
            from PIL import Image
            with Image.open(path) as im:
                cache[path] = im.width * im.height
        except Exception:
            cache[path] = 0
    return cache[path]


def build_graph(
    pairs: Iterable[Pair],
    *,
    seuil: float = 0.5,
    resolutions: dict[Path, int] | None = None,
) -> DuplicateGraph:
    """Assemble les paires vérifiées en groupes, inclusions et cas incertains.

    *seuil* s'applique au mérite : en deçà, la paire est ignorée.  Les paires en
    régime sémantique sont mises à part sans être perdues — elles n'ont pas de
    preuve géométrique, mais peuvent valoir un coup d'œil.
    """
    retenues = [p for p in pairs if p.merit.merite >= seuil]
    symetriques = [p for p in retenues if p.symmetric]
    inclusions = [p for p in retenues
                  if p.merit.regime == REGIME_PARTIEL]
    incertaines = [p for p in retenues
                   if p.merit.regime == REGIME_SEMANTIQUE]

    # --- Composantes connexes, sur les SEULES arêtes symétriques -----------
    parent: dict[Path, Path] = {}

    def racine(x: Path) -> Path:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]      # compression de chemin
            x = parent[x]
        return x

    def unir(x: Path, y: Path) -> None:
        rx, ry = racine(x), racine(y)
        if rx != ry:
            parent[rx] = ry

    for p in symetriques:
        unir(p.a, p.b)

    composantes: dict[Path, list[Path]] = {}
    for sommet in list(parent):
        composantes.setdefault(racine(sommet), []).append(sommet)

    preuves: dict[Path, list[Pair]] = {}
    for p in symetriques:
        preuves.setdefault(racine(p.a), []).append(p)

    cache = dict(resolutions or {})
    groupes: list[Group] = []
    for tete, membres in composantes.items():
        if len(membres) < 2:
            continue                            # un singleton n'est pas un doublon
        membres = sorted(membres)
        representant = max(membres, key=lambda m: (_resolution(m, cache), str(m)))
        groupes.append(Group(members=membres, representative=representant,
                             pairs=sorted(preuves.get(tete, []),
                                          key=lambda p: p.merit.merite, reverse=True)))

    groupes.sort(key=lambda g: (-g.size, str(g.representative)))
    return DuplicateGraph(groups=groupes, inclusions=inclusions, uncertain=incertaines)


def group_of(graph: DuplicateGraph, path: Path) -> Group | None:
    """Groupe contenant *path*, ou ``None``."""
    for g in graph.groups:
        if path in g.members:
            return g
    return None


def describe_inclusion(pair: Pair) -> str:
    """Phrase décrivant une relation d'inclusion (voir :meth:`Merit.explain`)."""
    return pair.merit.explain(pair.a.name, pair.b.name)

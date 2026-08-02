"""Tests du cœur de détection de doublons (media_restorer.engines.duplicates).

Sans Qt.  Les tests géométriques utilisent de **vraies images de synthèse** et
des transformations connues : la vérité terrain est parfaite, on sait exactement
ce que la décomposition doit retrouver.

Deux tests portent l'essentiel du risque :

* :func:`test_blocks_give_exactly_the_same_result_as_one_shot` — le calcul par
  blocs doit être *exact*, pas approché ; c'est lui qui permet le passage à
  l'échelle sans réécriture.
* :func:`test_inclusion_edges_do_not_merge_groups` — la transitivité des
  inclusions agglomérerait tout un fonds en un seul groupe.
"""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from media_restorer.engines.duplicates import (
    REGIME_GEOMETRIQUE,
    REGIME_PARTIEL,
    REGIME_SEMANTIQUE,
    STAGE_CANDIDATE,
    STAGE_VERIFY,
    DuplicateGraph,
    Merit,
    Pair,
    build_graph,
    coverage,
    decompose,
    default_keys,
    describe,
    methods_for,
    normalise,
    stroke_width,
    top_k,
    verify_pair,
)
from media_restorer.engines.duplicates import candidates, merit, methods
from media_restorer.engines.duplicates import tags as dup_tags


# ---------------------------------------------------------------------------
# Images de synthèse
# ---------------------------------------------------------------------------

def _drawing(seed: int = 0, size=(400, 400)) -> np.ndarray:
    """Un « dessin » synthétique : traits noirs sur fond clair, riche en coins.

    Assez texturé pour qu'ORB y trouve des centaines de points — un aplat uni
    ne permettrait aucune vérification géométrique.
    """
    import cv2

    rng = np.random.default_rng(seed)
    img = np.full(size, 245, np.uint8)
    for _ in range(40):
        p1 = rng.integers(0, size[0], 2)
        p2 = rng.integers(0, size[0], 2)
        cv2.line(img, tuple(int(v) for v in p1), tuple(int(v) for v in p2), 30, 3)
    for _ in range(15):
        c = rng.integers(40, size[0] - 40, 2)
        cv2.circle(img, tuple(int(v) for v in c), int(rng.integers(10, 40)), 40, 2)
    return img


def _write(path: Path, img: np.ndarray) -> Path:
    Image.fromarray(img).save(path)
    return path


# ---------------------------------------------------------------------------
# Catalogue de méthodes
# ---------------------------------------------------------------------------

def test_method_catalogue_is_coherent():
    assert methods.check_catalogue() == []


def test_methods_are_filtered_by_stage():
    verif = methods_for(STAGE_VERIFY, tuple(m.key for m in methods.METHODS))
    assert {m.key for m in verif} == {"orb_magsac", "sift_magsac"}
    assert all(m.stage == STAGE_VERIFY for m in verif)


def test_unchecked_methods_disappear_from_the_chain():
    assert methods_for(STAGE_CANDIDATE, ("fourier_mellin",)) == (
        methods.METHODS_BY_KEY["fourier_mellin"],
    )
    assert methods_for(STAGE_CANDIDATE, ()) == ()


def test_default_selection_covers_every_stage_but_stays_cheap():
    cles = set(default_keys())
    assert "orb_magsac" in cles          # une vérification est indispensable
    assert "sift_magsac" not in cles     # 8× plus coûteux : sur demande seulement


# ---------------------------------------------------------------------------
# Décomposition de l'homographie
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("deg,facteur", [(0, 1.0), (15, 1.0), (45, 1.0),
                                         (0, 0.6), (30, 0.75)])
def test_decomposition_recovers_the_applied_transform(deg, facteur):
    """La transformation estimée EST l'explication — encore faut-il qu'elle soit juste."""
    import cv2

    g = _drawing(1)
    h, w = g.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, facteur)
    b = cv2.warpAffine(g, M, (w, h), borderValue=255)

    m = verify_pair(g, b)

    assert m.n_inliers >= merit.INLIERS_MINIMUM
    assert abs(abs(m.rotation_deg) - deg) < 1.0
    assert m.echelle == pytest.approx(facteur, abs=0.02)
    assert m.anisotropie == pytest.approx(1.0, abs=0.05)   # similitude pure


def test_decompose_reports_shear_as_anisotropy():
    """Un cisaillement (photo prise de biais) se lit dans l'anisotropie."""
    H = np.array([[1.0, 0.4, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    assert decompose(H)["anisotropie"] > 1.3


# ---------------------------------------------------------------------------
# Couverture asymétrique — le mécanisme qui détecte l'inclusion
# ---------------------------------------------------------------------------

def test_identity_covers_everything_both_ways():
    H = np.eye(3)

    assert coverage(H, (100, 200), (100, 200)) == pytest.approx(1.0, abs=0.01)


def test_coverage_equals_the_occupied_area_fraction():
    """Un dessin réduit de moitié occupe le quart de la page — la mesure le dit."""
    H = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1.0]])

    assert coverage(H, (200, 200), (200, 200)) == pytest.approx(0.25, abs=0.01)


def test_coverage_is_zero_when_projected_outside():
    H = np.array([[1.0, 0, 10_000], [0, 1.0, 10_000], [0, 0, 1.0]])

    assert coverage(H, (100, 100), (100, 100)) == 0.0


def test_a_drawing_inside_a_page_is_detected_as_partial():
    import cv2

    g = _drawing(2)
    h, w = g.shape
    petit = cv2.resize(g, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    page = np.full((h, w), 245, np.uint8)
    page[10:10 + petit.shape[0], 20:20 + petit.shape[1]] = petit

    m = verify_pair(g, page)

    assert m.regime == REGIME_PARTIEL
    # Le dessin occupe le quart de la page ; la page, elle, le contient tout entier.
    assert m.couverture_a_dans_b == pytest.approx(0.25, abs=0.05)
    assert m.couverture_b_dans_a == pytest.approx(1.0, abs=0.05)


def test_two_full_copies_are_geometric_not_partial():
    g = _drawing(3)

    m = verify_pair(g, g.copy())

    assert m.regime == REGIME_GEOMETRIQUE
    assert m.merite > 0.9


# ---------------------------------------------------------------------------
# Le dictionnaire, et sa restitution en français
# ---------------------------------------------------------------------------

def test_semantic_regime_invents_no_geometric_field():
    """Un dictionnaire partiellement vide EST une information."""
    m = merit.build(None, 0, 0, (10, 10), (10, 10), cosinus=0.7)

    assert m.regime == REGIME_SEMANTIQUE
    assert m.rotation_deg is None and m.couverture_a_dans_b is None
    assert m.cosinus_semantique == 0.7
    assert "Aucune transformation" in m.explain()


def test_explanation_names_the_container_and_the_detail():
    m = Merit(regime=REGIME_PARTIEL, merite=0.9, n_inliers=248,
              rotation_deg=15.0, echelle=0.25,
              couverture_a_dans_b=0.25, couverture_b_dans_a=1.0)

    phrase = m.explain("detail.jpg", "page.jpg")

    assert "detail.jpg est un détail de page.jpg" in phrase
    assert "couverture 25 %" in phrase
    assert "248 points concordants" in phrase


def test_explanation_ignores_negligible_differences():
    """Une rotation de 0,1° ne mérite pas d'être annoncée."""
    m = Merit(regime=REGIME_GEOMETRIQUE, merite=1.0, n_inliers=900,
              rotation_deg=0.1, echelle=1.001,
              couverture_a_dans_b=1.0, couverture_b_dans_a=1.0)

    phrase = m.explain("a", "b")

    assert "même dessin" in phrase
    assert "rotation" not in phrase and "échelle" not in phrase


def test_stroke_width_grows_with_a_thicker_line():
    import cv2

    fin = _drawing(4)
    epais = cv2.erode(fin, np.ones((3, 3), np.uint8))    # l'encre est sombre

    assert stroke_width(epais) > stroke_width(fin)
    assert stroke_width(np.full((50, 50), 255, np.uint8)) == 0.0   # page blanche


# ---------------------------------------------------------------------------
# Étage candidats — le calcul par blocs
# ---------------------------------------------------------------------------

def _direct_top_k(d: np.ndarray, k: int) -> set[tuple[int, int]]:
    """Référence naïve : matrice complète, sans blocs."""
    dn = normalise(d)
    sims = dn @ dn.T
    np.fill_diagonal(sims, -np.inf)
    paires = set()
    for i in range(len(dn)):
        for j in np.argsort(-sims[i])[:k]:
            paires.add((min(i, int(j)), max(i, int(j))))
    return paires


def test_blocks_give_exactly_the_same_result_as_one_shot():
    """Le découpage en blocs est une économie de mémoire, PAS une approximation.

    C'est ce qui permet de passer de 8 693 à 43 465 images sans réécrire
    l'étage : la matrice complète pèserait 7,6 Go, jamais matérialisés.
    """
    rng = np.random.default_rng(0)
    d = rng.normal(size=(97, 24)).astype(np.float32)

    reference = _direct_top_k(d, 5)
    for block in (1, 7, 32, 1000):
        obtenu = {(i, j) for i, j, _ in top_k(d, k=5, block=block)}
        assert obtenu == reference, f"bloc={block}"


def test_pairs_are_deduplicated_and_sorted():
    rng = np.random.default_rng(1)
    d = rng.normal(size=(20, 8)).astype(np.float32)

    paires = top_k(d, k=4, block=6)

    assert all(i < j for i, j, _ in paires)                 # canonique
    assert len({(i, j) for i, j, _ in paires}) == len(paires)  # sans doublon
    scores = [s for _, _, s in paires]
    assert scores == sorted(scores, reverse=True)


def test_threshold_drops_weak_candidates():
    d = np.array([[1, 0], [0, 1], [1, 0.01]], np.float32)

    fortes = top_k(d, k=2, seuil=0.9)

    assert all(s >= 0.9 for _, _, s in fortes)
    assert (0, 2) in {(i, j) for i, j, _ in fortes}          # quasi colinéaires


def test_a_zero_descriptor_does_not_divide_by_zero():
    d = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)

    n = normalise(d)

    assert np.isfinite(n).all()
    assert not top_k(d, k=1, seuil=0.5) or all(s >= 0.5 for _, _, s in top_k(d, k=1, seuil=0.5))


def test_prefilter_can_drop_pairs_without_reading_files():
    paires = [(0, 1, 0.9), (0, 2, 0.8)]

    assert candidates.apply_prefilter(paires, None) == paires          # décoché
    assert candidates.apply_prefilter(paires, lambda i, j: j != 2) == [(0, 1, 0.9)]


def test_descriptor_is_invariant_to_rotation(tmp_path):
    """Fourier-Mellin doit rapprocher une image de sa version tournée."""
    import cv2

    g = _drawing(5)
    tourne = cv2.warpAffine(g, cv2.getRotationMatrix2D((200, 200), 30, 1.0),
                            (400, 400), borderValue=245)
    autre = _drawing(99)

    d = np.stack([describe(_write(tmp_path / f"{n}.png", im), ("fourier_mellin",))
                  for n, im in (("a", g), ("b", tourne), ("c", autre))])
    dn = normalise(d)

    # La version tournée doit être plus proche que le dessin sans rapport.
    assert float(dn[0] @ dn[1]) > float(dn[0] @ dn[2])


# ---------------------------------------------------------------------------
# Graphe — le piège de la transitivité
# ---------------------------------------------------------------------------

def _pair(a: str, b: str, regime: str, merite: float = 0.9) -> Pair:
    return Pair(a=Path(a), b=Path(b),
                merit=Merit(regime=regime, merite=merite, n_inliers=100,
                            couverture_a_dans_b=0.2 if regime == REGIME_PARTIEL else 1.0,
                            couverture_b_dans_a=1.0))


def test_symmetric_edges_merge_into_one_group():
    g = build_graph([_pair("a.jpg", "b.jpg", REGIME_GEOMETRIQUE),
                     _pair("b.jpg", "c.jpg", REGIME_GEOMETRIQUE)])

    assert len(g.groups) == 1
    assert [p.name for p in g.groups[0].members] == ["a.jpg", "b.jpg", "c.jpg"]


def test_inclusion_edges_do_not_merge_groups():
    """A⊃B et B⊃C n'implique pas A≈C — sinon tout un fonds d'affiches fusionne."""
    g = build_graph([_pair("page1.jpg", "dessin.jpg", REGIME_PARTIEL),
                     _pair("dessin.jpg", "detail.jpg", REGIME_PARTIEL)])

    assert g.groups == []                     # aucun groupe créé
    assert len(g.inclusions) == 2             # mais les relations sont conservées


def test_singletons_are_not_groups():
    g = build_graph([_pair("a.jpg", "b.jpg", REGIME_GEOMETRIQUE, merite=0.1)],
                    seuil=0.5)

    assert g.groups == []                     # sous le seuil : rien du tout


def test_representative_is_the_highest_resolution_member():
    g = build_graph(
        [_pair("petit.jpg", "grand.jpg", REGIME_GEOMETRIQUE)],
        resolutions={Path("petit.jpg"): 1_000, Path("grand.jpg"): 9_000_000},
    )

    assert g.groups[0].representative == Path("grand.jpg")


def test_group_keeps_the_evidence_that_motivated_it():
    """La revue humaine doit pouvoir consulter la preuve, pas seulement l'agrégat."""
    g = build_graph([_pair("a.jpg", "b.jpg", REGIME_GEOMETRIQUE)])

    assert len(g.groups[0].pairs) == 1
    assert "même dessin" in g.groups[0].pairs[0].merit.explain()


def test_semantic_pairs_are_set_aside_not_lost():
    g = build_graph([_pair("a.jpg", "b.jpg", REGIME_SEMANTIQUE)])

    assert g.groups == [] and g.inclusions == []
    assert len(g.uncertain) == 1
    assert "incertaine" in g.summary()


# ---------------------------------------------------------------------------
# Étiquettes
# ---------------------------------------------------------------------------

def test_duplicates_own_only_their_branch():
    assert dup_tags.owns(["media_restorer", "Doublons", "Groupe 001"])
    assert not dup_tags.owns(["media_restorer", "Repère", "Left Eye"])
    assert not dup_tags.owns(["media_restorer", "Tri", "Orientation", "Portrait"])


def test_group_label_sorts_correctly_and_reads_alone():
    """« 042 » seul ne voudrait rien dire dans un champ plat de DigiKam."""
    assert dup_tags.group_label(42) == "Groupe 042"
    assert sorted([dup_tags.group_label(9), dup_tags.group_label(10)])[0] == "Groupe 009"


def test_written_tags_mark_the_representative():
    calls = []

    def runner(args):
        if any("=" in a for a in args if a.startswith("-")):
            calls.append(args)
            return ""
        return '[{"SourceFile": "x.jpg"}]'

    graph = build_graph(
        [_pair("petit.jpg", "grand.jpg", REGIME_GEOMETRIQUE)],
        resolutions={Path("petit.jpg"): 10, Path("grand.jpg"): 900},
    )
    ecrites, echecs = dup_tags.write_graph(graph, runner=runner)

    assert (ecrites, echecs) == (2, [])
    prefixe = "-XMP-digiKam:TagsList="
    tous = [a[len(prefixe):] for appel in calls for a in appel if a.startswith(prefixe)]
    assert "media_restorer/Doublons/Groupe 001" in tous
    assert tous.count("media_restorer/Doublons/Représentant") == 1


def test_a_locked_file_does_not_stop_the_batch():
    def runner(args):
        if any("=" in a for a in args if a.startswith("-")):
            if "petit.jpg" in args:
                raise RuntimeError("verrouillé")
            return ""
        return '[{"SourceFile": "x.jpg"}]'

    graph = build_graph([_pair("petit.jpg", "grand.jpg", REGIME_GEOMETRIQUE)])
    ecrites, echecs = dup_tags.write_graph(graph, runner=runner)

    assert ecrites == 1 and len(echecs) == 1

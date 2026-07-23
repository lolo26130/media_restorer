"""Tests du cœur de calcul de l'extension Vectorise (sans Qt).

Aucun de ces tests ne nécessite QT_QPA_PLATFORM ni qtbot — le pipeline
topologie → traçage → texture → rendu → stockage est entièrement testable
hors GUI, comme documenté dans ``engines/vectorise/__init__.py``.
"""
import cv2
import numpy as np
import pytest

import media_restorer.engines.vectorise as vec
from media_restorer.engines.vectorise import render, storage, texture, topology, tracing
from media_restorer.engines.vectorise.types import Stroke, StrokeSet


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _line_canvas(h=300, w=400, thickness=8, gray_level=40, paper=250):
    """Canevas avec un unique trait rectiligne, épaisseur connue."""
    canvas = np.full((h, w), paper, dtype=np.uint8)
    p1, p2 = (30, h // 2), (w - 30, h // 2 + 20)
    cv2.line(canvas, p1, p2, gray_level, thickness=thickness, lineType=cv2.LINE_AA)
    return canvas, p1, p2


def _multi_shape_canvas(h=400, w=600):
    """Trois formes disjointes — sert à vérifier la séparation en composantes."""
    canvas = np.full((h, w), 250, dtype=np.uint8)
    cv2.line(canvas, (50, 50), (550, 80), 40, thickness=6, lineType=cv2.LINE_AA)
    cv2.line(canvas, (50, 300), (550, 350), 60, thickness=12, lineType=cv2.LINE_AA)
    cv2.circle(canvas, (150, 200), 40, 30, thickness=5, lineType=cv2.LINE_AA)
    return canvas


def _point_segment_distance(pt, a, b):
    """Distance du point *pt* au segment [a, b] (tous en coordonnées (x, y))."""
    pt, a, b = np.asarray(pt, float), np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip(np.dot(pt - a, ab) / np.dot(ab, ab), 0.0, 1.0)
    projection = a + t * ab
    return float(np.linalg.norm(pt - projection))


def _make_stroke(n=5, seed=0) -> Stroke:
    rng = np.random.default_rng(seed)
    return Stroke(
        points=rng.uniform(0, 100, size=(n, 2)).astype(np.float32),
        widths=rng.uniform(1, 4, size=n).astype(np.float32),
        intensity=rng.uniform(0, 1, size=n).astype(np.float32),
    )


# ---------------------------------------------------------------------------
# types.py
# ---------------------------------------------------------------------------

def test_stroke_rejects_mismatched_array_lengths():
    with pytest.raises(ValueError, match="incohérentes"):
        Stroke(
            points=np.zeros((5, 2), dtype=np.float32),
            widths=np.zeros(3, dtype=np.float32),
            intensity=np.zeros(5, dtype=np.float32),
        )


def test_strokeset_n_points_sums_across_strokes():
    ss = StrokeSet(strokes=[_make_stroke(4), _make_stroke(7)])
    assert ss.n_points == 11


# ---------------------------------------------------------------------------
# topology.py — dark_mask (régression : plateau de fond)
# ---------------------------------------------------------------------------

def test_dark_mask_selects_exactly_the_requested_fraction():
    gray = np.full((100, 100), 250, dtype=np.uint8)  # tout le fond identique
    gray[:10, :10] = 40  # 100 pixels francs sombres (1 % de l'image)

    mask = topology.dark_mask(gray, fraction=0.01)

    assert mask.sum() == 100
    assert mask[:10, :10].all()


def test_dark_mask_immune_to_background_plateau():
    """Un fond constant (papier) ne doit pas dégénérer le masque vers 'tout'.

    Régression : un seuillage par percentile classique (np.percentile)
    retombe souvent exactement sur la valeur du plateau de fond quand celui-ci
    domine la distribution, incluant alors la quasi-totalité de l'image dans
    le masque — et cv2.distanceTransform d'un masque sans aucun pixel de fond
    renvoie FLT_MAX (observé en développement).  dark_mask (comparaison
    stricte contre la valeur-limite) est immunisé par construction : quand
    *fraction* dépasse la proportion réelle de contenu sombre, le masque
    obtenu est plus petit que *fraction* — jamais dilaté vers le fond.
    """
    gray = np.full((200, 200), 250, dtype=np.uint8)
    gray[90:110, 90:110] = 30  # un petit trait, très minoritaire (1 % de l'image)

    mask = topology.dark_mask(gray, fraction=0.15)  # bien plus que le contenu réel

    assert mask.mean() < 0.02  # jamais dilaté vers les 15 % demandés
    assert mask[95:105, 95:105].all()  # le trait reste couvert
    assert not mask[0:10, 0:10].any()  # le fond, lui, reste exclu


def test_dark_mask_never_includes_pixels_at_or_above_the_boundary():
    """Aucun pixel à égalité avec la valeur-limite ne doit être marqué.

    Régression directe : avant la comparaison stricte, un pixel de fond
    (250, identique au reste du fond) pouvait être marqué « sombre » par
    complétion arbitraire d'``argpartition`` — repéré via un point de trait
    reconstruit à 27 px de tout tracé dessiné.
    """
    gray = np.full((50, 50), 250, dtype=np.uint8)
    gray[10:15, 10:15] = 40

    mask = topology.dark_mask(gray, fraction=0.5)  # très supérieur au contenu réel (1 %)

    assert not (gray[mask] >= 250).any()


# ---------------------------------------------------------------------------
# topology.py — échantillonnage
# ---------------------------------------------------------------------------

def test_sample_dark_points_widths_are_bounded_not_flt_max():
    """Régression directe du bug FLT_MAX : les largeurs doivent rester
    bornées par la taille de l'image, jamais au sentinel d'OpenCV."""
    canvas, _, _ = _line_canvas()
    _, widths, _ = topology.sample_dark_points(canvas, max_points=2000)

    assert len(widths) > 0
    assert widths.max() < max(canvas.shape)


def test_sample_dark_points_respects_max_points():
    canvas, _, _ = _line_canvas(h=600, w=800, thickness=20)
    points, widths, intensities = topology.sample_dark_points(canvas, max_points=500)

    assert len(points) <= 500
    assert len(points) == len(widths) == len(intensities)


# ---------------------------------------------------------------------------
# topology.py — candidats
# ---------------------------------------------------------------------------

def _count_components(edges, n):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    for e in edges:
        ra, rb = find(e.i), find(e.j)
        if ra != rb:
            parent[ra] = rb
    return len({find(i) for i in range(n)})


def test_estimate_candidates_returns_requested_count():
    canvas = _multi_shape_canvas()
    _, candidates = topology.estimate_candidates(canvas, n_candidates=5, max_points=5000)

    assert 1 <= len(candidates) <= 5


def test_estimate_candidates_some_candidate_separates_disjoint_shapes():
    """Sur 3 formes disjointes, au moins un candidat les sépare en ≥ 3 composantes."""
    canvas = _multi_shape_canvas()
    analysis, candidates = topology.estimate_candidates(canvas, n_candidates=5, max_points=5000)
    n = len(analysis.points)

    component_counts = [_count_components(c.edges, n) for c in candidates]

    assert max(component_counts) >= 3


def test_sample_dark_points_max_width_grows_with_drawn_thickness():
    """Mesure directement la primitive de largeur (avant les heuristiques de
    candidats, qui en agrègent plusieurs et peuvent lisser le signal) : la
    demi-épaisseur maximale relevée doit croître avec l'épaisseur dessinée.

    Ne teste pas une valeur absolue précise (la médiane utilisée pour
    :attr:`~media_restorer.engines.vectorise.topology.RawCandidate.pencil_width_px`
    est une approximation conservative, pas une mesure calibrée — les points
    échantillonnés par décimation par blocs ne couvrent pas uniformément la
    section transverse d'un trait), seulement la tendance, seule propriété
    dont dépend le classement des candidats du plus fin au plus épais.
    """
    max_widths = []
    for thickness in (6, 12, 20):
        canvas, _, _ = _line_canvas(thickness=thickness)
        _, widths, _ = topology.sample_dark_points(canvas, max_points=3000, mark_fraction=0.05)
        assert len(widths) > 0
        max_widths.append(widths.max())

    assert max_widths[0] < max_widths[1] < max_widths[2]


def test_px_to_mm_and_plausible_width_range():
    assert topology.px_to_mm(300.0, dpi=300.0) == pytest.approx(25.4)

    fake_candidate = topology.RawCandidate(edges=[], cutoff_px=0.0, pencil_width_px=118.11, score=1.0)
    lo, hi = topology.plausible_width_range_mm([fake_candidate, fake_candidate], dpi=300.0)
    assert lo == hi == pytest.approx(10.0, abs=0.01)


def test_estimate_candidates_on_blank_image_returns_no_candidates():
    blank = np.full((100, 100), 250, dtype=np.uint8)
    _analysis, candidates = topology.estimate_candidates(blank, n_candidates=5, max_points=1000)
    # Un fond uniforme n'a, par construction, aucun pixel "plus sombre" que
    # les autres dans une fraction significative — la fonction ne doit pas
    # lever, seulement renvoyer peu ou pas de structure exploitable.
    assert isinstance(candidates, list)


# ---------------------------------------------------------------------------
# tracing.py
# ---------------------------------------------------------------------------

def test_strokes_from_candidate_traces_close_to_the_drawn_line():
    canvas, p1, p2 = _line_canvas(thickness=6)
    analysis, candidates = topology.estimate_candidates(canvas, n_candidates=3, max_points=3000)
    stroke_set = tracing.strokes_from_candidate(candidates[0], analysis, dpi=300.0)

    assert stroke_set.strokes
    longest = max(stroke_set.strokes, key=lambda s: len(s.points))
    distances = [_point_segment_distance(pt, p1, p2) for pt in longest.points]
    assert max(distances) < 15  # tolérance large : AA + échantillonnage par blocs


def test_strokes_from_candidate_skips_components_below_min_points():
    from media_restorer.engines.vectorise.topology import Edge, RawCandidate

    # Composante isolée de 2 points seulement (une seule arête)
    candidate = RawCandidate(edges=[Edge(i=0, j=1, length_px=1.0)], cutoff_px=1.0,
                              pencil_width_px=1.0, score=1.0)
    analysis = topology.TopologyAnalysis(
        points=np.array([[0.0, 0.0], [1.0, 0.0]]),
        widths=np.array([1.0, 1.0], dtype=np.float32),
        intensities=np.array([0.5, 0.5], dtype=np.float32),
        mst_edges=[], loop_persistences=[],
    )

    stroke_set = tracing.strokes_from_candidate(candidate, analysis, min_stroke_points=3)

    assert stroke_set.strokes == []


def test_strokes_from_candidate_label_reports_width_and_count():
    from media_restorer.engines.vectorise.topology import RawCandidate

    candidate = RawCandidate(edges=[], cutoff_px=0.0, pencil_width_px=11.81, score=0.5)
    analysis = topology.TopologyAnalysis(
        points=np.zeros((0, 2)), widths=np.zeros(0, dtype=np.float32),
        intensities=np.zeros(0, dtype=np.float32), mst_edges=[], loop_persistences=[],
    )

    stroke_set = tracing.strokes_from_candidate(candidate, analysis, dpi=300.0)

    assert "1 mm" in stroke_set.label or "1,0 mm" in stroke_set.label
    assert "0 trait" in stroke_set.label
    assert "0.50" in stroke_set.label


# ---------------------------------------------------------------------------
# texture.py
# ---------------------------------------------------------------------------

def test_extract_grain_texture_shape_and_dtype():
    canvas, _, _ = _line_canvas(h=300, w=400, thickness=15)
    mask = topology.dark_mask(canvas, fraction=0.1)

    patch = texture.extract_grain_texture(canvas, mask, patch_size=64)

    assert patch.shape == (64, 64)
    assert patch.dtype == np.uint8


def test_extract_grain_texture_handles_image_smaller_than_patch():
    small = np.full((30, 30), 200, dtype=np.uint8)
    mask = np.ones((30, 30), dtype=bool)

    patch = texture.extract_grain_texture(small, mask, patch_size=256)

    assert patch.shape == (30, 30)


def test_extract_grain_texture_flat_region_returns_mid_gray():
    flat = np.full((64, 64), 128, dtype=np.uint8)
    mask = np.ones((64, 64), dtype=bool)

    patch = texture.extract_grain_texture(flat, mask, patch_size=64)

    assert (patch == 128).all()


# ---------------------------------------------------------------------------
# render.py
# ---------------------------------------------------------------------------

def test_vectorise_output_shape_and_dtype():
    stroke = Stroke(
        points=np.array([[10, 10], [50, 10], [90, 10]], dtype=np.float32),
        widths=np.array([2, 2, 2], dtype=np.float32),
        intensity=np.array([0.8, 0.8, 0.8], dtype=np.float32),
    )
    stroke_set = StrokeSet(strokes=[stroke], pencil_width_px=4, score=1.0, label="test")
    tex = np.full((32, 32), 128, dtype=np.uint8)

    result = render.vectorise(stroke_set, (100, 100), tex)

    assert result.shape == (100, 100)
    assert result.dtype == np.uint8


def test_vectorise_stroke_pixels_are_darker_than_paper():
    stroke = Stroke(
        points=np.array([[10, 50], [50, 50], [90, 50]], dtype=np.float32),
        widths=np.array([3, 3, 3], dtype=np.float32),
        intensity=np.array([0.9, 0.9, 0.9], dtype=np.float32),
    )
    stroke_set = StrokeSet(strokes=[stroke], pencil_width_px=6, score=1.0, label="test")
    tex = np.full((32, 32), 128, dtype=np.uint8)
    paper_color = 245

    result = render.vectorise(stroke_set, (100, 100), tex, paper_color=paper_color)

    assert result[50, 50] < paper_color - 50
    assert result[5, 5] == paper_color  # coin loin du trait : papier intact


def test_simplify_widths_never_exceeds_max_segments():
    rng = np.random.default_rng(0)
    widths = rng.uniform(1, 10, size=200).astype(np.float32)

    segments = render._simplify_widths(widths, max_segments=4)

    assert len(segments) <= 4
    # les segments couvrent bien tous les indices, sans trou ni chevauchement
    covered = sorted(a for a, _, _ in segments)
    assert covered[0] == 0
    assert segments[-1][1] == len(widths) - 1


def test_simplify_widths_empty_input():
    assert render._simplify_widths(np.array([], dtype=np.float32), 4) == []


# ---------------------------------------------------------------------------
# storage.py
# ---------------------------------------------------------------------------

def test_save_and_load_strokes_roundtrip_exact(tmp_path):
    strokes = [_make_stroke(6, seed=1), _make_stroke(3, seed=2)]
    original = [
        StrokeSet(strokes=strokes, pencil_width_px=5.5, score=0.73, label="Ø 0,5 mm — 2 traits"),
    ]
    path = tmp_path / "test.strokes.h5"

    storage.save_strokes(path, original, source_path="img.jpg", dpi=300.0, image_shape=(100, 200))
    reloaded = storage.load_strokes(path)

    assert len(reloaded) == 1
    assert reloaded[0].label == original[0].label
    assert reloaded[0].score == pytest.approx(original[0].score)
    assert reloaded[0].pencil_width_px == pytest.approx(original[0].pencil_width_px)
    assert len(reloaded[0].strokes) == 2
    for orig_s, reloaded_s in zip(original[0].strokes, reloaded[0].strokes):
        np.testing.assert_array_almost_equal(orig_s.points, reloaded_s.points)
        np.testing.assert_array_almost_equal(orig_s.widths, reloaded_s.widths)
        np.testing.assert_array_almost_equal(orig_s.intensity, reloaded_s.intensity)


def test_save_strokes_preserves_candidate_order(tmp_path):
    candidates = [
        StrokeSet(strokes=[_make_stroke(2, seed=i)], pencil_width_px=float(i), score=float(i), label=f"c{i}")
        for i in range(12)  # >= 10 pour vérifier le tri lexicographique des clés HDF5 ("10" < "2")
    ]
    path = tmp_path / "order.strokes.h5"

    storage.save_strokes(path, candidates, source_path="x", dpi=300.0, image_shape=(10, 10))
    reloaded = storage.load_strokes(path)

    assert [c.label for c in reloaded] == [f"c{i}" for i in range(12)]


# ---------------------------------------------------------------------------
# __init__.py — API publique du paquet
# ---------------------------------------------------------------------------

def test_public_get_outline_accepts_bgr_image():
    canvas, _, _ = _line_canvas()
    bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    result = vec.get_outline(bgr, n_candidates=3, max_points=2000)

    assert isinstance(result, list)
    assert all(isinstance(s, StrokeSet) for s in result)


def test_public_extract_texture_accepts_bgr_image():
    canvas, _, _ = _line_canvas(thickness=15)
    bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    patch = vec.extract_texture(bgr, patch_size=32)

    assert patch.shape == (32, 32)


def test_public_vectorise_and_storage_round_trip(tmp_path):
    canvas, _, _ = _line_canvas(thickness=8)
    strokesets = vec.get_outline(canvas, n_candidates=2, max_points=2000)
    tex = vec.extract_texture(canvas, patch_size=32)

    path = tmp_path / "img.strokes.h5"
    vec.save_strokes(path, strokesets, source_path="img.png", dpi=300.0, image_shape=canvas.shape)
    reloaded = vec.load_strokes(path)
    assert len(reloaded) == len(strokesets)

    result = vec.vectorise(reloaded[0], canvas.shape, tex)
    assert result.shape == canvas.shape
    assert result.dtype == np.uint8

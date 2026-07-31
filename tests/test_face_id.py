"""Tests du cœur de détection zero-shot (media_restorer.engines.face_id).

Sans Qt et sans modèle : le détecteur est toujours un faux (aucun
téléchargement, aucune inférence transformers).
"""
import numpy as np

from media_restorer.engines.face_id import detect_landmarks
from media_restorer.engines.face_id.detect import _label_to_query, _side_rank

_IMG = np.zeros((100, 100, 3), np.uint8)


def _box(xmin, ymin, xmax, ymax):
    return {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}


def _det(score, label, box):
    return {"score": score, "label": label, "box": box}


# ---------------------------------------------------------------------------
# Requêtes dérivées des libellés
# ---------------------------------------------------------------------------

def test_label_to_query_strips_side_prefix():
    assert _label_to_query("Left Eye") == "eye"
    assert _label_to_query("Right Ear") == "ear"
    assert _label_to_query("Nose") == "nose"
    assert _label_to_query("Upper Lip") == "lip"


def test_side_rank_orders_left_center_right():
    assert _side_rank("Left Eye") < _side_rank("Nose") < _side_rank("Right Eye")


# ---------------------------------------------------------------------------
# Attribution des boîtes
# ---------------------------------------------------------------------------

def test_left_right_assigned_by_x_position():
    def detector(image, queries):
        # deux « eye » : un à gauche (x≈15), un à droite (x≈55)
        return [
            _det(0.8, "eye", _box(50, 20, 60, 30)),
            _det(0.9, "eye", _box(10, 20, 20, 30)),
        ]

    res = detect_landmarks(_IMG, ["Left Eye", "Right Eye"], detector=detector)

    assert res["Left Eye"] == (15, 25)
    assert res["Right Eye"] == (55, 25)


def test_single_label_takes_highest_scoring_box():
    def detector(image, queries):
        return [
            _det(0.3, "nose", _box(0, 0, 10, 10)),
            _det(0.95, "nose", _box(30, 40, 40, 50)),
        ]

    res = detect_landmarks(_IMG, ["Nose"], detector=detector)

    assert res["Nose"] == (35, 45)


def test_min_score_filters_weak_detections():
    def detector(image, queries):
        return [_det(0.01, "nose", _box(30, 40, 40, 50))]

    res = detect_landmarks(_IMG, ["Nose"], detector=detector, min_score=0.05)

    assert res["Nose"] is None


def test_missing_detection_yields_none():
    res = detect_landmarks(_IMG, ["Nose"], detector=lambda i, q: [])

    assert res["Nose"] is None


def test_fewer_boxes_than_labels_leaves_some_none():
    def detector(image, queries):
        return [_det(0.9, "eye", _box(10, 20, 20, 30))]  # une seule boîte pour deux yeux

    res = detect_landmarks(_IMG, ["Left Eye", "Right Eye"], detector=detector)

    # La seule boîte va au libellé « left » (plus à gauche) ; l'autre reste None.
    assert res["Left Eye"] == (15, 25)
    assert res["Right Eye"] is None


def test_result_preserves_label_order():
    def detector(image, queries):
        return [
            _det(0.9, "eye", _box(10, 20, 20, 30)),
            _det(0.8, "eye", _box(50, 20, 60, 30)),
            _det(0.7, "nose", _box(30, 40, 40, 50)),
        ]

    labels = ["Nose", "Left Eye", "Right Eye"]
    res = detect_landmarks(_IMG, labels, detector=detector)

    assert list(res) == labels


def test_custom_label_becomes_its_own_query():
    seen_queries = []

    def detector(image, queries):
        seen_queries.extend(queries)
        return [_det(0.9, "mouth", _box(40, 60, 50, 70))]

    res = detect_landmarks(_IMG, ["Mouth"], detector=detector)

    assert "mouth" in seen_queries
    assert res["Mouth"] == (45, 65)


def test_large_image_is_downscaled_and_coords_scaled_back():
    seen_shapes = []

    def detector(image, queries):
        seen_shapes.append(image.shape[:2])
        return [_det(0.9, "nose", _box(40, 40, 60, 60))]  # centre (50, 50) en espace réduit

    big = np.zeros((2000, 2000, 3), np.uint8)
    res = detect_landmarks(big, ["Nose"], detector=detector, max_side=100)

    assert max(seen_shapes[0]) == 100                 # le détecteur a vu l'image réduite
    assert res["Nose"] == (1000, 1000)                # (50,50) ×20 → repère d'origine


def test_small_image_is_not_downscaled():
    seen_shapes = []

    def detector(image, queries):
        seen_shapes.append(image.shape[:2])
        return [_det(0.9, "nose", _box(30, 40, 40, 50))]

    res = detect_landmarks(_IMG, ["Nose"], detector=detector, max_side=1536)

    assert seen_shapes[0] == (100, 100)               # pas de réduction
    assert res["Nose"] == (35, 45)


def test_grayscale_image_is_accepted():
    gray = np.zeros((100, 100), np.uint8)
    res = detect_landmarks(gray, ["Nose"], detector=lambda i, q: [_det(0.9, "nose", _box(30, 40, 40, 50))])

    assert res["Nose"] == (35, 45)

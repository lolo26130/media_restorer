"""Cœur de l'extension Signatures — sans Qt, sans modèle réel.

Aucun test ici ne charge OWL-ViT ni SigLIP, ni ne lance un vrai
``exiftool`` : détecteur et empreinteur sont de simples fonctions
injectées, déterministes.  Le choix du modèle par défaut
(``descriptors.DEFAULT_MODEL``) a été validé empiriquement AVANT ce code
— voir la docstring de :mod:`media_restorer.engines.signatures.descriptors`
et ``.claude/CLAUDE.md`` — ce fichier ne re-teste pas cette mesure, qui n'a
de sens qu'avec de vraies images.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from media_restorer.engines.signatures import library, matching, tags
from media_restorer.engines.signatures.location import Box, locate_signature
from media_restorer.engines.signatures.pipeline import (
    REASON_NO_LOCATION,
    REASON_NO_MATCH,
    scan_corpus,
)

# ---------------------------------------------------------------------------
# library — dédoublonnage et validation du nom
# ---------------------------------------------------------------------------


def test_add_entry_rejects_a_name_containing_a_digikam_separator():
    with pytest.raises(library.InvalidArtistName):
        library.validate_artist_name("Cabrol / Sennep")
    with pytest.raises(library.InvalidArtistName):
        library.validate_artist_name("Cabrol | Sennep")


def test_add_entry_rejects_an_empty_name():
    with pytest.raises(library.InvalidArtistName):
        library.validate_artist_name("   ")


def test_add_entry_deduplicates_within_the_same_artist(tmp_path):
    crop = np.full((10, 10, 3), 10, dtype="uint8")
    entry1, warning1 = library.add_entry(
        "Cabrol", crop, embedder=_same_vector_embedder(), root=tmp_path
    )
    entry2, warning2 = library.add_entry(
        "Cabrol", crop, embedder=_same_vector_embedder(),
        dedup_threshold=0.99, root=tmp_path,
    )
    assert entry2 == entry1
    assert warning1 is None and warning2 is None
    assert len(library.list_entries(root=tmp_path)) == 1


def _same_vector_embedder():
    """Empreinteur factice renvoyant TOUJOURS le même vecteur — dédoublonnage garanti."""

    def embedder(paths, on_progress=None):
        return np.stack([np.array([1.0, 0.0]) for _ in paths])

    return embedder


def _distinguishing_embedder():
    """Empreinteur factice : décode le PNG et code son niveau de gris moyen.

    Contrairement à ``_same_vector_embedder``, ceci lit vraiment le CONTENU
    du fichier (comme un vrai ``Embedder``) — deux crops visuellement
    différents obtiennent des empreintes différentes, deux crops identiques
    la même.
    """
    import cv2

    def embedder(paths, on_progress=None):
        vecs = []
        for p in paths:
            gray = float(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).mean())
            vecs.append(np.array([gray, 255.0 - gray]))
        vecs = np.stack(vecs)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    return embedder


def test_add_entry_warns_on_a_cross_artist_near_duplicate(tmp_path):
    crop = np.full((10, 10, 3), 10, dtype="uint8")
    embedder = _same_vector_embedder()

    entry_cabrol, warning = library.add_entry(
        "Cabrol", crop, embedder=embedder, root=tmp_path
    )
    assert warning is None

    # Un nom différent, un crop qui « ressemble » (même empreinte factice) :
    # le choix explicite de l'utilisateur est respecté (Sennep est bien créé),
    # mais l'appelant DOIT être averti — c'est le cas dangereux du plan.
    entry_sennep, warning = library.add_entry(
        "Sennep", crop, embedder=embedder, dedup_threshold=0.5, root=tmp_path
    )
    assert entry_sennep.artist == "Sennep"
    assert warning is not None
    assert warning.existing == entry_cabrol
    assert library.known_artists(root=tmp_path) == ["Cabrol", "Sennep"]


def test_add_entry_keeps_genuinely_different_crops(tmp_path):
    crop_a = np.full((10, 10, 3), 10, dtype="uint8")
    crop_b = np.full((10, 10, 3), 240, dtype="uint8")
    embedder = _distinguishing_embedder()

    library.add_entry("Cabrol", crop_a, embedder=embedder, root=tmp_path)
    entry_b, warning = library.add_entry(
        "Cabrol", crop_b, embedder=embedder, dedup_threshold=0.999, root=tmp_path
    )
    assert warning is None
    assert len(library.list_entries(root=tmp_path)) == 2
    assert entry_b.path.name == "0002.png"


# ---------------------------------------------------------------------------
# matching — classification pure, sur des scores synthétiques
# ---------------------------------------------------------------------------


def _candidate(artist: str, score: float, path="/x.png") -> matching.Candidate:
    from pathlib import Path

    return matching.Candidate(library.LibraryEntry(artist, Path(path)), score)


def test_classify_confident_when_a_single_artist_dominates():
    verdict = matching.classify(
        [_candidate("Cabrol", 0.95), _candidate("Cabrol", 0.90), _candidate("Sennep", 0.40)]
    )
    assert verdict.regime == matching.CONFIDENT
    assert verdict.artist == "Cabrol"


def test_classify_ambiguous_when_a_rival_is_too_close():
    verdict = matching.classify([_candidate("Cabrol", 0.90), _candidate("Sennep", 0.89)])
    assert verdict.regime == matching.AMBIGUOUS
    assert verdict.artist is None


def test_classify_no_match_below_the_confident_threshold():
    verdict = matching.classify([_candidate("Cabrol", 0.40)])
    assert verdict.regime == matching.NO_MATCH


def test_classify_no_match_on_an_empty_library():
    assert matching.classify([]).regime == matching.NO_MATCH


def test_rank_orders_by_decreasing_score():
    entries = [library.LibraryEntry("A", __import__("pathlib").Path("/a")),
               library.LibraryEntry("B", __import__("pathlib").Path("/b"))]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    candidates = matching.rank(np.array([0.9, 0.1]), entries, vectors)
    assert [c.entry.artist for c in candidates] == ["A", "B"]


# ---------------------------------------------------------------------------
# location — remise à l'échelle de la boîte (piège explicitement documenté)
# ---------------------------------------------------------------------------


def test_locate_signature_rescales_the_box_to_the_original_image():
    # Image d'origine deux fois plus grande que ce que la détection reçoit
    # (simulée via max_side) : la boîte détectée à l'échelle réduite doit
    # revenir remise à l'échelle des pixels d'origine.
    image = np.zeros((400, 800, 3), dtype="uint8")   # h=400, w=800

    def fake_detector(scaled_image, queries):
        sh, sw = scaled_image.shape[:2]
        assert (sh, sw) != (400, 800)          # bien réduite avant détection
        # Boîte occupant le quart supérieur-gauche de l'image RÉDUITE.
        return [{
            "score": 0.9, "label": "signature",
            "box": {"xmin": 0, "ymin": 0, "xmax": sw // 4, "ymax": sh // 4},
        }]

    box = locate_signature(image, detector=fake_detector, max_side=200)
    assert box is not None
    # Remise à l'échelle : un quart de la largeur/hauteur D'ORIGINE, pas de
    # la version réduite envoyée au détecteur.
    assert box.xmax == pytest.approx(200, abs=5)     # 800/4
    assert box.ymax == pytest.approx(100, abs=5)     # 400/4


def test_locate_signature_returns_none_below_min_score():
    image = np.zeros((100, 100, 3), dtype="uint8")

    def fake_detector(scaled_image, queries):
        return [{"score": 0.01, "label": "signature",
                  "box": {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10}}]

    assert locate_signature(image, detector=fake_detector, min_score=0.05) is None


def test_box_crop_extracts_the_right_region():
    image = np.arange(100).reshape(10, 10).astype("uint8")
    box = Box(xmin=2, ymin=3, xmax=5, ymax=6)
    crop = box.crop(image)
    assert crop.shape == (3, 3)
    assert np.array_equal(crop, image[3:6, 2:5])


# ---------------------------------------------------------------------------
# tags — écriture/lecture, non-destruction inter-branches
# ---------------------------------------------------------------------------


def _exiftool_json(**fields):
    return json.dumps([{"SourceFile": "x.jpg", **fields}])


def test_write_and_read_artist_round_trip():
    store: dict[str, list[str]] = {}

    def runner(args):
        assignations = [a for a in args if a.startswith("-") and "=" in a]
        if not assignations:
            return _exiftool_json(**store)
        nouveau: dict[str, list[str]] = {}
        for a in assignations:
            tag, _, valeur = a[1:].partition("=")
            nouveau.setdefault(tag.split(":")[-1], []).append(valeur)
        store.update({k: [v for v in vs if v] for k, vs in nouveau.items()})
        return "1 image files updated"

    tags.write_artist("/p.jpg", "Cabrol", runner=runner)
    assert tags.read_artist("/p.jpg", runner=runner) == "Cabrol"


def test_write_no_signature_round_trip():
    store: dict[str, list[str]] = {}

    def runner(args):
        assignations = [a for a in args if a.startswith("-") and "=" in a]
        if not assignations:
            return _exiftool_json(**store)
        for a in assignations:
            tag, _, valeur = a[1:].partition("=")
            store.setdefault(tag.split(":")[-1], []).append(valeur)
        return "1 image files updated"

    tags.write_no_signature("/p.jpg", runner=runner)
    assert tags.read_artist("/p.jpg", runner=runner) == tags.NO_SIGNATURE


def test_writing_an_artist_never_erases_another_branchs_tags():
    """Même verrou que ``test_two_owners_never_erase_each_other`` de digikam_tags.

    ``Dessinateur`` doit cohabiter avec ``Tri``/``Repère``/``Doublons`` sans
    jamais les effacer — sinon un scan de signatures détruirait en silence
    le pré-classement déjà fait sur le même corpus.
    """
    from media_restorer import digikam_tags as _tags

    store: dict[str, list[str]] = {}

    def runner(args):
        assignations = [a for a in args if a.startswith("-") and "=" in a]
        if not assignations:
            return _exiftool_json(**store)
        nouveau: dict[str, list[str]] = {}
        for a in assignations:
            tag, _, valeur = a[1:].partition("=")
            nouveau.setdefault(tag.split(":")[-1], []).append(valeur)
        store.update({k: [v for v in vs if v] for k, vs in nouveau.items()})
        return "1 image files updated"

    tri = _tags.branch_owner("Tri")
    _tags.write_tags("/p.jpg", [["media_restorer", "Tri", "Orientation", "Portrait"]],
                      owns=tri, runner=runner)
    tags.write_artist("/p.jpg", "Cabrol", runner=runner)

    raw = _tags.read_raw("/p.jpg", runner)
    assert _tags.read_tag_paths(raw, tri) == [["media_restorer", "Tri", "Orientation", "Portrait"]]
    assert _tags.read_tag_paths(raw, tags.owns) == [["media_restorer", "Dessinateur", "Cabrol"]]


# ---------------------------------------------------------------------------
# pipeline — orchestration, déterministe (détecteur/empreinteur injectés)
# ---------------------------------------------------------------------------


def _write_png(path, value=100):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((30, 30, 3), value, dtype="uint8"))


def test_scan_corpus_never_calls_a_real_model(tmp_path, monkeypatch):
    """Verrou de non-régression : aucun modèle réel n'est jamais construit ici."""
    img = tmp_path / "corpus" / "a.png"
    _write_png(img)

    def fake_detector(image, queries):
        return []   # jamais rien localisé

    def fake_embedder(paths, on_progress=None):
        return np.zeros((len(paths), 2))

    monkeypatch.setattr(
        "media_restorer.engines.signatures.pipeline.app_settings",
        lambda: __import__("types").SimpleNamespace(fileName=lambda: str(tmp_path / "settings.ini")),
    )

    outcomes = scan_corpus(
        img.parent, recursive=True, detector=fake_detector, embedder=fake_embedder,
    )
    assert len(outcomes) == 1
    assert outcomes[0].reason == REASON_NO_LOCATION
    assert outcomes[0].needs_review


def test_scan_corpus_skips_already_tagged_images_by_default(tmp_path, monkeypatch):
    img = tmp_path / "corpus" / "a.png"
    _write_png(img)

    monkeypatch.setattr(
        "media_restorer.engines.signatures.pipeline.app_settings",
        lambda: __import__("types").SimpleNamespace(fileName=lambda: str(tmp_path / "settings.ini")),
    )
    monkeypatch.setattr(
        "media_restorer.engines.signatures.pipeline.read_artist",
        lambda path, **kw: "Cabrol",   # déjà étiqueté
    )

    outcomes = scan_corpus(
        img.parent, recursive=True,
        detector=lambda *a, **k: [], embedder=lambda paths, **k: np.zeros((len(paths), 2)),
    )
    assert outcomes == []


def test_scan_outcome_reason_when_no_library_but_a_box_is_found(tmp_path, monkeypatch):
    img = tmp_path / "corpus" / "a.png"
    _write_png(img)

    monkeypatch.setattr(
        "media_restorer.engines.signatures.pipeline.app_settings",
        lambda: __import__("types").SimpleNamespace(fileName=lambda: str(tmp_path / "settings.ini")),
    )

    def fake_detector(image, queries):
        h, w = image.shape[:2]
        return [{"score": 0.9, "label": "signature",
                  "box": {"xmin": 0, "ymin": 0, "xmax": w // 2, "ymax": h // 2}}]

    outcomes = scan_corpus(
        img.parent, recursive=True, detector=fake_detector,
        embedder=lambda paths, **k: np.zeros((len(paths), 2)),
    )
    assert len(outcomes) == 1
    # Bibliothèque vide -> rien à comparer -> non reconnu, mais localisé.
    assert outcomes[0].reason == REASON_NO_MATCH
    assert outcomes[0].box is not None
    assert outcomes[0].crop_path is not None

"""Tests du tri grossier (media_restorer.engines.triage), sans Qt.

Deux niveaux :

* :mod:`~media_restorer.engines.triage.signals` est testé sur de **vraies
  images** de synthèse écrites en PNG (jamais en JPEG : la compression avec
  perte déplacerait les couleurs et fausserait les mesures de saturation) ;
* :mod:`~media_restorer.engines.triage.scan` est testé avec une fonction de
  mesure **injectée**, donc sans décoder la moindre image — c'est le parcours,
  l'échantillonnage et l'agrégation que l'on vérifie là.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from media_restorer.engines.triage import (
    CRITERIA,
    CRITERIA_BY_KEY,
    INK_DENSITIES,
    ORIENTATIONS,
    RESOLUTIONS,
    SUPPORTS,
    ImageSignals,
    TooLarge,
    format_summary,
    iter_images,
    measure_image,
    scan_directory,
    summarise,
)
from media_restorer.engines.triage import cache, criteria
from media_restorer.engines.triage import tags as triage_tags

PAPIER_BLANC = (255, 255, 255)
PAPIER_JAUNI = (232, 214, 168)      # jauni : écart max-min = 64
ENCRE_NOIRE = (25, 25, 25)          # écart 0
ENCRE_ROUGE = (200, 30, 30)         # écart 170


def _draw(path: Path, paper, ink, *, size=(200, 200), ink_rows=0.30) -> Path:
    """Écrit un PNG : bandes d'encre couvrant *ink_rows* de la hauteur.

    Les bandes occupent assez de surface (30 % par défaut) pour que le quintile
    le plus sombre ne contienne que de l'encre, et le reste que du papier.
    """
    w, h = size
    arr = np.zeros((h, w, 3), np.uint8)
    arr[:, :] = paper
    arr[: int(h * ink_rows), :] = ink
    Image.fromarray(arr).save(path)
    return path


# ---------------------------------------------------------------------------
# Chromatisme : encre colorée et papier jauni sont DEUX axes distincts
# ---------------------------------------------------------------------------

def test_black_ink_on_white_paper(tmp_path):
    s = measure_image(_draw(tmp_path / "a.png", PAPIER_BLANC, ENCRE_NOIRE))

    assert not s.coloured_ink
    assert not s.tinted_paper
    assert s.support == "trait noir / papier neutre"


def test_black_ink_on_yellowed_paper_is_not_mistaken_for_colour(tmp_path):
    """Le cas qui invalide la saturation moyenne : 36 % du corpus de référence."""
    s = measure_image(_draw(tmp_path / "b.png", PAPIER_JAUNI, ENCRE_NOIRE))

    assert not s.coloured_ink          # le dessin reste au trait noir…
    assert s.tinted_paper              # …seul le support est jauni
    assert s.support == "trait noir / papier jauni"


def test_coloured_ink_on_white_paper(tmp_path):
    s = measure_image(_draw(tmp_path / "c.png", PAPIER_BLANC, ENCRE_ROUGE))

    assert s.coloured_ink
    assert not s.tinted_paper
    assert s.support == "encre colorée / papier neutre"


def test_coloured_ink_on_yellowed_paper(tmp_path):
    s = measure_image(_draw(tmp_path / "d.png", PAPIER_JAUNI, ENCRE_ROUGE))

    assert s.coloured_ink and s.tinted_paper
    assert s.support == "encre colorée / papier jauni"


def test_every_support_label_is_reachable():
    """Les quatre libellés correspondent bien aux quatre combinaisons."""
    produced = {
        ImageSignals(Path("x"), 10, 10, 0.0, ink, paper).support
        for ink in (0.0, 100.0) for paper in (0.0, 100.0)
    }
    assert produced == set(SUPPORTS)


# ---------------------------------------------------------------------------
# Orientation et résolution : lues en taille NATIVE, pas sur la vignette
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("size,attendu", [
    ((200, 400), "portrait"),
    ((400, 200), "paysage"),
    ((300, 300), "carré"),
])
def test_orientation(tmp_path, size, attendu):
    s = measure_image(_draw(tmp_path / f"{attendu}.png", PAPIER_BLANC, ENCRE_NOIRE, size=size))

    assert s.orientation == attendu


def test_dimensions_are_native_not_thumbnail(tmp_path):
    """La vignette d'analyse ne doit jamais fuiter dans les dimensions publiées."""
    s = measure_image(_draw(tmp_path / "big.png", PAPIER_BLANC, ENCRE_NOIRE, size=(1200, 900)))

    assert (s.width, s.height) == (1200, 900)
    assert s.megapixels == pytest.approx(1.08)
    assert s.resolution_class == "1-6 Mpx"


def test_resolution_classes_cover_the_whole_range():
    for mpx, attendu in ((0.5, "< 1 Mpx"), (3, "1-6 Mpx"), (12, "6-20 Mpx"), (40, "> 20 Mpx")):
        side = int((mpx * 1e6) ** 0.5)
        assert ImageSignals(Path("x"), side, side, 0, 0, 0).resolution_class == attendu


# ---------------------------------------------------------------------------
# Densité d'encre
# ---------------------------------------------------------------------------

def test_ink_coverage_grows_with_the_inked_area(tmp_path):
    faible = measure_image(_draw(tmp_path / "f.png", PAPIER_BLANC, ENCRE_NOIRE, ink_rows=0.10))
    forte = measure_image(_draw(tmp_path / "g.png", PAPIER_BLANC, ENCRE_NOIRE, ink_rows=0.50))

    assert faible.ink_coverage < forte.ink_coverage
    assert faible.ink_coverage == pytest.approx(10, abs=2)
    assert forte.ink_coverage == pytest.approx(50, abs=2)


def test_ink_density_labels_follow_the_bins():
    for coverage, attendu in ((2, "très clair"), (10, "clair"), (25, "moyen"), (60, "dense")):
        assert ImageSignals(Path("x"), 10, 10, coverage, 0, 0).ink_density == attendu


def test_uniform_image_does_not_produce_nan(tmp_path):
    """Une image parfaitement unie vide un masque de quantile — sans NaN pour autant."""
    Image.fromarray(np.full((50, 50, 3), 200, np.uint8)).save(tmp_path / "uni.png")

    s = measure_image(tmp_path / "uni.png")

    assert not np.isnan([s.ink_saturation, s.paper_saturation, s.ink_coverage]).any()


# ---------------------------------------------------------------------------
# Parcours du corpus (fonction de mesure injectée : aucune image décodée)
# ---------------------------------------------------------------------------

def _fake_measure(path: Path) -> ImageSignals:
    return ImageSignals(path, 100, 200, 20.0, 0.0, 0.0)


def _touch(root: Path, *names: str) -> None:
    for name in names:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")


def test_iter_images_filters_suffixes_and_is_sorted(tmp_path):
    _touch(tmp_path, "b.jpg", "a.png", "notes.txt", "sous/c.TIF")

    found = iter_images(tmp_path)

    assert [p.name for p in found] == ["a.png", "b.jpg", "c.TIF"]  # trié, .txt exclu


def test_iter_images_skips_the_digikam_trash(tmp_path):
    _touch(tmp_path, "garde.jpg", ".dtrash/files/jete.jpg")

    assert [p.name for p in iter_images(tmp_path)] == ["garde.jpg"]


def test_iter_images_non_recursive_stays_at_the_top(tmp_path):
    _touch(tmp_path, "haut.jpg", "sous/bas.jpg")

    assert [p.name for p in iter_images(tmp_path, recursive=False)] == ["haut.jpg"]


def test_scan_uses_the_injected_measurer(tmp_path):
    _touch(tmp_path, "a.jpg", "b.jpg")

    result = scan_directory(tmp_path, measure=_fake_measure)

    assert len(result.signals) == 2
    assert all(s.width == 100 for s in result.signals)  # substitut, pas le disque
    assert result.found == 2 and not result.unreadable


def test_sampling_is_reproducible_for_a_given_seed(tmp_path):
    _touch(tmp_path, *[f"img{i:03d}.jpg" for i in range(50)])

    first = scan_directory(tmp_path, sample=10, seed=7, measure=_fake_measure)
    again = scan_directory(tmp_path, sample=10, seed=7, measure=_fake_measure)
    other = scan_directory(tmp_path, sample=10, seed=8, measure=_fake_measure)

    assert [s.path for s in first.signals] == [s.path for s in again.signals]
    assert [s.path for s in first.signals] != [s.path for s in other.signals]


def test_sample_larger_than_the_corpus_measures_everything(tmp_path):
    _touch(tmp_path, "a.jpg", "b.jpg")

    assert len(scan_directory(tmp_path, sample=99, measure=_fake_measure).signals) == 2


def test_unreadable_files_are_skipped_not_fatal(tmp_path):
    _touch(tmp_path, "ok.jpg", "casse.jpg")

    def measure(path):
        if path.name == "casse.jpg":
            raise OSError("fichier tronqué")
        return _fake_measure(path)

    result = scan_directory(tmp_path, measure=measure)

    assert [s.path.name for s in result.signals] == ["ok.jpg"]
    # L'incident est signalé, pas noyé : l'appelant peut alerter.
    assert [p.name for p in result.unreadable] == ["casse.jpg"]
    assert result.found == 2


def test_progress_is_reported_for_every_file_including_failures(tmp_path):
    _touch(tmp_path, "a.jpg", "casse.jpg", "c.jpg")
    seen = []

    def measure(path):
        if path.name == "casse.jpg":
            raise OSError
        return _fake_measure(path)

    scan_directory(tmp_path, measure=measure, on_progress=lambda i, n: seen.append((i, n)))

    assert seen == [(1, 3), (2, 3), (3, 3)]  # l'échec ne fait pas sauter un cran


# ---------------------------------------------------------------------------
# Agrégation
# ---------------------------------------------------------------------------

def test_summary_counts_every_axis():
    signals = [
        ImageSignals(Path("a"), 100, 200, 20.0, 0.0, 0.0),    # portrait, moyen
        ImageSignals(Path("b"), 200, 100, 2.0, 0.0, 0.0),     # paysage, très clair
    ]

    summary = summarise(signals)

    assert set(summary) == {"orientation", "densité d'encre", "résolution", "support"}
    assert summary["orientation"]["portrait"] == 1
    assert summary["orientation"]["paysage"] == 1
    assert summary["densité d'encre"]["moyen"] == 1


def test_summary_keeps_empty_classes():
    """Une classe à zéro est une information, pas une ligne à masquer."""
    summary = summarise([ImageSignals(Path("a"), 100, 200, 20.0, 0.0, 0.0)])

    assert summary["orientation"]["carré"] == 0
    assert set(summary["orientation"]) == set(ORIENTATIONS)
    assert set(summary["densité d'encre"]) == set(INK_DENSITIES)
    assert set(summary["résolution"]) == set(RESOLUTIONS)
    assert set(summary["support"]) == set(SUPPORTS)


def test_summary_of_nothing_does_not_divide_by_zero():
    summary = summarise([])

    assert all(count == 0 for counts in summary.values() for count in counts.values())
    assert "orientation" in format_summary(summary)   # rendu sans lever


def test_format_summary_shows_share_and_count():
    text = format_summary(summarise([ImageSignals(Path("a"), 100, 200, 20.0, 0.0, 0.0)]))

    assert "portrait" in text
    assert "100.0 %" in text


# ---------------------------------------------------------------------------
# Plafond de résolution — écarté AVANT tout décodage
# ---------------------------------------------------------------------------

def test_measure_raises_too_large_before_decoding(tmp_path):
    """L'en-tête suffit à trancher : aucun pixel n'est décodé pour rien."""
    img = _draw(tmp_path / "enorme.png", PAPIER_BLANC, ENCRE_NOIRE, size=(3000, 2000))

    with pytest.raises(TooLarge):
        measure_image(img, max_megapixels=5)          # 6 Mpx > 5

    assert measure_image(img, max_megapixels=10).megapixels == pytest.approx(6.0)


def test_scan_lists_oversized_images_apart_from_unreadable_ones(tmp_path):
    """Un choix de l'utilisateur et un incident ne doivent pas être confondus."""
    _draw(tmp_path / "petite.png", PAPIER_BLANC, ENCRE_NOIRE, size=(100, 100))
    _draw(tmp_path / "enorme.png", PAPIER_BLANC, ENCRE_NOIRE, size=(3000, 2000))
    (tmp_path / "casse.jpg").write_bytes(b"ceci n'est pas une image")

    result = scan_directory(tmp_path, max_megapixels=5)

    assert [s.path.name for s in result.signals] == ["petite.png"]
    assert [p.name for p in result.skipped_large] == ["enorme.png"]
    assert [p.name for p in result.unreadable] == ["casse.jpg"]
    assert result.found == 3


def test_no_cap_measures_everything(tmp_path):
    _draw(tmp_path / "enorme.png", PAPIER_BLANC, ENCRE_NOIRE, size=(3000, 2000))

    result = scan_directory(tmp_path)

    assert len(result.signals) == 1 and not result.skipped_large


# ---------------------------------------------------------------------------
# Catalogue de critères
# ---------------------------------------------------------------------------

def test_every_class_label_is_usable_as_a_digikam_tag():
    """« / » est le séparateur de TagsList : un libellé qui en contient casse l'arbre."""
    assert criteria.check_labels() == []


def test_class_labels_are_self_sufficient():
    """dc:Subject est plat : seule la feuille survit, elle doit se suffire.

    « Moyen » seul ne voudrait rien dire ; « Encre moyenne » si.
    """
    densite = CRITERIA_BY_KEY["ink_density"]
    assert all(label.startswith("Encre") for label in densite.classes)


def test_every_criterion_extracts_one_of_its_own_classes():
    """Aucun critère ne peut produire une valeur absente de ses classes."""
    echantillons = [
        ImageSignals(Path("x"), w, h, cov, ink, paper)
        for w, h in ((100, 200), (200, 100), (100, 100))
        for cov in (2.0, 25.0, 60.0)
        for ink in (0.0, 100.0)
        for paper in (0.0, 100.0)
    ]
    for criterion in CRITERIA:
        produits = {criterion.extract(s) for s in echantillons}
        assert produits <= set(criterion.classes), criterion.key


def test_select_keeps_the_catalogue_order_not_the_caller_order():
    """Colonnes et étiquettes restent stables quel que soit l'ordre des cases cochées."""
    choisis = criteria.select(["resolution", "orientation"])

    assert [c.key for c in choisis] == ["orientation", "resolution"]


def test_select_ignores_an_unknown_key():
    """Une configuration persistée peut mentionner un critère retiré depuis."""
    assert [c.key for c in criteria.select(["orientation", "disparu"])] == ["orientation"]


def test_criteria_for_filters_on_method():
    assert criteria.criteria_for(criteria.METHOD_SIGNALS) == CRITERIA
    assert criteria.criteria_for("clip") == ()          # crochet, rien encore


# ---------------------------------------------------------------------------
# Étiquettes de tri
# ---------------------------------------------------------------------------

def test_tag_paths_sit_under_the_triage_branch():
    s = ImageSignals(Path("x"), 200, 100, 25.0, 0.0, 100.0)   # paysage, jauni

    chemins = triage_tags.tag_paths(s, criteria.select(["orientation", "paper"]))

    assert chemins == [
        ["media_restorer", "Tri", "Orientation", "Paysage"],
        ["media_restorer", "Tri", "Papier", "Papier jauni"],
    ]


def test_triage_owns_only_its_own_branch():
    """Sans cette restriction, écrire un tri effacerait les repères."""
    assert triage_tags.owns(["media_restorer", "Tri", "Orientation", "Paysage"])
    assert not triage_tags.owns(["media_restorer", "Repère", "Left Eye"])


def test_write_and_read_classification_round_trip(tmp_path):
    from media_restorer import digikam_tags as T

    store: dict[str, list[str]] = {}

    def runner(args):
        assignations = [a for a in args if a.startswith("-") and "=" in a]
        if not assignations:
            return json.dumps([{"SourceFile": "x.jpg", **store}])
        nouveau: dict[str, list[str]] = {}
        for a in assignations:
            tag, _, valeur = a[1:].partition("=")
            nouveau.setdefault(tag.split(":")[-1], []).append(valeur)
        store.update({k: [v for v in vs if v] for k, vs in nouveau.items()})
        return ""

    s = ImageSignals(Path("x"), 200, 100, 25.0, 0.0, 100.0)
    triage_tags.write_signals(tmp_path / "p.jpg", s, runner=runner)

    assert triage_tags.read_classification(tmp_path / "p.jpg", runner=runner) == {
        "Orientation": "Paysage",
        "Densité d'encre": "Encre moyenne",
        "Couleur d'encre": "Trait noir",
        "Papier": "Papier jauni",
        "Résolution": "< 1 Mpx",
    }


# ---------------------------------------------------------------------------
# Cache des mesures
# ---------------------------------------------------------------------------

def test_cache_round_trips_measures(tmp_path):
    img = _draw(tmp_path / "a.png", PAPIER_BLANC, ENCRE_NOIRE)
    fichier = tmp_path / "cache.json"
    mesure = measure_image(img)

    cache.save([mesure], fichier)
    relu = cache.load(fichier)

    assert relu[img].ink_coverage == pytest.approx(mesure.ink_coverage)
    assert relu[img].width == mesure.width


def test_cache_entry_is_invalidated_when_the_file_changes(tmp_path):
    img = _draw(tmp_path / "a.png", PAPIER_BLANC, ENCRE_NOIRE)
    fichier = tmp_path / "cache.json"
    cache.save([measure_image(img)], fichier)

    # Le fichier est remplacé : taille et mtime changent, l'entrée doit tomber.
    _draw(img, PAPIER_JAUNI, ENCRE_ROUGE, size=(300, 300))

    assert cache.load(fichier) == {}


def test_cache_ignores_a_vanished_file(tmp_path):
    img = _draw(tmp_path / "a.png", PAPIER_BLANC, ENCRE_NOIRE)
    fichier = tmp_path / "cache.json"
    cache.save([measure_image(img)], fichier)
    img.unlink()

    assert cache.load(fichier) == {}


def test_cache_never_raises_on_a_damaged_file(tmp_path):
    fichier = tmp_path / "cache.json"
    fichier.write_text("{ pas du json", encoding="utf-8")

    assert cache.load(fichier) == {}          # au pire on remesure, jamais faux
    assert cache.load(tmp_path / "absent.json") == {}


def test_cache_of_another_format_version_is_discarded(tmp_path):
    fichier = tmp_path / "cache.json"
    fichier.write_text(json.dumps({"version": 999, "entries": {}}), encoding="utf-8")

    assert cache.load(fichier) == {}

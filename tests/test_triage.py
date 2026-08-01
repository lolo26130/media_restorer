"""Tests du tri grossier (media_restorer.engines.triage), sans Qt.

Deux niveaux :

* :mod:`~media_restorer.engines.triage.signals` est testé sur de **vraies
  images** de synthèse écrites en PNG (jamais en JPEG : la compression avec
  perte déplacerait les couleurs et fausserait les mesures de saturation) ;
* :mod:`~media_restorer.engines.triage.scan` est testé avec une fonction de
  mesure **injectée**, donc sans décoder la moindre image — c'est le parcours,
  l'échantillonnage et l'agrégation que l'on vérifie là.
"""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from media_restorer.engines.triage import (
    INK_DENSITIES,
    ORIENTATIONS,
    RESOLUTIONS,
    SUPPORTS,
    ImageSignals,
    format_summary,
    iter_images,
    measure_image,
    scan_directory,
    summarise,
)

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

    signals = scan_directory(tmp_path, measure=_fake_measure)

    assert len(signals) == 2
    assert all(s.width == 100 for s in signals)  # valeurs du substitut, pas du disque


def test_sampling_is_reproducible_for_a_given_seed(tmp_path):
    _touch(tmp_path, *[f"img{i:03d}.jpg" for i in range(50)])

    first = scan_directory(tmp_path, sample=10, seed=7, measure=_fake_measure)
    again = scan_directory(tmp_path, sample=10, seed=7, measure=_fake_measure)
    other = scan_directory(tmp_path, sample=10, seed=8, measure=_fake_measure)

    assert [s.path for s in first] == [s.path for s in again]
    assert [s.path for s in first] != [s.path for s in other]


def test_sample_larger_than_the_corpus_measures_everything(tmp_path):
    _touch(tmp_path, "a.jpg", "b.jpg")

    assert len(scan_directory(tmp_path, sample=99, measure=_fake_measure)) == 2


def test_unreadable_files_are_skipped_not_fatal(tmp_path):
    _touch(tmp_path, "ok.jpg", "casse.jpg")

    def measure(path):
        if path.name == "casse.jpg":
            raise OSError("fichier tronqué")
        return _fake_measure(path)

    signals = scan_directory(tmp_path, measure=measure)

    assert [s.path.name for s in signals] == ["ok.jpg"]


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

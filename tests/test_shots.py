"""Tests de l'inventaire des correspondances RAW ↔ JPEG.

Sans Qt.  ``exiftool`` est toujours **injecté** : aucun test ne lance le binaire
ni ne lit de vrai fichier RAW.

Le test qui porte l'essentiel du risque est
:func:`test_same_name_but_different_shots_are_not_paired` — le compteur des
boîtiers Nikon reboucle, si bien qu'apparier par le nom produit de faux
appariements *silencieux*.
"""
import json
from pathlib import Path

import pytest

from media_restorer.engines.shots import (
    MATCH_NAME,
    MATCH_SURE,
    MATCH_TIME,
    RAW_SUFFIXES,
    ShotInventory,
    ShotPair,
    is_reliable,
    iter_files,
    keys_of,
    name_key,
    scan_shots,
    sure_key,
    time_key,
)
from media_restorer.engines.shots import preview as _preview
from media_restorer.engines.shots import sidecars


def _meta(path, **champs):
    """Un enregistrement exiftool."""
    return {"SourceFile": str(path), **champs}


def _runner(enregistrements):
    """Faux exiftool : rend les enregistrements des fichiers demandés."""
    par_chemin = {r["SourceFile"]: r for r in enregistrements}

    def runner(args):
        demandes = [a for a in args if not a.startswith("-")]
        return json.dumps([par_chemin[d] for d in demandes if d in par_chemin])

    return runner


def _touch(root: Path, *noms: str) -> list[Path]:
    crees = []
    for nom in noms:
        cible = root / nom
        cible.parent.mkdir(parents=True, exist_ok=True)
        cible.write_bytes(b"x")
        crees.append(cible)
    return crees


# ---------------------------------------------------------------------------
# Les RAW restent hors des autres chaînes
# ---------------------------------------------------------------------------

def test_raw_suffixes_are_disjoint_from_the_processing_chain():
    """Un .NEF ne doit JAMAIS entrer dans le pré-classement ni les doublons."""
    from media_restorer.engines.triage import IMAGE_SUFFIXES

    assert RAW_SUFFIXES & IMAGE_SUFFIXES == set()
    assert ".nef" in RAW_SUFFIXES and ".nef" not in IMAGE_SUFFIXES


def test_iter_files_separates_raw_from_derived(tmp_path):
    _touch(tmp_path, "a.NEF", "a.JPG", "b.nef", "notes.txt", "NEF/c.nef", ".dtrash/d.nef")

    raws, derives = iter_files(tmp_path)

    assert sorted(p.name for p in raws) == ["a.NEF", "b.nef", "c.nef"]  # corbeille exclue
    assert [p.name for p in derives] == ["a.JPG"]


# ---------------------------------------------------------------------------
# Les clés, et leur ordre de fiabilité
# ---------------------------------------------------------------------------

def test_sure_key_needs_both_serial_and_shutter_count():
    assert sure_key({"SerialNumber": "600", "ShutterCount": "14427"}) is not None
    assert sure_key({"SerialNumber": "600"}) is None
    assert sure_key({"ShutterCount": "14427"}) is None


def test_time_key_includes_the_camera_model():
    """Deux boîtiers peuvent déclencher à la même seconde."""
    a = time_key({"Model": "D800", "DateTimeOriginal": "2018:04:29 22:26:40"})
    b = time_key({"Model": "D70s", "DateTimeOriginal": "2018:04:29 22:26:40"})

    assert a != b


def test_keys_are_ordered_from_most_to_least_reliable():
    """C'est cet ordre qui garantit qu'une clé sûre l'emporte toujours."""
    cles = keys_of(Path("/c/DSC_1.NEF"),
                   {"SerialNumber": "600", "ShutterCount": "1",
                    "Model": "D800", "DateTimeOriginal": "2018:01:01 00:00:00"})

    assert [c[0] for c in cles] == [MATCH_SURE, MATCH_TIME, MATCH_NAME]


def test_name_is_never_considered_reliable():
    assert is_reliable(MATCH_SURE) and is_reliable(MATCH_TIME)
    assert not is_reliable(MATCH_NAME)


def test_name_key_ignores_case_and_extension():
    assert name_key(Path("/c/DSC_1.NEF")) == name_key(Path("/autre/dsc_1.jpg"))


# ---------------------------------------------------------------------------
# LE test : le compteur Nikon reboucle
# ---------------------------------------------------------------------------

def test_same_name_but_different_shots_are_not_paired(tmp_path):
    """Le cas réel ``330ND800/DSC_6769.nef`` vs ``329ND800/DSC_6769.jpg``.

    Mêmes noms, prises différentes.  Les deux portent une clé sûre, et ces clés
    diffèrent : l'appariement par le nom ne doit **pas** avoir lieu, sans quoi
    l'inventaire affirmerait une correspondance fausse.
    """
    nef, jpg = _touch(tmp_path, "330ND800/DSC_6769.nef", "329ND800/DSC_6769.jpg")
    runner = _runner([
        _meta(nef, SerialNumber="600", ShutterCount="18245", Model="NIKON D800",
              DateTimeOriginal="2020:03:02 21:49:20"),
        _meta(jpg, SerialNumber="600", ShutterCount="99999", Model="NIKON D800",
              DateTimeOriginal="2019:01:01 10:00:00"),
    ])

    inv = scan_shots(tmp_path, runner=runner)

    assert inv.pairs == []                       # aucune paire fiable
    assert inv.raw_only == [nef]                 # le RAW est bien orphelin
    assert inv.jpeg_only == [jpg]
    assert inv.unconfirmed == []                 # et pas d'appariement par nom


def test_a_shared_shutter_count_pairs_them(tmp_path):
    """Le cas normal : même boîtier, même déclenchement."""
    nef, jpg = _touch(tmp_path, "NEF/DSC_4149.NEF", "DSC_4149.JPG")
    runner = _runner([
        _meta(nef, SerialNumber="600", ShutterCount="14427", Model="NIKON D800"),
        _meta(jpg, SerialNumber="600", ShutterCount="14427", Model="NIKON D800"),
    ])

    inv = scan_shots(tmp_path, runner=runner)

    assert len(inv.pairs) == 1
    paire = inv.pairs[0]
    assert (paire.raw, paire.derived) == (nef, jpg)
    assert paire.method == MATCH_SURE and paire.reliable


def test_pairing_works_across_folders(tmp_path):
    """Le JPEG n'est pas forcément à côté de son RAW."""
    nef, jpg = _touch(tmp_path, "un/deux/NEF/x.nef", "tout/ailleurs/y.jpg")
    runner = _runner([
        _meta(nef, SerialNumber="7", ShutterCount="42"),
        _meta(jpg, SerialNumber="7", ShutterCount="42"),
    ])

    assert scan_shots(tmp_path, runner=runner).pairs[0].derived == jpg


def test_time_key_is_used_when_makernotes_are_missing(tmp_path):
    """Un JPEG ré-exporté perd ses MakerNotes, donc son ShutterCount."""
    nef, jpg = _touch(tmp_path, "a.nef", "b.jpg")
    runner = _runner([
        _meta(nef, SerialNumber="7", ShutterCount="42", Model="D800",
              DateTimeOriginal="2018:04:29 22:26:40", SubSecTimeOriginal="70"),
        _meta(jpg, Model="D800", DateTimeOriginal="2018:04:29 22:26:40",
              SubSecTimeOriginal="70"),
    ])

    inv = scan_shots(tmp_path, runner=runner)

    assert len(inv.pairs) == 1
    assert inv.pairs[0].method == MATCH_TIME


def test_name_match_lands_in_unconfirmed_not_pairs(tmp_path):
    """Sans métadonnées, l'appariement par nom est un candidat, pas un résultat."""
    nef, jpg = _touch(tmp_path, "DSC_1365.NEF", "DSC_1365.JPG")
    runner = _runner([_meta(nef), _meta(jpg)])

    inv = scan_shots(tmp_path, runner=runner)

    assert inv.pairs == []
    assert len(inv.unconfirmed) == 1
    assert inv.unconfirmed[0].method == MATCH_NAME
    assert not inv.unconfirmed[0].reliable


# ---------------------------------------------------------------------------
# Les quatre catégories
# ---------------------------------------------------------------------------

def test_categories_are_exhaustive_and_disjoint(tmp_path):
    """Tout fichier rencontré apparaît dans une catégorie, et une seule."""
    paire_n, paire_j, seul_n, seul_j = _touch(
        tmp_path, "p.nef", "p.jpg", "orphelin.nef", "scan.jpg")
    runner = _runner([
        _meta(paire_n, SerialNumber="1", ShutterCount="10"),
        _meta(paire_j, SerialNumber="1", ShutterCount="10"),
        _meta(seul_n, SerialNumber="1", ShutterCount="20"),
        _meta(seul_j),
    ])

    inv = scan_shots(tmp_path, runner=runner)

    assert [p.raw for p in inv.pairs] == [paire_n]
    assert inv.raw_only == [seul_n]
    assert inv.jpeg_only == [seul_j]
    assert inv.unconfirmed == []
    assert inv.n_raw == 2 and inv.n_derived == 2


def test_summary_highlights_the_actionable_category(tmp_path):
    """« RAW sans JPEG » est ce que l'utilisateur veut voir en premier."""
    inv = ShotInventory(pairs=[], raw_only=[Path("a.nef")], jpeg_only=[], unconfirmed=[])

    assert "1 RAW sans JPEG" in inv.summary()


def test_summary_stays_silent_about_empty_categories():
    inv = ShotInventory(pairs=[ShotPair(Path("a.nef"), Path("a.jpg"), MATCH_SURE)],
                        raw_only=[], jpeg_only=[], unconfirmed=[])

    resume = inv.summary()

    assert "100.0 % appariés" in resume
    assert "à confirmer" not in resume


def test_pair_describes_why_it_was_matched():
    """L'interface doit pouvoir justifier l'appariement affiché."""
    paire = ShotPair(Path("a.nef"), Path("a.jpg"), MATCH_SURE,
                     meta={"Model": "NIKON D800", "ShutterCount": "14427",
                           "DateTimeOriginal": "2018:04:29 22:26:40"})

    phrase = paire.describe()

    assert "NIKON D800" in phrase and "14427" in phrase
    assert "série" in phrase                  # la méthode d'appariement est dite


def test_an_unreadable_batch_does_not_abort_the_scan(tmp_path):
    """Mieux vaut un inventaire partiel qu'aucun inventaire."""
    _touch(tmp_path, "a.nef", "a.jpg")

    def runner(args):
        raise RuntimeError("exiftool a échoué")

    inv = scan_shots(tmp_path, runner=runner)

    # Sans métadonnées, il ne reste que le nom : candidat à confirmer.
    assert len(inv.unconfirmed) == 1 and inv.pairs == []


# ---------------------------------------------------------------------------
# Aperçu — jamais Pillow sur un RAW
# ---------------------------------------------------------------------------

def test_preview_tries_the_largest_embedded_image_first():
    demandes = []

    def runner(args):
        demandes.append([a for a in args if a.startswith("-") and a != "-b"])
        return b"\xff\xd8" + b"x" * 4096          # assez gros pour être retenu

    _preview.extract_preview(Path("/c/a.nef"), runner=runner, cache=False)

    assert demandes[0] == ["-JpgFromRaw"]         # l'aperçu pleine taille d'abord


def test_preview_falls_back_through_the_tags():
    appels = []

    def runner(args):
        balise = [a for a in args if a.startswith("-") and a != "-b"][0]
        appels.append(balise)
        return b"" if balise == "-JpgFromRaw" else b"\xff\xd8" + b"x" * 4096

    resultat = _preview.extract_preview(Path("/c/a.nef"), runner=runner, cache=False)

    assert appels[:2] == ["-JpgFromRaw", "-PreviewImage"]
    assert resultat is not None


def test_preview_rejects_a_truncated_payload():
    """Quelques octets ne sont pas une image : mieux vaut essayer la suite."""
    def runner(args):
        return b"\xff\xd8"                        # 2 octets

    assert _preview.extract_preview(Path("/c/a.nef"), runner=runner, cache=False) is None


def test_preview_never_raises_on_a_failing_runner():
    def boom(args):
        raise RuntimeError("exiftool absent")

    assert _preview.extract_preview(Path("/c/a.nef"), runner=boom, cache=False) is None


def test_cached_preview_name_changes_when_the_file_changes(tmp_path):
    """L'empreinte (taille, mtime) est dans le nom : pas d'index à tenir."""
    fichier = tmp_path / "a.nef"
    fichier.write_bytes(b"x" * 10)
    avant = _preview.cached_path(fichier)

    fichier.write_bytes(b"y" * 999)
    apres = _preview.cached_path(fichier)

    assert avant != apres


# ---------------------------------------------------------------------------
# Développé ou jamais développé — l'annexe de développement
# ---------------------------------------------------------------------------

def test_the_sidecar_is_named_after_the_full_raw_name(tmp_path):
    """Convention relevée sur le corpus : ``DSC_1.NEF.xmp``, jamais ``DSC_1.xmp``.

    1 050 des 1 877 NEF portent la première forme, aucun la seconde.  Se
    tromper de convention rendrait « jamais développé » pour tout le corpus,
    sans la moindre erreur.
    """
    candidats = sidecars.sidecar_candidates(tmp_path / "DSC_1.NEF")

    assert "dsc_1.nef.xmp" in candidats
    assert "dsc_1.xmp" not in candidats


def test_a_raw_is_developed_when_a_sidecar_sits_beside_it(tmp_path):
    brut, = _touch(tmp_path, "a.nef")
    _touch(tmp_path, "a.nef.xmp")
    autre, = _touch(tmp_path, "b.nef")

    assert sidecars.is_developed(brut) is True
    assert sidecars.is_developed(autre) is False


def test_a_sidecar_in_another_folder_does_not_count(tmp_path):
    """L'annexe appartient au RAW qu'elle jouxte, pas à son homonyme ailleurs."""
    brut, = _touch(tmp_path, "330ND800/DSC_6769.nef")
    _touch(tmp_path, "329ND800/DSC_6769.nef.xmp")

    index = sidecars.index_sidecars(sidecars.iter_sidecars(tmp_path))
    assert sidecars.is_developed(brut, index) is False


def test_sidecar_case_is_ignored(tmp_path):
    """La casse est mêlée dans le corpus : ``.XMP`` accompagne aussi ``.NEF``."""
    brut, = _touch(tmp_path, "A.NEF")
    _touch(tmp_path, "A.NEF.XMP")

    assert sidecars.is_developed(brut) is True
    index = sidecars.index_sidecars(sidecars.iter_sidecars(tmp_path))
    assert sidecars.is_developed(brut, index) is True


def test_the_scan_splits_orphan_raws_by_development(tmp_path):
    """La distinction demandée : reste-t-il tout à faire, ou juste à exporter ?"""
    fait, a_faire = _touch(tmp_path, "fait.nef", "a_faire.nef")
    _touch(tmp_path, "fait.nef.pp3")
    runner = _runner([
        _meta(fait, SerialNumber="1", ShutterCount="10"),
        _meta(a_faire, SerialNumber="1", ShutterCount="20"),
    ])

    inv = scan_shots(tmp_path, runner=runner)

    assert sorted(inv.raw_only) == sorted([fait, a_faire])
    assert inv.raw_only_developed == [fait]
    assert inv.raw_only_never_developed == [a_faire]
    # Les deux sous-ensembles partitionnent raw_only — aucune perte, aucun doublon.
    assert sorted(inv.raw_only_developed + inv.raw_only_never_developed) == sorted(inv.raw_only)


def test_a_paired_raw_carries_its_development_status(tmp_path):
    """Le JPEG existe déjà ; savoir s'il vient d'un développement reste utile."""
    brut, jpeg = _touch(tmp_path, "p.nef", "p.jpg")
    _touch(tmp_path, "p.nef.xmp")
    runner = _runner([
        _meta(brut, SerialNumber="1", ShutterCount="10"),
        _meta(jpeg, SerialNumber="1", ShutterCount="10"),
    ])

    (paire,) = scan_shots(tmp_path, runner=runner).pairs
    assert paire.developed is True

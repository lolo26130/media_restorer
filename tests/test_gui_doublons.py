"""Tests de la fenêtre Doublons (recherche, revue, étiquetage).

Qt offscreen.  La chaîne de détection est toujours **injectée** (aucune image
décodée, aucun appariement) et ``exiftool`` aussi (aucun fichier modifié).  Les
workers sont rendus synchrones — ``start`` → ``run``, jamais de vrai thread OS :
voir le piège de segfault documenté dans ``.claude/CLAUDE.md``.
"""
from pathlib import Path

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox

import media_restorer.extensions.doublons  # noqa: F401 — enregistre l'extension
from media_restorer.engines.duplicates import (
    METHODS,
    REGIME_GEOMETRIQUE,
    REGIME_PARTIEL,
    DuplicateGraph,
    Merit,
    Pair,
    build_graph,
)
from media_restorer.extensions.doublons.gui import (
    DoublonsGUI,
    _SearchWorker,
    _WriteWorker,
)


@pytest.fixture(autouse=True)
def _flush_qt_deletions():
    """Purge les suppressions Qt différées : deux ``pg.ImageView`` par fenêtre."""
    yield
    import gc

    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is not None:
        app.processEvents()
        gc.collect()
        app.processEvents()


@pytest.fixture(autouse=True)
def _sync_workers(monkeypatch):
    monkeypatch.setattr(_SearchWorker, "start", _SearchWorker.run)
    monkeypatch.setattr(_WriteWorker, "start", _WriteWorker.run)


def _pair(a: str, b: str, regime=REGIME_GEOMETRIQUE, merite=0.9) -> Pair:
    partiel = regime == REGIME_PARTIEL
    return Pair(a=Path(a), b=Path(b), merit=Merit(
        regime=regime, merite=merite, n_inliers=200,
        rotation_deg=15.0, echelle=0.25 if partiel else 1.0,
        couverture_a_dans_b=0.25 if partiel else 1.0,
        couverture_b_dans_a=1.0,
    ))


def _graph() -> DuplicateGraph:
    """Un groupe de deux images, plus une inclusion."""
    return build_graph(
        [_pair("/c/a.jpg", "/c/b.jpg"),
         _pair("/c/page.jpg", "/c/detail.jpg", regime=REGIME_PARTIEL)],
        resolutions={Path("/c/a.jpg"): 100, Path("/c/b.jpg"): 900},
    )


def _fake_search(*_args, **kwargs):
    on_progress = kwargs.get("on_progress")
    on_stage = kwargs.get("on_stage")
    if on_stage:
        on_stage("Description…")
    if on_progress:
        on_progress(1, 2)
        on_progress(2, 2)
    return _graph()


def _make_window(qtbot, tmp_path, *, search_fn=_fake_search, runner=None, target=None):
    win = DoublonsGUI(
        target_path=tmp_path if target is None else target,
        recursive=True, search_fn=search_fn,
        exiftool_runner=runner or (lambda args: '[{"SourceFile": "x.jpg"}]'),
    )
    qtbot.addWidget(win)
    return win


# ---------------------------------------------------------------------------
# Panneau de méthodes — engendré depuis le catalogue
# ---------------------------------------------------------------------------

def test_every_catalogue_method_gets_a_checkbox(qtbot, tmp_path):
    """Ajouter une méthode au catalogue doit suffire à la faire apparaître."""
    win = _make_window(qtbot, tmp_path)

    for m in METHODS:
        assert win._param_root.child(m.stage, m.key) is not None


def test_methods_are_grouped_by_stage(qtbot, tmp_path):
    win = _make_window(qtbot, tmp_path)

    etages = [c.name() for c in win._param_root.children() if c.type() == "group"]
    assert etages == ["prefilter", "candidate", "verify"]


def test_expensive_method_is_unchecked_by_default(qtbot, tmp_path):
    """SIFT est 8× plus coûteux qu'ORB : à cocher sciemment."""
    win = _make_window(qtbot, tmp_path)

    assert win._param_root["verify", "orb_magsac"] is True
    assert win._param_root["verify", "sift_magsac"] is False


def test_selection_follows_the_catalogue_order(qtbot, tmp_path):
    win = _make_window(qtbot, tmp_path)
    win._param_root.child("candidate", "fourier_mellin").setValue(False)

    choisies = win._selected_methods()

    assert "fourier_mellin" not in choisies
    assert list(choisies) == [m.key for m in METHODS if m.key in set(choisies)]


def test_searching_without_verification_is_refused(qtbot, tmp_path, monkeypatch):
    """Sans vérification géométrique, aucun mérite ne peut être calculé."""
    win = _make_window(qtbot, tmp_path)
    for cle in ("orb_magsac", "sift_magsac"):
        win._param_root.child("verify", cle).setValue(False)
    vus = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **kw: vus.append(a))

    win.on_actionSearch_triggered()

    assert vus
    assert not win.actionWriteTags.isEnabled()


# ---------------------------------------------------------------------------
# Recherche et arbre de résultats
# ---------------------------------------------------------------------------

def test_actions_unlock_only_after_a_search(qtbot, tmp_path):
    win = _make_window(qtbot, tmp_path)
    assert win.actionSearch.isEnabled()
    assert not win.actionWriteTags.isEnabled()

    win.on_actionSearch_triggered()

    assert win.actionWriteTags.isEnabled()
    assert win.actionExport.isEnabled()


def test_a_single_file_target_cannot_be_searched(qtbot, tmp_path):
    fichier = tmp_path / "une.jpg"
    fichier.write_bytes(b"x")

    win = _make_window(qtbot, tmp_path, target=fichier)

    assert not win.actionSearch.isEnabled()


def test_tree_separates_groups_from_inclusions(qtbot, tmp_path):
    """La distinction structurelle doit rester visible à l'écran."""
    win = _make_window(qtbot, tmp_path)
    win.on_actionSearch_triggered()

    arbre = win._ui.treeGroups
    racines = [arbre.topLevelItem(i).text(0) for i in range(arbre.topLevelItemCount())]
    assert any(r.startswith("Groupes") for r in racines)
    assert any(r.startswith("Inclusions") for r in racines)


def test_status_summarises_the_graph(qtbot, tmp_path):
    win = _make_window(qtbot, tmp_path)
    win.on_actionSearch_triggered()

    message = win.statusBar().currentMessage()
    assert "groupe(s)" in message and "inclusion(s)" in message


# ---------------------------------------------------------------------------
# Revue : l'explication en français
# ---------------------------------------------------------------------------

def _first_pair_item(win):
    arbre = win._ui.treeGroups
    for i in range(arbre.topLevelItemCount()):
        racine = arbre.topLevelItem(i)
        for j in range(racine.childCount()):
            noeud = racine.child(j)
            if noeud.data(0, Qt.ItemDataRole.UserRole) is not None:
                return noeud
            for k in range(noeud.childCount()):
                if noeud.child(k).data(0, Qt.ItemDataRole.UserRole) is not None:
                    return noeud.child(k)
    return None


def test_selecting_a_pair_shows_both_images_and_the_sentence(qtbot, tmp_path):
    """« rotation 15°, couverture 25 % » se vérifie ; « score 0,83 » non."""
    win = _make_window(qtbot, tmp_path)
    win.on_actionSearch_triggered()

    item = _first_pair_item(win)
    assert item is not None
    win._ui.treeGroups.setCurrentItem(item)

    texte = win._explanation.text()
    assert "points concordants" in texte
    assert "dessin" in texte or "détail" in texte


def test_selection_drives_the_root_docks(qtbot, tmp_path):
    win = _make_window(qtbot, tmp_path)
    win.on_actionSearch_triggered()
    vus = []
    win.current_image_changed.connect(vus.append)

    win._ui.treeGroups.setCurrentItem(_first_pair_item(win))

    assert any(isinstance(v, Path) for v in vus)


def test_a_new_search_clears_the_previous_review(qtbot, tmp_path):
    win = _make_window(qtbot, tmp_path)
    win.on_actionSearch_triggered()
    win._ui.treeGroups.setCurrentItem(_first_pair_item(win))
    assert win._explanation.text() != win._EMPTY

    win.on_actionSearch_triggered()

    assert win._explanation.text() == win._EMPTY
    assert win._preview_a.current_path is None


# ---------------------------------------------------------------------------
# Écriture des étiquettes
# ---------------------------------------------------------------------------

def _capturing_runner(calls):
    def runner(args):
        if any("=" in a for a in args if a.startswith("-")):
            calls.append(args)
            return ""
        return '[{"SourceFile": "x.jpg"}]'
    return runner


def test_writing_asks_once_and_tags_under_the_duplicates_branch(qtbot, tmp_path, monkeypatch):
    questions = []
    monkeypatch.setattr(
        QMessageBox, "question",
        lambda *a, **kw: questions.append(a) or QMessageBox.StandardButton.Yes)
    calls = []
    win = _make_window(qtbot, tmp_path, runner=_capturing_runner(calls))
    win.on_actionSearch_triggered()

    win.on_actionWriteTags_triggered()

    assert len(questions) == 1
    prefixe = "-XMP-digiKam:TagsList="
    ecrits = [a[len(prefixe):] for appel in calls for a in appel if a.startswith(prefixe)]
    assert any(e.startswith("media_restorer/Doublons/") for e in ecrits)
    assert all(e.startswith("media_restorer/Doublons/") for e in ecrits)


def test_declining_writes_nothing(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **kw: QMessageBox.StandardButton.No)
    calls = []
    win = _make_window(qtbot, tmp_path, runner=_capturing_runner(calls))
    win.on_actionSearch_triggered()

    win.on_actionWriteTags_triggered()

    assert calls == []
    assert "annulée" in win.statusBar().currentMessage()


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def test_export_writes_both_csv_and_html(qtbot, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog

    cible = tmp_path / "doublons.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        lambda *a, **kw: (str(cible), "CSV (*.csv)"))
    win = _make_window(qtbot, tmp_path)
    win.on_actionSearch_triggered()

    win.on_actionExport_triggered()

    assert cible.exists()
    assert cible.with_suffix(".html").exists()
    entete = cible.read_text(encoding="utf-8").splitlines()[0]
    assert entete.startswith("image_a,image_b,regime,merite")
    # Le HTML doit rester lisible s'il est déplacé : vignettes incorporées.
    assert "<html" in cible.with_suffix(".html").read_text(encoding="utf-8")


def test_cancelling_the_dialog_writes_no_file(qtbot, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **kw: ("", ""))
    win = _make_window(qtbot, tmp_path)
    win.on_actionSearch_triggered()

    win.on_actionExport_triggered()

    assert list(tmp_path.glob("*.csv")) == []


def test_a_search_failure_is_reported_not_swallowed(qtbot, tmp_path, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("répertoire illisible")

    erreurs = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **kw: erreurs.append(a))
    win = _make_window(qtbot, tmp_path, search_fn=boom)

    win.on_actionSearch_triggered()

    assert erreurs
    assert "Échec" in win.statusBar().currentMessage()

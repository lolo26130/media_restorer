"""Fenêtre Qt de l'extension Doublons — recherche, revue et étiquetage.

Orchestre le cœur sans Qt :mod:`media_restorer.engines.duplicates`.  Trois
étapes distinctes, dans cet ordre et pour cette raison :

1. **chercher** — ne touche à aucun fichier, entièrement réversible ;
2. **revoir** — deux aperçus côte à côte et la phrase explicative ; c'est
   l'humain qui tranche ;
3. **écrire** — modifie les métadonnées, sur confirmation unique.

Le panneau de paramètres est engendré **depuis**
:data:`~media_restorer.engines.duplicates.methods.METHODS` : cocher, décocher ou
ajouter une méthode ne demande aucune modification de ce module.
"""
from __future__ import annotations

from pathlib import Path

from pyqtgraph.parametertree import Parameter, ParameterTree
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QDockWidget,
    QPushButton,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from media_restorer import tabular
from media_restorer.app_settings import TOOLTIP_MODE_KEY, app_settings
from media_restorer.digikam_tags import ExiftoolRunner
from media_restorer.engines.duplicates import (
    METHODS,
    REGIME_SEMANTIQUE,
    STAGE_ORDER,
    STAGE_TITLES,
    DuplicateGraph,
    default_keys,
)
from media_restorer.engines.duplicates import benchmark as _benchmark
from media_restorer.engines.duplicates import device as _device
from media_restorer.engines.duplicates import embeddings as _emb
from media_restorer.engines.duplicates import pipeline as _pipeline
from media_restorer.engines.duplicates import verdicts as _verdicts
from media_restorer.engines.duplicates import report as _report
from media_restorer.engines.duplicates import tags as _dup_tags
from media_restorer.gui_widgets import (
    ImagePreview,
    restore_layout,
    save_layout,
)
from OutilsQt.Utils_Qt import compile_qrc, compile_ui, tooltips_from_code

_UI_SRC = Path(__file__).parent / "views" / "main.ui"
_UI_PY = Path(__file__).parent / "views" / "ui_main.py"
_QRC_SRC = Path(__file__).parents[2] / "resources" / "icons" / "media_restorer.qrc"
_QRC_PY = Path(__file__).parents[2] / "resources" / "icons" / "media_restorer_rc.py"

_TOOLTIP_TYPES: list[tuple] = [(QAction, "action", "triggered")]

#: Préfixe QSettings de la disposition (voir gui_widgets.save_layout).
#: Les deux aperçus côte à côte gardent la place qu'on leur a faite.
_LAYOUT_PREFIX = "doublons"

_SETTINGS_PREFIX = "doublons/"
_KEY_METHODS = _SETTINGS_PREFIX + "methods"
_KEY_TOPK = _SETTINGS_PREFIX + "top_k"
_KEY_THRESHOLD = _SETTINGS_PREFIX + "threshold"
_KEY_MAX_MPX = _SETTINGS_PREFIX + "max_megapixels"
_KEY_DEVICE = _SETTINGS_PREFIX + "device"
_KEY_MODEL = _SETTINGS_PREFIX + "embedding_model"
_KEY_SEM_THRESHOLD = _SETTINGS_PREFIX + "semantic_threshold"

DEFAULT_TOP_K = 20
DEFAULT_THRESHOLD = 0.50
DEFAULT_MAX_MEGAPIXELS = 50
#: Seuil sémantique de départ, DISTINCT du seuil géométrique.  Volontairement
#: élevé : sur des dessins au trait, les empreintes se ressemblent toutes.
#: Il n'a pas vocation à rester — la calibration par verdicts le remplace.
DEFAULT_SEMANTIC_THRESHOLD = 0.85

compile_ui(_UI_SRC, _UI_PY)
compile_qrc(_QRC_SRC, _QRC_PY)

from media_restorer.resources.icons import media_restorer_rc as _rc  # noqa: F401, E402
from media_restorer.extensions.doublons.views.ui_main import Ui_MainWindow  # noqa: E402


class _SearchWorker(QThread):
    """Exécute toute la chaîne dans un fil dédié.

    Signal de résultat nommé ``result_ready`` et **jamais** ``finished``, qui
    shadowe :attr:`QThread.finished` (convention du ``CLAUDE.md``).
    """

    result_ready = pyqtSignal(object)      # DuplicateGraph
    progress = pyqtSignal(int, int)
    stage_changed = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, root: Path, options: dict, *, search_fn=None) -> None:
        super().__init__()
        self._root = root
        self._options = options
        # Injection de dépendance : les tests substituent une chaîne instantanée.
        self._search_fn = search_fn or _pipeline.find_duplicates

    def run(self) -> None:
        try:
            graphe = self._search_fn(
                self._root,
                on_progress=lambda i, n: self.progress.emit(i, n),
                on_stage=lambda s: self.stage_changed.emit(s),
                **self._options,
            )
        except Exception as exc:
            self.error.emit(str(exc))
            return
        self.result_ready.emit(graphe)


class _WriteWorker(QThread):
    """Écrit les étiquettes de tout le graphe, un appel ``exiftool`` par image."""

    result_ready = pyqtSignal(int, list)
    progress = pyqtSignal(int, int)
    image_changed = pyqtSignal(str)

    def __init__(self, graph: DuplicateGraph, *, runner: ExiftoolRunner | None = None) -> None:
        super().__init__()
        self._graph = graph
        self._runner = runner

    def run(self) -> None:
        ecrites, echecs = _dup_tags.write_graph(
            self._graph, runner=self._runner,
            on_progress=lambda i, n: self.progress.emit(i, n),
            on_image=lambda p: self.image_changed.emit(str(p)),
        )
        self.result_ready.emit(ecrites, [(str(p), m) for p, m in echecs])


class DoublonsGUI(QMainWindow):
    """Fenêtre « Doublons » — cherche, laisse revoir, puis étiquette.

    Signaux
    -------
    current_image_changed : pyqtSignal(object)
        ``Path`` de l'image examinée, ``None`` quand il n'y en a plus.  Contrat
        optionnel documenté dans :mod:`media_restorer.extensions` : les docks
        « Aperçu » et « Infos, Exif » de la racine suivent la sélection.
    """

    current_image_changed = pyqtSignal(object)

    _EMPTY = "Aucune recherche effectuée."

    def __init__(
        self,
        target_path: Path | None = None,
        recursive: bool = True,
        search_fn=None,
        exiftool_runner: ExiftoolRunner | None = None,
    ) -> None:
        super().__init__()
        self._search_fn = search_fn
        self._exiftool_runner = exiftool_runner
        self._recursive = recursive
        self._target: Path | None = None
        self._graph: DuplicateGraph | None = None
        self._worker: QThread | None = None

        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)
        self.actionSearch = self._ui.actionSearch
        self.actionWriteTags = self._ui.actionWriteTags
        self.actionExport = self._ui.actionExport

        self._build_pair_panel()
        self._build_parameters()
        tooltips_from_code(
            self, mode=app_settings().value(TOOLTIP_MODE_KEY, "docstrings"),
            liste_types_actions=_TOOLTIP_TYPES,
        )
        self._ui.treeGroups.itemSelectionChanged.connect(self._on_selection)
        self._apply_target(target_path)
        restore_layout(self, _LAYOUT_PREFIX)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_pair_panel(self) -> None:
        """Deux aperçus côte à côte et la phrase explicative sous eux."""
        self._preview_a = ImagePreview()
        self._preview_b = ImagePreview()
        cote_a_cote = QHBoxLayout()
        cote_a_cote.setContentsMargins(0, 0, 0, 0)
        cote_a_cote.addWidget(self._preview_a)
        cote_a_cote.addWidget(self._preview_b)

        # La phrase est le vrai livrable de la revue : « rotation 15°, couverture
        # 25 % » se vérifie d'un coup d'œil, « score 0,83 » ne se vérifie pas.
        self._explanation = QLabel(self._EMPTY)
        self._explanation.setWordWrap(True)
        self._explanation.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._explanation.setStyleSheet("padding:6px;")

        # Verdicts : ils ne servent QUE pour les variantes redessinées, seul
        # régime sans preuve géométrique.  Confirmer une paire que la géométrie
        # a déjà démontrée n'apporterait rien.
        self._btn_confirm = QPushButton("✓ C'est bien une variante")
        self._btn_reject = QPushButton("✗ Non, sans rapport")
        self._btn_confirm.clicked.connect(lambda: self._record_verdict(True))
        self._btn_reject.clicked.connect(lambda: self._record_verdict(False))
        verdicts_ligne = QHBoxLayout()
        verdicts_ligne.addWidget(self._btn_confirm)
        verdicts_ligne.addWidget(self._btn_reject)
        self._set_verdict_enabled(False)

        conteneur = QWidget()
        vertical = QVBoxLayout(conteneur)
        vertical.setContentsMargins(0, 0, 0, 0)
        vertical.addLayout(cote_a_cote, stretch=1)
        vertical.addWidget(self._explanation)
        vertical.addLayout(verdicts_ligne)
        self._ui.pairLayout.addWidget(conteneur)

    def _set_verdict_enabled(self, actif: bool) -> None:
        for bouton in (self._btn_confirm, self._btn_reject):
            bouton.setEnabled(actif)
            bouton.setVisible(actif)

    def _build_parameters(self) -> None:
        """Panneau engendré depuis le catalogue de méthodes, groupé par étage."""
        settings = app_settings()
        actives = settings.value(_KEY_METHODS)
        actives = set(actives) if actives else set(default_keys())

        etages = [
            {
                "name": stage, "title": STAGE_TITLES[stage], "type": "group",
                "children": [
                    {"name": m.key, "title": m.title, "type": "bool",
                     "value": m.key in actives, "tip": m.description}
                    for m in METHODS if m.stage == stage
                ],
            }
            for stage in STAGE_ORDER
        ]
        self._param_root = Parameter.create(
            name="params", type="group",
            children=[
                *etages,
                {"name": "top_k", "title": "Candidats par image", "type": "int",
                 "value": int(settings.value(_KEY_TOPK, DEFAULT_TOP_K)),
                 "limits": (1, 200), "step": 5},
                {"name": "threshold", "title": "Seuil de mérite", "type": "float",
                 "value": float(settings.value(_KEY_THRESHOLD, DEFAULT_THRESHOLD)),
                 "limits": (0.0, 1.0), "step": 0.05},
                {"name": "max_mpx", "title": "Plafond de résolution (Mpx)",
                 "type": "int",
                 "value": int(settings.value(_KEY_MAX_MPX, DEFAULT_MAX_MEGAPIXELS)),
                 "limits": (1, 1000), "step": 5},
                {"name": "calcul", "title": "Calcul", "type": "group", "children": [
                    {"name": "device", "title": "Appareil", "type": "list",
                     "value": settings.value(_KEY_DEVICE, _device.DEVICE_AUTO),
                     "limits": {_device.DEVICE_TITLES[d]: d
                                for d in _device.DEVICE_ORDER}},
                    {"name": "model", "title": "Modèle d'empreinte", "type": "list",
                     "value": settings.value(_KEY_MODEL, _emb.DEFAULT_MODEL),
                     "limits": {m.title: m.key for m in _emb.EMBEDDING_MODELS}},
                    {"name": "sem_threshold", "title": "Seuil sémantique",
                     "type": "float",
                     "value": float(settings.value(_KEY_SEM_THRESHOLD,
                                                   DEFAULT_SEMANTIC_THRESHOLD)),
                     "limits": (0.0, 1.0), "step": 0.01},
                ]},
            ],
        )
        for nom in ("top_k", "threshold", "max_mpx"):
            self._param_root.child(nom).sigValueChanged.connect(self._save_settings)
        for nom in ("device", "model", "sem_threshold"):
            self._param_root.child("calcul", nom).sigValueChanged.connect(self._save_settings)
        for m in METHODS:
            self._param_root.child(m.stage, m.key).sigValueChanged.connect(self._save_settings)

        arbre = ParameterTree(showHeader=False)
        arbre.setParameters(self._param_root)

        # Rappel visible : ce lot ne couvre pas les variantes redessinées.
        self._avertissement = QLabel()
        self._avertissement.setWordWrap(True)
        self._avertissement.setStyleSheet("color:#A33A2E; padding:4px;")
        self._refresh_warning()
        for m in METHODS:
            self._param_root.child(m.stage, m.key).sigValueChanged.connect(
                self._refresh_warning
            )
        avertissement = self._avertissement

        self._btn_compare = QPushButton("Comparer les modèles sur mes verdicts")
        self._btn_compare.clicked.connect(self._compare_models)

        conteneur = QWidget()
        vertical = QVBoxLayout(conteneur)
        vertical.setContentsMargins(0, 0, 0, 0)
        vertical.addWidget(arbre)
        vertical.addWidget(avertissement)
        vertical.addWidget(self._btn_compare)
        vertical.addStretch(1)

        dock = QDockWidget("Méthodes et seuils", self)
        dock.setObjectName("dockMethodes")  # sans objectName, restoreState ignore le dock
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        dock.setWidget(conteneur)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self._param_dock = dock

    def _refresh_warning(self) -> None:
        """L'avertissement ne dit que ce qui est vrai à cet instant.

        Tant que la méthode sémantique n'est pas cochée, les variantes
        redessinées ne sont pas couvertes et il faut le dire.  Une fois cochée,
        c'est la nature *présomptive* du résultat qu'il faut rappeler.
        """
        if self._param_root["verify", "semantic_variants"]:
            self._avertissement.setText(
                "<i>Les <b>variantes redessinées</b> reposent sur une simple "
                "ressemblance de contenu : ce sont des <b>présomptions</b>, "
                "à confirmer une par une à la revue.</i>"
            )
        else:
            self._avertissement.setText(
                "<i>Détecte les mêmes dessins re-numérisés, republiés ou "
                "recadrés.<br>Les <b>variantes redessinées</b> ne sont pas "
                "couvertes — cochez la méthode sémantique pour les chercher.</i>"
            )

    def _compare_models(self) -> None:
        """Classe les modèles d'empreinte sur les verdicts déjà rendus.

        C'est la réponse mesurée à une question que la littérature ne tranche
        pas : DINOv2 ou CLIP sur des caricatures redessinées.  Sans empreintes
        en cache pour plusieurs modèles, le banc d'essai le dit franchement
        plutôt que de comparer ce qu'il n'a pas.
        """
        juges = list(_verdicts.load().values())
        modeles = {}
        for modele in _emb.EMBEDDING_MODELS:
            vecteurs = _emb.load_cache(modele.key)
            if vecteurs:
                modeles[modele.key] = vecteurs
        if not modeles:
            QMessageBox.information(
                self, "Comparaison impossible",
                "Aucune empreinte en cache. Lancez une recherche avec la méthode "
                "sémantique cochée, sur au moins deux modèles différents, avant "
                "de les comparer.",
            )
            return
        resultat = _benchmark.compare_models(juges, modeles)
        QMessageBox.information(
            self, "Comparaison des modèles",
            _benchmark.format_comparison(resultat),
        )

    def _save_settings(self) -> None:
        settings = app_settings()
        settings.setValue(_KEY_METHODS, list(self._selected_methods()))
        settings.setValue(_KEY_TOPK, self._param_root["top_k"])
        settings.setValue(_KEY_THRESHOLD, self._param_root["threshold"])
        settings.setValue(_KEY_MAX_MPX, self._param_root["max_mpx"])
        settings.setValue(_KEY_DEVICE, self._param_root["calcul", "device"])
        settings.setValue(_KEY_MODEL, self._param_root["calcul", "model"])
        settings.setValue(_KEY_SEM_THRESHOLD, self._param_root["calcul", "sem_threshold"])

    def _selected_methods(self) -> tuple[str, ...]:
        """Clés cochées, dans l'ordre du catalogue (jamais celui du décochage)."""
        return tuple(m.key for m in METHODS if self._param_root[m.stage, m.key])

    def _apply_target(self, path: Path | None) -> None:
        self._target = path
        if path is None:
            self.statusBar().showMessage("Aucune cible.")
            return
        if not path.is_dir():
            self.statusBar().showMessage(
                f"« {path.name} » est un fichier — choisissez un répertoire."
            )
            return
        self.actionSearch.setEnabled(True)
        self.statusBar().showMessage(f"Cible : {path}")

    # ------------------------------------------------------------------
    # Recherche
    # ------------------------------------------------------------------

    @pyqtSlot()
    def on_actionSearch_triggered(self) -> None:
        """Parcourt la cible et détecte les doublons selon les méthodes cochées.

        Ne modifie aucun fichier : la recherche est entièrement réversible,
        seule l'écriture des étiquettes touche aux images.
        """
        self._start_search()

    def _start_search(self) -> None:
        if self._target is None or self._worker is not None:
            return
        methodes = self._selected_methods()
        if not any(m.key in methodes and m.stage == "verify" for m in METHODS):
            QMessageBox.information(
                self, "Aucune vérification",
                "Cochez au moins une méthode de vérification géométrique : "
                "sans elle, aucun mérite ne peut être calculé.",
            )
            return

        options = {
            "methods": methodes,
            "recursive": bool(self._recursive),
            "top_k": int(self._param_root["top_k"]),
            "threshold": float(self._param_root["threshold"]),
            "max_megapixels": float(self._param_root["max_mpx"]),
            "semantic_threshold": float(self._param_root["calcul", "sem_threshold"]),
            "embedding_model": self._param_root["calcul", "model"],
        }
        if "semantic_variants" in methodes and self._search_fn is None:
            # L'extracteur n'est construit QUE si l'utilisateur a coché la
            # méthode : sinon on téléchargerait des poids pour rien.
            appareil, explication = _device.resolve(
                self._param_root["calcul", "device"]
            )
            self.statusBar().showMessage(f"Calcul sur {explication}")
            try:
                options["embedder"] = _emb.build_embedder(
                    options["embedding_model"], device=appareil
                )
            except Exception as exc:
                QMessageBox.critical(
                    self, "Modèle indisponible",
                    f"Impossible de charger « {options['embedding_model']} » :\n{exc}\n\n"
                    "La recherche continue sans les variantes redessinées.",
                )
                options["methods"] = tuple(
                    k for k in methodes if k != "semantic_variants"
                )

        self._set_busy(True, "Recherche en cours…")
        worker = _SearchWorker(self._target, options, search_fn=self._search_fn)
        worker.progress.connect(self._on_progress)
        worker.stage_changed.connect(lambda s: self.statusBar().showMessage(s))
        worker.result_ready.connect(self._on_found)
        worker.error.connect(self._on_error)
        self._worker = worker
        worker.start()

    @pyqtSlot(object)
    def _on_found(self, graph: DuplicateGraph) -> None:
        self._worker = None
        self._graph = graph
        self._fill_tree(graph)
        self.actionWriteTags.setEnabled(bool(graph.groups or graph.inclusions))
        self.actionExport.setEnabled(bool(graph.groups or graph.inclusions))
        self._set_busy(False, graph.summary())
        self.current_image_changed.emit(None)

    def _fill_tree(self, graph: DuplicateGraph) -> None:
        """Groupes puis inclusions, chaque nœud portant sa paire ou son chemin."""
        arbre = self._ui.treeGroups
        arbre.clear()
        self._preview_a.clear()
        self._preview_b.clear()
        self._explanation.setText(self._EMPTY)

        if graph.groups:
            racine = QTreeWidgetItem(arbre, [f"Groupes ({len(graph.groups)})"])
            racine.setExpanded(True)
            for numero, groupe in enumerate(graph.groups, start=1):
                noeud = QTreeWidgetItem(
                    racine, [f"{_dup_tags.group_label(numero)} — {groupe.size} images"]
                )
                for paire in groupe.pairs:
                    feuille = QTreeWidgetItem(
                        noeud, [f"{paire.a.name} ↔ {paire.b.name}  "
                                f"(mérite {paire.merit.merite:.2f})"]
                    )
                    feuille.setData(0, Qt.ItemDataRole.UserRole, paire)

        if graph.inclusions:
            racine = QTreeWidgetItem(arbre, [f"Inclusions ({len(graph.inclusions)})"])
            racine.setExpanded(True)
            for paire in graph.inclusions:
                feuille = QTreeWidgetItem(
                    racine, [f"{paire.a.name} ⊃ {paire.b.name}  "
                             f"(mérite {paire.merit.merite:.2f})"]
                )
                feuille.setData(0, Qt.ItemDataRole.UserRole, paire)

        if graph.uncertain:
            # Nommée « possibles » et non « incertaines » : aucune transformation
            # ne les relie, ce sont des présomptions à confirmer, pas des
            # résultats douteux d'une mesure fiable.
            racine = QTreeWidgetItem(
                arbre, [f"Variantes possibles ({len(graph.uncertain)}) — à confirmer"]
            )
            racine.setExpanded(True)
            for paire in graph.uncertain:
                deja = _verdicts.verdict_for(paire.a, paire.b)
                marque = ""
                if deja is not None:
                    marque = "  ✓ confirmée" if deja.confirmed else "  ✗ rejetée"
                feuille = QTreeWidgetItem(
                    racine, [f"{paire.a.name} ≈ {paire.b.name}  "
                             f"({paire.merit.merite:.2f}){marque}"]
                )
                feuille.setData(0, Qt.ItemDataRole.UserRole, paire)

    @pyqtSlot()
    def _on_selection(self) -> None:
        """Affiche la paire sélectionnée et son explication."""
        items = self._ui.treeGroups.selectedItems()
        paire = items[0].data(0, Qt.ItemDataRole.UserRole) if items else None
        if paire is None:
            return
        self._preview_a.show_path(paire.a)
        self._preview_b.show_path(paire.b)
        self._explanation.setText(paire.merit.explain(paire.a.name, paire.b.name))
        self.current_image_changed.emit(paire.a)
        self._current_pair = paire
        # Les verdicts ne concernent que le régime sémantique : confirmer une
        # paire dont la géométrie fait déjà la preuve n'apprendrait rien.
        self._set_verdict_enabled(paire.merit.regime == REGIME_SEMANTIQUE)

    def _record_verdict(self, confirme: bool) -> None:
        """Enregistre le jugement humain et passe à la paire suivante."""
        paire = getattr(self, "_current_pair", None)
        if paire is None:
            return
        _verdicts.record(
            paire.a, paire.b, confirme,
            model=self._param_root["calcul", "model"],
            cosine=float(paire.merit.cosinus_semantique or paire.merit.merite),
        )
        calibration = _verdicts.calibrate(
            _verdicts.load().values(), model=self._param_root["calcul", "model"]
        )
        self.statusBar().showMessage(_verdicts.describe(calibration))
        arbre = self._ui.treeGroups
        suivant = arbre.itemBelow(arbre.currentItem()) if arbre.currentItem() else None
        if suivant is not None:
            arbre.setCurrentItem(suivant)

    # ------------------------------------------------------------------
    # Écriture et export
    # ------------------------------------------------------------------

    @pyqtSlot()
    def on_actionWriteTags_triggered(self) -> None:
        """Enregistre les groupes dans les métadonnées au format DigiKam.

        Modifie les fichiers (``exiftool`` conserve une copie ``_original``) :
        une confirmation unique est demandée pour tout le lot.  Les étiquettes
        existantes — repères, pré-classement — sont préservées.
        """
        self._write_tags()

    def _write_tags(self) -> None:
        if self._graph is None or self._worker is not None:
            return
        n = sum(g.size for g in self._graph.groups)
        reply = QMessageBox.question(
            self, "Écrire les étiquettes ?",
            f"Écrire {len(self._graph.groups)} groupe(s) de doublons "
            f"({n} image(s)) dans les métadonnées ?\n\n"
            "Les fichiers seront modifiés (copie « _original » conservée par "
            "exiftool). Aucune image n'est déplacée ni supprimée, et les "
            "étiquettes déjà présentes sont préservées.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            self.statusBar().showMessage("Écriture annulée.")
            return

        self._set_busy(True, "Écriture des étiquettes…")
        worker = _WriteWorker(self._graph, runner=self._exiftool_runner)
        worker.progress.connect(self._on_progress)
        worker.image_changed.connect(
            lambda p: self.current_image_changed.emit(Path(p))
        )
        worker.result_ready.connect(self._on_written)
        self._worker = worker
        worker.start()

    @pyqtSlot(int, list)
    def _on_written(self, written: int, failures: list) -> None:
        self._worker = None
        self.current_image_changed.emit(None)
        self._set_busy(False, f"{written} image(s) étiquetée(s).")
        if failures:
            apercu = "\n".join(f"• {Path(p).name} : {msg}" for p, msg in failures[:10])
            QMessageBox.warning(
                self, "Écritures en échec",
                f"{len(failures)} image(s) n'ont pas pu être étiquetées :\n\n{apercu}",
            )

    @pyqtSlot()
    def on_actionExport_triggered(self) -> None:
        """Enregistre un rapport CSV et HTML des doublons, sans modifier les images.

        Le HTML est autonome (vignettes incorporées) : il reste lisible s'il est
        déplacé ou transmis.
        """
        self._export()

    def _export(self) -> None:
        if self._graph is None:
            return
        chemin, _ = QFileDialog.getSaveFileName(
            self, "Exporter le rapport", "doublons.csv", tabular.FILE_FILTER
        )
        if not chemin:
            return
        base = Path(chemin)
        paires = [p for g in self._graph.groups for p in g.pairs] + self._graph.inclusions
        try:
            _report.write_csv(base, paires)
            _report.write_html(base.with_suffix(".html"), self._graph)
        except OSError as exc:
            QMessageBox.critical(self, "Erreur — export", str(exc))
            return
        self.statusBar().showMessage(
            f"Rapport exporté : {base.name} et {base.with_suffix('.html').name}"
        )

    # ------------------------------------------------------------------
    # État
    # ------------------------------------------------------------------

    def _set_busy(self, busy: bool, message: str) -> None:
        self._ui.progressBar.setVisible(busy)
        self.actionSearch.setEnabled(not busy and self._target is not None)
        if busy:
            self.actionWriteTags.setEnabled(False)
            self.actionExport.setEnabled(False)
        self.statusBar().showMessage(message)

    @pyqtSlot(int, int)
    def _on_progress(self, index: int, total: int) -> None:
        self._ui.progressBar.setMaximum(max(total, 1))
        self._ui.progressBar.setValue(index)

    @pyqtSlot(str)
    def _on_error(self, message: str) -> None:
        self._worker = None
        self._set_busy(False, "Échec de la recherche.")
        QMessageBox.critical(self, "Erreur — recherche de doublons", message)

    def closeEvent(self, event) -> None:      # noqa: N802 — API Qt
        """Mémorise la disposition des docks avant de fermer.

        Seul chemin par lequel passe toute fermeture de la fenêtre : y placer
        l'enregistrement garantit qu'aucune sortie ne l'oublie.
        """
        save_layout(self, _LAYOUT_PREFIX)
        super().closeEvent(event)

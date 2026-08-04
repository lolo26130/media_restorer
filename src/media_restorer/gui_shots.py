"""Panneau « Correspondances RAW » — inventaire des prises de vue.

Affiche le résultat de :func:`~media_restorer.engines.shots.scan_shots` en
quatre onglets, chacun portant son effectif, avec un aperçu à droite.

Deux partis pris d'interface :

**Le scan ne part jamais tout seul.**  Il coûte une minute sur un corpus de
10 000 fichiers ; le déclencher à chaque changement de cible serait pénible.
Un bouton explicite s'en charge.

**Les appariements douteux ont leur propre onglet.**  Ceux qui ne reposent que
sur le nom de fichier ne sont pas fiables (le compteur Nikon reboucle) : les
mêler aux paires sûres leur prêterait une autorité qu'ils n'ont pas.

Placé dans son propre module plutôt que dans :mod:`media_restorer.gui_widgets` :
ce panneau connaît le cœur ``engines.shots``, là où les widgets de
``gui_widgets`` sont des briques génériques sans dépendance métier.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from PyQt6.QtCore import QByteArray, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from media_restorer import tabular
from media_restorer.app_settings import app_settings
from media_restorer.gui_widgets import ImagePreview

#: Clé QSettings de la répartition gauche/droite du séparateur.
_SPLITTER_KEY = "shots/splitter"

#: Signature de la fonction d'inventaire (injectable pour les tests).
ScanFn = Callable[..., object]


class _ScanWorker(QThread):
    """Inventorie un corpus dans un fil dédié.

    Signal de résultat nommé ``result_ready`` et **jamais** ``finished``, qui
    shadowe :attr:`QThread.finished` (convention du ``CLAUDE.md``).
    """

    result_ready = pyqtSignal(object)      # ShotInventory
    progress = pyqtSignal(int, int)
    stage_changed = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, root: Path, *, recursive: bool = True, scan_fn: ScanFn | None = None):
        super().__init__()
        self._root = root
        self._recursive = recursive
        self._scan_fn = scan_fn

    def run(self) -> None:
        scan = self._scan_fn
        if scan is None:
            from media_restorer.engines.shots import scan_shots

            scan = scan_shots
        try:
            inventaire = scan(
                self._root, recursive=self._recursive,
                on_progress=lambda i, n: self.progress.emit(i, n),
                on_stage=lambda s: self.stage_changed.emit(s),
            )
        except Exception as exc:
            self.error.emit(str(exc))
            return
        self.result_ready.emit(inventaire)


class ShotInventoryPanel(QWidget):
    """Quatre onglets de correspondances, un aperçu, un bouton de scan.

    Signaux
    -------
    current_image_changed : pyqtSignal(object)
        ``Path`` du fichier sélectionné, ``None`` s'il n'y en a plus.  Permet à
        la fenêtre hôte de tenir à jour ses autres docks.
    """

    current_image_changed = pyqtSignal(object)

    _IDLE = "Aucun inventaire — cliquez sur « Scanner »."

    #: (clé d'attribut, titre, en-têtes de colonnes)
    _TABS = (
        ("pairs", "Paires", ("RAW", "JPEG")),
        ("raw_only", "RAW sans JPEG", ("RAW", "Développé ?")),
        ("jpeg_only", "JPEG sans RAW", ("JPEG",)),
        ("unconfirmed", "⚠ À confirmer", ("RAW", "JPEG")),
    )

    def __init__(self, *, scan_fn: ScanFn | None = None,
                 preview_fn: Callable[[Path], Path | None] | None = None) -> None:
        super().__init__()
        self._scan_fn = scan_fn
        self._preview_fn = preview_fn
        self._target: Path | None = None
        self._inventory = None
        self._worker: QThread | None = None

        self._button = QPushButton("Scanner")
        self._button.setEnabled(False)
        self._button.clicked.connect(self._start_scan)
        self._export = QPushButton("Exporter CSV")
        self._export.setEnabled(False)
        self._export.clicked.connect(self._export_csv)
        self._status = QLabel(self._IDLE)
        self._status.setWordWrap(True)

        self._progress = QProgressBar()
        self._progress.setVisible(False)

        entete = QHBoxLayout()
        entete.addWidget(self._button)
        entete.addWidget(self._status, stretch=1)
        entete.addWidget(self._export)

        self._tabs = QTabWidget()
        self._trees: dict[str, QTreeWidget] = {}
        for cle, titre, colonnes in self._TABS:
            arbre = QTreeWidget()
            arbre.setColumnCount(len(colonnes))
            arbre.setHeaderLabels(list(colonnes))
            arbre.setRootIsDecorated(False)
            arbre.setAlternatingRowColors(True)
            arbre.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            arbre.header().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            # Branché sur la sélection et non sur le clic : les flèches du
            # clavier doivent parcourir la liste comme la souris.
            arbre.itemSelectionChanged.connect(self._on_selection)
            self._trees[cle] = arbre
            self._tabs.addTab(arbre, titre)
        self._tabs.currentChanged.connect(lambda _: self._on_selection())

        self._preview = ImagePreview()
        self._details = QLabel("")
        self._details.setWordWrap(True)
        self._details.setStyleSheet("color:#444; padding:4px;")

        droite = QWidget()
        vd = QVBoxLayout(droite)
        vd.setContentsMargins(0, 0, 0, 0)
        vd.addWidget(self._preview, stretch=1)
        vd.addWidget(self._details)

        separateur = QSplitter(Qt.Orientation.Horizontal)
        separateur.addWidget(self._tabs)
        separateur.addWidget(droite)
        separateur.setStretchFactor(0, 3)
        separateur.setStretchFactor(1, 2)
        # Ce panneau vit DANS un dock : la répartition gauche/droite du
        # séparateur n'est pas couverte par le ``saveState`` de la fenêtre
        # racine (qui ne connaît que les docks).  Elle se mémorise donc ici,
        # à chaque déplacement de la poignée — sans quoi l'aperçu reprendrait
        # sa part par défaut à chaque lancement.
        etat = app_settings().value(_SPLITTER_KEY)
        if isinstance(etat, QByteArray) and not etat.isEmpty():
            separateur.restoreState(etat)
        separateur.splitterMoved.connect(
            lambda *_: app_settings().setValue(_SPLITTER_KEY, separateur.saveState())
        )
        self._splitter = separateur

        principal = QVBoxLayout(self)
        principal.setContentsMargins(4, 4, 4, 4)
        principal.addLayout(entete)
        principal.addWidget(separateur, stretch=1)
        principal.addWidget(self._progress)

    # ------------------------------------------------------------------
    # Cible et scan
    # ------------------------------------------------------------------

    def set_target(self, path: Path | None, *, is_directory: bool = True) -> None:
        """Déclare la cible courante — **sans lancer de scan**."""
        self._target = path if (path is not None and is_directory) else None
        self._button.setEnabled(self._target is not None)
        if self._target is None:
            self._status.setText(
                "Choisissez un répertoire pour inventorier ses correspondances RAW."
            )
        elif self._inventory is None:
            self._status.setText(self._IDLE)

    @pyqtSlot()
    def _start_scan(self) -> None:
        if self._target is None or self._worker is not None:
            return
        self._button.setEnabled(False)
        self._progress.setVisible(True)
        self._status.setText("Inventaire en cours…")
        worker = _ScanWorker(self._target, scan_fn=self._scan_fn)
        worker.progress.connect(self._on_progress)
        worker.stage_changed.connect(self._status.setText)
        worker.result_ready.connect(self._on_scanned)
        worker.error.connect(self._on_error)
        self._worker = worker
        worker.start()

    @pyqtSlot(object)
    def _on_scanned(self, inventaire) -> None:
        self._worker = None
        self._inventory = inventaire
        self._fill(inventaire)
        self._progress.setVisible(False)
        self._button.setEnabled(True)
        self._export.setEnabled(True)
        self._status.setText(inventaire.summary())

    def _fill(self, inventaire) -> None:
        """Peuple les quatre onglets et met leur effectif dans le titre."""
        for index, (cle, titre, _colonnes) in enumerate(self._TABS):
            entrees = getattr(inventaire, cle)
            arbre = self._trees[cle]
            arbre.clear()
            for entree in entrees:
                if isinstance(entree, Path):
                    # Pour un RAW orphelin, la seconde colonne dit s'il a déjà
                    # été développé : « déjà développé » signifie que le JPEG a
                    # existé puis a été égaré — l'action à mener diffère.
                    if cle == "raw_only":
                        deja = inventaire.is_developed(entree)
                        colonnes = [entree.name,
                                    "✓ développé" if deja else "— jamais développé"]
                    else:
                        colonnes = [entree.name]
                    item = QTreeWidgetItem(arbre, colonnes)
                    item.setData(0, Qt.ItemDataRole.UserRole, entree)
                    item.setData(1, Qt.ItemDataRole.UserRole, None)
                else:
                    item = QTreeWidgetItem(arbre, [entree.raw.name, entree.derived.name])
                    item.setData(0, Qt.ItemDataRole.UserRole, entree.raw)
                    item.setData(1, Qt.ItemDataRole.UserRole, entree)
            self._tabs.setTabText(index, f"{titre} ({len(entrees)})")
        self._preview.clear()
        self._details.setText("")

    # ------------------------------------------------------------------
    # Sélection et aperçu
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_selection(self) -> None:
        # L'arbre ÉMETTEUR, et non l'onglet courant : les deux coïncident à
        # l'usage, mais s'appuyer sur l'onglet rendrait le panneau muet dès
        # qu'une sélection change autrement qu'à la souris.
        arbre = self.sender()
        if not isinstance(arbre, QTreeWidget):
            arbre = self._tabs.currentWidget()
        if not isinstance(arbre, QTreeWidget):
            return
        items = arbre.selectedItems()
        if not items:
            return
        chemin = items[0].data(0, Qt.ItemDataRole.UserRole)
        paire = items[0].data(1, Qt.ItemDataRole.UserRole)
        if chemin is None:
            return
        chemin = Path(chemin)
        souci = self._show(chemin)
        # Les deux informations sont INDÉPENDANTES : l'explication de
        # l'appariement (boîtier, déclenchement, clé employée) garde tout son
        # sens même quand l'aperçu n'a pas pu être extrait.  On les compose au
        # lieu de laisser l'une écraser l'autre.
        explication = paire.describe() if paire is not None else chemin.name
        self._details.setText(f"{explication}  —  {souci}" if souci else explication)
        self.current_image_changed.emit(chemin)

    def _show(self, chemin: Path) -> str | None:
        """Affiche *chemin* ; renvoie un message si l'aperçu a échoué.

        ⚠ Un RAW n'est **jamais** confié directement à Pillow : il en lirait la
        vignette 160×120 sans le dire (voir
        :mod:`media_restorer.engines.shots.preview`).
        """
        from media_restorer.engines.shots.pairing import RAW_SUFFIXES

        if chemin.suffix.lower() not in RAW_SUFFIXES:
            return None if self._preview.show_path(chemin) else "image illisible"
        extraire = self._preview_fn
        if extraire is None:
            from media_restorer.engines.shots.preview import extract_preview

            extraire = extract_preview
        apercu = extraire(chemin)
        if apercu is None:
            self._preview.clear()
            return "aucun aperçu extractible"
        return None if self._preview.show_path(apercu) else "aperçu illisible"

    # ------------------------------------------------------------------
    # Export et état
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _export_csv(self) -> None:
        if self._inventory is None:
            return
        chemin, _ = QFileDialog.getSaveFileName(
            self, "Exporter l'inventaire", "correspondances_raw.csv",
            tabular.FILE_FILTER
        )
        if not chemin:
            return
        try:
            with open(chemin, "w", newline="", encoding="utf-8") as f:
                writer = tabular.writer(f)
                writer.writerow(["categorie", "raw", "jpeg", "methode",
                                 "appareil", "developpe"])
                for cle, titre, _c in self._TABS:
                    for entree in getattr(self._inventory, cle):
                        if isinstance(entree, Path):
                            est_raw = cle == "raw_only"
                            developpe = ("oui" if self._inventory.is_developed(entree)
                                         else "non") if est_raw else ""
                            writer.writerow([titre, str(entree) if est_raw else "",
                                             "" if est_raw else str(entree), "", "",
                                             developpe])
                        else:
                            writer.writerow([titre, str(entree.raw), str(entree.derived),
                                             entree.method,
                                             entree.meta.get("Model", ""),
                                             "oui" if entree.developed else "non"])
        except OSError as exc:
            QMessageBox.critical(self, "Erreur — export", str(exc))
            return
        self._status.setText(f"Inventaire exporté : {chemin}")

    @pyqtSlot(int, int)
    def _on_progress(self, index: int, total: int) -> None:
        self._progress.setMaximum(max(total, 1))
        self._progress.setValue(index)

    @pyqtSlot(str)
    def _on_error(self, message: str) -> None:
        self._worker = None
        self._progress.setVisible(False)
        self._button.setEnabled(True)
        self._status.setText("Échec de l'inventaire.")
        QMessageBox.critical(self, "Erreur — correspondances RAW", message)

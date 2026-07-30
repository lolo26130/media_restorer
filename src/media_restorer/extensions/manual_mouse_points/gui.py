"""Fenêtre Qt de l'extension Manual Mouse Points — désignation de repères.

Mise en page définie dans ``views/main.ui`` (compilée automatiquement, même
mécanisme que :mod:`media_restorer.extensions.vectorise.gui`).  La fenêtre
orchestre :class:`~media_restorer.extensions.manual_mouse_points.image_click.ImageClick`
(le pointage à la souris, inhérent à Qt) et
:class:`~media_restorer.landmarks.LandmarkSet` (la lecture/écriture des repères
dans les métadonnées, sans Qt) : charger l'image, désigner des repères, les
enregistrer dans les métadonnées (après confirmation), relire ceux déjà
présents.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from pyqtgraph.parametertree import Parameter, ParameterTree
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QDockWidget,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from media_restorer.app_settings import TOOLTIP_MODE_KEY, app_settings
from media_restorer.extensions.manual_mouse_points import config as _config
from media_restorer.extensions.manual_mouse_points.image_click import (
    DEFAULT_LABELS,
    ImageClick,
)
from media_restorer.image_io import imread_oriented
from media_restorer.landmarks import ExiftoolRunner, LandmarkSet
from OutilsQt.Utils_Qt import compile_ui, compile_qrc, tooltips_from_code

_UI_SRC  = Path(__file__).parent / "views" / "main.ui"
_UI_PY   = Path(__file__).parent / "views" / "ui_main.py"
# resources/ est partagé au niveau racine du paquet, comme pour les autres
# extensions — même remontée à 2 niveaux (voir vectorise/gui.py).
_QRC_SRC = Path(__file__).parents[2] / "resources" / "icons" / "media_restorer.qrc"
_QRC_PY  = Path(__file__).parents[2] / "resources" / "icons" / "media_restorer_rc.py"

_TOOLTIP_TYPES: list[tuple] = [
    (QAction, "action", "triggered"),
]

compile_ui(_UI_SRC, _UI_PY)
compile_qrc(_QRC_SRC, _QRC_PY)

from media_restorer.resources.icons import media_restorer_rc as _rc  # noqa: F401, E402
from media_restorer.extensions.manual_mouse_points.views.ui_main import Ui_MainWindow  # noqa: E402


class ManualMousePointsGUI(QMainWindow):
    """Fenêtre « Manual Mouse Points » — désigne des repères et les stocke dans les métadonnées.

    Comme les autres extensions, ne choisit pas elle-même sa cible : le fichier
    lui est transmis à la construction par la fenêtre racine
    :class:`~media_restorer.gui_root.ImageTreatmentWindow`.  Outil à image
    unique — un répertoire est signalé mais non traité.

    Paramètres
    ----------
    target_path : Path | None
        Image à charger dès la construction.  ``None`` construit la fenêtre
        vide (tests, usage direct hors fenêtre racine).
    exiftool_runner : ExiftoolRunner | None
        Substitut du lanceur ``exiftool`` pour les tests (voir
        :mod:`media_restorer.landmarks`) — évite de dépendre du vrai binaire
        et d'écrire sur disque pendant les tests.
    config_path : Path | None
        Fichier de configuration TOML des repères (voir
        :mod:`~media_restorer.extensions.manual_mouse_points.config`).  ``None``
        utilise l'emplacement standard ; les tests passent un fichier jetable.
    """

    _EMPTY_RESULTS  = "Aucun repère désigné."
    _EMPTY_METADATA = "Aucun repère lu dans les métadonnées."

    def __init__(
        self,
        target_path: Path | None = None,
        exiftool_runner: ExiftoolRunner | None = None,
        config_path: Path | None = None,
    ) -> None:
        super().__init__()
        self._exiftool_runner = exiftool_runner
        self._config_path = config_path or _config.config_path()

        self._target_path: Path | None = None
        self._tags: dict[str, tuple[int, int] | None] = {}

        # ── UI de base (.ui compilé) ────────────────────────────────────
        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)

        # Expose les QActions sur self pour tooltips_from_code (convention
        # commune à toutes les fenêtres — voir vectorise/gui.py).
        self.actionStartTagging   = self._ui.actionStartTagging
        self.actionSaveToMetadata = self._ui.actionSaveToMetadata
        self.actionShowMetadata   = self._ui.actionShowMetadata

        # Repères persistés dans le TOML de l'extension (à défaut, jeu par
        # défaut) — mémorisés d'une session à l'autre.
        initial_labels = _config.load_labels(DEFAULT_LABELS, path=self._config_path)

        # ── Vue de pointage ─────────────────────────────────────────────
        self._image_click = ImageClick()
        self._image_click.setup(labels=list(initial_labels))
        self._image_click.tagging_finished.connect(self._on_tagging_finished)
        self._image_click.point_marked.connect(self._on_point_marked)
        self._image_click.instruction_changed.connect(self.statusBar().showMessage)
        self._ui.imageLayout.addWidget(self._image_click)

        # ── Dock Paramètres — libellés modifiables et persistés ─────────
        self._param_root = Parameter.create(
            name="params", type="group",
            children=[
                {
                    "name": "labels", "title": "Repères (un par ligne)",
                    "type": "text", "value": "\n".join(initial_labels),
                },
                {
                    "name": "add_label", "title": "Ajouter un repère…",
                    "type": "action",
                },
            ],
        )
        # Toute édition du texte est persistée dans le TOML ; le bouton
        # « Ajouter » ouvre une saisie puis passe par la même valeur (donc le
        # même enregistrement).  Connexions après create() pour que la valeur
        # initiale ci-dessus ne déclenche pas d'écriture parasite.
        self._param_root.child("labels").sigValueChanged.connect(self._save_labels)
        self._param_root.child("add_label").sigActivated.connect(self._on_add_label)

        tree = ParameterTree(showHeader=False)
        tree.setParameters(self._param_root)

        # Deux listes indépendantes en lecture seule :
        #  • les repères désignés à la souris (mis à jour en direct au marquage) ;
        #  • les repères lus dans les métadonnées de l'image (à l'ouverture et
        #    via « Afficher les repères enregistrés »).
        self._results_label = QLabel(self._EMPTY_RESULTS)
        self._metadata_label = QLabel(self._EMPTY_METADATA)
        for lbl in (self._results_label, self._metadata_label):
            lbl.setWordWrap(True)
            lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        container = QWidget()
        vlayout = QVBoxLayout(container)
        vlayout.setContentsMargins(0, 0, 0, 0)
        vlayout.addWidget(tree)
        vlayout.addWidget(QLabel("<b>Repères désignés (souris) :</b>"))
        vlayout.addWidget(self._results_label)
        vlayout.addWidget(QLabel("<b>Repères enregistrés (métadonnées) :</b>"))
        vlayout.addWidget(self._metadata_label)
        vlayout.addStretch(1)
        dock = QDockWidget("Paramètres", self)
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        dock.setWidget(container)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        # ── Tooltips ─────────────────────────────────────────────────────
        tooltips_from_code(
            self, mode=app_settings().value(TOOLTIP_MODE_KEY, "docstrings"),
            liste_types_actions=_TOOLTIP_TYPES,
        )

        # ── Cible pré-chargée par la fenêtre racine ─────────────────────
        self._apply_target(target_path)

    # ------------------------------------------------------------------
    # Cible
    # ------------------------------------------------------------------

    def _apply_target(self, target_path: Path | None) -> None:
        if target_path is None:
            return
        if target_path.is_dir():
            self.statusBar().showMessage(
                "Manual Mouse Points ne traite qu'un fichier à la fois — "
                "choisissez une image depuis Image Treatment."
            )
            return
        img = imread_oriented(target_path)
        if img is None:
            QMessageBox.critical(self, "Erreur", f"Impossible de lire : {target_path}")
            return
        self._target_path = target_path
        self._image_click.set_image(img)
        self._ui.actionStartTagging.setEnabled(True)
        self._ui.actionShowMetadata.setEnabled(True)
        self.statusBar().showMessage(f"Image chargée : {target_path}")
        # Affiche d'emblée les repères déjà enregistrés (superposés sur l'image
        # + liste métadonnées du dock), sans fenêtre modale : à l'ouverture,
        # une image sans repères — ou un exiftool absent — ne doit pas
        # interrompre l'utilisateur.
        self._refresh_metadata_display(announce=False)

    def _read_labels(self) -> list[str]:
        raw = self._param_root.child("labels").value()
        labels = [line.strip() for line in raw.splitlines() if line.strip()]
        return labels or list(DEFAULT_LABELS)

    def _on_add_label(self, *_args) -> None:
        """Demande un nouveau repère et l'ajoute à la liste (puis la persiste).

        Passe par la valeur du champ texte — l'écriture dans le TOML se fait
        donc via le même chemin que l'édition manuelle
        (:meth:`_save_labels`, sur ``sigValueChanged``).
        """
        text, ok = QInputDialog.getText(self, "Ajouter un repère", "Nom du repère :")
        if not ok:
            return
        label = text.strip()
        if not label:
            return
        current = self._read_labels()
        if label in current:
            self.statusBar().showMessage(f"« {label} » est déjà dans la liste.")
            return
        current.append(label)
        self._param_root.child("labels").setValue("\n".join(current))
        self.statusBar().showMessage(f"Repère ajouté : {label}")

    def _save_labels(self, *_args) -> None:
        """Enregistre la liste courante des repères dans le TOML de l'extension."""
        _config.save_labels(self._read_labels(), path=self._config_path)

    # ------------------------------------------------------------------
    # Désignation
    # ------------------------------------------------------------------

    @pyqtSlot()
    def on_actionStartTagging_triggered(self) -> None:
        """Commence (ou recommence) la désignation des repères sur l'image.

        Réinitialise le pointage avec les libellés du panneau de paramètres,
        puis attend les frappes clavier sur la vue : survol de la souris,
        ``Entrée`` pour marquer le repère courant, ``Espace`` pour le passer,
        ``Q`` pour terminer.  « Enregistrer dans les métadonnées » s'active à
        la fin.
        """
        self._image_click.data_setup(labels=self._read_labels())
        self._image_click.setFocus()
        self._tags = {}
        self._ui.actionSaveToMetadata.setEnabled(False)
        self._render_results()
        self.statusBar().showMessage(
            "Désignation en cours — survol + Entrée/Espace/Q (voir la consigne sur l'image)."
        )

    def _on_point_marked(self, label: str, point: object) -> None:
        # Mise à jour en direct du panneau au fur et à mesure du marquage.
        self._tags[label] = point
        self._render_results()
        where = "passé" if point is None else f"({point[0]}, {point[1]})"
        self.statusBar().showMessage(f"{label} : {where}")

    def _on_tagging_finished(self, tags: dict) -> None:
        self._tags = dict(tags)
        self._render_results()
        marked = sum(1 for v in tags.values() if v is not None)
        self._ui.actionSaveToMetadata.setEnabled(bool(tags))
        self.statusBar().showMessage(
            f"Désignation terminée — {marked}/{len(tags)} repère(s) marqué(s). "
            "« Enregistrer dans les métadonnées » est disponible."
        )

    @staticmethod
    def _format_points(points: dict[str, tuple[int, int] | None], empty_text: str) -> str:
        """Formate *points* en texte pour un ``QLabel`` (un repère par ligne).

        Un point à ``None`` (passé) est rendu « (passé) » ; l'absence totale de
        repère renvoie *empty_text*.
        """
        if not points:
            return empty_text
        return "\n".join(
            f"{label} : {'(passé)' if pt is None else f'({pt[0]}, {pt[1]})'}"
            for label, pt in points.items()
        )

    def _render_results(self, points: dict[str, tuple[int, int] | None] | None = None) -> None:
        """Rafraîchit la liste « repères désignés (souris) » (défaut : les repères courants)."""
        points = self._tags if points is None else points
        self._results_label.setText(self._format_points(points, self._EMPTY_RESULTS))

    def _render_metadata(self, points: dict[str, tuple[int, int] | None]) -> None:
        """Rafraîchit la liste « repères enregistrés (métadonnées) », indépendante de la précédente."""
        self._metadata_label.setText(self._format_points(points, self._EMPTY_METADATA))

    # ------------------------------------------------------------------
    # Écriture dans les métadonnées
    # ------------------------------------------------------------------

    @pyqtSlot()
    def on_actionSaveToMetadata_triggered(self) -> None:
        """Écrit les repères désignés dans les métadonnées de l'image, après confirmation.

        L'écriture modifie le fichier image lui-même (tag ``UserComment`` via
        ``exiftool``, qui conserve une copie ``<image>_original``).  Comme
        c'est une action sur le fichier d'origine, elle demande d'abord
        confirmation à l'utilisateur.
        """
        self._save_to_metadata()

    def _save_to_metadata(self) -> None:
        if not self._tags or self._target_path is None:
            return
        marked = sum(1 for v in self._tags.values() if v is not None)
        reply = QMessageBox.question(
            self, "Écrire dans les métadonnées ?",
            f"Écrire {marked} repère(s) marqué(s) dans les métadonnées de :\n"
            f"{self._target_path.name} ?\n\n"
            "Le fichier image sera modifié (une copie « _original » est conservée "
            "par exiftool).",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            self.statusBar().showMessage("Écriture annulée.")
            return
        try:
            LandmarkSet(points=dict(self._tags)).write_to_metadata(
                self._target_path, runner=self._exiftool_runner
            )
        except Exception as exc:  # exiftool absent, fichier verrouillé…
            QMessageBox.critical(self, "Erreur — écriture des métadonnées", str(exc))
            self.statusBar().showMessage("Échec de l'écriture des métadonnées.")
            return
        self.statusBar().showMessage(
            f"Repères enregistrés dans les métadonnées : {self._target_path.name}"
        )

    # ------------------------------------------------------------------
    # Lecture des métadonnées
    # ------------------------------------------------------------------

    @pyqtSlot()
    def on_actionShowMetadata_triggered(self) -> None:
        """Relit et affiche les repères déjà présents dans les métadonnées de l'image.

        N'écrit rien : lit le tag ``UserComment`` et présente les repères
        trouvés (ou signale qu'il n'y en a aucun à notre schéma).
        """
        self._refresh_metadata_display(announce=True)

    def _refresh_metadata_display(self, *, announce: bool) -> None:
        """Relit les repères des métadonnées et rafraîchit leurs affichages.

        Met à jour la liste « métadonnées » du dock **et** la superposition sur
        l'image (couleur distincte des repères désignés à la souris).  Ne touche
        jamais à la liste « souris » ni aux repères en cours de désignation.

        *announce* — utilisé par l'action « Afficher les repères enregistrés » :
        signale par une fenêtre modale le résultat (liste, absence, ou erreur).
        À ``False`` (ouverture de l'image), reste silencieux : l'affichage se
        met simplement à jour, sans interrompre l'utilisateur.
        """
        if self._target_path is None:
            return
        try:
            landmarks = LandmarkSet.read_from_metadata(
                self._target_path, runner=self._exiftool_runner
            )
        except Exception as exc:
            if announce:
                QMessageBox.critical(self, "Erreur — lecture des métadonnées", str(exc))
            self.statusBar().showMessage("Échec de la lecture des métadonnées.")
            return

        self._render_metadata(landmarks.points)
        self._image_click.show_existing_points(landmarks.points)

        if not landmarks.points:
            if announce:
                QMessageBox.information(
                    self, "Repères enregistrés",
                    "Aucun repère enregistré dans les métadonnées de cette image.",
                )
            self.statusBar().showMessage("Aucun repère enregistré dans les métadonnées.")
            return

        if announce:
            QMessageBox.information(
                self, "Repères enregistrés",
                self._format_points(landmarks.points, self._EMPTY_METADATA),
            )
        self.statusBar().showMessage(
            f"{len(landmarks.points)} repère(s) lu(s) dans les métadonnées."
        )

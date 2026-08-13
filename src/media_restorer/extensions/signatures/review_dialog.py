"""Revue d'un dessin non classé automatiquement : crop manuel, choix, ou passage.

Un item à la fois, **modal** (``QDialog.exec()`` — attente d'un clic de
souris, jamais d'un thread).  Ce n'est PAS une analogie avec le piège
documenté dans ``CLAUDE.md`` (boucle d'événements imbriquée attendant le
signal d'un ``QThread`` vivant, cause d'un segfault reproductible) : le
``_ScanWorker`` a déjà terminé et s'est arrêté avant que cette revue ne
commence (voir ``gui.py``) — rien n'attend ici un fil vivant.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from media_restorer.engines.signatures.pipeline import (
    REASON_AMBIGUOUS,
    REASON_NO_LOCATION,
    REASON_NO_MATCH,
    ScanOutcome,
)
from media_restorer.extensions.signatures.image_crop import ImageCrop

_REASON_MESSAGES = {
    REASON_NO_LOCATION: (
        "Aucune zone de signature localisée automatiquement — ajustez le "
        "cadre rouge sur la signature, indiquez le dessinateur, puis validez."
    ),
    REASON_NO_MATCH: (
        "Une zone a été localisée, mais elle ne ressemble à aucune signature "
        "déjà connue — ajustez le cadre si besoin, indiquez le dessinateur."
    ),
    REASON_AMBIGUOUS: (
        "Plusieurs dessinateurs sont plausibles — choisissez l'un des "
        "candidats ci-dessous, ou indiquez un autre nom."
    ),
}

#: Nombre maximal de candidats DISTINCTS proposés en boutons de choix rapide.
MAX_CANDIDATE_BUTTONS = 4


@dataclass(frozen=True)
class ReviewDecision:
    """Ce que l'utilisateur a décidé pour UN dessin de la revue."""

    #: Nom donné — un nouveau crop est à ajouter à la bibliothèque sous ce nom.
    artist: str | None = None
    #: « Pas de signature visible » — feuille dédiée, voir ``tags.NO_SIGNATURE``.
    no_signature: bool = False
    #: Reporté à une prochaine revue (rescan), rien n'est écrit.
    skipped: bool = False
    #: Zone choisie (ajustée ou dessinée), en pixels de l'image d'origine.
    region: tuple[int, int, int, int] | None = None


class SignatureReviewDialog(QDialog):
    """Boîte modale de revue d'UN dessin non classé automatiquement."""

    def __init__(self, *, known_artists: list[str] | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Revue — signature non reconnue")
        self.setModal(True)
        self._known_artists = list(known_artists or [])
        self._decision: ReviewDecision | None = None

        self._crop = ImageCrop()
        self._crop.setup()

        self._reason_label = QLabel()
        self._reason_label.setWordWrap(True)

        self._candidates_layout = QVBoxLayout()
        candidates_widget = QWidget()
        candidates_widget.setLayout(self._candidates_layout)

        self._name_combo = QComboBox()
        self._name_combo.setEditable(True)
        self._name_combo.addItems(self._known_artists)

        confirm_button = QPushButton("Valider")
        confirm_button.clicked.connect(self._on_confirm)
        confirm_row = QHBoxLayout()
        confirm_row.addWidget(QLabel("Dessinateur :"))
        confirm_row.addWidget(self._name_combo, 1)
        confirm_row.addWidget(confirm_button)

        no_signature_button = QPushButton("Pas de signature visible")
        no_signature_button.clicked.connect(self._on_no_signature)
        skip_button = QPushButton("Passer (revoir plus tard)")
        skip_button.clicked.connect(self._on_skip)
        buttons_row = QHBoxLayout()
        buttons_row.addWidget(no_signature_button)
        buttons_row.addWidget(skip_button)
        buttons_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addWidget(self._reason_label)
        layout.addWidget(self._crop, 1)
        layout.addWidget(candidates_widget)
        layout.addLayout(confirm_row)
        layout.addLayout(buttons_row)
        self.resize(900, 700)

    def set_outcome(self, outcome: ScanOutcome, image: np.ndarray) -> None:
        """Charge un nouveau dessin à revoir — un appel par item de la file."""
        self._decision = None
        self._crop.set_image(image)
        if outcome.box is not None:
            self._crop.set_region(
                outcome.box.xmin, outcome.box.ymin, outcome.box.xmax, outcome.box.ymax
            )
        self._reason_label.setText(
            f"{outcome.path.name} — {_REASON_MESSAGES.get(outcome.reason, '')}"
        )
        self._fill_candidates(outcome)
        self._name_combo.setCurrentText("")

    def _fill_candidates(self, outcome: ScanOutcome) -> None:
        while self._candidates_layout.count():
            item = self._candidates_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()

        # Un bouton par auteur DISTINCT — plusieurs crops du même auteur en
        # tête du classement ne sont pas des choix différents pour l'utilisateur.
        seen: set[str] = set()
        for candidate in outcome.candidates:
            artist = candidate.entry.artist
            if artist in seen:
                continue
            seen.add(artist)
            button = QPushButton(f"{artist}  ({candidate.score:.2f})")
            button.clicked.connect(lambda _checked=False, a=artist: self._choose(a))
            self._candidates_layout.addWidget(button)
            if len(seen) >= MAX_CANDIDATE_BUTTONS:
                break

    def _choose(self, artist: str) -> None:
        self._name_combo.setCurrentText(artist)
        self._on_confirm()

    def _on_confirm(self) -> None:
        name = self._name_combo.currentText().strip()
        if not name:
            return
        self._decision = ReviewDecision(artist=name, region=self._crop.get_region())
        self.accept()

    def _on_no_signature(self) -> None:
        self._decision = ReviewDecision(no_signature=True)
        self.accept()

    def _on_skip(self) -> None:
        self._decision = ReviewDecision(skipped=True)
        self.accept()

    def decision(self) -> ReviewDecision | None:
        """Décision de l'utilisateur pour le dernier dessin présenté."""
        return self._decision

"""Bibliothèque de signatures de référence, constituée progressivement.

Un sous-dossier par dessinateur (nom donné par l'utilisateur), contenant les
crops enregistrés au fil de la revue.  Emplacement **hors ``~/Pictures``**
(comme :mod:`media_restorer.landmark_config`) :
``Path(app_settings().fileName()).with_name("signatures")``.

:mod:`~media_restorer.engines.duplicates.embeddings` n'accepte que des
chemins de fichiers, jamais de tableaux en mémoire — chaque crop est donc
toujours écrit sur disque avant d'être comparé, jamais gardé seulement en
mémoire (voir :mod:`~media_restorer.engines.signatures.pipeline`).

Dédoublonnage **global** : un nouveau crop est comparé à toute la
bibliothèque, tous auteurs confondus — pas seulement aux entrées du même
auteur.  Un quasi-doublon sous le MÊME auteur ne réécrit rien (le lot reste
sans doublons).  Un quasi-doublon sous un auteur DIFFÉRENT est le cas
dangereux (faute de frappe : « Sennep » / « Senep ») qui rendrait toute
correspondance future ambiguë en permanence — jamais silencieux, un
:class:`DuplicateWarning` est renvoyé pour que la revue puisse le signaler.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from media_restorer.app_settings import app_settings
from media_restorer.engines.duplicates.embeddings import Embedder

_LIBRARY_DIRNAME = "signatures"

#: Séparateurs des champs d'étiquettes DigiKam (TagsList utilise « / »,
#: HierarchicalSubject/CatalogSets « | ») — un nom qui en contiendrait un
#: corromprait ces champs en silence (``digikam_tags`` ne valide rien en
#: amont). Voir ``write_tags`` dans ``digikam_tags.py``.
FORBIDDEN_CHARACTERS = ("/", "|")


class InvalidArtistName(ValueError):
    """Nom de dessinateur contenant un séparateur d'étiquette DigiKam."""


@dataclass(frozen=True)
class LibraryEntry:
    """Une signature de référence enregistrée dans la bibliothèque."""

    artist: str
    path: Path


@dataclass(frozen=True)
class DuplicateWarning:
    """Le nouveau crop ressemble fortement à une entrée d'un AUTRE auteur."""

    existing: LibraryEntry
    score: float


def library_dir() -> Path:
    """Répertoire de la bibliothèque, créé au besoin."""
    path = Path(app_settings().fileName()).with_name(_LIBRARY_DIRNAME)
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_artist_name(name: str) -> str:
    """Nom nettoyé (espaces superflus retirés), ou lève :class:`InvalidArtistName`."""
    cleaned = name.strip()
    if not cleaned:
        raise InvalidArtistName("le nom du dessinateur est vide")
    for char in FORBIDDEN_CHARACTERS:
        if char in cleaned:
            raise InvalidArtistName(
                f"le nom du dessinateur ne peut pas contenir {char!r} "
                "(séparateur des étiquettes DigiKam)"
            )
    return cleaned


def list_entries(*, root: Path | None = None) -> list[LibraryEntry]:
    """Toutes les entrées de la bibliothèque, triées par auteur puis fichier."""
    root = root or library_dir()
    return [
        LibraryEntry(artist=artist_dir.name, path=f)
        for artist_dir in sorted(p for p in root.iterdir() if p.is_dir())
        for f in sorted(artist_dir.glob("*.png"))
    ]


def known_artists(*, root: Path | None = None) -> list[str]:
    """Noms de dessinateurs déjà présents, triés — pour l'auto-complétion de la revue."""
    root = root or library_dir()
    return sorted(d.name for d in root.iterdir() if d.is_dir())


def add_entry(
    artist: str,
    crop_bgr: np.ndarray,
    *,
    embedder: Embedder,
    dedup_threshold: float = 0.97,
    root: Path | None = None,
) -> tuple[LibraryEntry, DuplicateWarning | None]:
    """Ajoute *crop_bgr* à la bibliothèque sous *artist*.

    Compare d'abord à toute la bibliothèque existante.  Quasi-doublon du
    MÊME auteur : rien de nouveau n'est écrit, l'entrée existante est
    renvoyée.  Quasi-doublon d'un AUTRE auteur : le choix explicite de
    l'utilisateur est respecté (l'entrée est bien créée), mais un
    :class:`DuplicateWarning` accompagne le retour.
    """
    from media_restorer.engines.signatures import descriptors as _descriptors

    artist = validate_artist_name(artist)
    root = root or library_dir()
    existing = list_entries(root=root)

    artist_dir = root / artist
    artist_dir.mkdir(parents=True, exist_ok=True)
    next_index = len(list(artist_dir.glob("*.png"))) + 1
    new_path = artist_dir / f"{next_index:04d}.png"
    cv2.imwrite(str(new_path), crop_bgr)

    if not existing:
        return LibraryEntry(artist=artist, path=new_path), None

    query = _descriptors.embed_one(new_path, embedder)
    vectors = _descriptors.embed_many([e.path for e in existing], embedder)
    scores = vectors @ query
    best_i = int(np.argmax(scores))
    best_score = float(scores[best_i])
    best = existing[best_i]

    if best_score < dedup_threshold:
        return LibraryEntry(artist=artist, path=new_path), None

    if best.artist == artist:
        # Même auteur : quasi-doublon, rien de nouveau à garder.
        new_path.unlink()
        return best, None

    # Auteur différent : le choix de l'utilisateur est respecté, mais
    # signalé — voir docstring de module.
    return LibraryEntry(artist=artist, path=new_path), DuplicateWarning(
        existing=best, score=best_score
    )

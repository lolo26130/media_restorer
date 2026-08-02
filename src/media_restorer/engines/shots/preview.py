"""Aperçu d'un fichier RAW, extrait de l'image intégrée.

.. warning::

   **Ne jamais ouvrir un RAW avec Pillow.**  ``Image.open()`` sur un ``.NEF``
   **réussit** : Pillow le prend pour un TIFF et renvoie la **vignette de
   160×120** qu'il contient — sans erreur, sans avertissement.  Analyser un RAW
   par cette voie produirait donc un résultat faux *en silence*, exactement le
   genre de défaut qu'aucun test ne rattrape si l'on ne le connaît pas.

   La bonne porte d'entrée est ``exiftool -b -JpgFromRaw``, qui rend l'aperçu
   pleine résolution que le boîtier a écrit dans le fichier : **7360×4912 en
   0,12 s**, pour 3,7 Mo au lieu des 41 Mo du RAW.

L'extraction est **mise en cache** sur disque : rouvrir un aperçu déjà vu ne
relit pas le RAW.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Sequence

#: Balises d'aperçu, de la plus grande à la plus petite.  ``JpgFromRaw`` est
#: l'aperçu pleine taille ; ``PreviewImage`` est plus petit mais toujours
#: exploitable ; ``ThumbnailImage`` est le dernier recours (160×120).
PREVIEW_TAGS = ("JpgFromRaw", "PreviewImage", "ThumbnailImage")

BinaryRunner = Callable[[Sequence[str]], bytes]

_CACHE_DIRNAME = "raw_previews"


def default_runner(args: Sequence[str]) -> bytes:
    """Lance ``exiftool`` et renvoie sa sortie **binaire**."""
    exe = shutil.which("exiftool")
    if exe is None:
        raise RuntimeError(
            "exiftool introuvable — installez-le (paquet « libimage-exiftool-perl ») "
            "pour afficher un aperçu de fichier RAW."
        )
    return subprocess.run([exe, *args], capture_output=True, timeout=120).stdout


def cache_dir() -> Path:
    """Répertoire des aperçus extraits (à côté du ``.ini`` du ``QSettings``)."""
    from media_restorer.app_settings import app_settings

    return Path(app_settings().fileName()).with_name(_CACHE_DIRNAME)


def cached_path(raw: Path) -> Path:
    """Emplacement de l'aperçu de *raw* dans le cache.

    Le nom mêle un condensé du chemin **et** l'empreinte ``(taille, mtime)`` du
    fichier : un RAW modifié ou remplacé produit un autre nom, donc un aperçu
    recalculé, sans qu'il faille tenir un index d'invalidation.
    """
    try:
        st = raw.stat()
        empreinte = f"{st.st_size}-{st.st_mtime_ns}"
    except OSError:
        empreinte = "absent"
    condense = hashlib.sha1(f"{raw.resolve()}|{empreinte}".encode()).hexdigest()[:20]
    return cache_dir() / f"{raw.stem}-{condense}.jpg"


def extract_preview(raw: Path | str, *, runner: BinaryRunner | None = None,
                    cache: bool = True) -> Path | None:
    """Extrait l'aperçu intégré de *raw* et renvoie son chemin, ou ``None``.

    Essaie les balises de :data:`PREVIEW_TAGS` de la plus grande à la plus
    petite.  Ne lève jamais : un RAW d'un format inconnu, ou sans aperçu,
    renvoie ``None`` — à l'appelant d'afficher un message plutôt que de planter
    au milieu d'un inventaire.
    """
    raw = Path(raw)
    destination = cached_path(raw)
    if cache and destination.exists() and destination.stat().st_size > 0:
        return destination

    runner = runner or default_runner
    for balise in PREVIEW_TAGS:
        try:
            donnees = runner(["-b", f"-{balise}", str(raw)])
        except Exception:
            return None
        # Un aperçu plausible dépasse largement quelques centaines d'octets ;
        # en deçà, exiftool a renvoyé du vide et il faut essayer la balise
        # suivante plutôt que d'écrire un fichier illisible.
        if donnees and len(donnees) > 1024:
            if not cache:
                return _ecrire_temporaire(donnees)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(donnees)
            return destination
    return None


def _ecrire_temporaire(donnees: bytes) -> Path:
    import tempfile

    fichier = Path(tempfile.mkstemp(suffix=".jpg", prefix="apercu-raw-")[1])
    fichier.write_bytes(donnees)
    return fichier


def clear_cache() -> int:
    """Vide le cache d'aperçus.  Renvoie le nombre de fichiers supprimés."""
    repertoire = cache_dir()
    if not repertoire.is_dir():
        return 0
    n = 0
    for fichier in repertoire.glob("*.jpg"):
        try:
            fichier.unlink()
            n += 1
        except OSError:
            pass
    return n

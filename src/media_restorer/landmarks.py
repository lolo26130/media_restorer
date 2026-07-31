"""Lecture/écriture de points nommés (repères) dans les métadonnées d'une image.

Duplique — par copie, sans importation — puis recentre la gestion de
métadonnées de ``TraiteImages.classes.data_classes.DataImages`` (son
``read_tags`` via ExifTool, son ``exif_string_to_nested_dict``).  La classe
d'origine était un couteau suisse (chargement d'image, rotation, mise à
l'échelle, alignement de visage *et* EXIF) ; on n'en garde ici que la brique
métadonnées, réduite à un seul rôle : **stocker et relire un jeu de points
nommés** (« Left Eye », « Nose »…) désignés à la souris (voir
:class:`~media_restorer.image_click.ImageClick`).

Améliorations par rapport à la source
-------------------------------------
- Aucune dépendance à ``PyExifTool`` (non installé ici) : appel direct du
  binaire ``exiftool`` en sous-processus — le *runner* est injectable, si bien
  que les tests n'ont jamais besoin de lancer le vrai binaire (convention
  d'injection de dépendances du ``CLAUDE.md`` racine).
- Charge utile en **JSON auto-contenu** dans un seul tag (``UserComment``),
  sous une clé de schéma reconnaissable : un commentaire sans rapport n'est
  jamais confondu avec nos données, et les points passés (``None``) sont
  conservés tels quels (distincts d'un point simplement absent).
- Sans état d'image : ``LandmarkSet`` ne transporte que les points, pas les
  pixels — la lecture/écriture des pixels reste à :mod:`media_restorer.image_io`.

Format et interopérabilité
--------------------------
Le tag ``UserComment`` (EXIF) est un champ libre, non destructif pour les
champs descriptifs qu'un logiciel de catalogage curerait (légende, mots-clés).
``exiftool`` conserve par défaut une copie ``<image>_original`` : l'écriture
est donc réversible.  Un schéma standard de régions de visage (MWG-rs /
digiKam, relu par Lightroom) serait plus interopérable mais nettement plus
lourd à écrire — évolution possible sans changer l'API publique de ce module.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# Un point : coordonnées en **pourcentage** (0–100) de la largeur (x) et de la
# hauteur (y) de l'image, ou ``None`` si le repère a été passé.  Les
# pourcentages sont indépendants de la résolution : aucune conversion n'est
# nécessaire si l'image est redimensionnée (c'est tout l'intérêt de ce choix).
Point = tuple[float, float] | None

# Fonction qui lance ``exiftool`` avec *args* (sans le nom de l'exécutable) et
# renvoie sa sortie standard.  Injectable pour les tests — voir _default_runner.
ExiftoolRunner = Callable[[list[str]], str]

# Tag porteur (champ EXIF libre) et clé de schéma qui identifie nos données.
_TAG = "UserComment"
_SCHEMA_KEY = "media_restorer_landmarks"


def _default_runner(args: list[str]) -> str:
    """Lance le vrai binaire ``exiftool`` et renvoie sa sortie standard.

    Runner de production (les tests en injectent un faux).  Lève un
    :class:`RuntimeError` explicite si ``exiftool`` n'est pas installé, plutôt
    que le ``FileNotFoundError`` brut de :func:`subprocess.run`.
    """
    exe = shutil.which("exiftool")
    if exe is None:
        raise RuntimeError(
            "exiftool introuvable — installez-le (paquet « libimage-exiftool-perl » "
            "sous Debian/Ubuntu) pour lire/écrire les repères dans les métadonnées."
        )
    result = subprocess.run(
        [exe, *args], capture_output=True, text=True, check=True, timeout=60
    )
    return result.stdout


@dataclass
class LandmarkSet:
    """Jeu de points nommés désignés sur une image, sérialisable dans ses métadonnées.

    Version recentrée de ``DataImages`` : ne porte que les repères, pas les
    pixels ni les transformations d'image.

    Attributs
    ---------
    points : dict[str, Point]
        Repère → coordonnées ``(x, y)`` en **pourcentage** (0–100) de la
        largeur/hauteur, ou ``None`` si passé.  L'ordre d'insertion (garanti
        par ``dict``) reflète l'ordre de désignation.
    """

    points: dict[str, Point] = field(default_factory=dict)

    # -- Sérialisation JSON (indépendante du support de stockage) -----------

    def to_json(self) -> str:
        """Charge utile JSON (une ligne) sous la clé de schéma reconnaissable.

        Coordonnées en pourcentage (float) — indépendantes de la résolution.
        """
        serialisable = {
            label: ([float(pt[0]), float(pt[1])] if pt is not None else None)
            for label, pt in self.points.items()
        }
        return json.dumps({_SCHEMA_KEY: serialisable}, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "LandmarkSet | None":
        """Reconstruit un :class:`LandmarkSet` depuis *raw*, ou ``None``.

        Renvoie ``None`` (et non un jeu vide) si *raw* n'est pas notre schéma :
        c'est ce qui permet à :meth:`read_from_metadata` de distinguer « pas
        de repères enregistrés » d'un ``UserComment`` sans rapport laissé par
        un autre logiciel.
        """
        try:
            blob = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(blob, dict) or _SCHEMA_KEY not in blob:
            return None
        payload = blob[_SCHEMA_KEY]
        if not isinstance(payload, dict):
            return None
        points: dict[str, Point] = {}
        for label, value in payload.items():
            if value is None:
                points[label] = None
            elif isinstance(value, (list, tuple)) and len(value) == 2:
                points[label] = (float(value[0]), float(value[1]))
            # tout autre forme est ignorée (donnée corrompue) plutôt que fatale
        return cls(points=points)

    # -- Support métadonnées (exiftool en sous-processus) -------------------

    def write_to_metadata(self, path: Path | str, *, runner: ExiftoolRunner | None = None) -> None:
        """Écrit les repères dans le tag ``UserComment`` de *path*.

        Écriture non destructive pour les pixels ; ``exiftool`` conserve par
        défaut une copie ``<image>_original``.  *runner* est injectable pour
        les tests (défaut : le vrai binaire, voir :func:`_default_runner`).
        """
        runner = runner or _default_runner
        runner([f"-{_TAG}={self.to_json()}", str(path)])

    @classmethod
    def read_from_metadata(cls, path: Path | str, *, runner: ExiftoolRunner | None = None) -> "LandmarkSet":
        """Relit les repères depuis les métadonnées de *path*.

        Renvoie un jeu vide si le fichier ne porte pas de repères à notre
        schéma (aucune métadonnée, ou ``UserComment`` sans rapport).  Ne
        distingue pas volontairement l'un de l'autre côté appelant : dans les
        deux cas, il n'y a rien à afficher.
        """
        runner = runner or _default_runner
        out = runner([f"-{_TAG}", "-j", str(path)])
        try:
            records = json.loads(out)
        except json.JSONDecodeError:
            return cls()
        if not isinstance(records, list) or not records:
            return cls()
        raw = records[0].get(_TAG)
        if not isinstance(raw, str):
            return cls()
        return cls.from_json(raw) or cls()

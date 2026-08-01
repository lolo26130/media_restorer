"""Cache des mesures de tri, pour ne pas relire un corpus deux fois.

Mesurer 8 700 images coûte environ sept minutes ; rejouer un classement avec
d'autres seuils ne doit rien recoûter.

**C'est précisément pourquoi**
:class:`~media_restorer.engines.triage.signals.ImageSignals` sépare les
*mesures* (attributs : couverture d'encre, saturations, dimensions) des
*catégories* (propriétés recalculées à la volée depuis les seuils du module).
Seules les mesures sont mises en cache — changer un seuil ne périme donc
strictement rien.

Format et invalidation
----------------------
Un JSON par corpus, dans le répertoire de configuration de l'application (même
emplacement que :mod:`media_restorer.landmark_config`, chemin dérivé de
``app_settings().fileName()`` — donc automatiquement isolé en test par la
redirection ``QSettings`` de ``conftest.py``).

Chaque entrée porte ``(taille, mtime)`` du fichier mesuré : une image retouchée
ou remplacée est remesurée, une image inchangée est relue instantanément.  Le
chemin absolu sert de clé.  Format lisible à la main, ce qui compte pour un
fichier qu'on voudra parfois inspecter ou supprimer sans outil.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from media_restorer.engines.triage.signals import ImageSignals

_CACHE_FILENAME = "triage_cache.json"
#: Version du format : incrémenter invalide tout le cache d'un coup, ce qui est
#: le comportement voulu si la définition d'une mesure venait à changer.
_FORMAT_VERSION = 1


def cache_path() -> Path:
    """Emplacement standard du cache (à côté du ``.ini`` du ``QSettings``)."""
    from media_restorer.app_settings import app_settings

    return Path(app_settings().fileName()).with_name(_CACHE_FILENAME)


def _stamp(path: Path) -> tuple[int, int] | None:
    """``(taille, mtime_ns)`` du fichier, ou ``None`` s'il a disparu."""
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_size, st.st_mtime_ns)


def load(path: Path | None = None) -> dict[Path, ImageSignals]:
    """Mesures en cache, **déjà validées** contre l'état actuel des fichiers.

    Ne lève jamais : un cache absent, tronqué ou d'une version antérieure
    revient à un cache vide — au pire on remesure, ce qui est lent mais jamais
    faux.  C'est le bon compromis pour un fichier purement dérivé.
    """
    path = path or cache_path()
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(blob, dict) or blob.get("version") != _FORMAT_VERSION:
        return {}

    entrees = blob.get("entries")
    if not isinstance(entrees, dict):
        return {}

    valides: dict[Path, ImageSignals] = {}
    for brut, valeur in entrees.items():
        fichier = Path(brut)
        if not isinstance(valeur, dict) or _stamp(fichier) != tuple(valeur.get("stamp", ())):
            continue                       # fichier modifié, remplacé ou disparu
        try:
            valides[fichier] = ImageSignals(
                path=fichier,
                width=int(valeur["width"]),
                height=int(valeur["height"]),
                ink_coverage=float(valeur["ink_coverage"]),
                ink_saturation=float(valeur["ink_saturation"]),
                paper_saturation=float(valeur["paper_saturation"]),
            )
        except (KeyError, TypeError, ValueError):
            continue                       # entrée corrompue : simplement ignorée
    return valides


def save(signals: Iterable[ImageSignals], path: Path | None = None) -> None:
    """Écrit *signals* dans le cache, en remplaçant son contenu.

    Les fichiers disparus entre la mesure et l'écriture sont omis plutôt que
    stockés avec une empreinte nulle, qui les ferait paraître valides.
    """
    path = path or cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    entrees: dict[str, dict] = {}
    for s in signals:
        stamp = _stamp(s.path)
        if stamp is None:
            continue
        entrees[str(s.path)] = {
            "stamp": list(stamp),
            "width": s.width,
            "height": s.height,
            "ink_coverage": s.ink_coverage,
            "ink_saturation": s.ink_saturation,
            "paper_saturation": s.paper_saturation,
        }
    path.write_text(
        json.dumps({"version": _FORMAT_VERSION, "entries": entrees}, ensure_ascii=False),
        encoding="utf-8",
    )

"""Tri grossier d'un corpus de dessins par signaux « gratuits », sans Qt.

Cœur de calcul du **premier niveau de tri** : les quatre grandeurs qui se
mesurent sans aucun modèle (orientation, résolution, densité d'encre,
chromatisme encre/papier).  Voir ``docs/rapport-tri-grossier-corpus.tex`` pour
le raisonnement et les mesures sur le corpus de référence.

Comme :mod:`~media_restorer.engines.vectorise` et
:mod:`~media_restorer.engines.face_id`, ce paquet **ne dérive pas de**
:class:`~media_restorer.engines.base.BaseEngine` : le contrat de ce dernier va
d'une image vers une image, alors qu'on va ici d'une image vers un jeu de
mesures scalaires, et d'un répertoire vers des distributions.  Il n'est donc pas
non plus déclaré dans l'énumération ``Engine`` de
:mod:`media_restorer.engines`.

Utilisation typique — calibrer des seuils sur un échantillon, puis trier tout :

.. code-block:: python

    from media_restorer.engines.triage import scan_directory, summarise, format_summary

    signals = scan_directory("~/Pictures/Dessins", sample=500)
    print(format_summary(summarise(signals)))

Seule dépendance : ``Pillow`` et ``numpy``, toutes deux déjà requises par le
projet.  Aucun téléchargement de poids, aucun GPU.
"""
from media_restorer.engines.triage.criteria import (
    CRITERIA,
    CRITERIA_BY_KEY,
    METHOD_SIGNALS,
    METHODS,
    Criterion,
    criteria_for,
    select,
)
from media_restorer.engines.triage.scan import (
    IMAGE_SUFFIXES,
    ScanResult,
    format_summary,
    iter_images,
    scan_directory,
    summarise,
)
from media_restorer.engines.triage.signals import (
    INK_DENSITIES,
    ORIENTATIONS,
    RESOLUTIONS,
    SUPPORTS,
    ImageSignals,
    TooLarge,
    measure_image,
)

__all__ = [
    "CRITERIA",
    "CRITERIA_BY_KEY",
    "IMAGE_SUFFIXES",
    "INK_DENSITIES",
    "METHODS",
    "METHOD_SIGNALS",
    "ORIENTATIONS",
    "RESOLUTIONS",
    "SUPPORTS",
    "Criterion",
    "ImageSignals",
    "ScanResult",
    "TooLarge",
    "criteria_for",
    "format_summary",
    "iter_images",
    "measure_image",
    "scan_directory",
    "select",
    "summarise",
]

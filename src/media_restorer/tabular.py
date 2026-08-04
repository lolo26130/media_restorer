"""Convention commune d'export tabulaire, sans Qt.

Tous les exports « CSV » du projet emploient la **tabulation** comme séparateur,
et non la virgule.

.. warning::

   **Un chemin de fichier peut contenir une virgule.**  Le corpus en contient
   (« 3 Républiques Cabrol et Sennep », « Photos, scans »…), et le module
   :mod:`csv` de la bibliothèque standard échapperait alors le champ par des
   guillemets — correct, mais illisible dans un tableur mal configuré, et
   fragile dès qu'un outil tiers relit le fichier avec un découpage naïf.

   Une tabulation, elle, **ne peut pas apparaître dans un nom de fichier** sur
   les systèmes de fichiers usuels : le découpage reste sans ambiguïté, sans
   échappement, et lisible tel quel.

L'extension reste ``.csv`` : c'est celle que les tableurs proposent d'ouvrir, et
tous savent choisir la tabulation à l'import.
"""
from __future__ import annotations

import csv
from typing import IO, Sequence

#: Séparateur de champs — voir l'avertissement en tête de module.
DELIMITER = "\t"

#: Filtre de tableur associé, pour les boîtes de dialogue d'enregistrement.
FILE_FILTER = "Texte tabulé (*.csv *.tsv)"


def writer(fichier: IO[str]):
    """Rédacteur ``csv`` configuré à la convention du projet.

    ``QUOTE_MINIMAL`` reste actif : si un champ contenait malgré tout une
    tabulation ou un saut de ligne, il serait correctement protégé plutôt que
    de casser silencieusement le tableau.
    """
    return csv.writer(fichier, delimiter=DELIMITER, quoting=csv.QUOTE_MINIMAL)


def dict_writer(fichier: IO[str], fieldnames: Sequence[str],
                **kwargs) -> csv.DictWriter:
    """Variante par dictionnaire, même convention."""
    return csv.DictWriter(fichier, fieldnames=list(fieldnames),
                          delimiter=DELIMITER, quoting=csv.QUOTE_MINIMAL, **kwargs)


def reader(fichier: IO[str]):
    """Lecteur correspondant — pour relire un export du projet."""
    return csv.reader(fichier, delimiter=DELIMITER)

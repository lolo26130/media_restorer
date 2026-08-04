"""Convention d'export tabulaire — la tabulation plutôt que la virgule.

Le vrai enjeu tient en un test : un chemin **contenant une virgule** (le corpus
en a) doit ressortir tel quel, sans guillemets d'échappement, et se relire
découpé au bon endroit.
"""
from __future__ import annotations

import csv
import io

from media_restorer import tabular


def test_the_separator_is_a_tabulation():
    """Une tabulation ne peut pas figurer dans un nom de fichier ; une virgule si."""
    assert tabular.DELIMITER == "\t"


def test_a_path_with_a_comma_survives_untouched():
    """Aucun guillemet d'échappement : le champ reste lisible tel quel."""
    chemin = "/fonds/3 Républiques Cabrol, Sennep/dessin.jpg"
    tampon = io.StringIO()
    tabular.writer(tampon).writerow([chemin, "42"])

    ligne = tampon.getvalue().rstrip("\r\n")
    assert '"' not in ligne
    assert ligne == f"{chemin}\t42"


def test_a_path_with_a_comma_is_read_back_as_one_field():
    """Le piège qu'on évite : la virgule couperait ce chemin en deux colonnes."""
    chemin = "/fonds/Photos, scans/a.jpg"
    tampon = io.StringIO()
    tabular.writer(tampon).writerow([chemin, "42"])
    tampon.seek(0)

    (relu,) = list(tabular.reader(tampon))
    assert relu == [chemin, "42"]

    # Contre-épreuve — la raison d'être de ce module : découpé à la virgule, ce
    # même chemin se brise en deux, et rien ne signale l'erreur.
    tampon.seek(0)
    brise = next(csv.reader(tampon))
    assert brise[0] == "/fonds/Photos"
    assert len(brise) == 2


def test_the_dict_writer_follows_the_same_convention():
    """Les exports par dictionnaire (rapport de doublons) ne divergent pas."""
    tampon = io.StringIO()
    redacteur = tabular.dict_writer(tampon, ["a", "b"])
    redacteur.writeheader()
    redacteur.writerow({"a": "x, y", "b": "z"})

    entete, ligne = tampon.getvalue().splitlines()
    assert entete == "a\tb"
    assert ligne == "x, y\tz"


def test_the_file_filter_accepts_both_extensions():
    """``.csv`` reste proposé : c'est l'extension que les tableurs ouvrent."""
    assert "*.csv" in tabular.FILE_FILTER
    assert "*.tsv" in tabular.FILE_FILTER

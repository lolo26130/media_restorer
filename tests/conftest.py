"""Configuration commune aux tests.

Force le backend Qt « offscreen » **avant** tout import de PyQt6, afin que la
suite tourne sans serveur graphique.  Passer par ``conftest.py`` plutôt que par
``env = [...]`` dans ``pyproject.toml`` évite d'ajouter la dépendance
``pytest-env``.
"""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _isolated_qsettings(tmp_path):
    """Redirige QSettings("media_restorer", "media_restorer") vers un fichier jetable.

    Sans ça, instancier PhotoRestorationGUI en test (choix de skin dans
    __init__) écrirait dans le vrai ``~/.config/media_restorer/…`` de
    l'utilisateur — un test ne doit pas laisser de trace dans sa config réelle.
    """
    from PyQt6.QtCore import QSettings

    # NativeFormat et IniFormat sont deux entrées indépendantes pour
    # QSettings.setPath, alors que ``QSettings("org", "app")`` (constructeur à
    # 2 arguments) retombe sur NativeFormat — rediriger seulement IniFormat
    # laisserait fuiter vers la vraie config tout code qui utilise ce
    # constructeur sans préciser le format (piège dans lequel ce projet est
    # déjà tombé une fois, voir gui._skin_settings).
    for fmt in (QSettings.Format.IniFormat, QSettings.Format.NativeFormat):
        QSettings.setPath(fmt, QSettings.Scope.UserScope, str(tmp_path))

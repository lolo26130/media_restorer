"""Configuration commune aux tests.

Force le backend Qt « offscreen » **avant** tout import de PyQt6, afin que la
suite tourne sans serveur graphique.  Passer par ``conftest.py`` plutôt que par
``env = [...]`` dans ``pyproject.toml`` évite d'ajouter la dépendance
``pytest-env``.
"""
import os
from contextlib import contextmanager

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


@contextmanager
def _noop_performance_mode(*args, **kwargs):
    yield


@pytest.fixture(autouse=True)
def _no_real_power_management(monkeypatch):
    """Neutralise ``performance_mode`` dans tous les workers Qt réels de la suite.

    ``_RestoreWorker``/``_BatchRestoreWorker`` (Media Restorer) et
    ``_GetOutlineWorker`` (Vectorise) l'appellent depuis un vrai ``QThread``.
    Le laisser spawner un vrai sous-processus (``powerprofilesctl``,
    ``systemd-inhibit`` — voir ``media_restorer.power``) pendant qu'un thread
    de test attend son résultat a provoqué un segfault reproductible en
    développement (accumulation de fenêtres/threads Qt sur toute une session
    de tests, pas isolable à un seul fichier).  Indépendamment de la cause
    exacte, un test ne doit de toute façon jamais dépendre d'un vrai
    changement de profil d'alimentation système.

    Patché dans l'espace de noms de chaque *consommateur* (pas dans
    ``media_restorer.power`` lui-même) : ``from media_restorer.power import
    performance_mode`` lie le nom au moment de l'import, bien avant qu'un
    test ne s'exécute — corriger la source après coup ne changerait rien à
    la référence déjà capturée par chaque module consommateur.
    """
    targets = (
        "media_restorer.extensions.media_restorer.gui.performance_mode",
        "media_restorer.extensions.vectorise.gui.performance_mode",
        "media_restorer.extensions.auto_face_id_register.gui.performance_mode",
    )
    for target in targets:
        module_path, _, _ = target.rpartition(".")
        try:
            __import__(module_path)
        except ImportError:
            continue
        monkeypatch.setattr(target, _noop_performance_mode)

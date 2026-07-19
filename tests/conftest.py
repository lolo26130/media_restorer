"""Configuration commune aux tests.

Force le backend Qt « offscreen » **avant** tout import de PyQt6, afin que la
suite tourne sans serveur graphique.  Passer par ``conftest.py`` plutôt que par
``env = [...]`` dans ``pyproject.toml`` évite d'ajouter la dépendance
``pytest-env``.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

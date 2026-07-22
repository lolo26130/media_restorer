"""Préférences persistantes de l'application (skin, mode tooltips).

Point d'entrée unique construisant l'objet ``QSettings`` partagé par les
fenêtres de l'application (:class:`~media_restorer.gui_root.ImageTreatmentWindow`
et :class:`~media_restorer.gui.PhotoRestorationGUI`).

Pourquoi ce module existe
-------------------------
``QSettings("org", "app")`` (constructeur à deux arguments) retombe sur
``QSettings.Format.NativeFormat``, qui est un format *distinct* d'
``IniFormat`` du point de vue de ``QSettings.setPath`` : rediriger l'un des
deux formats vers un répertoire temporaire (ce que font les tests pour ne
jamais écrire dans la vraie configuration de l'utilisateur) laisse fuiter
l'autre.  Ce piège s'est déjà produit une fois dans ce projet (préférence de
skin écrite dans ``~/.config/media_restorer/`` malgré une tentative
d'isolation).  Centraliser la construction du ``QSettings`` avec un format
explicite, ici et nulle part ailleurs, l'élimine structurellement.
"""
from __future__ import annotations

from PyQt6.QtCore import QSettings

_ORGANIZATION = "media_restorer"
_APPLICATION = "media_restorer"

# Clés partagées entre ImageTreatmentWindow (qui les règle) et
# PhotoRestorationGUI (qui les lit) — une seule définition pour éviter toute
# divergence de nom de clé entre les deux fenêtres.
SKIN_KEY = "skin"
TOOLTIP_MODE_KEY = "tooltip_mode"


def app_settings() -> QSettings:
    """Objet ``QSettings`` partagé de l'application, en ``IniFormat`` explicite.

    À utiliser pour toute préférence persistante — ne jamais construire de
    ``QSettings`` directement ailleurs dans le code de l'application (voir
    docstring de module pour la raison).
    """
    return QSettings(
        QSettings.Format.IniFormat,
        QSettings.Scope.UserScope,
        _ORGANIZATION,
        _APPLICATION,
    )

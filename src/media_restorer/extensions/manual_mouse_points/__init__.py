"""Extension Manual Mouse Points — désignation de repères à la souris.

Sous-paquet isolé, même schéma que les autres extensions
(:mod:`~media_restorer.extensions.vectorise`) : sa fenêtre (``gui.py``), son
widget de pointage (``image_click.py``, copié puis adapté de
``TraiteImages``) et son interface ``.ui`` (``views/main.ui``) lui sont
propres.  La lecture/écriture des repères dans les métadonnées vit séparément
dans :mod:`media_restorer.landmarks`, sans dépendance Qt.  Importer ce paquet
suffit à enregistrer :class:`ManualMousePointsExtension` — voir
:mod:`media_restorer.extensions` pour le mécanisme général.
"""
from __future__ import annotations

from PyQt6.QtWidgets import QMainWindow

from media_restorer.extensions import ExtensionContext, register


class ManualMousePointsExtension:
    """Désignation manuelle de repères (yeux, nez…) et stockage dans les métadonnées.

    Enveloppe
    :class:`~media_restorer.extensions.manual_mouse_points.gui.ManualMousePointsGUI`
    derrière le protocole :class:`~media_restorer.extensions.Extension`.
    """

    name = "Manual Mouse Points"
    description = (
        "Désigner des repères (yeux, bouche…) à la souris et les stocker dans "
        "les métadonnées de l'image (via exiftool)"
    )
    icon = ":/icons/eye.png"

    def launch(self, context: ExtensionContext) -> QMainWindow:
        """Ouvre la fenêtre avec *context* déjà chargé.

        Importé ici (et non en tête de module) pour ne compiler l'interface
        ``.ui`` et les ressources Qt qu'au moment du lancement réel — même
        raison que :meth:`~media_restorer.extensions.vectorise.VectoriseExtension.launch`.
        """
        from media_restorer.extensions.manual_mouse_points.gui import ManualMousePointsGUI

        window = ManualMousePointsGUI(target_path=context.path)
        window.show()
        return window


register(ManualMousePointsExtension())

"""Extension Auto Face ID Register — détection automatique de repères.

Sous-paquet isolé, même schéma que les autres extensions.  Le cœur de calcul
(détection zero-shot) vit sans Qt dans :mod:`media_restorer.engines.face_id` ;
le widget de revue (:class:`~media_restorer.image_click.ImageClick`) et la liste
de repères (:mod:`media_restorer.landmark_config`) sont partagés au niveau du
cœur — cette extension ne dépend donc pas de
:mod:`~media_restorer.extensions.manual_mouse_points`.  Importer ce paquet
suffit à enregistrer :class:`AutoFaceIdRegisterExtension`.
"""
from __future__ import annotations

from PyQt6.QtWidgets import QMainWindow

from media_restorer.extensions import ExtensionContext, register


class AutoFaceIdRegisterExtension:
    """Détecte automatiquement des repères (yeux, nez…) sur un dessin, puis les fait valider.

    Enveloppe
    :class:`~media_restorer.extensions.auto_face_id_register.gui.AutoFaceIdRegisterGUI`
    derrière le protocole :class:`~media_restorer.extensions.Extension`.
    """

    name = "Auto Face ID Register"
    description = (
        "Détecter automatiquement les repères (yeux, nez…) sur un dessin/"
        "caricature via un modèle de vision, corriger à la souris, puis "
        "enregistrer dans les métadonnées"
    )
    icon = ":/icons/smiley.png"

    def launch(self, context: ExtensionContext) -> QMainWindow:
        """Ouvre la fenêtre avec *context* déjà chargé.

        Import différé (comme les autres extensions) pour ne compiler l'UI et
        les ressources qu'au lancement réel.
        """
        from media_restorer.extensions.auto_face_id_register.gui import (
            AutoFaceIdRegisterGUI,
        )

        window = AutoFaceIdRegisterGUI(target_path=context.path)
        window.show()
        return window


register(AutoFaceIdRegisterExtension())

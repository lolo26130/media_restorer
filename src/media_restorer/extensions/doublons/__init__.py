"""Extension Doublons — détection de dessins en double.

Sous-paquet isolé, même schéma que les autres extensions.  Le cœur de calcul
vit sans Qt dans :mod:`media_restorer.engines.duplicates` ; cette extension n'en
est que l'interface.  Importer ce paquet suffit à enregistrer
:class:`DoublonsExtension`.
"""
from __future__ import annotations

from PyQt6.QtWidgets import QMainWindow

from media_restorer.extensions import ExtensionContext, register


class DoublonsExtension:
    """Trouve les dessins en double, même retournés, redimensionnés ou recadrés.

    Enveloppe :class:`~media_restorer.extensions.doublons.gui.DoublonsGUI`
    derrière le protocole :class:`~media_restorer.extensions.Extension`.
    """

    name = "Doublons"
    description = (
        "Détecter les dessins en double indépendamment d'une rotation, d'un "
        "changement d'échelle, de contraste ou d'un recadrage — y compris "
        "lorsqu'un dessin est contenu dans un autre — puis étiqueter les "
        "groupes pour DigiKam"
    )
    icon = ":/icons/graph-32x32-14320.png"

    def launch(self, context: ExtensionContext) -> QMainWindow:
        """Ouvre la fenêtre avec *context* déjà chargé.

        Import différé (comme les autres extensions) pour ne compiler l'UI et
        les ressources qu'au lancement réel.
        """
        from media_restorer.extensions.doublons.gui import DoublonsGUI

        window = DoublonsGUI(target_path=context.path, recursive=context.recursive)
        window.show()
        return window


register(DoublonsExtension())

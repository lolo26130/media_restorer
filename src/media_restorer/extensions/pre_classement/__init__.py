"""Extension Pré-classement — tri grossier d'un corpus de dessins.

Sous-paquet isolé, même schéma que les autres extensions.  Le cœur de calcul
vit sans Qt dans :mod:`media_restorer.engines.triage` (mesures, catalogue de
critères, écriture des étiquettes) ; cette extension n'en est que l'interface.
Importer ce paquet suffit à enregistrer :class:`PreClassementExtension`.
"""
from __future__ import annotations

from PyQt6.QtWidgets import QMainWindow

from media_restorer.extensions import ExtensionContext, register


class PreClassementExtension:
    """Classe un répertoire d'images selon des critères globaux, puis les étiquette.

    Enveloppe
    :class:`~media_restorer.extensions.pre_classement.gui.PreClassementGUI`
    derrière le protocole :class:`~media_restorer.extensions.Extension`.
    """

    name = "Pré-classement"
    description = (
        "Classer tout un répertoire selon des critères globaux robustes "
        "(orientation, densité et couleur d'encre, teinte du papier, "
        "résolution), puis écrire le résultat en étiquettes DigiKam"
    )
    icon = ":/icons/checkboard-32x32-456335.png"

    def launch(self, context: ExtensionContext) -> QMainWindow:
        """Ouvre la fenêtre avec *context* déjà chargé.

        Import différé (comme les autres extensions) pour ne compiler l'UI et
        les ressources qu'au lancement réel.
        """
        from media_restorer.extensions.pre_classement.gui import PreClassementGUI

        window = PreClassementGUI(
            target_path=context.path, recursive=context.recursive
        )
        window.show()
        return window


register(PreClassementExtension())

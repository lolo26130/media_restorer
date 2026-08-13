"""Extension Signatures — classement des dessins par signature de dessinateur.

Sous-paquet isolé, même schéma que les autres extensions.  Le cœur de calcul
vit sans Qt dans :mod:`media_restorer.engines.signatures` (localisation,
comparaison, bibliothèque de signatures, écriture des étiquettes) ; cette
extension n'en est que l'interface.  Importer ce paquet suffit à enregistrer
:class:`SignaturesExtension`.
"""
from __future__ import annotations

from PyQt6.QtWidgets import QMainWindow

from media_restorer.extensions import ExtensionContext, register


class SignaturesExtension:
    """Classe les dessins d'un répertoire selon la signature de leur auteur.

    Enveloppe :class:`~media_restorer.extensions.signatures.gui.SignaturesGUI`
    derrière le protocole :class:`~media_restorer.extensions.Extension`.
    """

    name = "Signatures"
    description = (
        "Reconnaître le dessinateur d'un dessin à sa signature, à partir "
        "d'un lot de signatures de référence qui se constitue "
        "progressivement (crops fournis par l'utilisateur quand la "
        "reconnaissance échoue ou hésite), puis écrire le résultat en "
        "étiquette DigiKam"
    )
    icon = ":/icons/tag-label.png"

    def launch(self, context: ExtensionContext) -> QMainWindow:
        """Ouvre la fenêtre avec *context* déjà chargé.

        Import différé (comme les autres extensions) pour ne compiler l'UI et
        les ressources qu'au lancement réel.
        """
        from media_restorer.extensions.signatures.gui import SignaturesGUI

        window = SignaturesGUI(
            target_path=context.path, recursive=context.recursive
        )
        window.show()
        return window


register(SignaturesExtension())

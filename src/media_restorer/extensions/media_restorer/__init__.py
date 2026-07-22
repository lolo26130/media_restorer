"""Extension Media Restorer — restauration interactive de photos anciennes.

Sous-paquet isolé : sa fenêtre (``gui.py``), son mixin Colab
(``colab_calc.py``) et son interface ``.ui`` (``views/main.ui``) sont propres
à cette extension et n'en sortent pas.  Importer ce paquet suffit à
enregistrer :class:`MediaRestorerExtension` — voir
:mod:`media_restorer.extensions` pour le mécanisme général et la marche à
suivre pour ajouter une nouvelle extension.
"""
from __future__ import annotations

from PyQt6.QtWidgets import QMainWindow

from media_restorer.extensions import ExtensionContext, register


class MediaRestorerExtension:
    """Restauration de photos anciennes — l'extension historique de l'application.

    Enveloppe :class:`~media_restorer.extensions.media_restorer.gui.PhotoRestorationGUI`
    (Real-ESRGAN, SwinIR, LaMa, GFPGAN, Double-exposition) derrière le
    protocole :class:`~media_restorer.extensions.Extension`, pour que la
    fenêtre racine puisse la lancer sans connaître ses détails d'implémentation.
    """

    name = "Media Restorer"
    description = (
        "Restauration de photos anciennes — Real-ESRGAN, SwinIR, LaMa, "
        "GFPGAN, Double-exposition"
    )
    icon = ":/icons/object-rotate-right.png"

    def launch(self, context: ExtensionContext) -> QMainWindow:
        """Ouvre Media Restorer avec *context* déjà chargé.

        Importé ici (et non en tête de module) pour ne charger
        :mod:`media_restorer.extensions.media_restorer.gui` — qui compile
        l'interface ``.ui`` et les ressources Qt au premier import — qu'au
        moment où l'extension est réellement lancée, pas dès que le registre
        est peuplé.
        """
        from media_restorer.extensions.media_restorer.gui import PhotoRestorationGUI

        window = PhotoRestorationGUI(
            target_path=context.path, recursive=context.recursive
        )
        window.show()
        return window


register(MediaRestorerExtension())

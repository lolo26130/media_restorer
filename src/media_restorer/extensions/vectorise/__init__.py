"""Extension Vectorise — analyse topologique et retraçage de dessins.

Sous-paquet isolé, même schéma que
:mod:`~media_restorer.extensions.media_restorer` : sa fenêtre (``gui.py``)
et son interface ``.ui`` (``views/main.ui``) sont propres à cette extension.
Le cœur de calcul (GUDHI, squelettisation, texture, rendu) vit séparément
dans :mod:`media_restorer.engines.vectorise`, sans dépendance Qt.  Importer
ce paquet suffit à enregistrer :class:`VectoriseExtension` — voir
:mod:`media_restorer.extensions` pour le mécanisme général.
"""
from __future__ import annotations

from PyQt6.QtWidgets import QMainWindow

from media_restorer.extensions import ExtensionContext, register


class VectoriseExtension:
    """Analyse topologique et retraçage de dessins — imagine des tracés plausibles.

    Enveloppe :class:`~media_restorer.extensions.vectorise.gui.VectoriseGUI`
    derrière le protocole :class:`~media_restorer.extensions.Extension`, pour
    que la fenêtre racine puisse la lancer sans connaître ses détails
    d'implémentation.
    """

    name = "Vectorise"
    description = (
        "Analyse topologique (GUDHI) et retraçage de dessins — imagine des "
        "tracés plausibles, texture de grain, recomposition vectorisée"
    )
    icon = ":/icons/graph-32x32-14320.png"

    def launch(self, context: ExtensionContext) -> QMainWindow:
        """Ouvre Vectorise avec *context* déjà chargé.

        Importé ici (et non en tête de module) pour ne charger
        :mod:`media_restorer.extensions.vectorise.gui` — qui compile
        l'interface ``.ui`` et les ressources Qt au premier import — qu'au
        moment où l'extension est réellement lancée.  Un répertoire choisi
        dans la fenêtre racine n'est pas un problème : outil à image unique,
        géré côté fenêtre (voir ``VectoriseGUI._apply_target``).
        """
        from media_restorer.extensions.vectorise.gui import VectoriseGUI

        window = VectoriseGUI(target_path=context.path)
        window.show()
        return window


register(VectoriseExtension())

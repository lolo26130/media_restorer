Media Restorer — Documentation
================================

Application PyQt6 de restauration de vieilles photos utilisant Real-ESRGAN,
SwinIR, LaMa et GFPGAN.  Le traitement peut s'exécuter sur CPU, sur un GPU
AMD local via ROCm, ou sur un GPU distant via Google Colab.

Le point d'entrée de l'application est la fenêtre racine « Image Treatment »
(:mod:`media_restorer.gui_root`) : elle choisit une cible (fichier ou
répertoire) et une apparence/mode de tooltips, puis lance l'un des outils
enregistrés dans :mod:`media_restorer.extensions` — Media Restorer
aujourd'hui (:mod:`media_restorer.gui`), d'autres outils demain sans
modification de la fenêtre racine.

.. toctree::
   :maxdepth: 2
   :caption: Guides

   acceleration


.. toctree::
   :maxdepth: 2
   :caption: Référence API

   api/gui_root
   api/extensions
   api/engines
   api/colab
   api/gui
   api/utils


Moteurs de restauration
-----------------------

.. autosummary::
   :nosignatures:

   media_restorer.engines.realesrgan_engine.RealESRGANEngine
   media_restorer.engines.swinir_engine.SwinIREngine
   media_restorer.engines.lama_engine.LaMaEngine
   media_restorer.engines.gfpgan_engine.GFPGANEngine

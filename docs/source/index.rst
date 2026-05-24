Media Restorer — Documentation
================================

Application PyQt6 de restauration de vieilles photos utilisant Real-ESRGAN,
SwinIR, LaMa et GFPGAN.  Le traitement peut s'exécuter sur CPU, sur un GPU
AMD local via ROCm, ou sur un GPU distant via Google Colab.

.. toctree::
   :maxdepth: 2
   :caption: Guides

   acceleration


.. toctree::
   :maxdepth: 2
   :caption: Référence API

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

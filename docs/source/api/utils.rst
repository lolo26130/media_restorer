Utilitaires
===========

Gestion de l'alimentation
--------------------------

.. automodule:: media_restorer.power
   :members:

Interface en ligne de commande
-------------------------------

.. automodule:: media_restorer.cli
   :members:

.. note::

   ``cli.py`` positionne ``HSA_OVERRIDE_GFX_VERSION=11.0.0`` au démarrage
   pour activer ROCm sur les GPU AMD non listés officiellement (voir
   :doc:`/acceleration`).

Compatibilité torchvision
--------------------------

.. automodule:: media_restorer._compat
   :members:

Préférences persistantes
--------------------------

.. automodule:: media_restorer.app_settings
   :members:

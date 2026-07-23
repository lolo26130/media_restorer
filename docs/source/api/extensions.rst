Extensions
==========

.. automodule:: media_restorer.extensions
   :members:
   :undoc-members:
   :show-inheritance:

Extension : Media Restorer
---------------------------

L'extension historique de l'application — restauration interactive de
photos anciennes.  Isolée dans son propre sous-paquet
(``extensions/media_restorer/``) : sa fenêtre, son mixin Colab et son
interface ``.ui`` n'en sortent pas.

.. automodule:: media_restorer.extensions.media_restorer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: media_restorer.extensions.media_restorer.gui
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: media_restorer.extensions.media_restorer.colab_calc
   :members:
   :undoc-members:
   :show-inheritance:

Extension : Vectorise
-----------------------

Analyse topologique (GUDHI) et retraçage de dessins.  Isolée dans son
propre sous-paquet (``extensions/vectorise/``) : sa fenêtre et son
interface ``.ui`` n'en sortent pas ; le cœur de calcul (sans dépendance Qt)
vit séparément, voir :doc:`engines` — section « Vectorise ».

.. automodule:: media_restorer.extensions.vectorise
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: media_restorer.extensions.vectorise.gui
   :members:
   :undoc-members:
   :show-inheritance:

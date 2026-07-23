Moteurs de restauration
=======================

.. automodule:: media_restorer.engines
   :members:
   :undoc-members:

.. autoclass:: media_restorer.engines.base.BaseEngine
   :members:
   :show-inheritance:

Real-ESRGAN
-----------

.. autoclass:: media_restorer.engines.realesrgan_engine.RealESRGANEngine
   :members:
   :show-inheritance:

SwinIR
------

.. autoclass:: media_restorer.engines.swinir_engine.SwinIREngine
   :members:
   :show-inheritance:

LaMa
----

.. autoclass:: media_restorer.engines.lama_engine.LaMaEngine
   :members:
   :show-inheritance:

GFPGAN
------

.. autoclass:: media_restorer.engines.gfpgan_engine.GFPGANEngine
   :members:
   :show-inheritance:

Double-exposition
------------------

.. autoclass:: media_restorer.engines.dual_engine.DualExposureEngine
   :members:
   :show-inheritance:

Vectorise
---------

Cœur de calcul de l'extension Vectorise — analyse topologique (GUDHI) et
retraçage de dessins.  Ne dérive pas de :class:`~media_restorer.engines.base.BaseEngine`
(contrat différent : une image en entrée, plusieurs candidats de retraçage
en sortie) — voir la docstring de module pour la justification complète.

.. automodule:: media_restorer.engines.vectorise
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: media_restorer.engines.vectorise.topology
   :members:
   :show-inheritance:

.. automodule:: media_restorer.engines.vectorise.tracing
   :members:
   :show-inheritance:

.. automodule:: media_restorer.engines.vectorise.texture
   :members:
   :show-inheritance:

.. automodule:: media_restorer.engines.vectorise.render
   :members:
   :show-inheritance:

.. automodule:: media_restorer.engines.vectorise.storage
   :members:
   :show-inheritance:

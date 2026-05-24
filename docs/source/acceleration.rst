Accélération GPU
================

Media Restorer supporte deux modes d'accélération GPU : un **GPU local AMD
via ROCm** et un **GPU distant via Google Colab**.  Les deux chemins partagent
la même interface côté Python — le code de restauration ne change pas.

.. contents:: Sommaire
   :local:
   :depth: 2


GPU local — AMD ROCm
--------------------

Stratégie générale
~~~~~~~~~~~~~~~~~~

PyTorch ROCm expose les GPU AMD sous la même API que CUDA (``torch.cuda.*``).
Chaque moteur interroge ``torch.cuda.is_available()`` au moment de la
construction de son backend ; si la réponse est ``True``, le modèle est chargé
en mémoire GPU avec ``half=True`` (précision FP16), sinon il reste sur CPU en
FP32.

Ce choix est entièrement automatique : aucune modification de code n'est
nécessaire entre une machine avec GPU et une machine sans.

**Cache du modèle.** Le chargement des poids depuis le disque et leur transfert
vers le GPU représente une fraction significative du temps total sur les petites
images.  Pour éviter de le répéter à chaque restauration, chaque moteur met en
cache son backend interne dès le premier appel et le réutilise pour toute la
durée de vie de l'instance.  L'interface graphique maintient une instance par
moteur, donc le chargement n'a lieu qu'une seule fois par session.

**Découpage en tuiles (tile).** Les grandes images sont découpées en tuiles
traitées indépendamment pour éviter les débordements de mémoire GPU.  La valeur
par défaut de 256 px offre le meilleur rapport vitesse/mémoire sur les GPU
intégrés ; augmenter cette valeur accélère les GPU discrets disposant d'une
large bande passante mémoire.

**FP16 (half precision).** Activé automatiquement sur GPU, il divise l'empreinte
mémoire par deux et accélère les convolutions sur les architectures RDNA et
Ampere/Ada sans perte visible de qualité pour la super-résolution.

Activation
~~~~~~~~~~

Installer PyTorch avec le support ROCm ::

    uv sync   # le pyproject.toml pointe déjà sur l'index pytorch-rocm5.7

PyTorch ROCm est sélectionné via l'index ``pytorch-rocm`` dans
``pyproject.toml`` ::

    [tool.uv.sources]
    torch = [{ index = "pytorch-rocm" }]

    [[tool.uv.index]]
    name = "pytorch-rocm"
    url  = "https://download.pytorch.org/whl/rocm5.7"
    explicit = true

Note spécifique — AMD Radeon 780M (Ryzen 8845HS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

La Radeon 780M est un GPU intégré (iGPU) basé sur l'architecture RDNA3,
identifié par ROCm comme **gfx1103** (Phoenix APU).  ROCm 5.7 ne liste
officiellement que jusqu'à **gfx1102** ; gfx1103 partage cependant la même
microarchitecture et les mêmes noyaux de calcul que gfx1100 (RX 7900).

Le flag d'environnement positionné dans :mod:`media_restorer.cli` au démarrage ::

    HSA_OVERRIDE_GFX_VERSION=11.0.0

demande au runtime HSA de substituer ``11.0.0`` (gfx1100) à l'identifiant
détecté (gfx1103) lors de la sélection des noyaux compilés.  L'exécution est
stable ; aucun résultat incorrect n'a été observé en pratique.

La 780M est une **iGPU** : elle ne dispose pas de mémoire dédiée mais
adresse directement la RAM système.  PyTorch voit ~47 Go de « VRAM » (sur
92 Go de RAM système).  Les transferts CPU ↔ GPU sont des copies en mémoire
partagée, moins coûteuses que sur une carte discrète mais non négligeables —
d'où l'importance du cache du modèle.

**Performances mesurées** sur cette machine (RealESRGAN ×4, photo 602 × 596 px,
9 tuiles 256 px, modèle déjà en cache) :

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Configuration
     - Temps
     - Gain
     - Notes
   * - GPU Radeon 780M (ROCm 5.7, FP16)
     - ~3,8 s
     - ×10
     - modèle en cache
   * - CPU Ryzen 8845HS (16 cœurs)
     - ~38,6 s
     - ×1 (référence)
     - FP32
   * - Premier appel GPU (modèle non chargé)
     - ~37 s
     - —
     - inclut chargement + transfert


Google Colab
------------

Architecture
~~~~~~~~~~~~

Lorsqu'aucun GPU local n'est disponible (ou pour des images très grandes),
les calculs peuvent être délégués à un **serveur FastAPI** hébergé dans un
notebook Google Colab exposé via un tunnel `cloudflared
<https://github.com/cloudflare/cloudflared>`_.  Le flux est :

.. code-block:: text

    App locale  ──POST /process──►  FastAPI (Colab GPU T4/L4)
                ◄── image PNG base64 ──────────────────────────

La classe :class:`~media_restorer.colab_calc.ColabCalc` est un **mixin pur
Python** dont hérite :class:`~media_restorer.gui.PhotoRestorationGUI`.  Elle
gère l'URL du tunnel, le worker de communication et le même signal
``result_ready`` que les moteurs locaux — le slot d'affichage et la sauvegarde
sont identiques dans les deux cas.

Mise en route
~~~~~~~~~~~~~

1. Ouvrir ``colab/server.ipynb`` dans `Google Colab
   <https://colab.research.google.com>`_.
2. **Runtime › Run all** — les cellules installent les librairies et téléchargent
   les modèles sur Google Drive (une seule fois), démarrent le serveur FastAPI
   et lancent le tunnel.
3. La cellule 5 affiche une URL du type
   ``https://abc-def-123.trycloudflare.com``.
4. Dans Media Restorer : **Colab › Connecter Colab** (``Ctrl+Shift+C``),
   coller l'URL.
5. Charger une image, puis **Colab › Restaurer (Colab)** (``Ctrl+Shift+R``).

.. note::

   L'URL change à chaque démarrage du notebook.  Les modèles et librairies
   sur Drive sont conservés entre les sessions ; seul le tunnel doit être
   recréé.

Moteurs disponibles côté Colab
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Moteur
     - Supporté
     - Notes
   * - Real-ESRGAN ×4
     - Oui
     - ``RealESRGAN_x4plus.pth`` sur Drive
   * - GFPGAN v1.4
     - Oui
     - ``GFPGANv1.4.pth`` + modèles facexlib sur Drive
   * - SwinIR
     - Non
     - non intégré dans le notebook pour l'instant
   * - LaMa
     - Non
     - non intégré dans le notebook pour l'instant

Isolation de l'état Colab
~~~~~~~~~~~~~~~~~~~~~~~~~

Les attributs de connexion (URL, worker) sont **name-mangés** dans
:class:`~media_restorer.colab_calc.ColabCalc` (``__url`` →
``_ColabCalc__url``) pour les isoler des attributs de la fenêtre principale.
Les méthodes d'accès sont préfixées ``_colab__`` pour matérialiser cette
frontière dans le code.


Traitement par lot — stabilité et progression
---------------------------------------------

Prévention du crash DRM sur iGPU (flip_done timedout)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sur un iGPU partagé (AMD Radeon 780M, et de façon générale tout GPU
intégré), le compositor KDE (KWin) et PyTorch/ROCm se disputent le même
silicium.  Lors d'un traitement de répertoire en rafale, PyTorch monopolise
la file de commandes GPU pendant plusieurs minutes ; KDE dépasse alors le
délai d'attente pour ses rafraîchissements d'écran (*vsync flip*) et le
driver DRM panique :

.. code-block:: none

    amdgpu 0000:65:00.0: [drm] *ERROR* flip_done timedout
    amdgpu 0000:65:00.0: [drm] *ERROR* [CRTC:88:crtc-2] commit wait timed out

**Correction appliquée dans** :class:`~media_restorer.gui._BatchRestoreWorker` :

- ``torch.cuda.synchronize()`` — vide la file de kernels GPU après chaque
  image (sans cela les kernels ROCm se cumulent en mémoire tampon)
- ``time.sleep(0.05)`` — pause de 50 ms (~3 cycles vsync à 60 Hz) pour que
  KWin puisse effectuer ses flips d'affichage avant l'image suivante

Le surcoût est négligeable (< 2 % sur des images de plusieurs secondes).

**Correctif noyau recommandé** — ajouter dans ``/etc/default/grub`` ::

    GRUB_CMDLINE_LINUX_DEFAULT="... amdgpu.gpu_recovery=1"

puis ``sudo update-grub`` et redémarrage.  Ce paramètre active la
récupération automatique du GPU en cas de timeout résiduel, évitant le crash
de session même si la pause ne suffit pas.

Exclusion des sorties précédentes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Le worker de lot détecte et ignore automatiquement tout fichier situé dans
un sous-répertoire portant le nom d'un moteur (``Real-ESRGAN/``,
``SwinIR/``, ``LaMa/``, ``GFPGAN/``), quelle que soit leur profondeur dans
l'arborescence.  Les images déjà restaurées ne sont donc jamais retraitées.

Progression en temps réel
~~~~~~~~~~~~~~~~~~~~~~~~~~

La barre de statut affiche après chaque image :

.. code-block:: none

    3 / 12 images  —  moy. 4.2 s/img  —  photo_007.jpg

Le temps affiché est le **temps réel** (horloge murale), qui inclut le calcul
GPU contrairement à ``time.process_time()`` qui n'aurait mesuré que le CPU.

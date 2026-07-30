Projet
media_restorer — PyQt6, restauration photo
- Repo : ~/Documents/Python/aaa_modules/media_restorer
- Package : src/media_restorer/

Architecture
- gui_root.py     → fenêtre racine ImageTreatmentWindow (« Image Treatment »),
                    point d'entrée de l'app (cli.py → gui_root.run_gui()) ;
                    choisit une cible (fichier/répertoire) puis lance une
                    extension du registre media_restorer.extensions
- views/root.ui   → .ui de la fenêtre racine, compilé automatiquement
- extensions/     → registre léger (Extension, ExtensionContext, register(),
                    all_extensions()) — chaque extension = sous-paquet isolé
  - extensions/media_restorer/ → l'extension historique (restauration photo)
    - gui.py          → fenêtre PhotoRestorationGUI (Media Restorer)
    - colab_calc.py   → mixin ColabCalc (QThread, propre à cette extension)
    - views/main.ui   → .ui de Media Restorer, compilé automatiquement
  - extensions/vectorise/ → analyse topologique (GUDHI) et retraçage de dessins
    - gui.py          → fenêtre VectoriseGUI (Vectorise)
    - views/main.ui   → .ui de Vectorise, compilé automatiquement
  - extensions/manual_mouse_points/ → désignation de repères (yeux, bouche…)
                    à la souris, stockés dans les métadonnées de l'image
    - image_click.py  → widget ImageClick (pg.ImageView) : survol + Entrée/
                    Espace/Q, signaux tagging_finished/point_marked. Copié
                    puis adapté (sans importation) de TraiteImages
                    (classes/data_classes.py), mixins CommonQtMethods retirés
    - gui.py          → fenêtre ManualMousePointsGUI (Manual Mouse Points)
    - views/main.ui   → .ui, compilé automatiquement
- engines/        → RealESRGAN, SwinIR, LaMa, GFPGAN, DualExposure (héritent de BaseEngine)
                    DualExposure = fusion front light / back light, sans réseau (OpenCV pur)
                    bibliothèque cœur, sans dépendance Qt — réutilisée par
                    restore.py (CLI photo) indépendamment de toute extension
  - engines/vectorise/ → cœur de calcul de l'extension Vectorise, ne dérive
                    PAS de BaseEngine (contrat différent — voir sa docstring) :
                    topology.py (GUDHI : AlphaComplex + homologie persistante H0/H1
                    → arbre couvrant minimal = squelette du dessin, sans
                    scikit-image ni networkx), tracing.py (décomposition en
                    traits, Python pur), texture.py, render.py, storage.py (HDF5)
- imaging.py      → opérations image partagées (ex. lowpass(), utilisé par
                    dual_engine.py ET engines/vectorise/texture.py)
- landmarks.py    → LandmarkSet : lecture/écriture de points nommés dans les
                    métadonnées via exiftool (sous-processus, runner injectable
                    pour les tests). Cœur sans Qt de l'extension
                    manual_mouse_points. Duplique+recentre la gestion EXIF de
                    DataImages (TraiteImages). Tag UserComment, JSON sous la
                    clé « media_restorer_landmarks » ; exiftool garde <img>_original
- download_models.py → téléchargement des poids (MODEL_REGISTRY), point
                    d'entrée autonome : python -m media_restorer.download_models
- resources/icons/ → icônes partagées entre la racine et toutes les extensions
                    (ne pas dupliquer par extension — voir extensions/__init__.py)
- OutilsQt/Utils_Qt.py → compile_ui, compile_qrc, tooltips_from_code

## Conventions importantes

    - @pyqtSlot() sur tous les on_<widget>_<signal> → connectSlotsByName, PAS de connect() manuel
    - QAction dans PyQt6.QtGui (pas QtWidgets)
    - Workers QThread : signal nommé result_ready (pas finished, qui shadowe QThread)
    - Imports lourds (torch, basicsr) différés dans les fonctions

## État actuel
    - [toute la gui est sous forme de fichier .ui, avec des ressources (icones) compilées au lancement]

## État actuel (suite)
    - [Colab implémenté : PhotoRestorationGUI hérite de ColabCalc (name mangling), voir extensions/media_restorer/colab_calc.py]
    - [fenêtre racine + registre d'extensions en place ; prochaine extension candidate : film.py (restauration vidéo, actuellement un stub CLI NotImplementedError, aucune GUI)]
    - [extension Vectorise en place : get_outline/show/save/save_texture_from_image/select_texture/vectorise]
    - [extension Manual Mouse Points en place : désignation de repères à la
       souris (image_click.py, copié de TraiteImages) + écriture dans les
       métadonnées via exiftool (landmarks.py), avec confirmation utilisateur]

## Piège connu — tests Qt avec de vrais QThread/pg.ImageView
    Démarrer un vrai QThread (worker.start()) puis attendre son résultat via
    une boucle d'événements imbriquée (qtbot.waitSignal, qtbot.waitUntil, ou
    une QEventLoop manuelle) a provoqué un segfault reproductible une fois
    assez de fenêtres pg.ImageView accumulées dans le même process de test
    (voir tests/test_gui_vectorise.py::_run_get_outline). Dans les tests,
    appeler worker.run() directement (méthode Python ordinaire, sans jamais
    démarrer de vrai thread OS) au lieu de worker.start() + attente — mêmes
    signaux émis, aucun risque. performance_mode() est neutralisé pour toute
    la suite dans tests/conftest.py (fixture _no_real_power_management).


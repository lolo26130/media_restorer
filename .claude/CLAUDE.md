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
- engines/        → RealESRGAN, SwinIR, LaMa, GFPGAN, DualExposure (héritent de BaseEngine)
                    DualExposure = fusion front light / back light, sans réseau (OpenCV pur)
                    bibliothèque cœur, sans dépendance Qt — réutilisée par
                    restore.py (CLI photo) indépendamment de toute extension
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


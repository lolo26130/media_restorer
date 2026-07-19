Projet
media_restorer — PyQt6, restauration photo
- Repo : ~/Documents/Python/aaa_modules/media_restorer
- Package : src/media_restorer/

Architecture
- gui.py          → fenêtre principale PhotoRestorationGUI
- engines/        → RealESRGAN, SwinIR, LaMa, GFPGAN, DualExposure (héritent de BaseEngine)
                    DualExposure = fusion front light / back light, sans réseau (OpenCV pur)
- download_models.py → téléchargement des poids (MODEL_REGISTRY)
- views/main.ui   → compilé automatiquement via compile_ui()
- OutilsQt/Utils_Qt.py → compile_ui, compile_qrc, tooltips_from_code

## Conventions importantes

    - @pyqtSlot() sur tous les on_<widget>_<signal> → connectSlotsByName, PAS de connect() manuel
    - QAction dans PyQt6.QtGui (pas QtWidgets)
    - Workers QThread : signal nommé result_ready (pas finished, qui shadowe QThread)
    - Imports lourds (torch, basicsr) différés dans les fonctions

## État actuel
    - [toute la gui est sous forme de fichier .ui, avec des ressources (icones) compilées au lancement]

## Problème en cours
    [il reste à ajouter la possibilité d'effectuer les traitements d'image sur google collab]
    [la classe PhotoRestorationGUI va hériter de ColabCalc, on va utiliser du name mangling dans cette classe qui contiendra les méthodes dédiées à Google-collab]


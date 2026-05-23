Voici les deux commandes à mémoriser :

Dans le worktree (pour commiter) :
git add -A && git commit

Depuis n'importe où (pour merger dans main) :
git -C /home/laurent/Documents/Python/aaa_modules/Bringing-Old-Photos-Back-to-Life merge --ff-only worktree-ui-engines-refactor

Si le merge échoue à cause de fichiers stagés dans le dépôt principal (comme gui.py plusieurs fois) :
git -C /home/laurent/Documents/Python/aaa_modules/Bringing-Old-Photos-Back-to-Life reset HEAD && git -C /home/laurent/Documents/Python/aaa_modules/Bringing-Old-Photos-Back-to-Life checkout -- .
puis relancer le merge.




      Voici un template compact à remplir à la main, basé sur ce qu'on a fait :


Projet
Bringing-Old-Photos-Back-to-Life — PyQt6, restauration photo
- Repo : ~/Documents/Python/aaa_modules/Bringing-Old-Photos-Back-to-Life
- Package : src/media_restorer/

Architecture
- gui.py          → fenêtre principale PhotoRestorationGUI
- engines/        → RealESRGAN, SwinIR, LaMa, GFPGAN (héritent de BaseEngine)
- download_models.py → téléchargement des poids (MODEL_REGISTRY)
 - views/main.ui   → compilé automatiquement via compile_ui()
      - OutilsQt/Utils_Qt.py → compile_ui, compile_qrc, tooltips_from_code

      ## Conventions importantes
    - @pyqtSlot() sur tous les on_<widget>_<signal> → connectSlotsByName, PAS de connect() manuel
    - QAction dans PyQt6.QtGui (pas QtWidgets)
    - Workers QThread : signal nommé result_ready (pas finished, qui shadowe QThread)
    - Imports lourds (torch, basicsr) différés dans les fonctions

    ## État actuel
    - [ce que tu as fait / ce qui marche]
    - [ce qui reste à faire]

    ## Problème en cours
    [description précise]

    ---
    Ce qu'il faut garder, ce qu'on peut couper


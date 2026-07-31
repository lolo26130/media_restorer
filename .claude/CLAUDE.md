Projet
media_restorer — PyQt6, restauration photo
- Repo : ~/Documents/Python/aaa_modules/media_restorer
- Package : src/media_restorer/

Architecture
- gui_root.py     → fenêtre racine ImageTreatmentWindow (« Image Treatment »),
                    point d'entrée de l'app (cli.py → gui_root.run_gui()) ;
                    choisit une cible (fichier/répertoire) puis lance une
                    extension du registre media_restorer.extensions.
                    Dock « Infos, Exif » (InfoExifPanel) : métadonnées de la
                    cible (image → EXIF ; répertoire → résumé). Se met à jour
                    en direct sur l'image en cours pendant un traitement par
                    lot si l'extension expose le signal OPTIONNEL
                    current_image_changed(object) — Path affiche, None revient
                    au résumé (contrat duck-typé, doc dans extensions/__init__)
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
    - gui.py          → fenêtre ManualMousePointsGUI (Manual Mouse Points).
                    Utilise le widget partagé image_click.py + landmark_config.py
                    (déplacés au cœur, voir plus bas) + landmarks.py
    - views/main.ui   → .ui, compilé automatiquement
  - extensions/auto_face_id_register/ → détection AUTO de repères sur dessins/
                    caricatures (là où les détecteurs entraînés sur photos
                    échouent), puis revue/correction à la souris, puis
                    enregistrement dans les métadonnées
    - gui.py          → fenêtre AutoFaceIdRegisterGUI ; worker _DetectWorker
                    (QThread, result_ready) ; réutilise ImageClick pour la
                    correction (un « passer » garde la valeur détectée, seul un
                    re-marquage écrase) ; repères détectés superposés en cyan
    - config.py       → modèle de détection + appareil (detection_device
                    cpu/gpu, DÉFAUT cpu car le GPU ROCm 780M peut se figer au
                    chargement du modèle) choisis au 1er usage, persistés dans
                    SON propre TOML (auto_face_id_register.toml) ; la LISTE de
                    repères, elle, vient du landmark_config PARTAGÉ
    - views/main.ui   → .ui, compilé automatiquement
- image_click.py  → widget PARTAGÉ ImageClick (pg.ImageView) : survol + Entrée/
                    Espace/Q, signaux tagging_finished/point_marked, overlay
                    show_existing_points. Coordonnées émises/stockées en
                    POURCENTAGE (0–100, float) de la largeur/hauteur (converties
                    depuis/vers les pixels pour le dessin via _px_to_pct/
                    _pct_to_px). Copié+adapté (sans import) de
                    TraiteImages (data_classes.py), mixins CommonQtMethods
                    retirés. Au cœur (pas dans une extension) → réutilisé par
                    manual_mouse_points ET auto_face_id_register sans qu'elles
                    dépendent l'une de l'autre (comme gui_widgets.ResultWindow)
- exif_info.py    → lecteur de métadonnées SANS Qt pour le dock « Infos, Exif »
                    (read_image_info / read_directory_summary). Un seul lecteur :
                    exiftool -j (runner injectable comme landmarks.py), repli
                    Pillow si exiftool absent — ne lève jamais
- landmark_config.py → liste de repères PARTAGÉE, persistée dans UN TOML unique
                    (~/.config/media_restorer/manual_mouse_points.toml, chemin
                    dérivé de app_settings().fileName() → isolé en test via la
                    redirection QSettings de conftest). Lecture tomllib (stdlib),
                    écriture à la main (pas de tomli_w) via json.dumps par libellé.
                    Au cœur car partagée par les deux extensions à repères
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
  - engines/face_id/ → cœur de auto_face_id_register, sans Qt, ne dérive PAS de
                    BaseEngine. detect.py : détection ZERO-SHOT (vocabulaire
                    ouvert) pilotée par la liste de repères = requêtes texte
                    (« Left Eye » → « eye »), centre de la meilleure boîte,
                    gauche/droite attribués par abscisse. build_detector() =
                    pipeline transformers (import lourd différé, modèle
                    téléchargé au 1er usage) ; detect_landmarks() prend un
                    detector INJECTABLE → tests sans téléchargement ni inférence
- imaging.py      → opérations image partagées (ex. lowpass(), utilisé par
                    dual_engine.py ET engines/vectorise/texture.py)
- landmarks.py    → LandmarkSet : lecture/écriture de points nommés dans les
                    métadonnées via exiftool (sous-processus, runner injectable
                    pour les tests). Cœur sans Qt, partagé par manual_mouse_points
                    ET auto_face_id_register. Duplique+recentre la gestion EXIF de
                    DataImages (TraiteImages). Tag UserComment, JSON sous la
                    clé « media_restorer_landmarks » ; exiftool garde <img>_original.
                    Points en POURCENTAGE (0–100, float) → indépendants de la
                    résolution, aucune conversion lors d'un changement de résolution
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
    - [extension Auto Face ID Register en place : détection zero-shot
       (transformers, engines/face_id) des repères sur dessins + revue/
       correction à la souris (ImageClick partagé) + enregistrement métadonnées.
       image_click.py et landmark_config.py DÉPLACÉS au cœur (étaient dans
       manual_mouse_points) pour partage sans dépendance inter-extensions.
       Dépendance transformers ajoutée (uv add)]

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

    Corollaire (extension Manual Mouse Points) : ImageClick EST un pg.ImageView,
    donc chaque fenêtre de test en crée un de plus et rapproche la suite du même
    seuil d'accumulation — sans même de QThread. Correctif : fixture autouse
    _flush_qt_deletions dans tests/test_gui_manual_mouse_points.py (processEvents
    + gc.collect après chaque test) pour purger les deleteLater entre les tests.
    Toute future extension à base de pg.ImageView devrait faire de même.


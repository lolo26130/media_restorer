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
                    au résumé (contrat duck-typé, doc dans extensions/__init__).
                    Dock « Aperçu » (gui_widgets.ImagePreview) piloté par la MÊME
                    règle et le MÊME signal : cible fichier → l'image, cible
                    répertoire → vide (des milliers d'images, aucune à montrer).
                    Une extension qui émet current_image_changed pilote donc les
                    DEUX docks sans rien savoir de leur existence
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
  - extensions/pre_classement/ → tri grossier d'un corpus puis écriture des
                    étiquettes DigiKam. ParameterTree ENGENDRÉ depuis CRITERIA ;
                    _ClassifyWorker et _WriteWorker (QThread, result_ready) ;
                    classer est RÉVERSIBLE, écrire modifie les fichiers → deux
                    actions distinctes, UNE seule confirmation pour tout le lot.
                    Expose current_image_changed (dock « Infos, Exif »).
                    Plafond Mpx (défaut 50) et critères cochés persistés QSettings.
                    DOCK « Aperçu » (gui_widgets.ImagePreview) : branché sur
                    tableResults.itemSelectionChanged — donc AUSSI au CLAVIER
                    (flèches), pas seulement au clic. Chargement DIFFÉRÉ par un
                    QTimer (PREVIEW_DEBOUNCE_MS=120) : une flèche maintenue
                    traverse des dizaines de lignes, seule celle où l'on s'arrête
                    est chargée. Le chemin complet voyage sur la cellule
                    (Qt.ItemDataRole.UserRole) car le tableau est TRIABLE :
                    l'index de ligne ne désigne plus le bon fichier après un tri
  - extensions/doublons/ → recherche, revue et étiquetage des doublons.
                    ParameterTree ENGENDRÉ depuis METHODS (groupé par étage) ;
                    _SearchWorker et _WriteWorker (QThread, result_ready) ;
                    arbre séparant GROUPES et INCLUSIONS ; deux ImagePreview
                    côte à côte + la PHRASE explicative (« B est un détail de
                    A — rotation 15°, couverture 25 % ») qui se vérifie d'un
                    coup d'œil, là où « score 0,83 » ne se vérifie pas.
                    Non destructif : étiquettes + export CSV/HTML, JAMAIS de
                    déplacement ni de suppression (fonds patrimonial)
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
- gui_shots.py    → ShotInventoryPanel : dock « Correspondances RAW » de la
                    racine, TABIFIÉ avec « Infos, Exif » pour ne pas empiler un
                    3e dock. QTabWidget à 4 onglets portant leur effectif +
                    ImagePreview + phrase justifiant l'appariement + export CSV.
                    ⚠ Le scan NE PART JAMAIS TOUT SEUL (bouton explicite) : il
                    coûte ~1 min sur 10 000 fichiers. _ScanWorker (QThread,
                    result_ready), scan_fn et preview_fn INJECTABLES.
                    _on_selection lit l'arbre ÉMETTEUR (self.sender()) et non
                    l'onglet courant. Explication et aperçu sont INDÉPENDANTS :
                    l'appariement reste explicable même sans aperçu.
                    Module séparé de gui_widgets car il connaît engines.shots,
                    là où gui_widgets ne porte que des briques génériques
- gui_widgets.py  → widgets Qt PARTAGÉS : ResultWindow (fenêtre de résultat),
                    InfoExifPanel (arbre des métadonnées), et ImagePreview —
                    pg.ImageView d'aperçu chargé en RÉSOLUTION RÉDUITE via
                    Image.draft() + thumbnail (max 1600 px) : indispensable pour
                    parcourir un tableau au clavier. Mesuré sur le corpus réel :
                    ~55 ms sur un JPEG de 3 Mpx, ~250 ms sur un TIFF de 23 Mpx
                    (draft() n'accélère QUE le JPEG). Transpose (1,0,2) avant
                    setImage — pyqtgraph est col-major, NumPy row-major (même
                    convention que image_click.py). Orientation EXIF appliquée
                    via image_io.apply_exif_orientation (jamais redupliquée).
                    Ne lève jamais : un fichier abîmé affiche un message
- exif_info.py    → lecteur de métadonnées SANS Qt pour le dock « Infos, Exif »
                    (read_image_info / read_directory_summary). Renvoie une
                    hiérarchie {groupe: {tag: valeur}} via exiftool -g -j
                    (regroupement par provenance : File/EXIF/XMP/MakerNotes…,
                    runner injectable comme landmarks.py), repli Pillow (groupe
                    « Image ») si exiftool absent — ne lève jamais. InfoExifPanel
                    (gui_widgets) l'affiche en QTreeWidget repliable (pas
                    pg.DataTreeWidget : 2 colonnes propres, sans colonne « type »)
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
  - engines/triage/ → cœur du PRÉ-CLASSEMENT, sans Qt, ne dérive PAS de
                    BaseEngine (image → mesures scalaires, pas image → image ;
                    donc absent de l'enum Engine, comme vectorise et face_id).
                    signals.py : ImageSignals — MESURES en attributs, CATÉGORIES
                    en propriétés recalculées depuis des seuils en constantes de
                    module (⇒ rejouer un tri avec d'autres seuils ne relit aucun
                    fichier, seules les mesures sont mises en cache). Deux
                    saturations SÉPARÉES : encre (20 % + sombres) = axe CONTENU,
                    papier (40 % + clairs) = axe CONDITION de numérisation ; la
                    saturation MOYENNE les confondrait et rangerait 36 % du
                    corpus (trait noir sur papier jauni) en « couleur ».
                    measure_image(max_megapixels=) lève TooLarge APRÈS lecture
                    d'en-tête et AVANT décodage → plafond gratuit.
                    scan.py : ScanResult(signals, skipped_large, unreadable) —
                    écartée par le plafond ≠ illisible, ne jamais les additionner.
                    iter_images TRIE (sans quoi un échantillon à graine fixée
                    n'est pas reproductible : 21 recouvrements sur 500 mesurés).
                    criteria.py : catalogue DÉCLARATIF (Criterion/CRITERIA) qui
                    pilote le ParameterTree ET les colonnes ⇒ ajouter un critère
                    ne touche aucune ligne de Qt. Libellés de classe
                    AUTO-SUFFISANTS et SANS « / » (séparateur de TagsList).
                    tags.py : branche media_restorer/Tri/<branche>/<classe>,
                    owns restreint. cache.py : JSON, invalidation (taille, mtime)
  - engines/duplicates/ → cœur de la DÉTECTION DE DOUBLONS, sans Qt, ne dérive
                    PAS de BaseEngine (contrat : corpus → graphe). Voir
                    docs/rapport-doublons-dessins.tex pour la méthode et ses
                    mesures. TROIS RÉGIMES : R1 copie géométrique, R2 inclusion
                    partielle (les deux couverts), R3 variante redessinée (PAS
                    couvert — exige des descripteurs appris, lot ultérieur).
                    methods.py : catalogue DÉCLARATIF (Method/METHODS, 3 étages)
                    qui pilote le ParameterTree ⇒ ajouter une méthode ne touche
                    aucune ligne de Qt. descriptors.py : Fourier-Mellin
                    log-polaire (rotation+échelle+translation) et profil
                    d'orientations (peu sensible à l'épaisseur du trait).
                    candidates.py : similarité PAR BLOCS, la matrice complète
                    n'est JAMAIS matérialisée — à 8 693 images elle pèse 0,3 Go,
                    à 43 465 (×5 prévu) 7,6 Go. Résultat EXACT, identique au
                    calcul direct (test verrouillé). Aucun index approché
                    nécessaire sous ~10⁵ images. verify.py : ORB + USAC_MAGSAC
                    (ORB mesuré 8× plus rapide que SIFT et aussi invariant sur
                    ces dessins — contraire à sa réputation, établie sur des
                    photos). merit.py : LA TRANSFORMATION ESTIMÉE EST
                    L'EXPLICATION — décomposition SVD de H (rotation, échelle,
                    anisotropie, exactes à 0,02° et 0,001), COUVERTURES
                    ASYMÉTRIQUES (0,25/1,00 ⇒ « A est un détail de B » ;
                    1,00/1,00 ⇒ même dessin), photométrie et épaisseur mesurées
                    APRÈS recalage. Merit.explain() rend une PHRASE lisible.
                    groups.py : ⚠ PIÈGE DE TRANSITIVITÉ — les composantes
                    connexes ne se calculent QUE sur les arêtes symétriques ;
                    A⊃B et B⊃C n'implique pas A≈C, sinon tout un fonds
                    d'affiches fusionne en un groupe géant. pipeline.py : écarte
                    les paires en régime SÉMANTIQUE (sans famille D, ce ne sont
                    que des candidats rejetés — les remonter ferait 32 « paires
                    incertaines » sur 36, mesuré). tags.py : branche
                    media_restorer/Doublons/, owns RESTREINT.
                    — v2 (variantes redessinées, R3) —
                    device.py : DEVICE_AUTO/CPU/GPU. ⚠ LA SONDE GPU S'EXÉCUTE EN
                    SOUS-PROCESSUS, jamais en thread : un pilote bloqué n'est PAS
                    interruptible depuis le processus appelant (ni Ctrl-C ni
                    join(timeout)) — c'est ce qui avait figé l'app avec OWLv2. Un
                    sous-processus se tue. RÈGLE À SUIVRE POUR TOUTE SONDE
                    MATÉRIELLE FUTURE. HSA_OVERRIDE_GFX_VERSION=11.0.0 posé
                    automatiquement (11.0.2 fige la machine). Mesuré : GPU OK
                    (matmul 4,6×, conv2d OK, ViT 2,2-2,5×) — donc CONFORT, pas
                    prérequis (ViT-B sur Cabrol : 11,4 min CPU / 4,6 min GPU).
                    embeddings.py : catalogue EMBEDDING_MODELS (DINOv2 S/B, CLIP,
                    SigLIP) — transformers 4.57.6 les fournit tous, AUCUNE
                    dépendance nouvelle. Cache invalidé par (taille, mtime) ET
                    CLÉ DE MODÈLE : l'oublier ferait comparer des vecteurs DINOv2
                    à des CLIP, silencieusement. Le rappel on_progress fait
                    partie du CONTRAT (pas d'introspection par except TypeError).
                    verdicts.py : LA BOUCLE QUI FAIT PROGRESSER L'OUTIL. Le seuil
                    sémantique NE PEUT PAS être fixé a priori (v1 : 32 paires
                    « incertaines » sur 36) — il vient des verdicts rendus à la
                    revue. Clé de paire = chemins TRIÉS puis hachés (symétrique,
                    survit à un reclassement). calibrate() optimise le F1 et
                    REFUSE de parler sous MIN_VERDICTS=10.
                    benchmark.py : compare les modèles sur LE MÊME jeu de
                    verdicts — répond par la mesure à ce que la littérature ne
                    tranche pas (DINOv2 devant CLIP, mais sur des PHOTOS).
                    SSCD volontairement absent : entraîné sur des COPIES
                    transformées, inutile pour un redessin.
                    tags.py : feuille « Variante » réservée aux paires
                    CONFIRMÉES — une présomption non validée n'a rien à faire
                    dans les métadonnées d'un fonds patrimonial
  - engines/shots/ → INVENTAIRE des correspondances RAW ↔ JPEG, sans Qt.
                    Répond à « quel JPEG vient de quel NEF » — jointure par
                    MÉTADONNÉES, pas analyse d'image : exacte, sans seuil.
                    ⚠ RAW_SUFFIXES est DISJOINT d'IMAGE_SUFFIXES : les .NEF sont
                    des sauvegardes pour développement ultérieur et n'entrent
                    dans AUCUNE autre chaîne. Test de verrouillage :
                    test_raw_suffixes_are_disjoint_from_the_processing_chain.
                    keys.py : trois clés par fiabilité DÉCROISSANTE — SURE
                    (SerialNumber+ShutterCount, identifiant unique d'une prise,
                    porté par 100 % des NEF), TIME (Model+DateTimeOriginal+
                    SubSecTimeOriginal, repli pour JPEG ré-exportés), NAME (nom
                    de fichier, JAMAIS fiable).
                    ⚠ PIÈGE — LE NOM DE FICHIER N'EST PAS UNE CLÉ. Le compteur
                    Nikon reboucle : 62 stems de NEF en double dans le corpus, et
                    330ND800/DSC_6769.nef ≠ 329ND800/DSC_6769.jpg. D'où
                    pairing.contradicts() : si les DEUX fichiers portent une clé
                    fiable et qu'elles DIFFÈRENT, c'est une preuve positive de
                    prises distinctes — on ne retombe pas sur le nom.
                    ⚠ PIÈGE — exiftool -fast2 SUPPRIME ShutterCount (il s'arrête
                    avant les MakerNotes). Une mesure a annoncé 0 paire au lieu
                    de 1440, SANS ERREUR. Employer -fast (niveau 1).
                    ⚠ PIÈGE — PIL.Image.open() sur un NEF RÉUSSIT et renvoie la
                    vignette 160×120 (il le prend pour un TIFF). Analyser un RAW
                    par PIL donne un faux résultat EN SILENCE. Passer par
                    preview.extract_preview() → exiftool -b -JpgFromRaw, qui rend
                    7360×4912 en 0,12 s (3,7 Mo au lieu de 41).
                    Mesuré sur Dessins/ : 1877 RAW, 1440 paires sûres (76,7 %),
                    391 RAW SANS JPEG (à produire), 46 à confirmer. Scan 33-63 s
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
- digikam_tags.py → moteur PARTAGÉ d'étiquettes hiérarchiques DigiKam, sans Qt
                    (extrait de landmarks.py). read_raw / read_tag_paths /
                    tag_args / write_tags / to_struct, runner injectable.
                    Écrit les 6 champs que DigiKam synchronise (relevés dans
                    ~/.config/digikamrc, section [DMetadata Settings]
                    [readTagsNamespaces]) : XMP-digiKam:TagsList (sép. « / »,
                    source de vérité en lecture), XMP-lr:HierarchicalSubject
                    (« | »), XMP-microsoft:LastKeywordXMP (« / »),
                    XMP-mediapro:CatalogSets (« | »), XMP-dc:Subject et
                    IPTC:Keywords (tagPaths=0 → PLATS, feuille seule),
                    + IPTC:CodedCharacterSet=UTF8.
                    ⚠ PIÈGE MAJEUR — APPARTENANCE PAR BRANCHE. exiftool -TAG=
                    REMPLACE la liste entière : l'écriture est une lecture-
                    fusion-écriture qui reconstruit chaque champ. Le prédicat
                    `owns` décide de ce qui est à nous (remplacé) vs étranger
                    (relu puis réécrit). Un `owns` trop large DÉTRUIT EN SILENCE
                    les étiquettes d'un autre écrivain. Tout nouvel écrivain
                    DOIT utiliser branch_owner("SaBranche") et jamais la racine
                    entière. Verrouillé par test_digikam_tags.py::
                    test_two_owners_never_erase_each_other
- landmarks.py    → LandmarkSet : lecture/écriture de points nommés dans les
                    métadonnées via exiftool (sous-processus, runner injectable
                    pour les tests). Cœur sans Qt, partagé par manual_mouse_points
                    ET auto_face_id_register. Duplique+recentre la gestion EXIF de
                    DataImages (TraiteImages) ; exiftool garde <img>_original.
                    Points en POURCENTAGE (0–100, float) → indépendants de la
                    résolution, aucune conversion lors d'un changement de résolution.
                    STOCKAGE COMPATIBLE DIGIKAM (remplace l'ancien UserComment JSON,
                    encore RELU en repli pour les images antérieures) :
                    • ce qui est filtrable → étiquettes hiérarchiques écrites dans
                      les 6 champs que DigiKam synchronise (liste et séparateurs
                      RELEVÉS dans ~/.config/digikamrc, section [DMetadata Settings]
                      [readTagsNamespaces] : clés « separator » et « tagPaths ») :
                      XMP-digiKam:TagsList (« / », source de vérité en lecture),
                      XMP-lr:HierarchicalSubject (« | »),
                      XMP-microsoft:LastKeywordXMP (« / »),
                      XMP-mediapro:CatalogSets (« | »),
                      XMP-dc:Subject et IPTC:Keywords (tagPaths=0 → PLATS, feuille
                      seule). + IPTC:CodedCharacterSet=UTF8 (IPTC n'est PAS UTF-8
                      par défaut et nos libellés sont accentués — DigiKam écrit la
                      même déclaration). 7e champ XMP-acdsee:Categories écrit par
                      DigiKam en XML imbriqué : volontairement laissé de côté (seul
                      format non « liste de chemins », aucun champ relu n'en dépend).
                      Arborescence sous la racine
                      « media_restorer » (qui remplace l'ancienne clé de schéma) :
                        media_restorer/Repère/<Libellé>
                        media_restorer/Repère ignoré/<Libellé>   (point passé, None)
                        media_restorer/Repérage manuel | automatique  (provenance,
                          LandmarkSet.source = "manual"/"auto" selon l'extension)
                        media_restorer/Repérage complet          (aucun point passé)
                      Feuilles VOLONTAIREMENT auto-suffisantes (« Repérage complet »
                      d'un seul tenant, pas « Repérage/Complet ») : dc:Subject et
                      IPTC:Keywords étant plats, seul le dernier segment y survit
                    • coordonnées → XMP-mwg-rs:RegionInfo (standard MWG lu par
                      DigiKam), aires déjà normalisées 0–1 (nos % /100). Type=Focus
                      et NON Face : DigiKam n'importe que les « Face » dans son
                      arbre Personnes. Les points passés (None) n'ont pas de région
                      et ne survivent que par leur étiquette « Repère ignoré ».
                      Étiquette « Repère » SANS région correspondante → le repère
                      est OMIS à la lecture, jamais rendu « passé » : le rétrograder
                      graverait l'erreur au prochain enregistrement
                    • FUSION NON DESTRUCTIVE OBLIGATOIRE : ces champs sont curés par
                      l'utilisateur et « exiftool -TAG=… » REMPLACE la liste entière.
                      write_to_metadata fait donc lecture→fusion→écriture : les
                      étiquettes hors racine media_restorer et les régions de type
                      ≠ Focus (visages DigiKam) sont relues puis réécrites
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


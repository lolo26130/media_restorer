# CLAUDE.md — Gabarit d'architecture réutilisable

Ce fichier décrit uniquement la **forme** : organisation des fichiers,
conventions d'écriture, patterns d'architecture. Aucun contenu métier
(algorithmes, formats de données traités, domaine applicatif) — à adapter
au domaine du projet qui réutilise ce gabarit.

Complète (ne remplace pas) le CLAUDE.md générique hérité au niveau parent
(gestionnaire de paquets `uv`, structure de projet standard, tests,
Sphinx, injection de dépendances — voir ce fichier pour ces points communs
à tous les projets).

## Principe directeur

Une application desktop PyQt6 = **une fenêtre racine** (sélection d'une
cible + préférences globales) qui lance des **outils indépendants**
enregistrés dans un registre léger, chacun isolé dans son propre
sous-paquet. Ajouter un outil ne modifie jamais la fenêtre racine — une
seule ligne d'import suffit (l'import déclenche l'auto-enregistrement).

## Structure de projet

```
src/mon_projet/
  gui_root.py          → fenêtre racine, point d'entrée unique de l'app ;
                          choisit une cible puis lance un outil du registre
  views/root.ui         → .ui de la fenêtre racine, compilé automatiquement
                          au lancement (jamais de ui_*.py édité à la main)
  extensions/            → registre léger : Extension (protocole), 
                          ExtensionContext, register(), all_extensions()
    __init__.py           → registre + garde-fou "icône manquante" (voir
                          plus bas)
    mon_outil/            → un sous-paquet par outil, isolé
      __init__.py           → classe Extension (name, description, icon,
                          launch()) + appel register() au niveau module
      gui.py                 → fenêtre QMainWindow de cet outil
      views/main.ui          → .ui de cet outil, compilé automatiquement
  engines/                → bibliothèque de calcul, SANS dépendance Qt —
                          réutilisable telle quelle par une CLI, un script,
                          ou n'importe quel outil du registre. Chaque
                          moteur hérite d'une interface commune SAUF
                          quand le contrat ne correspond vraiment pas
                          (entrée/sortie de forme différente) — dans ce
                          cas le départ de convention est documenté dans
                          la docstring du module, pas caché.
  gui_widgets.py           → widgets Qt partagés entre plusieurs outils
                          (ex. fenêtre de résultat générique) — extraits
                          dès qu'un deuxième outil en a besoin, jamais
                          dupliqués
  resources/icons/         → icônes partagées entre la racine et tous les
                          outils (un seul .qrc central, jamais un par
                          outil — sinon un outil ajouté sans mettre à jour
                          le .qrc casse silencieusement au lancement)
  OutilsQt/ (ou équivalent) → utilitaires transverses : compile_ui,
                          compile_qrc, génération de tooltips depuis le
                          code
```

### Garde-fou « icône manquante » lors de l'ajout d'un outil

Centraliser les icônes dans un seul `.qrc` partagé introduit un risque :
un outil ajouté sans y référencer son icône casse au lancement avec une
icône vide, silencieusement. Contre-mesure : un test qui itère
`all_extensions()` et vérifie que chaque `extension.icon` est bien une
ressource présente dans le `.qrc` compilé — échoue à la CI plutôt qu'à
l'exécution.

## Conventions PyQt6

- `@pyqtSlot()` sur **tous** les `on_<widget>_<signal>` → repose sur
  `connectSlotsByName`, jamais de `.connect()` manuel sur ces méthodes
- `QAction` s'importe depuis `PyQt6.QtGui` (pas `QtWidgets`)
- Workers `QThread` : signal de résultat nommé `result_ready` (jamais
  `finished`, qui shadowe `QThread.finished` et provoque des bugs de
  connexion silencieux)
- Imports lourds (calcul numérique, ML, décodage de formats propriétaires)
  **différés** dans les fonctions/méthodes qui en ont besoin, jamais au
  niveau module — le lancement de l'app ne doit pas payer le coût
  d'import de bibliothèques dont l'utilisateur n'utilisera peut-être
  jamais la fonctionnalité associée
- Fichiers `.ui` compilés automatiquement au lancement via une fonction
  `compile_ui()` (jamais de `ui_*.py` généré une fois puis édité à la main
  — le `.ui` reste la seule source de vérité)
- Panneaux de paramètres réglables : privilégier un composant générique
  piloté par déclaration (type `ParameterTree`) plutôt que des widgets
  Qt câblés à la main un par un
- Préférences persistées (apparence, dernier mode choisi, etc.) via
  `QSettings`, relues au prochain lancement — jamais de configuration
  perdue entre deux sessions

## Injection de dépendances pour les tests

Les workers (et plus généralement tout code qui appelle une fonction de
calcul potentiellement lourde/lente) reçoivent cette fonction en
paramètre (`_xxx_factory` ou équivalent) au lieu de l'importer en dur.
En production, on passe l'import réel ; dans les tests, `conftest.py`
injecte un remplaçant rapide. Permet de tester le câblage Qt (activation
des actions, peuplement des combos, gestion d'erreur) sans jamais exécuter
le vrai calcul.

## Piège connu — tests Qt avec de vrais QThread et boucle d'événements imbriquée

Démarrer un vrai `QThread` (`worker.start()`) puis attendre son résultat
via une boucle d'événements imbriquée (`qtbot.waitSignal`,
`qtbot.waitUntil`, ou une `QEventLoop` manuelle) peut provoquer un
segfault reproductible une fois assez de fenêtres/widgets lourds
accumulés dans le même process de test. Dans les tests, appeler
`worker.run()` directement (méthode Python ordinaire, sans jamais
démarrer de vrai thread OS) au lieu de `worker.start()` + attente — mêmes
signaux émis, aucun risque. Toute routine de gestion d'énergie/priorité
process (type `performance_mode()`) doit être neutralisée globalement
pour la suite de tests (fixture autouse dans `conftest.py`), pas au cas
par cas.

## Documentation Sphinx

Structure `docs/source/api/*.rst` qui **reflète l'arborescence du
package** (un fichier par sous-paquet significatif : engines, extensions,
utils…), généré via `automodule`. Éviter les avertissements de référence
croisée ambiguë : ne pas dupliquer un `automodule` sur un sous-module dont
le contenu (`__all__`) est déjà couvert par l'`automodule` du paquet
parent.

## Menu d'aide

La fenêtre racine expose un menu « Aide » qui ouvre la documentation
Sphinx générée dans le navigateur par défaut (`QDesktopServices.openUrl`),
et la reconstruit à la volée (avec timeout, jamais un blocage indéfini de
l'UI) si elle n'a jamais été générée — pour ne jamais présenter de lien
mort. Même logique côté script shell (`open-docs.sh`) que côté GUI, pas de
divergence entre les deux points d'entrée.

"""Registre des extensions Media Restorer.

Une extension est un outil de traitement lancé par la fenêtre racine
(:class:`~media_restorer.gui_root.ImageTreatmentWindow`, « Image Treatment »)
une fois qu'un fichier ou un répertoire a été choisi.  Le mécanisme est
volontairement léger — pas de paquet Python séparé à installer, pas de
découverte via ``importlib.metadata.entry_points`` — car il n'a besoin de
servir qu'un seul programme mono-utilisateur : chaque extension est un petit
module qui s'enregistre lui-même à l'import (même idée que les registres déjà
présents dans ce projet, par exemple
:data:`~media_restorer.engines.ENGINE_PARAMS` ou
:data:`~media_restorer.download_models.MODEL_REGISTRY`).

:class:`~media_restorer.extensions.media_restorer.gui.PhotoRestorationGUI` (Media Restorer) est
elle-même enregistrée comme la première extension, via
:mod:`~media_restorer.extensions.media_restorer` — ce n'est pas un cas
spécial câblé en dur dans la fenêtre racine : c'est ce qui vérifie que
l'abstraction ``Extension`` est effectivement suffisante pour un outil réel,
plutôt que de la laisser purement théorique en attendant une deuxième
extension.

Ajouter une extension
----------------------
1. Créer un sous-paquet isolé sous ``src/media_restorer/extensions/<nom>/``
   (voir :mod:`~media_restorer.extensions.media_restorer` comme modèle : sa
   fenêtre, son ``.ui``, ses modules propres n'en sortent pas).  Son
   ``__init__.py`` définit une classe qui satisfait le protocole
   :class:`Extension` (``name``, ``description``, ``icon``, ``launch``).
2. Si ``icon`` référence un fichier pas déjà dans
   ``resources/icons/media_restorer.qrc`` (partagé, voir plus bas), l'y
   ajouter (fichier ``.png`` + entrée ``<file>``) — sinon
   ``QIcon(icon).isNull()`` sera vrai en silence, sans erreur ni exception :
   le test ``test_every_extension_icon_is_registered_in_the_shared_qrc``
   (``tests/test_extensions.py``) est là pour transformer cet oubli en échec
   de suite plutôt qu'en bouton vide découvert en production.
3. Appeler :func:`register` sur une instance de cette classe, au niveau du
   module (elle s'enregistre donc dès l'import du paquet).
4. Importer ce paquet quelque part avant que la fenêtre racine ne peuple son
   menu — voir :meth:`~media_restorer.gui_root.ImageTreatmentWindow.__init__`,
   qui importe explicitement chaque paquet d'extension connu.  Aucune autre
   modification de la fenêtre racine n'est nécessaire : le menu et la
   toolbar « Extensions » se peuplent dynamiquement depuis
   :func:`all_extensions`.

Les icônes (``resources/icons/``) restent centralisées et partagées entre la
racine et toutes les extensions plutôt que dupliquées par extension : c'est
déjà le cas d'usage réel de
:data:`~media_restorer.extensions.media_restorer.MediaRestorerExtension.icon`,
qui réutilise l'icône de l'action ``actionRestore`` de sa propre fenêtre.

Contrat optionnel — dock « Infos, Exif » de la racine
-----------------------------------------------------
Une fenêtre d'extension *peut* exposer un signal Qt ::

    current_image_changed = pyqtSignal(object)  # Path | None

La fenêtre racine s'y branche automatiquement (par duck-typing, après
``launch``) pour tenir à jour son dock « Infos, Exif » pendant un traitement
qui parcourt plusieurs images : émettre le ``Path`` de l'image en cours affiche
ses métadonnées ; émettre ``None`` (fin de parcours) fait revenir le dock au
résumé de la cible.  Les extensions à image unique n'ont rien à exposer — le
signal est facultatif.  Exemple réel :
:class:`~media_restorer.extensions.media_restorer.gui.PhotoRestorationGUI`,
qui le relaie depuis son worker de traitement par lot.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from PyQt6.QtWidgets import QMainWindow


@dataclass(frozen=True)
class ExtensionContext:
    """Ce que la fenêtre racine transmet à une extension lors de son lancement.

    Paramètres
    ----------
    path : Path
        Fichier ou répertoire choisi par l'utilisateur dans la fenêtre racine.
    recursive : bool
        Mode récursif du parcours de répertoire.  Sans objet si *path*
        désigne un fichier — c'est à l'extension de l'ignorer dans ce cas,
        au même titre qu'elle ignorerait un paramètre non pertinent pour le
        mode dans lequel elle est appelée.
    """

    path: Path
    recursive: bool = False


@runtime_checkable
class Extension(Protocol):
    """Contrat qu'un outil doit satisfaire pour apparaître dans la fenêtre racine.

    Attributs
    ---------
    name : str
        Nom affiché dans le menu « Extensions » et utilisé comme clé du
        registre (deux extensions ne peuvent pas partager le même nom — le
        second appel à :func:`register` écraserait le premier).
    description : str
        Une phrase, affichée en tooltip de l'action de lancement.
    icon : str
        Chemin de ressource Qt (``:/icons/...``) pour l'icône de l'action.
    """

    name: str
    description: str
    icon: str

    def launch(self, context: ExtensionContext) -> QMainWindow:
        """Lance l'outil sur *context* et retourne sa fenêtre principale.

        La fenêtre retournée doit déjà être construite (mais peut ne pas
        être encore affichée — c'est l'appelant, la fenêtre racine, qui gère
        l'affichage et la durée de vie de la référence, exactement comme
        :class:`~media_restorer.extensions.media_restorer.gui.PhotoRestorationGUI` le fait déjà pour
        ses propres fenêtres de résultat).
        """
        ...


_REGISTRY: dict[str, Extension] = {}


def register(extension: Extension) -> None:
    """Enregistre *extension* sous son nom (:attr:`Extension.name`).

    Un second enregistrement sous le même nom remplace silencieusement le
    précédent — pratique pour recharger une extension en développement, mais
    à surveiller si deux extensions distinctes choisissent le même nom par
    inadvertance.
    """
    _REGISTRY[extension.name] = extension


def all_extensions() -> list[Extension]:
    """Toutes les extensions enregistrées, dans leur ordre d'enregistrement.

    Repose sur l'ordre d'insertion des ``dict`` (garanti par le langage
    depuis Python 3.7) : la première extension enregistrée est donc toujours
    la première de la liste — c'est ce qui permet à la fenêtre racine de
    proposer « la première extension enregistrée » comme action par défaut
    sans configuration supplémentaire.
    """
    return list(_REGISTRY.values())

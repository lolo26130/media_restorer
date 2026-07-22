"""Tests du registre d'extensions (media_restorer.extensions)."""
import pytest
from PyQt6.QtWidgets import QMainWindow

from media_restorer.extensions import ExtensionContext, all_extensions, register


class _FakeExtension:
    """Extension minimale pour tester le registre sans dépendre de Media Restorer."""

    def __init__(self, name="Fake", description="Une extension factice", icon=":/icons/gear--plus.png"):
        self.name = name
        self.description = description
        self.icon = icon
        self.launched_with: ExtensionContext | None = None

    def launch(self, context: ExtensionContext) -> QMainWindow:
        self.launched_with = context
        return QMainWindow()


@pytest.fixture
def register_fake():
    """``register()`` une extension factice, puis la retire à la fin du test.

    Ne touche qu'aux noms explicitement enregistrés par ce biais — jamais au
    reste du registre.  C'est important : ``media_restorer_ext`` enregistre
    « Media Restorer » de façon *permanente* dès son import (c'est le
    comportement voulu, voir
    ``test_media_restorer_extension_is_registered_on_import`` plus bas), et
    un nettoyage qui reviendrait à un instantané complet d'avant-test la
    désinscrirait pour le reste de la session sans que rien ne la
    réenregistre (l'import Python n'est pas rejoué).
    """
    import media_restorer.extensions as ext_mod

    names: list[str] = []

    def _register(extension):
        register(extension)
        names.append(extension.name)
        return extension

    yield _register
    for name in names:
        ext_mod._REGISTRY.pop(name, None)


def test_register_then_all_extensions_returns_it(register_fake):
    ext = register_fake(_FakeExtension())

    assert ext in all_extensions()


def test_all_extensions_preserves_registration_order(register_fake):
    """L'ordre d'enregistrement est celui que la racine propose par défaut."""
    register_fake(_FakeExtension("Premier"))
    register_fake(_FakeExtension("Second"))

    names = [e.name for e in all_extensions()]
    assert names.index("Premier") < names.index("Second")


def test_registering_the_same_name_twice_replaces_the_previous_extension(register_fake):
    register_fake(_FakeExtension("Même nom", description="v1"))
    register_fake(_FakeExtension("Même nom", description="v2"))

    matches = [e for e in all_extensions() if e.name == "Même nom"]
    assert len(matches) == 1
    assert matches[0].description == "v2"


def test_extension_context_defaults_recursive_to_false():
    ctx = ExtensionContext(path=__import__("pathlib").Path("/tmp/x"))

    assert ctx.recursive is False


def test_media_restorer_extension_is_registered_on_import():
    """Importer le module d'extension suffit à l'enregistrer (voir gui_root.py)."""
    import media_restorer.extensions.media_restorer_ext as mre

    names = [e.name for e in all_extensions()]
    assert mre.MediaRestorerExtension.name in names


def test_media_restorer_extension_launch_forwards_context(tmp_path, qtbot):
    from media_restorer.extensions.media_restorer_ext import MediaRestorerExtension
    from media_restorer.gui import PhotoRestorationGUI

    context = ExtensionContext(path=tmp_path, recursive=True)
    window = MediaRestorerExtension().launch(context)
    qtbot.addWidget(window)

    assert isinstance(window, PhotoRestorationGUI)
    assert window._batch_dir == tmp_path
    assert window._batch_recursive is True

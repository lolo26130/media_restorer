"""Tests pour OutilsQt.Utils_Qt.compile_ui et compile_qrc."""
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from OutilsQt.Utils_Qt import compile_ui, compile_qrc


# ---------------------------------------------------------------------------
# compile_ui
# ---------------------------------------------------------------------------

def test_compile_ui_creates_when_py_missing(tmp_path):
    """compile_ui appelle uic.compileUi si ui_py n'existe pas."""
    ui_src = tmp_path / "main.ui"
    ui_py  = tmp_path / "ui_main.py"
    ui_src.write_text("<ui></ui>")

    with patch("OutilsQt.Utils_Qt._uic.compileUi") as mock_compile:
        compile_ui(ui_src, ui_py)

    mock_compile.assert_called_once_with(str(ui_src), pytest.approx(mock_compile.call_args[0][1]))


def test_compile_ui_skips_when_py_is_newer(tmp_path):
    """compile_ui ne recompile pas si ui_py est plus récent que ui_src."""
    ui_src = tmp_path / "main.ui"
    ui_py  = tmp_path / "ui_main.py"
    ui_src.write_text("<ui></ui>")
    ui_py.write_text("# generated")
    time.sleep(0.05)
    ui_py.touch()  # ui_py plus récent

    with patch("OutilsQt.Utils_Qt._uic.compileUi") as mock_compile:
        compile_ui(ui_src, ui_py)

    mock_compile.assert_not_called()


def test_compile_ui_recompiles_when_ui_is_newer(tmp_path):
    """compile_ui recompile si ui_src est plus récent que ui_py."""
    ui_py  = tmp_path / "ui_main.py"
    ui_py.write_text("# old")
    time.sleep(0.05)
    ui_src = tmp_path / "main.ui"
    ui_src.write_text("<ui></ui>")  # ui_src créé après ui_py → plus récent

    with patch("OutilsQt.Utils_Qt._uic.compileUi") as mock_compile:
        compile_ui(ui_src, ui_py)

    mock_compile.assert_called_once()


# ---------------------------------------------------------------------------
# compile_qrc
# ---------------------------------------------------------------------------

def _fake_rcc_ok(qrc_py: Path):
    """Retourne un side_effect simulant rcc qui écrit le fichier."""
    def _side_effect(cmd, **kw):
        qrc_py.write_text("from PySide6 import QtCore\n")
        m = MagicMock()
        m.returncode = 0
        return m
    return _side_effect


def test_compile_qrc_patches_pyside6_to_pyqt6(tmp_path):
    """compile_qrc remplace 'from PySide6 import QtCore' par PyQt6."""
    qrc_src = tmp_path / "icons.qrc"
    qrc_py  = tmp_path / "icons_rc.py"
    qrc_src.write_text('<RCC><qresource/></RCC>')

    with patch("OutilsQt.Utils_Qt._subprocess.run", side_effect=_fake_rcc_ok(qrc_py)):
        compile_qrc(qrc_src, qrc_py)

    text = qrc_py.read_text()
    assert "from PyQt6 import QtCore" in text
    assert "PySide6" not in text


def test_compile_qrc_raises_on_rcc_failure(tmp_path):
    """compile_qrc lève RuntimeError si rcc retourne un code non-zéro."""
    qrc_src = tmp_path / "icons.qrc"
    qrc_py  = tmp_path / "icons_rc.py"
    qrc_src.write_text('<RCC/>')

    mock_result = MagicMock(returncode=1, stderr="rcc: bad input")
    with patch("OutilsQt.Utils_Qt._subprocess.run", return_value=mock_result):
        with pytest.raises(RuntimeError, match="rcc failed"):
            compile_qrc(qrc_src, qrc_py)


def test_compile_qrc_skips_when_py_is_newer(tmp_path):
    """compile_qrc ne relance pas rcc si qrc_py est plus récent que qrc_src."""
    qrc_src = tmp_path / "icons.qrc"
    qrc_py  = tmp_path / "icons_rc.py"
    qrc_src.write_text('<RCC/>')
    qrc_py.write_text("# already compiled")
    time.sleep(0.05)
    qrc_py.touch()  # plus récent

    with patch("OutilsQt.Utils_Qt._subprocess.run") as mock_run:
        compile_qrc(qrc_src, qrc_py)

    mock_run.assert_not_called()


def test_compile_qrc_creates_when_py_missing(tmp_path):
    """compile_qrc appelle rcc si qrc_py n'existe pas."""
    qrc_src = tmp_path / "icons.qrc"
    qrc_py  = tmp_path / "icons_rc.py"
    qrc_src.write_text('<RCC/>')

    with patch("OutilsQt.Utils_Qt._subprocess.run", side_effect=_fake_rcc_ok(qrc_py)):
        compile_qrc(qrc_src, qrc_py)

    assert qrc_py.exists()


def test_compile_qrc_custom_rcc_path(tmp_path):
    """compile_qrc utilise le chemin rcc fourni en paramètre."""
    qrc_src = tmp_path / "icons.qrc"
    qrc_py  = tmp_path / "icons_rc.py"
    qrc_src.write_text('<RCC/>')

    calls = []

    def _capture(cmd, **kw):
        calls.append(cmd)
        qrc_py.write_text("from PySide6 import QtCore\n")
        return MagicMock(returncode=0)

    with patch("OutilsQt.Utils_Qt._subprocess.run", side_effect=_capture):
        compile_qrc(qrc_src, qrc_py, rcc="/custom/path/rcc")

    assert calls[0][0] == "/custom/path/rcc"

"""Sphinx configuration for media-restorer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

project = "media-restorer"
author = "Laurent"
release = "0.1.0"
copyright = f"2026, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

html_theme = "sphinx_rtd_theme"
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False
autosummary_generate = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy":  ("https://numpy.org/doc/stable", None),
    "torch":  ("https://pytorch.org/docs/stable", None),
}

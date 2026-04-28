# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "abstracts-explorer"
copyright = "2025, Abstracts Explorer Contributors"
author = "Abstracts Explorer Contributors"

# Dynamically read the version from the installed package so the HTML title
# always shows the real release version instead of a hard-coded placeholder.
try:
    from abstracts_explorer._version import __version__ as release
except ImportError:
    release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Auto-generate documentation from docstrings
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "sphinx.ext.viewcode",  # Add links to highlighted source code
    "sphinx.ext.intersphinx",  # Link to other project's documentation
    "sphinx.ext.coverage",  # Check documentation coverage
    "myst_parser",  # Markdown support
    "sphinx_autodoc_typehints",  # Type hints support
    "sphinxcontrib.mermaid",  # Mermaid diagram support
]

# Napoleon settings for NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Mock imports for optional dependencies
autodoc_mock_imports = [
    "flask",
    "flask_cors",
    "waitress",
]

# Type hints settings
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# MyST Parser settings
myst_enable_extensions = [
    "colon_fence",  # ::: fences
    "deflist",  # Definition lists
    "tasklist",  # Task lists
    "linkify",  # Auto-detect URLs
]
myst_heading_anchors = 4  # Auto-generate anchors for h1-h4 headings

# Source suffix
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = []

# Theme options
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
}

# Version selector for multi-version GitHub Pages deployment.
# DOCS_VERSION is set by CI: 'stable' for the main branch, 'dev' for the develop branch.
# DOCS_BASE_URL can be overridden for custom domains; defaults to the GitHub Pages URL.
_docs_version = os.environ.get("DOCS_VERSION", "stable")
_docs_base_url = os.environ.get("DOCS_BASE_URL", "https://thawn.github.io/abstracts-explorer")

html_context = {
    "display_version": True,
    "current_version": _docs_version,
    "versions": [
        ("stable", f"{_docs_base_url}/"),
        ("dev", f"{_docs_base_url}/dev/"),
    ],
}

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# Configure intersphinx to not fail on network errors
intersphinx_timeout = 5  # Timeout after 5 seconds instead of hanging
suppress_warnings = ["app.add_node"]  # Suppress node warnings

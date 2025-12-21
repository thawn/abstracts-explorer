"""
Web UI module for NeurIPS Abstracts.

This module provides a Flask-based web interface for exploring
the NeurIPS abstracts database.

Note: This module requires Flask and related dependencies.
Install with: pip install neurips-abstracts[web]
"""


def _initialize():
    """Lazy initialization of web UI components."""
    import importlib

    app_module = importlib.import_module(".app", package=__name__)

    # Set as module attributes to make them importable
    import sys

    current_module = sys.modules[__name__]
    current_module.app = app_module.app
    current_module.run_server = app_module.run_server


def __getattr__(name):
    """
    Lazy-load web UI components to avoid importing Flask unless needed.

    This allows the CLI and other parts of the package to work without
    the web dependencies installed.
    """
    if name in ("app", "run_server"):
        _initialize()
        return globals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["app", "run_server"]

"""
Abstracts Explorer Package
===========================

Abstracts Explorer - A Python package for downloading conference data and loading it into a SQLite database.

Main Components
---------------
- database: Load JSON data into SQLite database
- embeddings: Generate and store text embeddings for papers
- plugins: Extensible plugin system for different data sources

Plugin System
-------------
The package includes a plugin system for downloading papers from different sources:

- **neurips**: Official NeurIPS conference data (1987 onward)
- **ml4ps**: ML4PS (Machine Learning for Physical Sciences) workshop

Example usage with plugins::

    from abstracts_explorer.plugins import get_plugin, list_plugins

    # List available plugins
    plugins = list_plugins()
    for plugin in plugins:
        print(f"{plugin['name']}: {plugin['description']}")

    # Use a specific plugin
    plugin = get_plugin('neurips')
    data = plugin.download(year=2025, output_path='neurips_2025.json')

    # Use ML4PS plugin
    ml4ps_plugin = get_plugin('ml4ps')
    data = ml4ps_plugin.download(year=2025)

Creating Custom Plugins
------------------------
Subclass ``DownloaderPlugin`` and return
:class:`~abstracts_explorer.plugin.LightweightPaper` objects::

    from abstracts_explorer.plugins import (
        DownloaderPlugin,
        LightweightPaper,
        register_plugin,
    )

    class MyWorkshopPlugin(DownloaderPlugin):
        plugin_name = "myworkshop"
        plugin_description = "My workshop downloader"
        _start_year = 2025

        def get_url(self, year):
            return f"https://example.com/{year}"

        def _download_uncached(self, year, output_path, options):
            return [
                LightweightPaper(
                    title="Paper Title",
                    authors=["John Doe", "Jane Smith"],
                    abstract="Paper abstract...",
                    session="Morning Session",
                    poster_position="A1",
                    year=year,
                    conference="MyWorkshop",
                )
            ]

        def get_metadata(self):
            return {
                "name": self.plugin_name,
                "description": self.plugin_description,
                "supported_years": self.supported_years,
            }

    register_plugin(MyWorkshopPlugin())

The base ``download()`` template handles year resolution, validation, caching,
and saving. HTTP-based plugins can subclass ``HTTPDownloaderPlugin`` for shared
retry and rate-limit handling. See ``docs/plugins.md`` for details.
"""

import logging


def _configure_package_logging() -> None:
    """
    Configure package-level logging at import time.

    Sets the log level for the abstracts_explorer package based on the
    LOG_LEVEL environment variable or .env file. Defaults to WARNING to
    prevent plugin import messages from appearing during normal startup.

    This runs before plugin imports to ensure INFO-level plugin registration
    messages are suppressed unless LOG_LEVEL is explicitly configured.
    """
    from abstracts_explorer.config import get_config

    config = get_config()
    level_name = config.log_level or "WARNING"
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    level = level_map.get(level_name, logging.WARNING)
    logging.getLogger("abstracts_explorer").setLevel(level)


_configure_package_logging()

from typing import TYPE_CHECKING, Any  # noqa: E402

if TYPE_CHECKING:
    from abstracts_explorer.config import Config, get_config
    from abstracts_explorer.database import DatabaseManager
    from abstracts_explorer.embeddings import EmbeddingsManager
    from abstracts_explorer.clustering import ClusteringManager, ClusteringError, perform_clustering
    from abstracts_explorer.rag import RAGChat
    from abstracts_explorer.registry import RegistryClient, RegistryError
    from abstracts_explorer.plugins import (
        DownloaderPlugin,
        PluginRegistry,
        register_plugin,
        get_plugin,
        list_plugins,
        list_plugin_names,
    )

_LAZY_IMPORTS = {
    "Config": "abstracts_explorer.config",
    "get_config": "abstracts_explorer.config",
    "DatabaseManager": "abstracts_explorer.database",
    "EmbeddingsManager": "abstracts_explorer.embeddings",
    "ClusteringManager": "abstracts_explorer.clustering",
    "ClusteringError": "abstracts_explorer.clustering",
    "perform_clustering": "abstracts_explorer.clustering",
    "RAGChat": "abstracts_explorer.rag",
    "RegistryClient": "abstracts_explorer.registry",
    "RegistryError": "abstracts_explorer.registry",
    "DownloaderPlugin": "abstracts_explorer.plugins",
    "PluginRegistry": "abstracts_explorer.plugins",
    "register_plugin": "abstracts_explorer.plugins",
    "get_plugin": "abstracts_explorer.plugins",
    "list_plugins": "abstracts_explorer.plugins",
    "list_plugin_names": "abstracts_explorer.plugins",
}


def __getattr__(name: str) -> Any:
    """Lazily import heavy public API symbols on first access (PEP 562)."""
    module_path = _LAZY_IMPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))


try:
    from abstracts_explorer._version import __version__
except ImportError:
    __version__ = "0.1.0"
__all__ = [
    "Config",
    "get_config",
    "DatabaseManager",
    "EmbeddingsManager",
    "ClusteringManager",
    "ClusteringError",
    "perform_clustering",
    "RAGChat",
    "RegistryClient",
    "RegistryError",
    "DownloaderPlugin",
    "PluginRegistry",
    "register_plugin",
    "get_plugin",
    "list_plugins",
    "list_plugin_names",
]

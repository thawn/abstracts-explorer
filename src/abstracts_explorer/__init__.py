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

- **neurips**: Official NeurIPS conference data (2013-2025)
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
You can create custom downloader plugins by subclassing `DownloaderPlugin` for full control,
or `LightweightDownloaderPlugin` for a simpler interface::

    # Full schema plugin
    from abstracts_explorer.plugins import DownloaderPlugin, register_plugin

    class MyCustomPlugin(DownloaderPlugin):
        plugin_name = "mycustom"
        plugin_description = "My custom data source"
        supported_years = [2024, 2025]

        def download(self, year=None, output_path=None, force_download=False, **kwargs):
            # Implement your download logic
            return {'count': 0, 'next': None, 'previous': None, 'results': []}

        def get_metadata(self):
            return {
                'name': self.plugin_name,
                'description': self.plugin_description,
                'supported_years': self.supported_years
            }

    # Register your plugin
    register_plugin(MyCustomPlugin())

Lightweight Plugin API
-----------------------
For simpler use cases, use `LightweightDownloaderPlugin` which only requires essential fields::

    from abstracts_explorer.plugins import (
        LightweightDownloaderPlugin,
        LightweightPaper,
        register_plugin
    )

    class MyLightweightPlugin(LightweightDownloaderPlugin):
        plugin_name = "myworkshop"
        plugin_description = "My workshop downloader"
        supported_years = [2025]

        def download(self, year=None, output_path=None, force_download=False, **kwargs):
            # Return papers in lightweight format as LightweightPaper objects
            papers_data = [
                {
                    'title': 'Paper Title',
                    'authors': ['John Doe', 'Jane Smith'],
                    'abstract': 'Paper abstract...',
                    'session': 'Morning Session',
                    'poster_position': 'A1',
                    'year': year,
                    'conference': 'MyWorkshop',
                    'paper_pdf_url': 'https://example.com/paper.pdf',  # optional
                    'url': 'https://example.com/paper',  # optional
                    'keywords': ['ML', 'Physics'],  # optional
                }
            ]

            # Return as LightweightPaper objects
            return [LightweightPaper(**paper) for paper in papers_data]

        def get_metadata(self):
            return {
                'name': self.plugin_name,
                'description': self.plugin_description,
                'supported_years': self.supported_years
            }

    register_plugin(MyLightweightPlugin())
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

from abstracts_explorer.config import Config, get_config  # noqa: E402
from abstracts_explorer.database import DatabaseManager  # noqa: E402
from abstracts_explorer.embeddings import EmbeddingsManager  # noqa: E402
from abstracts_explorer.clustering import ClusteringManager, ClusteringError, perform_clustering  # noqa: E402
from abstracts_explorer.rag import RAGChat  # noqa: E402
from abstracts_explorer.registry import RegistryClient, RegistryError  # noqa: E402
from abstracts_explorer.plugins import (  # noqa: E402
    DownloaderPlugin,
    LightweightDownloaderPlugin,
    PluginRegistry,
    register_plugin,
    get_plugin,
    list_plugins,
    list_plugin_names,
)

# Import plugins to auto-register them
import abstracts_explorer.plugins  # noqa: E402, F401

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
    "LightweightDownloaderPlugin",
    "PluginRegistry",
    "register_plugin",
    "get_plugin",
    "list_plugins",
    "list_plugin_names",
]

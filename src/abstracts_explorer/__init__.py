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

from .config import Config, get_config
from .database import DatabaseManager
from .embeddings import EmbeddingsManager
from .clustering import ClusteringManager, ClusteringError, perform_clustering
from .rag import RAGChat
from .plugins import (
    DownloaderPlugin,
    LightweightDownloaderPlugin,
    PluginRegistry,
    register_plugin,
    get_plugin,
    list_plugins,
    list_plugin_names,
)

# Import plugins to auto-register them
from . import plugins  # noqa: F401

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
    "DownloaderPlugin",
    "LightweightDownloaderPlugin",
    "PluginRegistry",
    "register_plugin",
    "get_plugin",
    "list_plugins",
    "list_plugin_names",
]

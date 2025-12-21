"""
NeurIPS Abstracts Package
==========================

A Python package for downloading NeurIPS conference data and loading it into a SQLite database.

Main Components
---------------
- downloader: Download JSON data from configurable URLs
- database: Load JSON data into SQLite database
- embeddings: Generate and store text embeddings for papers
- plugins: Extensible plugin system for different data sources

Plugin System
-------------
The package includes a plugin system for downloading papers from different sources:

- **neurips**: Official NeurIPS conference data (2013-2025)
- **ml4ps**: ML4PS (Machine Learning for Physical Sciences) workshop

Example usage with plugins::

    from neurips_abstracts.plugins import get_plugin, list_plugins

    # List available plugins
    plugins = list_plugins()
    for plugin in plugins:
        print(f"{plugin['name']}: {plugin['description']}")

    # Use a specific plugin
    plugin = get_plugin('neurips')
    data = plugin.download(year=2025, output_path='neurips_2025.json')

    # Use ML4PS plugin
    ml4ps_plugin = get_plugin('ml4ps')
    data = ml4ps_plugin.download(year=2025, fetch_abstracts=True)

Creating Custom Plugins
------------------------
You can create custom downloader plugins by subclassing `DownloaderPlugin` for full control,
or `LightweightDownloaderPlugin` for a simpler interface::

    # Full schema plugin
    from neurips_abstracts.plugins import DownloaderPlugin, register_plugin

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

    from neurips_abstracts.plugins import (
        LightweightDownloaderPlugin,
        convert_lightweight_to_neurips_schema,
        register_plugin
    )

    class MyLightweightPlugin(LightweightDownloaderPlugin):
        plugin_name = "myworkshop"
        plugin_description = "My workshop downloader"
        supported_years = [2025]

        def download(self, year=None, output_path=None, force_download=False, **kwargs):
            # Return papers in lightweight format
            papers = [
                {
                    'id': 1,
                    'title': 'Paper Title',
                    'authors': ['John Doe', 'Jane Smith'],
                    'abstract': 'Paper abstract...',
                    'session': 'Morning Session',
                    'poster_position': 'A1',
                    'paper_pdf_url': 'https://example.com/paper.pdf',  # optional
                    'url': 'https://example.com/paper',  # optional
                    'keywords': ['ML', 'Physics'],  # optional
                }
            ]

            # Convert to full schema
            return convert_lightweight_to_neurips_schema(
                papers,
                session_default='My Workshop 2025',
                event_type='Workshop Poster',
                source_url='https://myworkshop.com'
            )

        def get_metadata(self):
            return {
                'name': self.plugin_name,
                'description': self.plugin_description,
                'supported_years': self.supported_years
            }

    register_plugin(MyLightweightPlugin())
"""

from .config import Config, get_config
from .downloader import download_json, download_neurips_data
from .database import DatabaseManager
from .embeddings import EmbeddingsManager
from .rag import RAGChat
from .plugins import (
    DownloaderPlugin,
    LightweightDownloaderPlugin,
    PluginRegistry,
    register_plugin,
    get_plugin,
    list_plugins,
    list_plugin_names,
    convert_lightweight_to_neurips_schema,
)

# Import plugins to auto-register them
from . import plugins

__version__ = "0.1.0"
__all__ = [
    "Config",
    "get_config",
    "download_json",
    "download_neurips_data",
    "DatabaseManager",
    "EmbeddingsManager",
    "RAGChat",
    "DownloaderPlugin",
    "LightweightDownloaderPlugin",
    "PluginRegistry",
    "register_plugin",
    "get_plugin",
    "list_plugins",
    "list_plugin_names",
    "convert_lightweight_to_neurips_schema",
]

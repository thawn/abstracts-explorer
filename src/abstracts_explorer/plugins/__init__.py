"""
Download Plugins
================

This module provides downloadable plugin implementations for different data sources.

Available plugins:
- CHIDownloaderPlugin: ACM CHI conference data (from SIGCHI program JSON)
- HAICONDownloaderPlugin: HAICON (Helmholtz AI Conference) data via ConfTool
- ICLRDownloaderPlugin: Official ICLR conference data
- ICMLDownloaderPlugin: Official ICML conference data
- IGARSSDownloaderPlugin: IGARSS conference data (via IEEE Xplore)
- IEEEVISDownloaderPlugin: Official IEEE VIS conference data
- ML4PSDownloaderPlugin: ML4PS workshop data
- NeurIPSDownloaderPlugin: Official NeurIPS conference data

The plugin framework is provided by the abstracts_explorer.plugin module.
"""

# Re-export plugin framework from abstracts_explorer.plugin
from abstracts_explorer.plugin import (
    # Plugin base classes
    DownloaderPlugin,
    LightweightDownloaderPlugin,
    # Registry
    PluginRegistry,
    register_plugin,
    get_plugin,
    get_all_plugins,
    list_plugins,
    list_plugin_names,
    # Conversion utilities
    convert_to_lightweight_schema,
    # Pydantic models
    LightweightPaper,
    # Validation functions
    sanitize_author_names,
    validate_lightweight_paper,
    validate_lightweight_papers,
)

# Import actual plugin implementations
from abstracts_explorer.plugins.chi_downloader import CHIDownloaderPlugin
from abstracts_explorer.plugins.haicon_downloader import HAICONDownloaderPlugin
from abstracts_explorer.plugins.iclr_downloader import ICLRDownloaderPlugin
from abstracts_explorer.plugins.icml_downloader import ICMLDownloaderPlugin
from abstracts_explorer.plugins.igarss_downloader import IGARSSDownloaderPlugin
from abstracts_explorer.plugins.ieeevis_downloader import IEEEVISDownloaderPlugin
from abstracts_explorer.plugins.ml4ps_downloader import ML4PSDownloaderPlugin
from abstracts_explorer.plugins.neurips_downloader import NeurIPSDownloaderPlugin

__all__ = [
    # Plugin base classes
    "DownloaderPlugin",
    "LightweightDownloaderPlugin",
    # Registry
    "PluginRegistry",
    "register_plugin",
    "get_plugin",
    "get_all_plugins",
    "list_plugins",
    "list_plugin_names",
    # Conversion utilities
    "convert_to_lightweight_schema",
    # Pydantic models
    "LightweightPaper",
    # Validation functions
    "sanitize_author_names",
    "validate_lightweight_paper",
    "validate_lightweight_papers",
    # Plugin implementations
    "CHIDownloaderPlugin",
    "HAICONDownloaderPlugin",
    "ICLRDownloaderPlugin",
    "ICMLDownloaderPlugin",
    "IGARSSDownloaderPlugin",
    "IEEEVISDownloaderPlugin",
    "ML4PSDownloaderPlugin",
    "NeurIPSDownloaderPlugin",
]

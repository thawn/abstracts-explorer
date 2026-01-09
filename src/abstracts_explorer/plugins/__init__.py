"""
Download Plugins
================

This module provides downloadable plugin implementations for different data sources.

Available plugins:
- ICLRDownloaderPlugin: Official ICLR conference data
- ICMLDownloaderPlugin: Official ICML conference data
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
from .iclr_downloader import ICLRDownloaderPlugin
from .icml_downloader import ICMLDownloaderPlugin
from .ml4ps_downloader import ML4PSDownloaderPlugin
from .neurips_downloader import NeurIPSDownloaderPlugin

__all__ = [
    # Plugin base classes
    "DownloaderPlugin",
    "LightweightDownloaderPlugin",
    # Registry
    "PluginRegistry",
    "register_plugin",
    "get_plugin",
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
    "ICLRDownloaderPlugin",
    "ICMLDownloaderPlugin",
    "ML4PSDownloaderPlugin",
    "NeurIPSDownloaderPlugin",
]

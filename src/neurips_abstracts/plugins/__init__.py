"""
Download Plugins
================

This module provides downloadable plugin implementations for different data sources.

Available plugins:
- ML4PSDownloaderPlugin: ML4PS workshop data
- NeurIPSDownloaderPlugin: Official NeurIPS conference data

The plugin framework is provided by the neurips_abstracts.plugin module.
"""

# Re-export plugin framework from neurips_abstracts.plugin
from neurips_abstracts.plugin import (
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
    convert_lightweight_to_neurips_schema,
    # Pydantic models
    LightweightAuthor,
    LightweightPaper,
    EventMediaModel,
    AuthorModel,
    PaperModel,
    # Validation functions
    validate_lightweight_paper,
    validate_lightweight_papers,
    validate_paper,
    validate_papers,
)

# Import actual plugin implementations
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
    "convert_lightweight_to_neurips_schema",
    # Pydantic models
    "LightweightAuthor",
    "LightweightPaper",
    "EventMediaModel",
    "AuthorModel",
    "PaperModel",
    # Validation functions
    "validate_lightweight_paper",
    "validate_lightweight_papers",
    "validate_paper",
    "validate_papers",
    # Plugin implementations
    "ML4PSDownloaderPlugin",
    "NeurIPSDownloaderPlugin",
]

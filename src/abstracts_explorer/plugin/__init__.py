"""
Plugin Framework
================

Extensible downloader plugin framework for abstracts-explorer.

The framework consists of:

- :mod:`~abstracts_explorer.plugin.base` — ``DownloaderPlugin`` and
  ``LightweightDownloaderPlugin`` base classes
- :mod:`~abstracts_explorer.plugin.http` — shared HTTP retry/cooldown handling
  and ``HTTPDownloaderPlugin``
- :mod:`~abstracts_explorer.plugin.models` — ``LightweightPaper`` and
  validation/serialization helpers
- :mod:`~abstracts_explorer.plugin.converters` —
  ``convert_to_lightweight_schema``
- :mod:`~abstracts_explorer.plugin.registry` — plugin registry and lookup
- :mod:`~abstracts_explorer.plugin.persistence` — cache/sidecar path helpers

All public names are re-exported here so ``from abstracts_explorer.plugin
import ...`` continues to work.
"""

from abstracts_explorer.plugin.base import DownloadOptions, DownloaderPlugin, LightweightDownloaderPlugin
from abstracts_explorer.plugin.converters import convert_to_lightweight_schema
from abstracts_explorer.plugin.http import HTTPDownloaderPlugin, fetch_with_retries
from abstracts_explorer.plugin.models import (
    LightweightPaper,
    deserialize_authors_from_string,
    deserialize_keywords_from_string,
    sanitize_author_names,
    serialize_authors_to_string,
    serialize_keywords_to_string,
    validate_lightweight_paper,
    validate_lightweight_papers,
)
from abstracts_explorer.plugin.persistence import get_failed_downloads_path
from abstracts_explorer.plugin.registry import (
    PluginRegistry,
    get_all_plugins,
    get_available_filters,
    get_plugin,
    list_plugin_names,
    list_plugins,
    register_plugin,
    resolve_conference_from_url,
)
from abstracts_explorer.rate_limiter import RateLimitCooldown

__all__ = [
    "DownloaderPlugin",
    "LightweightDownloaderPlugin",
    "HTTPDownloaderPlugin",
    "DownloadOptions",
    "RateLimitCooldown",
    "PluginRegistry",
    "register_plugin",
    "get_plugin",
    "get_all_plugins",
    "list_plugins",
    "list_plugin_names",
    "get_available_filters",
    "resolve_conference_from_url",
    "convert_to_lightweight_schema",
    "fetch_with_retries",
    "get_failed_downloads_path",
    "LightweightPaper",
    "sanitize_author_names",
    "serialize_authors_to_string",
    "deserialize_authors_from_string",
    "serialize_keywords_to_string",
    "deserialize_keywords_from_string",
    "validate_lightweight_paper",
    "validate_lightweight_papers",
]

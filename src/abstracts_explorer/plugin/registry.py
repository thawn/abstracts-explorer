"""
Plugin registry.

Manages registration and lookup of downloader plugins.
"""

import logging
from typing import Any, Dict, List, Optional

from abstracts_explorer.plugin.base import DownloaderPlugin

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Registry for managing downloader plugins."""

    def __init__(self) -> None:
        self._plugins: Dict[str, DownloaderPlugin] = {}

    def register(self, plugin: DownloaderPlugin) -> None:
        """
        Register a new plugin.

        Parameters
        ----------
        plugin : DownloaderPlugin
            Plugin instance to register
        """
        if not isinstance(plugin, DownloaderPlugin):
            raise TypeError(f"Plugin must be an instance of DownloaderPlugin, got {type(plugin)}")

        self._plugins[plugin.plugin_name] = plugin
        logger.info(f"Registered plugin: {plugin.plugin_name}")

    def get(self, plugin_name: str) -> Optional[DownloaderPlugin]:
        """
        Get a plugin by name.

        Parameters
        ----------
        plugin_name : str
            Name of plugin to retrieve

        Returns
        -------
        DownloaderPlugin or None
            Plugin instance or None if not found
        """
        return self._plugins.get(plugin_name)

    def list_plugins(self) -> List[Dict[str, Any]]:
        """
        List all registered plugins with their metadata.

        Returns
        -------
        list
            List of plugin metadata dictionaries
        """
        return [plugin.get_metadata() for plugin in self._plugins.values()]

    def list_plugin_names(self) -> List[str]:
        """
        List names of all registered plugins.

        Returns
        -------
        list
            List of plugin names
        """
        return list(self._plugins.keys())


_registry = PluginRegistry()


def register_plugin(plugin: DownloaderPlugin) -> None:
    """
    Register a plugin with the global registry.

    Parameters
    ----------
    plugin : DownloaderPlugin
        Plugin instance to register
    """
    _registry.register(plugin)


def get_plugin(plugin_name: str) -> Optional[DownloaderPlugin]:
    """
    Get a plugin from the global registry.

    Parameters
    ----------
    plugin_name : str
        Name of plugin to retrieve

    Returns
    -------
    DownloaderPlugin or None
        Plugin instance or None if not found
    """
    return _registry.get(plugin_name)


def list_plugins() -> List[Dict[str, Any]]:
    """
    List all registered plugins.

    Returns
    -------
    list
        List of plugin metadata dictionaries
    """
    return _registry.list_plugins()


def list_plugin_names() -> List[str]:
    """
    List names of all registered plugins.

    Returns
    -------
    list
        List of plugin names
    """
    return _registry.list_plugin_names()


def get_all_plugins() -> List[DownloaderPlugin]:
    """
    Get all registered plugin instances.

    Returns
    -------
    list of DownloaderPlugin
        List of all registered plugin instances
    """
    plugins = []
    for name in _registry.list_plugin_names():
        plugin = _registry.get(name)
        if plugin is not None:
            plugins.append(plugin)
    return plugins


def get_available_filters() -> Dict[str, Any]:
    """
    Get available conferences and years from registered plugins.

    Returns a mapping of conferences to their supported years based on
    the registered downloader plugins.

    Returns
    -------
    dict
        Dictionary with:
        - conferences: list of conference names
        - years: list of all unique years across all plugins
        - conference_years: dict mapping conference names to their supported years

    Examples
    --------
    >>> filters = get_available_filters()
    >>> print(filters['conferences'])  # ['NeurIPS', 'ICLR', ...]
    >>> print(filters['years'])  # [2025, 2024, 2023, ...]
    >>> print(filters['conference_years'])  # {'NeurIPS': [2025, 2024], ...}
    """
    plugins = list_plugins()

    conference_years: Dict[str, List[int]] = {}
    all_years: set = set()

    for plugin_info in plugins:
        conference_name = plugin_info.get("conference_name")
        supported_years = plugin_info.get("supported_years", [])

        if conference_name and supported_years:
            if conference_name not in conference_years:
                conference_years[conference_name] = []
            conference_years[conference_name].extend(supported_years)
            all_years.update(supported_years)

    all_years_sorted = sorted(list(all_years), reverse=True)
    conferences = sorted(conference_years.keys())

    for conf in conference_years:
        conference_years[conf] = sorted(conference_years[conf], reverse=True)

    return {"conferences": conferences, "years": all_years_sorted, "conference_years": conference_years}


def resolve_conference_from_url(url_path: str) -> Optional[str]:
    """
    Resolve a URL path segment to a canonical conference name using plugins.

    Performs case-insensitive matching against:
    1. Plugin conference names (e.g. ``NeurIPS``, ``ICLR``)
    2. Plugin names (e.g. ``neurips``, ``iclr``)

    Parameters
    ----------
    url_path : str
        URL path segment to resolve (e.g. ``"neurips"``, ``"ICLR"``).

    Returns
    -------
    str or None
        The canonical conference name if a match is found, ``None`` otherwise.

    Examples
    --------
    >>> resolve_conference_from_url("neurips")  # matches plugin name
    'NeurIPS'
    >>> resolve_conference_from_url("ICLR")  # matches conference name
    'ICLR'
    >>> resolve_conference_from_url("unknown") is None
    True
    """
    lookup: Dict[str, str] = {}
    available = get_available_filters()
    for conf in available.get("conferences", []):
        lookup[conf.lower()] = conf

    for meta in list_plugins():
        plugin_name = meta.get("name", "")
        conf_name = meta.get("conference_name", "")
        if plugin_name and conf_name:
            lookup[plugin_name.lower()] = conf_name

    return lookup.get(url_path.lower())

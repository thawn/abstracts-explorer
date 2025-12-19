"""
Download Plugins
================

This module provides a plugin system for different data downloaders.

The plugin system allows extending the package with new data sources
while maintaining a consistent interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class DownloaderPlugin(ABC):
    """
    Base class for all downloader plugins.

    Each plugin must implement the download method and provide metadata
    about its capabilities.
    """

    # Plugin metadata (should be overridden in subclasses)
    plugin_name: str = "base"
    plugin_description: str = "Base downloader plugin"
    supported_years: List[int] = []

    @abstractmethod
    def download(
        self,
        year: Optional[int] = None,
        output_path: Optional[str] = None,
        force_download: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Download papers from the data source.

        Parameters
        ----------
        year : int, optional
            Year to download papers for (if applicable)
        output_path : str, optional
            Path to save the downloaded data
        force_download : bool
            Force re-download even if data exists
        **kwargs : Any
            Additional plugin-specific parameters

        Returns
        -------
        dict
            Downloaded data in the standardized format:
            {
                'count': int,
                'next': None,
                'previous': None,
                'results': [list of papers]
            }
        """
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get plugin metadata.

        Returns
        -------
        dict
            Plugin metadata including name, description, supported years, etc.
        """
        pass

    def validate_year(self, year: Optional[int]) -> None:
        """
        Validate that the requested year is supported.

        Parameters
        ----------
        year : int or None
            Year to validate

        Raises
        ------
        ValueError
            If year is not supported by this plugin
        """
        if year is not None and self.supported_years and year not in self.supported_years:
            raise ValueError(
                f"Year {year} not supported by {self.plugin_name}. " f"Supported years: {self.supported_years}"
            )


class PluginRegistry:
    """Registry for managing downloader plugins."""

    def __init__(self):
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

    def unregister(self, plugin_name: str) -> None:
        """
        Unregister a plugin.

        Parameters
        ----------
        plugin_name : str
            Name of plugin to unregister
        """
        if plugin_name in self._plugins:
            del self._plugins[plugin_name]
            logger.info(f"Unregistered plugin: {plugin_name}")
        else:
            logger.warning(f"Plugin not found: {plugin_name}")

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


# Global plugin registry
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


__all__ = [
    "DownloaderPlugin",
    "PluginRegistry",
    "register_plugin",
    "get_plugin",
    "list_plugins",
    "list_plugin_names",
]

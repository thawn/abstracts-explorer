"""
NeurIPS Official Downloader Plugin
===================================

Plugin for downloading papers from the official NeurIPS conference data.
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
import logging

from neurips_abstracts.plugin import DownloaderPlugin
from neurips_abstracts.downloader import download_neurips_data

logger = logging.getLogger(__name__)


class NeurIPSDownloaderPlugin(DownloaderPlugin):
    """
    Plugin for downloading papers from official NeurIPS conference.

    This plugin wraps the existing NeurIPS downloader functionality
    and provides it through the plugin interface.
    """

    plugin_name = "neurips"
    plugin_description = "Official NeurIPS conference data downloader"
    supported_years = list(range(2013, 2026))  # NeurIPS years available

    def __init__(self, timeout: int = 30, verify_ssl: bool = True):
        """
        Initialize the NeurIPS downloader plugin.

        Parameters
        ----------
        timeout : int, default=30
            Request timeout in seconds
        verify_ssl : bool, default=True
            Whether to verify SSL certificates
        """
        self.timeout = timeout
        self.verify_ssl = verify_ssl

    def download(
        self,
        year: Optional[int] = None,
        output_path: Optional[str] = None,
        force_download: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Download papers from NeurIPS conference.

        Parameters
        ----------
        year : int, optional
            Conference year to download (default: 2025)
        output_path : str, optional
            Path to save the downloaded JSON file
        force_download : bool
            Force re-download even if file exists
        **kwargs : Any
            Additional parameters (timeout, verify_ssl can override defaults)

        Returns
        -------
        dict
            Downloaded data in format:
            {
                'count': int,
                'next': None,
                'previous': None,
                'results': [list of papers]
            }
        """
        if year is None:
            year = 2025

        # Validate year
        self.validate_year(year)

        # Get timeout and verify_ssl from kwargs or use defaults
        timeout = kwargs.get("timeout", self.timeout)
        verify_ssl = kwargs.get("verify_ssl", self.verify_ssl)

        logger.info(f"Downloading NeurIPS {year} data...")

        # Use the existing download function
        data = download_neurips_data(
            year=year, output_path=output_path, timeout=timeout, force_download=force_download
        )

        # Add year and conference fields to each paper
        if "results" in data and isinstance(data["results"], list):
            for paper in data["results"]:
                paper["year"] = year
                paper["conference"] = "NeurIPS"

        logger.info(f"Successfully downloaded {data.get('count', 0)} papers from NeurIPS {year}")

        return data

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get plugin metadata.

        Returns
        -------
        dict
            Plugin metadata including name, description, supported years
        """
        return {
            "name": self.plugin_name,
            "description": self.plugin_description,
            "supported_years": self.supported_years,
            "parameters": {
                "year": {
                    "type": "int",
                    "required": True,
                    "description": "Conference year to download",
                    "default": 2025,
                },
                "output_path": {"type": "str", "required": False, "description": "Path to save the downloaded data"},
                "force_download": {
                    "type": "bool",
                    "required": False,
                    "description": "Force re-download even if file exists",
                    "default": False,
                },
                "timeout": {
                    "type": "int",
                    "required": False,
                    "description": "Request timeout in seconds",
                    "default": 30,
                },
            },
        }


# Auto-register the plugin when imported
def _register():
    """Auto-register the NeurIPS plugin."""
    from neurips_abstracts.plugins import register_plugin

    plugin = NeurIPSDownloaderPlugin()
    register_plugin(plugin)
    logger.debug("NeurIPS downloader plugin registered")


# Register on import
_register()

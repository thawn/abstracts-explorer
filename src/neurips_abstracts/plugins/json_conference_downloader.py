"""
JSON Conference Downloader Base Class
======================================

Base class for conference downloader plugins that fetch JSON data from a URL.
This reduces code duplication between similar conference data downloaders.
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
import logging
import json
import requests

from neurips_abstracts.plugin import DownloaderPlugin, convert_neurips_to_lightweight_schema, LightweightPaper

logger = logging.getLogger(__name__)


class JSONConferenceDownloaderPlugin(DownloaderPlugin):
    """
    Base class for conference downloaders that fetch JSON data from a URL.

    Subclasses need to override:
    - plugin_name: Name of the plugin
    - plugin_description: Description of the plugin
    - supported_years: List of supported years
    - conference_name: Full conference name (e.g., "NeurIPS", "ICLR")
    - get_url(year): Method to construct the URL for a specific year
    """

    plugin_name = "json_conference_base"
    plugin_description = "Base class for JSON conference downloaders"
    supported_years = []
    conference_name = "Conference"

    def __init__(self, timeout: int = 30, verify_ssl: bool = True):
        """
        Initialize the JSON conference downloader plugin.

        Parameters
        ----------
        timeout : int, default=30
            Request timeout in seconds
        verify_ssl : bool, default=True
            Whether to verify SSL certificates
        """
        self.timeout = timeout
        self.verify_ssl = verify_ssl

    def get_url(self, year: int) -> str:
        """
        Get the download URL for a specific year.

        This method must be overridden by subclasses.

        Parameters
        ----------
        year : int
            Conference year

        Returns
        -------
        str
            URL to download JSON data from
        """
        raise NotImplementedError("Subclasses must implement get_url()")

    def download(
        self,
        year: Optional[int] = None,
        output_path: Optional[str] = None,
        force_download: bool = False,
        **kwargs: Any,
    ) -> List[LightweightPaper]:
        """
        Download papers from conference.

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
        list of LightweightPaper
            List of validated paper objects ready for database insertion

        Raises
        ------
        ValueError
            If the year is not supported
        RuntimeError
            If the download or JSON parsing fails
        """
        if year is None:
            year = 2025

        # Validate year
        self.validate_year(year)

        # Get timeout and verify_ssl from kwargs or use defaults
        timeout = kwargs.get("timeout", self.timeout)
        verify_ssl = kwargs.get("verify_ssl", self.verify_ssl)

        # Check if file already exists and should be loaded
        if output_path and not force_download:
            output_file = Path(output_path)
            if output_file.exists():
                logger.info(f"Loading existing data from: {output_file}")
                try:
                    with open(output_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    papers = [LightweightPaper(**paper) for paper in data]
                    logger.info(f"Successfully loaded {len(papers)} papers from local file")
                    return papers
                except (json.JSONDecodeError, IOError, Exception) as e:
                    logger.warning(f"Failed to load local file: {str(e)}. Downloading from URL...")

        logger.info(f"Downloading {self.conference_name} {year} data...")

        # Construct URL for the specific year
        url = self.get_url(year)

        # Download the data
        try:
            response = requests.get(url, timeout=timeout, verify=verify_ssl)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to download from {url}: {str(e)}") from e

        try:
            data = response.json()
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON response from {url}: {str(e)}") from e

        # Convert papers to lightweight schema
        papers_data = []
        if "results" in data and isinstance(data["results"], list):
            # Add year and conference fields before conversion
            for paper in data["results"]:
                paper["year"] = year
                paper["conference"] = self.conference_name

            # Convert to lightweight schema (returns list of dicts)
            papers_data = convert_neurips_to_lightweight_schema(data["results"])

        # Convert to LightweightPaper objects
        papers = [LightweightPaper(**paper) for paper in papers_data]

        # Save to file if path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            # Save as list of dicts for readability
            papers_json = [paper.model_dump() for paper in papers]
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(papers_json, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved JSON data to: {output_file}")

        logger.info(f"Successfully downloaded {len(papers)} papers from {self.conference_name} {year}")

        return papers

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get plugin metadata.

        Returns
        -------
        dict
            Plugin metadata including name, description, supported years, conference name
        """
        return {
            "name": self.plugin_name,
            "description": self.plugin_description,
            "supported_years": self.supported_years,
            "conference_name": self.conference_name,
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

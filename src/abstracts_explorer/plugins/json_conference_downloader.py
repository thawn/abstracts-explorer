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

from abstracts_explorer.plugin import (
    DownloaderPlugin,
    convert_to_lightweight_schema,
    LightweightPaper,
    validate_lightweight_papers,
)

logger = logging.getLogger(__name__)


class JSONConferenceDownloaderPlugin(DownloaderPlugin):
    """
    Base class for conference downloaders that fetch JSON data from a URL.

    Subclasses need to override:
    - plugin_name: Name of the plugin
    - plugin_description: Description of the plugin
    - _start_year: First supported year (used to compute supported_years dynamically)
    - conference_name: Full conference name (e.g., "NeurIPS", "ICLR")
    - get_url(year): Method to construct the URL for a specific year


    Starting 2026, EventHosts, the CMS behind NeurIPS/ICML/ICLR/CVPR and more
    split metadata (-orals-posters.json) and abstracts (-abstracts.json) into separate files
    Therefore, we now automatically download not only the main ``-orals-posters.json``, but also the
    corresponding ``-abstracts.json`` file when records have no inline abstract.
    """

    plugin_name = "json_conference_base"
    plugin_description = "Base class for JSON conference downloaders"
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

    def get_companion_abstracts_url(self, year: int) -> Optional[str]:
        """
        Get the companion abstract URL for an EventHosts paper URL.

        Subclasses with a different companion URL convention may override this
        method. Returning ``None`` disables companion abstract retrieval.

        Parameters
        ----------
        year : int
            Conference year

        Returns
        -------
        str or None
            Companion abstract URL, or ``None`` when the main URL does not use
            the EventHosts naming convention
        """
        url = self.get_url(year)
        suffix = "-orals-posters.json"
        if not url.endswith(suffix):
            return None
        return f"{url.removesuffix(suffix)}-abstracts.json"

    def _merge_companion_abstracts(
        self,
        records: List[Dict[str, Any]],
        year: int,
        timeout: int,
        verify_ssl: bool,
    ) -> None:
        """Add missing abstracts from an EventHosts companion JSON object."""
        missing_records = [
            record
            for record in records
            if (not isinstance(record.get("abstract"), str) or not record["abstract"].strip())
        ]
        if not missing_records:
            return

        url = self.get_companion_abstracts_url(year)
        if url is None:
            return

        try:
            response = requests.get(url, timeout=timeout, verify=verify_ssl)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            logger.warning("Failed to download companion abstracts from %s: %s", url, exc)
            return

        try:
            abstracts = response.json()
        except json.JSONDecodeError as exc:
            logger.warning("Invalid JSON in companion abstracts response from %s: %s", url, exc)
            return

        if not isinstance(abstracts, dict):
            logger.warning(
                "Invalid companion abstracts response from %s: expected an object",
                url,
            )
            return

        matched = 0
        for record in missing_records:
            paper_id = record.get("id")
            abstract = abstracts.get(str(paper_id)) if paper_id is not None else None
            if isinstance(abstract, str) and abstract.strip():
                record["abstract"] = abstract
                matched += 1

        unmatched = len(missing_records) - matched
        if unmatched:
            logger.warning(
                "No companion abstract found for %d of %d records from %s",
                unmatched,
                len(missing_records),
                url,
            )

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
                    papers = validate_lightweight_papers(data)
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
        source_record_count = 0
        if "results" in data and isinstance(data["results"], list):
            source_record_count = len(data["results"])
            self._merge_companion_abstracts(data["results"], year, timeout, verify_ssl)

            # Add year and conference fields before conversion
            for paper in data["results"]:
                paper["year"] = year
                paper["conference"] = self.conference_name

            # Convert to lightweight schema (returns list of dicts)
            papers_data = convert_to_lightweight_schema(data["results"])

        # Convert to LightweightPaper objects (skip papers that fail validation)
        papers = validate_lightweight_papers(papers_data)
        if source_record_count and not papers:
            logger.warning(
                "No valid papers produced from %d source records for %s %d",
                source_record_count,
                self.conference_name,
                year,
            )

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

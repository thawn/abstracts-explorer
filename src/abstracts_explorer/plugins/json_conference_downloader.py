"""
JSON Conference Downloader Base Class
======================================

Base class for conference downloader plugins that fetch JSON data from a URL.
This reduces code duplication between similar conference data downloaders.
"""

from typing import Any, Dict, List, Optional
import logging
import json

from abstracts_explorer.plugin import (
    DownloadOptions,
    HTTPDownloaderPlugin,
    convert_to_lightweight_schema,
    LightweightPaper,
    validate_lightweight_papers,
)
from abstracts_explorer.plugin.persistence import FailureSidecar

logger = logging.getLogger(__name__)


class JSONConferenceDownloaderPlugin(HTTPDownloaderPlugin):
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
    _SOURCE_LABEL = "conference data"

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
        timeout: float,
        verify_ssl: bool,
    ) -> List[Dict[str, Any]]:
        """
        Add missing abstracts from an EventHosts companion JSON object.

        Records whose companion abstract cannot be retrieved because the
        companion file itself failed to download or parse are reported in the
        returned failure list so they can be retried on the next run. Records
        that are simply absent from a successfully downloaded companion file
        are treated as genuinely lacking an abstract and are only logged.

        Parameters
        ----------
        records : list of dict
            Event record dictionaries parsed from the main JSON file.
        year : int
            Conference year.
        timeout : float
            HTTP timeout in seconds.
        verify_ssl : bool
            Whether to verify TLS certificates.

        Returns
        -------
        list of dict
            Failure entries suitable for :class:`FailureSidecar`.
        """
        missing_records = [
            record
            for record in records
            if (not isinstance(record.get("abstract"), str) or not record["abstract"].strip())
        ]
        if not missing_records:
            return []

        url = self.get_companion_abstracts_url(year)
        if url is None:
            return []

        missing_ids = [str(record.get("id")) for record in missing_records if record.get("id") is not None]

        try:
            response = self._request(url, timeout, verify_ssl)
        except RuntimeError as exc:
            logger.warning("Failed to download companion abstracts from %s: %s", url, exc)
            return [{"url": url, "type": "companion", "ids": missing_ids, "reason": str(exc)}]

        try:
            abstracts = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Invalid JSON in companion abstracts response from %s: %s", url, exc)
            return [{"url": url, "type": "companion", "ids": missing_ids, "reason": str(exc)}]

        if not isinstance(abstracts, dict):
            logger.warning(
                "Invalid companion abstracts response from %s: expected an object",
                url,
            )
            return [
                {
                    "url": url,
                    "type": "companion",
                    "ids": missing_ids,
                    "reason": "expected a JSON object",
                }
            ]

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
                "No companion abstract found for %d of %d records from %s; "
                "they are treated as having no abstract.",
                unmatched,
                len(missing_records),
                url,
            )
        return []

    def _download_uncached(
        self,
        year: int,
        output_path: Optional[str],
        options: DownloadOptions,
    ) -> List[LightweightPaper]:
        """
        Fetch and parse the conference JSON for *year*.

        Parameters
        ----------
        year : int
            Conference year to download.
        output_path : str, optional
            Cache path (unused by this base class; kept for the template).
        options : DownloadOptions
            Resolved download options.

        Returns
        -------
        list of LightweightPaper
            List of validated paper objects ready for database insertion.

        Raises
        ------
        RuntimeError
            If the download or JSON parsing fails.
        """
        timeout = options.timeout
        verify_ssl = options.verify_ssl

        logger.info(f"Downloading {self.conference_name} {year} data...")

        url = self.get_url(year)

        try:
            response = self._request(url, timeout, verify_ssl)
        except RuntimeError as e:
            raise RuntimeError(f"Failed to download from {url}: {str(e)}") from e

        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(f"Invalid JSON response from {url}: {str(e)}") from e

        papers_data = []
        source_record_count = 0
        failures: List[Dict[str, Any]] = []
        if "results" in data and isinstance(data["results"], list):
            source_record_count = len(data["results"])
            failures = self._merge_companion_abstracts(data["results"], year, timeout, verify_ssl)

            for paper in data["results"]:
                paper["year"] = year
                paper["conference"] = self.conference_name

            papers_data = convert_to_lightweight_schema(data["results"])

        papers = validate_lightweight_papers(papers_data)
        if source_record_count and not papers:
            logger.warning(
                "No valid papers produced from %d source records for %s %d",
                source_record_count,
                self.conference_name,
                year,
            )

        FailureSidecar(output_path, self.conference_name, year).save(failures)

        logger.info(f"Successfully downloaded {len(papers)} papers from {self.conference_name} {year}")

        return papers

    def _handle_cached(
        self,
        cached: List[LightweightPaper],
        year: int,
        output_path: str,
        options: DownloadOptions,
    ) -> List[LightweightPaper]:
        """
        Refresh a cached result when companion abstracts previously failed.

        Parameters
        ----------
        cached : list of LightweightPaper
            Papers loaded from *output_path*.
        year : int
            Requested year.
        output_path : str
            Cache file that was loaded.
        options : DownloadOptions
            Resolved download options.

        Returns
        -------
        list of LightweightPaper
            Refreshed papers, or the cached papers when no sidecar exists.
        """
        failures = FailureSidecar(output_path, self.conference_name, year).load()
        if not failures:
            return cached

        logger.info(
            "Retrying %d previously failed %s %d companion abstract fetch(es).",
            len(failures),
            self.conference_name,
            year,
        )
        papers = self._download_uncached(year, output_path, options)
        self._save_papers_json(papers, output_path, self._save_label)
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
                    "default": max(self.supported_years, default=self._start_year),
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

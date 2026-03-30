"""
IEEE VIS Official Downloader Plugin
=====================================

Plugin for downloading papers from the official IEEE VIS conference.
"""

import json
import logging
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional

from abstracts_explorer.plugin import sanitize_author_names, LightweightPaper, validate_lightweight_papers
from pydantic import ValidationError
from abstracts_explorer.plugins.json_conference_downloader import JSONConferenceDownloaderPlugin

logger = logging.getLogger(__name__)


class IEEEVISDownloaderPlugin(JSONConferenceDownloaderPlugin):
    """
    Plugin for downloading papers from the official IEEE VIS conference.

    This plugin downloads data from the IEEE VIS program website using
    their JSON API endpoint. The JSON format is a direct array of paper
    objects (unlike other conferences that wrap results in a ``results`` key).

    Notes
    -----
    The IEEE VIS JSON format differs from other conference plugins:

    - The response is a JSON array (not ``{"results": [...]}``)
    - Authors are given as ``{"name": "...", "email": null}`` objects
    - The session is stored in ``session_title`` (not ``session``)
    - The room is stored in ``session_room`` (not ``room_name``)
    - The DOI field is used to construct a URL
    """

    plugin_name = "ieeevis"
    plugin_description = "Official IEEE VIS conference data downloader"
    _start_year = 2025
    conference_name = "IEEE VIS"

    def get_url(self, year: int) -> str:
        """
        Get the download URL for IEEE VIS data.

        Parameters
        ----------
        year : int
            Conference year

        Returns
        -------
        str
            URL to download IEEE VIS JSON data
        """
        return f"https://ieeevis.org/year/{year}/program/papers.json"

    def _convert_paper(self, item: Dict[str, Any], year: int) -> Optional[LightweightPaper]:
        """
        Convert a single IEEE VIS paper record to a ``LightweightPaper``.

        Parameters
        ----------
        item : dict
            Raw paper record from the IEEE VIS JSON feed
        year : int
            Conference year

        Returns
        -------
        LightweightPaper or None
            Converted paper, or ``None`` if the record has no title or authors
        """
        title = (item.get("title") or "").strip()
        if not title:
            return None

        # Authors: list of {"name": "...", "email": null}
        raw_authors = item.get("authors", [])
        authors: List[str] = []
        for author in raw_authors:
            if isinstance(author, dict):
                name = (author.get("name") or "").strip()
                if name:
                    authors.append(name)
            elif isinstance(author, str):
                name = author.strip()
                if name:
                    authors.append(name)
        authors = sanitize_author_names(authors)

        if not authors:
            return None

        # Session comes from session_title; fall back to event_title
        session = (item.get("session_title") or item.get("event_title") or "No session").strip()

        paper_dict: Dict[str, Any] = {
            "title": title,
            "authors": authors,
            "abstract": (item.get("abstract") or "").strip(),
            "session": session,
            "poster_position": "",
            "year": year,
            "conference": self.conference_name,
        }

        # Optional fields
        if item.get("session_room"):
            paper_dict["room_name"] = item["session_room"]

        keywords = item.get("keywords")
        if keywords and isinstance(keywords, list):
            paper_dict["keywords"] = keywords

        if item.get("pdf_url"):
            paper_dict["paper_pdf_url"] = item["pdf_url"]

        doi = item.get("doi")
        if doi:
            paper_dict["url"] = f"https://doi.org/{doi}"

        award = (item.get("award") or "").strip()
        if award:
            paper_dict["award"] = award

        if item.get("time_stamp"):
            paper_dict["starttime"] = item["time_stamp"]

        try:
            return LightweightPaper(**paper_dict)
        except ValidationError as exc:
            logger.warning("Skipping paper '%s': validation failed: %s", title, exc)
            return None

    def download(
        self,
        year: Optional[int] = None,
        output_path: Optional[str] = None,
        force_download: bool = False,
        **kwargs: Any,
    ) -> List[LightweightPaper]:
        """
        Download papers from IEEE VIS.

        Parameters
        ----------
        year : int, optional
            Conference year to download (default: 2025)
        output_path : str, optional
            Path to save the downloaded JSON file
        force_download : bool
            Force re-download even if file exists
        **kwargs : Any
            Additional parameters (``timeout``, ``verify_ssl`` can override defaults)

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

        self.validate_year(year)

        timeout = kwargs.get("timeout", self.timeout)
        verify_ssl = kwargs.get("verify_ssl", self.verify_ssl)

        # Load from local file if it exists and force_download is False
        if output_path and not force_download:
            output_file = Path(output_path)
            if output_file.exists():
                logger.info(f"Loading existing data from: {output_file}")
                try:
                    with open(output_file, "r", encoding="utf-8") as f:
                        cached_data = json.load(f)
                    cached_papers = validate_lightweight_papers(cached_data)
                    logger.info(f"Successfully loaded {len(cached_papers)} papers from local file")
                    return cached_papers
                except (json.JSONDecodeError, IOError, Exception) as e:
                    logger.warning(f"Failed to load local file: {str(e)}. Downloading from URL...")

        logger.info(f"Downloading {self.conference_name} {year} data...")

        url = self.get_url(year)

        try:
            response = requests.get(url, timeout=timeout, verify=verify_ssl)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to download from {url}: {str(e)}") from e

        try:
            data = response.json()
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON response from {url}: {str(e)}") from e

        # IEEE VIS returns a direct JSON array, not {"results": [...]}
        if not isinstance(data, list):
            logger.warning(f"Unexpected JSON structure from {url}: expected a list, got {type(data).__name__}")
            data = []

        papers: List[LightweightPaper] = []
        for item in data:
            paper = self._convert_paper(item, year)
            if paper is not None:
                papers.append(paper)

        # Save to file if path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            papers_json = [paper.model_dump() for paper in papers]
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(papers_json, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved JSON data to: {output_file}")

        logger.info(f"Successfully downloaded {len(papers)} papers from {self.conference_name} {year}")

        return papers


# Auto-register the plugin when imported
def _register():
    """Auto-register the IEEE VIS plugin."""
    from abstracts_explorer.plugins import register_plugin

    plugin = IEEEVISDownloaderPlugin()
    register_plugin(plugin)
    logger.debug("IEEE VIS downloader plugin registered")


# Register on import
_register()

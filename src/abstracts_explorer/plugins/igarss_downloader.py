"""
IGARSS Conference Downloader Plugin
=====================================

Plugin for downloading papers from the IEEE International Geoscience and
Remote Sensing Symposium (IGARSS) via the IEEE Xplore search API.

Data source: ``ieeexplore.ieee.org``

The plugin queries the IEEE Xplore REST search API for papers published
under the IGARSS conference proceedings for a given year and converts
the results into :class:`~abstracts_explorer.plugin.LightweightPaper` objects.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from pydantic import ValidationError

from abstracts_explorer.plugin import (
    LightweightDownloaderPlugin,
    LightweightPaper,
    sanitize_author_names,
    validate_lightweight_papers,
)

logger = logging.getLogger(__name__)


class IGARSSDownloaderPlugin(LightweightDownloaderPlugin):
    """
    Plugin for downloading papers from the IGARSS conference via IEEE Xplore.

    IGARSS (IEEE International Geoscience and Remote Sensing Symposium) is
    the premier conference in geoscience and remote sensing, organized by the
    IEEE Geoscience and Remote Sensing Society.

    This plugin uses the IEEE Xplore REST search API to retrieve paper
    metadata including titles, abstracts, authors, keywords, and DOIs.
    Results are paginated and converted to
    :class:`~abstracts_explorer.plugin.LightweightPaper` objects.

    Notes
    -----
    The IEEE Xplore REST API returns up to 100 records per page. The plugin
    automatically paginates through all results and includes a configurable
    delay between pages to be respectful of the API.

    Year 2005 is excluded from the supported years because the IEEE Xplore
    records for IGARSS 2005 do not include abstracts.  Entries without
    abstracts match too well against arbitrary short queries and would
    produce misleading search results.
    """

    plugin_name = "igarss"
    plugin_description = (
        "IGARSS (IEEE International Geoscience and Remote Sensing Symposium) "
        "conference data downloader via IEEE Xplore"
    )
    _start_year = 1994
    conference_name = "IGARSS"

    #: Years that are excluded from the supported years because their records
    #: do not include abstracts.
    _excluded_years: List[int] = [2005]

    @property
    def supported_years(self) -> List[int]:
        """
        Dynamically computed supported years, excluding years without abstracts.

        Returns
        -------
        list of int
            Supported conference years (excludes years listed in ``_excluded_years``).
        """
        all_years = super().supported_years
        return [y for y in all_years if y not in self._excluded_years]

    @supported_years.setter
    def supported_years(self, value: List[int]) -> None:
        self._supported_years_cache = value

    #: IEEE Xplore REST search API endpoint.
    _API_URL = "https://ieeexplore.ieee.org/rest/search"

    #: Maximum records per page allowed by the IEEE Xplore API.
    _ROWS_PER_PAGE = 25

    #: Default delay between paginated requests (seconds).
    _PAGE_DELAY = 1.0

    def __init__(self, timeout: int = 30, verify_ssl: bool = True):
        """
        Initialize the IGARSS downloader plugin.

        Parameters
        ----------
        timeout : int, optional
            HTTP request timeout in seconds (default: 30).
        verify_ssl : bool, optional
            Whether to verify SSL certificates (default: True).
        """
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json, text/plain, */*",
                "Content-Type": "application/json",
                "Origin": "https://ieeexplore.ieee.org",
                "Referer": "https://ieeexplore.ieee.org/search/searchresult.jsp",
            }
        )
        adapter = HTTPAdapter(pool_maxsize=5)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    def get_url(self, year: int) -> str:
        """
        Get the IEEE Xplore search URL for IGARSS papers of a given year.

        Parameters
        ----------
        year : int
            Conference year.

        Returns
        -------
        str
            URL to the IEEE Xplore search results page for IGARSS papers.
        """
        return (
            "https://ieeexplore.ieee.org/search/searchresult.jsp"
            f"?action=search&newsearch=true&matchBoolean=true"
            f"&queryText=(%22Publication%20Title%22:igarss%20{year})"
        )

    def _build_search_payload(self, year: int, page_number: int = 1) -> Dict[str, Any]:
        """
        Build the JSON payload for an IEEE Xplore search API request.

        Parameters
        ----------
        year : int
            Conference year to search for.
        page_number : int, optional
            Page number for pagination (default: 1).

        Returns
        -------
        dict
            JSON payload for the POST request.
        """
        return {
            "newsearch": True,
            "queryText": f'("Publication Title":igarss {year})',
            "highlight": True,
            "returnFacets": ["ALL"],
            "returnType": "SEARCH",
            "matchPubs": True,
            "rowsPerPage": self._ROWS_PER_PAGE,
            "pageNumber": page_number,
        }

    def _fetch_page(
        self,
        year: int,
        page_number: int,
        timeout: int,
        verify_ssl: bool,
    ) -> Dict[str, Any]:
        """
        Fetch a single page of search results from IEEE Xplore.

        Parameters
        ----------
        year : int
            Conference year.
        page_number : int
            Page number to fetch.
        timeout : int
            HTTP request timeout in seconds.
        verify_ssl : bool
            Whether to verify SSL certificates.

        Returns
        -------
        dict
            Parsed JSON response from the API.

        Raises
        ------
        RuntimeError
            If the HTTP request fails or the response is not valid JSON.
        """
        payload = self._build_search_payload(year, page_number)

        try:
            response = self._session.post(
                self._API_URL,
                json=payload,
                timeout=timeout,
                verify=verify_ssl,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to fetch IGARSS {year} page {page_number} from IEEE Xplore: {e}") from e

        try:
            return response.json()
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON response from IEEE Xplore (page {page_number}): {e}") from e

    def _convert_paper(self, record: Dict[str, Any], year: int) -> Optional[LightweightPaper]:
        """
        Convert a single IEEE Xplore record to a :class:`LightweightPaper`.

        Parameters
        ----------
        record : dict
            A single record from the IEEE Xplore search results.
        year : int
            Conference year.

        Returns
        -------
        LightweightPaper or None
            Converted paper, or ``None`` if essential fields are missing.
        """
        title = (record.get("articleTitle") or "").strip()
        # Remove highlight tags that IEEE Xplore may add
        title = _strip_highlight_tags(title)
        if not title:
            return None

        # Authors
        raw_authors = record.get("authors", [])
        authors: List[str] = []
        for author in raw_authors:
            if isinstance(author, dict):
                # Try preferredName first, then authorName, then full name parts
                name = (
                    author.get("preferredName") or author.get("authorName") or author.get("normalizedName") or ""
                ).strip()
                name = _strip_highlight_tags(name)
                if name:
                    authors.append(name)
            elif isinstance(author, str):
                name = _strip_highlight_tags(author.strip())
                if name:
                    authors.append(name)
        authors = sanitize_author_names(authors)

        if not authors:
            return None

        # Abstract
        abstract = _strip_highlight_tags((record.get("abstract") or "").strip())

        # Session / publication title
        session = _strip_highlight_tags((record.get("publicationTitle") or "IGARSS").strip())

        paper_dict: Dict[str, Any] = {
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "session": session,
            "poster_position": str(record.get("articleNumber") or ""),
            "year": year,
            "conference": self.conference_name,
        }

        # DOI → URL
        doi = record.get("doi")
        if doi:
            paper_dict["url"] = f"https://doi.org/{doi}"

        # PDF URL
        pdf_link = record.get("pdfUrl") or record.get("pdfLink")
        if pdf_link:
            if pdf_link.startswith("/"):
                pdf_link = f"https://ieeexplore.ieee.org{pdf_link}"
            paper_dict["paper_pdf_url"] = pdf_link

        # Article number as original_id
        article_number = record.get("articleNumber")
        if article_number:
            try:
                paper_dict["original_id"] = int(article_number)
            except (ValueError, TypeError):
                pass

        # Keywords from index terms
        keywords = self._extract_keywords(record)
        if keywords:
            paper_dict["keywords"] = keywords

        # Start/end page info
        start_page = record.get("startPage")
        end_page = record.get("endPage")
        if start_page and end_page:
            paper_dict["poster_position"] = f"pp. {start_page}-{end_page}"

        try:
            return LightweightPaper(**paper_dict)
        except ValidationError as exc:
            logger.warning("Skipping paper '%s': validation failed: %s", title, exc)
            return None

    def _extract_keywords(self, record: Dict[str, Any]) -> Optional[List[str]]:
        """
        Extract keywords from an IEEE Xplore record's index terms.

        Parameters
        ----------
        record : dict
            IEEE Xplore record.

        Returns
        -------
        list of str or None
            Combined keywords from author and IEEE index terms, or ``None``
            if no keywords are found.
        """
        keywords: List[str] = []
        index_terms = record.get("indexTerms") or {}

        for term_group_key in ("author_terms", "ieee_terms", "mesh_terms", "controlledTerms"):
            term_group = index_terms.get(term_group_key)
            if isinstance(term_group, dict):
                terms = term_group.get("terms", [])
                if isinstance(terms, list):
                    for term in terms:
                        cleaned = _strip_highlight_tags(str(term).strip())
                        if cleaned and cleaned not in keywords:
                            keywords.append(cleaned)

        return keywords if keywords else None

    def download(
        self,
        year: Optional[int] = None,
        output_path: Optional[str] = None,
        force_download: bool = False,
        **kwargs: Any,
    ) -> List[LightweightPaper]:
        """
        Download IGARSS papers from IEEE Xplore for a given year.

        Parameters
        ----------
        year : int, optional
            Conference year to download (default: latest supported year).
        output_path : str, optional
            Path to save the downloaded JSON file. If the file already exists
            and ``force_download`` is ``False``, it is loaded directly.
        force_download : bool
            Force re-download even if a cached file exists.
        **kwargs : Any
            Additional parameters:

            - ``timeout`` (int): Override default HTTP timeout.
            - ``verify_ssl`` (bool): Override default SSL verification.
            - ``page_delay`` (float): Delay between paginated requests.

        Returns
        -------
        list of LightweightPaper
            List of validated paper objects ready for database insertion.

        Raises
        ------
        ValueError
            If the year is not supported.
        RuntimeError
            If the download or JSON parsing fails.
        """
        if year is None:
            year = max(self.supported_years) if self.supported_years else 2025

        self.validate_year(year)

        timeout = kwargs.get("timeout", self.timeout)
        verify_ssl = kwargs.get("verify_ssl", self.verify_ssl)
        page_delay = kwargs.get("page_delay", self._PAGE_DELAY)

        # Load from local file if it exists and force_download is False
        if output_path and not force_download:
            output_file = Path(output_path)
            if output_file.exists():
                logger.info("Loading existing IGARSS data from: %s", output_file)
                try:
                    with open(output_file, "r", encoding="utf-8") as f:
                        cached_data = json.load(f)
                    cached_papers = validate_lightweight_papers(cached_data)
                    logger.info(
                        "Successfully loaded %d papers from local file",
                        len(cached_papers),
                    )
                    return cached_papers
                except (json.JSONDecodeError, IOError, Exception) as e:
                    logger.warning(
                        "Failed to load local file: %s. Downloading from IEEE Xplore...",
                        e,
                    )

        logger.info("Downloading IGARSS %d data from IEEE Xplore...", year)

        # Fetch first page to get total record count
        first_page_data = self._fetch_page(year, 1, timeout, verify_ssl)
        total_records = first_page_data.get("totalRecords", 0)
        records = first_page_data.get("records", [])

        logger.info(
            "IGARSS %d: found %d total records on IEEE Xplore",
            year,
            total_records,
        )

        # Fetch remaining pages
        if total_records > self._ROWS_PER_PAGE:
            total_pages = (total_records + self._ROWS_PER_PAGE - 1) // self._ROWS_PER_PAGE
            for page_num in range(2, total_pages + 1):
                if page_delay > 0:
                    time.sleep(page_delay)
                logger.info(
                    "Fetching page %d/%d...",
                    page_num,
                    total_pages,
                )
                page_data = self._fetch_page(year, page_num, timeout, verify_ssl)
                page_records = page_data.get("records", [])
                if not page_records:
                    break
                records.extend(page_records)

        # Convert records to LightweightPaper objects
        papers: List[LightweightPaper] = []
        for record in records:
            paper = self._convert_paper(record, year)
            if paper is not None:
                papers.append(paper)

        # Save to file if path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            papers_json = [paper.model_dump() for paper in papers]
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(papers_json, f, indent=2, ensure_ascii=False)
            logger.info("Saved IGARSS JSON data to: %s", output_file)

        logger.info(
            "Successfully downloaded %d papers from IGARSS %d",
            len(papers),
            year,
        )

        return papers

    def get_metadata(self) -> Dict[str, Any]:
        """
        Return metadata describing this plugin.

        Returns
        -------
        dict
            Plugin name, description, supported years, and accepted parameters.
        """
        return {
            "name": self.plugin_name,
            "description": self.plugin_description,
            "conference_name": self.conference_name,
            "supported_years": self.supported_years,
            "parameters": {
                "year": "int – Conference year",
                "output_path": "str – Path to save downloaded JSON",
                "force_download": "bool – Force re-download even when output_path exists",
                "timeout": "int – HTTP request timeout in seconds",
                "verify_ssl": "bool – Whether to verify SSL certificates",
                "page_delay": "float – Delay between paginated requests (seconds)",
            },
        }


def _strip_highlight_tags(text: str) -> str:
    """
    Remove IEEE Xplore highlight markup tags from text.

    The IEEE Xplore API wraps matched search terms in ``<highlight>``
    tags which need to be stripped for clean text storage.

    Parameters
    ----------
    text : str
        Text that may contain ``<highlight>...</highlight>`` tags.

    Returns
    -------
    str
        Text with highlight tags removed.
    """
    return re.sub(r"</?highlight>", "", text)


# ---------------------------------------------------------------------------
# Auto-registration
# ---------------------------------------------------------------------------


def _register() -> None:
    """Auto-register the IGARSS plugin when this module is imported."""
    from abstracts_explorer.plugins import register_plugin

    plugin = IGARSSDownloaderPlugin()
    register_plugin(plugin)
    logger.debug("IGARSSDownloaderPlugin registered")


_register()

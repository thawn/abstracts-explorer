"""
HAICON Conference Downloader Plugin
=====================================

Plugin for downloading papers from the Helmholtz AI Conference (HAICON) via
the ConfTool conference management system.

Data source: ``www.conftool.pro/haicon{YY}``

The plugin scrapes the public ConfTool agenda pages of the HAICON conference.
It first fetches the sessions overview to discover all session IDs, then
retrieves each session's detail page (with ``abstracts=show``) to parse paper
titles, authors, abstracts, and metadata.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import requests
from bs4 import BeautifulSoup
from pydantic import ValidationError

from abstracts_explorer.plugin import (
    LightweightDownloaderPlugin,
    LightweightPaper,
    sanitize_author_names,
    validate_lightweight_papers,
)

logger = logging.getLogger(__name__)


class HAICONDownloaderPlugin(LightweightDownloaderPlugin):
    """
    Plugin for downloading papers from the Helmholtz AI Conference (HAICON).

    HAICON is an annual conference organised by Helmholtz AI, covering
    artificial intelligence research across scientific domains. This plugin
    scrapes the public ConfTool agenda pages at
    ``www.conftool.pro/haicon{YY}`` to retrieve paper metadata including
    titles, abstracts, authors, and session information.

    The plugin:

    1. Fetches the sessions overview page to discover all session IDs.
    2. For each session, retrieves the detail page with ``abstracts=show``.
    3. Parses each paper's metadata from the structured HTML.

    Notes
    -----
    Only papers with non-empty abstracts are included. Duplicate papers
    (i.e., papers appearing in multiple sessions) are de-duplicated by their
    ConfTool paper ID.

    Rate limiting is applied between requests to be respectful of the
    ConfTool server.

    Usage
    -----
    ::

        abstracts-explorer download --plugin haicon --year 2026
    """

    plugin_name = "haicon"
    plugin_description = "HAICON (Helmholtz AI Conference) data downloader via ConfTool"
    _start_year = 2026
    conference_name = "HAICON"

    #: Base URL template for ConfTool; ``{short_year}`` is the 2-digit year.
    _CONFTOOL_BASE = "https://www.conftool.pro/haicon{short_year}"

    #: Delay in seconds between paginated HTTP requests.
    _REQUEST_DELAY = 0.5

    def __init__(self, timeout: int = 30, verify_ssl: bool = True):
        """
        Initialise the HAICON downloader plugin.

        Parameters
        ----------
        timeout : int, optional
            HTTP request timeout in seconds (default: 30).
        verify_ssl : bool, optional
            Whether to verify SSL certificates (default: ``True``).
        """
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64; rv:120.0) " "Gecko/20100101 Firefox/120.0"),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_url(self, year: int) -> str:
        """
        Get the ConfTool sessions overview URL for a specific year.

        Parameters
        ----------
        year : int
            Conference year (e.g., 2026).

        Returns
        -------
        str
            URL to the sessions overview page for the given year.
        """
        short_year = str(year)[-2:]
        return f"{self._CONFTOOL_BASE.format(short_year=short_year)}/sessions.php"

    def download(
        self,
        year: Optional[int] = None,
        output_path: Optional[str] = None,
        force_download: bool = False,
        **kwargs: Any,
    ) -> List[LightweightPaper]:
        """
        Download HAICON conference papers from ConfTool for a given year.

        Parameters
        ----------
        year : int, optional
            Conference year to download (default: 2026).
        output_path : str, optional
            Path to save the downloaded JSON file.  If the file already
            exists and ``force_download`` is ``False``, it is loaded directly
            without making any HTTP requests.
        force_download : bool
            Force re-download even when a cached file exists.
        **kwargs : Any
            Additional parameters:

            - ``timeout`` (int): Override the default HTTP timeout.
            - ``verify_ssl`` (bool): Override the default SSL verification.
            - ``request_delay`` (float): Override the delay between requests.

        Returns
        -------
        list of LightweightPaper
            List of validated paper objects ready for database insertion.

        Raises
        ------
        ValueError
            If the year is not supported.
        RuntimeError
            If the HTTP requests fail or parsing produces no papers.
        """
        if year is None:
            year = 2026

        self.validate_year(year)

        timeout = kwargs.get("timeout", self.timeout)
        verify_ssl = kwargs.get("verify_ssl", self.verify_ssl)
        request_delay = kwargs.get("request_delay", self._REQUEST_DELAY)

        # Load from local file if it exists and force_download is False
        if output_path and not force_download:
            output_file = Path(output_path)
            if output_file.exists():
                logger.info("Loading existing HAICON data from: %s", output_file)
                try:
                    with open(output_file, "r", encoding="utf-8") as fh:
                        cached_data = json.load(fh)
                    cached_papers = validate_lightweight_papers(cached_data)
                    logger.info(
                        "Successfully loaded %d papers from local file",
                        len(cached_papers),
                    )
                    return cached_papers
                except (json.JSONDecodeError, OSError, Exception) as exc:
                    logger.warning(
                        "Failed to load local file: %s. Downloading from ConfTool...",
                        exc,
                    )

        short_year = str(year)[-2:]
        base_url = self._CONFTOOL_BASE.format(short_year=short_year)

        logger.info("Downloading HAICON %d data from ConfTool...", year)

        # Step 1: Get all session IDs from the overview page
        session_ids = self._fetch_session_ids(base_url, timeout, verify_ssl)
        logger.info("Found %d sessions to scrape", len(session_ids))

        # Step 2: For each session, fetch the detail page and parse papers
        papers_by_id: Dict[int, LightweightPaper] = {}
        for session_id in session_ids:
            if request_delay > 0:
                time.sleep(request_delay)
            session_papers = self._fetch_session_papers(base_url, session_id, year, timeout, verify_ssl)
            for paper in session_papers:
                # De-duplicate by original_id (ConfTool paper ID)
                if paper.original_id is not None and paper.original_id not in papers_by_id:
                    papers_by_id[paper.original_id] = paper
                elif paper.original_id is None:
                    # Use title as key for papers without an ID
                    title_key = -(hash(paper.title) % (10**9))
                    if title_key not in papers_by_id:
                        papers_by_id[title_key] = paper

        papers = list(papers_by_id.values())
        logger.info(
            "Successfully downloaded %d unique papers from HAICON %d",
            len(papers),
            year,
        )

        # Save to file if path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            papers_json = [paper.model_dump() for paper in papers]
            with open(output_file, "w", encoding="utf-8") as fh:
                json.dump(papers_json, fh, indent=2, ensure_ascii=False)
            logger.info("Saved HAICON JSON data to: %s", output_file)

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
                "year": "int – Conference year (e.g., 2026)",
                "output_path": "str – Path to save downloaded JSON",
                "force_download": "bool – Force re-download even when output_path exists",
                "timeout": "int – HTTP request timeout in seconds",
                "verify_ssl": "bool – Whether to verify SSL certificates",
                "request_delay": "float – Delay between HTTP requests (seconds)",
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_session_ids(
        self,
        base_url: str,
        timeout: int,
        verify_ssl: bool,
    ) -> List[int]:
        """
        Fetch all session IDs from the ConfTool sessions overview page.

        Extracts unique ``form_session`` values from links on the page.

        Parameters
        ----------
        base_url : str
            Base URL for the ConfTool conference (e.g.,
            ``https://www.conftool.pro/haicon26``).
        timeout : int
            HTTP request timeout in seconds.
        verify_ssl : bool
            Whether to verify SSL certificates.

        Returns
        -------
        list of int
            Sorted list of unique session IDs found on the overview page.

        Raises
        ------
        RuntimeError
            If the HTTP request fails.
        """
        url = f"{base_url}/sessions.php"
        try:
            response = self._session.get(url, timeout=timeout, verify=verify_ssl)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Failed to fetch HAICON sessions overview from {url}: {exc}") from exc

        soup = BeautifulSoup(response.text, "html.parser")
        session_ids: Set[int] = set()

        for link in soup.find_all("a", href=True):
            href = link["href"]
            match = re.search(r"form_session=(\d+)", href)
            if match:
                session_ids.add(int(match.group(1)))

        return sorted(session_ids)

    def _fetch_session_papers(
        self,
        base_url: str,
        session_id: int,
        year: int,
        timeout: int,
        verify_ssl: bool,
    ) -> List[LightweightPaper]:
        """
        Fetch and parse all papers from a single ConfTool session detail page.

        Parameters
        ----------
        base_url : str
            Base URL for the ConfTool conference.
        session_id : int
            ConfTool session identifier.
        year : int
            Conference year (used for :class:`LightweightPaper` metadata).
        timeout : int
            HTTP request timeout in seconds.
        verify_ssl : bool
            Whether to verify SSL certificates.

        Returns
        -------
        list of LightweightPaper
            Papers parsed from this session (may be empty if the session
            has no presentations with abstracts).
        """
        url = (
            f"{base_url}/index.php"
            f"?page=browseSessions"
            f"&abstracts=show"
            f"&form_date=all"
            f"&form_session={session_id}"
            f"&presentations=show"
        )

        try:
            response = self._session.get(url, timeout=timeout, verify=verify_ssl)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            logger.warning(
                "Failed to fetch session %d: %s – skipping",
                session_id,
                exc,
            )
            return []

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract session name from the page header
        session_name = self._extract_session_name(soup)

        # Find all paper divs (id starts with 'paperID')
        papers: List[LightweightPaper] = []
        for paper_div in soup.find_all("div", id=re.compile(r"^paperID\d+")):
            paper = self._parse_paper_div(paper_div, session_name, year)
            if paper is not None:
                papers.append(paper)

        if papers:
            logger.debug(
                "Session %d (%s): parsed %d papers",
                session_id,
                session_name,
                len(papers),
            )

        return papers

    def _extract_session_name(self, soup: BeautifulSoup) -> str:
        """
        Extract the session name from a ConfTool session detail page.

        The session name is contained in a ``<span class='font11'><b>…</b></span>``
        element within the page.

        Parameters
        ----------
        soup : BeautifulSoup
            Parsed HTML of the session detail page.

        Returns
        -------
        str
            Session name, or ``"unknown session at HAICON"`` if it cannot be found.
        """
        tag = soup.find("span", class_="font11")
        if tag:
            bold = tag.find("b")
            if bold:
                return bold.get_text(strip=True)
        return "unknown session at HAICON"

    def _parse_paper_div(
        self,
        paper_div: Any,
        session_name: str,
        year: int,
    ) -> Optional[LightweightPaper]:
        """
        Parse a single ConfTool paper ``<div>`` element into a
        :class:`LightweightPaper`.

        Parameters
        ----------
        paper_div : bs4.element.Tag
            The ``<div id='paperID{N}'>`` element.
        session_name : str
            Name of the session this paper belongs to.
        year : int
            Conference year.

        Returns
        -------
        LightweightPaper or None
            ``None`` if the paper has no title, authors, or abstract.
        """
        # ---- Paper ID ----
        id_span = paper_div.find("span", class_="paper_id")
        paper_id: Optional[int] = None
        if id_span:
            match = re.search(r"ID:\s*(\d+)", id_span.get_text())
            if match:
                paper_id = int(match.group(1))

        # ---- Poster / slot position ----
        session_span = paper_div.find("span", class_="paper_session")
        poster_position = ""
        if session_span:
            pos_text = session_span.get_text(strip=True)
            # Strip leading " / " separator
            poster_position = re.sub(r"^\s*/\s*", "", pos_text).strip()

        # ---- Title ----
        title_tag = paper_div.find("p", class_="paper_title")
        title = title_tag.get_text(strip=True) if title_tag else ""
        if not title:
            return None

        # ---- Authors ----
        author_tag = paper_div.find("p", class_="paper_author")
        authors: List[str] = []
        if author_tag:
            authors = self._extract_authors(author_tag)
        if not authors:
            return None

        # ---- Abstract ----
        # The abstract wrapper is the innermost div.paper that contains
        # p.paper_abstract elements but is NOT the outermost paper wrapper.
        abstract_paragraphs: List[str] = []
        # Look for the nested div.paper that wraps the abstract
        for inner_div in paper_div.find_all("div", class_="paper"):
            # Skip the outer paper div itself
            if inner_div == paper_div:
                continue
            for para in inner_div.find_all("p", class_="paper_abstract"):
                text = para.get_text(separator=" ", strip=True)
                if text:
                    abstract_paragraphs.append(text)
            if abstract_paragraphs:
                break

        abstract = "\n\n".join(abstract_paragraphs)
        if not abstract.strip():
            return None

        # ---- Keywords from topics ----
        keywords: List[str] = []
        for topic_span in paper_div.find_all("span", class_="paper_topics"):
            label_span = topic_span.find_previous_sibling("span", class_="paper_label")
            label = label_span.get_text(strip=True).rstrip(":").strip() if label_span else ""
            for kw in topic_span.get_text(separator=",").split(","):
                kw = kw.strip()
                if kw:
                    keywords.append(f"{label}: {kw}" if label else kw)

        # ---- Start time (optional) ----
        starttime: Optional[str] = None
        time_span = paper_div.find("span", class_="paper_time_value")
        if time_span:
            starttime = time_span.get_text(strip=True)

        try:
            return LightweightPaper(
                title=title,
                abstract=abstract,
                authors=authors,
                session=session_name,
                poster_position=poster_position,
                year=year,
                conference=self.conference_name,
                original_id=paper_id,
                keywords=keywords if keywords else None,
                starttime=starttime,
            )
        except ValidationError as exc:
            logger.warning("Skipping paper '%s': validation failed: %s", title, exc)
            return None

    def _extract_authors(self, author_tag: Any) -> List[str]:
        """
        Extract a clean list of author names from a ``<p class="paper_author">``
        element.

        Superscript affiliation indices and underline (presenting author)
        markup are removed; only the plain name text is kept.

        Parameters
        ----------
        author_tag : bs4.element.Tag
            The ``<p class="paper_author">`` element.

        Returns
        -------
        list of str
            Cleaned, sanitised author names.
        """
        # Remove all <sup> elements (affiliation indices) before reading text
        for sup in author_tag.find_all("sup"):
            sup.decompose()

        raw = author_tag.get_text(separator=",")
        # Split on commas, strip whitespace
        names = [n.strip() for n in raw.split(",") if n.strip()]
        return sanitize_author_names(names)


# ---------------------------------------------------------------------------
# Auto-registration
# ---------------------------------------------------------------------------


def _register() -> None:
    """Auto-register the HAICON plugin when this module is imported."""
    from abstracts_explorer.plugins import register_plugin

    plugin = HAICONDownloaderPlugin()
    register_plugin(plugin)
    logger.debug("HAICONDownloaderPlugin registered")


_register()

"""
IEEE VIS Official Downloader Plugin
=====================================

Plugin for downloading papers from the official IEEE VIS conference.
"""

import logging
from bs4 import BeautifulSoup, Tag
from typing import Any, Dict, List, Optional

from abstracts_explorer.plugin import sanitize_author_names, LightweightPaper
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

    Starting with IEEE VIS 2026 the official site no longer publishes
    ``papers.json``. Those years are read from the server-rendered HTML
    program page instead, keeping every entry that has a title, authors,
    and a non-empty abstract.
    """

    plugin_name = "ieeevis"
    plugin_description = "Official IEEE VIS conference data downloader"
    _start_year = 2025
    conference_name = "IEEE VIS"
    _SOURCE_LABEL = "IEEE VIS data"

    #: First year whose program is published as an HTML page rather than JSON.
    _HTML_PROGRAM_START_YEAR = 2026

    def _uses_html_program(self, year: int) -> bool:
        """
        Return whether *year* uses the new HTML program page.

        Parameters
        ----------
        year : int
            Conference year.

        Returns
        -------
        bool
            ``True`` for years served by the Astro program page.
        """
        return year >= self._HTML_PROGRAM_START_YEAR

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
            URL to the IEEE VIS JSON feed (2025 and earlier) or the HTML
            program page (2026 and later).
        """
        if self._uses_html_program(year):
            return f"https://ieeevis.org/year/{year}/program/content"
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

        url = item.get("url")
        doi = item.get("doi")
        if url:
            paper_dict["url"] = url
        elif doi:
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

    @staticmethod
    def _card_title(card: Tag) -> str:
        """Return the display title from an HTML program card."""
        link = card.select_one("a.program-card__title-link")
        if link is not None:
            title = link.get_text(" ", strip=True)
            if title:
                return title
        return str(card.get("data-title") or "").strip()

    @staticmethod
    def _card_authors(card: Tag) -> List[str]:
        """Return author names from an HTML program card."""
        names = [link.get_text(" ", strip=True) for link in card.select("a.program-card__person-link")]
        if not names:
            raw = str(card.get("data-authors") or "").strip()
            names = [name.strip() for name in raw.split(",") if name.strip()]
        return sanitize_author_names([name for name in names if name])

    @staticmethod
    def _card_session(card: Tag) -> str:
        """Return the session title from an HTML program card."""
        for link in card.select("a.program-card__action-link"):
            href = str(link.get("href") or "")
            if "/program/session/" not in href:
                continue
            text = link.get_text(" ", strip=True)
            if " - " in text:
                return text.split(" - ", 1)[1].strip()
        return str(card.get("data-session") or "").strip() or "No session"

    @staticmethod
    def _card_keywords(card: Tag) -> List[str]:
        """Return keywords from an HTML program card."""
        raw = str(card.get("data-keywords-label") or card.get("data-keywords") or "").strip()
        if not raw or raw.lower() == "no keywords listed":
            return []
        return [keyword.strip() for keyword in raw.split(",") if keyword.strip()]

    @staticmethod
    def _card_award(card: Tag) -> str:
        """Return a normalized award label from an HTML program card."""
        award = str(card.get("data-award") or "").strip()
        return award.title() if award else ""

    def _parse_program_html(self, html: str, year: int) -> List[LightweightPaper]:
        """
        Parse the IEEE VIS HTML program page into lightweight papers.

        Parameters
        ----------
        html : str
            HTML of the IEEE VIS program content page.
        year : int
            Conference year.

        Returns
        -------
        list of LightweightPaper
            Papers with a title and a non-empty abstract.
        """
        soup = BeautifulSoup(html, "html.parser")
        papers: List[LightweightPaper] = []
        for card in soup.select("[data-program-card][data-paper-id]"):
            title = self._card_title(card)
            abstract = str(card.get("data-abstract") or "").strip()
            if not title or not abstract:
                continue

            authors = self._card_authors(card)
            if not authors:
                continue

            title_link = card.select_one("a.program-card__title-link")
            href = str(title_link.get("href") or "") if title_link is not None else ""
            detail_url = f"https://ieeevis.org{href}" if href.startswith("/") else href

            item: Dict[str, Any] = {
                "title": title,
                "abstract": abstract,
                "authors": [{"name": name} for name in authors],
                "keywords": self._card_keywords(card),
                "session_title": self._card_session(card),
                "event_title": str(card.get("data-event") or "").strip(),
                "url": detail_url,
                "award": self._card_award(card),
                "time_stamp": str(card.get("data-start") or "").strip(),
            }
            paper = self._convert_paper(item, year)
            if paper is not None:
                papers.append(paper)

        return papers

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
            If the download or parsing fails
        """
        if year is None:
            year = 2025

        self.validate_year(year)

        timeout = kwargs.get("timeout", self.timeout)
        verify_ssl = kwargs.get("verify_ssl", self.verify_ssl)

        # Load from local file if it exists and force_download is False
        if output_path and not force_download:
            cached_papers = self._load_papers_json(output_path, "URL")
            if cached_papers is not None:
                return cached_papers

        logger.info(f"Downloading {self.conference_name} {year} data...")

        url = self.get_url(year)

        try:
            response = self._request(url, timeout, verify_ssl)
        except RuntimeError as e:
            raise RuntimeError(f"Failed to download from {url}: {str(e)}") from e

        if self._uses_html_program(year):
            response.encoding = "utf-8"
            html_papers = self._parse_program_html(response.text, year)
            self._save_papers_json(html_papers, output_path, self.conference_name)
            logger.info(f"Successfully downloaded {len(html_papers)} papers from {self.conference_name} {year}")
            return html_papers

        try:
            data = response.json()
        except ValueError as e:
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

        # Save to file only if path provided and papers were found
        self._save_papers_json(papers, output_path, self.conference_name)

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

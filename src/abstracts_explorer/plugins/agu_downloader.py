"""
AGU Conference Downloader Plugin
================================

Downloads abstracts from the AGU annual meeting's public m-anage meeting API.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import requests
from bs4 import BeautifulSoup
from pydantic import ValidationError

from abstracts_explorer.plugin import LightweightDownloaderPlugin, LightweightPaper, sanitize_author_names

logger = logging.getLogger(__name__)


class AGUDownloaderPlugin(LightweightDownloaderPlugin):
    """
    Download abstracts from the AGU annual meeting program.

    The plugin reads sections and sessions from the public m-anage meeting API,
    then fetches each scheduled contribution and its author records.

    Annual meetings from 2014 onward are supported. Meetings through 2023 use
    the legacy ``fmYY`` app slug, while 2024 and later use ``aguYY``.
    """

    plugin_name = "agu"
    plugin_description = "AGU annual meeting data downloader via m-anage"
    conference_name = "AGU"
    _start_year = 2014
    _BATCH_SIZE = 20
    _REQUEST_DELAY = 0.1
    _BASE_URL = "https://studio.m-anage.com/agu"
    _LEGACY_SLUG_LAST_YEAR = 2023

    def __init__(self, timeout: int = 30, verify_ssl: bool = True):
        """
        Initialize the AGU downloader.

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

    def get_url(self, year: int) -> str:
        """
        Get the AGU meeting-app URL for a conference year.

        Meetings through 2023 use the legacy ``fmYY`` slug, while 2024 and
        later use ``aguYY``.

        Parameters
        ----------
        year : int
            Conference year.

        Returns
        -------
        str
            Public meeting-app URL.
        """
        prefix = "fm" if year <= self._LEGACY_SLUG_LAST_YEAR else "agu"
        short_year = str(year)[-2:]
        return f"{self._BASE_URL}/{prefix}{short_year}/meetingapp.cgi"

    def download(
        self,
        year: Optional[int] = None,
        output_path: Optional[str] = None,
        force_download: bool = False,
        **kwargs: Any,
    ) -> List[LightweightPaper]:
        """
        Download and validate AGU abstracts for a conference year.

        Parameters
        ----------
        year : int, optional
            Conference year (defaults to the latest supported year).
        output_path : str, optional
            Path to save the lightweight JSON file.
        force_download : bool, optional
            Re-fetch data even when a cached file exists.
        **kwargs : Any
            Optional ``timeout``, ``verify_ssl``, and ``request_delay`` overrides.

        Returns
        -------
        list of LightweightPaper
            Validated conference abstracts.

        Raises
        ------
        ValueError
            If the year is not supported.
        RuntimeError
            If the API request fails or no abstracts are found.
        """
        if year is None:
            year = max(self.supported_years, default=self._start_year)
        self.validate_year(year)

        if output_path and not force_download:
            cached_papers = self._load_papers_json(output_path, "AGU")
            if cached_papers is not None:
                return cached_papers

        timeout = kwargs.get("timeout", self.timeout)
        verify_ssl = kwargs.get("verify_ssl", self.verify_ssl)
        request_delay = kwargs.get("request_delay", self._REQUEST_DELAY)
        meeting_url = self.get_url(year)
        api_url = meeting_url.replace("meetingapp.cgi", "meetingapi.cgi")

        programs = self._fetch_json(
            f"{api_url}/ModuleProgramBook/0/ChildList_Program",
            timeout,
            verify_ssl,
            {"date": datetime.now().isoformat(timespec="seconds")},
        )
        session_refs: Set[str] = set()
        for program in self._records(programs):
            for session_key in ("ChildList_Session", "ChildList_SessionReadOnly", "ChildList_UnscheduledSession"):
                session_refs.update(program.get(session_key) or [])

        sessions = self._fetch_batch_records(api_url, sorted(session_refs), timeout, verify_ssl, request_delay)
        slot_refs = {
            slot_ref for session in sessions.values() for slot_ref in (session.get("ChildList_PaperSlot") or [])
        }
        slots = self._fetch_batch_records(api_url, sorted(slot_refs), timeout, verify_ssl, request_delay)

        slot_by_paper: Dict[str, Dict[str, Any]] = {}
        paper_refs: Set[str] = set()
        for slot in slots.values():
            for paper_ref in slot.get("ChildList_Entry") or []:
                paper_refs.add(paper_ref)
                slot_by_paper.setdefault(paper_ref, slot)

        papers = self._fetch_batch_records(api_url, sorted(paper_refs), timeout, verify_ssl, request_delay)
        person_refs = {
            person_ref for paper in papers.values() for person_ref in (paper.get("ChildList_Person") or [])
        }
        people = self._fetch_batch_records(api_url, sorted(person_refs), timeout, verify_ssl, request_delay)

        unique_papers: Dict[Any, LightweightPaper] = {}
        for paper_ref, paper_data in papers.items():
            paper = self._convert_paper(paper_data, slot_by_paper.get(paper_ref, {}), people, year, meeting_url)
            if paper is None:
                continue
            paper_key: Any = (
                paper.original_id if paper.original_id is not None else (paper.title.casefold(), paper.session)
            )
            unique_papers.setdefault(paper_key, paper)

        result = list(unique_papers.values())
        if not result:
            raise RuntimeError(f"No abstracts with authors found for AGU {year}")

        self._save_papers_json(result, output_path, "AGU")
        logger.info("Downloaded %d AGU %d abstracts", len(result), year)
        return result

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
                "force_download": "bool – Force re-download when output_path exists",
                "timeout": "int – HTTP request timeout in seconds",
                "verify_ssl": "bool – Whether to verify SSL certificates",
                "request_delay": "float – Delay between API batches in seconds",
            },
        }

    def _fetch_json(
        self,
        url: str,
        timeout: int,
        verify_ssl: bool,
        params: Optional[Dict[str, str]] = None,
    ) -> Any:
        """Fetch and decode one m-anage API response."""
        try:
            response = self._session.get(url, params=params, timeout=timeout, verify=verify_ssl)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            raise RuntimeError(f"Failed to fetch AGU meeting data from {url}: {exc}") from exc

    def _fetch_batch_records(
        self,
        api_url: str,
        refs: List[str],
        timeout: int,
        verify_ssl: bool,
        request_delay: float,
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch m-anage records in bounded BatchRead requests."""
        records: Dict[str, Dict[str, Any]] = {}
        for start in range(0, len(refs), self._BATCH_SIZE):
            batch = refs[start : start + self._BATCH_SIZE]
            payload = self._fetch_json(
                f"{api_url}/BatchRead/0",
                timeout,
                verify_ssl,
                {"ids": ",".join(batch)},
            )
            models = payload.get("models", {}) if isinstance(payload, dict) else {}
            if isinstance(models, dict):
                records.update({ref: record for ref, record in models.items() if isinstance(record, dict)})
            if request_delay > 0 and start + self._BATCH_SIZE < len(refs):
                time.sleep(request_delay)
        return records

    @staticmethod
    def _records(payload: Any) -> List[Dict[str, Any]]:
        """Return record values from an indexed m-anage child-list response."""
        if isinstance(payload, list):
            return [record for record in payload if isinstance(record, dict)]
        if isinstance(payload, dict):
            return [record for record in payload.values() if isinstance(record, dict)]
        return []

    def _convert_paper(
        self,
        paper_data: Dict[str, Any],
        slot_data: Dict[str, Any],
        people: Dict[str, Dict[str, Any]],
        year: int,
        meeting_url: str,
    ) -> Optional[LightweightPaper]:
        """Convert an m-anage paper record into the lightweight schema."""
        title = str(paper_data.get("Title") or "").strip()
        abstract = BeautifulSoup(str(paper_data.get("Abstract") or ""), "html.parser").get_text(" ", strip=True)
        if not title or not abstract:
            return None

        authors: List[str] = []
        for person_ref in paper_data.get("ChildList_Person") or []:
            person = people.get(person_ref, {})
            name = " ".join(
                part.strip()
                for part in (
                    str(person.get("FirstName") or ""),
                    str(person.get("MiddleName") or ""),
                    str(person.get("LastName") or ""),
                )
                if part.strip()
            )
            if not name:
                name = str(person.get("FullName") or person.get("Name") or "").strip()
            if name:
                authors.append(name)
        authors = sanitize_author_names(authors)
        if not authors:
            return None

        session = str(paper_data.get("SessionTitle") or slot_data.get("SessionTitle") or "").strip()
        if not session:
            session = str(paper_data.get("ProgramTitle") or slot_data.get("ProgramTitle") or f"AGU {year}").strip()

        raw_id = paper_data.get("id")
        try:
            original_id = int(raw_id) if raw_id is not None else None
        except (TypeError, ValueError):
            original_id = None

        poster_position = (
            paper_data.get("FinalNumber")
            or paper_data.get("PosterNumber")
            or slot_data.get("Entry_FinalNumber")
            or slot_data.get("Entry_PosterNumber")
            or ""
        )

        try:
            return LightweightPaper(
                title=title,
                abstract=abstract,
                authors=authors,
                session=session,
                poster_position=str(poster_position),
                year=year,
                conference=self.conference_name,
                original_id=original_id,
                url=str(paper_data.get("URL") or f"{meeting_url}/Paper/{raw_id}"),
            )
        except ValidationError as exc:
            logger.warning("Skipping AGU abstract '%s': validation failed: %s", title, exc)
            return None


def _register() -> None:
    """Auto-register the AGU plugin when this module is imported."""
    from abstracts_explorer.plugins import register_plugin

    register_plugin(AGUDownloaderPlugin())


_register()

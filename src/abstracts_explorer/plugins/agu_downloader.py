"""
AGU Conference Downloader Plugin
================================

Downloads abstracts from the AGU annual meeting's public m-anage meeting API.
"""

import logging
import string
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
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
    _REQUEST_DELAY = 0.5
    _BASE_URL = "https://studio.m-anage.com/agu"
    _LEGACY_SLUG_LAST_YEAR = 2023
    _MAX_FETCH_ATTEMPTS = 6
    _RETRY_BASE_DELAY = 5.0
    _MAX_RETRY_DELAY = 120.0
    _RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
    _PEOPLE_INDEX_LETTERS = tuple(string.ascii_uppercase)

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
        self._rate_limit_until = 0.0
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json, text/plain, */*",
            }
        )

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
        programs = self._records(programs)
        sessions = self._fetch_sessions(api_url, programs, timeout, verify_ssl, request_delay)
        papers = self._fetch_papers(api_url, sessions, timeout, verify_ssl, request_delay)
        if not papers:
            raise RuntimeError(f"No abstracts with authors found for AGU {year}")

        person_refs = {
            person_ref for paper in papers.values() for person_ref in (paper.get("ChildList_Person") or [])
        }
        people = self._fetch_people(api_url, sorted(person_refs), timeout, verify_ssl, request_delay)

        unique_papers: Dict[Any, LightweightPaper] = {}
        for paper_data in papers.values():
            paper = self._convert_paper(paper_data, {}, people, year, meeting_url)
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
        """
        Fetch and decode one m-anage API response with retry handling.

        Transient failures (HTTP 429/5xx and connection errors) are retried
        with exponential backoff. A 429 response sets a session-wide cooldown
        based on the ``Retry-After`` header so subsequent requests also wait.
        """
        for attempt in range(self._MAX_FETCH_ATTEMPTS):
            self._wait_for_rate_limit()
            try:
                response = self._session.get(url, params=params, timeout=timeout, verify=verify_ssl)
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                status = self._status_code(exc)
                retryable = (
                    status in self._RETRYABLE_STATUSES
                    if status is not None
                    else isinstance(exc, (requests.ConnectionError, requests.Timeout))
                )
                if not retryable or attempt == self._MAX_FETCH_ATTEMPTS - 1:
                    detail = f"HTTP {status}: {exc}" if status is not None else str(exc)
                    raise RuntimeError(f"Failed to fetch AGU meeting data from {url}: {detail}") from exc

                delay = self._retry_delay(exc, attempt)
                if status == 429:
                    self._rate_limit_until = max(self._rate_limit_until, time.monotonic() + delay)
                    logger.warning("AGU m-anage rate limit hit; waiting %.0f seconds before retrying %s", delay, url)
                else:
                    logger.warning("Retrying AGU request to %s in %.0f seconds after: %s", url, delay, exc)
                    time.sleep(delay)

        raise RuntimeError(f"Failed to fetch AGU meeting data from {url}")

    def _wait_for_rate_limit(self) -> None:
        """Wait until the session-wide rate-limit cooldown has elapsed."""
        remaining = self._rate_limit_until - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
            self._rate_limit_until = 0.0

    @staticmethod
    def _status_code(exc: Exception) -> Optional[int]:
        """Return the HTTP status code attached to an exception, if any."""
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        return status if isinstance(status, int) else None

    def _retry_delay(self, exc: Exception, attempt: int) -> float:
        """Compute the retry delay, honouring ``Retry-After`` when present."""
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", {}) or {}
        retry_after = headers.get("Retry-After")
        delay = self._RETRY_BASE_DELAY * (2**attempt)
        if retry_after:
            try:
                delay = float(retry_after)
            except (TypeError, ValueError):
                try:
                    retry_at = parsedate_to_datetime(str(retry_after))
                    if retry_at.tzinfo is None:
                        retry_at = retry_at.replace(tzinfo=timezone.utc)
                    delay = (retry_at - datetime.now(timezone.utc)).total_seconds()
                except (TypeError, ValueError, OverflowError):
                    pass
        return max(0.0, min(delay, self._MAX_RETRY_DELAY))

    def _fetch_batch_records(
        self,
        api_url: str,
        refs: List[str],
        timeout: int,
        verify_ssl: bool,
        request_delay: float,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch m-anage records in bounded BatchRead requests.

        m-anage may answer a BatchRead with only some of the requested records,
        listing the rest under ``unanswered`` or ``single``. As the m-anage web
        app does, records missing from the batch response are fetched
        individually so throttling does not silently drop data.
        """
        records: Dict[str, Dict[str, Any]] = {}
        single_fetches = 0
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

            refused = payload.get("refused", {}) if isinstance(payload, dict) else {}
            for ref in batch:
                if ref in records:
                    continue
                status = refused.get(ref) if isinstance(refused, dict) else None
                try:
                    status_code = int(status) if status is not None else None
                except (TypeError, ValueError):
                    status_code = None
                if status_code is not None and status_code not in self._RETRYABLE_STATUSES:
                    logger.debug("Skipping unavailable AGU record %s (HTTP %s)", ref, status_code)
                    continue

                record = self._fetch_json(f"{api_url}/{ref}", timeout, verify_ssl)
                if isinstance(record, dict):
                    records[ref] = record
                single_fetches += 1
                if request_delay > 0:
                    time.sleep(request_delay)

            if request_delay > 0 and start + self._BATCH_SIZE < len(refs):
                time.sleep(request_delay)

        if single_fetches:
            logger.info(
                "Fetched %d AGU record(s) individually after incomplete batch responses",
                single_fetches,
            )
        return records

    def _fetch_sessions(
        self,
        api_url: str,
        programs: List[Dict[str, Any]],
        timeout: int,
        verify_ssl: bool,
        request_delay: float,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch all sessions through the program child-list endpoints.

        Program-level child lists return complete session records and avoid the
        partial BatchRead responses that m-anage produces under load.
        """
        sessions: Dict[str, Dict[str, Any]] = {}
        fallback_refs: Set[str] = set()
        params = {"date": datetime.now().isoformat(timespec="seconds")}
        session_keys = ("ChildList_Session", "ChildList_SessionReadOnly", "ChildList_UnscheduledSession")

        for program in programs:
            parent = str(program.get("_url") or f"Program/{program.get('id')}")
            for session_key in session_keys:
                refs = program.get(session_key) or []
                if not refs:
                    continue
                data = self._fetch_json(f"{api_url}/{parent}/{session_key}", timeout, verify_ssl, params)
                for record in self._records(data):
                    ref = str(record.get("_url") or f"Session/{record.get('id')}")
                    sessions[ref] = record
                fallback_refs.update(ref for ref in refs if ref not in sessions)
                if request_delay > 0:
                    time.sleep(request_delay)

        if fallback_refs:
            sessions.update(
                self._fetch_batch_records(api_url, sorted(fallback_refs), timeout, verify_ssl, request_delay)
            )
        return sessions

    def _fetch_papers(
        self,
        api_url: str,
        sessions: Dict[str, Dict[str, Any]],
        timeout: int,
        verify_ssl: bool,
        request_delay: float,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch full paper records through the session child-list endpoints.

        Session child lists return complete paper records, avoiding the partial
        BatchRead responses m-anage produces under load. Sessions without a
        direct paper list fall back to their paper slots.
        """
        papers: Dict[str, Dict[str, Any]] = {}
        expected_refs: Set[str] = set()
        slot_refs: Set[str] = set()
        params = {"date": datetime.now().isoformat(timespec="seconds")}

        for session in sessions.values():
            session_ref = str(session.get("_url") or f"Session/{session.get('id')}")
            refs = session.get("ChildList_Paper") or []
            if not refs:
                slot_refs.update(session.get("ChildList_PaperSlot") or [])
                continue
            expected_refs.update(refs)
            data = self._fetch_json(f"{api_url}/{session_ref}/ChildList_Paper", timeout, verify_ssl, params)
            for paper in self._records(data):
                ref = str(paper.get("_url") or f"Paper/{paper.get('id')}")
                papers[ref] = paper
            if request_delay > 0:
                time.sleep(request_delay)

        if slot_refs:
            slots = self._fetch_batch_records(api_url, sorted(slot_refs), timeout, verify_ssl, request_delay)
            for slot in slots.values():
                expected_refs.update(slot.get("ChildList_Entry") or [])

        missing = sorted(ref for ref in expected_refs if ref not in papers)
        if missing:
            papers.update(self._fetch_batch_records(api_url, missing, timeout, verify_ssl, request_delay))
        return papers

    def _fetch_people(
        self,
        api_url: str,
        person_refs: List[str],
        timeout: int,
        verify_ssl: bool,
        request_delay: float,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch people through the alphabetic people index.

        The people index returns every person for a letter in one request, so
        the complete people directory takes only a few requests. Any referenced
        people missing from the index are fetched through BatchRead.
        """
        if not person_refs:
            return {}

        people: Dict[str, Dict[str, Any]] = {}
        params = {"date": datetime.now().isoformat(timespec="seconds")}
        for letter in self._PEOPLE_INDEX_LETTERS:
            try:
                data = self._fetch_json(
                    f"{api_url}/PeopleIndex/{letter}/ChildList_Person",
                    timeout,
                    verify_ssl,
                    params,
                )
            except RuntimeError as exc:
                if "HTTP 404" in str(exc):
                    logger.debug("No AGU people index for letter %s", letter)
                    continue
                raise
            for person in self._records(data):
                ref = str(person.get("_url") or f"Person/{person.get('id')}")
                people[ref] = person
            if request_delay > 0:
                time.sleep(request_delay)

        missing = sorted(ref for ref in person_refs if ref not in people)
        if missing:
            people.update(self._fetch_batch_records(api_url, missing, timeout, verify_ssl, request_delay))
        return people

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

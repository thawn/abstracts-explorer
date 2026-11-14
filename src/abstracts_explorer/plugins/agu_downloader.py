"""
AGU Conference Downloader Plugin
================================

Downloads abstracts from the AGU annual meeting's public m-anage meeting API.
"""

import logging
import string
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from bs4 import BeautifulSoup
from pydantic import ValidationError
from tqdm import tqdm

from abstracts_explorer.plugin import HTTPDownloaderPlugin, LightweightPaper, sanitize_author_names

logger = logging.getLogger(__name__)


class AGUDownloaderPlugin(HTTPDownloaderPlugin):
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
    _PEOPLE_INDEX_LETTERS = tuple(string.ascii_uppercase)
    _USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    )
    _ACCEPT = "application/json, text/plain, */*"
    _SOURCE_LABEL = "AGU meeting data"

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
            Optional ``timeout``, ``verify_ssl``, ``request_delay``, and
            ``progress`` overrides.

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
        show_progress = kwargs.get("progress")
        meeting_url = self.get_url(year)
        api_url = meeting_url.replace("meetingapp.cgi", "meetingapi.cgi")

        programs = self._fetch_json(
            f"{api_url}/ModuleProgramBook/0/ChildList_Program",
            timeout,
            verify_ssl,
            {"date": datetime.now().isoformat(timespec="seconds")},
        )
        programs = self._records(programs)
        sessions = self._fetch_sessions(api_url, programs, timeout, verify_ssl, request_delay, show_progress)
        papers = self._fetch_papers(api_url, sessions, timeout, verify_ssl, request_delay, show_progress)
        if not papers:
            raise RuntimeError(f"No abstracts with authors found for AGU {year}")

        person_refs = {
            person_ref for paper in papers.values() for person_ref in (paper.get("ChildList_Person") or [])
        }
        people = self._fetch_people(api_url, sorted(person_refs), timeout, verify_ssl, request_delay, show_progress)

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
                "progress": "bool – Show progress bars during the download (default: auto)",
            },
        }

    def _fetch_batch_records(
        self,
        api_url: str,
        refs: List[str],
        timeout: int,
        verify_ssl: bool,
        request_delay: float,
        show_progress: Optional[bool] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch m-anage records in bounded BatchRead requests.

        m-anage may answer a BatchRead with only some of the requested records,
        listing the rest under ``unanswered`` or ``single``. As the m-anage web
        app does, records missing from the batch response are fetched
        individually so throttling does not silently drop data.

        Parameters
        ----------
        show_progress : bool, optional
            Whether to display a progress bar. ``None`` (default) shows the
            bar only when stderr is an interactive terminal.
        """
        if not refs:
            return {}

        records: Dict[str, Dict[str, Any]] = {}
        single_fetches = 0
        disable = None if show_progress is None else not show_progress
        with tqdm(
            total=len(refs),
            desc="Fetching AGU records",
            unit="record",
            disable=disable,
        ) as progress:
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

                progress.update(len(batch))
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
        show_progress: Optional[bool] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch all sessions through the program child-list endpoints.

        Program-level child lists return complete session records and avoid the
        partial BatchRead responses that m-anage produces under load.

        Parameters
        ----------
        show_progress : bool, optional
            Whether to display a progress bar. ``None`` (default) shows the
            bar only when stderr is an interactive terminal.
        """
        sessions: Dict[str, Dict[str, Any]] = {}
        fallback_refs: Set[str] = set()
        params = {"date": datetime.now().isoformat(timespec="seconds")}
        session_keys = ("ChildList_Session", "ChildList_SessionReadOnly", "ChildList_UnscheduledSession")
        program_requests = [
            (program, session_key) for program in programs for session_key in session_keys if program.get(session_key)
        ]
        disable = None if show_progress is None else not show_progress

        for program, session_key in tqdm(
            program_requests,
            total=len(program_requests),
            desc="Fetching AGU sessions",
            unit="request",
            disable=disable,
        ):
            parent = str(program.get("_url") or f"Program/{program.get('id')}")
            refs = program[session_key]
            data = self._fetch_json(f"{api_url}/{parent}/{session_key}", timeout, verify_ssl, params)
            for record in self._records(data):
                ref = str(record.get("_url") or f"Session/{record.get('id')}")
                sessions[ref] = record
            fallback_refs.update(ref for ref in refs if ref not in sessions)
            if request_delay > 0:
                time.sleep(request_delay)

        if fallback_refs:
            sessions.update(
                self._fetch_batch_records(
                    api_url,
                    sorted(fallback_refs),
                    timeout,
                    verify_ssl,
                    request_delay,
                    show_progress,
                )
            )
        return sessions

    def _fetch_papers(
        self,
        api_url: str,
        sessions: Dict[str, Dict[str, Any]],
        timeout: int,
        verify_ssl: bool,
        request_delay: float,
        show_progress: Optional[bool] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch full paper records through the session child-list endpoints.

        Session child lists return complete paper records, avoiding the partial
        BatchRead responses m-anage produces under load. Sessions without a
        direct paper list fall back to their paper slots.

        Parameters
        ----------
        show_progress : bool, optional
            Whether to display a progress bar. ``None`` (default) shows the
            bar only when stderr is an interactive terminal.
        """
        papers: Dict[str, Dict[str, Any]] = {}
        expected_refs: Set[str] = set()
        slot_refs: Set[str] = set()
        params = {"date": datetime.now().isoformat(timespec="seconds")}
        disable = None if show_progress is None else not show_progress

        for session in tqdm(
            sessions.values(),
            total=len(sessions),
            desc="Fetching AGU papers",
            unit="session",
            disable=disable,
        ):
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
            slots = self._fetch_batch_records(
                api_url,
                sorted(slot_refs),
                timeout,
                verify_ssl,
                request_delay,
                show_progress,
            )
            for slot in slots.values():
                expected_refs.update(slot.get("ChildList_Entry") or [])

        missing = sorted(ref for ref in expected_refs if ref not in papers)
        if missing:
            papers.update(
                self._fetch_batch_records(
                    api_url,
                    missing,
                    timeout,
                    verify_ssl,
                    request_delay,
                    show_progress,
                )
            )
        return papers

    def _fetch_people(
        self,
        api_url: str,
        person_refs: List[str],
        timeout: int,
        verify_ssl: bool,
        request_delay: float,
        show_progress: Optional[bool] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch people through the alphabetic people index.

        The people index returns every person for a letter in one request, so
        the complete people directory takes only a few requests. Any referenced
        people missing from the index are fetched through BatchRead.

        Parameters
        ----------
        show_progress : bool, optional
            Whether to display a progress bar. ``None`` (default) shows the
            bar only when stderr is an interactive terminal.
        """
        if not person_refs:
            return {}

        people: Dict[str, Dict[str, Any]] = {}
        params = {"date": datetime.now().isoformat(timespec="seconds")}
        disable = None if show_progress is None else not show_progress
        for letter in tqdm(
            self._PEOPLE_INDEX_LETTERS,
            total=len(self._PEOPLE_INDEX_LETTERS),
            desc="Fetching AGU people",
            unit="letter",
            disable=disable,
        ):
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
            people.update(
                self._fetch_batch_records(
                    api_url,
                    missing,
                    timeout,
                    verify_ssl,
                    request_delay,
                    show_progress,
                )
            )
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

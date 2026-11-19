"""
AGU Conference Downloader Plugin
================================

Downloads abstracts from the AGU annual meeting's public m-anage meeting API.
"""

import json
import logging
import string
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from bs4 import BeautifulSoup
from pydantic import ValidationError
from tqdm import tqdm

from abstracts_explorer.plugin import HTTPDownloaderPlugin, LightweightPaper, sanitize_author_names
from abstracts_explorer.plugin.persistence import append_jsonl, clear_directory, load_json, load_jsonl, save_json

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
    _MAX_WORKERS: int = 16
    _REQUEST_DELAY: float = 0.1
    _DEFAULT_REQUEST_DELAY = _REQUEST_DELAY
    _DEFAULT_MAX_WORKERS = _MAX_WORKERS
    _BASE_URL = "https://studio.m-anage.com/agu"
    _LEGACY_SLUG_LAST_YEAR = 2023
    _MAX_FETCH_ATTEMPTS = 6
    _RETRY_BASE_DELAY = 5.0
    _MAX_RETRY_DELAY = 120.0
    _PEOPLE_INDEX_LETTERS = tuple(string.ascii_uppercase)
    _CHECKPOINT_VERSION = 1
    _CHECKPOINT_FLUSH_INTERVAL = 50
    _USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    )
    _ACCEPT = "application/json, text/plain, */*"
    _SOURCE_LABEL = "AGU meeting data"

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the AGU downloader and its shared request delay state."""
        super().__init__(*args, **kwargs)
        self._delay_lock = threading.Lock()
        self._last_request_at = 0.0

    def _wait_for_request_slot(self, request_delay: float) -> None:
        """Wait so concurrent workers share one aggregate request rate."""
        if request_delay <= 0:
            return
        with self._delay_lock:
            now = time.monotonic()
            wait = max(0.0, self._last_request_at + request_delay - now)
            self._last_request_at = now + wait
        if wait > 0:
            time.sleep(wait)

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

    def _run_parallel(
        self,
        items: Sequence[Any],
        worker: Callable[[Any], Any],
        *,
        max_workers: int,
        desc: str,
        unit: str,
        show_progress: Optional[bool] = None,
        initial: int = 0,
        total: Optional[int] = None,
    ) -> List[Any]:
        """
        Run *worker* over *items* concurrently while preserving input order.

        Progress is updated from the calling thread only so ``tqdm`` does not
        need to be thread-safe. Results are stored by input index, so callers
        can merge them deterministically even though requests finish out of
        order.

        Parameters
        ----------
        initial : int, optional
            Number of entries already restored from a checkpoint. The progress
            bar starts at this value while still using the full *total*.
        total : int, optional
            Total number of entries including checkpointed ones. Defaults to
            ``len(items)``.
        """
        results: List[Any] = [None] * len(items)
        disable = None if show_progress is None else not show_progress
        bar_total = len(items) if total is None else total
        if not items:
            if total is not None:
                self._show_completed_progress(desc, unit, bar_total, show_progress, initial=initial)
            return results

        errors: List[Exception] = []
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            futures = {executor.submit(worker, item): index for index, item in enumerate(items)}
            with tqdm(
                total=bar_total,
                initial=initial,
                desc=desc,
                unit=unit,
                disable=disable,
            ) as progress:
                for future in as_completed(futures):
                    try:
                        results[futures[future]] = future.result()
                    except Exception as exc:
                        errors.append(exc)
                        logger.warning("AGU request failed: %s", exc)
                    finally:
                        progress.update(1)

        if errors:
            first_error = errors[0]
            raise RuntimeError(
                f"{len(errors)} of {len(items)} {desc.lower()} request(s) failed; " f"first error: {first_error}"
            ) from first_error
        return results

    @staticmethod
    def _show_completed_progress(
        desc: str,
        unit: str,
        total: int,
        show_progress: Optional[bool],
        initial: Optional[int] = None,
    ) -> None:
        """Display a progress bar for work already satisfied by a checkpoint."""
        disable = None if show_progress is None else not show_progress
        progress = total if initial is None else initial
        with tqdm(total=total, initial=progress, desc=desc, unit=unit, disable=disable):
            pass

    @staticmethod
    def _checkpoint_dir(output_path: str) -> Path:
        """Return the checkpoint directory for an output file."""
        return Path(f"{output_path}.checkpoint")

    def _prepare_checkpoint(
        self,
        output_path: Optional[str],
        year: int,
        api_url: str,
        resume: bool,
    ) -> Optional[Path]:
        """
        Initialise a checkpoint directory, reusing it only when it matches.

        A checkpoint stores the year and API URL it belongs to. Mismatched or
        incomplete metadata causes the directory to be cleared so stale data
        can never be mixed with a different conference year.
        """
        if not output_path:
            return None

        directory = self._checkpoint_dir(output_path)
        meta_path = directory / "meta.json"
        expected = {"version": self._CHECKPOINT_VERSION, "year": year, "api_url": api_url}
        if resume and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                meta = None
            if meta == expected:
                return directory

        self._clear_checkpoint(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._save_json(meta_path, expected)
        return directory

    @staticmethod
    def _clear_checkpoint(directory: Path) -> None:
        """Remove a checkpoint directory if it exists."""
        clear_directory(directory)

    @staticmethod
    def _load_json(path: Optional[Path]) -> Any:
        """Load a JSON checkpoint, returning ``None`` when it is unusable."""
        return load_json(path)

    @staticmethod
    def _save_json(path: Path, data: Any) -> None:
        """Write JSON atomically so an interrupted write cannot corrupt it."""
        save_json(path, data)

    @staticmethod
    def _load_jsonl(path: Optional[Path]) -> List[Dict[str, Any]]:
        """Load append-only JSONL records, ignoring a corrupt trailing line."""
        return load_jsonl(path)

    @staticmethod
    def _append_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
        """Append records to a JSONL checkpoint."""
        append_jsonl(path, records)

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
            Optional ``timeout``, ``verify_ssl``, ``request_delay``,
            ``max_workers``, ``resume``, and ``progress`` overrides.

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
        year = self._resolve_year(year)
        self.validate_year(year)

        options = self.resolve_options(kwargs)

        if output_path and not force_download:
            cached_papers = self._load_papers_json(output_path, "AGU")
            if cached_papers is not None and self._cache_matches_requested_year(cached_papers, year):
                self._clear_checkpoint(self._checkpoint_dir(output_path))
                return cached_papers

        timeout = options.timeout
        verify_ssl = options.verify_ssl
        request_delay = options.request_delay
        max_workers = max(1, int(options.max_workers))
        resume = options.resume
        show_progress = options.progress
        meeting_url = self.get_url(year)
        api_url = meeting_url.replace("meetingapp.cgi", "meetingapi.cgi")

        self.configure_pool(max_workers)
        checkpoint_dir = self._prepare_checkpoint(output_path, year, api_url, resume)

        sessions: Optional[Dict[str, Dict[str, Any]]] = None
        if checkpoint_dir is not None:
            stored_sessions = self._load_json(checkpoint_dir / "sessions.json")
            if isinstance(stored_sessions, dict):
                sessions = {str(ref): record for ref, record in stored_sessions.items() if isinstance(record, dict)}

        if sessions is None:
            programs = self._fetch_json(
                f"{api_url}/ModuleProgramBook/0/ChildList_Program",
                timeout,
                verify_ssl,
                {"date": datetime.now().isoformat(timespec="seconds")},
            )
            programs = self._records(programs)
            sessions = self._fetch_sessions(
                api_url,
                programs,
                timeout,
                verify_ssl,
                request_delay,
                show_progress,
                max_workers=max_workers,
            )
            if checkpoint_dir is not None:
                self._save_json(checkpoint_dir / "sessions.json", sessions)
        elif sessions:
            self._show_completed_progress(
                "Fetching AGU sessions",
                "session",
                len(sessions),
                show_progress,
            )

        existing_papers: Dict[str, Dict[str, Any]] = {}
        done_sessions: Set[str] = set()
        existing_people: Dict[str, Dict[str, Any]] = {}
        done_letters: Set[str] = set()
        if checkpoint_dir is not None:
            for record in self._load_jsonl(checkpoint_dir / "papers.jsonl"):
                ref = record.get("ref")
                data = record.get("data")
                if isinstance(ref, str) and isinstance(data, dict):
                    existing_papers[ref] = data
            done_sessions = {
                record["ref"]
                for record in self._load_jsonl(checkpoint_dir / "paper_sessions.jsonl")
                if isinstance(record.get("ref"), str)
            }
            for record in self._load_jsonl(checkpoint_dir / "people.jsonl"):
                ref = record.get("ref")
                data = record.get("data")
                if isinstance(ref, str) and isinstance(data, dict):
                    existing_people[ref] = data
            done_letters = {
                record["letter"]
                for record in self._load_jsonl(checkpoint_dir / "people_letters.jsonl")
                if isinstance(record.get("letter"), str)
            }

        papers = self._fetch_papers(
            api_url,
            sessions,
            timeout,
            verify_ssl,
            request_delay,
            show_progress,
            max_workers=max_workers,
            existing_papers=existing_papers,
            done_sessions=done_sessions,
            checkpoint_dir=checkpoint_dir,
        )
        if not papers:
            raise RuntimeError(f"No abstracts with authors found for AGU {year}")

        person_refs = {
            person_ref for paper in papers.values() for person_ref in (paper.get("ChildList_Person") or [])
        }
        people = self._fetch_people(
            api_url,
            sorted(person_refs),
            timeout,
            verify_ssl,
            request_delay,
            show_progress,
            max_workers=max_workers,
            existing_people=existing_people,
            done_letters=done_letters,
            checkpoint_dir=checkpoint_dir,
        )

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
        if checkpoint_dir is not None:
            self._clear_checkpoint(checkpoint_dir)
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
                "request_delay": "float – Delay before each concurrent request in seconds",
                "max_workers": "int – Maximum concurrent HTTP requests (default: 16)",
                "resume": "bool – Resume an interrupted download from its checkpoint",
                "progress": "bool – Show progress bars during the download (default: auto)",
            },
        }

    def _fetch_batch_records(
        self,
        api_url: str,
        refs: List[str],
        timeout: float,
        verify_ssl: bool,
        request_delay: float,
        show_progress: Optional[bool] = None,
        max_workers: int = 1,
        on_batch: Optional[Callable[[Dict[str, Dict[str, Any]]], None]] = None,
        initial: int = 0,
        total: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch m-anage records in bounded BatchRead requests.

        m-anage may answer a BatchRead with only some of the requested records,
        listing the rest under ``unanswered`` or ``single``. As the m-anage web
        app does, records missing from the batch response are fetched
        individually (concurrently) so throttling does not silently drop data.

        Parameters
        ----------
        show_progress : bool, optional
            Whether to display a progress bar. ``None`` (default) shows the
            bar only when stderr is an interactive terminal.
        max_workers : int, optional
            Maximum concurrent individual fallback requests (default: 1).
        on_batch : callable, optional
            Called with each batch's newly fetched records as soon as they are
            available, so callers can checkpoint partial progress.
        initial : int, optional
            Number of records already restored from a checkpoint.
        total : int, optional
            Total number of records including checkpointed ones. Defaults to
            ``len(refs)``.
        """
        if not refs:
            if total is not None:
                self._show_completed_progress(
                    "Fetching AGU records",
                    "record",
                    total,
                    show_progress,
                    initial=initial,
                )
            return {}

        records: Dict[str, Dict[str, Any]] = {}
        pending: List[str] = []
        single_fetches = 0
        disable = None if show_progress is None else not show_progress
        bar_total = len(refs) if total is None else total
        with tqdm(
            total=bar_total,
            initial=initial,
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
                batch_records = (
                    {ref: record for ref, record in models.items() if isinstance(record, dict)}
                    if isinstance(models, dict)
                    else {}
                )
                records.update(batch_records)
                if on_batch is not None and batch_records:
                    on_batch(batch_records)
                progress.update(len(batch_records))

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
                        progress.update(1)
                        continue
                    pending.append(ref)

                if start + self._BATCH_SIZE < len(refs):
                    self._wait_for_request_slot(request_delay)

            for start in range(0, len(pending), self._BATCH_SIZE):
                chunk = pending[start : start + self._BATCH_SIZE]
                chunk_records = self._fetch_records_individually(
                    api_url,
                    chunk,
                    timeout,
                    verify_ssl,
                    request_delay,
                    max_workers,
                )
                records.update(chunk_records)
                single_fetches += len(chunk_records)
                if on_batch is not None and chunk_records:
                    on_batch(chunk_records)
                progress.update(len(chunk))

        if single_fetches:
            logger.info(
                "Fetched %d AGU record(s) individually after incomplete batch responses",
                single_fetches,
            )
        return records

    def _fetch_records_individually(
        self,
        api_url: str,
        refs: List[str],
        timeout: float,
        verify_ssl: bool,
        request_delay: float,
        max_workers: int,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch records that a BatchRead response omitted, concurrently.

        Permanently unavailable records (HTTP 4xx other than 429) are skipped
        instead of aborting the whole download.

        Parameters
        ----------
        refs : list of str
            Record refs to fetch individually.
        max_workers : int
            Maximum concurrent requests.
        """

        def fetch_one(ref: str) -> Any:
            self._wait_for_request_slot(request_delay)
            try:
                record = self._fetch_json(f"{api_url}/{ref}", timeout, verify_ssl)
            except RuntimeError as exc:
                if self._is_unavailable_error(exc):
                    logger.debug("Skipping unavailable AGU record %s: %s", ref, exc)
                    return ref, None
                raise
            return ref, record if isinstance(record, dict) else None

        results = self._run_parallel(
            refs,
            fetch_one,
            max_workers=max_workers,
            desc="Fetching AGU records",
            unit="record",
            show_progress=False,
        )
        return {ref: record for ref, record in results if isinstance(record, dict)}

    @staticmethod
    def _is_unavailable_error(error: Exception) -> bool:
        """Return whether a fetch error is a permanent missing-record error."""
        message = str(error)
        return any(f"HTTP {status}" in message for status in (400, 403, 404, 410, 422))

    def _fetch_sessions(
        self,
        api_url: str,
        programs: List[Dict[str, Any]],
        timeout: float,
        verify_ssl: bool,
        request_delay: float,
        show_progress: Optional[bool] = None,
        max_workers: int = 1,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch all sessions through the program child-list endpoints.

        Program-level child lists return complete session records and avoid the
        partial BatchRead responses that m-anage produces under load. The
        independent program requests are fetched concurrently.

        Parameters
        ----------
        show_progress : bool, optional
            Whether to display a progress bar. ``None`` (default) shows the
            bar only when stderr is an interactive terminal.
        max_workers : int, optional
            Maximum concurrent requests (default: 1).
        """
        sessions: Dict[str, Dict[str, Any]] = {}
        fallback_refs: Set[str] = set()
        params = {"date": datetime.now().isoformat(timespec="seconds")}
        session_keys = ("ChildList_Session", "ChildList_SessionReadOnly", "ChildList_UnscheduledSession")
        program_requests = [
            (program, session_key) for program in programs for session_key in session_keys if program.get(session_key)
        ]

        def fetch(item: Any) -> Any:
            program, session_key = item
            parent = str(program.get("_url") or f"Program/{program.get('id')}")
            self._wait_for_request_slot(request_delay)
            data = self._fetch_json(f"{api_url}/{parent}/{session_key}", timeout, verify_ssl, params)
            return program, session_key, data

        results = self._run_parallel(
            program_requests,
            fetch,
            max_workers=max_workers,
            desc="Fetching AGU sessions",
            unit="request",
            show_progress=show_progress,
        )
        for program, session_key, data in results:
            refs = program[session_key]
            for record in self._records(data):
                ref = str(record.get("_url") or f"Session/{record.get('id')}")
                sessions[ref] = record
            fallback_refs.update(ref for ref in refs if ref not in sessions)

        if fallback_refs:
            sessions.update(
                self._fetch_batch_records(
                    api_url,
                    sorted(fallback_refs),
                    timeout,
                    verify_ssl,
                    request_delay,
                    show_progress,
                    max_workers=max_workers,
                )
            )
        return sessions

    def _fetch_papers(
        self,
        api_url: str,
        sessions: Dict[str, Dict[str, Any]],
        timeout: float,
        verify_ssl: bool,
        request_delay: float,
        show_progress: Optional[bool] = None,
        max_workers: int = 1,
        existing_papers: Optional[Dict[str, Dict[str, Any]]] = None,
        done_sessions: Optional[Set[str]] = None,
        checkpoint_dir: Optional[Path] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch full paper records through the session child-list endpoints.

        Session child lists return complete paper records, avoiding the partial
        BatchRead responses m-anage produces under load. Sessions without a
        direct paper list fall back to their paper slots. Independent session
        requests are fetched concurrently and checkpointed as they complete.

        Parameters
        ----------
        show_progress : bool, optional
            Whether to display a progress bar. ``None`` (default) shows the
            bar only when stderr is an interactive terminal.
        max_workers : int, optional
            Maximum concurrent session requests (default: 1).
        existing_papers : dict, optional
            Paper records restored from a checkpoint.
        done_sessions : set of str, optional
            Session refs already recorded in the checkpoint.
        checkpoint_dir : pathlib.Path, optional
            Directory used to persist completed session fetches.
        """
        papers: Dict[str, Dict[str, Any]] = dict(existing_papers or {})
        done_sessions = set(done_sessions or ())
        expected_refs: Set[str] = set()
        slot_refs: Set[str] = set()
        params = {"date": datetime.now().isoformat(timespec="seconds")}
        session_list = list(sessions.values())
        pending_sessions: List[Dict[str, Any]] = []
        paper_session_count = 0

        for session in session_list:
            refs = session.get("ChildList_Paper") or []
            if not refs:
                slot_refs.update(session.get("ChildList_PaperSlot") or [])
                continue
            expected_refs.update(refs)
            paper_session_count += 1
            session_ref = str(session.get("_url") or f"Session/{session.get('id')}")
            if session_ref not in done_sessions:
                pending_sessions.append(session)

        skipped_sessions = paper_session_count - len(pending_sessions)
        if skipped_sessions:
            logger.info(
                "Resuming AGU papers: %d of %d paper session(s) already checkpointed",
                skipped_sessions,
                paper_session_count,
            )

        def fetch(session: Dict[str, Any]) -> Any:
            session_ref = str(session.get("_url") or f"Session/{session.get('id')}")
            self._wait_for_request_slot(request_delay)
            data = self._fetch_json(f"{api_url}/{session_ref}/ChildList_Paper", timeout, verify_ssl, params)
            return session_ref, data

        results = self._run_parallel(
            pending_sessions,
            fetch,
            max_workers=max_workers,
            desc="Fetching AGU papers",
            unit="session",
            show_progress=show_progress,
            initial=skipped_sessions,
            total=paper_session_count,
        )

        new_papers: List[Dict[str, Any]] = []
        new_sessions: List[Dict[str, Any]] = []
        for session_ref, data in results:
            if data is None:
                continue
            for paper in self._records(data):
                ref = str(paper.get("_url") or f"Paper/{paper.get('id')}")
                papers[ref] = paper
                new_papers.append({"ref": ref, "data": paper})
            new_sessions.append({"ref": session_ref})
            if checkpoint_dir is not None and len(new_sessions) >= self._CHECKPOINT_FLUSH_INTERVAL:
                self._append_jsonl(checkpoint_dir / "papers.jsonl", new_papers)
                self._append_jsonl(checkpoint_dir / "paper_sessions.jsonl", new_sessions)
                new_papers = []
                new_sessions = []

        if checkpoint_dir is not None:
            self._append_jsonl(checkpoint_dir / "papers.jsonl", new_papers)
            self._append_jsonl(checkpoint_dir / "paper_sessions.jsonl", new_sessions)

        if slot_refs:
            slots = self._fetch_batch_records(
                api_url,
                sorted(slot_refs),
                timeout,
                verify_ssl,
                request_delay,
                show_progress,
                max_workers=max_workers,
            )
            for slot in slots.values():
                expected_refs.update(slot.get("ChildList_Entry") or [])

        missing = sorted(ref for ref in expected_refs if ref not in papers)
        if missing:

            def checkpoint_papers(records: Dict[str, Dict[str, Any]]) -> None:
                if checkpoint_dir is not None:
                    self._append_jsonl(
                        checkpoint_dir / "papers.jsonl",
                        [{"ref": ref, "data": paper} for ref, paper in records.items() if isinstance(paper, dict)],
                    )

            fetched = self._fetch_batch_records(
                api_url,
                missing,
                timeout,
                verify_ssl,
                request_delay,
                show_progress,
                max_workers=max_workers,
                on_batch=checkpoint_papers,
                initial=len(expected_refs) - len(missing),
                total=len(expected_refs),
            )
            papers.update(fetched)
        return papers

    def _fetch_people(
        self,
        api_url: str,
        person_refs: List[str],
        timeout: float,
        verify_ssl: bool,
        request_delay: float,
        show_progress: Optional[bool] = None,
        max_workers: int = 1,
        existing_people: Optional[Dict[str, Dict[str, Any]]] = None,
        done_letters: Optional[Set[str]] = None,
        checkpoint_dir: Optional[Path] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch people through the alphabetic people index.

        The people index returns every person for a letter in one request, so
        the complete people directory takes only a few concurrent requests.
        Any referenced people missing from the index are fetched through
        BatchRead.

        Parameters
        ----------
        show_progress : bool, optional
            Whether to display a progress bar. ``None`` (default) shows the
            bar only when stderr is an interactive terminal.
        max_workers : int, optional
            Maximum concurrent index requests (default: 1).
        existing_people : dict, optional
            People records restored from a checkpoint.
        done_letters : set of str, optional
            Index letters already recorded in the checkpoint.
        checkpoint_dir : pathlib.Path, optional
            Directory used to persist completed index letters.
        """
        if not person_refs:
            return {}

        people: Dict[str, Dict[str, Any]] = dict(existing_people or {})
        done_letters = set(done_letters or ())
        params = {"date": datetime.now().isoformat(timespec="seconds")}
        if all(ref in people for ref in person_refs):
            letters: List[str] = []
        else:
            letters = [letter for letter in self._PEOPLE_INDEX_LETTERS if letter not in done_letters]

        def fetch(letter: str) -> Any:
            self._wait_for_request_slot(request_delay)
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
                    return letter, []
                raise
            return letter, self._records(data)

        results = self._run_parallel(
            letters,
            fetch,
            max_workers=max_workers,
            desc="Fetching AGU people",
            unit="letter",
            show_progress=show_progress,
            initial=len(self._PEOPLE_INDEX_LETTERS) - len(letters),
            total=len(self._PEOPLE_INDEX_LETTERS),
        )
        for letter, records in results:
            new_people: List[Dict[str, Any]] = []
            for person in records:
                ref = str(person.get("_url") or f"Person/{person.get('id')}")
                people[ref] = person
                new_people.append({"ref": ref, "data": person})
            if checkpoint_dir is not None:
                self._append_jsonl(checkpoint_dir / "people.jsonl", new_people)
                self._append_jsonl(checkpoint_dir / "people_letters.jsonl", [{"letter": letter}])

        missing = sorted(ref for ref in person_refs if ref not in people)
        if missing:
            cached_people = len(person_refs) - len(missing)

            def checkpoint_people(records: Dict[str, Dict[str, Any]]) -> None:
                people.update(records)
                if checkpoint_dir is not None:
                    self._append_jsonl(
                        checkpoint_dir / "people.jsonl",
                        [{"ref": ref, "data": person} for ref, person in records.items() if isinstance(person, dict)],
                    )

            self._fetch_batch_records(
                api_url,
                missing,
                timeout,
                verify_ssl,
                request_delay,
                show_progress,
                max_workers=max_workers,
                on_batch=checkpoint_people,
                initial=cached_people,
                total=len(person_refs),
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

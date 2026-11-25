"""
Plugin base classes.

Defines :class:`DownloaderPlugin`, including year handling and the per-year
JSON cache helpers shared by all downloaders, plus the HTTP transport base
class :class:`HTTPDownloaderPlugin` and its retry/backoff helpers.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter

from abstracts_explorer.plugin.models import LightweightPaper, validate_lightweight_papers
from abstracts_explorer.plugin.persistence import save_json
from abstracts_explorer.rate_limiter import RateLimitCooldown
from abstracts_explorer.utils.time_utils import utcnow

logger = logging.getLogger(__name__)

_RETRYABLE_HTTP_STATUSES = frozenset({429, 500, 502, 503, 504})
_DEFAULT_MAX_RETRY_DELAY = 120.0
_IDEMPOTENT_METHODS = frozenset({"GET", "HEAD", "PUT", "DELETE", "OPTIONS", "TRACE"})


@dataclass
class DownloadOptions:
    """
    Common per-download options shared by downloader plugins.

    Plugins resolve this from the ``**kwargs`` passed to
    :meth:`DownloaderPlugin.download`, layering their own defaults on top of
    the fields below.
    """

    timeout: float = 30
    verify_ssl: bool = True
    request_delay: float = 0.0
    max_workers: int = 1
    page_delay: float = 0.0
    resume: bool = True
    progress: Optional[bool] = None
    openreview_token: Optional[str] = None
    openreview_email: Optional[str] = None
    openreview_password: Optional[str] = None

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any], defaults: Optional[Dict[str, Any]] = None) -> "DownloadOptions":
        """
        Build options from a download ``**kwargs`` mapping.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments passed to ``download``.
        defaults : dict, optional
            Plugin-specific defaults that take precedence over the dataclass
            defaults when a key is absent from *kwargs*.

        Returns
        -------
        DownloadOptions
            Resolved options.
        """
        defaults = defaults or {}
        values = {}
        for field_name in cls.__dataclass_fields__:
            values[field_name] = kwargs.get(field_name, defaults.get(field_name, getattr(cls, field_name)))
        return cls(**values)


class DownloaderPlugin(ABC):
    """
    Base class for all downloader plugins.

    Each plugin must implement the download method and provide metadata
    about its capabilities.

    Subclasses should set ``_start_year`` and override ``get_url(year)``
    to get automatic ``supported_years`` computation with a current-year
    availability check.

    Plugins return :class:`~abstracts_explorer.plugin.LightweightPaper`
    objects; see that model for the required and optional per-paper fields.
    The default :meth:`download` template handles year resolution, validation,
    caching, and saving, delegating the source-specific work to
    :meth:`_download_uncached`.
    """

    # Plugin metadata (should be overridden in subclasses)
    plugin_name: str = "base"
    plugin_description: str = "Base downloader plugin"
    requires_manual_input: bool = False
    _start_year: int = 0

    @property
    def supported_years(self) -> List[int]:
        """
        Dynamically computed supported years.

        Builds the pure range ``[_start_year, current_year]`` without any
        network access.  Current-year availability is probed only when a
        download actually needs a default year.

        Returns
        -------
        list of int
            Supported conference years.
        """
        if not hasattr(self, "_supported_years_cache"):
            if self._start_year > 0:
                current_year = datetime.now().year
                self._supported_years_cache: List[int] = list(range(self._start_year, current_year + 1))
            else:
                self._supported_years_cache = []
        return self._supported_years_cache

    @supported_years.setter
    def supported_years(self, value: List[int]) -> None:
        self._supported_years_cache = value

    def get_url(self, year: int) -> str:
        """
        Get the data URL for a specific year.

        Override in subclasses to enable current-year availability checking
        when resolving a default download year.

        Parameters
        ----------
        year : int
            Conference year

        Returns
        -------
        str
            URL used for downloading or probing availability.

        Raises
        ------
        NotImplementedError
            When the subclass does not provide an implementation.
        """
        raise NotImplementedError("Subclasses must implement get_url()")

    def _check_current_year_available(self, current_year: int) -> bool:
        """
        Check whether the data URL for *current_year* is already reachable.

        Sends a HEAD request to ``get_url(current_year)`` with a 3-second
        timeout.  Returns ``False`` when ``get_url`` is not implemented or
        the request fails.

        Parameters
        ----------
        current_year : int
            Year to probe.

        Returns
        -------
        bool
            ``True`` when the URL responds with HTTP 200.
        """
        try:
            url = self.get_url(current_year)
            response = requests.head(url, timeout=3, allow_redirects=True)
            return response.status_code == 200
        except (requests.RequestException, NotImplementedError):
            logger.debug(
                "%s: current-year availability check failed for year %d",
                self.plugin_name,
                current_year,
            )
            return False

    def download(
        self,
        year: Optional[int] = None,
        output_path: Optional[str] = None,
        force_download: bool = False,
        **kwargs: Any,
    ) -> List[LightweightPaper]:
        """
        Download papers from the data source.

        This template resolves the year and options, serves a cached JSON
        file when available, delegates the source-specific work to
        :meth:`_download_uncached`, and persists the result. Plugins with
        fundamentally different input sources (e.g. a manual JSON file) may
        override this method and call the shared helpers directly.

        Parameters
        ----------
        year : int, optional
            Year to download papers for. Defaults to the newest supported year.
        output_path : str, optional
            Path to save the downloaded data. When the file exists and
            *force_download* is false, it is loaded instead of downloading.
        force_download : bool
            Force a fresh download even when *output_path* exists.
        **kwargs : Any
            Plugin-specific options (see :class:`DownloadOptions`).

        Returns
        -------
        list of LightweightPaper
            List of validated paper objects ready for database insertion.
        """
        options = self.resolve_options(kwargs)
        year = self._resolve_year(year)
        self.validate_year(year)

        if output_path and not force_download:
            cached = self._load_papers_json(output_path, self._cache_source_label)
            if cached is not None and self._cache_matches_requested_year(cached, year):
                return self._handle_cached(cached, year, output_path, options)
            if cached is not None:
                logger.info(
                    "%s cache does not match requested year %d; downloading fresh data",
                    self.plugin_name,
                    year,
                )

        papers = self._download_uncached(year, output_path, options)
        self._save_papers_json(papers, output_path, self._save_label)
        return papers

    @property
    def _cache_source_label(self) -> str:
        """Label used when reporting cache load failures."""
        return getattr(self, "_SOURCE_LABEL", "data")

    @property
    def _save_label(self) -> str:
        """Label used when logging saved papers."""
        return self._canonical_conference_name

    @property
    def _canonical_conference_name(self) -> str:
        """Canonical conference name assigned to every paper from this plugin."""
        return getattr(self, "conference_name", self.plugin_name)

    def resolve_options(self, kwargs: Dict[str, Any], **defaults: Any) -> DownloadOptions:
        """
        Resolve :class:`DownloadOptions` from download kwargs.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments passed to :meth:`download`.
        **defaults : Any
            Plugin-specific defaults overriding the dataclass defaults.

        Returns
        -------
        DownloadOptions
            Resolved options.
        """
        base: Dict[str, Any] = {
            "timeout": getattr(self, "timeout", 30),
            "verify_ssl": getattr(self, "verify_ssl", True),
            "request_delay": getattr(self, "_DEFAULT_REQUEST_DELAY", 0.0),
            "max_workers": getattr(self, "_DEFAULT_MAX_WORKERS", 1),
            "page_delay": getattr(self, "_DEFAULT_PAGE_DELAY", 0.0),
            "resume": getattr(self, "_DEFAULT_RESUME", True),
            "progress": getattr(self, "_DEFAULT_PROGRESS", None),
        }
        base.update(defaults)
        return DownloadOptions.from_kwargs(kwargs, base)

    def _resolve_year(self, year: Optional[int]) -> int:
        """
        Resolve the download year, defaulting to the newest supported year.

        Parameters
        ----------
        year : int, optional
            Requested year.

        Returns
        -------
        int
            The requested year, or ``max(supported_years)`` when omitted.
        """
        if year is not None:
            return year
        years = self.supported_years
        current_year = datetime.now().year
        if current_year in years and not self._check_current_year_available(current_year):
            older = [year for year in years if year < current_year]
            if older:
                return max(older)
        return max(years, default=self._start_year)

    @staticmethod
    def _cache_matches_requested_year(cached: List[LightweightPaper], year: int) -> bool:
        """Return whether every cached paper belongs to *year*."""
        return bool(cached) and all(paper.year == year for paper in cached)

    def _handle_cached(
        self,
        cached: List[LightweightPaper],
        year: int,
        output_path: str,
        options: DownloadOptions,
    ) -> List[LightweightPaper]:
        """
        Apply plugin-specific handling to a successful cache hit.

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
            Papers to return to the caller.
        """
        canonical = self._canonical_conference_name
        normalized: List[LightweightPaper] = []
        for paper in cached:
            if paper.conference != canonical:
                logger.info(
                    "Normalizing cached conference %r to %r via %s",
                    paper.conference,
                    canonical,
                    self.plugin_name,
                )
                paper = paper.model_copy(update={"conference": canonical})
            normalized.append(paper)
        return normalized

    def _download_uncached(
        self,
        year: int,
        output_path: Optional[str],
        options: DownloadOptions,
    ) -> List[LightweightPaper]:
        """
        Perform the source-specific download.

        Parameters
        ----------
        year : int
            Year to download.
        output_path : str, optional
            Cache path (for plugins that track sidecars/checkpoints).
        options : DownloadOptions
            Resolved download options.

        Returns
        -------
        list of LightweightPaper
            Downloaded and validated papers.
        """
        raise NotImplementedError("Subclasses must implement _download_uncached()")

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get plugin metadata.

        Returns
        -------
        dict
            Plugin metadata including name, description, supported years, etc.
        """
        pass

    def validate_year(self, year: Optional[int]) -> None:
        """
        Validate that the requested year is supported.

        Parameters
        ----------
        year : int or None
            Year to validate

        Raises
        ------
        ValueError
            If year is not supported by this plugin
        """
        if year is not None and self.supported_years and year not in self.supported_years:
            raise ValueError(
                f"Year {year} not supported by {self.plugin_name}. Supported years: {self.supported_years}"
            )

    def _save_papers_json(self, papers: List["LightweightPaper"], output_path: Optional[str], label: str) -> None:
        """
        Save validated papers to a per-year lightweight JSON file.

        Skips the write entirely when *papers* is empty so a year with no
        abstracts does not leave a stray ``[]`` file in the data directory.

        Parameters
        ----------
        papers : list of LightweightPaper
            Papers to persist.
        output_path : str or None
            Destination path. When ``None``, nothing is written.
        label : str
            Conference/source name used only in log messages.
        """
        if output_path and papers:
            save_json(Path(output_path), [paper.model_dump() for paper in papers])
            logger.info("Saved %d %s papers to %s", len(papers), label, output_path)
        elif output_path:
            logger.info("No papers found; skipping save of %s", output_path)

    def _load_papers_json(self, output_path: Optional[str], source_label: str) -> Optional[List["LightweightPaper"]]:
        """
        Load cached lightweight papers from a previously written JSON file.

        Returns ``None`` when there is nothing to load (no path given, file
        absent, or the file cannot be parsed), signalling the caller to fall
        through to a fresh download.

        Parameters
        ----------
        output_path : str or None
            Path to an existing lightweight JSON file.
        source_label : str
            Where the fresh download comes from, used only in the fallback
            warning message.

        Returns
        -------
        list of LightweightPaper or None
        """
        if output_path and Path(output_path).exists():
            logger.info("Loading existing %s from: %s", source_label, output_path)
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                papers = validate_lightweight_papers(cached_data)
                logger.info("Successfully loaded %d papers from local file", len(papers))
                return papers
            except (json.JSONDecodeError, Exception) as e:
                logger.warning("Failed to load local file: %s. Downloading from %s...", e, source_label)
        return None


def _http_status_code(exc: Exception) -> Optional[int]:
    """Return the HTTP status code attached to an exception, if any."""
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _compute_retry_delay(
    exc: Exception,
    attempt: int,
    retry_base_delay: float,
    max_retry_delay: Optional[float],
) -> float:
    """Compute a retry delay, honouring ``Retry-After`` when present."""
    response = getattr(exc, "response", None)
    retry_after = getattr(response, "headers", {}).get("Retry-After")
    delay = retry_base_delay * (2**attempt)
    if retry_after:
        try:
            delay = float(retry_after)
        except (TypeError, ValueError):
            try:
                retry_at = parsedate_to_datetime(str(retry_after))
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                delay = (retry_at - utcnow()).total_seconds()
            except (TypeError, ValueError, OverflowError):
                pass
    if delay <= 0 and retry_after:
        delay = retry_base_delay * (2**attempt)
    cap = max_retry_delay if max_retry_delay is not None else _DEFAULT_MAX_RETRY_DELAY
    return max(0.0, min(delay, cap))


def fetch_with_retries(
    session: requests.Session,
    url: str,
    *,
    timeout: float,
    verify: bool = True,
    request_delay: float = 0.0,
    max_attempts: int = 3,
    retryable_statuses: frozenset[int] = _RETRYABLE_HTTP_STATUSES,
    params: Optional[Dict[str, Any]] = None,
    retry_base_delay: float = 1.0,
    max_retry_delay: Optional[float] = _DEFAULT_MAX_RETRY_DELAY,
    cooldown: Optional[RateLimitCooldown] = None,
    method: str = "GET",
    json_body: Optional[Any] = None,
    headers: Optional[Dict[str, str]] = None,
    retry_non_idempotent: bool = False,
) -> requests.Response:
    """
    Fetch a URL, retrying transient failures with exponential backoff.

    Connection errors and timeouts are retried, as are HTTP responses in
    ``retryable_statuses``. Permanent failures, such as HTTP 404, are raised
    immediately without retrying.

    Parameters
    ----------
    session : requests.Session
        Session used to issue the request.
    url : str
        URL to fetch.
    timeout : float
        HTTP timeout in seconds.
    verify : bool, optional
        Verify TLS certificates (default: True).
    request_delay : float, optional
        Delay before every attempt in seconds (default: 0).
    max_attempts : int, optional
        Maximum number of attempts, including the first (default: 3).
    retryable_statuses : frozenset of int, optional
        HTTP statuses that should be retried.
    params : dict, optional
        Query parameters forwarded to ``session.get`` when provided.
    retry_base_delay : float, optional
        Base delay for exponential backoff in seconds (default: 1).
    max_retry_delay : float, optional
        Maximum retry delay in seconds (default: 120 seconds).
    cooldown : RateLimitCooldown, optional
        Shared cooldown applied after HTTP 429 responses.
    method : str, optional
        HTTP method to use (default: ``"GET"``).
    json_body : Any, optional
        JSON body for POST requests.
    headers : dict, optional
        Additional request headers.
    retry_non_idempotent : bool, optional
        Retry non-idempotent methods such as ``POST`` (default: False).
        Only enable this for requests that are safe to repeat, such as
        read-only search queries.

    Returns
    -------
    requests.Response
        Successful response.

    Raises
    ------
    requests.RequestException
        The last request failure after all retry attempts are exhausted.
    """
    last_error: Optional[requests.RequestException] = None
    method_is_idempotent = method.upper() in _IDEMPOTENT_METHODS
    for attempt in range(max_attempts):
        if cooldown is not None:
            cooldown.wait()
        if request_delay > 0:
            time.sleep(request_delay)
        request_kwargs: Dict[str, Any] = {"timeout": timeout, "verify": verify}
        if params is not None:
            request_kwargs["params"] = params
        if headers is not None:
            request_kwargs["headers"] = headers
        try:
            upper_method = method.upper()
            if upper_method == "GET":
                response = session.get(url, **request_kwargs)
            elif upper_method == "POST":
                response = session.post(url, json=json_body, **request_kwargs)
            else:
                response = session.request(method, url, json=json_body, **request_kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            status = _http_status_code(exc)
            retryable = (
                status in retryable_statuses
                if status is not None
                else isinstance(exc, (requests.ConnectionError, requests.Timeout))
            )
            if not method_is_idempotent and not retry_non_idempotent:
                retryable = False
            if not retryable or attempt == max_attempts - 1:
                break
            delay = _compute_retry_delay(exc, attempt, retry_base_delay, max_retry_delay)
            logger.warning("Retrying request to %s in %.0fs after: %s", url, delay, exc)
            if status == 429 and cooldown is not None:
                cooldown.defer(delay)
            else:
                time.sleep(delay)
    assert last_error is not None
    raise last_error


class HTTPDownloaderPlugin(DownloaderPlugin):
    """Downloader base class with shared HTTP transport handling."""

    _USER_AGENT = "abstracts-explorer"
    _ACCEPT: Optional[str] = None
    _EXTRA_HEADERS: Dict[str, str] = {}
    _SOURCE_LABEL = "data"
    _MAX_FETCH_ATTEMPTS = 3
    _RETRY_BASE_DELAY = 1.0
    _MAX_RETRY_DELAY: Optional[float] = _DEFAULT_MAX_RETRY_DELAY
    _RETRYABLE_STATUSES = _RETRYABLE_HTTP_STATUSES
    _RETRY_NON_IDEMPOTENT: bool = False

    def __init__(self, timeout: float = 30, verify_ssl: bool = True) -> None:
        """
        Initialize the shared HTTP session.

        Parameters
        ----------
        timeout : int, optional
            HTTP timeout in seconds (default: 30).
        verify_ssl : bool, optional
            Verify TLS certificates (default: True).
        """
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self._rate_limit = RateLimitCooldown()
        self._session = requests.Session()
        headers = {"User-Agent": self._USER_AGENT, **self._EXTRA_HEADERS}
        if self._ACCEPT:
            headers["Accept"] = self._ACCEPT
        self._session.headers.update(headers)

    def configure_pool(self, max_workers: int) -> None:
        """
        Size the shared connection pool for concurrent requests.

        Parameters
        ----------
        max_workers : int
            Number of concurrent workers the pool must accommodate.
        """
        pool_size = max(32, max_workers + 5)
        adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    def _request(
        self,
        url: str,
        timeout: float,
        verify_ssl: bool,
        params: Optional[Dict[str, Any]] = None,
        request_delay: float = 0.0,
        method: str = "GET",
        json_body: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        max_attempts: Optional[int] = None,
    ) -> requests.Response:
        """Fetch a URL through the shared retry and cooldown handling."""
        try:
            return fetch_with_retries(
                self._session,
                url,
                timeout=timeout,
                verify=verify_ssl,
                request_delay=request_delay,
                max_attempts=self._MAX_FETCH_ATTEMPTS if max_attempts is None else max_attempts,
                retryable_statuses=self._RETRYABLE_STATUSES,
                params=params,
                retry_base_delay=self._RETRY_BASE_DELAY,
                max_retry_delay=self._MAX_RETRY_DELAY,
                cooldown=self._rate_limit,
                method=method,
                json_body=json_body,
                headers=headers,
                retry_non_idempotent=self._RETRY_NON_IDEMPOTENT,
            )
        except requests.RequestException as exc:
            status = _http_status_code(exc)
            detail = f" (HTTP {status})" if status is not None else ""
            raise RuntimeError(f"Failed to fetch {self._SOURCE_LABEL} from {url}{detail}: {exc}") from exc

    def _fetch_json(
        self,
        url: str,
        timeout: float,
        verify_ssl: bool,
        params: Optional[Dict[str, Any]] = None,
        request_delay: float = 0.0,
        method: str = "GET",
        json_body: Optional[Any] = None,
        max_attempts: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        """Fetch and decode a JSON response."""
        response = self._request(
            url,
            timeout,
            verify_ssl,
            params=params,
            request_delay=request_delay,
            method=method,
            json_body=json_body,
            max_attempts=max_attempts,
            headers=headers,
        )
        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError(f"Failed to decode {self._SOURCE_LABEL} JSON from {url}: {exc}") from exc

    def _fetch_soup(
        self,
        url: str,
        timeout: float,
        verify_ssl: bool,
        request_delay: float = 0.0,
        max_attempts: Optional[int] = None,
    ) -> BeautifulSoup:
        """Fetch a response and parse it as HTML."""
        response = self._request(
            url,
            timeout,
            verify_ssl,
            request_delay=request_delay,
            max_attempts=max_attempts,
        )
        return BeautifulSoup(response.text, "html.parser")

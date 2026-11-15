"""
HTTP transport helpers for downloader plugins.

Provides the shared retry/backoff/cooldown logic and the
:class:`HTTPDownloaderPlugin` base class that most conference downloaders use.
"""

import logging
import time
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Optional

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter

from abstracts_explorer.plugin.base import LightweightDownloaderPlugin
from abstracts_explorer.rate_limiter import RateLimitCooldown
from abstracts_explorer.utils.time_utils import utcnow

logger = logging.getLogger(__name__)

_RETRYABLE_HTTP_STATUSES = frozenset({429, 500, 502, 503, 504})


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
    if max_retry_delay is not None:
        delay = min(delay, max_retry_delay)
    return max(0.0, delay)


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
    max_retry_delay: Optional[float] = None,
    cooldown: Optional[RateLimitCooldown] = None,
    method: str = "GET",
    json_body: Optional[Any] = None,
    headers: Optional[Dict[str, str]] = None,
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
        Maximum retry delay in seconds (default: no cap).
    cooldown : RateLimitCooldown, optional
        Shared cooldown applied after HTTP 429 responses.
    method : str, optional
        HTTP method to use (default: ``"GET"``).
    json_body : Any, optional
        JSON body for POST requests.
    headers : dict, optional
        Additional request headers.

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


class HTTPDownloaderPlugin(LightweightDownloaderPlugin):
    """Lightweight plugin base class with shared HTTP transport handling."""

    _USER_AGENT = "abstracts-explorer"
    _ACCEPT: Optional[str] = None
    _EXTRA_HEADERS: Dict[str, str] = {}
    _SOURCE_LABEL = "data"
    _MAX_FETCH_ATTEMPTS = 3
    _RETRY_BASE_DELAY = 1.0
    _MAX_RETRY_DELAY: Optional[float] = None
    _RETRYABLE_STATUSES = _RETRYABLE_HTTP_STATUSES

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

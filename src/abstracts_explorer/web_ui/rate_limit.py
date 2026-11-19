"""Per-client rate limiting for expensive web UI endpoints."""

import math
import threading
import time
from collections import OrderedDict
from typing import Optional, Tuple

from abstracts_explorer.config import get_config
from abstracts_explorer.rate_limiter import TokenBucketRateLimiter
from abstracts_explorer.web_ui.api_utils import ApiError

_MAX_CLIENTS = 4096

_client_limiter: Optional["ClientRateLimiter"] = None
_client_limiter_lock = threading.Lock()


class ClientRateLimiter:
    """
    Thread-safe per-client token bucket limiter with bounded client storage.

    Parameters
    ----------
    requests_per_minute : int
        Maximum requests allowed per client per minute.
    max_clients : int, optional
        Maximum number of client buckets retained in memory.
    """

    def __init__(self, requests_per_minute: int, max_clients: int = _MAX_CLIENTS) -> None:
        self._requests_per_minute = requests_per_minute
        self._max_clients = max_clients
        self._buckets: "OrderedDict[str, Tuple[TokenBucketRateLimiter, float]]" = OrderedDict()
        self._lock = threading.Lock()

    @property
    def requests_per_minute(self) -> int:
        """Return the configured requests per minute."""
        return self._requests_per_minute

    def check(self, client_id: str) -> Tuple[bool, float]:
        """
        Consume one token for *client_id*.

        Parameters
        ----------
        client_id : str
            Stable identifier for the client (for example, ``request.remote_addr``).

        Returns
        -------
        tuple
            ``(allowed, retry_after)`` where ``retry_after`` is the number of
            seconds until the next token is available.
        """
        now = time.monotonic()
        with self._lock:
            entry = self._buckets.get(client_id)
            if entry is None:
                while len(self._buckets) >= self._max_clients:
                    self._buckets.popitem(last=False)
                entry = (TokenBucketRateLimiter(requests_per_minute=self._requests_per_minute), now)

            bucket, _last_seen = entry
            allowed = bucket.try_acquire()
            retry_after = 0.0 if allowed else bucket.time_until_available()
            self._buckets[client_id] = (bucket, now)
            self._buckets.move_to_end(client_id)
            return allowed, retry_after

    def reset(self) -> None:
        """Clear all client buckets."""
        with self._lock:
            self._buckets.clear()


def get_client_rate_limiter() -> ClientRateLimiter:
    """Return the process-level client rate limiter."""
    global _client_limiter
    requests_per_minute = get_config().web_rate_limit_per_minute
    with _client_limiter_lock:
        if _client_limiter is None or _client_limiter.requests_per_minute != requests_per_minute:
            _client_limiter = ClientRateLimiter(requests_per_minute=requests_per_minute)
        return _client_limiter


def reset_client_rate_limiter() -> None:
    """Reset the process-level client rate limiter (used by tests)."""
    global _client_limiter
    with _client_limiter_lock:
        _client_limiter = None


def enforce_client_rate_limit(client_id: str) -> None:
    """
    Enforce the per-client rate limit for expensive endpoints.

    Parameters
    ----------
    client_id : str
        Stable identifier for the client.

    Raises
    ------
    ApiError
        With HTTP 429 when the client has exhausted its quota.
    """
    if get_config().web_rate_limit_per_minute <= 0:
        return
    allowed, retry_after = get_client_rate_limiter().check(client_id)
    if allowed:
        return
    retry_seconds = max(1, math.ceil(retry_after)) if math.isfinite(retry_after) else 60
    raise ApiError(
        "Rate limit exceeded. Please retry later.",
        429,
        headers={"Retry-After": str(retry_seconds)},
    )

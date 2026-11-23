"""
Thread-safe and async-compatible token bucket rate limiter.

Provides accurate global rate limiting that works correctly with
Waitress's multi-threaded WSGI server and async agents (Pydantic AI).

``RateLimitedTransport`` adapts the shared limiter to ``httpx2`` clients so
sync and async OpenAI clients are throttled transparently.
"""

import asyncio
import logging
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

import httpx2

logger = logging.getLogger(__name__)

_global_rate_limiter: Optional["TokenBucketRateLimiter"] = None
_global_cooldown: Optional["RateLimitCooldown"] = None


class TokenBucketRateLimiter:
    """
    Thread-safe token bucket rate limiter.

    Tokens are added at a constant rate up to a maximum bucket capacity.
    Each request consumes one token. If no tokens available, request waits.

    The rate limiter is designed to be shared across multiple threads,
    providing accurate global rate limiting rather than per-thread limits.
    """

    def __init__(self, requests_per_minute: int = 60, burst: Optional[int] = None) -> None:
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests allowed per minute.
            burst: Maximum number of tokens that can accumulate. Defaults to
                ``requests_per_minute``. Set to ``1`` for smooth pacing
                without an initial burst.
        """
        if requests_per_minute < 0:
            raise ValueError("requests_per_minute must be non-negative")

        if burst is None:
            burst = requests_per_minute
        if burst < 0:
            raise ValueError("burst must be non-negative")
        if requests_per_minute > 0 and burst == 0:
            burst = 1

        self._rpm = requests_per_minute
        self._rate = requests_per_minute / 60.0  # tokens per second
        self._burst = burst
        self._max_tokens = float(burst)
        self._tokens = self._max_tokens
        self._last_update = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> None:
        """
        Acquire tokens from the bucket. Blocks if not enough tokens available.

        Args:
            tokens: Number of tokens to acquire (default: 1).
        """
        with self._lock:
            self._refill()

            while self._tokens < tokens:
                needed = tokens - self._tokens
                wait_time = needed / self._rate

                self._lock.release()
                try:
                    time.sleep(wait_time)
                finally:
                    self._lock.acquire()

                self._refill()

            self._tokens -= tokens

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """
        Try to acquire tokens without blocking.

        Parameters
        ----------
        tokens : float, optional
            Number of tokens to acquire (default: 1).

        Returns
        -------
        bool
            ``True`` when the tokens were acquired, ``False`` otherwise.
        """
        with self._lock:
            self._refill()
            if self._tokens < tokens:
                return False
            self._tokens -= tokens
            return True

    def time_until_available(self, tokens: float = 1.0) -> float:
        """
        Return the seconds until the requested tokens are available.

        Parameters
        ----------
        tokens : float, optional
            Number of tokens required (default: 1).

        Returns
        -------
        float
            Non-negative wait time in seconds.
        """
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                return 0.0
            if self._rate <= 0:
                return float("inf")
            return (tokens - self._tokens) / self._rate

    async def async_acquire(self, tokens: float = 1.0) -> None:
        """
        Acquire tokens from the bucket asynchronously.

        Args:
            tokens: Number of tokens to acquire (default: 1).
        """
        with self._lock:
            self._refill()

            while self._tokens < tokens:
                needed = tokens - self._tokens
                wait_time = needed / self._rate

                self._lock.release()
                try:
                    await asyncio.sleep(wait_time)
                finally:
                    self._lock.acquire()

                self._refill()

            self._tokens -= tokens

    def _refill(self) -> None:
        """
        Refill tokens based on elapsed time.

        Must be called with lock held.
        """
        now = time.monotonic()
        elapsed = now - self._last_update
        self._tokens = min(self._max_tokens, self._tokens + elapsed * self._rate)
        self._last_update = now

    @property
    def requests_per_minute(self) -> int:
        """Return the configured requests per minute."""
        return self._rpm

    @property
    def burst(self) -> int:
        """Return the configured burst capacity."""
        return self._burst

    def reset(self) -> None:
        """Reset the rate limiter to full capacity."""
        with self._lock:
            self._tokens = self._max_tokens
            self._last_update = time.monotonic()


class RateLimitedTransport(httpx2.BaseTransport, httpx2.AsyncBaseTransport):
    """
    An httpx2 transport that enforces a maximum requests-per-minute rate.

    Supports both synchronous and asynchronous clients.  Wraps an existing
    transport and acquires tokens from a shared token bucket rate limiter before
    forwarding requests.  This provides thread-safe, global rate limiting that
    works correctly under Waitress's multi-threaded WSGI server as well as with
    Pydantic AI's async OpenAI provider.

    Parameters
    ----------
    transport : httpx2.BaseTransport or httpx2.AsyncBaseTransport
        The underlying transport to delegate requests to.
    requests_per_minute : int
        Maximum number of requests per minute. Must be > 0.
    burst : int, optional
        Maximum number of tokens that can accumulate (default: 1). A burst of
        one paces requests smoothly so per-minute provider quotas are not
        exhausted by an initial burst.
    """

    def __init__(
        self,
        transport: httpx2.BaseTransport | httpx2.AsyncBaseTransport,
        requests_per_minute: int,
        burst: int = 1,
    ) -> None:
        self._transport = transport

        global_rate_limiter = get_global_rate_limiter()
        if global_rate_limiter is None:
            global_rate_limiter = TokenBucketRateLimiter(
                requests_per_minute=requests_per_minute,
                burst=burst,
            )
            set_global_rate_limiter(global_rate_limiter)
        elif global_rate_limiter.requests_per_minute != requests_per_minute:
            logger.warning(
                "Reusing existing global rate limiter with %d requests/minute; "
                "requested rate of %d requests/minute is ignored",
                global_rate_limiter.requests_per_minute,
                requests_per_minute,
            )

        self._rate_limiter = global_rate_limiter

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        """Send *request* after acquiring a rate-limit token."""
        if not isinstance(self._transport, httpx2.BaseTransport):
            raise TypeError("Cannot use a synchronous client with an async transport.")
        self._rate_limiter.acquire(tokens=1.0)
        return self._transport.handle_request(request)

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        """Send *request* after acquiring a rate-limit token asynchronously."""
        if not isinstance(self._transport, httpx2.AsyncBaseTransport):
            raise TypeError("Cannot use an async client with a synchronous transport.")
        await self._rate_limiter.async_acquire(tokens=1.0)
        return await self._transport.handle_async_request(request)

    def close(self) -> None:
        """Close the underlying sync transport."""
        if isinstance(self._transport, httpx2.BaseTransport):
            self._transport.close()

    async def aclose(self) -> None:
        """Close the underlying async transport."""
        if isinstance(self._transport, httpx2.AsyncBaseTransport):
            await self._transport.aclose()


def get_global_rate_limiter() -> Optional["TokenBucketRateLimiter"]:
    """Get or create the global shared rate limiter instance."""
    global _global_rate_limiter
    return _global_rate_limiter


def set_global_rate_limiter(
    limiter: Optional["TokenBucketRateLimiter"],
) -> None:
    """Set or reset the global shared rate limiter instance."""
    global _global_rate_limiter
    _global_rate_limiter = limiter


class RateLimitCooldown:
    """Thread-safe shared cooldown window after a rate-limit response."""

    def __init__(self) -> None:
        self._until = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        """Sleep until the current cooldown expires."""
        with self._lock:
            remaining = self._until - time.monotonic()
            if remaining <= 0:
                self._until = 0.0
                return
        time.sleep(remaining)
        with self._lock:
            if self._until <= time.monotonic():
                self._until = 0.0

    async def async_wait(self) -> None:
        """Sleep until the current cooldown expires without blocking the event loop."""
        with self._lock:
            remaining = self._until - time.monotonic()
            if remaining <= 0:
                self._until = 0.0
                return
        await asyncio.sleep(remaining)
        with self._lock:
            if self._until <= time.monotonic():
                self._until = 0.0

    def defer(self, delay: float) -> None:
        """Extend the cooldown by at least *delay* seconds."""
        with self._lock:
            self._until = max(self._until, time.monotonic() + max(0.0, delay))


def parse_retry_after_seconds(headers: Optional[httpx2.Headers], *, max_wait: float = 120.0) -> Optional[float]:
    """
    Parse a ``Retry-After`` header into a non-negative delay in seconds.

    ``retry-after-ms`` takes precedence over ``Retry-After`` (matching the
    OpenAI SDK). Numeric values are treated as seconds; HTTP-date values are
    converted to the remaining time. The result is capped at *max_wait*.

    Parameters
    ----------
    headers : httpx2.Headers or None
        Response headers.
    max_wait : float, optional
        Upper bound for the returned delay in seconds (default 120).

    Returns
    -------
    float or None
        Delay in seconds, or ``None`` when no valid header is present.
    """
    if not headers:
        return None

    for header, divisor in (("retry-after-ms", 1000.0), ("retry-after", 1.0)):
        value = headers.get(header)
        if value is None:
            continue
        try:
            delay = float(value) / divisor
        except (TypeError, ValueError):
            continue
        return max(0.0, min(delay, max_wait))

    value = headers.get("retry-after")
    if not value:
        return None
    try:
        retry_at = parsedate_to_datetime(str(value))
    except (TypeError, ValueError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    return max(0.0, min((retry_at - datetime.now(timezone.utc)).total_seconds(), max_wait))


class CooldownAwareTransport(httpx2.BaseTransport, httpx2.AsyncBaseTransport):
    """
    Transport that pauses all LLM requests after a rate-limit response.

    Wraps another transport (typically :class:`RateLimitedTransport`) and:

    - waits for the shared :class:`RateLimitCooldown` before sending;
    - on ``429``, parses ``Retry-After`` / ``retry-after-ms`` and extends the
      shared cooldown so parallel requests pause too.

    Retrying itself is left to the OpenAI SDK, which already implements
    retryable statuses, exponential backoff, ``Retry-After`` and attempt limits.

    Parameters
    ----------
    transport : httpx2.BaseTransport or httpx2.AsyncBaseTransport
        Underlying transport.
    rate_limit_delay : float, optional
        Cooldown used for a ``429`` without a usable ``Retry-After`` header
        (default 60 seconds).
    max_retry_after : float, optional
        Upper bound for a cooldown derived from ``Retry-After`` (default 120
        seconds).
    cooldown : RateLimitCooldown, optional
        Shared cooldown instance. Defaults to the process-global cooldown.
    """

    def __init__(
        self,
        transport: httpx2.BaseTransport | httpx2.AsyncBaseTransport,
        *,
        rate_limit_delay: float = 60.0,
        max_retry_after: float = 120.0,
        cooldown: Optional[RateLimitCooldown] = None,
    ) -> None:
        self._transport = transport
        self._rate_limit_delay = max(0.0, rate_limit_delay)
        self._max_retry_after = max(0.0, max_retry_after)
        self._cooldown = cooldown or get_global_cooldown()

    def _handle_response(self, response: httpx2.Response) -> None:
        """Extend the shared cooldown when *response* is a rate-limit response."""
        if response.status_code != 429:
            return
        delay = parse_retry_after_seconds(response.headers, max_wait=self._max_retry_after)
        if delay is None:
            delay = self._rate_limit_delay
        self._cooldown.defer(delay)
        logger.warning("LLM request rate-limited; pausing all LLM requests for %.1fs", delay)

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        """Send *request* after waiting for any active cooldown."""
        if not isinstance(self._transport, httpx2.BaseTransport):
            raise TypeError("Cannot use a synchronous client with an async transport.")
        self._cooldown.wait()
        response = self._transport.handle_request(request)
        self._handle_response(response)
        return response

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        """Send *request* asynchronously after waiting for any active cooldown."""
        if not isinstance(self._transport, httpx2.AsyncBaseTransport):
            raise TypeError("Cannot use an async client with a synchronous transport.")
        await self._cooldown.async_wait()
        response = await self._transport.handle_async_request(request)
        self._handle_response(response)
        return response

    def close(self) -> None:
        """Close the underlying sync transport."""
        if isinstance(self._transport, httpx2.BaseTransport):
            self._transport.close()

    async def aclose(self) -> None:
        """Close the underlying async transport."""
        if isinstance(self._transport, httpx2.AsyncBaseTransport):
            await self._transport.aclose()


def get_global_cooldown() -> RateLimitCooldown:
    """Return the process-global cooldown shared by all LLM transports."""
    global _global_cooldown
    if _global_cooldown is None:
        _global_cooldown = RateLimitCooldown()
    return _global_cooldown


def set_global_cooldown(cooldown: Optional[RateLimitCooldown]) -> None:
    """Replace or reset the process-global LLM rate-limit cooldown."""
    global _global_cooldown
    _global_cooldown = cooldown or RateLimitCooldown()

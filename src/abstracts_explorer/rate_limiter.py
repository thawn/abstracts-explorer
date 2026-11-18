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
from typing import Optional

import httpx2

logger = logging.getLogger(__name__)

_global_rate_limiter: Optional["TokenBucketRateLimiter"] = None


class TokenBucketRateLimiter:
    """
    Thread-safe token bucket rate limiter.

    Tokens are added at a constant rate up to a maximum bucket capacity.
    Each request consumes one token. If no tokens available, request waits.

    The rate limiter is designed to be shared across multiple threads,
    providing accurate global rate limiting rather than per-thread limits.
    """

    def __init__(self, requests_per_minute: int = 60) -> None:
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests allowed per minute.
        """
        if requests_per_minute < 0:
            raise ValueError("requests_per_minute must be non-negative")

        self._rpm = requests_per_minute
        self._rate = requests_per_minute / 60.0  # tokens per second
        self._max_tokens = float(requests_per_minute)
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
    """

    def __init__(
        self,
        transport: httpx2.BaseTransport | httpx2.AsyncBaseTransport,
        requests_per_minute: int,
    ) -> None:
        self._transport = transport

        global_rate_limiter = get_global_rate_limiter()
        if global_rate_limiter is None:
            global_rate_limiter = TokenBucketRateLimiter(requests_per_minute=requests_per_minute)
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

    def defer(self, delay: float) -> None:
        """Extend the cooldown by at least *delay* seconds."""
        with self._lock:
            self._until = max(self._until, time.monotonic() + max(0.0, delay))

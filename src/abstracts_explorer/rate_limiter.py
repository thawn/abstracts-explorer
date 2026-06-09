"""
Thread-safe and async-compatible token bucket rate limiter.

Provides accurate global rate limiting that works correctly with
Waitress's multi-threaded WSGI server and async agents (Pydantic AI).
"""

import asyncio
import threading
import time
from typing import Optional

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

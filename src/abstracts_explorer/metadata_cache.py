"""
Thread-safe TTL cache
=====================

Small in-memory cache used for database metadata queries.  The cache is
process-local; separate processes do not share entries.
"""

from __future__ import annotations

import copy
import threading
import time
from typing import Any, Callable, Hashable, Optional, Tuple


class MetadataCache:
    """
    A thread-safe, time-to-live cache for metadata query results.

    Parameters
    ----------
    clock : callable, optional
        Monotonic clock used for expiry checks.  Defaults to
        :func:`time.monotonic`.

    Examples
    --------
    >>> cache = MetadataCache()
    >>> cache.set("key", {"years": [2025]}, ttl_seconds=60)
    >>> cache.get("key", ttl_seconds=60)
    {'years': [2025]}
    >>> cache.get("key", ttl_seconds=60) is cache.get("key", ttl_seconds=60)
    False
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic, max_entries: int = 1000):
        self._entries: dict[Hashable, Tuple[float, Any]] = {}
        self._lock = threading.RLock()
        self._clock = clock
        self._max_entries = max(1, max_entries)

    def get(self, key: Hashable, ttl_seconds: int) -> Optional[Any]:
        """
        Return a cached value, or ``None`` when absent or expired.

        Parameters
        ----------
        key : hashable
            Cache key.
        ttl_seconds : int
            Maximum entry age.  Values less than or equal to zero disable the
            cache.

        Returns
        -------
        Any or None
            A deep copy of the cached value, or ``None``.
        """
        if ttl_seconds <= 0:
            return None

        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None

            expires_at, value = entry
            if self._clock() >= expires_at:
                del self._entries[key]
                return None

            return copy.deepcopy(value)

    def set(self, key: Hashable, value: Any, ttl_seconds: int) -> None:
        """
        Store a value in the cache.

        Parameters
        ----------
        key : hashable
            Cache key.
        value : Any
            Value to cache.
        ttl_seconds : int
            Entry lifetime.  Values less than or equal to zero disable the
            cache.
        """
        if ttl_seconds <= 0:
            return

        with self._lock:
            now = self._clock()
            if len(self._entries) >= self._max_entries:
                expired = [entry_key for entry_key, (expires_at, _) in self._entries.items() if now >= expires_at]
                for entry_key in expired:
                    del self._entries[entry_key]
            while len(self._entries) >= self._max_entries:
                oldest_key = next(iter(self._entries))
                del self._entries[oldest_key]
            self._entries[key] = (now + ttl_seconds, copy.deepcopy(value))

    def invalidate_database(self, database_url: str) -> None:
        """
        Remove all entries whose key starts with *database_url*.

        Parameters
        ----------
        database_url : str
            Database URL used as the first element of cache keys.
        """
        prefix = (database_url,)
        with self._lock:
            keys = [key for key in self._entries if isinstance(key, tuple) and key[:1] == prefix]
            for key in keys:
                del self._entries[key]

    def clear(self) -> None:
        """Remove all cache entries."""
        with self._lock:
            self._entries.clear()

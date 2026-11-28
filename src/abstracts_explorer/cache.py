"""
Process-global, thread-safe cache infrastructure.

The caches in this module are shared by all threads in a process.  Cache
locks are held only for short bookkeeping operations; expensive database,
ChromaDB, or LLM work always happens outside the lock.
"""

from __future__ import annotations

import copy
import json
import pickle
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Hashable, Optional, Tuple

from abstracts_explorer.config import get_config

DEFAULT_MAX_ENTRIES = 1024
DEFAULT_MAX_ENTRY_BYTES = 1024 * 1024
DEFAULT_SCOPE_UID_MAX_BYTES = 64 * 1024 * 1024
MAX_CACHED_SCOPE_UIDS = 50_000

_MISSING = object()
_UNSET = object()


def normalize_query_text(text: str) -> str:
    """
    Normalize query text for cache keys.

    Parameters
    ----------
    text : str
        Query text.

    Returns
    -------
    str
        Whitespace-collapsed, case-folded query text.
    """
    return " ".join(text.split()).casefold()


def cache_key_text(value: Any) -> str:
    """
    Serialize a value into a deterministic cache-key component.

    Parameters
    ----------
    value : Any
        Value to serialize.

    Returns
    -------
    str
        Stable JSON representation.
    """
    return json.dumps(value, sort_keys=True, default=str)


def _estimate_size(value: Any) -> int:
    """Return an approximate in-memory size for *value* in bytes."""
    try:
        return len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        try:
            return len(repr(value).encode("utf-8"))
        except Exception:
            return 1


class _Flight:
    """In-flight marker for a single-flight computation."""

    __slots__ = ("event",)

    def __init__(self) -> None:
        self.event = threading.Event()


class TtlLruCache:
    """
    Thread-safe bounded cache with TTL, LRU eviction, and byte caps.

    Parameters
    ----------
    max_entries : int, optional
        Maximum number of entries, by default 1024.
    max_total_bytes : int, optional
        Maximum total approximate bytes across all entries. ``None`` disables
        the total-byte cap.
    max_entry_bytes : int, optional
        Entries larger than this are not cached. ``None`` disables the
        per-entry limit.
    clock : callable, optional
        Monotonic clock used for expiry checks.
    default_ttl : float or None, optional
        Default entry lifetime in seconds. ``None`` means entries never
        expire. Values less than or equal to zero disable caching.

    Examples
    --------
    >>> cache = TtlLruCache(max_entries=2, default_ttl=60)
    >>> cache.set("key", [1, 2, 3])
    >>> cache.get("key")
    [1, 2, 3]
    """

    def __init__(
        self,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_total_bytes: Optional[int] = None,
        max_entry_bytes: Optional[int] = None,
        clock: Callable[[], float] = time.monotonic,
        default_ttl: Optional[float] = None,
    ) -> None:
        self._entries: "OrderedDict[Hashable, Tuple[Optional[float], int, Any]]" = OrderedDict()
        self._inflight: dict[Hashable, _Flight] = {}
        self._lock = threading.RLock()
        self._clock = clock
        self._max_entries = max(1, max_entries)
        self._max_total_bytes = max_total_bytes
        self._max_entry_bytes = max_entry_bytes
        self._default_ttl = default_ttl
        self._total_bytes = 0

    def _resolve_ttl(self, ttl_seconds: Any) -> Optional[float]:
        """Return the TTL to use for an operation."""
        if ttl_seconds is _UNSET:
            return self._default_ttl
        return ttl_seconds

    @staticmethod
    def _ttl_disabled(ttl_seconds: Optional[float]) -> bool:
        """Return whether *ttl_seconds* disables caching."""
        return ttl_seconds is not None and ttl_seconds <= 0

    def _get_locked(self, key: Hashable) -> Any:
        """Return an entry reference while the lock is held."""
        entry = self._entries.get(key)
        if entry is None:
            return _MISSING

        expires_at, size, value = entry
        if expires_at is not None and self._clock() >= expires_at:
            del self._entries[key]
            self._total_bytes -= size
            return _MISSING

        self._entries.move_to_end(key)
        return value

    def _store_locked(
        self,
        key: Hashable,
        value: Any,
        size: int,
        ttl_seconds: Optional[float],
    ) -> None:
        """Store a value while the lock is held."""
        if self._max_entry_bytes is not None and size > self._max_entry_bytes:
            return
        if self._max_total_bytes is not None and size > self._max_total_bytes:
            return

        now = self._clock()
        existing = self._entries.pop(key, None)
        if existing is not None:
            self._total_bytes -= existing[1]

        expired_keys = [
            entry_key
            for entry_key, (expires_at, _, _) in self._entries.items()
            if expires_at is not None and now >= expires_at
        ]
        for entry_key in expired_keys:
            _, entry_size, _ = self._entries.pop(entry_key)
            self._total_bytes -= entry_size

        while len(self._entries) >= self._max_entries:
            _, entry = self._entries.popitem(last=False)
            self._total_bytes -= entry[1]

        if self._max_total_bytes is not None:
            while self._entries and self._total_bytes + size > self._max_total_bytes:
                _, entry = self._entries.popitem(last=False)
                self._total_bytes -= entry[1]

        expires_at = None if ttl_seconds is None else now + ttl_seconds
        self._entries[key] = (expires_at, size, value)
        self._total_bytes += size

    def get(self, key: Hashable, ttl_seconds: Any = _UNSET) -> Optional[Any]:
        """
        Return a deep copy of a cached value, or ``None`` when unavailable.

        Parameters
        ----------
        key : hashable
            Cache key.
        ttl_seconds : float or None, optional
            Override the default TTL. Values less than or equal to zero disable
            caching. ``None`` means entries never expire.

        Returns
        -------
        Any or None
            Deep copy of the cached value, or ``None``.
        """
        ttl = self._resolve_ttl(ttl_seconds)
        if self._ttl_disabled(ttl):
            return None

        with self._lock:
            value = self._get_locked(key)

        if value is _MISSING:
            return None
        return copy.deepcopy(value)

    def set(self, key: Hashable, value: Any, ttl_seconds: Any = _UNSET) -> None:
        """
        Store a value in the cache.

        Parameters
        ----------
        key : hashable
            Cache key.
        value : Any
            Value to cache.
        ttl_seconds : float or None, optional
            Override the default TTL. Values less than or equal to zero disable
            caching. ``None`` means the entry never expires.
        """
        ttl = self._resolve_ttl(ttl_seconds)
        if self._ttl_disabled(ttl):
            return

        size = _estimate_size(value)
        with self._lock:
            self._store_locked(key, value, size, ttl)

    def get_or_compute(
        self,
        key: Hashable,
        compute: Callable[[], Any],
        ttl_seconds: Any = _UNSET,
        store_predicate: Optional[Callable[[Any], bool]] = None,
    ) -> Any:
        """
        Return a cached value, computing it once for concurrent callers.

        The first thread for a missing key computes the value outside the cache
        lock. Concurrent threads for the same key wait and reuse the cached
        result. Exceptions propagate to the caller and clear the in-flight
        marker so later calls can retry.

        Parameters
        ----------
        key : hashable
            Cache key.
        compute : callable
            Zero-argument callable that produces the value.
        ttl_seconds : float or None, optional
            Override the default TTL. Values less than or equal to zero disable
            caching.
        store_predicate : callable, optional
            Predicate applied to the computed value. When it returns ``False``
            the value is returned to the caller but not cached. This is used to
            skip extremely large entries such as whole-database UID scopes.

        Returns
        -------
        Any
            Deep copy of the cached or newly computed value.
        """
        ttl = self._resolve_ttl(ttl_seconds)
        if self._ttl_disabled(ttl):
            return compute()

        while True:
            leader = False
            flight: Optional[_Flight] = None
            with self._lock:
                value = self._get_locked(key)
                if value is not _MISSING:
                    hit = True
                else:
                    hit = False
                    flight = self._inflight.get(key)
                    if flight is None:
                        flight = _Flight()
                        self._inflight[key] = flight
                        leader = True

            if hit:
                return copy.deepcopy(value)
            if leader:
                break
            assert flight is not None
            flight.event.wait()

        try:
            result = compute()
        except BaseException:
            with self._lock:
                pending = self._inflight.pop(key, None)
                if pending is not None:
                    pending.event.set()
            raise

        size = _estimate_size(result)
        with self._lock:
            if store_predicate is None or store_predicate(result):
                self._store_locked(key, result, size, ttl)
            pending = self._inflight.pop(key, None)
            if pending is not None:
                pending.event.set()

        return copy.deepcopy(result)

    def invalidate(self, key: Hashable) -> None:
        """Remove *key* from the cache."""
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is not None:
                self._total_bytes -= entry[1]

    def invalidate_prefix(self, prefix: Tuple[Any, ...]) -> None:
        """
        Remove all entries whose key tuple starts with *prefix*.

        Parameters
        ----------
        prefix : tuple
            Key prefix to match.
        """
        if not isinstance(prefix, tuple):
            prefix = (prefix,)
        with self._lock:
            keys = [
                key
                for key in self._entries
                if isinstance(key, tuple) and len(key) >= len(prefix) and key[: len(prefix)] == prefix
            ]
            for key in keys:
                _, size, _ = self._entries.pop(key)
                self._total_bytes -= size

    def clear(self) -> None:
        """Remove all cache entries."""
        with self._lock:
            self._entries.clear()
            self._inflight.clear()
            self._total_bytes = 0

    def __len__(self) -> int:
        """Return the number of stored entries."""
        with self._lock:
            return len(self._entries)


def _new_result_cache() -> TtlLruCache:
    """Build a result cache using current configuration."""
    return TtlLruCache(
        max_entries=DEFAULT_MAX_ENTRIES,
        max_total_bytes=max(0, get_config().search_cache_max_bytes),
        max_entry_bytes=DEFAULT_MAX_ENTRY_BYTES,
    )


QUERY_EMBEDDING_CACHE = TtlLruCache(
    max_entries=max(1, get_config().query_embedding_cache_max_entries),
    default_ttl=None,
)
SEARCH_RESULT_CACHE = _new_result_cache()
TOPIC_EVOLUTION_RESULT_CACHE = _new_result_cache()
SCOPE_UID_CACHE = TtlLruCache(
    max_entries=DEFAULT_MAX_ENTRIES,
    max_total_bytes=DEFAULT_SCOPE_UID_MAX_BYTES,
    max_entry_bytes=DEFAULT_MAX_ENTRY_BYTES * 8,
)

_QUERY_EMBEDDING_SIGNATURE: Optional[Tuple[str, str]] = None
_QUERY_EMBEDDING_SIGNATURE_LOCK = threading.Lock()


def get_or_compute_query_embedding(
    signature: Tuple[Any, Any],
    text: str,
    compute: Callable[[], list[float]],
) -> list[float]:
    """
    Return a query embedding from the process-global cache.

    Parameters
    ----------
    signature : tuple
        Active ``(backend_url, model_name)`` pair. A different signature
        clears the cache to avoid cross-model contamination.
    text : str
        Query text.
    compute : callable
        Zero-argument callable that generates the embedding.

    Returns
    -------
    list of float
        Query embedding vector.
    """
    global _QUERY_EMBEDDING_SIGNATURE

    active = (str(signature[0]), str(signature[1]))
    with _QUERY_EMBEDDING_SIGNATURE_LOCK:
        if _QUERY_EMBEDDING_SIGNATURE != active:
            QUERY_EMBEDDING_CACHE.clear()
            _QUERY_EMBEDDING_SIGNATURE = active

    key = normalize_query_text(text)
    return QUERY_EMBEDDING_CACHE.get_or_compute(key, compute, ttl_seconds=None)


def _result_cache_ttl() -> int:
    """Return the configured result-cache TTL in seconds."""
    return max(0, int(get_config().metadata_cache_ttl))


def get_or_compute_search_result(key: Hashable, compute: Callable[[], Any]) -> Any:
    """
    Return a cached semantic-search result or compute it once.

    Parameters
    ----------
    key : hashable
        Search cache key.
    compute : callable
        Zero-argument callable that runs the search.

    Returns
    -------
    Any
        Deep copy of the ``(papers, total_similar)`` result.
    """
    return SEARCH_RESULT_CACHE.get_or_compute(key, compute, ttl_seconds=_result_cache_ttl())


def get_or_compute_topic_evolution_result(key: Hashable, compute: Callable[[], Any]) -> Any:
    """
    Return a cached topic-evolution result or compute it once.

    Parameters
    ----------
    key : hashable
        Topic-evolution cache key.
    compute : callable
        Zero-argument callable that computes the parsed result.

    Returns
    -------
    Any
        Deep copy of the parsed topic-evolution result.
    """
    return TOPIC_EVOLUTION_RESULT_CACHE.get_or_compute(key, compute, ttl_seconds=_result_cache_ttl())


def get_or_compute_scope_uids(key: Hashable, compute: Callable[[], list[str]]) -> list[str]:
    """
    Return cached paper UIDs for a filter scope or compute them once.

    Extremely large scopes are returned without caching so a single broad
    filter cannot evict the rest of the cache or consume unbounded memory.

    Parameters
    ----------
    key : hashable
        Scope cache key. It must start with the database URL so write
        invalidation can remove entries with a prefix.
    compute : callable
        Zero-argument callable that resolves the UIDs.

    Returns
    -------
    list of str
        Deep copy of the cached or newly resolved UID list.
    """
    return SCOPE_UID_CACHE.get_or_compute(
        key,
        compute,
        ttl_seconds=_result_cache_ttl(),
        store_predicate=lambda value: len(value) <= MAX_CACHED_SCOPE_UIDS,
    )


def invalidate_result_caches(database_url: Optional[str] = None) -> None:
    """
    Invalidate result caches after a write.

    Parameters
    ----------
    database_url : str, optional
        When provided, only entries for this database are removed. When
        omitted (for example after an embedding-store write), all result
        caches are cleared.
    """
    if database_url is None:
        SEARCH_RESULT_CACHE.clear()
        TOPIC_EVOLUTION_RESULT_CACHE.clear()
        SCOPE_UID_CACHE.clear()
        return

    prefix = (database_url,)
    SEARCH_RESULT_CACHE.invalidate_prefix(prefix)
    TOPIC_EVOLUTION_RESULT_CACHE.invalidate_prefix(prefix)
    SCOPE_UID_CACHE.invalidate_prefix(prefix)


def reset_query_embedding_cache() -> None:
    """Clear the query embedding cache and active signature."""
    global _QUERY_EMBEDDING_SIGNATURE

    QUERY_EMBEDDING_CACHE.clear()
    with _QUERY_EMBEDDING_SIGNATURE_LOCK:
        _QUERY_EMBEDDING_SIGNATURE = None


def reset_result_caches() -> None:
    """Clear the semantic-search, topic-evolution, and scope UID result caches."""
    SEARCH_RESULT_CACHE.clear()
    TOPIC_EVOLUTION_RESULT_CACHE.clear()
    SCOPE_UID_CACHE.clear()


def reset_all_caches() -> None:
    """Clear all process-global caches."""
    reset_query_embedding_cache()
    reset_result_caches()

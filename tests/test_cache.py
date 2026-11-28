"""
Tests for the shared process-global cache infrastructure.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from abstracts_explorer.cache import TtlLruCache, normalize_query_text


class FakeClock:
    """Controllable monotonic clock for TTL tests."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        """Advance the clock by *seconds*."""
        self.now += seconds


class TestTtlLruCache:
    """Tests for TTL, LRU, byte-cap, and copy semantics."""

    def test_ttl_expiry_with_injected_clock(self):
        clock = FakeClock()
        cache = TtlLruCache(default_ttl=60, clock=clock)
        cache.set("key", [2025])

        clock.advance(59)
        assert cache.get("key") == [2025]

        clock.advance(2)
        assert cache.get("key") is None

    def test_lru_eviction(self):
        cache = TtlLruCache(max_entries=2, default_ttl=60)
        cache.set("a", 1)
        cache.set("b", 2)

        assert cache.get("a") == 1
        cache.set("c", 3)

        assert cache.get("b") is None
        assert cache.get("a") == 1
        assert cache.get("c") == 3

    def test_byte_cap_evicts_lru(self):
        cache = TtlLruCache(max_entries=10, max_total_bytes=200, max_entry_bytes=1000, default_ttl=60)
        cache.set("a", b"a" * 100)
        cache.set("b", b"b" * 100)

        assert cache.get("a") is None
        assert cache.get("b") == b"b" * 100

    def test_oversized_entry_skipped(self):
        cache = TtlLruCache(max_entries=10, max_entry_bytes=50, default_ttl=60)
        cache.set("big", b"x" * 200)

        assert cache.get("big") is None
        assert len(cache) == 0

    def test_get_returns_deep_copy(self):
        cache = TtlLruCache(default_ttl=60)
        original = {"years": [2025]}
        cache.set("key", original)

        first = cache.get("key")
        assert first is not None
        first["years"].append(2024)

        assert cache.get("key") == {"years": [2025]}
        assert original == {"years": [2025]}

    def test_zero_ttl_disables_cache(self):
        cache = TtlLruCache(default_ttl=0)
        cache.set("key", [2025])

        assert cache.get("key") is None

    def test_invalidate_prefix(self):
        cache = TtlLruCache(default_ttl=60)
        cache.set(("sqlite:///a", "x"), 1)
        cache.set(("sqlite:///b", "x"), 2)

        cache.invalidate_prefix(("sqlite:///a",))

        assert cache.get(("sqlite:///a", "x")) is None
        assert cache.get(("sqlite:///b", "x")) == 2


class TestSingleFlight:
    """Tests for the single-flight compute helper."""

    def test_concurrent_same_key_compute_runs_once(self):
        cache = TtlLruCache(default_ttl=60)
        calls = 0
        calls_lock = threading.Lock()
        start = threading.Barrier(8)

        def compute() -> list[int]:
            nonlocal calls
            with calls_lock:
                calls += 1
            time.sleep(0.05)
            return [42]

        def worker() -> list[int]:
            start.wait()
            return cache.get_or_compute("key", compute)

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda _: worker(), range(8)))

        assert calls == 1
        assert results == [[42]] * 8

    def test_exception_does_not_poison_cache(self):
        cache = TtlLruCache(default_ttl=60)
        calls = 0

        def failing() -> str:
            nonlocal calls
            calls += 1
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            cache.get_or_compute("key", failing)

        assert cache.get("key") is None

        def succeeding() -> str:
            nonlocal calls
            calls += 1
            return "ok"

        assert cache.get_or_compute("key", succeeding) == "ok"
        assert calls == 2
        assert cache.get("key") == "ok"


def test_normalize_query_text():
    assert normalize_query_text("  Hello\n\tWorld  ") == "hello world"
    assert normalize_query_text("HELLO   world") == "hello world"

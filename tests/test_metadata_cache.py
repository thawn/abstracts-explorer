"""
Tests for the metadata cache.
"""

from __future__ import annotations

from abstracts_explorer.metadata_cache import MetadataCache


class FakeClock:
    """Controllable monotonic clock for TTL tests."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        """Advance the clock by *seconds*."""
        self.now += seconds


class TestMetadataCache:
    """Tests for MetadataCache get/set/expiry behavior."""

    def test_set_and_get(self):
        cache = MetadataCache()

        cache.set("key", {"years": [2025]}, ttl_seconds=60)

        assert cache.get("key", ttl_seconds=60) == {"years": [2025]}

    def test_get_returns_copy(self):
        cache = MetadataCache()
        cache.set("key", {"years": [2025]}, ttl_seconds=60)

        first = cache.get("key", ttl_seconds=60)
        assert first is not None
        first["years"].append(2024)
        second = cache.get("key", ttl_seconds=60)

        assert second == {"years": [2025]}

    def test_expiry(self):
        clock = FakeClock()
        cache = MetadataCache(clock=clock)
        cache.set("key", [2025], ttl_seconds=60)

        clock.advance(59)
        assert cache.get("key", ttl_seconds=60) == [2025]

        clock.advance(2)
        assert cache.get("key", ttl_seconds=60) is None

    def test_zero_ttl_disables_cache(self):
        cache = MetadataCache()

        cache.set("key", [2025], ttl_seconds=0)

        assert cache.get("key", ttl_seconds=0) is None

    def test_invalidate_database_only_matching_entries(self):
        cache = MetadataCache()
        cache.set(("postgresql://a", "years"), [2025], ttl_seconds=60)
        cache.set(("postgresql://b", "years"), [2024], ttl_seconds=60)
        cache.set("unrelated", [2023], ttl_seconds=60)

        cache.invalidate_database("postgresql://a")

        assert cache.get(("postgresql://a", "years"), ttl_seconds=60) is None
        assert cache.get(("postgresql://b", "years"), ttl_seconds=60) == [2024]
        assert cache.get("unrelated", ttl_seconds=60) == [2023]

    def test_clear(self):
        cache = MetadataCache()
        cache.set("key", [2025], ttl_seconds=60)

        cache.clear()

        assert cache.get("key", ttl_seconds=60) is None

"""
Tests for time utility helpers.
"""

from datetime import datetime, timedelta, timezone

from abstracts_explorer.utils.time_utils import format_timestamp, utcnow


class TestUtcnow:
    """Tests for utcnow."""

    def test_returns_timezone_aware_datetime(self):
        now = utcnow()
        assert isinstance(now, datetime)
        assert now.tzinfo is not None
        assert now.utcoffset() == timedelta(0)

    def test_close_to_wall_clock_utc(self):
        before = datetime.now(timezone.utc)
        now = utcnow()
        after = datetime.now(timezone.utc)
        assert before <= now <= after


class TestFormatTimestamp:
    """Tests for format_timestamp."""

    def test_formats_datetime_with_default_format(self):
        dt = datetime(2025, 1, 2, 3, 4)
        assert format_timestamp(dt) == "2025-01-02 03:04"

    def test_formats_datetime_with_custom_format(self):
        dt = datetime(2025, 1, 2, 3, 4, 5)
        assert format_timestamp(dt, "%Y-%m-%d %H:%M:%S") == "2025-01-02 03:04:05"

    def test_none_returns_fallback(self):
        assert format_timestamp(None) == "unknown"

    def test_custom_fallback(self):
        assert format_timestamp(None, fallback="unknown date") == "unknown date"

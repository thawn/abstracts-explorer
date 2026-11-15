"""
Time helpers shared across the package.

Provides a single source of truth for timezone-aware UTC timestamps and for
rendering datetimes in CLI/API output.
"""

from datetime import datetime, timezone
from typing import Optional

DEFAULT_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M"


def utcnow() -> datetime:
    """
    Return the current UTC time as a timezone-aware datetime.

    Returns
    -------
    datetime
        Current time with ``tzinfo=timezone.utc``.

    Examples
    --------
    >>> utcnow().tzinfo is not None
    True
    """
    return datetime.now(timezone.utc)


def format_timestamp(
    value: Optional[datetime],
    fmt: str = DEFAULT_TIMESTAMP_FORMAT,
    fallback: str = "unknown",
) -> str:
    """
    Format a datetime for display, with a fallback for missing values.

    Parameters
    ----------
    value : datetime, optional
        Timestamp to format. Values that are not ``datetime`` instances
        (including ``None``) yield *fallback*.
    fmt : str, optional
        ``strftime`` format string (default ``"%Y-%m-%d %H:%M"``).
    fallback : str, optional
        Text returned when *value* is not a datetime (default ``"unknown"``).

    Returns
    -------
    str
        Formatted timestamp, or *fallback* when no datetime was given.

    Examples
    --------
    >>> from datetime import datetime
    >>> format_timestamp(datetime(2025, 1, 2, 3, 4))
    '2025-01-02 03:04'
    >>> format_timestamp(None)
    'unknown'
    """
    if isinstance(value, datetime):
        return value.strftime(fmt)
    return fallback

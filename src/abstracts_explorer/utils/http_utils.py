"""
HTTP timeout helpers shared across the package.
"""

import asyncio
from collections.abc import Iterator
from typing import Optional

import httpx2

_TIMEOUT_CLASS_NAMES = frozenset(
    {
        "APITimeoutError",
        "ConnectTimeout",
        "ModelRequestTimeout",
        "ModelTimeoutError",
        "PoolTimeout",
        "ReadTimeout",
        "TimeoutError",
        "WriteTimeout",
    }
)

_CONNECTION_CLASS_NAMES = frozenset(
    {
        "APIConnectionError",
        "ConnectError",
        "NetworkError",
        "RemoteProtocolError",
    }
)


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    """Yield *exc* and its chained causes without repeating exceptions."""
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def is_llm_timeout(exc: BaseException) -> bool:
    """
    Return whether *exc* represents an LLM or embedding backend timeout.

    Parameters
    ----------
    exc : BaseException
        Exception raised while talking to an OpenAI-compatible backend.

    Returns
    -------
    bool
        True when the exception or one of its underlying causes is a timeout.

    Examples
    --------
    >>> import httpx2
    >>> is_llm_timeout(httpx2.ReadTimeout("timed out"))
    True
    """
    for item in _exception_chain(exc):
        if isinstance(item, (httpx2.TimeoutException, TimeoutError, asyncio.TimeoutError)):
            return True
        if item.__class__.__name__ in _TIMEOUT_CLASS_NAMES:
            return True
        if "timed out" in str(item).lower():
            return True
    return False


def is_llm_connection_error(exc: BaseException) -> bool:
    """
    Return whether *exc* represents an LLM or embedding backend connection failure.

    Timeouts are excluded; :func:`is_llm_timeout` should be checked first when
    choosing a user-facing message.

    Parameters
    ----------
    exc : BaseException
        Exception raised while talking to an OpenAI-compatible backend.

    Returns
    -------
    bool
        True when the exception or one of its underlying causes is a
        non-timeout connection failure.
    """
    for item in _exception_chain(exc):
        if is_llm_timeout(item):
            continue
        if isinstance(item, httpx2.TransportError):
            return True
        if item.__class__.__name__ in _CONNECTION_CLASS_NAMES:
            return True
    return False


def build_llm_timeout(
    request_timeout: float,
    connect_timeout: float = 10.0,
) -> httpx2.Timeout:
    """
    Build a structured timeout for LLM and embedding HTTP requests.

    Parameters
    ----------
    request_timeout : float
        Read/write timeout in seconds for a single HTTP request.
    connect_timeout : float, optional
        TCP/TLS connect and connection-pool timeout in seconds, by default 10.0.

    Returns
    -------
    httpx2.Timeout
        Timeout with connect/pool set to ``connect_timeout`` and read/write
        set to ``request_timeout``.

    Examples
    --------
    >>> timeout = build_llm_timeout(60, connect_timeout=10)
    >>> timeout.connect, timeout.read
    (10.0, 60)
    """
    return httpx2.Timeout(
        connect=connect_timeout,
        read=request_timeout,
        write=request_timeout,
        pool=connect_timeout,
    )

"""
HTTP timeout helpers shared across the package.
"""

import httpx2


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

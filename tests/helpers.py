"""
Shared test helper utilities.

This module contains common utility functions used across multiple test files
to reduce code duplication and ensure consistency.
"""

import functools
import socket
import threading
from collections.abc import Iterator
from typing import Any

import pytest

import httpx2
import openai
from sqlalchemy import text

from abstracts_explorer.config import get_config
from abstracts_explorer.embeddings import EmbeddingsManager
from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.utils.http_utils import is_llm_connection_error, is_llm_timeout

from tests.conftest import get_env_test_path

# Cache for LM Studio availability check to avoid multiple API calls during test collection
_lm_studio_available_cache = None


def raw_query(db_manager: DatabaseManager, sql: str, parameters: tuple = ()) -> list[dict[str, Any]]:
    """
    Execute a raw SQL query against the database (test escape hatch).

    Test assertions sometimes need to inspect raw table contents instead of
    going through the ``DatabaseManager`` API.  This helper keeps that
    capability in the test suite only.

    Parameters
    ----------
    db_manager : DatabaseManager
        Manager providing the session to execute against.
    sql : str
        SQL query to execute (use ``?`` placeholders or named parameters).
    parameters : tuple, optional
        Query parameters for ``?`` placeholders.

    Returns
    -------
    list of dict
        Query results as list of dictionaries.
    """
    converted_sql = sql
    for i in range(sql.count("?")):
        converted_sql = converted_sql.replace("?", f":param{i}", 1)
    param_dict = {f"param{i}": parameters[i] for i in range(len(parameters))}
    result = db_manager.session.execute(text(converted_sql), param_dict)
    return [dict(row._mapping) for row in result]


class FakeTime:
    """
    Controllable clock for tests that avoids real sleeps.

    Parameters
    ----------
    start : float, optional
        Initial virtual time in seconds, by default 1000.0.

    Attributes
    ----------
    current : float
        Current virtual time in seconds.
    sleeps : list of float
        Recorded sleep durations.
    """

    def __init__(self, start: float = 1000.0) -> None:
        self.current = start
        self.sleeps: list[float] = []
        self._lock = threading.Lock()

    def monotonic(self) -> float:
        """Return the current virtual time."""
        return self.current

    def sleep(self, seconds: float) -> None:
        """Record a sleep and advance the virtual clock."""
        with self._lock:
            self.sleeps.append(seconds)
            self.current += max(seconds, 1e-6)


def reset_lm_studio_availability_cache():
    """Reset the cached result of LM Studio availability check."""
    global _lm_studio_available_cache
    _lm_studio_available_cache = None


def mock_plugin_current_year_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Disable live current-year availability probes for all plugin classes.

    Plugin ``supported_years`` properties probe the conference website for the
    current year. Unit tests should not perform those requests, so this helper
    patches the probe on the base class and on plugins that override it.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture used to replace the probe methods.
    """
    from abstracts_explorer.plugin import DownloaderPlugin
    from abstracts_explorer.plugins.aaai_downloader import AAAIDownloaderPlugin
    from abstracts_explorer.plugins.igarss_downloader import IGARSSDownloaderPlugin

    for plugin_class in (DownloaderPlugin, AAAIDownloaderPlugin, IGARSSDownloaderPlugin):
        monkeypatch.setattr(plugin_class, "_check_current_year_available", lambda self, year: True)


def create_test_db_with_paper(db_path, paper_data):
    """
    Create a test database with a single paper.

    Parameters
    ----------
    db_path : Path or str
        Path where the database should be created
    paper_data : dict
        Dictionary containing paper data with keys: uid, title, abstract, authors,
        session, poster_position, keywords, year, conference

    Returns
    -------
    DatabaseManager
        Connected database manager instance. Caller is responsible for closing it.

    Notes
    -----
    This helper reduces duplication in integration tests that need to manually
    create databases with specific paper data. Default values are provided for
    optional fields if not specified.

    This function temporarily sets the PAPER_DB environment variable to configure
    the DatabaseManager with the specified path.

    Examples
    --------
    >>> db = create_test_db_with_paper(
    ...     tmp_path / "test.db",
    ...     {
    ...         "uid": "1",
    ...         "title": "Test Paper",
    ...         "abstract": "Test abstract",
    ...         "authors": "Test Author",
    ...         "session": "Test Session",
    ...         "keywords": "test",
    ...         "year": 2025,
    ...         "conference": "NeurIPS"
    ...     }
    ... )
    >>> # Use db...
    >>> db.close()
    """
    from abstracts_explorer.plugin import LightweightPaper
    from tests.conftest import set_test_db

    # Set environment variable and reload config
    set_test_db(db_path)

    db = DatabaseManager()
    db.create_tables()

    # Create LightweightPaper from paper_data
    # Convert authors if needed
    authors = paper_data.get("authors", [])
    if isinstance(authors, str):
        authors = [authors]

    paper = LightweightPaper(
        original_id=paper_data.get("uid"),
        title=paper_data["title"],
        abstract=paper_data.get("abstract", ""),
        authors=authors,
        session=paper_data.get("session", "Session"),
        poster_position=paper_data.get("poster_position", ""),
        keywords=paper_data.get("keywords", "").split(", ") if paper_data.get("keywords") else None,
        year=paper_data.get("year", 2025),
        conference=paper_data.get("conference", "TestConf"),
    )

    db.add_paper(paper)

    return db


def check_lm_studio_available():
    """
    Check if OpenAI-compatible API is available with the configured models.

    Returns
    -------
    bool
        True if the API is available with required models, False otherwise.

    Notes
    -----
    This function checks:
    1. If the API server is running
    2. If any models are loaded
    3. If an embedding generation request works with the configured model

    Used to skip tests that require a running OpenAI-compatible API backend
    (such as LM Studio or blablador).

    The result is cached to avoid multiple API calls during test collection.
    """
    global _lm_studio_available_cache

    # Return cached result if available
    if _lm_studio_available_cache is not None:
        return _lm_studio_available_cache

    try:
        config = get_config(reload=True, env_path=get_env_test_path())
        em = EmbeddingsManager(
            backend_url=config.llm_backend_url,
            auth_token=config.llm_backend_auth_token,
            model_name=config.embedding_model,
        )

        try:
            # Connect and perform a lightweight liveness check via the embeddings API
            if not em.test_api_connection():
                _lm_studio_available_cache = False
                return False

            _lm_studio_available_cache = True
            return True
        finally:
            em.close()

    except (httpx2.HTTPError, openai.OpenAIError, OSError, ConnectionError):
        # Catch network and API errors: the openai client uses httpx2
        # internally, so network errors surface as httpx2/openai exceptions
        # rather than requests exceptions.
        _lm_studio_available_cache = False
        return False


def find_free_port():
    """
    Find a free port to use for testing.

    Returns
    -------
    int
        Available port number

    Notes
    -----
    Uses a temporary socket to find an available port on the system.
    Useful for starting test servers that need unique ports.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def stop_test_server(server: Any, server_thread: threading.Thread, timeout: float = 30.0) -> None:
    """
    Stop a WSGI test server and wait for in-flight requests to finish.

    Parameters
    ----------
    server : Any
        WSGI server returned by ``werkzeug.serving.make_server``.
    server_thread : threading.Thread
        Thread running ``server.serve_forever()``.
    timeout : float, optional
        Maximum seconds to wait for the serve loop to stop (default: 30).

    Raises
    ------
    RuntimeError
        If the server thread does not stop within *timeout*.
    """
    server.shutdown()
    server_thread.join(timeout=timeout)
    if server_thread.is_alive():
        raise RuntimeError(f"Test server did not stop within {timeout:.0f}s")
    server.server_close()


def requires_lm_studio(func):
    """
    Mark a test as slow and requiring a reachable OpenAI-compatible API backend.

    The availability check is performed lazily at test setup time by the
    ``pytest_runtest_setup`` hook in ``tests/conftest.py``, so deselected slow
    tests never trigger a network request during collection.

    Usage
    -----
    @requires_lm_studio
    def test_something_with_api():
        ...
    """
    func = pytest.mark.slow(func)
    func = pytest.mark.requires_lm_studio(func)
    return func


_TRANSIENT_MESSAGE_MARKERS = (
    "502",
    "503",
    "504",
    "bad gateway",
    "service unavailable",
    "gateway timeout",
    "timed out",
    "connection error",
    "connection reset",
    "temporary failure",
)


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    """Yield *exc* and its chained causes without repeating exceptions."""
    seen: set[int] = set()
    current: Any = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def is_transient_backend_message(text: str) -> bool:
    """
    Return whether a backend error message indicates a transient outage.

    Parameters
    ----------
    text : str
        Warning or error text produced by the OpenAI-compatible backend.

    Returns
    -------
    bool
        True when the text looks like a transient server/network failure.
    """
    if not text:
        return False
    lowered = text.lower()
    return any(marker in lowered for marker in _TRANSIENT_MESSAGE_MARKERS)


def is_transient_backend_error(exc: BaseException) -> bool:
    """
    Return whether *exc* represents a transient backend outage.

    Traverses the exception chain so wrapped errors such as ``EmbeddingsError``
    caused by an ``openai.InternalServerError`` are recognised. Assertions are
    never treated as transient backend failures.

    Parameters
    ----------
    exc : BaseException
        Exception raised while talking to an OpenAI-compatible backend.

    Returns
    -------
    bool
        True for 5xx responses, connection failures, and timeouts.
    """
    if isinstance(exc, AssertionError):
        return False

    for item in _exception_chain(exc):
        status = getattr(item, "status_code", None)
        if isinstance(status, int) and status >= 500:
            return True
        if isinstance(item, (openai.APITimeoutError, openai.APIConnectionError)):
            return True
        if isinstance(item, httpx2.HTTPStatusError) and item.response.status_code >= 500:
            return True
        if is_llm_timeout(item) or is_llm_connection_error(item):
            return True
        if is_transient_backend_message(str(item)):
            return True
    return False


def skip_on_backend_unavailable(func):
    """
    Skip a test when the OpenAI-compatible backend becomes unavailable.

    The pre-test availability check can succeed while the backend fails later
    with a 5xx, timeout, or connection error. Those are infrastructure outages,
    not test failures, so the test is skipped instead.

    Usage
    -----
    @requires_lm_studio
    @skip_on_backend_unavailable
    def test_something_with_api():
        ...
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            if not is_transient_backend_error(exc):
                raise
            pytest.skip(f"LLM backend unavailable during test: {exc}")

    return wrapper

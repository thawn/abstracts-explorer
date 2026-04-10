"""
Shared test helper utilities.

This module contains common utility functions used across multiple test files
to reduce code duplication and ensure consistency.
"""

import os
import socket
import pytest

import httpx
import openai

from abstracts_explorer.config import get_config
from abstracts_explorer.embeddings import EmbeddingsManager
from abstracts_explorer.database import DatabaseManager

from tests.conftest import get_env_test_path

# Cache for LM Studio availability check to avoid multiple API calls during test collection
_lm_studio_available_cache = None


def reset_lm_studio_availability_cache():
    """Reset the cached result of LM Studio availability check."""
    global _lm_studio_available_cache
    _lm_studio_available_cache = None


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
    db.connect()
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
            lm_studio_url=config.llm_backend_url,
            auth_token=config.llm_backend_auth_token,
            model_name=config.embedding_model,
        )

        try:
            # Connect and perform a lightweight liveness check via the embeddings API
            if not em.test_lm_studio_connection():
                _lm_studio_available_cache = False
                return False

            _lm_studio_available_cache = True
            return True
        finally:
            em.close()

    except (httpx.HTTPError, openai.OpenAIError, OSError, ConnectionError):
        # Catch network and API errors: the openai client uses httpx
        # internally, so network errors surface as httpx/openai exceptions
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


def requires_lm_studio(func):
    """
    Decorator that marks tests as slow and skips them if OpenAI API backend is not available.

    This decorator:
    1. Marks the test as 'slow' (so it's skipped by default with -m "not slow")
    2. Skips the test if the configured OpenAI-compatible API is not running or no model is loaded

    Usage
    -----
    @requires_lm_studio
    def test_something_with_api():
        ...
    """
    # Apply both slow marker and skipif condition
    func = pytest.mark.slow(func)
    func = pytest.mark.skipif(
        not check_lm_studio_available(),
        reason="OpenAI-compatible API not available. Check configuration and ensure the API backend (LM Studio or blablador) is accessible with the configured models.",
    )(func)
    return func


# ---------------------------------------------------------------------------
# Browser driver helpers (shared between test_web_e2e.py and staging tests)
# ---------------------------------------------------------------------------

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.firefox.service import Service as FirefoxService
    from selenium.webdriver.firefox.options import Options as FirefoxOptions
    from webdriver_manager.chrome import ChromeDriverManager
    from webdriver_manager.firefox import GeckoDriverManager
    from webdriver_manager.core.os_manager import ChromeType

    _SELENIUM_AVAILABLE = True
except ImportError:
    _SELENIUM_AVAILABLE = False

# Module-level cache for driver paths – prevents redundant installs when
# running multiple E2E tests in the same session.
_driver_cache: dict[str, str | bool | None] = {
    "chrome": None,
    "firefox": None,
    "chrome_available": None,
    "firefox_available": None,
}


def _get_browser_preference() -> str:
    """
    Return the browser preference from the ``E2E_BROWSER`` environment variable.

    Returns
    -------
    str
        ``'chrome'``, ``'firefox'``, or ``'auto'`` (default).
    """
    return os.environ.get("E2E_BROWSER", "auto").lower()


def _check_chrome_available() -> bool:
    """
    Return ``True`` if a Chrome/Chromium binary is present on PATH.

    Does **not** install the driver – only checks for the browser executable.

    Returns
    -------
    bool
    """
    if _driver_cache["chrome_available"] is not None:
        return _driver_cache["chrome_available"]
    try:
        import shutil

        chrome_binary = shutil.which("chromium") or shutil.which("chrome") or shutil.which("google-chrome")
        result = chrome_binary is not None
        _driver_cache["chrome_available"] = result
        return result
    except Exception:
        _driver_cache["chrome_available"] = False
        return False


def _check_firefox_available() -> bool:
    """
    Return ``True`` if a Firefox binary is present on PATH.

    Does **not** install the driver – only checks for the browser executable.

    Returns
    -------
    bool
    """
    if _driver_cache["firefox_available"] is not None:
        return _driver_cache["firefox_available"]
    try:
        import shutil

        firefox_binary = shutil.which("firefox")
        result = firefox_binary is not None
        _driver_cache["firefox_available"] = result
        return result
    except Exception:
        _driver_cache["firefox_available"] = False
        return False


def _create_chrome_driver():
    """
    Create and return a headless Chrome ``WebDriver`` instance.

    The ChromeDriver binary is installed once per test session via
    ``webdriver_manager`` (cached in ``_driver_cache``).

    Returns
    -------
    webdriver.Chrome

    Raises
    ------
    Exception
        If driver installation or browser creation fails.
    """
    chrome_options = ChromeOptions()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--ignore-certificate-errors")

    if _driver_cache["chrome"] is None:
        try:
            _driver_cache["chrome"] = ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install()
        except Exception as exc:
            _driver_cache["chrome_available"] = False
            raise Exception(f"Failed to install ChromeDriver: {exc}") from exc

    service = ChromeService(_driver_cache["chrome"])
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.implicitly_wait(10)
    return driver


def _create_firefox_driver():
    """
    Create and return a headless Firefox ``WebDriver`` instance.

    The GeckoDriver binary is installed once per test session via
    ``webdriver_manager`` (cached in ``_driver_cache``).

    Returns
    -------
    webdriver.Firefox

    Raises
    ------
    Exception
        If driver installation or browser creation fails.
    """
    firefox_options = FirefoxOptions()
    firefox_options.add_argument("--headless")
    firefox_options.add_argument("--width=1920")
    firefox_options.add_argument("--height=1080")

    if _driver_cache["firefox"] is None:
        try:
            _driver_cache["firefox"] = GeckoDriverManager().install()
        except Exception as exc:
            _driver_cache["firefox_available"] = False
            raise Exception(f"Failed to install GeckoDriver: {exc}") from exc

    service = FirefoxService(_driver_cache["firefox"])
    driver = webdriver.Firefox(service=service, options=firefox_options)
    driver.implicitly_wait(10)
    return driver


def create_webdriver():
    """
    Create a ``WebDriver`` instance using the configured browser preference.

    Reads the ``E2E_BROWSER`` environment variable to choose Chrome, Firefox, or
    auto (tries Chrome first, falls back to Firefox).  If no browser is available
    the calling test is skipped via ``pytest.skip``.

    Returns
    -------
    webdriver.Chrome or webdriver.Firefox

    Notes
    -----
    Callers are responsible for calling ``driver.quit()`` when done.
    """
    pref = _get_browser_preference()

    if pref == "chrome":
        if not _check_chrome_available():
            pytest.skip("Chrome browser not available for E2E testing")
        try:
            return _create_chrome_driver()
        except Exception as exc:
            pytest.skip(f"Failed to create Chrome driver: {exc}")
    elif pref == "firefox":
        if not _check_firefox_available():
            pytest.skip("Firefox browser not available for E2E testing")
        try:
            return _create_firefox_driver()
        except Exception as exc:
            pytest.skip(f"Failed to create Firefox driver: {exc}")
    else:
        # auto – try Chrome, fall back to Firefox
        if _check_chrome_available():
            try:
                return _create_chrome_driver()
            except Exception:
                if _check_firefox_available():
                    try:
                        return _create_firefox_driver()
                    except Exception as exc:
                        pytest.skip(
                            f"Failed to create browser drivers: Chrome and Firefox both failed. Last error: {exc}"
                        )
                else:
                    pytest.skip("Chrome driver installation failed and Firefox not available")
        elif _check_firefox_available():
            try:
                return _create_firefox_driver()
            except Exception as exc:
                pytest.skip(f"Failed to create Firefox driver: {exc}")
        else:
            pytest.skip("Neither Chrome nor Firefox browser available for E2E testing")

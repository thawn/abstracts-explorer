"""
Shared pytest fixtures for all test modules.

This module contains common fixtures used across multiple test files to reduce
code duplication and ensure consistency in test setup.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock

from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.embeddings import EmbeddingsManager
from abstracts_explorer.plugin import LightweightPaper
from abstracts_explorer.config import get_config


@pytest.fixture(autouse=True)
def _reset_global_rate_limiter():
    """Reset the global rate limiter between tests to avoid test pollution."""
    from abstracts_explorer.rate_limiter import set_global_rate_limiter
    from abstracts_explorer.web_ui.rate_limit import reset_client_rate_limiter

    set_global_rate_limiter(None)
    reset_client_rate_limiter()
    yield
    reset_client_rate_limiter()


@pytest.fixture(autouse=True)
def _reset_process_caches():
    """Reset process-level schema and metadata caches between tests."""
    from abstracts_explorer.database import reset_metadata_cache, reset_schema_cache

    reset_schema_cache()
    reset_metadata_cache()
    yield
    reset_schema_cache()
    reset_metadata_cache()


@pytest.fixture(autouse=True)
def _reset_mcp_resources():
    """Close cached MCP database/embeddings resources between tests."""
    from abstracts_explorer.mcp import close_mcp_resources

    close_mcp_resources()
    yield
    close_mcp_resources()


def get_base_env_test_path() -> Path:
    """
    Get the path to the tracked .env.tests file.

    Returns
    -------
    Path
        Path to the tracked .env.tests file in the repository root.
    """
    repo_root = Path(__file__).parent.parent
    return repo_root / ".env.tests"


def get_env_test_path() -> Path:
    """
    Get the path to the test env file.

    If a ``.env.tests.local`` file exists next to the tracked ``.env.tests``,
    it is used instead.  Copy ``.env.tests`` to ``.env.tests.local`` and edit
    the copy to point slow tests at a local LM Studio backend or to use a
    personal Blablador token.  Process environment variables still take
    precedence over the file.

    Returns
    -------
    Path
        Path to ``.env.tests.local`` when it exists, otherwise the tracked
        ``.env.tests`` file.

    Examples
    --------
    >>> env_test = get_env_test_path()
    >>> config = get_config(reload=True, env_path=env_test)
    """
    base_path = get_base_env_test_path()
    local_path = base_path.with_name(f"{base_path.name}.local")
    if local_path.exists():
        return local_path
    return base_path


def pytest_addoption(parser):
    """
    Register custom command-line options for pytest.

    Parameters
    ----------
    parser : argparse.Parser
        The pytest argument parser.
    """
    parser.addoption(
        "--staging-url",
        action="store",
        default=None,
        help="Base URL of the staging deployment to run staging e2e tests against (e.g. http://localhost:5000)",
    )
    parser.addoption(
        "--server-url",
        action="store",
        default=None,
        help="Base URL of the deployment to run stress tests against (e.g. http://localhost:5000)",
    )
    parser.addoption(
        "--stress-mcp-tool",
        action="store",
        default=None,
        help="MCP tool to target with stress-test prompts (default: search_papers)",
    )


def pytest_runtest_setup(item: pytest.Item) -> None:
    """
    Skip tests marked ``requires_lm_studio`` when the backend is unavailable.

    The check runs only for selected tests, avoiding a network request during
    collection when slow tests are deselected.

    Parameters
    ----------
    item : pytest.Item
        Test item that is about to run.
    """
    if item.get_closest_marker("requires_lm_studio") is None:
        return

    from tests.helpers import check_lm_studio_available

    if not check_lm_studio_available():
        pytest.skip(
            "OpenAI-compatible API not available. Check configuration and ensure the API backend "
            "(LM Studio or blablador) is accessible with the configured models."
        )


@pytest.fixture
def mock_current_year_probe(monkeypatch):
    """
    Disable live current-year plugin availability probes.

    Tests that exercise plugin metadata or supported years can request this
    fixture instead of reimplementing the same monkeypatches.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture.
    """
    from tests.helpers import mock_plugin_current_year_probe

    mock_plugin_current_year_probe(monkeypatch)


@pytest.fixture(scope="session")
def staging_url(request):
    """
    Resolve the staging deployment URL from CLI option or environment variable.

    The URL is taken from ``--staging-url`` first, falling back to the
    ``STAGING_URL`` environment variable.  Tests that depend on this fixture
    are skipped when neither source provides a value.

    Parameters
    ----------
    request : FixtureRequest
        Pytest request object (provides access to CLI options).

    Returns
    -------
    str
        The base URL of the staging deployment (without trailing slash).
    """
    import os

    url = request.config.getoption("--staging-url") or os.environ.get("STAGING_URL")
    if not url:
        pytest.skip("Staging URL not provided. Use --staging-url or set STAGING_URL env var.")
    return url.rstrip("/")


@pytest.fixture(scope="session")
def server_url(request):
    """
    Resolve the stress-test deployment URL from CLI option or environment variable.

    The URL is taken from ``--server-url`` first, falling back to the
    ``SERVER_URL`` environment variable.  Tests that depend on this fixture
    are skipped when neither source provides a value.

    Parameters
    ----------
    request : FixtureRequest
        Pytest request object (provides access to CLI options).

    Returns
    -------
    str
        The base URL of the deployment (without trailing slash).
    """
    import os

    url = request.config.getoption("--server-url") or os.environ.get("SERVER_URL")
    if not url:
        pytest.skip("Server URL not provided. Use --server-url or set SERVER_URL env var.")
    return url.rstrip("/")


@pytest.fixture(scope="session")
def stress_mcp_tool(request):
    """
    Resolve the MCP tool targeted by stress-test prompts.

    The tool is taken from ``--stress-mcp-tool`` first, falling back to the
    ``STRESS_MCP_TOOL`` environment variable and finally to
    ``search_papers``.

    Parameters
    ----------
    request : FixtureRequest
        Pytest request object (provides access to CLI options).

    Returns
    -------
    str
        Name of the MCP tool to target.
    """
    import os

    return request.config.getoption("--stress-mcp-tool") or os.environ.get("STRESS_MCP_TOOL") or "search_papers"


@pytest.fixture(scope="module")
def staging_page(browser):
    """
    Create a module-scoped Playwright page for staging e2e tests.

    A single browser context and page are shared across all tests in a module
    to avoid the overhead of launching a new browser for every test.  The
    fixture relies on the session-scoped ``browser`` fixture provided by
    ``pytest-playwright``.

    Yields
    ------
    playwright.sync_api.Page
        A page using a fresh browser context.  The context is closed after the
        module finishes.
    """
    context = browser.new_context()
    page = context.new_page()
    yield page
    context.close()


def _collect_console_errors(page) -> list[str]:
    """
    Attach console and page-error listeners to *page* and return the collector.

    Parameters
    ----------
    page : playwright.sync_api.Page
        Page whose console output should be collected.

    Returns
    -------
    list[str]
        Messages for console entries of type ``error`` and uncaught page
        exceptions.  The list is populated as the page runs.
    """
    errors: list[str] = []
    page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    return errors


@pytest.fixture
def console_errors(page):
    """
    Collect browser console errors and uncaught page errors.

    The listeners are attached before the test body runs, so errors emitted
    during navigation and asynchronous rendering are captured.

    Parameters
    ----------
    page : playwright.sync_api.Page
        Playwright page fixture.

    Returns
    -------
    list[str]
        Messages for console entries of type ``error`` and uncaught page
        exceptions.  The list is populated as the test runs.
    """
    return _collect_console_errors(page)


@pytest.fixture
def staging_console_errors(staging_page):
    """
    Collect console errors for the module-scoped staging page.

    Parameters
    ----------
    staging_page : playwright.sync_api.Page
        Module-scoped Playwright page used by staging tests.

    Returns
    -------
    list[str]
        Messages for console entries of type ``error`` and uncaught page
        exceptions.  The list is populated as the test runs.
    """
    return _collect_console_errors(staging_page)


@pytest.fixture(scope="session")
def monkeypatch_session():
    """
    Session-scoped monkeypatch fixture for module-level fixtures.

    This allows module-scoped fixtures to use monkeypatch functionality.
    """
    from _pytest.monkeypatch import MonkeyPatch

    m = MonkeyPatch()
    yield m
    m.undo()


def set_test_db(db_path):
    """
    Helper function to set PAPER_DB environment variable and reload config.

    This reduces code duplication across test files where the pattern
    os.environ["PAPER_DB"] = str(db_path) followed by get_config(reload=True)
    is repeated many times.

    Uses .env.test for all other config values to ensure test consistency.

    Parameters
    ----------
    db_path : str or Path
        Path to the database file

    Examples
    --------
    >>> set_test_db(tmp_path / "test.db")

    Notes
    -----
    This function uses os.environ directly instead of monkeypatch.setenv
    so it can be used in contexts where monkeypatch is not available
    (e.g., in helper functions).
    """
    import os

    os.environ["PAPER_DB"] = str(db_path)

    # Reload config with .env.test to ensure consistent test configuration
    # The PAPER_DB environment variable will override the value from .env.test
    get_config(reload=True, env_path=get_env_test_path())


def set_test_embedding_db(embedding_db_path):
    """
    Helper function to set PAPER_DB environment variable and reload config.

    This reduces code duplication across test files where the pattern
    os.environ["PAPER_DB"] = str(db_path) followed by get_config(reload=True)
    is repeated many times.

    Uses .env.test for all other config values to ensure test consistency.

    Parameters
    ----------
    db_path : str or Path
        Path to the database file

    Examples
    --------
    >>> set_test_db(tmp_path / "test.db")

    Notes
    -----
    This function uses os.environ directly instead of monkeypatch.setenv
    so it can be used in contexts where monkeypatch is not available
    (e.g., in helper functions).
    """
    import os

    os.environ["EMBEDDING_DB"] = str(embedding_db_path)

    # Reload config with .env.test to ensure consistent test configuration
    # The PAPER_DB environment variable will override the value from .env.test
    get_config(reload=True, env_path=get_env_test_path())


@pytest.fixture(scope="session", autouse=True)
def test_config():
    """
    Ensure tests use predictable configuration from .env.test.

    This fixture loads configuration directly from .env.test to provide
    consistent defaults for all tests, preventing tests from being affected
    by the user's custom .env file.

    The fixture runs automatically for all tests (autouse=True) at session scope.
    Individual tests can override specific environment variables as needed.

    Note: End-to-end tests that require a working LM Studio connection should
    not use .env.test and will need to reload config without env_path.
    """
    from abstracts_explorer.config import get_config

    # Find .env.test file
    env_test = get_env_test_path()

    if not env_test.exists():
        # If .env.test doesn't exist, skip this fixture
        yield
        return

    # Force config reload with .env.test
    # This ensures all tests start with predictable configuration values
    get_config(reload=True, env_path=env_test)

    yield

    # Reload config without env_path to restore user's configuration
    # This uses the default behavior (searching for .env file)
    get_config(reload=True)


@pytest.fixture
def db_manager(tmp_path, monkeypatch):
    """
    Create a DatabaseManager instance with a temporary database.

    Parameters
    ----------
    tmp_path : Path
        Pytest's temporary path fixture
    monkeypatch : MonkeyPatch
        Pytest's monkeypatch fixture for setting environment variables

    Returns
    -------
    DatabaseManager
        Database manager instance with temporary database

    Notes
    -----
    Sets PAPER_DB environment variable to point to a temporary database
    and reloads config to pick up the change.
    """
    db_path = tmp_path / "test.db"
    set_test_db(db_path)

    return DatabaseManager()


@pytest.fixture
def connected_db(db_manager):
    """
    Create and connect to a database with tables created.

    Parameters
    ----------
    db_manager : DatabaseManager
        Database manager fixture

    Yields
    ------
    DatabaseManager
        Connected database manager with tables created

    Notes
    -----
    Automatically closes the database connection after the test.
    """
    db_manager.create_tables()
    yield db_manager
    db_manager.close()


@pytest.fixture
def sample_neurips_data():
    """
    Sample data with lightweight schema for testing.

    Returns
    -------
    list
        List of paper dictionaries with lightweight schema

    Notes
    -----
    This data uses the lightweight schema with semicolon-separated authors.
    Includes two papers for testing purposes.
    """
    return [
        {
            "id": 123456,
            "uid": "abc123",
            "title": "Deep Learning with Neural Networks",
            "abstract": "This paper explores deep neural networks",
            "authors": ["Miaomiao Huang", "John Smith"],
            "keywords": ["deep learning", "neural networks"],
            "session": "Session A",
            "poster_position": "A-1",
            "room_name": "Hall A",
            "url": "https://openreview.net/forum?id=abc123",
            "paper_pdf_url": "https://openreview.net/pdf?id=abc123",
            "starttime": "2025-12-10T10:00:00",
            "endtime": "2025-12-10T12:00:00",
            "year": 2025,
            "conference": "NeurIPS",
        },
        {
            "id": 123457,
            "uid": "def456",
            "title": "Advances in Computer Vision",
            "abstract": "This paper discusses computer vision techniques",
            "authors": ["John Smith", "Jane Doe"],
            "keywords": ["computer vision", "image processing"],
            "session": "Session B",
            "poster_position": "B-2",
            "room_name": "Hall B",
            "url": "https://openreview.net/forum?id=def456",
            "paper_pdf_url": "https://openreview.net/pdf?id=def456",
            "starttime": "2025-12-10T14:00:00",
            "endtime": "2025-12-10T16:00:00",
            "award": "Best Paper Award",
            "year": 2025,
            "conference": "NeurIPS",
        },
    ]


@pytest.fixture
def test_database(tmp_path, monkeypatch):
    """
    Create a test database with sample papers for testing.

    Parameters
    ----------
    tmp_path : Path
        Pytest's temporary path fixture
    monkeypatch : MonkeyPatch
        Pytest's monkeypatch fixture for setting environment variables

    Returns
    -------
    Path
        Path to the test database file

    Notes
    -----
    Creates a database with 3 papers using the lightweight schema
    via DatabaseManager for testing search and retrieval functionality.
    Sets PAPER_DB environment variable to point to the test database.
    """
    from abstracts_explorer.database import DatabaseManager
    from abstracts_explorer.plugin import LightweightPaper

    db_path = tmp_path / "test.db"

    # Set PAPER_DB to point to our test database
    set_test_db(db_path)

    # Use DatabaseManager to create the database with proper schema
    with DatabaseManager() as db:
        db.create_tables()

        # Create sample papers using LightweightPaper model
        papers = [
            LightweightPaper(
                original_id=1,
                title="Deep Learning Paper",
                abstract="This paper presents a novel deep learning approach.",
                authors=["John Doe", "Jane Smith"],
                keywords=["deep learning", "neural networks"],
                session="ML Session 1",
                poster_position="A12",
                paper_pdf_url="https://papers.nips.cc/paper/1",
                year=2025,
                conference="NeurIPS",
            ),
            LightweightPaper(
                original_id=2,
                title="NLP Paper",
                abstract="We introduce a new natural language processing method.",
                authors=["Alice Johnson"],
                keywords=["NLP", "transformers"],
                session="NLP Session 2",
                poster_position="B05",
                paper_pdf_url="https://papers.nips.cc/paper/2",
                year=2025,
                conference="NeurIPS",
            ),
            LightweightPaper(
                original_id=3,
                title="Computer Vision Paper",
                abstract="This paper explores computer vision with CNNs.",
                authors=["Bob Wilson"],
                keywords=["vision", "CNN"],
                session="CV Session 3",
                poster_position="",
                paper_pdf_url="",
                year=2025,
                conference="NeurIPS",
            ),
        ]

        # Add papers to database
        for paper in papers:
            db.add_paper(paper)

    return db_path


@pytest.fixture
def mock_lm_studio():
    """
    Mock LM Studio API responses for testing embeddings.

    Yields
    ------
    Mock
        Mock OpenAI client with configured responses

    Notes
    -----
    Mocks both embedding and chat completion endpoints with typical responses.
    """
    from unittest.mock import patch, Mock

    with patch("abstracts_explorer.embeddings.models.OpenAI") as mock_openai_class:
        # Create mock OpenAI client instance
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock models.list() for connection test
        mock_models = Mock()
        mock_client.models.list.return_value = mock_models

        # Mock embeddings.create() for single and batched embedding generation
        def _create_embeddings(*args, **kwargs):
            inputs = kwargs.get("input")
            texts = [inputs] if isinstance(inputs, str) else list(inputs or [])
            response = Mock()
            response.data = []
            for index in range(len(texts)):
                item = Mock()
                item.embedding = [0.1] * 4096
                item.index = index
                response.data.append(item)
            return response

        mock_client.embeddings.create.side_effect = _create_embeddings

        yield mock_client


@pytest.fixture
def embeddings_manager(tmp_path, monkeypatch):
    """
    Create an EmbeddingsManager instance for testing.

    Parameters
    ----------
    tmp_path : Path
        Pytest's temporary path fixture
    monkeypatch : MonkeyPatch
        Pytest's monkeypatch fixture

    Returns
    -------
    EmbeddingsManager
        Embeddings manager with temporary ChromaDB path

    Notes
    -----
    Uses a test collection name and temporary ChromaDB storage path.
    Mocks the OpenAI client to avoid real API calls.
    """
    from unittest.mock import Mock

    chroma_path = tmp_path / "test_chroma"

    # Set the embedding_db path via environment variable
    monkeypatch.setenv("EMBEDDING_DB", str(chroma_path))
    # Keep the default collection name in sync with the fixture manager so that
    # managers created inside production code use the same test collection.
    monkeypatch.setenv("COLLECTION_NAME", "test_collection")

    # Force config reload to pick up the environment variable
    # Use .env.test for consistent test configuration
    from abstracts_explorer.config import get_config

    _ = get_config(reload=True, env_path=get_env_test_path())  # Force reload but don't need the result

    # Create the manager
    em = EmbeddingsManager(
        backend_url="http://localhost:1234",
        collection_name="test_collection",
    )

    # Replace the OpenAI client with a mock
    mock_client = Mock()

    # Mock embeddings.create() for single and batched embedding generation
    def _create_embeddings(*args, **kwargs):
        inputs = kwargs.get("input")
        texts = [inputs] if isinstance(inputs, str) else list(inputs or [])
        response = Mock()
        response.data = []
        for index in range(len(texts)):
            item = Mock()
            item.embedding = [0.1] * 4096
            item.index = index
            response.data.append(item)
        return response

    mock_client.embeddings.create.side_effect = _create_embeddings

    # Mock models.list()
    mock_client.models.list.return_value = Mock()

    # Set the private attribute directly to avoid triggering lazy loading
    em._openai_client = mock_client

    return em


@pytest.fixture
def mock_embeddings_manager():
    """
    Create a mock embeddings manager for RAG testing.

    Returns
    -------
    Mock
        Mock EmbeddingsManager with configured search results

    Notes
    -----
    Returns mock search results with 3 papers about transformers and language models.
    Uses string UIDs as required by the lightweight database schema.
    """
    mock_em = Mock(spec=EmbeddingsManager)

    # Mock successful search results with STRING UIDs (lightweight schema)
    mock_em.search_similar.return_value = {
        "ids": [["1", "2", "3"]],
        "distances": [[0.1, 0.2, 0.3]],
    }

    return mock_em


@pytest.fixture
def mock_response(sample_neurips_data):
    """
    Mock HTTP response for downloader tests.

    Parameters
    ----------
    sample_neurips_data : list
        Sample NeurIPS data fixture

    Returns
    -------
    Mock
        Mock response object with JSON data

    Notes
    -----
    Simulates a successful HTTP response from the NeurIPS API.
    """
    mock = Mock()
    mock.status_code = 200
    mock.json.return_value = sample_neurips_data
    mock.raise_for_status = Mock()
    return mock


@pytest.fixture(scope="module")
def web_test_papers():
    """
    Create a list of test papers for web interface testing.

    Returns
    -------
    list of LightweightPaper
        Three test papers with diverse content for web testing

    Notes
    -----
    This fixture provides consistent test data for both test_web.py and
    test_web_integration.py, reducing code duplication.
    """
    return [
        LightweightPaper(
            title="Attention is All You Need",
            authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
            abstract="We propose the Transformer, a model architecture based solely on attention mechanisms.",
            session="Oral Session 1",
            poster_position="O1",
            year=2017,
            conference="NeurIPS",
            keywords=["attention", "transformer", "neural networks"],
        ),
        LightweightPaper(
            title="BERT: Pre-training of Deep Bidirectional Transformers",
            authors=["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee"],
            abstract="We introduce BERT, which stands for Bidirectional Encoder Representations from Transformers.",
            session="Poster Session 2",
            poster_position="P42",
            year=2019,
            conference="NeurIPS",
            keywords=["bert", "pretraining", "transformers"],
        ),
        LightweightPaper(
            title="Deep Residual Learning for Image Recognition",
            authors=["Kaiming He", "Xiangyu Zhang", "Shaoqing Ren"],
            abstract="We present a residual learning framework to ease the training of networks.",
            session="Oral Session 3",
            poster_position="O2",
            year=2016,
            conference="NeurIPS",
            keywords=["resnet", "computer vision", "deep learning"],
        ),
    ]

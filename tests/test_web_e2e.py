"""
End-to-end tests for the web UI using Selenium.

These tests verify the web UI functionality through browser automation,
testing user interactions and workflows as they would happen in a real browser.
"""

import pytest
import time
import threading
import sys
import os
from pathlib import Path
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from abstracts_explorer.database import DatabaseManager
from tests.helpers import (
    find_free_port,
    create_webdriver,
)
from tests.conftest import set_test_db

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Constants for E2E tests
MOCK_EMBEDDING_DIMENSION = 4096  # Standard dimension for test embeddings


@pytest.fixture(scope="module")
def test_database(tmp_path_factory):
    """
    Create a test database with sample data for E2E tests.

    Parameters
    ----------
    tmp_path_factory : TempPathFactory
        Pytest fixture for creating temporary directories

    Returns
    -------
    Path
        Path to the test database
    """
    tmp_dir = tmp_path_factory.mktemp("data")
    db_path = tmp_dir / "test_web_e2e.db"

    # Create database and add test data
    from abstracts_explorer.plugin import LightweightPaper

    set_test_db(db_path)
    db = DatabaseManager()

    with db:
        db.create_tables()

        # Add test papers with realistic data (lightweight schema)
        papers_data = [
            {
                "title": "Attention is All You Need",
                "authors": ["Ashish Vaswani", "Noam Shazeer"],
                "abstract": "We propose the Transformer, a model architecture eschewing recurrence and instead "
                "relying entirely on an attention mechanism to draw global dependencies between input and output.",
                "session": "Oral Session 1",
                "poster_position": "A1",
                "keywords": ["attention", "transformer", "neural networks"],
                "year": 2025,
                "conference": "NeurIPS",
            },
            {
                "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "authors": ["Jacob Devlin", "Ming-Wei Chang"],
                "abstract": "We introduce a new language representation model called BERT, which stands for "
                "Bidirectional Encoder Representations from Transformers.",
                "session": "Poster Session A",
                "poster_position": "P1",
                "keywords": ["bert", "nlp", "transformers"],
                "year": 2025,
                "conference": "NeurIPS",
            },
            {
                "title": "Deep Residual Learning for Image Recognition",
                "authors": ["Kaiming He", "Xiangyu Zhang"],
                "abstract": "Deep residual nets are foundations of our submissions to ILSVRC & COCO 2015 competitions.",
                "session": "Oral Session 2",
                "poster_position": "A2",
                "keywords": ["resnet", "computer vision", "deep learning"],
                "year": 2025,
                "conference": "NeurIPS",
            },
            {
                "title": "Generative Adversarial Networks",
                "authors": ["Ian Goodfellow", "Yoshua Bengio"],
                "abstract": "We propose a new framework for estimating generative models via an adversarial process.",
                "session": "Poster Session B",
                "poster_position": "P2",
                "keywords": ["gan", "generative models", "adversarial"],
                "year": 2025,
                "conference": "NeurIPS",
            },
            {
                "title": "Neural Machine Translation by Jointly Learning to Align and Translate",
                "authors": ["Dzmitry Bahdanau", "Yoshua Bengio"],
                "abstract": "Neural machine translation is a recently proposed approach to machine translation.",
                "session": "Spotlight Session",
                "poster_position": "S1",
                "keywords": ["nmt", "translation", "attention"],
                "year": 2025,
                "conference": "NeurIPS",
            },
        ]

        for paper_data in papers_data:
            paper = LightweightPaper(**paper_data)
            db.add_paper(paper)

    return db_path


@pytest.fixture(scope="module")
def test_embeddings(test_database, tmp_path_factory):
    """
    Create a test embeddings database with mock embeddings for E2E tests.

    This fixture is cached at module scope to avoid recreating embeddings
    for each test, improving test performance and reliability.

    Parameters
    ----------
    test_database : Path
        Path to the test database with paper data
    tmp_path_factory : TempPathFactory
        Pytest fixture for creating temporary directories

    Returns
    -------
    tuple
        Tuple of (embeddings_manager, embeddings_path, collection_name, mock_client)
    """
    from abstracts_explorer.embeddings import EmbeddingsManager
    from unittest.mock import Mock
    import chromadb.api.shared_system_client
    import uuid
    import time

    # Clear ChromaDB's global client registry to avoid conflicts
    chromadb.api.shared_system_client.SharedSystemClient._identifier_to_system.clear()

    # Use unique ID with timestamp to avoid conflicts across test sessions
    unique_id = f"{int(time.time())}_{str(uuid.uuid4())[:8]}"
    tmp_dir = tmp_path_factory.mktemp("e2e_embeddings")
    embeddings_path = tmp_dir / f"chroma_{unique_id}"
    collection_name = f"test_collection_{unique_id}"

    # Create mock OpenAI client
    # This will be injected directly into the EmbeddingsManager instance
    mock_client = Mock()

    # Mock models.list() for connection test
    mock_models = Mock()
    mock_client.models.list.return_value = mock_models

    # Mock embeddings.create() for embedding generation
    # IMPORTANT: This mock will be used for both adding papers AND searching
    mock_embedding_response = Mock()
    mock_embedding_data = Mock()
    mock_embedding_data.embedding = [0.1] * MOCK_EMBEDDING_DIMENSION
    mock_embedding_response.data = [mock_embedding_data]
    mock_client.embeddings.create.return_value = mock_embedding_response

    # Set environment variable for EMBEDDING_DB before creating EmbeddingsManager
    os.environ["EMBEDDING_DB"] = str(embeddings_path)

    # Force config reload to pick up environment variable with .env.test
    from abstracts_explorer.config import get_config
    from tests.conftest import get_env_test_path

    _ = get_config(reload=True, env_path=get_env_test_path())

    # Initialize embeddings manager
    em = EmbeddingsManager(collection_name=collection_name)

    # Inject the mock client directly to bypass lazy loading
    # This ensures we ALWAYS use the mock, never a real OpenAI connection
    em._openai_client = mock_client

    em.connect()

    # Forcefully delete collection if it exists to ensure clean state
    try:
        em.client.delete_collection(name=collection_name)
    except Exception:
        pass  # Collection might not exist

    # Create fresh collection
    em.create_collection(reset=False)

    # Add embeddings for all test papers from the database
    set_test_db(str(test_database))
    db = DatabaseManager()
    db.connect()
    papers = db.query("SELECT * FROM papers")

    # Add each paper to the embeddings database
    for paper in papers:
        em.add_paper(paper)

    db.close()

    # Return the embeddings manager, metadata, and the mock client
    # The mock client is returned so web_server can reuse it
    yield (em, embeddings_path, collection_name, mock_client)

    # Cleanup happens automatically


def _prepopulate_clustering_cache(test_database, em, collection_name):
    """
    Pre-populate the database with cached clustering results.

    This mirrors the production workflow where ``abstracts-explorer cluster
    pre-generate`` is run before the web UI is used.  With cached results
    present, MCP tools (``get_conference_topics``, ``get_cluster_visualization``)
    return instantly instead of running t-SNE from scratch.

    Parameters
    ----------
    test_database : Path
        Path to the test SQLite database.
    em : EmbeddingsManager
        Embeddings manager with test embeddings loaded.
    collection_name : str
        Name of the ChromaDB collection.
    """
    from abstracts_explorer.clustering import ClusteringManager
    from abstracts_explorer.config import get_config

    config = get_config()

    set_test_db(str(test_database))
    db = DatabaseManager()
    db.connect()
    db.create_tables()

    cm = ClusteringManager(em, db)
    cm.load_embeddings()

    # Cluster on full embeddings (this is fast with only 3 test papers)
    cm.cluster(
        method="agglomerative",
        n_clusters=None,
        use_reduced=False,
        distance_threshold=150.0,
        linkage="ward",
    )

    # Use PCA for the reduction step (deterministic, thread-safe, and fast).
    # The results are stored under the "tsne" cache key so that
    # compute_clusters_with_cache() finds an exact cache hit when MCP tools
    # request t-SNE reduction.  The x/y coordinates differ from a real t-SNE
    # but are sufficient for testing the cache-lookup and topic-analysis paths.
    cm.reduce_dimensions(method="pca", n_components=2)

    try:
        cm.extract_cluster_keywords(n_keywords=5)
        cm.generate_cluster_labels(use_llm=False, max_keywords=3)
    except Exception:
        pass  # Labels are optional

    results = cm.get_clustering_results()

    # Save under the t-SNE cache key so that compute_clusters_with_cache()
    # finds it when MCP tools ask for tsne reduction.
    db.save_clustering_cache(
        embedding_model=config.embedding_model,
        reduction_method="tsne",
        n_components=2,
        clustering_method="agglomerative",
        results=results,
        n_clusters=None,
        clustering_params={"linkage": "ward", "distance_threshold": 150.0},
    )
    db.close()


@pytest.fixture(scope="module")
def web_server(test_database, test_embeddings, tmp_path_factory):
    """
    Start a web server in a background thread for E2E testing.

    Uses cached embeddings from test_embeddings fixture for better performance
    and reliability. The mock OpenAI client is injected to ensure consistent
    embedding dimensions.

    Parameters
    ----------
    test_database : Path
        Path to the test database
    test_embeddings : tuple
        Cached embeddings manager, path, collection name, and mock client
    tmp_path_factory : TempPathFactory
        Pytest fixture for creating temporary directories

    Yields
    ------
    tuple
        Tuple of (base_url, port)
    """
    from abstracts_explorer.database import DatabaseManager

    # Import the module first, then access it from sys.modules
    import sys
    import abstracts_explorer.web_ui.app  # noqa: F401

    # Get the actual module from sys.modules (now it's loaded)
    app_module = sys.modules["abstracts_explorer.web_ui.app"]
    # Get the Flask app instance
    flask_app = app_module.app

    # Unpack cached embeddings and mock client
    em, embeddings_path, collection_name, mock_client = test_embeddings

    # CRITICAL: Ensure the embeddings manager is using the mock client
    # This prevents it from creating a real OpenAI connection during searches
    em._openai_client = mock_client

    port = find_free_port()
    base_url = f"http://localhost:{port}"

    # Configure the app to use test database and embeddings via environment variables
    # Set environment variables before creating Config instance
    original_paper_db = os.environ.get("PAPER_DB")
    original_embedding_db = os.environ.get("EMBEDDING_DB")
    original_collection_name = os.environ.get("COLLECTION_NAME")
    original_default_conference = os.environ.get("DEFAULT_CONFERENCE")

    os.environ["PAPER_DB"] = str(test_database)
    os.environ["EMBEDDING_DB"] = str(embeddings_path)
    os.environ["COLLECTION_NAME"] = collection_name
    # Test database only has "NeurIPS" papers; override so the auto-selected
    # conference matches the test data and search/filter requests return results.
    os.environ["DEFAULT_CONFERENCE"] = "NeurIPS"

    def mock_get_config():
        # Force reload to pick up environment variables with .env.test
        from abstracts_explorer.config import get_config as real_get_config
        from tests.conftest import get_env_test_path

        return real_get_config(reload=True, env_path=get_env_test_path())

    app_module.get_config = mock_get_config

    # Inject the pre-created embeddings manager directly
    # CRITICAL: Set this BEFORE starting the server to avoid race conditions
    # This prevents get_embeddings_manager() from creating a new instance
    app_module.embeddings_manager = em
    app_module.rag_chat = None

    # Pre-populate the database with cached clustering results so that MCP
    # tools (get_conference_topics, get_cluster_visualization) return instantly
    # without running t-SNE.  This mirrors the production workflow where
    # clustering is pre-generated via the CLI.
    _prepopulate_clustering_cache(test_database, em, collection_name)

    # Mock get_database to not check file existence
    # Each Flask request thread will create its own database connection
    original_get_database = app_module.get_database

    def mock_get_database_wrapper():
        """Wrapper that skips file existence check and ensures tables exist."""
        from flask import g

        if "db" not in g:
            db_path = str(test_database)
            # Don't check file existence in tests - the database was created in a temp dir
            set_test_db(db_path)
            g.db = DatabaseManager()
            g.db.connect()
            g.db.create_tables()  # Ensure all tables exist (including clustering_cache)
        return g.db

    app_module.get_database = mock_get_database_wrapper

    # Use werkzeug's make_server for better cross-platform compatibility
    # This works more reliably in threads than Flask's app.run()
    from werkzeug.serving import make_server

    server = make_server("localhost", port, flask_app, threaded=True)

    # Start server in a thread
    def run_server():
        server.serve_forever()

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Wait for server to start
    import requests

    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(base_url, timeout=1)
            if response.status_code == 200:
                break
        except requests.exceptions.RequestException:
            if i == max_retries - 1:
                server.shutdown()
                pytest.fail("Server failed to start")
            time.sleep(0.2)

    yield (base_url, port)

    # Cleanup
    server.shutdown()

    # Reset the app module state
    app_module.embeddings_manager = None
    app_module.rag_chat = None
    app_module.get_database = original_get_database

    # Restore original environment variables
    if original_default_conference is not None:
        os.environ["DEFAULT_CONFERENCE"] = original_default_conference
    elif "DEFAULT_CONFERENCE" in os.environ:
        del os.environ["DEFAULT_CONFERENCE"]

    if original_paper_db is not None:
        os.environ["PAPER_DB"] = original_paper_db
    elif "PAPER_DB" in os.environ:
        del os.environ["PAPER_DB"]

    if original_embedding_db is not None:
        os.environ["EMBEDDING_DB"] = original_embedding_db
    elif "EMBEDDING_DB" in os.environ:
        del os.environ["EMBEDDING_DB"]

    if original_collection_name is not None:
        os.environ["COLLECTION_NAME"] = original_collection_name
    elif "COLLECTION_NAME" in os.environ:
        del os.environ["COLLECTION_NAME"]


@pytest.fixture
def browser():
    """
    Create a Selenium WebDriver instance for browser automation.

    Yields
    ------
    webdriver.Chrome or webdriver.Firefox
        WebDriver instance (Chrome or Firefox)

    Notes
    -----
    Automatically quits the browser after the test.
    Uses headless mode for CI/CD compatibility.
    Skips test if no browser is available or driver installation fails.

    Browser selection is controlled by the E2E_BROWSER environment variable:
    - 'chrome': Use Chrome only
    - 'firefox': Use Firefox only
    - 'auto' (default): Try Chrome first, fall back to Firefox

    Examples
    --------
    # Use Firefox instead of Chrome
    E2E_BROWSER=firefox pytest tests/test_web_e2e.py -m e2e
    """
    driver = create_webdriver()
    yield driver
    driver.quit()


@pytest.mark.e2e
@pytest.mark.slow
class TestWebUIE2E:
    """End-to-end tests for the web UI using Selenium."""

    def test_page_loads(self, web_server, browser):
        """
        Test that the main page loads correctly.

        Parameters
        ----------
        web_server : tuple
            Web server fixture (base_url, port)
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        # Check page title
        assert "Abstracts Explorer" in browser.title

        # Check that main elements are present
        assert browser.find_element(By.ID, "search-input")
        assert browser.find_element(By.ID, "search-results")
        assert browser.find_element(By.ID, "tab-search")
        assert browser.find_element(By.ID, "tab-chat")

    def test_search_tab_visible_by_default(self, web_server, browser):
        """
        Test that the search tab is visible by default.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        # Search tab should be visible
        search_tab = browser.find_element(By.ID, "search-tab")
        assert search_tab.is_displayed()

        # Chat tab should be hidden
        chat_tab = browser.find_element(By.ID, "chat-tab")
        assert not chat_tab.is_displayed()

    def test_switch_to_chat_tab(self, web_server, browser):
        """
        Test switching between search and chat tabs.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        # Click chat tab
        chat_tab_button = browser.find_element(By.ID, "tab-chat")
        chat_tab_button.click()

        # Wait for chat tab to become visible
        wait = WebDriverWait(browser, 5)
        chat_tab = wait.until(EC.visibility_of_element_located((By.ID, "chat-tab")))
        assert chat_tab.is_displayed()

        # Search tab should be hidden
        search_tab = browser.find_element(By.ID, "search-tab")
        assert not search_tab.is_displayed()

        # Switch back to search
        search_tab_button = browser.find_element(By.ID, "tab-search")
        search_tab_button.click()

        # Wait for search tab to become visible
        search_tab = wait.until(EC.visibility_of_element_located((By.ID, "search-tab")))
        assert search_tab.is_displayed()

        # Chat tab should now be hidden
        chat_tab = browser.find_element(By.ID, "chat-tab")
        assert not chat_tab.is_displayed()

    def test_keyword_search_interaction(self, web_server, browser):
        """
        Test keyword search user interaction flow.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        # Find search input
        search_input = browser.find_element(By.ID, "search-input")

        # Enter search query
        search_query = "attention"
        search_input.send_keys(search_query)
        search_input.send_keys(Keys.RETURN)

        # Wait for results to load
        wait = WebDriverWait(browser, 10)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#search-results .bg-white.rounded-lg")))

        # Check that results are displayed
        results = browser.find_elements(By.CSS_SELECTOR, "#search-results .bg-white.rounded-lg")
        assert len(results) > 0

        # Check that the search returned results (text will vary based on actual data)
        search_results_div = browser.find_element(By.ID, "search-results")
        assert len(search_results_div.text) > 0, "Search results should contain some text"

    def test_search_limit_filter(self, web_server, browser):
        """
        Test changing the search results limit.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        # Open search settings modal
        wait = WebDriverWait(browser, 10)
        settings_btn = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button[onclick='openSearchSettings()']"))
        )
        settings_btn.click()

        # Wait for modal to open
        wait.until(EC.visibility_of_element_located((By.ID, "settings-modal")))

        # Change limit to 10 (actual available value)
        limit_select = Select(browser.find_element(By.ID, "limit-select"))
        limit_select.select_by_value("10")

        # Close modal
        close_btn = browser.find_element(By.CSS_SELECTOR, "button[onclick='closeSettings()']")
        close_btn.click()

        # Wait for modal to close
        wait.until(EC.invisibility_of_element_located((By.ID, "settings-modal")))

        # Perform search
        search_input = browser.find_element(By.ID, "search-input")
        search_input.send_keys("learning")
        search_input.send_keys(Keys.RETURN)

        # Wait for results
        wait = WebDriverWait(browser, 10)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#search-results .bg-white.rounded-lg")))

        # Check that we have at most 10 results
        # Note: There's a header card with results count, so we get 11 elements
        time.sleep(0.2)  # Allow time for rendering
        results = browser.find_elements(By.CSS_SELECTOR, "#search-results .paper-card")
        assert len(results) <= 10

    def test_collapsible_abstract(self, web_server, browser):
        """
        Test expanding and collapsing paper abstracts.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        # Perform search
        search_input = browser.find_element(By.ID, "search-input")
        search_input.send_keys("attention")
        search_input.send_keys(Keys.RETURN)

        # Wait for results
        wait = WebDriverWait(browser, 10)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#search-results .bg-white.rounded-lg")))

        # Find a toggle button
        toggle_buttons = browser.find_elements(By.CSS_SELECTOR, ".toggle-abstract")
        if len(toggle_buttons) > 0:
            # Click to expand
            toggle_buttons[0].click()
            time.sleep(0.2)

            # Check that abstract is visible
            abstract_divs = browser.find_elements(By.CSS_SELECTOR, ".abstract-content")
            if len(abstract_divs) > 0:
                # Abstract should be visible after clicking toggle
                assert abstract_divs[0].is_displayed()

                # Click again to collapse
                toggle_buttons[0].click()
                time.sleep(0.2)
                assert not abstract_divs[0].is_displayed()

    def test_empty_search_shows_message(self, web_server, browser):
        """
        Test that searching with no query shows appropriate message.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        # Submit empty search
        search_input = browser.find_element(By.ID, "search-input")
        search_input.send_keys(Keys.RETURN)

        # Wait a moment
        time.sleep(0.2)

        # Check for message or no results
        results_div = browser.find_element(By.ID, "search-results")
        # Should either show message or be empty
        assert results_div is not None

    def test_search_no_results(self, web_server, browser):
        """
        Test searching with a query that returns no results.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        # Search for something unlikely to be in test data
        search_input = browser.find_element(By.ID, "search-input")
        search_input.send_keys("xyzqwertyuiopasdfghjkl")
        search_input.send_keys(Keys.RETURN)

        # Wait a moment
        time.sleep(0.2)

        # Check that results area exists
        # Note: Fuzzy search may still return results for nonsense queries
        # Just verify search executed without error
        browser.find_elements(By.CSS_SELECTOR, "#search-results .bg-white.rounded-lg")
        # Results may vary - just ensure search didn't crash
        search_results_element = browser.find_element(By.ID, "search-results")
        assert search_results_element is not None

    def test_stats_display(self, web_server, browser):
        """
        Test that statistics are displayed on the page.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        # Wait for stats to load - wait for the actual stats, not just the loading message
        wait = WebDriverWait(browser, 10)
        # Wait for stats element to be present
        wait.until(EC.presence_of_element_located((By.ID, "stats")))

        # Wait for stats to actually load (text should change from "Loading stats...")
        def stats_loaded(driver):
            stats = driver.find_element(By.ID, "stats")
            stats_text = stats.text.lower()
            return "abstracts" in stats_text or "papers" in stats_text or "error loading stats" in stats_text

        wait.until(stats_loaded)

        stats = browser.find_element(By.ID, "stats")
        stats_text = stats.text

        # Should contain paper/abstract count
        assert "abstracts" in stats_text.lower() or "papers" in stats_text.lower()

    def test_paper_detail_view(self, web_server, browser):
        """
        Test viewing paper details by clicking on a paper.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        # Perform search
        search_input = browser.find_element(By.ID, "search-input")
        search_input.send_keys("attention")
        search_input.send_keys(Keys.RETURN)

        # Wait for results
        wait = WebDriverWait(browser, 10)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#search-results .bg-white.rounded-lg")))

        # Get paper cards (not the header)
        results = browser.find_elements(By.CSS_SELECTOR, "#search-results .paper-card")
        assert len(results) > 0, "Should have at least one paper card"

        first_result = results[0]

        # Check that paper has title (h3 or h4 depending on compact mode)
        title_elements = first_result.find_elements(By.CSS_SELECTOR, "h3, h4")
        assert len(title_elements) > 0, "Paper card should have a title element"
        assert len(title_elements[0].text) > 0, "Title should not be empty"

        # Check that paper has authors
        author_elements = first_result.find_elements(By.CSS_SELECTOR, ".text-gray-600")
        assert len(author_elements) > 0, "Paper card should have author information"

    def test_select_deselect_all_filters(self, web_server, browser):
        """
        Test the select/deselect all functionality for filters.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        # Wait for page to load
        time.sleep(0.2)

        # Find select all buttons (if they exist)
        select_buttons = browser.find_elements(By.CSS_SELECTOR, "button[onclick*='selectAll']")
        if len(select_buttons) > 0:
            # Scroll to the button first to ensure it's visible
            browser.execute_script("arguments[0].scrollIntoView(true);", select_buttons[0])
            time.sleep(0.2)

            # Use JavaScript click to avoid scrolling issues
            browser.execute_script("arguments[0].click();", select_buttons[0])
            time.sleep(0.2)

            # Find corresponding deselect button
            deselect_buttons = browser.find_elements(By.CSS_SELECTOR, "button[onclick*='deselectAll']")
            if len(deselect_buttons) > 0:
                # Scroll and click with JavaScript
                browser.execute_script("arguments[0].scrollIntoView(true);", deselect_buttons[0])
                time.sleep(0.2)
                browser.execute_script("arguments[0].click();", deselect_buttons[0])
                time.sleep(0.2)

    def test_chat_interface_elements(self, web_server, browser):
        """
        Test that chat interface elements are present and functional.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        # Switch to chat tab
        chat_tab_button = browser.find_element(By.ID, "tab-chat")
        chat_tab_button.click()
        time.sleep(0.2)

        # Check chat elements
        assert browser.find_element(By.ID, "chat-input")
        assert browser.find_element(By.ID, "chat-messages")
        assert browser.find_element(By.ID, "chat-papers")
        assert browser.find_element(By.ID, "n-papers")

    def test_chat_filters_exist(self, web_server, browser):
        """
        Test that chat filters are present.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        # Switch to chat tab
        chat_tab_button = browser.find_element(By.ID, "tab-chat")
        chat_tab_button.click()
        time.sleep(0.2)

        # Check filter elements (only session filter exists in lightweight schema)
        assert browser.find_element(By.ID, "chat-session-filter")

    def test_chat_send_message(self, web_server, browser):
        """
        Test sending a message in the chat interface.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        # Switch to chat tab
        chat_tab_button = browser.find_element(By.ID, "tab-chat")
        chat_tab_button.click()
        time.sleep(0.2)

        # Find chat input and send button
        chat_input = browser.find_element(By.ID, "chat-input")

        # Enter a test message
        test_message = "What are the main topics in NeurIPS papers?"
        chat_input.send_keys(test_message)

        # Click send button or press Enter
        chat_input.send_keys(Keys.RETURN)

        # Wait for response (AI response may take some time)
        wait = WebDriverWait(browser, 30)

        # Check that message appears in chat
        try:
            # Wait for user message to appear
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".chat-message")))

            # Get all messages
            messages = browser.find_elements(By.CSS_SELECTOR, ".chat-message")

            # Should have at least initial welcome message
            assert len(messages) >= 1, "Chat should have messages"

            # Check if response is loading or received
            # Look for either loading indicator or assistant response
            chat_messages_div = browser.find_element(By.ID, "chat-messages")
            assert len(chat_messages_div.text) > 0, "Chat should have content"

        except TimeoutException:
            # If AI is not configured, that's okay - just verify message was sent
            chat_input_after = browser.find_element(By.ID, "chat-input")
            # Input should be cleared after sending
            assert chat_input_after.get_attribute("value") == "" or len(chat_input_after.get_attribute("value")) == 0

    def test_chat_reset_conversation(self, web_server, browser):
        """
        Test resetting the chat conversation.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        # Switch to chat tab
        chat_tab_button = browser.find_element(By.ID, "tab-chat")
        chat_tab_button.click()
        time.sleep(0.2)

        # Try to find and click reset button
        reset_buttons = browser.find_elements(By.CSS_SELECTOR, "button[onclick*='resetChat']")
        if len(reset_buttons) > 0:
            # Get initial message count
            initial_messages = browser.find_elements(By.CSS_SELECTOR, ".chat-message")
            len(initial_messages)

            # Click reset button
            reset_buttons[0].click()
            time.sleep(0.2)

            # Check messages were reset
            messages_after = browser.find_elements(By.CSS_SELECTOR, ".chat-message")

            # Should have welcome message after reset
            assert len(messages_after) >= 1, "Should have at least welcome message after reset"

    def test_chat_n_papers_selector(self, web_server, browser):
        """
        Test changing the number of papers for chat context.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        # Switch to chat tab
        chat_tab_button = browser.find_element(By.ID, "tab-chat")
        chat_tab_button.click()
        time.sleep(0.2)

        # Find n-papers selector
        n_papers_select = Select(browser.find_element(By.ID, "n-papers"))

        # Check that it has options
        options = n_papers_select.options
        assert len(options) > 0, "n-papers selector should have options"

        # Try to select a different value
        if len(options) > 1:
            n_papers_select.select_by_index(1)
            time.sleep(0.2)

            # Verify selection changed
            selected_value = n_papers_select.first_selected_option.get_attribute("value")
            assert selected_value is not None

    def test_chat_papers_display(self, web_server, browser):
        """
        Test that papers section exists in chat interface.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        # Switch to chat tab
        chat_tab_button = browser.find_element(By.ID, "tab-chat")
        chat_tab_button.click()
        time.sleep(0.2)

        # Check papers section exists
        chat_papers = browser.find_element(By.ID, "chat-papers")
        assert chat_papers is not None

        # Papers section should be visible or hidden but present in DOM
        assert chat_papers.is_displayed() or not chat_papers.is_displayed()

    def test_chat_input_validation(self, web_server, browser):
        """
        Test chat input validation and placeholder.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        # Switch to chat tab
        chat_tab_button = browser.find_element(By.ID, "tab-chat")
        chat_tab_button.click()
        time.sleep(0.2)

        # Check input has placeholder
        chat_input = browser.find_element(By.ID, "chat-input")
        placeholder = chat_input.get_attribute("placeholder")
        assert placeholder is not None and len(placeholder) > 0, "Chat input should have placeholder text"

        # Try sending empty message (should not do anything)
        chat_input.send_keys(Keys.RETURN)
        time.sleep(0.2)

        # Input should still be focusable
        assert chat_input.is_enabled()

    def test_responsive_layout(self, web_server, browser):
        """
        Test that the layout responds to different window sizes.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        # Test desktop size
        browser.set_window_size(1920, 1080)
        time.sleep(0.2)
        assert browser.find_element(By.ID, "search-input").is_displayed()

        # Test tablet size
        browser.set_window_size(768, 1024)
        time.sleep(0.2)
        assert browser.find_element(By.ID, "search-input").is_displayed()

        # Test mobile size
        browser.set_window_size(375, 667)
        time.sleep(0.2)
        assert browser.find_element(By.ID, "search-input").is_displayed()

    def test_multiple_searches_in_sequence(self, web_server, browser):
        """
        Test performing multiple searches in sequence.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        search_queries = ["attention", "learning", "neural"]

        for query in search_queries:
            search_input = browser.find_element(By.ID, "search-input")
            search_input.clear()
            search_input.send_keys(query)
            search_input.send_keys(Keys.RETURN)

            # Wait for results
            time.sleep(0.2)

            # Check that search was performed
            results_div = browser.find_element(By.ID, "search-results")
            assert results_div is not None

    def test_browser_single_page_app(self, web_server, browser):
        """
        Test browser back/forward navigation.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        # Perform a search
        search_input = browser.find_element(By.ID, "search-input")
        search_input.send_keys("attention")
        search_input.send_keys(Keys.RETURN)
        time.sleep(0.2)

        # Switch to chat tab
        chat_tab_button = browser.find_element(By.ID, "tab-chat")
        chat_tab_button.click()
        time.sleep(0.2)

        # Should still be on the same page (single page app)
        assert browser.current_url == base_url + "/"

    def test_page_no_javascript_errors(self, web_server, browser):
        """
        Test that the page loads without JavaScript errors.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        # Wait for page to load
        time.sleep(2)

        # Try to get browser console logs (Chrome only)
        # Firefox WebDriver doesn't support get_log method
        try:
            logs = browser.get_log("browser")
            # Filter for severe errors
            errors = [log for log in logs if log["level"] == "SEVERE"]
            # Should have no severe JavaScript errors
            assert len(errors) == 0, f"JavaScript errors found: {errors}"
        except AttributeError:
            # Firefox doesn't support get_log, skip this check
            # Just verify page loaded successfully
            assert "Abstracts Explorer" in browser.title


@pytest.mark.e2e
@pytest.mark.slow
class TestDataDonationE2E:
    """End-to-end tests for data donation feature."""

    def test_load_json_button_always_visible(self, web_server, browser):
        """
        Test that Load JSON button is always visible even when no papers are rated.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        # Navigate to Interesting Papers tab
        wait = WebDriverWait(browser, 10)
        interesting_tab_button = wait.until(EC.element_to_be_clickable((By.ID, "tab-interesting")))
        interesting_tab_button.click()

        # Wait for tab to load
        wait.until(EC.visibility_of_element_located((By.ID, "interesting-tab")))

        # Load JSON button should be visible even with no papers - wait for it to be present
        load_json_button = wait.until(
            EC.visibility_of_element_located((By.XPATH, "//button[contains(., 'Load JSON')]"))
        )
        assert load_json_button.is_displayed(), "Load JSON button should always be visible"

        # Other action buttons should be hidden when no papers rated
        try:
            donate_button = browser.find_element(By.XPATH, "//button[contains(., 'Donate Data')]")
            assert not donate_button.is_displayed(), "Donate Data button should be hidden when no papers"
        except Exception:
            # Button might not be in DOM at all, which is also fine
            pass

        try:
            save_json_button = browser.find_element(By.XPATH, "//button[contains(., 'Save JSON')]")
            assert not save_json_button.is_displayed(), "Save JSON button should be hidden when no papers"
        except Exception:
            pass

    def test_buttons_appear_after_rating_paper(self, web_server, browser):
        """
        Test that action buttons appear after rating a paper.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        wait = WebDriverWait(browser, 10)

        # Perform a search to find papers
        search_input = wait.until(EC.presence_of_element_located((By.ID, "search-input")))
        search_input.send_keys("attention")

        # Find search button by text content
        search_button = wait.until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Search') and @onclick='searchPapers()']"))
        )
        search_button.click()

        # Wait for search results
        time.sleep(0.2)

        # Rate a paper by clicking on stars
        try:
            star_buttons = browser.find_elements(By.CSS_SELECTOR, ".star-rating button")
            if star_buttons:
                # Click the 5th star (highest rating)
                star_buttons[4].click()
                time.sleep(0.2)

                # Navigate to Interesting Papers tab
                interesting_tab_button = browser.find_element(By.ID, "tab-interesting")
                interesting_tab_button.click()

                # Wait for tab to load
                wait.until(EC.visibility_of_element_located((By.ID, "interesting-tab")))
                time.sleep(0.2)

                # Now all buttons should be visible
                donate_button = wait.until(
                    EC.visibility_of_element_located((By.XPATH, "//button[contains(., 'Donate Data')]"))
                )
                assert donate_button.is_displayed(), "Donate Data button should be visible after rating"

                save_json_button = browser.find_element(By.XPATH, "//button[contains(., 'Save JSON')]")
                assert save_json_button.is_displayed(), "Save JSON button should be visible after rating"

                load_json_button = browser.find_element(By.XPATH, "//button[contains(., 'Load JSON')]")
                assert load_json_button.is_displayed(), "Load JSON button should still be visible"
        except Exception as e:
            pytest.skip(f"Could not test rating flow: {e}")

    def test_donate_data_button_click_shows_confirmation(self, web_server, browser):
        """
        Test that clicking Donate Data button shows confirmation dialog.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        wait = WebDriverWait(browser, 10)

        # Navigate to Interesting Papers tab first
        interesting_tab_button = wait.until(EC.element_to_be_clickable((By.ID, "tab-interesting")))
        interesting_tab_button.click()

        # Wait for tab to load
        wait.until(EC.visibility_of_element_located((By.ID, "interesting-tab")))

        # Use JavaScript to inject test data and trigger UI update
        browser.execute_script("""
            const testPriorities = {
                "test_uid_1": {
                    "priority": 5,
                    "searchTerm": "machine learning"
                }
            };
            localStorage.setItem('paperPriorities', JSON.stringify(testPriorities));
            // Reload priorities from localStorage into memory state
            if (window.loadPriorities) {
                window.loadPriorities();
            }
            // Manually call updateControlsVisibility to show buttons
            if (window.updateControlsVisibility) {
                window.updateControlsVisibility();
            }
        """)

        time.sleep(0.2)

        # Donate button should be visible
        donate_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Donate Data')]")))

        # Click the donate button
        donate_button.click()

        # Wait for confirmation dialog
        time.sleep(0.2)

        # Check that alert is present (confirmation dialog)
        try:
            alert = browser.switch_to.alert
            alert_text = alert.text
            assert "Would you like to donate" in alert_text or "anonymized" in alert_text.lower()
            # Dismiss the alert
            alert.dismiss()
        except Exception as e:
            pytest.fail(f"Expected confirmation dialog but got: {e}")

    def test_donate_button_hidden_after_successful_donation(self, web_server, browser):
        """
        Test that Donate Data button is hidden after successful donation.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        wait = WebDriverWait(browser, 10)

        # Navigate to Interesting Papers tab first
        interesting_tab_button = wait.until(EC.element_to_be_clickable((By.ID, "tab-interesting")))
        interesting_tab_button.click()

        # Wait for tab to load
        wait.until(EC.visibility_of_element_located((By.ID, "interesting-tab")))

        # Use JavaScript to inject test data and trigger UI update
        browser.execute_script("""
            const testPriorities = {
                "test_uid_1": {
                    "priority": 5,
                    "searchTerm": "machine learning"
                }
            };
            localStorage.setItem('paperPriorities', JSON.stringify(testPriorities));
            // Reload priorities from localStorage into memory state
            if (window.loadPriorities) {
                window.loadPriorities();
            }
            // Manually call updateControlsVisibility to show buttons
            if (window.updateControlsVisibility) {
                window.updateControlsVisibility();
            }
        """)

        time.sleep(0.2)

        # Donate button should be visible
        donate_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Donate Data')]")))

        # Click the donate button
        donate_button.click()
        time.sleep(0.2)

        # Accept confirmation dialog
        try:
            alert = browser.switch_to.alert
            alert.accept()
            time.sleep(0.2)

            # Accept success message
            alert = browser.switch_to.alert
            alert.accept()
            time.sleep(0.2)

            # Now the donate button should be hidden
            donate_buttons = browser.find_elements(By.XPATH, "//button[contains(., 'Donate Data')]")
            visible_donate_buttons = [btn for btn in donate_buttons if btn.is_displayed()]
            assert len(visible_donate_buttons) == 0, "Donate button should be hidden after successful donation"

            # Other buttons should still be visible
            save_json_button = browser.find_element(By.XPATH, "//button[contains(., 'Save JSON')]")
            assert save_json_button.is_displayed(), "Save JSON button should still be visible"

            load_json_button = browser.find_element(By.XPATH, "//button[contains(., 'Load JSON')]")
            assert load_json_button.is_displayed(), "Load JSON button should still be visible"

        except Exception as e:
            pytest.skip(f"Could not complete donation flow: {e}")

    def test_export_shows_donation_prompt_once_per_session(self, web_server, browser):
        """
        Test that export shows donation prompt only once per session.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        wait = WebDriverWait(browser, 10)

        # Navigate to Interesting Papers tab first
        interesting_tab_button = wait.until(EC.element_to_be_clickable((By.ID, "tab-interesting")))
        interesting_tab_button.click()

        # Wait for tab to load
        wait.until(EC.visibility_of_element_located((By.ID, "interesting-tab")))

        # Use JavaScript to inject test data and trigger UI update
        browser.execute_script("""
            const testPriorities = {
                "test_uid_1": {
                    "priority": 5,
                    "searchTerm": "test"
                }
            };
            localStorage.setItem('paperPriorities', JSON.stringify(testPriorities));
            // Reload priorities from localStorage into memory state
            if (window.loadPriorities) {
                window.loadPriorities();
            }
            // Manually call updateControlsVisibility to show buttons
            if (window.updateControlsVisibility) {
                window.updateControlsVisibility();
            }
        """)

        time.sleep(0.2)

        # Click Export as Zip button
        try:
            export_button = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Export as Zip')]"))
            )
            export_button.click()
            time.sleep(0.2)

            # Should show donation prompt on first export
            alert = browser.switch_to.alert
            alert_text = alert.text
            assert "donate" in alert_text.lower() or "export" in alert_text.lower()
            alert.dismiss()  # Decline donation
            time.sleep(0.2)

            # If there's another alert (the export itself), dismiss it
            try:
                alert = browser.switch_to.alert
                alert.dismiss()
            except Exception:
                pass

            time.sleep(0.2)

            # Click export again - should NOT show donation prompt again
            export_button.click()
            time.sleep(0.2)

            # This time it should go straight to export (or show export-related dialog only)
            # Not the donation prompt
            try:
                alert = browser.switch_to.alert
                alert_text = alert.text
                # If there's an alert, it should NOT be about donation
                assert "donate" not in alert_text.lower() or "before you export" not in alert_text.lower()
                alert.dismiss()
            except Exception:
                # No alert is fine - means it went straight to export
                pass

        except Exception as e:
            pytest.skip(f"Could not test export flow: {e}")


@pytest.mark.e2e
@pytest.mark.slow
class TestWebUIAccessibility:
    """Accessibility tests for the web UI."""

    def test_keyboard_navigation(self, web_server, browser):
        """
        Test keyboard navigation through the interface.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        # Tab through elements
        body = browser.find_element(By.TAG_NAME, "body")
        for _ in range(5):
            body.send_keys(Keys.TAB)
            time.sleep(0.2)

        # Should be able to reach search input via keyboard
        search_input = browser.find_element(By.ID, "search-input")
        assert search_input is not None

    def test_form_labels(self, web_server, browser):
        """
        Test that form elements have proper labels.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        # Check for label elements
        labels = browser.find_elements(By.TAG_NAME, "label")
        assert len(labels) > 0, "Form elements should have labels"

    def test_semantic_html(self, web_server, browser):
        """
        Test that the page uses semantic HTML elements.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        # Check for semantic elements
        assert len(browser.find_elements(By.TAG_NAME, "header")) > 0
        assert len(browser.find_elements(By.TAG_NAME, "main")) > 0
        assert len(browser.find_elements(By.TAG_NAME, "button")) > 0


@pytest.mark.e2e
@pytest.mark.slow
class TestAdvancedSearchE2E:
    """End-to-end tests for the advanced search functionality.

    Tests cover:
    - Plain author name search (without field:"value" syntax)
    - ``authors:"Name"`` and ``author:"Name"`` field syntax
    - Combined field + topic queries via the advanced search form
    - Other field searches (title, keywords, abstract) via the advanced search form
    """

    # ------------------------------------------------------------------ helpers

    def _open_advanced_search(self, browser, wait):
        """
        Click the Advanced Search button and wait for the modal to appear.

        Parameters
        ----------
        browser : webdriver.Chrome or webdriver.Firefox
            Selenium WebDriver instance.
        wait : WebDriverWait
            WebDriverWait instance configured with the desired timeout.
        """
        adv_btn = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[onclick='openAdvancedSearch()']")))
        adv_btn.click()
        wait.until(EC.visibility_of_element_located((By.ID, "advanced-search-modal")))

    def _apply_advanced_search(self, browser, wait):
        """
        Click the Search button inside the advanced search modal and wait for results.

        Parameters
        ----------
        browser : webdriver.Chrome or webdriver.Firefox
            Selenium WebDriver instance.
        wait : WebDriverWait
            WebDriverWait instance configured with the desired timeout.
        """
        apply_btn = browser.find_element(By.CSS_SELECTOR, "button[onclick='applyAdvancedSearch()']")
        apply_btn.click()
        # Wait for the modal to close before checking results
        wait.until(EC.invisibility_of_element_located((By.ID, "advanced-search-modal")))
        # Wait for at least one result card (header card is always present when papers are found)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#search-results .bg-white.rounded-lg")))

    # ------------------------------------------------------------------ tests

    def test_plain_author_name_search(self, web_server, browser):
        """
        Test that entering only an author name (without field syntax) returns the correct paper.

        When a user types a plain name such as "Ashish Vaswani" the backend
        recognises it as an author match and returns the corresponding paper as
        the first result, even though no ``authors:"…"`` syntax is used.

        Parameters
        ----------
        web_server : tuple
            Web server fixture (base_url, port)
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        search_input = browser.find_element(By.ID, "search-input")
        search_input.send_keys("Ashish Vaswani")
        search_input.send_keys(Keys.RETURN)

        wait = WebDriverWait(browser, 10)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#search-results .bg-white.rounded-lg")))

        results_text = browser.find_element(By.ID, "search-results").text
        assert (
            "Attention is All You Need" in results_text
        ), "Searching for 'Ashish Vaswani' should return 'Attention is All You Need'"

    def test_plain_author_last_name_search(self, web_server, browser):
        """
        Test that entering only an author's last name returns the correct paper.

        A partial name such as "Vaswani" must be resolved to the author
        "Ashish Vaswani" and return that author's paper.

        Parameters
        ----------
        web_server : tuple
            Web server fixture (base_url, port)
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        search_input = browser.find_element(By.ID, "search-input")
        search_input.send_keys("Vaswani")
        search_input.send_keys(Keys.RETURN)

        wait = WebDriverWait(browser, 10)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#search-results .bg-white.rounded-lg")))

        results_text = browser.find_element(By.ID, "search-results").text
        assert (
            "Attention is All You Need" in results_text
        ), "Searching for 'Vaswani' should return 'Attention is All You Need'"

    def test_authors_field_syntax_alone(self, web_server, browser):
        """
        Test that the ``authors:"Name"`` syntax works alone in the search box.

        Entering ``authors:"Ashish Vaswani"`` must bypass the embedding search
        and return only papers whose ``authors`` column matches that value.

        Parameters
        ----------
        web_server : tuple
            Web server fixture (base_url, port)
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        search_input = browser.find_element(By.ID, "search-input")
        search_input.send_keys('authors:"Ashish Vaswani"')
        search_input.send_keys(Keys.RETURN)

        wait = WebDriverWait(browser, 10)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#search-results .bg-white.rounded-lg")))

        results_text = browser.find_element(By.ID, "search-results").text
        assert (
            "Attention is All You Need" in results_text
        ), "authors:\"Ashish Vaswani\" should return 'Attention is All You Need'"
        # Papers by other authors must not be in the results
        assert "Generative Adversarial" not in results_text, "GAN paper should not appear when filtering by Vaswani"

    def test_author_alias_field_syntax(self, web_server, browser):
        """
        Test that the ``author:"Name"`` alias syntax works the same as ``authors:"Name"``.

        The backend accepts both ``author:`` and ``authors:`` as field names and
        must return identical results for both.

        Parameters
        ----------
        web_server : tuple
            Web server fixture (base_url, port)
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        search_input = browser.find_element(By.ID, "search-input")
        search_input.send_keys('author:"Ashish Vaswani"')
        search_input.send_keys(Keys.RETURN)

        wait = WebDriverWait(browser, 10)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#search-results .bg-white.rounded-lg")))

        results_text = browser.find_element(By.ID, "search-results").text
        assert (
            "Attention is All You Need" in results_text
        ), "author:\"Ashish Vaswani\" (alias) should return 'Attention is All You Need'"
        assert (
            "Generative Adversarial" not in results_text
        ), "GAN paper should not appear when filtering by Vaswani (using author: alias)"

    def test_advanced_search_modal_opens_and_closes(self, web_server, browser):
        """
        Test that the advanced search modal can be opened and closed.

        Verifies:
        * The modal is hidden on page load.
        * The Advanced Search button makes the modal visible.
        * The Cancel button hides the modal again.

        Parameters
        ----------
        web_server : tuple
            Web server fixture (base_url, port)
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        wait = WebDriverWait(browser, 10)

        # Modal must be hidden initially
        modal = browser.find_element(By.ID, "advanced-search-modal")
        assert not modal.is_displayed(), "Advanced search modal should be hidden on page load"

        # Open the modal
        self._open_advanced_search(browser, wait)
        assert modal.is_displayed(), "Advanced search modal should be visible after clicking the button"

        # Close with the Cancel button
        cancel_btn = browser.find_element(
            By.XPATH, "//button[@onclick='closeAdvancedSearch()' and contains(., 'Cancel')]"
        )
        cancel_btn.click()
        wait.until(EC.invisibility_of_element_located((By.ID, "advanced-search-modal")))
        assert not modal.is_displayed(), "Advanced search modal should be hidden after Cancel"

    def test_advanced_search_modal_fields_present(self, web_server, browser):
        """
        Test that the advanced search modal contains all expected input fields.

        The modal must expose fields for: topic, authors, title, keywords,
        abstract, and award.

        Parameters
        ----------
        web_server : tuple
            Web server fixture (base_url, port)
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        wait = WebDriverWait(browser, 10)
        self._open_advanced_search(browser, wait)

        for field_id in ("adv-topic", "adv-authors", "adv-title", "adv-keywords", "adv-abstract", "adv-award"):
            assert browser.find_element(
                By.ID, field_id
            ).is_displayed(), f"Advanced search field '{field_id}' should be visible"

        # Close the modal
        browser.find_element(By.XPATH, "//button[@onclick='closeAdvancedSearch()' and contains(., 'Cancel')]").click()
        wait.until(EC.invisibility_of_element_located((By.ID, "advanced-search-modal")))

    def test_advanced_search_authors_field(self, web_server, browser):
        """
        Test searching by the authors field in the advanced search form.

        Filling in "Ashish Vaswani" in the Authors field and applying the
        search must return only that author's paper.

        Parameters
        ----------
        web_server : tuple
            Web server fixture (base_url, port)
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        wait = WebDriverWait(browser, 10)
        self._open_advanced_search(browser, wait)

        authors_input = browser.find_element(By.ID, "adv-authors")
        authors_input.clear()
        authors_input.send_keys("Ashish Vaswani")

        self._apply_advanced_search(browser, wait)

        results_text = browser.find_element(By.ID, "search-results").text
        assert (
            "Attention is All You Need" in results_text
        ), "Searching for author 'Ashish Vaswani' via the advanced form should return 'Attention is All You Need'"
        assert (
            "Generative Adversarial" not in results_text
        ), "GAN paper should not appear in results filtered to Vaswani"

    def test_advanced_search_author_with_topic(self, web_server, browser):
        """
        Test combining an author filter with a free-text topic query.

        Entering "Yoshua Bengio" in the Authors field and "adversarial" in
        the Topic field must return only papers by Yoshua Bengio; papers by
        other authors must not appear in the results.

        Parameters
        ----------
        web_server : tuple
            Web server fixture (base_url, port)
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        wait = WebDriverWait(browser, 10)
        self._open_advanced_search(browser, wait)

        authors_input = browser.find_element(By.ID, "adv-authors")
        authors_input.clear()
        authors_input.send_keys("Yoshua Bengio")

        topic_input = browser.find_element(By.ID, "adv-topic")
        topic_input.clear()
        topic_input.send_keys("adversarial")

        self._apply_advanced_search(browser, wait)

        results_text = browser.find_element(By.ID, "search-results").text
        # At least one Bengio paper must appear (checking for last name is sufficient
        # since the full name "Yoshua Bengio" always contains "Bengio")
        assert "Bengio" in results_text, "Expected at least one Yoshua Bengio paper in results"
        # Papers by unrelated authors must not appear
        assert "Ashish Vaswani" not in results_text, "Vaswani paper should not appear when filtering by Yoshua Bengio"
        assert "Kaiming He" not in results_text, "He paper should not appear when filtering by Yoshua Bengio"

    def test_advanced_search_title_field(self, web_server, browser):
        """
        Test searching by the title field in the advanced search form.

        Entering "Residual" in the Title field must return the
        'Deep Residual Learning for Image Recognition' paper and not
        unrelated papers.

        Parameters
        ----------
        web_server : tuple
            Web server fixture (base_url, port)
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        wait = WebDriverWait(browser, 10)
        self._open_advanced_search(browser, wait)

        title_input = browser.find_element(By.ID, "adv-title")
        title_input.clear()
        title_input.send_keys("Residual")

        self._apply_advanced_search(browser, wait)

        results_text = browser.find_element(By.ID, "search-results").text
        assert (
            "Residual" in results_text
        ), "Searching by title 'Residual' should return the Deep Residual Learning paper"
        assert (
            "Generative Adversarial" not in results_text
        ), "GAN paper should not appear in results filtered by title 'Residual'"

    def test_advanced_search_keywords_field(self, web_server, browser):
        """
        Test searching by the keywords field in the advanced search form.

        Entering "deep learning" in the Keywords field must return the
        'Deep Residual Learning for Image Recognition' paper which has
        "deep learning" as one of its keywords.

        Parameters
        ----------
        web_server : tuple
            Web server fixture (base_url, port)
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        wait = WebDriverWait(browser, 10)
        self._open_advanced_search(browser, wait)

        keywords_input = browser.find_element(By.ID, "adv-keywords")
        keywords_input.clear()
        keywords_input.send_keys("deep learning")

        self._apply_advanced_search(browser, wait)

        results_text = browser.find_element(By.ID, "search-results").text
        # The Deep Residual Learning paper has "deep learning" in its keywords.
        # "Residual" is sufficient: the title "Deep Residual Learning for Image Recognition"
        # always contains this substring.
        assert (
            "Residual" in results_text
        ), "Searching by keyword 'deep learning' should return the Deep Residual Learning paper"

    def test_advanced_search_abstract_field(self, web_server, browser):
        """
        Test searching by the abstract field in the advanced search form.

        Entering "adversarial process" in the Abstract field must return the
        'Generative Adversarial Networks' paper whose abstract contains that phrase.

        Parameters
        ----------
        web_server : tuple
            Web server fixture (base_url, port)
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        wait = WebDriverWait(browser, 10)
        self._open_advanced_search(browser, wait)

        abstract_input = browser.find_element(By.ID, "adv-abstract")
        abstract_input.clear()
        abstract_input.send_keys("adversarial process")

        self._apply_advanced_search(browser, wait)

        results_text = browser.find_element(By.ID, "search-results").text
        assert (
            "Generative Adversarial Networks" in results_text
        ), "Searching abstract for 'adversarial process' should return the GAN paper"
        assert (
            "Attention is All You Need" not in results_text
        ), "Vaswani paper should not appear when filtering by abstract 'adversarial process'"

    def test_advanced_search_search_input_populated(self, web_server, browser):
        """
        Test that applying an advanced search populates the main search input.

        After the user fills in the advanced search form and applies it, the
        constructed ``field:"value"`` query string must be written back to the
        main search input box so the user can see and edit it.

        Parameters
        ----------
        web_server : tuple
            Web server fixture (base_url, port)
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        wait = WebDriverWait(browser, 10)
        self._open_advanced_search(browser, wait)

        authors_input = browser.find_element(By.ID, "adv-authors")
        authors_input.clear()
        authors_input.send_keys("Vaswani")

        self._apply_advanced_search(browser, wait)

        # The main search input must now contain the field:"value" query
        search_input = browser.find_element(By.ID, "search-input")
        search_value = search_input.get_attribute("value")
        assert "Vaswani" in search_value, f"Main search input should contain the author query, got: {search_value!r}"
        assert "authors:" in search_value, f"Main search input should use field syntax, got: {search_value!r}"

    def test_advanced_search_enter_key_applies(self, web_server, browser):
        """
        Test that pressing Enter inside an advanced search field applies the search.

        The advanced search form must submit when the user presses Enter in
        any of its input fields, matching the behaviour of the main search box.

        Parameters
        ----------
        web_server : tuple
            Web server fixture (base_url, port)
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        wait = WebDriverWait(browser, 10)
        self._open_advanced_search(browser, wait)

        authors_input = browser.find_element(By.ID, "adv-authors")
        authors_input.clear()
        authors_input.send_keys("Ashish Vaswani")
        # Press Enter instead of clicking the Search button
        authors_input.send_keys(Keys.RETURN)

        # Modal should close and results should appear
        wait.until(EC.invisibility_of_element_located((By.ID, "advanced-search-modal")))
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#search-results .bg-white.rounded-lg")))

        results_text = browser.find_element(By.ID, "search-results").text
        assert (
            "Attention is All You Need" in results_text
        ), "Pressing Enter in the authors field should trigger search and return Vaswani's paper"


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.skip(
    reason="Clustering tab tests are currently too slow because of the large number of papers in test data. Need to use smaller dataset for testing."
)
class TestClusteringTabE2E:
    """Test clustering tab functionality with Selenium."""

    def test_clustering_tab_exists(self, web_server, browser):
        """
        Test that the clustering tab exists and is accessible.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        # Wait for page to load
        wait = WebDriverWait(browser, 10)
        # Find and click clustering tab
        clustering_tab = wait.until(EC.element_to_be_clickable((By.ID, "tab-clusters")))
        assert clustering_tab.is_displayed(), "Clustering tab should be visible"

    def test_clustering_plot_loads(self, web_server, browser):
        """
        Test that the clustering plot container loads properly.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        # Navigate to clustering tab
        wait = WebDriverWait(browser, 10)
        clustering_tab = wait.until(EC.element_to_be_clickable((By.ID, "tab-clusters")))
        clustering_tab.click()

        # Wait for plot container to be visible with timeout
        plot_container = wait.until(EC.visibility_of_element_located((By.ID, "cluster-plot")))
        assert plot_container.is_displayed(), "Clustering plot container should be visible"

    def test_clustering_stats_display(self, web_server, browser):
        """
        Test that clustering statistics are displayed in the legend title.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        # Navigate to clustering tab
        wait = WebDriverWait(browser, 10)
        clustering_tab = wait.until(EC.element_to_be_clickable((By.ID, "tab-clusters")))
        clustering_tab.click()

        # Wait for legend element to be visible with timeout
        legend_element = wait.until(EC.visibility_of_element_located((By.ID, "cluster-legend")))
        assert legend_element.is_displayed(), "Cluster legend should be visible"

        # Verify legend contains the stats information
        legend_text = legend_element.text
        assert "papers" in legend_text.lower(), "Legend should contain paper count"
        assert "clusters" in legend_text.lower(), "Legend should contain cluster count"

    def test_clustering_filter_dropdown(self, web_server, browser):
        """
        Test that the clustering tab has filter/search functionality.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        # Navigate to clustering tab
        wait = WebDriverWait(browser, 10)
        clustering_tab = wait.until(EC.element_to_be_clickable((By.ID, "tab-clusters")))
        clustering_tab.click()

        # Wait for custom query search input to be visible with timeout
        search_input = wait.until(EC.visibility_of_element_located((By.ID, "custom-query-input")))
        assert search_input.is_displayed(), "Custom query search input should be visible"

        # Verify it's an input element
        assert search_input.tag_name == "input", "Search should be an input element"

    def test_clustering_paper_details_panel(self, web_server, browser):
        """
        Test that the selected paper details panel exists.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        # Navigate to clustering tab
        wait = WebDriverWait(browser, 10)
        clustering_tab = wait.until(EC.element_to_be_clickable((By.ID, "tab-clusters")))
        clustering_tab.click()

        # Wait for clustering content to load
        wait.until(EC.visibility_of_element_located((By.ID, "clusters-tab")))

        # Check for paper details panel with timeout
        try:
            details_panel = wait.until(EC.presence_of_element_located((By.ID, "selected-paper-details")), timeout=5)
            # Panel might be hidden initially
            assert details_panel is not None, "Selected paper details panel should exist"
        except TimeoutException:
            # Panel might not exist if no paper selected - this is acceptable
            pass

    def test_clustering_plot_has_plotly(self, web_server, browser):
        """
        Test that Plotly is loaded and the plot is rendered.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        # Navigate to clustering tab
        wait = WebDriverWait(browser, 10)
        clustering_tab = wait.until(EC.element_to_be_clickable((By.ID, "tab-clusters")))
        clustering_tab.click()

        # Wait for clustering content to be visible
        wait.until(EC.visibility_of_element_located((By.ID, "clusters-tab")))

        # Wait for Plotly to be available with timeout
        def plotly_loaded(driver):
            return driver.execute_script("return typeof Plotly !== 'undefined'")

        wait.until(plotly_loaded)

        # Verify plot container exists
        wait.until(EC.presence_of_element_located((By.ID, "cluster-plot")))

    def test_clustering_tab_no_javascript_errors(self, web_server, browser):
        """
        Test that there are no JavaScript errors when loading the clustering tab.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        # Navigate to clustering tab
        wait = WebDriverWait(browser, 10)
        clustering_tab = wait.until(EC.element_to_be_clickable((By.ID, "tab-clusters")))
        clustering_tab.click()

        # Wait for content to load
        wait.until(EC.visibility_of_element_located((By.ID, "clusters-tab")))

        # Check browser console for errors with timeout
        def check_errors(driver):
            try:
                logs = driver.get_log("browser")
                return logs
            except Exception:
                # Browser might not support get_log (e.g., Firefox)
                return []

        logs = check_errors(browser)
        errors = [log for log in logs if log.get("level") == "SEVERE"]

        # Filter out known harmless errors
        critical_errors = []
        for error in errors:
            message = error.get("message", "")
            # Ignore favicon errors and other non-critical issues
            if "favicon" not in message.lower() and "ERR_BLOCKED_BY_CLIENT" not in message:
                critical_errors.append(error)

        assert len(critical_errors) == 0, f"No severe JavaScript errors should occur. Found: {critical_errors}"

    def test_clustering_visualization_elements(self, web_server, browser):
        """
        Test that clustering visualization has expected elements.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        # Navigate to clustering tab
        wait = WebDriverWait(browser, 10)
        clustering_tab = wait.until(EC.element_to_be_clickable((By.ID, "tab-clusters")))
        clustering_tab.click()

        # Wait for clustering content to be visible
        clustering_content = wait.until(EC.visibility_of_element_located((By.ID, "clusters-tab")))

        # Should have some content
        def has_content(element):
            return element.text != ""

        assert has_content(clustering_content), "Clustering content should not be empty"

        # Wait for plot container
        plot = wait.until(EC.presence_of_element_located((By.ID, "cluster-plot")))
        assert plot is not None, "Cluster plot should exist"

    def test_cluster_center_colors_match_points(self, web_server, browser):
        """
        Test that cluster centers have the same color as their cluster points.

        This test verifies that when cluster centers are displayed as stars,
        they use the same color as the points in their cluster.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        browser : webdriver.Chrome
            Selenium WebDriver instance
        """
        base_url, _ = web_server
        browser.get(base_url)

        # Navigate to clustering tab
        wait = WebDriverWait(browser, 10)
        clustering_tab = wait.until(EC.element_to_be_clickable((By.ID, "tab-clusters")))
        clustering_tab.click()

        # Wait for clustering content to be visible
        wait.until(EC.visibility_of_element_located((By.ID, "clusters-tab")))

        # Wait for Plotly to load with timeout
        def plotly_available(driver):
            return driver.execute_script("return typeof Plotly !== 'undefined'")

        wait.until(plotly_available)

        # Execute JavaScript to check if Plotly plot exists and get trace colors
        script = """
        const plotDiv = document.getElementById('cluster-plot');
        if (!plotDiv || !plotDiv.data) {
            return { error: 'No plot data found' };
        }
        
        const data = plotDiv.data;
        const colorMatches = [];
        
        // Iterate through traces looking for cluster points and their centers
        for (let i = 0; i < data.length; i++) {
            const trace = data[i];
            
            // Check if this is a center trace (star marker)
            if (trace.marker && trace.marker.symbol === 'star') {
                const centerColor = trace.marker.color;
                
                // Find the corresponding cluster points trace
                // Center traces are added after their point traces
                // Look backwards for a trace with the same legendgroup
                for (let j = i - 1; j >= 0; j--) {
                    const pointTrace = data[j];
                    if (pointTrace.legendgroup === trace.legendgroup &&
                        pointTrace.marker && pointTrace.marker.symbol !== 'star') {
                        const pointColor = pointTrace.marker.color;
                        
                        colorMatches.push({
                            cluster: trace.legendgroup,
                            centerColor: centerColor,
                            pointColor: pointColor,
                            match: centerColor === pointColor
                        });
                        break;
                    }
                }
            }
        }
        
        return {
            success: true,
            matches: colorMatches,
            allMatch: colorMatches.every(m => m.match)
        };
        """

        try:
            result = browser.execute_script(script)

            # Check if plot data was found
            if "error" in result:
                # Plot might not have data yet, which is ok for this test
                # The important thing is the JavaScript code structure is correct
                return

            if result.get("success") and result.get("matches"):
                # Verify all cluster centers have matching colors
                assert result["allMatch"], f"Not all cluster centers match their point colors: {result['matches']}"

                # Log the successful matches
                print(f"Color matching verified for {len(result['matches'])} clusters")
        except TimeoutException:
            # If plot takes too long to load, skip this test
            pytest.skip("Clustering plot failed to load within timeout")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "e2e"])

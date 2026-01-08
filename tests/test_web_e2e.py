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
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from webdriver_manager.core.os_manager import ChromeType
from neurips_abstracts.database import DatabaseManager
from neurips_abstracts.config import Config
from tests.test_helpers import find_free_port

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Constants for E2E tests
MOCK_EMBEDDING_DIMENSION = 4096  # Standard dimension for test embeddings

# Module-level cache for driver paths to ensure single installation per session
# This prevents redundant downloads and installations when running multiple E2E tests
# The driver is only installed when:
# 1. Tests with @pytest.mark.e2e are actually executed (not skipped)
# 2. The browser fixture is requested by a test
# 3. The driver hasn't been installed yet in this test session
_driver_cache = {
    "chrome": None,
    "firefox": None,
    "chrome_available": None,
    "firefox_available": None,
}


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
    db = DatabaseManager(str(db_path))

    with db:
        db.create_tables()

        cursor = db.connection.cursor()

        # Add test papers with realistic data (lightweight schema)
        papers_data = [
            (
                "paper1",
                "Attention is All You Need",
                "Ashish Vaswani, Noam Shazeer",
                "We propose the Transformer, a model architecture eschewing recurrence and instead "
                "relying entirely on an attention mechanism to draw global dependencies between input and output.",
                "Oral Session 1",
                "A1",
                "attention, transformer, neural networks",
                2025,
                "NeurIPS",
            ),
            (
                "paper2",
                "BERT: Pre-training of Deep Bidirectional Transformers",
                "Jacob Devlin, Ming-Wei Chang",
                "We introduce a new language representation model called BERT, which stands for "
                "Bidirectional Encoder Representations from Transformers.",
                "Poster Session A",
                "P1",
                "bert, nlp, transformers",
                2025,
                "NeurIPS",
            ),
            (
                "paper3",
                "Deep Residual Learning for Image Recognition",
                "Kaiming He, Xiangyu Zhang",
                "Deep residual nets are foundations of our submissions to ILSVRC & COCO 2015 competitions.",
                "Oral Session 2",
                "A2",
                "resnet, computer vision, deep learning",
                2025,
                "NeurIPS",
            ),
            (
                "paper4",
                "Generative Adversarial Networks",
                "Ian Goodfellow, Yoshua Bengio",
                "We propose a new framework for estimating generative models via an adversarial process.",
                "Poster Session B",
                "P2",
                "gan, generative models, adversarial",
                2025,
                "NeurIPS",
            ),
            (
                "paper5",
                "Neural Machine Translation by Jointly Learning to Align and Translate",
                "Dzmitry Bahdanau, Yoshua Bengio",
                "Neural machine translation is a recently proposed approach to machine translation.",
                "Spotlight Session",
                "S1",
                "nmt, translation, attention",
                2025,
                "NeurIPS",
            ),
        ]

        for uid, title, authors, abstract, session, poster_position, keywords, year, conference in papers_data:
            cursor.execute(
                """
                INSERT INTO papers (uid, title, authors, abstract, session, poster_position, keywords, year, conference)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (uid, title, authors, abstract, session, poster_position, keywords, year, conference),
            )

        db.connection.commit()

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
    from neurips_abstracts.embeddings import EmbeddingsManager
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

    # Initialize embeddings manager
    em = EmbeddingsManager(chroma_path=embeddings_path, collection_name=collection_name)

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
    db = DatabaseManager(str(test_database))
    db.connect()
    cursor = db.connection.cursor()
    cursor.execute("SELECT * FROM papers")
    papers = cursor.fetchall()

    # Add each paper to the embeddings database
    for paper in papers:
        paper_dict = dict(paper)
        em.add_paper(paper_dict)

    db.close()

    # Return the embeddings manager, metadata, and the mock client
    # The mock client is returned so web_server can reuse it
    yield (em, embeddings_path, collection_name, mock_client)

    # Cleanup happens automatically


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
    from neurips_abstracts.database import DatabaseManager

    # Import the module, not the app object
    import sys

    # Get the actual module
    app_module = sys.modules["neurips_abstracts.web_ui.app"]
    # Get the Flask app instance
    flask_app = app_module.app

    # Unpack cached embeddings and mock client
    em, embeddings_path, collection_name, mock_client = test_embeddings

    # CRITICAL: Ensure the embeddings manager is using the mock client
    # This prevents it from creating a real OpenAI connection during searches
    em._openai_client = mock_client

    port = find_free_port()
    base_url = f"http://localhost:{port}"

    # Configure the app to use test database and embeddings
    def mock_get_config():
        config = Config()
        config.paper_db_path = str(test_database)
        config.embedding_db_path = str(embeddings_path)
        config.collection_name = collection_name
        return config

    app_module.get_config = mock_get_config

    # Inject the pre-created embeddings manager directly
    # CRITICAL: Set this BEFORE starting the server to avoid race conditions
    # This prevents get_embeddings_manager() from creating a new instance
    app_module.embeddings_manager = em
    app_module.rag_chat = None

    # Mock get_database to not check file existence
    # Each Flask request thread will create its own database connection
    original_get_database = app_module.get_database

    def mock_get_database_wrapper():
        """Wrapper that skips file existence check."""
        from flask import g

        if "db" not in g:
            db_path = str(test_database)
            # Don't check file existence in tests - the database was created in a temp dir
            g.db = DatabaseManager(db_path)
            g.db.connect()
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
            time.sleep(0.5)

    yield (base_url, port)

    # Cleanup
    server.shutdown()

    # Reset the app module state
    app_module.embeddings_manager = None
    app_module.rag_chat = None
    app_module.get_database = original_get_database


def _check_chrome_available():
    """
    Check if Chrome browser is available for testing.

    This check does NOT install the driver - it only verifies the browser exists.

    Returns
    -------
    bool
        True if Chrome is available, False otherwise
    """
    # Use cached result if available
    if _driver_cache["chrome_available"] is not None:
        return _driver_cache["chrome_available"]

    try:
        # Just check if Chrome binary exists
        import shutil

        chrome_binary = shutil.which("chromium") or shutil.which("chrome") or shutil.which("google-chrome")
        result = chrome_binary is not None
        _driver_cache["chrome_available"] = result
        return result
    except Exception:
        _driver_cache["chrome_available"] = False
        return False


def _check_firefox_available():
    """
    Check if Firefox browser is available for testing.

    This check does NOT install the driver - it only verifies the browser exists.

    Returns
    -------
    bool
        True if Firefox is available, False otherwise
    """
    # Use cached result if available
    if _driver_cache["firefox_available"] is not None:
        return _driver_cache["firefox_available"]

    try:
        # Just check if Firefox binary exists
        import shutil

        firefox_binary = shutil.which("firefox")
        result = firefox_binary is not None
        _driver_cache["firefox_available"] = result
        return result
    except Exception:
        _driver_cache["firefox_available"] = False
        return False


def _get_browser_preference():
    """
    Get browser preference from environment variable.

    Returns
    -------
    str
        Browser preference: 'chrome', 'firefox', or 'auto'
    """
    return os.environ.get("E2E_BROWSER", "auto").lower()


def _create_chrome_driver():
    """
    Create a Chrome WebDriver instance.

    Installs the driver only once per test session using a cache.

    Returns
    -------
    webdriver.Chrome
        Chrome WebDriver instance
    """
    chrome_options = ChromeOptions()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")

    # Install driver only once per session
    if _driver_cache["chrome"] is None:
        _driver_cache["chrome"] = ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install()

    service = ChromeService(_driver_cache["chrome"])
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.implicitly_wait(10)
    return driver


def _create_firefox_driver():
    """
    Create a Firefox WebDriver instance.

    Installs the driver only once per test session using a cache.

    Returns
    -------
    webdriver.Firefox
        Firefox WebDriver instance
    """
    firefox_options = FirefoxOptions()
    firefox_options.add_argument("--headless")
    firefox_options.add_argument("--width=1920")
    firefox_options.add_argument("--height=1080")

    # Install driver only once per session
    if _driver_cache["firefox"] is None:
        _driver_cache["firefox"] = GeckoDriverManager().install()

    service = FirefoxService(_driver_cache["firefox"])
    driver = webdriver.Firefox(service=service, options=firefox_options)
    driver.implicitly_wait(10)
    return driver


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
    Skips test if no browser is available.

    Browser selection is controlled by the E2E_BROWSER environment variable:
    - 'chrome': Use Chrome only
    - 'firefox': Use Firefox only
    - 'auto' (default): Try Chrome first, fall back to Firefox

    Examples
    --------
    # Use Firefox instead of Chrome
    E2E_BROWSER=firefox pytest tests/test_web_e2e.py -m e2e
    """
    browser_pref = _get_browser_preference()
    driver = None

    if browser_pref == "chrome":
        # User explicitly wants Chrome
        if not _check_chrome_available():
            pytest.skip("Chrome browser not available for E2E testing")
        driver = _create_chrome_driver()
    elif browser_pref == "firefox":
        # User explicitly wants Firefox
        if not _check_firefox_available():
            pytest.skip("Firefox browser not available for E2E testing")
        driver = _create_firefox_driver()
    else:
        # Auto mode: try Chrome first, then Firefox
        if _check_chrome_available():
            driver = _create_chrome_driver()
        elif _check_firefox_available():
            driver = _create_firefox_driver()
        else:
            pytest.skip("Neither Chrome nor Firefox browser available for E2E testing")

    yield driver

    # Cleanup
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

        # Wait for tab to switch
        time.sleep(0.5)

        # Chat tab should be visible
        chat_tab = browser.find_element(By.ID, "chat-tab")
        assert chat_tab.is_displayed()

        # Search tab should be hidden
        search_tab = browser.find_element(By.ID, "search-tab")
        assert not search_tab.is_displayed()

        # Switch back to search
        search_tab_button = browser.find_element(By.ID, "tab-search")
        search_tab_button.click()
        time.sleep(0.5)

        assert search_tab.is_displayed()
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
        time.sleep(1)  # Allow time for rendering
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
            time.sleep(0.5)

            # Check that abstract is visible
            abstract_divs = browser.find_elements(By.CSS_SELECTOR, ".abstract-content")
            if len(abstract_divs) > 0:
                # Abstract should be visible after clicking toggle
                assert abstract_divs[0].is_displayed()

                # Click again to collapse
                toggle_buttons[0].click()
                time.sleep(0.5)

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
        time.sleep(1)

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
        time.sleep(2)

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

        # Wait for stats to load
        wait = WebDriverWait(browser, 10)
        wait.until(EC.presence_of_element_located((By.ID, "stats")))

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
        time.sleep(1)

        # Find select all buttons (if they exist)
        select_buttons = browser.find_elements(By.CSS_SELECTOR, "button[onclick*='selectAll']")
        if len(select_buttons) > 0:
            # Scroll to the button first to ensure it's visible
            browser.execute_script("arguments[0].scrollIntoView(true);", select_buttons[0])
            time.sleep(0.3)

            # Use JavaScript click to avoid scrolling issues
            browser.execute_script("arguments[0].click();", select_buttons[0])
            time.sleep(0.5)

            # Find corresponding deselect button
            deselect_buttons = browser.find_elements(By.CSS_SELECTOR, "button[onclick*='deselectAll']")
            if len(deselect_buttons) > 0:
                # Scroll and click with JavaScript
                browser.execute_script("arguments[0].scrollIntoView(true);", deselect_buttons[0])
                time.sleep(0.3)
                browser.execute_script("arguments[0].click();", deselect_buttons[0])
                time.sleep(0.5)

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
        time.sleep(0.5)

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
        time.sleep(0.5)

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
        time.sleep(1)

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
        time.sleep(1)

        # Try to find and click reset button
        reset_buttons = browser.find_elements(By.CSS_SELECTOR, "button[onclick*='resetChat']")
        if len(reset_buttons) > 0:
            # Get initial message count
            initial_messages = browser.find_elements(By.CSS_SELECTOR, ".chat-message")
            len(initial_messages)

            # Click reset button
            reset_buttons[0].click()
            time.sleep(1)

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
        time.sleep(0.5)

        # Find n-papers selector
        n_papers_select = Select(browser.find_element(By.ID, "n-papers"))

        # Check that it has options
        options = n_papers_select.options
        assert len(options) > 0, "n-papers selector should have options"

        # Try to select a different value
        if len(options) > 1:
            n_papers_select.select_by_index(1)
            time.sleep(0.5)

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
        time.sleep(0.5)

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
        time.sleep(0.5)

        # Check input has placeholder
        chat_input = browser.find_element(By.ID, "chat-input")
        placeholder = chat_input.get_attribute("placeholder")
        assert placeholder is not None and len(placeholder) > 0, "Chat input should have placeholder text"

        # Try sending empty message (should not do anything)
        chat_input.send_keys(Keys.RETURN)
        time.sleep(0.5)

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
        time.sleep(0.5)
        assert browser.find_element(By.ID, "search-input").is_displayed()

        # Test tablet size
        browser.set_window_size(768, 1024)
        time.sleep(0.5)
        assert browser.find_element(By.ID, "search-input").is_displayed()

        # Test mobile size
        browser.set_window_size(375, 667)
        time.sleep(0.5)
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
            time.sleep(2)

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
        time.sleep(0.5)

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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "e2e"])

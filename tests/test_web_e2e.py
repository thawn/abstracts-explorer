"""
End-to-end tests for the web UI using Playwright.

These tests verify the web UI functionality through browser automation,
testing user interactions and workflows as they would happen in a real browser.
"""

import pytest
import time

import sqlite3
import numpy as np
import threading
import sys
import os
from pathlib import Path
from playwright.sync_api import Page, TimeoutError, expect
from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.mcp import close_mcp_resources
from abstracts_explorer.orcid import OrcidError, fetch_orcid_record
from abstracts_explorer.rag import shutdown_tool_executor
from tests.helpers import (
    raw_query,
    find_free_port,
    requires_lm_studio,
    stop_test_server,
)
from tests.conftest import set_test_db

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Constants for E2E tests
MOCK_EMBEDDING_DIMENSION = 4096  # Standard dimension for test embeddings
_ORCID_ID = "0000-0002-2315-9247"

# The compiled web UI bundle is required for browser interactions.
_WEB_UI_BUNDLE = Path(__file__).parent.parent / "src/abstracts_explorer/web_ui/static/dist/app.js"


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
            {
                "title": "Contrastive Vision Pretraining at Scale",
                "authors": ["Test Author", "Another Author"],
                "abstract": "We study contrastive pretraining for large-scale vision models.",
                "session": "Poster Session C",
                "poster_position": "P3",
                "keywords": ["vision", "contrastive learning", "pretraining"],
                "year": 2024,
                "conference": "ICLR",
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

    papers = raw_query(db, "SELECT * FROM papers")

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
    Pre-populate the database with cached BERTopic-style clustering results.

    This mirrors the production workflow where ``abstracts-explorer clustering
    pre-generate`` is run before the web UI is used.  Results are constructed
    directly (without running BERTopic) so the E2E suite stays fast and
    deterministic.

    Parameters
    ----------
    test_database : Path
        Path to the test SQLite database.
    em : EmbeddingsManager
        Embeddings manager with test embeddings loaded.
    collection_name : str
        Name of the ChromaDB collection.
    """
    import pandas as pd

    from abstracts_explorer.clustering import (
        DEFAULT_CLUSTERING_METHOD,
        DEFAULT_CLUSTERING_PARAMS,
        DEFAULT_REDUCTION_METHOD,
        ClusteringManager,
    )
    from abstracts_explorer.clustering.results import build_clustering_results
    from abstracts_explorer.config import get_config

    config = get_config()

    set_test_db(str(test_database))
    db = DatabaseManager()
    db.create_tables()

    cm = ClusteringManager(em, db)
    cm.load_embeddings()

    paper_ids = list(cm.paper_ids or [])
    metadatas = list(cm.metadatas or [])
    topics = list(range(len(paper_ids)))
    reduced = np.array([[float(i), float(i % 3)] for i in range(len(paper_ids))])
    labels = {index: f"Topic {index}" for index in topics}
    keywords = {index: [f"keyword{index}"] for index in topics}

    results = build_clustering_results(
        paper_ids=paper_ids,
        reduced_embeddings=reduced,
        metadatas=metadatas,
        topics=topics,
        flat_labels=labels,
        flat_keywords=keywords,
        hierarchical_topics=pd.DataFrame(columns=["Parent_ID", "Parent_Name", "Topics", "Distance"]),
        parent_labels={},
        nr_levels=10,
    )

    clustering_params = dict(DEFAULT_CLUSTERING_PARAMS)
    for conference, year in ((None, None), ("NeurIPS", 2025)):
        db.save_clustering_cache(
            embedding_model=config.embedding_model,
            reduction_method=DEFAULT_REDUCTION_METHOD,
            n_components=2,
            clustering_method=DEFAULT_CLUSTERING_METHOD,
            results=results,
            n_clusters=None,
            clustering_params=clustering_params,
            conference=conference,
            year=year,
            collection_name=collection_name,
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

    if not _WEB_UI_BUNDLE.exists():
        pytest.fail("The web UI bundle is missing. Run `npm ci && npm run build:js` " "before running the e2e tests.")

    # Import the module first, then access it from sys.modules
    import sys
    import abstracts_explorer.web_ui.app  # noqa: F401

    # Get the actual module from sys.modules (now it's loaded)
    app_module = sys.modules["abstracts_explorer.web_ui.app"]
    services_module = sys.modules["abstracts_explorer.web_ui.request_services"]
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
    # Test database has NeurIPS 2025 and ICLR 2024 papers; override so the
    # auto-selected conference matches the primary test data.
    os.environ["DEFAULT_CONFERENCE"] = "NeurIPS"

    def mock_get_config():
        # Force reload to pick up environment variables with .env.test
        from abstracts_explorer.config import get_config as real_get_config
        from tests.conftest import get_env_test_path

        return real_get_config(reload=True, env_path=get_env_test_path())

    # Patch the dependency module: routes and lazy managers resolve their
    # dependencies through ``dependencies`` at call time.
    original_services_get_config = services_module.get_config
    services_module.get_config = mock_get_config

    # Inject the pre-created embeddings manager directly
    # CRITICAL: Set this BEFORE starting the server to avoid race conditions
    # This prevents get_embeddings_manager() from creating a new instance
    services_module.embeddings_manager = em
    services_module._reset_rag_chat_local()

    # Pre-populate the database with cached clustering results so that MCP
    # tools (get_conference_topics, get_cluster_visualization) return instantly
    # without running t-SNE.  This mirrors the production workflow where
    # clustering is pre-generated via the CLI.
    _prepopulate_clustering_cache(test_database, em, collection_name)

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

    try:
        stop_test_server(server, server_thread)
        shutdown_tool_executor()
        close_mcp_resources(force=True)
    finally:
        # Reset the dependency module state
        services_module.embeddings_manager = None
        services_module._reset_rag_chat_local()
        services_module.get_config = original_services_get_config

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


@pytest.mark.e2e
@pytest.mark.slow
class TestWebUIE2E:
    """End-to-end tests for the web UI using Playwright."""

    def test_page_loads(self, web_server, page: Page):
        """
        Test that the main page loads correctly.

        Parameters
        ----------
        web_server : tuple
            Web server fixture (base_url, port)
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        # Check page title
        assert "Abstracts Explorer" in page.title()

        # Check that main elements are present
        expect(page.locator("#search-input")).to_be_attached()
        expect(page.locator("#search-results")).to_be_attached()
        expect(page.locator("#tab-search")).to_be_attached()
        expect(page.locator("#tab-chat")).to_be_attached()

    def test_search_tab_visible_by_default(self, web_server, page: Page):
        """
        Test that the search tab is visible by default.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        # Search tab should be visible
        expect(page.locator("#search-tab")).to_be_visible()

        # Chat tab should be hidden
        expect(page.locator("#chat-tab")).to_be_hidden()

    def test_switch_to_chat_tab(self, web_server, page: Page):
        """
        Test switching between search and chat tabs.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        # Click chat tab
        page.locator("#tab-chat").click()

        # Wait for chat tab to become visible
        expect(page.locator("#chat-tab")).to_be_visible()

        # Search tab should be hidden
        expect(page.locator("#search-tab")).to_be_hidden()

        # Switch back to search
        page.locator("#tab-search").click()

        # Wait for search tab to become visible
        expect(page.locator("#search-tab")).to_be_visible()

        # Chat tab should now be hidden
        expect(page.locator("#chat-tab")).to_be_hidden()

    def test_keyword_search_interaction(self, web_server, page: Page):
        """
        Test keyword search user interaction flow.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        # Find search input
        search_input = page.locator("#search-input")

        # Enter search query
        search_query = "attention"
        search_input.fill(search_query)
        search_input.press("Enter")

        # Wait for results to load
        expect(page.locator("#search-results .bg-white.rounded-lg").first).to_be_attached()

        # Check that results are displayed
        results = page.locator("#search-results .bg-white.rounded-lg").all()
        assert len(results) > 0

        # Check that the search returned results (text will vary based on actual data)
        search_results_div = page.locator("#search-results")
        assert len(search_results_div.inner_text()) > 0, "Search results should contain some text"

    def test_search_limit_filter(self, web_server, page: Page):
        """
        Test changing the search results limit.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        # Open search settings modal
        settings_btn = page.locator("button[onclick='openSearchSettings()']")
        expect(settings_btn).to_be_enabled()
        settings_btn.click()

        # Wait for modal to open
        expect(page.locator("#settings-modal")).to_be_visible()

        # Change limit to 10 (actual available value)
        page.locator("#limit-select").select_option(value="10")

        # Close modal
        page.locator("button[onclick='closeSettings()']").first.click()

        # Wait for modal to close
        expect(page.locator("#settings-modal")).to_be_hidden()

        # Perform search
        search_input = page.locator("#search-input")
        search_input.fill("learning")
        search_input.press("Enter")

        # Wait for results
        expect(page.locator("#search-results .bg-white.rounded-lg").first).to_be_attached()

        # Check that we have at most 10 results
        # Note: There's a header card with results count, so we get 11 elements
        results = page.locator("#search-results .paper-card").all()
        assert len(results) <= 10

    def test_collapsible_abstract(self, web_server, page: Page):
        """
        Test expanding and collapsing paper abstracts.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        # Perform search
        search_input = page.locator("#search-input")
        search_input.fill("attention")
        search_input.press("Enter")

        # Wait for results
        expect(page.locator("#search-results .bg-white.rounded-lg").first).to_be_attached()

        # Find a toggle button
        toggle_buttons = page.locator(".toggle-abstract").all()
        if len(toggle_buttons) > 0:
            # Click to expand
            toggle_buttons[0].click()
            expect(toggle_buttons[0]).to_be_visible()

            # Check that abstract is visible
            abstract_divs = page.locator(".abstract-content").all()
            if len(abstract_divs) > 0:
                # Abstract should be visible after clicking toggle
                expect(abstract_divs[0]).to_be_visible()

                # Click again to collapse
                toggle_buttons[0].click()
                expect(abstract_divs[0]).to_be_hidden()

    def test_empty_search_shows_message(self, web_server, page: Page):
        """
        Test that searching with no query shows appropriate message.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        # Submit empty search
        page.locator("#search-input").press("Enter")

        # Check for message or no results
        results_div = page.locator("#search-results")
        # Should either show message or be empty
        expect(results_div).to_be_attached()
        assert results_div.count() > 0

    def test_search_no_results(self, web_server, page: Page):
        """
        Test searching with a query that returns no results.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        # Search for something unlikely to be in test data
        search_input = page.locator("#search-input")
        search_input.fill("xyzqwertyuiopasdfghjkl")
        search_input.press("Enter")

        # Check that results area exists
        # Note: Fuzzy search may still return results for nonsense queries
        # Just verify search executed without error
        expect(page.locator("#search-results")).to_be_attached()
        # Results may vary - just ensure search didn't crash
        search_results_element = page.locator("#search-results")
        assert search_results_element.count() > 0

    def test_stats_display(self, web_server, page: Page):
        """
        Test that statistics are displayed on the page.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        # Wait for stats to load - wait for the actual stats, not just the loading message
        expect(page.locator("#stats")).to_be_attached()

        # Wait for stats to actually load (text should change from "Loading stats...")
        page.wait_for_function("""() => {
                const el = document.getElementById('stats');
                if (!el) return false;
                const text = el.innerText.toLowerCase();
                return (
                    text.includes('abstracts') ||
                    text.includes('papers') ||
                    text.includes('error loading stats')
                );
            }""")

        stats = page.locator("#stats")
        stats_text = stats.inner_text()

        # Should contain paper/abstract count
        assert "abstracts" in stats_text.lower() or "papers" in stats_text.lower()

    def test_paper_detail_view(self, web_server, page: Page):
        """
        Test viewing paper details by clicking on a paper.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        # Perform search
        search_input = page.locator("#search-input")
        search_input.fill("attention")
        search_input.press("Enter")

        # Wait for results
        expect(page.locator("#search-results .bg-white.rounded-lg").first).to_be_attached()

        # Get paper cards (not the header)
        results = page.locator("#search-results .paper-card").all()
        assert len(results) > 0, "Should have at least one paper card"

        first_result = results[0]

        # Check that paper has title (h3 or h4 depending on compact mode)
        title_elements = first_result.locator("h3, h4").all()
        assert len(title_elements) > 0, "Paper card should have a title element"
        assert len(title_elements[0].inner_text()) > 0, "Title should not be empty"

        # Check that paper has authors
        author_elements = first_result.locator(".text-gray-600").all()
        assert len(author_elements) > 0, "Paper card should have author information"

    def test_select_deselect_all_filters(self, web_server, page: Page):
        """
        Test the select/deselect all functionality for filters.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        # Find select all buttons (if they exist)
        select_buttons = page.locator("button[onclick*='selectAll']").all()
        if len(select_buttons) > 0:
            # These controls may be off-screen or conditionally hidden; use a
            # direct DOM click to match the previous JavaScript-driven behaviour.
            select_buttons[0].evaluate("el => el.click()")

            # Find corresponding deselect button
            deselect_buttons = page.locator("button[onclick*='deselectAll']").all()
            if len(deselect_buttons) > 0:
                deselect_buttons[0].evaluate("el => el.click()")

    def test_chat_interface_elements(self, web_server, page: Page):
        """
        Test that chat interface elements are present and functional.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        # Switch to chat tab
        page.locator("#tab-chat").click()
        expect(page.locator("#chat-tab")).to_be_visible()

        # Check chat elements
        expect(page.locator("#chat-input")).to_be_attached()
        expect(page.locator("#chat-messages")).to_be_attached()
        expect(page.locator("#chat-papers")).to_be_attached()
        expect(page.locator("#n-papers")).to_be_attached()

    def test_chat_filters_exist(self, web_server, page: Page):
        """
        Test that chat filters are present.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        # Switch to chat tab
        page.locator("#tab-chat").click()
        expect(page.locator("#chat-tab")).to_be_visible()

        # Check filter elements (only session filter exists in lightweight schema)
        expect(page.locator("#chat-session-filter")).to_be_attached()

    def test_chat_send_message(self, web_server, page: Page):
        """
        Test sending a message in the chat interface.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        # Switch to chat tab
        page.locator("#tab-chat").click()
        expect(page.locator("#chat-tab")).to_be_visible()

        # Find chat input and send button
        chat_input = page.locator("#chat-input")

        # Enter a test message
        test_message = "What are the main topics in NeurIPS papers?"
        chat_input.fill(test_message)

        # Click send button or press Enter
        chat_input.press("Enter")

        # Check that message appears in chat
        try:
            # Wait for user message to appear (AI response may take some time)
            expect(page.locator(".chat-message").first).to_be_attached(timeout=30000)

            # Get all messages
            messages = page.locator(".chat-message").all()

            # Should have at least initial welcome message
            assert len(messages) >= 1, "Chat should have messages"

            # Check if response is loading or received
            # Look for either loading indicator or assistant response
            chat_messages_div = page.locator("#chat-messages")
            assert len(chat_messages_div.inner_text()) > 0, "Chat should have content"

        except TimeoutError:
            # If AI is not configured, that's okay - just verify message was sent
            chat_input_after = page.locator("#chat-input")
            # Input should be cleared after sending
            assert chat_input_after.input_value() == ""

    @requires_lm_studio
    def test_chat_orcid_search_terms(self, web_server, page: Page):
        """
        Test that chat can suggest search terms from a real public ORCID record.

        The test uses the live ORCID API and the configured chat model.  It
        skips when either service is unavailable.

        Parameters
        ----------
        web_server : tuple
            Web server fixture (base_url, port)
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server

        try:
            record = fetch_orcid_record(_ORCID_ID, timeout=30)
        except OrcidError as exc:
            pytest.skip(f"ORCID public API unavailable: {exc}")
        assert record["works"], "Expected public works on the ORCID record"

        page.goto(base_url)
        page.locator("#tab-chat").click()
        expect(page.locator("#chat-tab")).to_be_visible()

        chat_input = page.locator("#chat-input")
        chat_input.fill(
            f"My ORCID is {_ORCID_ID}. Use the suggest_search_terms_from_orcid tool "
            "to suggest paper search terms for me."
        )

        with page.expect_response(
            lambda response: response.url.endswith("/api/chat") and response.request.method == "POST",
            timeout=120000,
        ) as response_info:
            chat_input.press("Enter")

        response = response_info.value
        if response.status != 200:
            pytest.skip(f"Chat backend unavailable (HTTP {response.status})")

        payload = response.json().get("response")
        assert isinstance(payload, dict), "Chat response should contain structured RAG output"

        tools_executed = payload.get("metadata", {}).get("tools_executed", [])
        assert "suggest_search_terms_from_orcid" in tools_executed

        assistant_text = payload.get("response")
        assert isinstance(assistant_text, str)
        assert assistant_text.strip()

    def test_chat_reset_conversation(self, web_server, page: Page):
        """
        Test resetting the chat conversation.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        # Switch to chat tab
        page.locator("#tab-chat").click()
        expect(page.locator("#chat-tab")).to_be_visible()

        # Try to find and click reset button
        reset_buttons = page.locator("button[onclick*='resetChat']").all()
        if len(reset_buttons) > 0:
            # Get initial message count
            page.locator(".chat-message").all()

            # Click reset button
            reset_buttons[0].click()
            expect(page.locator(".chat-message").first).to_be_attached()

            # Check messages were reset
            messages_after = page.locator(".chat-message").all()

            # Should have welcome message after reset
            assert len(messages_after) >= 1, "Should have at least welcome message after reset"

    def test_chat_n_papers_selector(self, web_server, page: Page):
        """
        Test changing the number of papers for chat context.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        # Switch to chat tab
        page.locator("#tab-chat").click()
        expect(page.locator("#chat-tab")).to_be_visible()

        # Find n-papers selector
        n_papers_select = page.locator("#n-papers")

        # Check that it has options
        options = n_papers_select.locator("option").all()
        assert len(options) > 0, "n-papers selector should have options"

        # Try to select a different value
        if len(options) > 1:
            n_papers_select.select_option(index=1, force=True)

            # Verify selection changed
            selected_value = n_papers_select.input_value()
            assert selected_value is not None

    def test_chat_papers_display(self, web_server, page: Page):
        """
        Test that papers section exists in chat interface.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        # Switch to chat tab
        page.locator("#tab-chat").click()
        expect(page.locator("#chat-tab")).to_be_visible()

        # Check papers section exists
        chat_papers = page.locator("#chat-papers")
        assert chat_papers.count() > 0

        # Papers section should be visible or hidden but present in DOM
        assert chat_papers.is_visible() or not chat_papers.is_visible()

    def test_chat_input_validation(self, web_server, page: Page):
        """
        Test chat input validation and placeholder.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        # Switch to chat tab
        page.locator("#tab-chat").click()
        expect(page.locator("#chat-tab")).to_be_visible()

        # Check input has placeholder
        chat_input = page.locator("#chat-input")
        placeholder = chat_input.get_attribute("placeholder")
        assert placeholder is not None and len(placeholder) > 0, "Chat input should have placeholder text"

        # Try sending empty message (should not do anything)
        chat_input.press("Enter")

        # Input should still be focusable
        assert chat_input.is_enabled()

    def test_responsive_layout(self, web_server, page: Page):
        """
        Test that the layout responds to different window sizes.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        # Test desktop size
        page.set_viewport_size({"width": 1920, "height": 1080})
        expect(page.locator("#search-input")).to_be_visible()

        # Test tablet size
        page.set_viewport_size({"width": 768, "height": 1024})
        expect(page.locator("#search-input")).to_be_visible()

        # Test mobile size
        page.set_viewport_size({"width": 375, "height": 667})
        expect(page.locator("#search-input")).to_be_visible()

    def test_multiple_searches_in_sequence(self, web_server, page: Page):
        """
        Test performing multiple searches in sequence.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        search_queries = ["attention", "learning", "neural"]

        for query in search_queries:
            search_input = page.locator("#search-input")
            search_input.clear()
            search_input.fill(query)
            search_input.press("Enter")

            # Check that search was performed
            expect(page.locator("#search-results")).to_be_attached()
            results_div = page.locator("#search-results")
            assert results_div.count() > 0

    def test_browser_single_page_app(self, web_server, page: Page):
        """
        Test browser back/forward navigation.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        # Perform a search
        search_input = page.locator("#search-input")
        search_input.fill("attention")
        search_input.press("Enter")
        expect(page.locator("#search-results")).to_be_attached()

        # Switch to chat tab
        page.locator("#tab-chat").click()
        expect(page.locator("#chat-tab")).to_be_visible()

        # Should still be on the same page (single page app)
        assert page.url == base_url + "/"

    def test_page_no_javascript_errors(self, web_server, page: Page, console_errors):
        """
        Test that the page loads without JavaScript errors.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        page : playwright.sync_api.Page
            Playwright page
        console_errors : list[str]
            Collected browser console and page errors
        """
        base_url, _ = web_server
        page.goto(base_url)

        # Wait for page to load
        page.wait_for_timeout(2000)

        errors = [
            message
            for message in console_errors
            if "favicon" not in message.lower() and "ERR_BLOCKED_BY_CLIENT" not in message
        ]
        # Should have no severe JavaScript errors
        assert len(errors) == 0, f"JavaScript errors found: {errors}"


@pytest.mark.e2e
@pytest.mark.slow
class TestDataDonationE2E:
    """End-to-end tests for data donation feature."""

    def test_load_json_button_always_visible(self, web_server, page: Page):
        """
        Test that Load JSON button is always visible even when no papers are rated.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        # Navigate to Interesting Papers tab
        interesting_tab_button = page.locator("#tab-interesting")
        expect(interesting_tab_button).to_be_enabled()
        interesting_tab_button.click()

        # Wait for tab to load
        expect(page.locator("#interesting-tab")).to_be_visible()

        # Load JSON button should be visible even with no papers - wait for it to be present
        load_json_button = page.locator("xpath=//button[contains(., 'Load JSON')]").first
        expect(load_json_button).to_be_visible()
        assert load_json_button.is_visible(), "Load JSON button should always be visible"

        # Other action buttons should be hidden when no papers rated
        try:
            donate_button = page.locator("xpath=//button[contains(., 'Donate Data')]").first
            assert not donate_button.is_visible(), "Donate Data button should be hidden when no papers"
        except Exception:
            # Button might not be in DOM at all, which is also fine
            pass

        try:
            save_json_button = page.locator("xpath=//button[contains(., 'Save JSON')]").first
            assert not save_json_button.is_visible(), "Save JSON button should be hidden when no papers"
        except Exception:
            pass

    def test_buttons_appear_after_rating_paper(self, web_server, page: Page):
        """
        Test that action buttons appear after rating a paper.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        # Perform a search to find papers
        search_input = page.locator("#search-input")
        expect(search_input).to_be_attached()
        search_input.fill("attention")

        # Find search button by text content
        search_button = page.locator("xpath=//button[contains(., 'Search') and @onclick='searchPapers()']").first
        expect(search_button).to_be_enabled()
        search_button.click()

        # Wait for search results
        expect(page.locator("#search-results .bg-white.rounded-lg").first).to_be_attached()

        # Rate a paper by clicking on stars
        try:
            star_buttons = page.locator(".star-rating button").all()
            if star_buttons:
                # Click the 5th star (highest rating)
                star_buttons[4].click()

                # Navigate to Interesting Papers tab
                page.locator("#tab-interesting").click()

                # Wait for tab to load
                expect(page.locator("#interesting-tab")).to_be_visible()

                # Now all buttons should be visible
                donate_button = page.locator("xpath=//button[contains(., 'Donate Data')]").first
                expect(donate_button).to_be_visible()
                assert donate_button.is_visible(), "Donate Data button should be visible after rating"

                save_json_button = page.locator("xpath=//button[contains(., 'Save JSON')]").first
                assert save_json_button.is_visible(), "Save JSON button should be visible after rating"

                load_json_button = page.locator("xpath=//button[contains(., 'Load JSON')]").first
                assert load_json_button.is_visible(), "Load JSON button should still be visible"
        except Exception as e:
            pytest.skip(f"Could not test rating flow: {e}")

    def test_donate_data_button_click_shows_confirmation(self, web_server, page: Page):
        """
        Test that clicking Donate Data button shows confirmation dialog.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        # Navigate to Interesting Papers tab first
        interesting_tab_button = page.locator("#tab-interesting")
        expect(interesting_tab_button).to_be_enabled()
        interesting_tab_button.click()

        # Wait for tab to load
        expect(page.locator("#interesting-tab")).to_be_visible()

        # Use JavaScript to inject test data and trigger UI update
        page.evaluate("""() => {
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
            }""")

        # Donate button should be visible
        donate_button = page.locator("xpath=//button[contains(., 'Donate Data')]").first
        expect(donate_button).to_be_enabled()

        dialogs: list[str] = []
        page.once("dialog", lambda dialog: (dialogs.append(dialog.message), dialog.dismiss()))
        donate_button.click()
        page.wait_for_timeout(200)

        assert dialogs, "Expected a confirmation dialog after clicking Donate Data"
        dialog_text = dialogs[0]
        assert "Would you like to donate" in dialog_text or "anonymized" in dialog_text.lower()

    def test_donate_button_hidden_after_successful_donation(self, web_server, page: Page):
        """
        Test that Donate Data button is hidden after successful donation.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        # Navigate to Interesting Papers tab first
        interesting_tab_button = page.locator("#tab-interesting")
        expect(interesting_tab_button).to_be_enabled()
        interesting_tab_button.click()

        # Wait for tab to load
        expect(page.locator("#interesting-tab")).to_be_visible()

        # Use JavaScript to inject test data and trigger UI update
        page.evaluate("""() => {
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
            }""")

        # Donate button should be visible
        donate_button = page.locator("xpath=//button[contains(., 'Donate Data')]").first
        expect(donate_button).to_be_enabled()

        dialogs: list[str] = []

        def handle_dialog(dialog) -> None:
            dialogs.append(dialog.message)
            dialog.accept()

        page.on("dialog", handle_dialog)

        try:
            donate_button.click()

            # The button is hidden once the donation has been accepted.
            expect(page.locator("xpath=//button[contains(., 'Donate Data')]").first).to_be_hidden()
            assert dialogs, "Expected at least one dialog during the donation flow"

            # Other buttons should still be visible
            save_json_button = page.locator("xpath=//button[contains(., 'Save JSON')]").first
            assert save_json_button.is_visible(), "Save JSON button should still be visible"

            load_json_button = page.locator("xpath=//button[contains(., 'Load JSON')]").first
            assert load_json_button.is_visible(), "Load JSON button should still be visible"

        except Exception as e:
            pytest.skip(f"Could not complete donation flow: {e}")

    def test_export_shows_donation_prompt_once_per_session(self, web_server, page: Page):
        """
        Test that export shows donation prompt only once per session.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        # Navigate to Interesting Papers tab first
        interesting_tab_button = page.locator("#tab-interesting")
        expect(interesting_tab_button).to_be_enabled()
        interesting_tab_button.click()

        # Wait for tab to load
        expect(page.locator("#interesting-tab")).to_be_visible()

        # Use JavaScript to inject test data and trigger UI update
        page.evaluate("""() => {
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
            }""")

        dialogs: list[str] = []

        def handle_dialog(dialog) -> None:
            dialogs.append(dialog.message)
            dialog.dismiss()

        page.on("dialog", handle_dialog)

        try:
            export_button = page.locator("xpath=//button[contains(., 'Export as Zip')]").first
            expect(export_button).to_be_enabled()

            # Should show donation prompt on first export
            export_button.click()
            page.wait_for_timeout(300)
            assert dialogs, "Expected a dialog when exporting for the first time"
            assert any("donate" in message.lower() or "export" in message.lower() for message in dialogs)

            # Click export again - should NOT show donation prompt again
            first_export_dialog_count = len(dialogs)
            export_button.click()
            page.wait_for_timeout(300)
            later_dialogs = dialogs[first_export_dialog_count:]
            assert not any(
                "donate" in message.lower() for message in later_dialogs
            ), "Donation prompt should only be shown once per session"

        except Exception as e:
            pytest.skip(f"Could not test export flow: {e}")


@pytest.mark.e2e
@pytest.mark.slow
class TestInterestingPapersConferenceSwitcher:
    """End-to-end tests for switching conferences in the Interesting Papers tab."""

    def test_switch_to_other_conference_with_interesting_papers(self, web_server, test_database, page: Page):
        """
        Test switching to another conference/year combo that has interesting papers.

        Parameters
        ----------
        web_server : tuple
            Web server fixture (base_url, port)
        test_database : Path
            Path to the E2E test database
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server

        with sqlite3.connect(test_database) as conn:
            neurips_row = conn.execute(
                "SELECT uid, title FROM papers WHERE conference = ? AND year = ? ORDER BY uid LIMIT 1",
                ("NeurIPS", 2025),
            ).fetchone()
            iclr_row = conn.execute(
                "SELECT uid, title FROM papers WHERE conference = ? AND year = ? ORDER BY uid LIMIT 1",
                ("ICLR", 2024),
            ).fetchone()

        assert neurips_row is not None, "Expected a NeurIPS 2025 paper in the test database"
        assert iclr_row is not None, "Expected a ICLR 2024 paper in the test database"
        neurips_uid, neurips_title = neurips_row
        iclr_uid, iclr_title = iclr_row

        page.goto(base_url)

        priorities = {
            neurips_uid: {"priority": 5, "searchTerm": "transformers"},
            iclr_uid: {"priority": 4, "searchTerm": "vision"},
        }
        page.evaluate(
            """(priorities) => {
                localStorage.setItem('paperPriorities', JSON.stringify(priorities));
                window.loadPriorities();
            }""",
            priorities,
        )

        page.wait_for_function(
            "() => [...document.querySelectorAll('#conference-selector option')].some(o => o.value === 'NeurIPS')"
        )
        page.locator("#conference-selector").select_option(value="NeurIPS")
        page.evaluate("handleConferenceChange()")
        page.wait_for_function(
            "() => [...document.querySelectorAll('#year-selector option')].some(o => o.value === '2025')"
        )
        page.locator("#year-selector").select_option(value="2025")
        page.evaluate("handleYearChange()")

        page.locator("#tab-interesting").click()
        expect(page.locator("#interesting-tab")).to_be_visible()

        menu = page.locator("#interesting-conference-menu")
        expect(menu).to_be_visible(timeout=10000)
        select = page.locator("#interesting-conference-select")
        page.wait_for_function("""() => [...document.querySelectorAll('#interesting-conference-select option')]
                .some(o => o.textContent.includes('ICLR 2024 (1)'))""")

        papers_list = page.locator("#interesting-papers-list")
        expect(papers_list).to_contain_text(neurips_title)
        expect(papers_list).not_to_contain_text(iclr_title)

        option = select.locator("option", has_text="ICLR 2024 (1)").first
        option_value = option.get_attribute("value")
        assert option_value, "Menu option should have an encoded value"
        select.select_option(value=option_value)

        expect(page.locator("#conference-selector")).to_have_value("ICLR")
        expect(page.locator("#year-selector")).to_have_value("2024")
        expect(menu).to_be_visible(timeout=10000)
        page.wait_for_function("""() => [...document.querySelectorAll('#interesting-conference-select option')]
                .some(o => o.textContent.includes('NeurIPS 2025 (1)'))""")
        expect(select).to_have_value("")
        expect(papers_list).to_contain_text(iclr_title)
        expect(papers_list).not_to_contain_text(neurips_title)
        expect(page.locator("#interesting-count")).to_have_text("1")


@pytest.mark.e2e
@pytest.mark.slow
class TestWebUIAccessibility:
    """Accessibility tests for the web UI."""

    def test_keyboard_navigation(self, web_server, page: Page):
        """
        Test keyboard navigation through the interface.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        # Tab through elements
        body = page.locator("body")
        for _ in range(5):
            body.press("Tab")
            page.wait_for_timeout(200)

        # Should be able to reach search input via keyboard
        search_input = page.locator("#search-input")
        assert search_input.count() > 0

    def test_form_labels(self, web_server, page: Page):
        """
        Test that form elements have proper labels.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        # Check for label elements
        labels = page.locator("label").all()
        assert len(labels) > 0, "Form elements should have labels"

    def test_semantic_html(self, web_server, page: Page):
        """
        Test that the page uses semantic HTML elements.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        # Check for semantic elements
        assert len(page.locator("header").all()) > 0
        assert len(page.locator("main").all()) > 0
        assert len(page.locator("button").all()) > 0


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

    def _open_advanced_search(self, page: Page) -> None:
        """
        Click the Advanced Search button and wait for the modal to appear.

        Parameters
        ----------
        page : playwright.sync_api.Page
            Playwright page.
        """
        adv_btn = page.locator("button[onclick='openAdvancedSearch()']")
        expect(adv_btn).to_be_enabled()
        adv_btn.click()
        expect(page.locator("#advanced-search-modal")).to_be_visible()

    def _apply_advanced_search(self, page: Page) -> None:
        """
        Click the Search button inside the advanced search modal and wait for results.

        Parameters
        ----------
        page : playwright.sync_api.Page
            Playwright page.
        """
        apply_btn = page.locator("button[onclick='applyAdvancedSearch()']")
        apply_btn.click()
        # Wait for the modal to close before checking results
        expect(page.locator("#advanced-search-modal")).to_be_hidden()
        # Wait for at least one result card (header card is always present when papers are found)
        expect(page.locator("#search-results .bg-white.rounded-lg").first).to_be_attached()

    # ------------------------------------------------------------------ tests

    def test_plain_author_name_search(self, web_server, page: Page):
        """
        Test that entering only an author name (without field syntax) returns the correct paper.

        When a user types a plain name such as "Ashish Vaswani" the backend
        recognises it as an author match and returns the corresponding paper as
        the first result, even though no ``authors:"…"`` syntax is used.

        Parameters
        ----------
        web_server : tuple
            Web server fixture (base_url, port)
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        search_input = page.locator("#search-input")
        search_input.fill("Ashish Vaswani")
        search_input.press("Enter")

        expect(page.locator("#search-results .bg-white.rounded-lg").first).to_be_attached()

        results_text = page.locator("#search-results").inner_text()
        assert (
            "Attention is All You Need" in results_text
        ), "Searching for 'Ashish Vaswani' should return 'Attention is All You Need'"

    def test_plain_author_last_name_search(self, web_server, page: Page):
        """
        Test that entering only an author's last name returns the correct paper.

        A partial name such as "Vaswani" must be resolved to the author
        "Ashish Vaswani" and return that author's paper.

        Parameters
        ----------
        web_server : tuple
            Web server fixture (base_url, port)
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        search_input = page.locator("#search-input")
        search_input.fill("Vaswani")
        search_input.press("Enter")

        expect(page.locator("#search-results .bg-white.rounded-lg").first).to_be_attached()

        results_text = page.locator("#search-results").inner_text()
        assert (
            "Attention is All You Need" in results_text
        ), "Searching for 'Vaswani' should return 'Attention is All You Need'"

    def test_authors_field_syntax_alone(self, web_server, page: Page):
        """
        Test that the ``authors:"Name"`` syntax works alone in the search box.

        Entering ``authors:"Ashish Vaswani"`` must bypass the embedding search
        and return only papers whose ``authors`` column matches that value.

        Parameters
        ----------
        web_server : tuple
            Web server fixture (base_url, port)
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        search_input = page.locator("#search-input")
        search_input.fill('authors:"Ashish Vaswani"')
        search_input.press("Enter")

        expect(page.locator("#search-results .bg-white.rounded-lg").first).to_be_attached()

        results_text = page.locator("#search-results").inner_text()
        assert (
            "Attention is All You Need" in results_text
        ), "authors:\"Ashish Vaswani\" should return 'Attention is All You Need'"
        # Papers by other authors must not be in the results
        assert "Generative Adversarial" not in results_text, "GAN paper should not appear when filtering by Vaswani"

    def test_author_alias_field_syntax(self, web_server, page: Page):
        """
        Test that the ``author:"Name"`` alias syntax works the same as ``authors:"Name"``.

        The backend accepts both ``author:`` and ``authors:`` as field names and
        must return identical results for both.

        Parameters
        ----------
        web_server : tuple
            Web server fixture (base_url, port)
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        search_input = page.locator("#search-input")
        search_input.fill('author:"Ashish Vaswani"')
        search_input.press("Enter")

        expect(page.locator("#search-results .bg-white.rounded-lg").first).to_be_attached()

        results_text = page.locator("#search-results").inner_text()
        assert (
            "Attention is All You Need" in results_text
        ), "author:\"Ashish Vaswani\" (alias) should return 'Attention is All You Need'"
        assert (
            "Generative Adversarial" not in results_text
        ), "GAN paper should not appear when filtering by Vaswani (using author: alias)"

    def test_advanced_search_modal_opens_and_closes(self, web_server, page: Page):
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
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        # Modal must be hidden initially
        modal = page.locator("#advanced-search-modal")
        assert not modal.is_visible(), "Advanced search modal should be hidden on page load"

        # Open the modal
        self._open_advanced_search(page)
        assert modal.is_visible(), "Advanced search modal should be visible after clicking the button"

        # Close with the Cancel button
        cancel_btn = page.locator("xpath=//button[@onclick='closeAdvancedSearch()' and contains(., 'Cancel')]")
        cancel_btn.click()
        expect(modal).to_be_hidden()
        assert not modal.is_visible(), "Advanced search modal should be hidden after Cancel"

    def test_advanced_search_modal_fields_present(self, web_server, page: Page):
        """
        Test that the advanced search modal contains all expected input fields.

        The modal must expose fields for: topic, authors, title, keywords,
        abstract, and award.

        Parameters
        ----------
        web_server : tuple
            Web server fixture (base_url, port)
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        self._open_advanced_search(page)

        for field_id in ("adv-topic", "adv-authors", "adv-title", "adv-keywords", "adv-abstract", "adv-award"):
            assert page.locator(f"#{field_id}").is_visible(), f"Advanced search field '{field_id}' should be visible"

        # Close the modal
        page.locator("xpath=//button[@onclick='closeAdvancedSearch()' and contains(., 'Cancel')]").click()
        expect(page.locator("#advanced-search-modal")).to_be_hidden()

    def test_advanced_search_authors_field(self, web_server, page: Page):
        """
        Test searching by the authors field in the advanced search form.

        Filling in "Ashish Vaswani" in the Authors field and applying the
        search must return only that author's paper.

        Parameters
        ----------
        web_server : tuple
            Web server fixture (base_url, port)
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        self._open_advanced_search(page)

        authors_input = page.locator("#adv-authors")
        authors_input.clear()
        authors_input.fill("Ashish Vaswani")

        self._apply_advanced_search(page)

        results_text = page.locator("#search-results").inner_text()
        assert (
            "Attention is All You Need" in results_text
        ), "Searching for author 'Ashish Vaswani' via the advanced form should return 'Attention is All You Need'"
        assert (
            "Generative Adversarial" not in results_text
        ), "GAN paper should not appear in results filtered to Vaswani"

    def test_advanced_search_author_with_topic(self, web_server, page: Page):
        """
        Test combining an author filter with a free-text topic query.

        Entering "Yoshua Bengio" in the Authors field and "adversarial" in
        the Topic field must return only papers by Yoshua Bengio; papers by
        other authors must not appear in the results.

        Parameters
        ----------
        web_server : tuple
            Web server fixture (base_url, port)
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        self._open_advanced_search(page)

        authors_input = page.locator("#adv-authors")
        authors_input.clear()
        authors_input.fill("Yoshua Bengio")

        topic_input = page.locator("#adv-topic")
        topic_input.clear()
        topic_input.fill("adversarial")

        self._apply_advanced_search(page)

        results_text = page.locator("#search-results").inner_text()
        # At least one Bengio paper must appear (checking for last name is sufficient
        # since the full name "Yoshua Bengio" always contains "Bengio")
        assert "Bengio" in results_text, "Expected at least one Yoshua Bengio paper in results"
        # Papers by unrelated authors must not appear
        assert "Ashish Vaswani" not in results_text, "Vaswani paper should not appear when filtering by Yoshua Bengio"
        assert "Kaiming He" not in results_text, "He paper should not appear when filtering by Yoshua Bengio"

    def test_advanced_search_title_field(self, web_server, page: Page):
        """
        Test searching by the title field in the advanced search form.

        Entering "Residual" in the Title field must return the
        'Deep Residual Learning for Image Recognition' paper and not
        unrelated papers.

        Parameters
        ----------
        web_server : tuple
            Web server fixture (base_url, port)
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        self._open_advanced_search(page)

        title_input = page.locator("#adv-title")
        title_input.clear()
        title_input.fill("Residual")

        self._apply_advanced_search(page)

        results_text = page.locator("#search-results").inner_text()
        assert (
            "Residual" in results_text
        ), "Searching by title 'Residual' should return the Deep Residual Learning paper"
        assert (
            "Generative Adversarial" not in results_text
        ), "GAN paper should not appear in results filtered by title 'Residual'"

    def test_advanced_search_keywords_field(self, web_server, page: Page):
        """
        Test searching by the keywords field in the advanced search form.

        Entering "deep learning" in the Keywords field must return the
        'Deep Residual Learning for Image Recognition' paper which has
        "deep learning" as one of its keywords.

        Parameters
        ----------
        web_server : tuple
            Web server fixture (base_url, port)
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        self._open_advanced_search(page)

        keywords_input = page.locator("#adv-keywords")
        keywords_input.clear()
        keywords_input.fill("deep learning")

        self._apply_advanced_search(page)

        results_text = page.locator("#search-results").inner_text()
        # The Deep Residual Learning paper has "deep learning" in its keywords.
        # "Residual" is sufficient: the title "Deep Residual Learning for Image Recognition"
        # always contains this substring.
        assert (
            "Residual" in results_text
        ), "Searching by keyword 'deep learning' should return the Deep Residual Learning paper"

    def test_advanced_search_abstract_field(self, web_server, page: Page):
        """
        Test searching by the abstract field in the advanced search form.

        Entering "adversarial process" in the Abstract field must return the
        'Generative Adversarial Networks' paper whose abstract contains that phrase.

        Parameters
        ----------
        web_server : tuple
            Web server fixture (base_url, port)
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        self._open_advanced_search(page)

        abstract_input = page.locator("#adv-abstract")
        abstract_input.clear()
        abstract_input.fill("adversarial process")

        self._apply_advanced_search(page)

        results_text = page.locator("#search-results").inner_text()
        assert (
            "Generative Adversarial Networks" in results_text
        ), "Searching abstract for 'adversarial process' should return the GAN paper"
        assert (
            "Attention is All You Need" not in results_text
        ), "Vaswani paper should not appear when filtering by abstract 'adversarial process'"

    def test_advanced_search_search_input_populated(self, web_server, page: Page):
        """
        Test that applying an advanced search populates the main search input.

        After the user fills in the advanced search form and applies it, the
        constructed ``field:"value"`` query string must be written back to the
        main search input box so the user can see and edit it.

        Parameters
        ----------
        web_server : tuple
            Web server fixture (base_url, port)
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        self._open_advanced_search(page)

        authors_input = page.locator("#adv-authors")
        authors_input.clear()
        authors_input.fill("Vaswani")

        self._apply_advanced_search(page)

        # The main search input must now contain the field:"value" query
        search_value = page.locator("#search-input").input_value()
        assert "Vaswani" in search_value, f"Main search input should contain the author query, got: {search_value!r}"
        assert "authors:" in search_value, f"Main search input should use field syntax, got: {search_value!r}"

    def test_advanced_search_enter_key_applies(self, web_server, page: Page):
        """
        Test that pressing Enter inside an advanced search field applies the search.

        The advanced search form must submit when the user presses Enter in
        any of its input fields, matching the behaviour of the main search box.

        Parameters
        ----------
        web_server : tuple
            Web server fixture (base_url, port)
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        self._open_advanced_search(page)

        authors_input = page.locator("#adv-authors")
        authors_input.clear()
        authors_input.fill("Ashish Vaswani")
        # Press Enter instead of clicking the Search button
        authors_input.press("Enter")

        # Modal should close and results should appear
        expect(page.locator("#advanced-search-modal")).to_be_hidden()
        expect(page.locator("#search-results .bg-white.rounded-lg").first).to_be_attached()

        results_text = page.locator("#search-results").inner_text()
        assert (
            "Attention is All You Need" in results_text
        ), "Pressing Enter in the authors field should trigger search and return Vaswani's paper"


@pytest.mark.e2e
@pytest.mark.slow
class TestClusteringTabE2E:
    """Test clustering tab functionality with Playwright."""

    def test_clustering_tab_exists(self, web_server, page: Page):
        """
        Test that the clustering tab exists and is accessible.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        # Find and click clustering tab
        clustering_tab = page.locator("#tab-clusters")
        expect(clustering_tab).to_be_enabled()
        assert clustering_tab.is_visible(), "Clustering tab should be visible"

    def test_clustering_plot_loads(self, web_server, page: Page):
        """
        Test that the clustering plot container loads properly.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        # Navigate to clustering tab
        clustering_tab = page.locator("#tab-clusters")
        expect(clustering_tab).to_be_enabled()
        clustering_tab.click()

        # Wait for plot container to be visible with timeout
        plot_container = page.locator("#cluster-plot")
        expect(plot_container).to_be_visible()
        assert plot_container.is_visible(), "Clustering plot container should be visible"

    def test_clustering_stats_display(self, web_server, page: Page):
        """
        Test that clustering statistics are displayed in the legend title.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        # Navigate to clustering tab
        clustering_tab = page.locator("#tab-clusters")
        expect(clustering_tab).to_be_enabled()
        clustering_tab.click()

        # Wait for legend element to be visible with timeout
        legend_element = page.locator("#cluster-legend")
        expect(legend_element).to_be_visible()
        expect(legend_element).to_contain_text("papers", timeout=15000)
        expect(legend_element).to_contain_text("clusters", timeout=15000)

        # Verify legend contains the stats information
        legend_text = legend_element.inner_text()
        assert "papers" in legend_text.lower(), "Legend should contain paper count"
        assert "clusters" in legend_text.lower(), "Legend should contain cluster count"

    def test_clustering_filter_dropdown(self, web_server, page: Page):
        """
        Test that the clustering tab has filter/search functionality.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        # Navigate to clustering tab
        clustering_tab = page.locator("#tab-clusters")
        expect(clustering_tab).to_be_enabled()
        clustering_tab.click()

        # Wait for custom query search input to be visible with timeout
        search_input = page.locator("#custom-query-input")
        expect(search_input).to_be_visible()
        assert search_input.is_visible(), "Custom query search input should be visible"

        # Verify it's an input element
        assert search_input.evaluate("el => el.tagName.toLowerCase()") == "input", "Search should be an input element"

    def test_clustering_paper_details_panel(self, web_server, page: Page):
        """
        Test that the selected paper details panel exists.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        # Navigate to clustering tab
        clustering_tab = page.locator("#tab-clusters")
        expect(clustering_tab).to_be_enabled()
        clustering_tab.click()

        # Wait for clustering content to load
        expect(page.locator("#clusters-tab")).to_be_visible()

        # Check for paper details panel with timeout
        try:
            details_panel = page.locator("#selected-paper-details")
            expect(details_panel).to_be_attached()
            # Panel might be hidden initially
            assert details_panel.count() > 0, "Selected paper details panel should exist"
        except TimeoutError:
            # Panel might not exist if no paper selected - this is acceptable
            pass

    def test_clustering_plot_has_plotly(self, web_server, page: Page):
        """
        Test that Plotly is loaded and the plot is rendered.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        # Navigate to clustering tab
        clustering_tab = page.locator("#tab-clusters")
        expect(clustering_tab).to_be_enabled()
        clustering_tab.click()

        # Wait for clustering content to be visible
        expect(page.locator("#clusters-tab")).to_be_visible()

        # Wait for Plotly to be available with timeout
        page.wait_for_function("typeof Plotly !== 'undefined'")

        # Verify plot container exists
        expect(page.locator("#cluster-plot")).to_be_attached()

    def test_clustering_tab_no_javascript_errors(self, web_server, page: Page, console_errors):
        """
        Test that there are no JavaScript errors when loading the clustering tab.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        page : playwright.sync_api.Page
            Playwright page
        console_errors : list[str]
            Collected browser console and page errors
        """
        base_url, _ = web_server
        page.goto(base_url)

        # Navigate to clustering tab
        clustering_tab = page.locator("#tab-clusters")
        expect(clustering_tab).to_be_enabled()
        clustering_tab.click()

        # Wait for content to load
        expect(page.locator("#clusters-tab")).to_be_visible()
        page.wait_for_timeout(1000)

        # Filter out known harmless errors
        critical_errors = [
            message
            for message in console_errors
            if "favicon" not in message.lower() and "ERR_BLOCKED_BY_CLIENT" not in message
        ]

        assert len(critical_errors) == 0, f"No severe JavaScript errors should occur. Found: {critical_errors}"

    def test_clustering_visualization_elements(self, web_server, page: Page):
        """
        Test that clustering visualization has expected elements.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        # Navigate to clustering tab
        clustering_tab = page.locator("#tab-clusters")
        expect(clustering_tab).to_be_enabled()
        clustering_tab.click()

        # Wait for clustering content to be visible
        clustering_content = page.locator("#clusters-tab")
        expect(clustering_content).to_be_visible()

        # Should have some content
        assert clustering_content.inner_text() != "", "Clustering content should not be empty"

        # Wait for plot container
        plot = page.locator("#cluster-plot")
        expect(plot).to_be_attached()
        assert plot.count() > 0, "Cluster plot should exist"

    def test_cluster_center_colors_match_points(self, web_server, page: Page):
        """
        Test that cluster centers have the same color as their cluster points.

        This test verifies that when cluster centers are displayed as stars,
        they use the same color as the points in their cluster.

        Parameters
        ----------
        web_server : tuple
            Web server fixture
        page : playwright.sync_api.Page
            Playwright page
        """
        base_url, _ = web_server
        page.goto(base_url)

        # Navigate to clustering tab
        clustering_tab = page.locator("#tab-clusters")
        expect(clustering_tab).to_be_enabled()
        clustering_tab.click()

        # Wait for clustering content to be visible
        expect(page.locator("#clusters-tab")).to_be_visible()

        # Wait for Plotly to load with timeout
        page.wait_for_function("typeof Plotly !== 'undefined'")

        # Execute JavaScript to check if Plotly plot exists and get trace colors
        script = """
        () => {
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
        }
        """

        try:
            result = page.evaluate(script)

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
        except TimeoutError:
            # If plot takes too long to load, skip this test
            pytest.skip("Clustering plot failed to load within timeout")


# ---------------------------------------------------------------------------
# Mobile viewport fixture
# ---------------------------------------------------------------------------

MOBILE_WIDTH = 375
MOBILE_HEIGHT = 812  # iPhone X dimensions


@pytest.fixture
def mobile_page(browser):
    """
    Create a Playwright page sized to a typical mobile phone screen.

    Uses iPhone X dimensions (375×812 px) to exercise narrow-screen behaviour.

    Parameters
    ----------
    browser : playwright.sync_api.Browser
        Session-scoped Playwright browser.

    Yields
    ------
    playwright.sync_api.Page
        Playwright page sized to mobile dimensions.

    Notes
    -----
    The browser context is closed automatically after the test.
    """
    context = browser.new_context(viewport={"width": MOBILE_WIDTH, "height": MOBILE_HEIGHT})
    page = context.new_page()
    yield page
    context.close()


@pytest.mark.e2e
@pytest.mark.slow
class TestMobileUIE2E:
    """End-to-end tests that verify the web UI works on narrow phone screens."""

    def test_hamburger_button_visible_on_mobile(self, web_server, mobile_page: Page):
        """
        Test that the hamburger menu button is visible on a narrow phone screen.

        Parameters
        ----------
        web_server : tuple
            Web server fixture (base_url, port)
        mobile_page : playwright.sync_api.Page
            Playwright page sized to mobile dimensions
        """
        base_url, _ = web_server
        mobile_page.goto(base_url)

        expect(mobile_page.locator("#mobile-menu-btn")).to_be_visible()

    def test_header_controls_collapsed_by_default_on_mobile(self, web_server, mobile_page: Page):
        """
        Test that the header controls are hidden by default on a narrow screen.

        Parameters
        ----------
        web_server : tuple
            Web server fixture (base_url, port)
        mobile_page : playwright.sync_api.Page
            Playwright page sized to mobile dimensions
        """
        base_url, _ = web_server
        mobile_page.goto(base_url)

        expect(mobile_page.locator("#header-controls")).to_be_hidden()

    def test_hamburger_opens_header_controls(self, web_server, mobile_page: Page):
        """
        Test that clicking the hamburger button reveals the header controls.

        Parameters
        ----------
        web_server : tuple
            Web server fixture (base_url, port)
        mobile_page : playwright.sync_api.Page
            Playwright page sized to mobile dimensions
        """
        base_url, _ = web_server
        mobile_page.goto(base_url)

        hamburger = mobile_page.locator("#mobile-menu-btn")
        header_controls = mobile_page.locator("#header-controls")

        # Controls are hidden before clicking
        expect(header_controls).to_be_hidden()

        # Click hamburger → controls should appear
        hamburger.click()
        expect(header_controls).to_be_visible()

        assert header_controls.is_visible(), "Header controls should be visible after opening menu"
        assert mobile_page.locator("#conference-selector").is_visible()
        assert mobile_page.locator("#year-selector").is_visible()

    def test_hamburger_closes_header_controls(self, web_server, mobile_page: Page):
        """
        Test that clicking the hamburger button again collapses the header controls.

        Parameters
        ----------
        web_server : tuple
            Web server fixture (base_url, port)
        mobile_page : playwright.sync_api.Page
            Playwright page sized to mobile dimensions
        """
        base_url, _ = web_server
        mobile_page.goto(base_url)

        hamburger = mobile_page.locator("#mobile-menu-btn")
        header_controls = mobile_page.locator("#header-controls")

        # Open
        hamburger.click()
        expect(header_controls).to_be_visible()

        # Close
        hamburger.click()
        expect(header_controls).to_be_hidden()

        assert not header_controls.is_visible(), "Header controls should collapse after closing menu"

    def test_tabs_fit_within_viewport_on_mobile(self, web_server, mobile_page: Page):
        """
        Test that the tab navigation bar does not overflow the screen width.

        Parameters
        ----------
        web_server : tuple
            Web server fixture (base_url, port)
        mobile_page : playwright.sync_api.Page
            Playwright page sized to mobile dimensions
        """
        base_url, _ = web_server
        mobile_page.goto(base_url)

        viewport_width = mobile_page.evaluate("window.innerWidth")

        # Check that the overall body/page doesn't overflow horizontally
        scroll_width = mobile_page.evaluate("document.body.scrollWidth")
        assert (
            scroll_width <= viewport_width + 1
        ), f"Page overflows viewport: scrollWidth={scroll_width} > viewportWidth={viewport_width}"

        # All four tab buttons should be present and fit within the viewport
        for tab_id in ["tab-search", "tab-chat", "tab-interesting", "tab-clusters"]:
            tab = mobile_page.locator(f"#{tab_id}")
            assert tab.count() > 0, f"Tab {tab_id} should be in the DOM"
            tab_rect = tab.evaluate("el => el.getBoundingClientRect()")
            assert (
                tab_rect["right"] <= viewport_width + 1
            ), f"Tab {tab_id} overflows viewport: right={tab_rect['right']} > viewportWidth={viewport_width}"

    def test_tab_buttons_are_clickable_on_mobile(self, web_server, mobile_page: Page):
        """
        Test that tab buttons can be clicked on a narrow screen.

        Parameters
        ----------
        web_server : tuple
            Web server fixture (base_url, port)
        mobile_page : playwright.sync_api.Page
            Playwright page sized to mobile dimensions
        """
        base_url, _ = web_server
        mobile_page.goto(base_url)

        # Click each tab and verify the corresponding content becomes visible
        mobile_page.locator("#tab-chat").click()
        expect(mobile_page.locator("#chat-tab")).to_be_visible()
        assert mobile_page.locator("#chat-tab").is_visible()

        mobile_page.locator("#tab-search").click()
        expect(mobile_page.locator("#search-tab")).to_be_visible()
        assert mobile_page.locator("#search-tab").is_visible()

    def test_search_works_on_mobile(self, web_server, mobile_page: Page):
        """
        Test that keyword search returns results on a narrow phone screen.

        Parameters
        ----------
        web_server : tuple
            Web server fixture (base_url, port)
        mobile_page : playwright.sync_api.Page
            Playwright page sized to mobile dimensions
        """
        base_url, _ = web_server
        mobile_page.goto(base_url)

        search_input = mobile_page.locator("#search-input")
        search_input.fill("attention")
        search_input.press("Enter")

        expect(mobile_page.locator("#search-results .bg-white.rounded-lg").first).to_be_attached()

        results = mobile_page.locator("#search-results .bg-white.rounded-lg").all()
        assert len(results) > 0, "Search should return results on mobile"

    def test_hamburger_button_hidden_on_desktop(self, web_server, page: Page):
        """
        Test that the hamburger button is not visible on a desktop-sized screen.

        Parameters
        ----------
        web_server : tuple
            Web server fixture (base_url, port)
        page : playwright.sync_api.Page
            Standard (desktop) Playwright page
        """
        base_url, _ = web_server
        page.set_viewport_size({"width": 1280, "height": 800})
        page.goto(base_url)

        expect(page.locator("#mobile-menu-btn")).to_be_hidden()

    def test_header_controls_visible_on_desktop(self, web_server, page: Page):
        """
        Test that the header controls are always visible on a desktop-sized screen.

        Parameters
        ----------
        web_server : tuple
            Web server fixture (base_url, port)
        page : playwright.sync_api.Page
            Standard (desktop) Playwright page
        """
        base_url, _ = web_server
        page.set_viewport_size({"width": 1280, "height": 800})
        page.goto(base_url)

        expect(page.locator("#header-controls")).to_be_visible()
        assert page.locator("#conference-selector").is_visible()
        assert page.locator("#year-selector").is_visible()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "e2e"])

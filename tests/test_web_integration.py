"""
Integration tests for the web UI.

These tests verify that the web UI server can start, serve pages,
and handle API requests correctly.
"""

import pytest
import sys
import time
import os
import requests
import threading
from pathlib import Path
from unittest.mock import patch, Mock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from abstracts_explorer.database import DatabaseManager
from tests.conftest import set_test_db
from tests.helpers import requires_lm_studio, find_free_port

# Helper functions imported from test_helpers:
# - check_lm_studio_available(): Check if LM Studio is running
# - requires_lm_studio: Skip marker for tests requiring LM Studio
# - find_free_port(): Find a free port for testing

# Constants for tests
MOCK_EMBEDDING_DIMENSION = 4096  # Standard dimension for test embeddings

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def test_database(tmp_path_factory, web_test_papers):
    """
    Create a test database with sample data.

    Parameters
    ----------
    tmp_path_factory : TempPathFactory
        Pytest fixture for creating temporary directories
    web_test_papers : list
        List of test papers from shared fixture

    Returns
    -------
    Path
        Path to the test database

    Notes
    -----
    Uses the shared web_test_papers fixture from conftest.py to ensure
    consistency across web-related tests.
    """
    tmp_dir = tmp_path_factory.mktemp("data")
    db_path = tmp_dir / "test_web_integration.db"

    # Create database and add test data using LightweightPaper
    set_test_db(str(db_path))
    db = DatabaseManager()

    with db:
        db.create_tables()
        db.add_papers(web_test_papers)

    return db_path


@pytest.fixture(scope="module")
def web_server(test_database, tmp_path_factory):
    """
    Start the web server for testing.

    Parameters
    ----------
    test_database : Path
        Path to the test database
    tmp_path_factory : TempPathFactory
        Pytest fixture for creating temporary directories

    Yields
    ------
    tuple
        (host, port, base_url)
    """
    # Check if Flask is installed
    try:
        import importlib.util

        if importlib.util.find_spec("flask") is None:
            pytest.skip("Flask not installed - web UI tests require 'uv sync --extra web'")
    except ImportError:
        pytest.skip("Flask not installed - web UI tests require 'uv sync --extra web'")

    import uuid
    import chromadb.api.shared_system_client

    # Clear ChromaDB's global client registry to avoid conflicts with other test modules
    chromadb.api.shared_system_client.SharedSystemClient._identifier_to_system.clear()

    port = find_free_port()
    host = "127.0.0.1"
    base_url = f"http://{host}:{port}"

    # Create test embeddings database with unique path
    unique_id = f"{int(time.time())}_{str(uuid.uuid4())[:8]}"
    tmp_dir = tmp_path_factory.mktemp("web_integration_embeddings")
    embeddings_path = tmp_dir / f"web_integration_chroma_{unique_id}"
    collection_name = f"test_collection_{unique_id}"

    # Set environment variables BEFORE importing the app modules
    # This ensures the config is loaded with the correct test paths
    original_paper_db = os.environ.get("PAPER_DB")
    original_embedding_db = os.environ.get("EMBEDDING_DB")
    original_collection_name = os.environ.get("COLLECTION_NAME")

    os.environ["PAPER_DB"] = str(test_database)
    os.environ["EMBEDDING_DB"] = str(embeddings_path)
    os.environ["COLLECTION_NAME"] = collection_name

    # Mock the OpenAI API for embedding generation
    # Integration tests should not require a real API connection
    mock_openai_patcher = patch("abstracts_explorer.embeddings.OpenAI")
    mock_openai_class = mock_openai_patcher.start()

    try:
        # Create mock OpenAI client instance
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock models.list() for connection test
        mock_models = Mock()
        mock_client.models.list.return_value = mock_models

        # Mock embeddings.create() for embedding generation
        mock_embedding_response = Mock()
        mock_embedding_data = Mock()
        mock_embedding_data.embedding = [0.1] * MOCK_EMBEDDING_DIMENSION
        mock_embedding_response.data = [mock_embedding_data]
        mock_client.embeddings.create.return_value = mock_embedding_response

        # Import after setting environment variables and mocking
        from abstracts_explorer.web_ui import app as flask_app
        from abstracts_explorer.embeddings import EmbeddingsManager
        from abstracts_explorer.config import get_config

        # Force reload config to pick up environment variables with .env.test
        from tests.conftest import get_env_test_path

        get_config(reload=True, env_path=get_env_test_path())

        # Initialize embeddings with test data
        em = EmbeddingsManager(collection_name=collection_name)
        em.connect()
        em.create_collection(reset=True)

        # Add embeddings for test papers
        from abstracts_explorer.database import DatabaseManager

        set_test_db(str(test_database))
        db = DatabaseManager()
        db.connect()
        papers = db.query("SELECT * FROM papers")

        for paper in papers:
            em.add_paper(paper)

        db.close()

        # Inject the pre-created embeddings manager directly
        import abstracts_explorer.web_ui.app as app_module

        app_module.embeddings_manager = em
        app_module.rag_chat = None

        # Use werkzeug's make_server for better cross-platform compatibility
        # This works more reliably in threads than Flask's app.run()
        from werkzeug.serving import make_server

        server = make_server(host, port, flask_app, threaded=True)

        # Start server in a thread
        def run_server():
            server.serve_forever()

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()

        # Wait for server to start
        max_retries = 30
        for i in range(max_retries):
            try:
                response = requests.get(base_url, timeout=1)
                if response.status_code in [200, 404]:
                    break
            except requests.exceptions.RequestException:
                if i == max_retries - 1:
                    server.shutdown()
                    pytest.fail("Web server failed to start")
                time.sleep(0.5)

        yield (host, port, base_url)

        # Shutdown server gracefully
        server.shutdown()

    finally:
        # Ensure mock is always stopped
        mock_openai_patcher.stop()

        # Restore original environment variables
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


class TestWebUIIntegration:
    """Integration tests for the web UI."""

    def test_server_starts(self, web_server):
        """Test that the web server starts and responds."""
        host, port, base_url = web_server

        response = requests.get(base_url, timeout=5)
        assert response.status_code == 200
        assert b"Abstracts Explorer" in response.content

    def test_static_files_served(self, web_server):
        """Test that static files are served correctly."""
        host, port, base_url = web_server

        # Test JavaScript file
        response = requests.get(f"{base_url}/static/app.js", timeout=5)
        assert response.status_code == 200
        assert b"searchPapers" in response.content or b"function" in response.content

        # Test CSS file
        response = requests.get(f"{base_url}/static/style.css", timeout=5)
        assert response.status_code == 200

    def test_api_stats_endpoint(self, web_server):
        """Test that the stats API endpoint works."""
        host, port, base_url = web_server

        response = requests.get(f"{base_url}/api/stats", timeout=5)
        if response.status_code != 200:
            # Print error for debugging
            try:
                error_data = response.json()
                print(f"Error response: {error_data}")
            except Exception:
                print(f"Error response (non-JSON): {response.text}")
        assert response.status_code == 200

        data = response.json()
        assert "total_papers" in data
        assert data["total_papers"] >= 3  # We added 3 test papers

    def test_api_search_keyword(self, web_server):
        """Test keyword search through the API."""
        host, port, base_url = web_server

        search_data = {"query": "transformer", "use_embeddings": False, "limit": 10}

        response = requests.post(f"{base_url}/api/search", json=search_data, timeout=5)

        assert response.status_code == 200

        data = response.json()
        assert "papers" in data
        assert "count" in data
        assert "query" in data
        assert data["query"] == "transformer"

        # Should find papers with "transformer" in name or abstract
        assert isinstance(data["papers"], list)

    def test_api_search_validation(self, web_server):
        """Test that search validates input correctly."""
        host, port, base_url = web_server

        # Test missing query
        response = requests.post(f"{base_url}/api/search", json={"use_embeddings": False}, timeout=5)
        assert response.status_code == 400

        data = response.json()
        assert "error" in data

        # Test empty query
        response = requests.post(f"{base_url}/api/search", json={"query": "", "use_embeddings": False}, timeout=5)
        assert response.status_code == 400

    def test_api_paper_detail(self, web_server):
        """Test getting paper details."""
        host, port, base_url = web_server

        # First, search for a paper to get its UID
        search_data = {"query": "attention", "use_embeddings": False, "limit": 1}

        response = requests.post(f"{base_url}/api/search", json=search_data, timeout=5)

        assert response.status_code == 200
        papers = response.json()["papers"]

        if papers:
            paper_uid = papers[0]["uid"]

            # Get paper details
            response = requests.get(f"{base_url}/api/paper/{paper_uid}", timeout=5)
            assert response.status_code == 200

            paper = response.json()
            assert "uid" in paper
            assert "title" in paper
            assert paper["uid"] == paper_uid

    def test_api_paper_not_found(self, web_server):
        """Test getting a non-existent paper."""
        host, port, base_url = web_server

        response = requests.get(f"{base_url}/api/paper/999999", timeout=5)
        assert response.status_code == 404

        data = response.json()
        assert "error" in data

    def test_api_chat_validation(self, web_server):
        """Test chat API validation."""
        host, port, base_url = web_server

        # Test missing message
        response = requests.post(f"{base_url}/api/chat", json={}, timeout=5)
        assert response.status_code == 400

        data = response.json()
        assert "error" in data

        # Test empty message
        response = requests.post(f"{base_url}/api/chat", json={"message": ""}, timeout=5)
        assert response.status_code == 400

    def test_api_chat_reset(self, web_server):
        """Test chat reset endpoint."""
        host, port, base_url = web_server

        response = requests.post(f"{base_url}/api/chat/reset", timeout=5)
        # Should return 200 or 500 (if LM Studio not running)
        assert response.status_code in [200, 500]

    def test_concurrent_requests(self, web_server):
        """Test that the server handles concurrent requests."""
        host, port, base_url = web_server

        results = []

        def make_request():
            try:
                response = requests.get(f"{base_url}/api/stats", timeout=5)
                results.append(response.status_code)
            except Exception:
                results.append(None)

        # Make 5 concurrent requests
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            thread.start()
            threads.append(thread)

        # Wait for all threads
        for thread in threads:
            thread.join(timeout=10)

        # All requests should succeed
        assert len(results) == 5
        assert all(status == 200 for status in results)

    def test_search_with_different_limits(self, web_server):
        """Test search with different limit values."""
        host, port, base_url = web_server

        for limit in [1, 5, 10]:
            search_data = {"query": "learning", "use_embeddings": False, "limit": limit}

            response = requests.post(f"{base_url}/api/search", json=search_data, timeout=5)

            assert response.status_code == 200

            data = response.json()
            assert len(data["papers"]) <= limit

    def test_search_special_characters(self, web_server):
        """Test search with special characters."""
        host, port, base_url = web_server

        special_queries = [
            "neural & networks",
            "attention-mechanisms",
            "deep learning (2024)",
            "BERT",
        ]

        for query in special_queries:
            search_data = {"query": query, "use_embeddings": False, "limit": 10}

            response = requests.post(f"{base_url}/api/search", json=search_data, timeout=5)

            # Should handle gracefully (either 200 or 400, not crash)
            assert response.status_code in [200, 400]

    def test_semantic_search_embeddings_manager_init(self, web_server):
        """
        Test that semantic search initializes EmbeddingsManager correctly.

        This test verifies that EmbeddingsManager uses configuration from
        environment variables rather than direct parameters.
        """
        host, port, base_url = web_server

        search_data = {
            "query": "transformer neural network",
            "use_embeddings": True,  # This triggers EmbeddingsManager initialization
            "limit": 5,
        }

        response = requests.post(f"{base_url}/api/search", json=search_data, timeout=10)

        # Should not crash with TypeError about unexpected keyword argument
        # The response might be an error if embeddings aren't set up, but it should be
        # a proper error response (400 or 500), not a crash
        assert response.status_code in [200, 400, 500]

        # If we get a response, it should be valid JSON
        data = response.json()
        assert isinstance(data, dict)

        # If successful (200), should have papers list
        if response.status_code == 200:
            assert "papers" in data
            assert isinstance(data["papers"], list)

        # If error (400/500), should have error message
        elif response.status_code in [400, 500]:
            assert "error" in data
            # The error should NOT be about unexpected keyword argument
            assert "unexpected keyword argument" not in data.get("error", "").lower()

    def test_keyword_search_end_to_end(self, web_server):
        """
        End-to-end test for keyword search functionality.

        Tests the complete workflow:
        1. Make search request with keyword search
        2. Verify response format
        3. Check that results contain the query term
        4. Verify all required fields are present
        """
        host, port, base_url = web_server

        search_data = {
            "query": "attention",
            "use_embeddings": False,
            "limit": 5,
        }

        response = requests.post(f"{base_url}/api/search", json=search_data, timeout=5)

        # Should succeed
        assert response.status_code == 200

        data = response.json()

        # Check response structure
        assert "papers" in data
        assert "count" in data
        assert "query" in data
        assert "use_embeddings" in data

        # Check response values
        assert data["query"] == "attention"
        assert data["use_embeddings"] is False
        assert isinstance(data["papers"], list)
        assert data["count"] == len(data["papers"])
        assert data["count"] <= 5

        # If we got results, check paper structure
        if data["papers"]:
            paper = data["papers"][0]
            required_fields = ["uid", "title", "abstract"]
            for field in required_fields:
                assert field in paper, f"Missing required field: {field}"

            # At least one paper should contain "attention" in title or abstract
            found_match = False
            for p in data["papers"]:
                if "attention" in p.get("title", "").lower() or "attention" in p.get("abstract", "").lower():
                    found_match = True
                    break
            assert found_match, "No papers matched the search query"

    def test_semantic_search_end_to_end(self, web_server):
        """
        End-to-end test for semantic search functionality.

        Tests the complete semantic search workflow:
        1. Make search request with embeddings enabled
        2. Verify response format
        3. Check that similarity scores are present
        4. Verify results are semantically relevant
        """
        host, port, base_url = web_server

        search_data = {
            "query": "deep learning neural networks",
            "use_embeddings": True,
            "limit": 3,
        }

        response = requests.post(f"{base_url}/api/search", json=search_data, timeout=10)

        # Should succeed (or fail gracefully if embeddings not available)
        assert response.status_code in [200, 500]

        data = response.json()

        if response.status_code == 200:
            # Check response structure
            assert "papers" in data
            assert "count" in data
            assert "query" in data
            assert "use_embeddings" in data

            # Check response values
            assert data["query"] == "deep learning neural networks"
            assert data["use_embeddings"] is True
            assert isinstance(data["papers"], list)
            assert data["count"] == len(data["papers"])
            assert data["count"] <= 3

            # Semantic search MUST return results - if it returns 0, it's a bug
            assert (
                data["count"] > 0
            ), "Semantic search must return at least 1 result. If this fails, check that embeddings are created and the collection is not empty."
            assert len(data["papers"]) > 0, "Semantic search must return at least 1 paper"

            # Check that results have similarity scores
            for paper in data["papers"]:
                assert "uid" in paper
                assert "title" in paper
                assert "abstract" in paper
                # Semantic search results should include similarity score
                assert "similarity" in paper, "Semantic search results should include similarity score"
                assert isinstance(paper["similarity"], (int, float))
                assert 0 <= paper["similarity"] <= 1, "Similarity should be between 0 and 1"
        else:
            # If failed, should have clear error message
            assert "error" in data

    def test_search_comparison_keyword_vs_semantic(self, web_server):
        """
        Compare keyword search vs semantic search results.

        Tests that both search methods work and can return different results
        for the same query.
        """
        host, port, base_url = web_server

        query = "transformer architecture"

        # Keyword search
        keyword_data = {
            "query": query,
            "use_embeddings": False,
            "limit": 5,
        }
        keyword_response = requests.post(f"{base_url}/api/search", json=keyword_data, timeout=5)
        assert keyword_response.status_code == 200
        keyword_results = keyword_response.json()

        # Semantic search
        semantic_data = {
            "query": query,
            "use_embeddings": True,
            "limit": 5,
        }
        semantic_response = requests.post(f"{base_url}/api/search", json=semantic_data, timeout=10)

        # Both should work
        assert keyword_response.status_code == 200

        # Check keyword results
        assert "papers" in keyword_results
        assert "use_embeddings" in keyword_results
        assert keyword_results["use_embeddings"] is False

        # Semantic search should work or fail gracefully
        if semantic_response.status_code == 200:
            semantic_results = semantic_response.json()
            assert "papers" in semantic_results
            assert "use_embeddings" in semantic_results
            assert semantic_results["use_embeddings"] is True

            # Semantic search MUST return results
            assert len(semantic_results["papers"]) > 0, "Semantic search must return at least 1 result"

            # Semantic results should have similarity scores
            assert "similarity" in semantic_results["papers"][0]

            # Keyword results should NOT have similarity scores
            if keyword_results["papers"]:
                assert "similarity" not in keyword_results["papers"][0]

    def test_empty_search_results(self, web_server):
        """
        Test handling of searches that return no results.
        """
        host, port, base_url = web_server

        search_data = {
            "query": "xyzabc123nonexistent",
            "use_embeddings": False,
            "limit": 10,
        }

        response = requests.post(f"{base_url}/api/search", json=search_data, timeout=5)

        assert response.status_code == 200

        data = response.json()
        assert "papers" in data
        assert data["count"] == 0
        assert len(data["papers"]) == 0

    def test_content_type_headers(self, web_server):
        """Test that appropriate content-type headers are set."""
        host, port, base_url = web_server

        # HTML page
        response = requests.get(base_url, timeout=5)
        assert "text/html" in response.headers.get("Content-Type", "")

        # JSON API
        response = requests.get(f"{base_url}/api/stats", timeout=5)
        assert "application/json" in response.headers.get("Content-Type", "")

        # JavaScript
        response = requests.get(f"{base_url}/static/app.js", timeout=5)
        assert any(ct in response.headers.get("Content-Type", "") for ct in ["javascript", "text/plain"])

    def test_cors_headers(self, web_server):
        """Test that CORS headers are present."""
        host, port, base_url = web_server

        response = requests.get(f"{base_url}/api/stats", timeout=5)

        # Should have CORS headers (Flask-CORS is enabled)
        assert "Access-Control-Allow-Origin" in response.headers


class TestWebUICommand:
    """Test the CLI command for starting the web UI."""

    def test_web_ui_command_exists(self):
        """Test that the web-ui command is registered."""
        from abstracts_explorer.cli import main

        # Test that calling with web-ui shows help or runs
        # We can't actually run it here, but we can check it's registered

        # This should not raise an error
        assert callable(main)

    def test_web_ui_import(self):
        """Test that web_ui module can be imported."""
        try:
            from abstracts_explorer.web_ui import run_server, app

            assert callable(run_server)
            assert app is not None
        except ImportError as e:
            pytest.skip(f"Web UI not available: {e}")


class TestWebUISemanticSearchWithResults:
    """Test semantic search with actual results to cover transformation code."""

    def test_semantic_search_with_multiple_results(self, web_server):
        """Test semantic search returns multiple results with similarity scores."""
        host, port, base_url = web_server

        # Use a query that should match test papers
        search_data = {
            "query": "deep learning transformers attention mechanism",
            "use_embeddings": True,
            "limit": 3,
        }

        response = requests.post(f"{base_url}/api/search", json=search_data, timeout=10)

        # May succeed or fail depending on embeddings setup
        if response.status_code == 200:
            data = response.json()

            # Check structure
            assert "papers" in data
            assert "count" in data
            assert "use_embeddings" in data
            assert data["use_embeddings"] is True

            # If we got results, verify they have similarity scores
            if data["papers"]:
                for paper in data["papers"]:
                    assert "uid" in paper
                    assert "title" in paper
                    # Semantic search should add similarity
                    assert "similarity" in paper
                    assert isinstance(paper["similarity"], (int, float))
                    assert 0 <= paper["similarity"] <= 1


class TestWebUIChatEndpointFull:
    """
    Test chat endpoint with full functionality.

    These integration tests require LM Studio to verify end-to-end chat functionality.
    For unit testing without LM Studio, see:
    - TestWebUIErrorHandlingPaths.test_chat_with_empty_message
    - TestWebUIErrorHandlingPaths.test_chat_without_message
    - Tests in test_rag.py for the underlying RAG functionality
    """

    @requires_lm_studio
    def test_chat_with_valid_message_and_response(self, web_server):
        """
        Test chat endpoint returns valid response.

        This integration test verifies the complete chat workflow with real API.
        For unit testing without LM Studio, see tests in test_rag.py.
        """
        host, port, base_url = web_server

        chat_data = {
            "message": "What are the main contributions of transformer models?",
            "n_papers": 3,
        }

        response = requests.post(f"{base_url}/api/chat", json=chat_data, timeout=60)

        # Chat requires LM Studio, so it may fail gracefully
        if response.status_code == 200:
            data = response.json()

            # Check response structure
            assert isinstance(data, dict)
            assert "response" in data or "message" in data

            # If successful, response should be a string
            if "response" in data:
                assert isinstance(data["response"], (str, dict))
        else:
            # Should return proper error
            assert response.status_code in [400, 500]
            data = response.json()
            assert "error" in data

    @requires_lm_studio
    def test_chat_with_reset_flag(self, web_server):
        """
        Test chat endpoint with reset flag.

        This integration test verifies conversation reset with real API.
        For unit testing without LM Studio, see TestRAGChatConversation.test_reset_conversation in test_rag.py.
        """
        host, port, base_url = web_server

        # First message
        chat_data1 = {
            "message": "What is a transformer?",
            "n_papers": 2,
        }

        response1 = requests.post(f"{base_url}/api/chat", json=chat_data1, timeout=60)

        # Second message with reset
        chat_data2 = {
            "message": "What is BERT?",
            "n_papers": 2,
            "reset": True,
        }

        response2 = requests.post(f"{base_url}/api/chat", json=chat_data2, timeout=60)

        # Both should work or fail gracefully
        assert response1.status_code in [200, 400, 500]
        assert response2.status_code in [200, 400, 500]

        # If they succeed, both should have valid structure
        if response1.status_code == 200:
            data1 = response1.json()
            assert isinstance(data1, dict)

        if response2.status_code == 200:
            data2 = response2.json()
            assert isinstance(data2, dict)

    @requires_lm_studio
    def test_chat_with_custom_n_papers(self, web_server):
        """
        Test chat endpoint with custom n_papers parameter.

        This integration test verifies custom paper count with real API.
        For unit testing without LM Studio, see TestRAGChatQuery.test_query_with_n_results in test_rag.py.
        """
        host, port, base_url = web_server

        chat_data = {
            "message": "Explain attention mechanisms",
            "n_papers": 5,  # Custom number
        }

        response = requests.post(f"{base_url}/api/chat", json=chat_data, timeout=60)

        # Should handle the parameter
        assert response.status_code in [200, 400, 500]

        data = response.json()
        assert isinstance(data, dict)

    def test_chat_reset_endpoint(self, web_server):
        """Test the dedicated chat reset endpoint."""
        host, port, base_url = web_server

        response = requests.post(f"{base_url}/api/chat/reset", timeout=10)

        # Should work or fail gracefully
        assert response.status_code in [200, 500]

        data = response.json()
        assert isinstance(data, dict)


class TestWebUIPaperEndpointDetails:
    """Test paper detail endpoint edge cases."""

    def test_get_paper_with_authors(self, web_server):
        """Test getting paper details includes authors."""
        host, port, base_url = web_server

        # First search for a paper
        search_data = {"query": "transformer", "use_embeddings": False, "limit": 1}
        search_response = requests.post(f"{base_url}/api/search", json=search_data, timeout=5)

        if search_response.status_code == 200:
            papers = search_response.json()["papers"]

            if papers:
                paper_uid = papers[0]["uid"]

                # Get full paper details
                response = requests.get(f"{base_url}/api/paper/{paper_uid}", timeout=5)

                assert response.status_code == 200
                paper = response.json()

                # Should have authors field
                assert "authors" in paper
                assert isinstance(paper["authors"], list)

    def test_get_paper_exception_handling(self, web_server):
        """Test paper endpoint handles database errors."""
        host, port, base_url = web_server

        # Try to get a paper with invalid ID (very large number)
        response = requests.get(f"{base_url}/api/paper/999999999", timeout=5)

        # Should return 404 or 500
        assert response.status_code in [404, 500]

        data = response.json()
        assert "error" in data


class TestWebUIErrorHandlingPaths:
    """Test error handling paths in web UI."""

    def test_search_with_missing_query_parameter(self, web_server):
        """Test search without query parameter."""
        host, port, base_url = web_server

        # Missing query
        search_data = {"use_embeddings": False, "limit": 10}

        response = requests.post(f"{base_url}/api/search", json=search_data, timeout=5)

        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    def test_search_with_empty_query(self, web_server):
        """Test search with empty query string."""
        host, port, base_url = web_server

        search_data = {"query": "", "use_embeddings": False, "limit": 10}

        response = requests.post(f"{base_url}/api/search", json=search_data, timeout=5)

        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    def test_chat_with_empty_message(self, web_server):
        """Test chat with empty message."""
        host, port, base_url = web_server

        chat_data = {"message": ""}

        response = requests.post(f"{base_url}/api/chat", json=chat_data, timeout=5)

        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    def test_chat_without_message(self, web_server):
        """Test chat without message parameter."""
        host, port, base_url = web_server

        chat_data = {}

        response = requests.post(f"{base_url}/api/chat", json=chat_data, timeout=5)

        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    def test_search_semantic_exception(self, web_server):
        """Test semantic search handles exceptions gracefully."""
        host, port, base_url = web_server

        # Try semantic search - it may fail if embeddings not set up
        search_data = {
            "query": "test query for exception",
            "use_embeddings": True,
            "limit": 5,
        }

        response = requests.post(f"{base_url}/api/search", json=search_data, timeout=10)

        # Should return valid response (200) or error (500)
        assert response.status_code in [200, 500]

        data = response.json()
        assert isinstance(data, dict)

        # If error, should have error message
        if response.status_code == 500:
            assert "error" in data

    def test_clusters_compute_endpoint(self, web_server):
        """Test that clusters compute endpoint returns 404 when no pre-computed data exists."""
        host, port, base_url = web_server

        # Request pre-computed clustering data (no clustering config needed)
        response = requests.post(f"{base_url}/api/clusters/compute", json={}, timeout=30)

        # Should return 404 since no pre-computed data exists
        assert response.status_code == 404

        data = response.json()
        assert isinstance(data, dict)
        assert "error" in data

    def test_clusters_cached_endpoint_not_found(self, web_server):
        """Test that cached clusters endpoint returns 404 when no cache exists."""
        host, port, base_url = web_server

        response = requests.get(f"{base_url}/api/clusters/cached", timeout=10)

        # Should return 404 when no cached file exists
        assert response.status_code == 404

        data = response.json()
        assert "error" in data

    def test_clusters_compute_with_conference_year_filter(self, web_server):
        """Test that clusters compute endpoint accepts conference/year filters."""
        host, port, base_url = web_server

        # Request with conference/year filter - no pre-computed data so expect 404
        cluster_data = {
            "conferences": ["NeurIPS"],
            "years": [2024],
        }

        response = requests.post(f"{base_url}/api/clusters/compute", json=cluster_data, timeout=30)

        # Should return 404 since no pre-computed data exists for this combo
        assert response.status_code == 404

        data = response.json()
        assert isinstance(data, dict)
        assert "error" in data

    def test_custom_cluster_search_with_embeddings(self, web_server):
        """Test custom cluster search endpoint with real embeddings."""
        host, port, base_url = web_server

        # Test the custom cluster search with a query
        search_data = {"query": "machine learning", "distance": 150.0}

        response = requests.post(f"{base_url}/api/clusters/search", json=search_data, timeout=30)

        # Should return 200 OK
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, dict)

        # Verify response structure
        assert "query" in data
        assert "distance" in data
        assert "papers" in data
        assert "count" in data
        assert "query_embedding" in data

        # Verify values
        assert data["query"] == "machine learning"
        assert data["distance"] == 150.0
        assert isinstance(data["papers"], list)
        assert isinstance(data["count"], int)
        assert data["count"] == len(data["papers"])

        # All papers should have a distance field
        for paper in data["papers"]:
            assert "distance" in paper
            assert isinstance(paper["distance"], (int, float))
            assert paper["distance"] <= 150.0

        # Papers should be sorted by distance
        if len(data["papers"]) > 1:
            for i in range(len(data["papers"]) - 1):
                assert data["papers"][i]["distance"] <= data["papers"][i + 1]["distance"]


class TestPaperCardDisplayFields:
    """
    Integration tests verifying paper cards display correctly everywhere.

    These tests ensure that every API endpoint which returns paper data
    includes the fields required for correct paper card rendering:
    - uid: required for star ratings and the paper details modal
    - title: the paper title
    - authors: list of author names (NOT a semicolon-separated string)
    - conference: shown as an indigo badge on every card
    - abstract: shown in the card body
    """

    def _get_any_paper_uid(self, base_url: str) -> str | None:
        """Helper to get a paper UID via keyword search."""
        response = requests.post(
            f"{base_url}/api/search",
            json={"query": "attention", "use_embeddings": False, "limit": 1},
            timeout=5,
        )
        if response.status_code == 200:
            papers = response.json().get("papers", [])
            if papers:
                return papers[0].get("uid")
        return None

    def test_keyword_search_paper_card_fields(self, web_server):
        """
        Verify keyword search returns all fields needed for paper card display.

        The paper card (formatPaperCard in paper-card.js) requires: uid, title,
        authors (as a list), conference, and abstract.
        """
        host, port, base_url = web_server

        response = requests.post(
            f"{base_url}/api/search",
            json={"query": "transformer", "use_embeddings": False, "limit": 5},
            timeout=5,
        )
        assert response.status_code == 200
        data = response.json()
        assert "papers" in data

        for paper in data["papers"]:
            # uid: required for star ratings and the detail modal onclick handler
            assert "uid" in paper, "Paper card requires 'uid' for star ratings and detail modal"

            # title: displayed as the card heading
            assert "title" in paper, "Paper card requires 'title'"

            # authors: must be a list; a string would make formatPaperCard throw TypeError
            assert "authors" in paper, "Paper card requires 'authors'"
            assert isinstance(
                paper["authors"], list
            ), f"authors must be a list for paper card display; got {type(paper['authors'])}"

            # conference: shown as indigo badge on card
            assert "conference" in paper, "Paper card requires 'conference' for conference badge"

    def test_semantic_search_paper_card_fields(self, web_server):
        """
        Verify semantic search returns all fields needed for paper card display.

        Semantic search results additionally include a similarity score.
        """
        host, port, base_url = web_server

        response = requests.post(
            f"{base_url}/api/search",
            json={"query": "deep learning", "use_embeddings": True, "limit": 3},
            timeout=10,
        )
        # Semantic search may fail if embeddings backend unavailable; accept graceful failure
        assert response.status_code in [200, 500]

        if response.status_code == 200:
            data = response.json()
            for paper in data["papers"]:
                assert "uid" in paper, "Paper card requires 'uid'"
                assert "title" in paper, "Paper card requires 'title'"
                assert "authors" in paper, "Paper card requires 'authors'"
                assert isinstance(
                    paper["authors"], list
                ), f"authors must be a list for paper card display; got {type(paper['authors'])}"
                assert "conference" in paper, "Paper card requires 'conference' for conference badge"

    def test_paper_detail_endpoint_card_fields(self, web_server):
        """
        Verify the paper detail endpoint returns all fields needed for the paper details modal.

        The showPaperDetails() function renders a modal with conference, authors, abstract, etc.
        """
        host, port, base_url = web_server

        paper_uid = self._get_any_paper_uid(base_url)
        if paper_uid is None:
            pytest.skip("No papers available in test database")

        response = requests.get(f"{base_url}/api/paper/{paper_uid}", timeout=5)
        assert response.status_code == 200

        paper = response.json()

        # Fields required by the paper details modal (showPaperDetails)
        assert "uid" in paper, "Paper details modal requires 'uid'"
        assert "title" in paper, "Paper details modal requires 'title'"

        # authors must be a list for the modal to join them correctly
        assert "authors" in paper, "Paper details modal requires 'authors'"
        assert isinstance(
            paper["authors"], list
        ), f"authors must be a list for paper details modal; got {type(paper['authors'])}"

        # conference shown as indigo badge in modal header
        assert "conference" in paper, "Paper details modal requires 'conference' for conference badge"

        # abstract shown in modal body
        assert "abstract" in paper, "Paper details modal requires 'abstract'"

    def test_paper_detail_conference_has_correct_value(self, web_server):
        """
        Verify the conference field in paper details has a non-empty string value.

        The conference badge is only rendered when paper.conference is truthy.
        """
        host, port, base_url = web_server

        paper_uid = self._get_any_paper_uid(base_url)
        if paper_uid is None:
            pytest.skip("No papers available in test database")

        response = requests.get(f"{base_url}/api/paper/{paper_uid}", timeout=5)
        assert response.status_code == 200

        paper = response.json()
        # All test papers are from NeurIPS conference
        assert paper["conference"] == "NeurIPS", f"Expected conference 'NeurIPS', got '{paper['conference']}'"

    def test_paper_detail_authors_is_list_of_strings(self, web_server):
        """
        Verify authors field is a list of strings (not a semicolon-separated string).

        The paper card renders authors with paper.authors.join(', '). If authors is a
        string, it throws a TypeError and the card fails to render.
        """
        host, port, base_url = web_server

        paper_uid = self._get_any_paper_uid(base_url)
        if paper_uid is None:
            pytest.skip("No papers available in test database")

        response = requests.get(f"{base_url}/api/paper/{paper_uid}", timeout=5)
        assert response.status_code == 200

        paper = response.json()
        authors = paper["authors"]

        assert isinstance(authors, list), f"authors must be a list, got {type(authors)}: {authors!r}"
        for author in authors:
            assert isinstance(author, str), f"Each author must be a string, got {type(author)}: {author!r}"
            # Authors should be trimmed of whitespace
            assert author == author.strip(), f"Author '{author}' has leading/trailing whitespace"

    def test_batch_papers_endpoint_card_fields(self, web_server):
        """
        Verify the batch papers endpoint returns all fields needed for paper card display.

        The /api/papers/batch endpoint is used by the interesting papers tab and the
        conference/year filter count update.
        """
        host, port, base_url = web_server

        paper_uid = self._get_any_paper_uid(base_url)
        if paper_uid is None:
            pytest.skip("No papers available in test database")

        response = requests.post(
            f"{base_url}/api/papers/batch",
            json={"paper_ids": [paper_uid]},
            timeout=5,
        )
        assert response.status_code == 200
        data = response.json()
        assert "papers" in data
        assert len(data["papers"]) > 0

        paper = data["papers"][0]
        assert "uid" in paper, "Batch endpoint must return 'uid' for paper cards"
        assert "title" in paper, "Batch endpoint must return 'title' for paper cards"
        assert "authors" in paper, "Batch endpoint must return 'authors' for paper cards"
        assert isinstance(
            paper["authors"], list
        ), f"authors must be a list for paper cards; got {type(paper['authors'])}"
        assert "conference" in paper, "Batch endpoint must return 'conference' for conference badge"

    def test_keyword_search_authors_are_not_semicolon_strings(self, web_server):
        """
        Verify that keyword search results never return authors as a raw semicolon-separated string.

        Returning a semicolon-separated string instead of a list would cause formatPaperCard
        to throw TypeError: 'authors must be an array'.
        """
        host, port, base_url = web_server

        response = requests.post(
            f"{base_url}/api/search",
            json={"query": "attention", "use_embeddings": False, "limit": 10},
            timeout=5,
        )
        assert response.status_code == 200
        data = response.json()

        for paper in data["papers"]:
            authors = paper.get("authors", [])
            assert isinstance(
                authors, list
            ), f"Paper '{paper.get('title')}' has authors as {type(authors).__name__} instead of list: {authors!r}"
            # Ensure no author string contains unprocessed semicolons from DB
            for author in authors:
                assert (
                    ";" not in author
                ), f"Author '{author}' still contains semicolon; authors were not split correctly"


class TestTopicEvolutionIntegration:
    """
    Integration tests for the /api/topic-evolution endpoint.

    These tests exercise the endpoint through the real web server and ChromaDB so
    that the full request→MCP tool→embeddings→database→response chain is verified.
    The OpenAI client is still mocked by the web_server fixture (no real LLM needed).
    """

    def test_topic_evolution_returns_valid_structure(self, web_server):
        """
        Test that /api/topic-evolution returns the expected JSON structure.

        All required top-level keys must be present and have the correct types.
        """
        host, port, base_url = web_server

        response = requests.post(
            f"{base_url}/api/topic-evolution",
            json={"topic_keywords": "transformers", "conferences": ["NeurIPS"]},
            timeout=15,
        )

        assert response.status_code == 200
        data = response.json()

        # Top-level keys required by the frontend
        assert "topic" in data
        assert "conferences" in data
        assert "conference_data" in data
        assert data["topic"] == "transformers"
        assert data["conferences"] == ["NeurIPS"]
        assert isinstance(data["conference_data"], dict)

    def test_topic_evolution_conference_data_structure(self, web_server):
        """
        Test that the per-conference data contains the expected sub-keys.

        The frontend's fetchAndDisplayTopicEvolution() reads year_relative to
        build the line chart, so that key is essential.
        """
        host, port, base_url = web_server

        response = requests.post(
            f"{base_url}/api/topic-evolution",
            json={"topic_keywords": "attention mechanisms", "conferences": ["NeurIPS"]},
            timeout=15,
        )

        assert response.status_code == 200
        data = response.json()

        assert "NeurIPS" in data["conference_data"]
        neurips_data = data["conference_data"]["NeurIPS"]

        # Keys read by the JS frontend
        assert "year_counts" in neurips_data
        assert "year_relative" in neurips_data
        assert "year_totals" in neurips_data

        # year_relative values must be floats (percentages)
        for year_key, value in neurips_data["year_relative"].items():
            assert isinstance(value, (int, float)), f"year_relative[{year_key}] is not a number"
            assert value >= 0, f"year_relative[{year_key}] is negative"

    def test_topic_evolution_missing_topic_keywords_returns_400(self, web_server):
        """Test that omitting topic_keywords returns HTTP 400."""
        host, port, base_url = web_server

        response = requests.post(
            f"{base_url}/api/topic-evolution",
            json={"conferences": ["NeurIPS"]},
            timeout=10,
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    def test_topic_evolution_empty_body_returns_400(self, web_server):
        """Test that an empty request body returns HTTP 400."""
        host, port, base_url = web_server

        response = requests.post(
            f"{base_url}/api/topic-evolution",
            json={},
            timeout=10,
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    def test_topic_evolution_no_conferences_returns_error_in_body(self, web_server):
        """
        Test that omitting conferences causes the MCP tool to return an error.

        The endpoint returns HTTP 200 but the response body contains an "error"
        key because the MCP tool itself enforces the conference requirement.
        """
        host, port, base_url = web_server

        response = requests.post(
            f"{base_url}/api/topic-evolution",
            json={"topic_keywords": "deep learning"},
            timeout=10,
        )

        # The MCP tool returns a JSON error rather than raising — endpoint stays 200
        assert response.status_code == 200
        data = response.json()
        assert "error" in data

    def test_topic_evolution_custom_distance_threshold(self, web_server):
        """
        Test that a custom distance_threshold is forwarded and respected.

        A very small threshold (0.0) should still return a valid response
        structure (just with zero-count years).
        """
        host, port, base_url = web_server

        response = requests.post(
            f"{base_url}/api/topic-evolution",
            json={
                "topic_keywords": "transformers",
                "conferences": ["NeurIPS"],
                "distance_threshold": 0.0,
            },
            timeout=15,
        )

        assert response.status_code == 200
        data = response.json()
        assert "conference_data" in data
        # distance_threshold should be echoed back
        assert data.get("distance_threshold") == 0.0

    def test_topic_evolution_covers_test_paper_years(self, web_server):
        """
        Test that the response includes all years present in the test database.

        The web_server fixture adds papers for years 2016, 2017, and 2019 in the
        NeurIPS conference, so the response must contain entries for those years.
        """
        host, port, base_url = web_server

        response = requests.post(
            f"{base_url}/api/topic-evolution",
            json={"topic_keywords": "neural networks", "conferences": ["NeurIPS"]},
            timeout=15,
        )

        assert response.status_code == 200
        data = response.json()

        assert "NeurIPS" in data["conference_data"]
        year_counts = data["conference_data"]["NeurIPS"]["year_counts"]

        # The three test papers span these years
        expected_years = {"2016", "2017", "2019"}
        assert expected_years.issubset(
            set(str(k) for k in year_counts.keys())
        ), f"Expected years {expected_years} not all present in {set(year_counts.keys())}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

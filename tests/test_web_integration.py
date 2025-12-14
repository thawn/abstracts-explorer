"""
Integration tests for the web UI.

These tests verify that the web UI server can start, serve pages,
and handle API requests correctly.
"""

import pytest
import sys
import time
import requests
import threading
from pathlib import Path
from multiprocessing import Process

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neurips_abstracts.database import DatabaseManager
from tests.test_helpers import requires_lm_studio, find_free_port

# Helper functions imported from test_helpers:
# - check_lm_studio_available(): Check if LM Studio is running
# - requires_lm_studio: Skip marker for tests requiring LM Studio
# - find_free_port(): Find a free port for testing


@pytest.fixture(scope="module")
def test_database(tmp_path_factory):
    """
    Create a test database with sample data.

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
    db_path = tmp_dir / "test_web_integration.db"

    # Create database and add test data
    db = DatabaseManager(str(db_path))

    with db:
        db.create_tables()

        cursor = db.connection.cursor()

        # Add test papers
        papers_data = [
            (
                "test1",
                "Attention is All You Need",
                "We propose the Transformer, a model architecture based solely on attention mechanisms.",
                "Accept (oral)",
            ),
            (
                "test2",
                "BERT: Pre-training of Deep Bidirectional Transformers",
                "We introduce BERT, which stands for Bidirectional Encoder Representations from Transformers.",
                "Accept (poster)",
            ),
            (
                "test3",
                "Deep Residual Learning for Image Recognition",
                "We present a residual learning framework to ease the training of networks.",
                "Accept (oral)",
            ),
        ]

        paper_ids = []
        for uid, name, abstract, decision in papers_data:
            cursor.execute(
                """
                INSERT INTO papers (uid, name, abstract, decision)
                VALUES (?, ?, ?, ?)
                """,
                (uid, name, abstract, decision),
            )
            paper_ids.append(cursor.lastrowid)

        # Add authors
        authors_data = [
            ("Ashish Vaswani", "Google Brain"),
            ("Jacob Devlin", "Google AI"),
            ("Kaiming He", "Microsoft Research"),
        ]

        for i, (fullname, institution) in enumerate(authors_data):
            cursor.execute("INSERT INTO authors (fullname, institution) VALUES (?, ?)", (fullname, institution))

        db.connection.commit()

    return db_path


def start_web_server(db_path, port):
    """
    Start the web server in a separate process.

    Parameters
    ----------
    db_path : Path
        Path to the test database
    port : int
        Port to run the server on
    """
    import os

    # Set environment variable for database path
    os.environ["PAPER_DB_PATH"] = str(db_path)

    # Import after setting env var
    from neurips_abstracts.web_ui import run_server
    from neurips_abstracts.config import get_config

    # Force reload config to pick up the environment variable
    # This is important on Linux where multiprocessing uses fork()
    get_config(reload=True)

    # Run server (config will be loaded lazily on first request)
    run_server(host="127.0.0.1", port=port, debug=False)


@pytest.fixture(scope="module")
def web_server(test_database):
    """
    Start the web server for testing.

    Parameters
    ----------
    test_database : Path
        Path to the test database

    Yields
    ------
    tuple
        (host, port, base_url)
    """
    # Check if Flask is installed
    try:
        import importlib.util

        if importlib.util.find_spec("flask") is None:
            pytest.skip("Flask not installed - web UI tests require 'pip install neurips-abstracts[web]'")
    except ImportError:
        pytest.skip("Flask not installed - web UI tests require 'pip install neurips-abstracts[web]'")

    port = find_free_port()
    host = "127.0.0.1"
    base_url = f"http://{host}:{port}"

    # Set environment variable before starting server
    import os
    from pathlib import Path

    # Ensure the database path is absolute
    abs_db_path = str(Path(test_database).absolute())
    os.environ["PAPER_DB_PATH"] = abs_db_path

    # Start server in a separate process
    server_process = Process(target=start_web_server, args=(abs_db_path, port), daemon=True)
    server_process.start()

    # Wait for server to start
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(base_url, timeout=1)
            if response.status_code in [200, 404]:
                break
        except requests.exceptions.RequestException:
            if i == max_retries - 1:
                server_process.terminate()
                pytest.fail("Web server failed to start")
            time.sleep(0.5)

    yield (host, port, base_url)

    # Cleanup
    server_process.terminate()
    server_process.join(timeout=5)


class TestWebUIIntegration:
    """Integration tests for the web UI."""

    def test_server_starts(self, web_server):
        """Test that the web server starts and responds."""
        host, port, base_url = web_server

        response = requests.get(base_url, timeout=5)
        assert response.status_code == 200
        assert b"NeurIPS Abstracts Explorer" in response.content

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

        # First, search for a paper to get its ID
        search_data = {"query": "attention", "use_embeddings": False, "limit": 1}

        response = requests.post(f"{base_url}/api/search", json=search_data, timeout=5)

        assert response.status_code == 200
        papers = response.json()["papers"]

        if papers:
            paper_id = papers[0]["id"]

            # Get paper details
            response = requests.get(f"{base_url}/api/paper/{paper_id}", timeout=5)
            assert response.status_code == 200

            paper = response.json()
            assert "id" in paper
            assert "name" in paper
            assert paper["id"] == paper_id

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

    def test_404_handling(self, web_server):
        """Test 404 error handling."""
        host, port, base_url = web_server

        response = requests.get(f"{base_url}/nonexistent", timeout=5)
        assert response.status_code == 404

        data = response.json()
        assert "error" in data

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

        This test specifically checks for the bug where EmbeddingsManager was being
        initialized with incorrect parameters (db_path instead of chroma_path).

        Bug: EmbeddingsManager.__init__() got an unexpected keyword argument 'db_path'
        Fix: Use correct parameters: lm_studio_url, model_name, chroma_path, collection_name
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
            required_fields = ["id", "name", "abstract"]
            for field in required_fields:
                assert field in paper, f"Missing required field: {field}"

            # At least one paper should contain "attention" in name or abstract
            found_match = False
            for p in data["papers"]:
                if "attention" in p.get("name", "").lower() or "attention" in p.get("abstract", "").lower():
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

            # If we got results, check they have similarity scores
            if data["papers"]:
                for paper in data["papers"]:
                    assert "id" in paper
                    assert "name" in paper
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

            # Semantic results should have similarity scores
            if semantic_results["papers"]:
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
        from neurips_abstracts.cli import main

        # Test that calling with web-ui shows help or runs
        # We can't actually run it here, but we can check it's registered

        # This should not raise an error
        assert callable(main)

    def test_web_ui_import(self):
        """Test that web_ui module can be imported."""
        try:
            from neurips_abstracts.web_ui import run_server, app

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
                    assert "id" in paper
                    assert "name" in paper or "title" in paper
                    # Semantic search should add similarity
                    assert "similarity" in paper
                    assert isinstance(paper["similarity"], (int, float))
                    assert 0 <= paper["similarity"] <= 1


class TestWebUIChatEndpointFull:
    """Test chat endpoint with full functionality."""

    @requires_lm_studio
    def test_chat_with_valid_message_and_response(self, web_server):
        """Test chat endpoint returns valid response."""
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
        """Test chat endpoint with reset flag."""
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
        """Test chat endpoint with custom n_papers parameter."""
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
                paper_id = papers[0]["id"]

                # Get full paper details
                response = requests.get(f"{base_url}/api/paper/{paper_id}", timeout=5)

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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

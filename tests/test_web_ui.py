"""
Tests for web_ui/app.py module.

This module contains unit tests for the web UI application including:
- Flask app configuration and initialization
- API endpoints (search, chat, stats, paper details)
- Semantic search functionality
- Error handling and edge cases
- Response formatting
"""

import json
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from abstracts_explorer.web_ui import app as flask_app
from abstracts_explorer.database import DatabaseManager


# ============================================================
# Tests from test_web.py
# ============================================================

@pytest.fixture
def app():
    """
    Create Flask app for testing.

    Returns
    -------
    Flask
        Flask application instance
    """
    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture
def client(app):
    """
    Create test client.

    Parameters
    ----------
    app : Flask
        Flask application

    Returns
    -------
    FlaskClient
        Test client for making requests
    """
    return app.test_client()


# Note: Flask app fixtures (app, client) are kept in this file as they're specific
# to web testing and not shared with other test modules.


@pytest.fixture
def test_db(tmp_path, web_test_papers):
    """
    Create a test database with sample papers for web testing.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory path
    web_test_papers : list
        List of test papers from shared fixture

    Returns
    -------
    DatabaseManager
        Database manager with test data

    Notes
    -----
    Uses the shared web_test_papers fixture from conftest.py to ensure
    consistency across web-related tests.
    """
    db_path = tmp_path / "test.db"
    db = DatabaseManager(str(db_path))

    with db:
        db.create_tables()
        db.add_papers(web_test_papers)

    return db


class TestWebInterface:
    """Test web interface endpoints."""

    def test_index_route(self, client):
        """Test that the main page loads."""
        response = client.get("/")
        assert response.status_code == 200
        assert b"Abstracts Explorer" in response.data

    def test_stats_endpoint_no_db(self, client):
        """Test stats endpoint when database doesn't exist."""
        response = client.get("/api/stats")
        # Should return error if DB doesn't exist
        assert response.status_code in [200, 500]

    def test_filters_endpoint_no_db(self, client):
        """Test filters endpoint when database doesn't exist."""
        response = client.get("/api/filters")
        # Should return error if DB doesn't exist
        assert response.status_code in [200, 500]

    def test_years_endpoint_no_db(self, client):
        """Test years endpoint when database doesn't exist."""
        response = client.get("/api/years")
        # Should return error if DB doesn't exist
        assert response.status_code in [200, 500]


class TestSearchEndpoint:
    """Test the search endpoint specifically."""

    def test_search_without_query(self, client):
        """Test search endpoint with missing query."""
        response = client.post("/api/search", data=json.dumps({}), content_type="application/json")
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_search_with_empty_query(self, client):
        """Test search endpoint with empty query."""
        response = client.post("/api/search", data=json.dumps({"query": ""}), content_type="application/json")
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_search_keyword_parameters(self, client, monkeypatch):
        """
        Test that keyword search uses correct parameters.

        This test ensures the bug where 'title' and 'abstract' were
        passed to search_papers() doesn't happen again. It verifies
        that search_papers is called with 'keyword' parameter only.
        """
        from unittest.mock import MagicMock, patch
        import sys

        # Get the actual app module (not the Flask app object)
        app_module = sys.modules["abstracts_explorer.web_ui.app"]

        # Create a mock database
        mock_db = MagicMock()
        mock_papers = [{"id": 1, "name": "Test Paper", "abstract": "Test abstract", "uid": "test1"}]

        # Setup the mock to return our test data
        mock_db.search_papers.return_value = mock_papers

        # Patch the get_database function to return our mock
        with patch.object(app_module, "get_database", return_value=mock_db):
            # Make search request
            response = client.post(
                "/api/search",
                data=json.dumps({"query": "test", "use_embeddings": False, "limit": 10}),
                content_type="application/json",
            )

        # Verify search_papers was called with correct parameters
        # This is the key test - it should use 'keyword', not 'title' or 'abstract'
        mock_db.search_papers.assert_called_once()
        call_args = mock_db.search_papers.call_args

        # Check that it was called with 'keyword' parameter
        assert "keyword" in call_args.kwargs, "search_papers should be called with 'keyword' parameter"
        assert call_args.kwargs["keyword"] == "test"
        assert call_args.kwargs["limit"] == 10

        # Verify it was NOT called with invalid parameters
        assert "title" not in call_args.kwargs, "search_papers should NOT be called with 'title' parameter"
        assert "abstract" not in call_args.kwargs, "search_papers should NOT be called with 'abstract' parameter"

        # Check response
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "papers" in data
        assert "count" in data
        assert "query" in data
        assert data["query"] == "test"

    def test_search_with_limit(self, client):
        """Test that limit parameter is passed correctly."""
        from unittest.mock import MagicMock, patch
        import sys

        # Get the actual app module (not the Flask app object)
        app_module = sys.modules["abstracts_explorer.web_ui.app"]

        # Create a mock database
        mock_db = MagicMock()
        mock_papers = [{"id": i, "name": f"Paper {i}", "abstract": "About transformers"} for i in range(5)]

        mock_db.search_papers.return_value = mock_papers[:2]  # Return limited results

        # Patch the get_database function
        with patch.object(app_module, "get_database", return_value=mock_db):
            # Search with limit
            response = client.post(
                "/api/search",
                data=json.dumps({"query": "transformers", "use_embeddings": False, "limit": 2}),
                content_type="application/json",
            )

        # Verify limit was passed
        call_args = mock_db.search_papers.call_args
        assert call_args.kwargs["limit"] == 2

        assert response.status_code == 200
        data = json.loads(response.data)

        # Should respect the limit
        assert len(data["papers"]) <= 2

    def test_search_response_format(self, client):
        """Test that search response has correct format."""
        response = client.post(
            "/api/search",
            data=json.dumps({"query": "test", "use_embeddings": False, "limit": 10}),
            content_type="application/json",
        )

        # Even if it fails, check response structure
        data = json.loads(response.data)

        if response.status_code == 200:
            assert "papers" in data
            assert "count" in data
            assert "query" in data
            assert "use_embeddings" in data
            assert isinstance(data["papers"], list)
            assert isinstance(data["count"], int)
        else:
            assert "error" in data


class TestChatEndpoint:
    """Test the chat endpoint."""

    def test_chat_without_message(self, client):
        """Test chat endpoint with missing message."""
        response = client.post("/api/chat", data=json.dumps({}), content_type="application/json")
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_chat_with_empty_message(self, client):
        """Test chat endpoint with empty message."""
        response = client.post("/api/chat", data=json.dumps({"message": ""}), content_type="application/json")
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_chat_reset(self, client):
        """Test chat reset endpoint."""
        response = client.post("/api/chat/reset")
        # May fail if LM Studio not running, but should not crash
        assert response.status_code in [200, 500]


class TestPaperEndpoint:
    """Test the paper details endpoint."""

    def test_get_paper_invalid_id(self, client):
        """Test getting a paper with invalid ID."""
        response = client.get("/api/paper/99999")
        # Should return 404 or 500
        assert response.status_code in [404, 500]


class TestDatabaseSearchIntegration:
    """Test database search_papers method directly."""

    def test_search_papers_with_keyword(self, test_db):
        """Test search_papers with keyword parameter."""
        with test_db:
            results = test_db.search_papers(keyword="transformer")
            assert len(results) > 0
            # Should find papers with "transformer" in name or abstract

    def test_search_papers_limit(self, test_db):
        """Test search_papers limit parameter."""
        with test_db:
            results = test_db.search_papers(keyword="", limit=2)
            assert len(results) <= 2

    def test_search_papers_no_invalid_params(self, test_db):
        """
        Test that search_papers rejects invalid parameters.

        This ensures we don't accidentally pass invalid parameters
        that don't exist in the function signature.
        """
        with test_db:
            # These should raise TypeError if we try to pass invalid params
            with pytest.raises(TypeError):
                test_db.search_papers(title="test")

            with pytest.raises(TypeError):
                test_db.search_papers(abstract="test")

            with pytest.raises(TypeError):
                test_db.search_papers(invalid_param="test")

    def test_search_papers_valid_params_only(self, test_db):
        """Test that only valid parameters work."""
        with test_db:
            # These should work
            results = test_db.search_papers(keyword="attention")
            assert isinstance(results, list)

            results = test_db.search_papers(year=2017)
            assert isinstance(results, list)

            results = test_db.search_papers(limit=5)
            assert isinstance(results, list)


class TestErrorHandling:
    """Test error handling in web interface."""

    def test_404_handler(self, client):
        """Test 404 error handler."""
        response = client.get("/nonexistent")
        assert response.status_code == 404
        data = json.loads(response.data)
        assert "error" in data

    def test_search_json_error(self, client):
        """Test search with invalid JSON."""
        response = client.post("/api/search", data="invalid json", content_type="application/json")
        assert response.status_code in [400, 500]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ============================================================
# Tests from test_web_ui_unit.py
# ============================================================

class TestWebUISemanticSearchDetails:
    """Test semantic search result processing (lines 165-183)."""

    def test_semantic_search_transforms_chromadb_results(self):
        """Test that semantic search correctly transforms ChromaDB results to paper format."""
        # Import inside test to avoid import-time issues
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            # Mock the get_embeddings_manager and get_database functions
            with patch("abstracts_explorer.web_ui.app.get_embeddings_manager") as mock_get_em:
                with patch("abstracts_explorer.web_ui.app.get_database") as mock_get_db:
                    # Setup mock embeddings manager with STRING UIDs
                    mock_em = Mock()
                    mock_em.search_similar.return_value = {
                        "ids": [["uid_123", "uid_456"]],  # Use string UIDs
                        "distances": [[0.1, 0.2]],
                        "documents": [["doc1", "doc2"]],
                    }
                    mock_get_em.return_value = mock_em

                    # Setup mock database with lightweight schema
                    mock_db = Mock()
                    mock_paper1 = {
                        "uid": "uid_123",
                        "title": "Test Paper 1",
                        "abstract": "Abstract 1",
                        "authors": "Author A, Author B",  # Comma-separated string
                    }
                    mock_paper2 = {
                        "uid": "uid_456",
                        "title": "Test Paper 2",
                        "abstract": "Abstract 2",
                        "authors": "Author C, Author D",  # Comma-separated string
                    }

                    # Mock query to return paper rows
                    def mock_query(sql, params):
                        paper_uid = params[0]
                        if paper_uid == "uid_123":
                            return [mock_paper1]
                        elif paper_uid == "uid_456":
                            return [mock_paper2]
                        return []

                    mock_db.query.side_effect = mock_query
                    mock_get_db.return_value = mock_db

                    # Make request
                    response = client.post(
                        "/api/search",
                        json={"query": "test query", "use_embeddings": True, "limit": 5},
                    )

                    assert response.status_code == 200
                    data = response.get_json()

                    # Verify transformation happened
                    assert "papers" in data
                    assert len(data["papers"]) == 2

                    # Check similarity scores were added
                    assert "similarity" in data["papers"][0]
                    assert "similarity" in data["papers"][1]

                    # Verify similarity calculation (1 - distance)
                    assert data["papers"][0]["similarity"] == pytest.approx(0.9, 0.01)
                    assert data["papers"][1]["similarity"] == pytest.approx(0.8, 0.01)

                    # Verify 'uid' field is present
                    assert data["papers"][0]["uid"] == "uid_123"
                    assert data["papers"][1]["uid"] == "uid_456"

    def test_semantic_search_handles_empty_results(self):
        """Test semantic search with no results."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_embeddings_manager") as mock_get_em:
                mock_em = Mock()
                mock_em.search_similar.return_value = {
                    "ids": [[]],
                    "distances": [[]],
                    "documents": [[]],
                }
                mock_get_em.return_value = mock_em

                response = client.post(
                    "/api/search",
                    json={"query": "nonexistent", "use_embeddings": True, "limit": 5},
                )

                assert response.status_code == 200
                data = response.get_json()
                assert data["papers"] == []
                assert data["count"] == 0


class TestWebUIChatEndpointSuccess:
    """Test chat endpoint success paths (lines 219-227, 255-270)."""

    def test_chat_with_valid_message_success(self):
        """Test successful chat response."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_rag_chat") as mock_get_rag:
                with patch("abstracts_explorer.web_ui.app.get_config") as mock_get_config:
                    # Setup mocks
                    mock_config = Mock()
                    mock_config.max_context_papers = 3
                    mock_get_config.return_value = mock_config

                    mock_rag = Mock()
                    mock_rag.query.return_value = {
                        "response": "This is a test response",
                        "papers": [{"title": "Paper 1"}],
                        "metadata": {"n_papers": 1},
                    }
                    mock_get_rag.return_value = mock_rag

                    # Make request
                    response = client.post(
                        "/api/chat",
                        json={"message": "What is a transformer?"},
                    )

                    assert response.status_code == 200
                    data = response.get_json()

                    # Verify response structure (lines 255-260)
                    assert "response" in data
                    assert "message" in data
                    assert data["message"] == "What is a transformer?"

                    # Verify query was called
                    mock_rag.query.assert_called_once()

    def test_chat_with_custom_n_papers(self):
        """Test chat with custom n_papers parameter."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_rag_chat") as mock_get_rag:
                with patch("abstracts_explorer.web_ui.app.get_config") as mock_get_config:
                    mock_config = Mock()
                    mock_config.max_context_papers = 3
                    mock_get_config.return_value = mock_config

                    mock_rag = Mock()
                    mock_rag.query.return_value = {
                        "response": "Test",
                        "papers": [],
                        "metadata": {},
                    }
                    mock_get_rag.return_value = mock_rag

                    response = client.post(
                        "/api/chat",
                        json={"message": "Test question", "n_papers": 5},
                    )

                    assert response.status_code == 200
                    # Verify n_papers was used
                    mock_rag.query.assert_called_with("Test question", n_results=5, metadata_filter=None)

    def test_chat_with_reset_flag(self):
        """Test chat with reset=True (lines 249-253)."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_rag_chat") as mock_get_rag:
                with patch("abstracts_explorer.web_ui.app.get_config") as mock_get_config:
                    mock_config = Mock()
                    mock_config.max_context_papers = 3
                    mock_get_config.return_value = mock_config

                    mock_rag = Mock()
                    mock_rag.query.return_value = {
                        "response": "Test",
                        "papers": [],
                        "metadata": {},
                    }
                    mock_get_rag.return_value = mock_rag

                    response = client.post(
                        "/api/chat",
                        json={"message": "Test", "reset": True},
                    )

                    assert response.status_code == 200
                    # Verify reset_conversation was called
                    mock_rag.reset_conversation.assert_called_once()


class TestWebUIErrorHandlingDetails:
    """Test error handling paths (lines 287-288, 304-305)."""

    def test_chat_reset_exception_handling(self):
        """Test chat reset endpoint handles exceptions (lines 287-288)."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_rag_chat") as mock_get_rag:
                mock_rag = Mock()
                mock_rag.reset_conversation.side_effect = Exception("Reset failed")
                mock_get_rag.return_value = mock_rag

                response = client.post("/api/chat/reset")

                assert response.status_code == 500
                data = response.get_json()
                assert "error" in data
                assert "Reset failed" in data["error"]

    def test_stats_endpoint_exception_handling(self):
        """Test stats endpoint handles exceptions (lines 304-305)."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_database") as mock_get_db:
                mock_db = Mock()
                mock_db.get_paper_count.side_effect = Exception("Database error")
                mock_get_db.return_value = mock_db

                response = client.get("/api/stats")

                assert response.status_code == 500
                data = response.get_json()
                assert "error" in data
                assert "Database error" in data["error"]


class TestWebUIGetPaperDetails:
    """Test get_paper endpoint (lines 219-227)."""

    def test_get_paper_with_authors_list(self):
        """Test that paper details include authors as list."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_database") as mock_get_db:
                mock_db = Mock()

                # Mock paper data with lightweight schema (authors as semicolon-separated string)
                paper_row = {
                    "uid": "test_uid_123",
                    "title": "Test Paper",
                    "abstract": "Test abstract",
                    "authors": "Author 1; Author 2",  # Semicolon-separated string
                    "session": "Poster Session 1",
                    "poster_position": "123",
                }
                mock_db.query.return_value = [paper_row]
                mock_get_db.return_value = mock_db

                # Use string UID (not integer ID)
                response = client.get("/api/paper/test_uid_123")

                assert response.status_code == 200
                data = response.get_json()

                # Verify authors are included as list (parsed from semicolon-separated string)
                assert "authors" in data
                assert data["authors"] == ["Author 1", "Author 2"]
                # Verify 'uid' field is present
                assert data["uid"] == "test_uid_123"


class TestWebUIStatsEndpoint:
    """Test stats endpoint details (line 319)."""

    def test_stats_returns_paper_count(self):
        """Test that stats endpoint returns paper count."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_database") as mock_get_db:
                mock_db = Mock()
                mock_db.get_paper_count.return_value = 42
                mock_get_db.return_value = mock_db

                response = client.get("/api/stats")

                assert response.status_code == 200
                data = response.get_json()

                assert "total_papers" in data
                assert data["total_papers"] == 42


class TestWebUISearchExceptionHandling:
    """Test search endpoint exception handling (line 195)."""

    def test_search_handles_database_exception(self):
        """Test that search endpoint handles database exceptions."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_database") as mock_get_db:
                mock_db = Mock()
                mock_db.search_papers.side_effect = Exception("Database connection failed")
                mock_db.get_paper_authors.return_value = []
                mock_get_db.return_value = mock_db

                response = client.post(
                    "/api/search",
                    json={"query": "test", "use_embeddings": False, "limit": 10},
                )

                assert response.status_code == 500
                data = response.get_json()
                assert "error" in data

    def test_search_handles_embeddings_exception(self):
        """Test that semantic search handles embeddings exceptions."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_embeddings_manager") as mock_get_em:
                mock_em = Mock()
                mock_em.search_similar.side_effect = Exception("Embeddings error")
                mock_get_em.return_value = mock_em

                response = client.post(
                    "/api/search",
                    json={"query": "test", "use_embeddings": True, "limit": 10},
                )

                assert response.status_code == 500
                data = response.get_json()
                assert "error" in data


class TestWebUIDatabaseNotFound:
    """Test database file not found handling (line 43)."""

    def test_get_database_file_not_found(self):
        """Test that get_database raises FileNotFoundError when database doesn't exist."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            # Patch os.path.exists to return False for the database path
            with patch("abstracts_explorer.web_ui.app.os.path.exists", return_value=False):
                with patch("abstracts_explorer.web_ui.app.get_config") as mock_get_config:
                    mock_config = Mock()
                    mock_config.paper_db_path = "/nonexistent/database.db"
                    mock_get_config.return_value = mock_config

                    # Try to access endpoint that uses database
                    response = client.get("/api/stats")

                    # Should fail because database doesn't exist
                    assert response.status_code == 500
                    data = response.get_json()
                    assert "error" in data
                    # Should mention the database file not found
                    assert "not found" in data["error"].lower() or "filenotfounderror" in str(data["error"]).lower()


class TestWebUIChatExceptionLines:
    """Test specific exception handling lines in chat endpoint."""

    def test_chat_exception_returns_500(self):
        """Test that chat exceptions return 500 with error message (lines 269-270)."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_rag_chat") as mock_get_rag:
                with patch("abstracts_explorer.web_ui.app.get_config") as mock_get_config:
                    mock_config = Mock()
                    mock_config.max_context_papers = 3
                    mock_get_config.return_value = mock_config

                    mock_rag = Mock()
                    mock_rag.query.side_effect = Exception("RAG system error")
                    mock_get_rag.return_value = mock_rag

                    response = client.post(
                        "/api/chat",
                        json={"message": "test"},
                    )

                    assert response.status_code == 500
                    data = response.get_json()
                    assert "error" in data
                    assert "RAG system error" in data["error"]


class TestWebUIGetPaperException:
    """Test get_paper exception handling."""

    def test_get_paper_not_found_returns_404(self):
        """Test that missing paper returns 404."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_database") as mock_get_db:
                mock_db = Mock()
                mock_db.query.return_value = []  # No paper found
                mock_get_db.return_value = mock_db

                response = client.get("/api/paper/999")

                assert response.status_code == 404
                data = response.get_json()
                assert "error" in data
                assert "not found" in data["error"].lower()

    def test_get_paper_database_error_returns_404(self):
        """Test that database exceptions are wrapped as PaperFormattingError and return 404.

        This is by design - our new API fails early and converts all database errors
        to PaperFormattingError which returns 404 (not found).
        """
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_database") as mock_get_db:
                mock_db = Mock()
                # Simulate a database exception - gets wrapped as PaperFormattingError
                mock_db.query.side_effect = RuntimeError("Database connection lost")
                mock_get_db.return_value = mock_db

                response = client.get("/api/paper/1")

                # Database errors are wrapped as PaperFormattingError, which returns 404
                assert response.status_code == 404
                data = response.get_json()
                assert "error" in data
                assert "Failed to retrieve paper" in data["error"]


class TestWebUIStatsExceptionHandling:
    """Test stats endpoint exception handling more thoroughly."""

    def test_stats_paper_count_calculation(self):
        """Test that stats correctly calls get_paper_count (line 319)."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_database") as mock_get_db:
                mock_db = Mock()
                mock_db.get_paper_count.return_value = 100
                mock_get_db.return_value = mock_db

                response = client.get("/api/stats")

                assert response.status_code == 200
                data = response.get_json()
                assert data["total_papers"] == 100

                # Verify get_paper_count was actually called
                mock_db.get_paper_count.assert_called_once()


class TestWebUIRunServer:
    """Test run_server function with Waitress production server."""

    def test_run_server_starts_waitress_server(self):
        """Test that run_server starts Waitress server by default."""
        from abstracts_explorer.web_ui import run_server

        with patch("abstracts_explorer.web_ui.app.os.path.exists", return_value=True):
            with patch("abstracts_explorer.web_ui.app.get_config") as mock_config:
                mock_cfg = Mock()
                mock_cfg.paper_db_path = "/some/path/test.db"
                mock_cfg.embedding_db_path = "/some/path/chroma_db"
                mock_config.return_value = mock_cfg
                
                # Mock Waitress serve with timeout
                with patch("waitress.serve") as mock_serve:
                    mock_serve.side_effect = KeyboardInterrupt()  # Simulate server stop
                    
                    with pytest.raises(KeyboardInterrupt):
                        run_server(host="127.0.0.1", port=5000, debug=False)

                    # Verify Waitress serve was called with correct parameters
                    mock_serve.assert_called_once()
                    call_args = mock_serve.call_args
                    assert call_args[1]["host"] == "127.0.0.1"
                    assert call_args[1]["port"] == 5000

    def test_run_server_with_debug_mode(self):
        """Test run_server with debug=True uses Flask dev server."""
        from abstracts_explorer.web_ui import run_server, app

        with patch("abstracts_explorer.web_ui.app.os.path.exists", return_value=True):
            with patch("abstracts_explorer.web_ui.app.get_config") as mock_config:
                mock_cfg = Mock()
                mock_cfg.paper_db_path = "/some/path/test.db"
                mock_cfg.embedding_db_path = "/some/path/chroma_db"
                mock_config.return_value = mock_cfg
                
                # Mock Flask app.run with timeout
                with patch.object(app, "run") as mock_run:
                    mock_run.side_effect = KeyboardInterrupt()  # Simulate server stop
                    
                    with pytest.raises(KeyboardInterrupt):
                        run_server(host="0.0.0.0", port=8080, debug=True)

                    # Verify Flask dev server was called with correct parameters
                    mock_run.assert_called_once_with(host="0.0.0.0", port=8080, debug=True)

    def test_run_server_database_not_found(self, capsys):
        """Test that run_server raises FileNotFoundError when database doesn't exist."""
        from abstracts_explorer.web_ui import run_server

        with patch("abstracts_explorer.web_ui.app.os.path.exists", return_value=False):
            with patch("abstracts_explorer.web_ui.app.get_config") as mock_config:
                mock_cfg = Mock()
                mock_cfg.paper_db_path = "/nonexistent/test.db"
                mock_cfg.embedding_db_path = "/some/path/chroma_db"
                mock_config.return_value = mock_cfg
                
                with pytest.raises(FileNotFoundError) as exc_info:
                    run_server(host="127.0.0.1", port=5000, debug=False)
                
                # Verify error message includes database path
                assert "/nonexistent/test.db" in str(exc_info.value)
        
        # Verify helpful error message was printed
        captured = capsys.readouterr()
        assert "Database not found" in captured.err
        assert "neurips-abstracts download" in captured.err
        assert "create-embeddings" in captured.err


class TestConferencePluginMapping:
    """Test conference name to plugin name mapping in download endpoint."""

    def test_ml4ps_conference_name_maps_to_plugin(self):
        """Test that 'ML4PS@Neurips' conference name correctly maps to 'ml4ps' plugin."""
        from abstracts_explorer.plugins import get_plugin, list_plugins
        
        # Get all plugins and build mapping
        plugins = list_plugins()
        conference_to_plugin = {}
        for plugin_info in plugins:
            conf_name = plugin_info.get("conference_name")
            plug_name = plugin_info.get("name")
            if conf_name and plug_name:
                conference_to_plugin[conf_name] = plug_name
        
        # Verify ML4PS@Neurips maps to ml4ps
        assert "ML4PS@Neurips" in conference_to_plugin
        assert conference_to_plugin["ML4PS@Neurips"] == "ml4ps"
        
        # Verify we can get the plugin
        plugin_name = conference_to_plugin["ML4PS@Neurips"]
        plugin = get_plugin(plugin_name)
        assert plugin is not None
        assert plugin.plugin_name == "ml4ps"

    def test_all_conferences_map_to_plugins(self):
        """Test that all conference names can be mapped to valid plugins."""
        from abstracts_explorer.plugins import get_plugin, list_plugins
        
        # Get all plugins
        plugins = list_plugins()
        
        # Build mapping and verify each conference has a valid plugin
        for plugin_info in plugins:
            conf_name = plugin_info.get("conference_name")
            plug_name = plugin_info.get("name")
            
            if conf_name and plug_name:
                # Verify we can get the plugin
                plugin = get_plugin(plug_name)
                assert plugin is not None, f"Plugin '{plug_name}' for conference '{conf_name}' not found"
                
                # Verify conference name and plugin name match
                assert plugin.conference_name == conf_name


class TestServerInitialization:
    """Test server initialization with production and development modes."""

    def test_run_server_with_waitress_available(self, tmp_path, monkeypatch):
        """Test that run_server uses Waitress by default when available."""
        from abstracts_explorer.web_ui.app import run_server
        from abstracts_explorer.database import DatabaseManager
        
        # Create a test database
        db_path = tmp_path / "test.db"
        db = DatabaseManager(str(db_path))
        with db:
            db.create_tables()
        
        # Mock config to use our test database
        with patch("abstracts_explorer.web_ui.app.get_config") as mock_config:
            mock_config.return_value = Mock(
                paper_db_path=str(db_path),
                embedding_db_path="chroma_db"
            )
            
            # Mock Waitress serve - need to patch the import inside the function
            with patch("waitress.serve") as mock_serve:
                mock_serve.side_effect = KeyboardInterrupt()  # Simulate Ctrl+C
                
                with pytest.raises(KeyboardInterrupt):
                    run_server(host="127.0.0.1", port=5000, debug=False, dev=False)
                
                # Verify Waitress was called
                mock_serve.assert_called_once()
                call_args = mock_serve.call_args
                assert call_args[1]["host"] == "127.0.0.1"
                assert call_args[1]["port"] == 5000

    def test_run_server_with_dev_flag(self, tmp_path, monkeypatch):
        """Test that run_server uses Flask dev server when dev=True."""
        from abstracts_explorer.web_ui.app import run_server, app
        from abstracts_explorer.database import DatabaseManager
        
        # Create a test database
        db_path = tmp_path / "test.db"
        db = DatabaseManager(str(db_path))
        with db:
            db.create_tables()
        
        # Mock config to use our test database
        with patch("abstracts_explorer.web_ui.app.get_config") as mock_config:
            mock_config.return_value = Mock(
                paper_db_path=str(db_path),
                embedding_db_path="chroma_db"
            )
            
            # Mock app.run
            with patch.object(app, "run") as mock_run:
                mock_run.side_effect = KeyboardInterrupt()  # Simulate Ctrl+C
                
                with pytest.raises(KeyboardInterrupt):
                    run_server(host="127.0.0.1", port=5000, debug=False, dev=True)
                
                # Verify Flask dev server was called
                mock_run.assert_called_once_with(host="127.0.0.1", port=5000, debug=False)

    def test_run_server_with_debug_flag(self, tmp_path, monkeypatch):
        """Test that run_server uses Flask dev server when debug=True."""
        from abstracts_explorer.web_ui.app import run_server, app
        from abstracts_explorer.database import DatabaseManager
        
        # Create a test database
        db_path = tmp_path / "test.db"
        db = DatabaseManager(str(db_path))
        with db:
            db.create_tables()
        
        # Mock config to use our test database
        with patch("abstracts_explorer.web_ui.app.get_config") as mock_config:
            mock_config.return_value = Mock(
                paper_db_path=str(db_path),
                embedding_db_path="chroma_db"
            )
            
            # Mock app.run
            with patch.object(app, "run") as mock_run:
                mock_run.side_effect = KeyboardInterrupt()  # Simulate Ctrl+C
                
                with pytest.raises(KeyboardInterrupt):
                    run_server(host="127.0.0.1", port=5000, debug=True, dev=False)
                
                # Verify Flask dev server was called with debug=True
                mock_run.assert_called_once_with(host="127.0.0.1", port=5000, debug=True)

    def test_run_server_waitress_not_available(self, tmp_path, monkeypatch):
        """Test that run_server falls back to Flask when Waitress is not available."""
        from abstracts_explorer.web_ui.app import run_server, app
        from abstracts_explorer.database import DatabaseManager
        
        # Create a test database
        db_path = tmp_path / "test.db"
        db = DatabaseManager(str(db_path))
        with db:
            db.create_tables()
        
        # Mock config to use our test database
        with patch("abstracts_explorer.web_ui.app.get_config") as mock_config:
            mock_config.return_value = Mock(
                paper_db_path=str(db_path),
                embedding_db_path="chroma_db"
            )
            
            # Mock waitress import to fail
            import builtins
            real_import = builtins.__import__
            
            def mock_import(name, *args, **kwargs):
                if name == "waitress":
                    raise ImportError("Waitress not installed")
                return real_import(name, *args, **kwargs)
            
            # Mock app.run
            with patch.object(app, "run") as mock_run:
                mock_run.side_effect = KeyboardInterrupt()
                
                with patch("builtins.__import__", side_effect=mock_import):
                    with pytest.raises(KeyboardInterrupt):
                        run_server(host="127.0.0.1", port=5000, debug=False, dev=False)
                
                # Verify Flask was called as fallback
                mock_run.assert_called_once_with(host="127.0.0.1", port=5000, debug=False)

    def test_run_server_missing_database(self, tmp_path):
        """Test that run_server raises FileNotFoundError when database is missing."""
        from abstracts_explorer.web_ui.app import run_server
        
        # Use a non-existent database path
        db_path = tmp_path / "nonexistent.db"
        
        # Mock config to use non-existent database
        with patch("abstracts_explorer.web_ui.app.get_config") as mock_config:
            mock_config.return_value = Mock(
                paper_db_path=str(db_path),
                embedding_db_path="chroma_db"
            )
            
            # Should raise FileNotFoundError
            with pytest.raises(FileNotFoundError, match="Database not found"):
                run_server(host="127.0.0.1", port=5000)


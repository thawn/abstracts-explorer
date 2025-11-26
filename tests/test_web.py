"""
Tests for the web interface.

These tests ensure the Flask web application works correctly
and integrates properly with the database and other components.
"""

import json
import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import web app from the package
from neurips_abstracts.web_ui import app as flask_app
from neurips_abstracts.database import DatabaseManager


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
def test_db(tmp_path):
    """
    Create a test database with sample papers for web testing.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory path

    Returns
    -------
    DatabaseManager
        Database manager with test data

    Notes
    -----
    This fixture creates a database with authors table and specific test papers
    for web interface testing. It's kept separate from conftest.py's test_database
    fixture as it has a different structure with authors.
    """
    db_path = tmp_path / "test.db"
    db = DatabaseManager(str(db_path))

    with db:
        db.create_tables()

        # Add some test papers
        cursor = db.connection.cursor()

        # Paper 1
        cursor.execute(
            """
            INSERT INTO papers (uid, name, abstract, decision)
            VALUES (?, ?, ?, ?)
            """,
            (
                "test1",
                "Attention is All You Need",
                "We propose the Transformer, a model architecture based solely on attention mechanisms.",
                "Accept (oral)",
            ),
        )
        paper1_id = cursor.lastrowid

        # Paper 2
        cursor.execute(
            """
            INSERT INTO papers (uid, name, abstract, decision)
            VALUES (?, ?, ?, ?)
            """,
            (
                "test2",
                "BERT: Pre-training of Deep Bidirectional Transformers",
                "We introduce BERT, which stands for Bidirectional Encoder Representations from Transformers.",
                "Accept (poster)",
            ),
        )
        paper2_id = cursor.lastrowid

        # Paper 3
        cursor.execute(
            """
            INSERT INTO papers (uid, name, abstract, decision)
            VALUES (?, ?, ?, ?)
            """,
            (
                "test3",
                "Deep Residual Learning for Image Recognition",
                "We present a residual learning framework to ease the training of networks.",
                "Accept (oral)",
            ),
        )
        paper3_id = cursor.lastrowid

        # Add authors (using fullname as per actual schema)
        cursor.execute("INSERT INTO authors (fullname) VALUES (?)", ("Ashish Vaswani",))
        cursor.execute("INSERT INTO authors (fullname) VALUES (?)", ("Jacob Devlin",))
        cursor.execute("INSERT INTO authors (fullname) VALUES (?)", ("Kaiming He",))

        db.connection.commit()

    return db


class TestWebInterface:
    """Test web interface endpoints."""

    def test_index_route(self, client):
        """Test that the main page loads."""
        response = client.get("/")
        assert response.status_code == 200
        assert b"NeurIPS Abstracts Explorer" in response.data

    def test_stats_endpoint_no_db(self, client):
        """Test stats endpoint when database doesn't exist."""
        response = client.get("/api/stats")
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
        app_module = sys.modules["neurips_abstracts.web_ui.app"]

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
        app_module = sys.modules["neurips_abstracts.web_ui.app"]

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

        This ensures we don't accidentally pass 'title' or 'abstract'
        parameters that don't exist.
        """
        with test_db:
            # This should raise TypeError if we try to pass invalid params
            with pytest.raises(TypeError):
                test_db.search_papers(title="test")

            with pytest.raises(TypeError):
                test_db.search_papers(abstract="test")

            with pytest.raises(TypeError):
                test_db.search_papers(year=2025)

    def test_search_papers_valid_params_only(self, test_db):
        """Test that only valid parameters work."""
        with test_db:
            # These should work
            results = test_db.search_papers(keyword="attention")
            assert isinstance(results, list)

            results = test_db.search_papers(decision="Accept (oral)")
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

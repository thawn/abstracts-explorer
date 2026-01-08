"""
Unit tests for web_ui/app.py to increase coverage to 90%+.

These tests specifically target uncovered lines in the web UI application.
"""

import pytest
from unittest.mock import Mock, patch


class TestWebUISemanticSearchDetails:
    """Test semantic search result processing (lines 165-183)."""

    def test_semantic_search_transforms_chromadb_results(self):
        """Test that semantic search correctly transforms ChromaDB results to paper format."""
        # Import inside test to avoid import-time issues
        from neurips_abstracts.web_ui.app import app

        with app.test_client() as client:
            # Mock the get_embeddings_manager and get_database functions
            with patch("neurips_abstracts.web_ui.app.get_embeddings_manager") as mock_get_em:
                with patch("neurips_abstracts.web_ui.app.get_database") as mock_get_db:
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
        from neurips_abstracts.web_ui.app import app

        with app.test_client() as client:
            with patch("neurips_abstracts.web_ui.app.get_embeddings_manager") as mock_get_em:
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
        from neurips_abstracts.web_ui.app import app

        with app.test_client() as client:
            with patch("neurips_abstracts.web_ui.app.get_rag_chat") as mock_get_rag:
                with patch("neurips_abstracts.web_ui.app.get_config") as mock_get_config:
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
        from neurips_abstracts.web_ui.app import app

        with app.test_client() as client:
            with patch("neurips_abstracts.web_ui.app.get_rag_chat") as mock_get_rag:
                with patch("neurips_abstracts.web_ui.app.get_config") as mock_get_config:
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
        from neurips_abstracts.web_ui.app import app

        with app.test_client() as client:
            with patch("neurips_abstracts.web_ui.app.get_rag_chat") as mock_get_rag:
                with patch("neurips_abstracts.web_ui.app.get_config") as mock_get_config:
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
        from neurips_abstracts.web_ui.app import app

        with app.test_client() as client:
            with patch("neurips_abstracts.web_ui.app.get_rag_chat") as mock_get_rag:
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
        from neurips_abstracts.web_ui.app import app

        with app.test_client() as client:
            with patch("neurips_abstracts.web_ui.app.get_database") as mock_get_db:
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
        from neurips_abstracts.web_ui.app import app

        with app.test_client() as client:
            with patch("neurips_abstracts.web_ui.app.get_database") as mock_get_db:
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
        from neurips_abstracts.web_ui.app import app

        with app.test_client() as client:
            with patch("neurips_abstracts.web_ui.app.get_database") as mock_get_db:
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
        from neurips_abstracts.web_ui.app import app

        with app.test_client() as client:
            with patch("neurips_abstracts.web_ui.app.get_database") as mock_get_db:
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
        from neurips_abstracts.web_ui.app import app

        with app.test_client() as client:
            with patch("neurips_abstracts.web_ui.app.get_embeddings_manager") as mock_get_em:
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
        from neurips_abstracts.web_ui.app import app

        with app.test_client() as client:
            # Patch os.path.exists to return False for the database path
            with patch("neurips_abstracts.web_ui.app.os.path.exists", return_value=False):
                with patch("neurips_abstracts.web_ui.app.get_config") as mock_get_config:
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
        from neurips_abstracts.web_ui.app import app

        with app.test_client() as client:
            with patch("neurips_abstracts.web_ui.app.get_rag_chat") as mock_get_rag:
                with patch("neurips_abstracts.web_ui.app.get_config") as mock_get_config:
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
        from neurips_abstracts.web_ui.app import app

        with app.test_client() as client:
            with patch("neurips_abstracts.web_ui.app.get_database") as mock_get_db:
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
        from neurips_abstracts.web_ui.app import app

        with app.test_client() as client:
            with patch("neurips_abstracts.web_ui.app.get_database") as mock_get_db:
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
        from neurips_abstracts.web_ui.app import app

        with app.test_client() as client:
            with patch("neurips_abstracts.web_ui.app.get_database") as mock_get_db:
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
    """Test run_server function (lines 335-342)."""

    def test_run_server_starts_flask_app(self):
        """Test that run_server starts the Flask app."""
        from neurips_abstracts.web_ui import run_server, app

        with patch.object(app, "run") as mock_run:
            with patch("neurips_abstracts.web_ui.app.os.path.exists", return_value=True):
                with patch("neurips_abstracts.web_ui.app.get_config") as mock_config:
                    mock_cfg = Mock()
                    mock_cfg.paper_db_path = "/some/path/test.db"
                    mock_cfg.embedding_db_path = "/some/path/chroma_db"
                    mock_config.return_value = mock_cfg
                    
                    run_server(host="127.0.0.1", port=5000, debug=False)

            # Verify app.run was called with correct parameters
            mock_run.assert_called_once_with(host="127.0.0.1", port=5000, debug=False)

    def test_run_server_with_debug_mode(self):
        """Test run_server with debug=True."""
        from neurips_abstracts.web_ui import run_server, app

        with patch.object(app, "run") as mock_run:
            with patch("neurips_abstracts.web_ui.app.os.path.exists", return_value=True):
                with patch("neurips_abstracts.web_ui.app.get_config") as mock_config:
                    mock_cfg = Mock()
                    mock_cfg.paper_db_path = "/some/path/test.db"
                    mock_cfg.embedding_db_path = "/some/path/chroma_db"
                    mock_config.return_value = mock_cfg
                    
                    run_server(host="0.0.0.0", port=8080, debug=True)

            mock_run.assert_called_once_with(host="0.0.0.0", port=8080, debug=True)

    def test_run_server_database_not_found(self, capsys):
        """Test that run_server raises FileNotFoundError when database doesn't exist."""
        from neurips_abstracts.web_ui import run_server

        with patch("neurips_abstracts.web_ui.app.os.path.exists", return_value=False):
            with patch("neurips_abstracts.web_ui.app.get_config") as mock_config:
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
        from neurips_abstracts.plugins import get_plugin, list_plugins
        
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
        from neurips_abstracts.plugins import get_plugin, list_plugins
        
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

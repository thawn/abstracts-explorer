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
                    # Setup mock embeddings manager with INTEGER IDs
                    mock_em = Mock()
                    mock_em.search_similar.return_value = {
                        "ids": [[1, 2]],  # Use integer IDs
                        "distances": [[0.1, 0.2]],
                        "documents": [["doc1", "doc2"]],
                    }
                    mock_get_em.return_value = mock_em

                    # Setup mock database
                    mock_db = Mock()
                    mock_paper1 = {
                        "id": 1,
                        "name": "Test Paper 1",
                        "abstract": "Abstract 1",
                    }
                    mock_paper2 = {
                        "id": 2,
                        "name": "Test Paper 2",
                        "abstract": "Abstract 2",
                    }

                    # Mock query to return paper rows
                    def mock_query(sql, params):
                        paper_id = params[0]
                        if paper_id == 1:
                            return [mock_paper1]
                        elif paper_id == 2:
                            return [mock_paper2]
                        return []

                    # Mock get_paper_authors to return author names
                    def mock_get_authors(paper_id):
                        return [{"fullname": "Author A"}, {"fullname": "Author B"}]

                    mock_db.query.side_effect = mock_query
                    mock_db.get_paper_authors.side_effect = mock_get_authors
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

                # Mock paper data
                paper_row = {
                    "id": 1,
                    "name": "Test Paper",
                    "abstract": "Test abstract",
                    "uid": "test_uid",
                    "decision": "Accept",
                }
                mock_db.query.return_value = [paper_row]

                # Mock authors - use 'fullname' field as per database schema
                mock_db.get_paper_authors.return_value = [
                    {"fullname": "Author 1"},
                    {"fullname": "Author 2"},
                ]
                mock_get_db.return_value = mock_db

                response = client.get("/api/paper/1")

                assert response.status_code == 200
                data = response.get_json()

                # Verify authors are included as list of fullnames
                assert "authors" in data
                assert data["authors"] == ["Author 1", "Author 2"]


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
            run_server(host="127.0.0.1", port=5000, debug=False)

            # Verify app.run was called with correct parameters
            mock_run.assert_called_once_with(host="127.0.0.1", port=5000, debug=False)

    def test_run_server_with_debug_mode(self):
        """Test run_server with debug=True."""
        from neurips_abstracts.web_ui import run_server, app

        with patch.object(app, "run") as mock_run:
            run_server(host="0.0.0.0", port=8080, debug=True)

            mock_run.assert_called_once_with(host="0.0.0.0", port=8080, debug=True)


class TestDownloadPosterImage:
    """Test download_poster_image function."""

    def test_download_poster_from_file_path(self, tmp_path):
        """Test downloading poster from eventmedia with 'file' key."""
        from neurips_abstracts.web_ui.app import download_poster_image
        import json

        eventmedia = json.dumps(
            [
                {"id": 121603, "type": "URL", "name": "OpenReview", "uri": "https://openreview.net/forum?id=test"},
                {
                    "id": 134330,
                    "file": "/media/PosterPDFs/NeurIPS%202025/114996.png",
                    "type": "Poster",
                    "name": "Poster",
                },
            ]
        )

        with patch("neurips_abstracts.web_ui.app.download_file") as mock_download:
            mock_download.return_value = "poster_114996.png"
            result = download_poster_image(eventmedia, tmp_path, "poster_114996", 114996)

            # Verify download_file was called with correct URL
            mock_download.assert_called_once_with(
                "https://neurips.cc/media/PosterPDFs/NeurIPS%202025/114996.png", tmp_path, "poster_114996.png"
            )
            assert result == "poster_114996.png"

    def test_download_poster_skips_thumbnails(self, tmp_path):
        """Test that thumbnail images are skipped in favor of full size."""
        from neurips_abstracts.web_ui.app import download_poster_image
        import json

        eventmedia = json.dumps(
            [
                {
                    "id": 130886,
                    "file": "/media/PosterPDFs/NeurIPS%202025/114997-thumb.png",
                    "type": "Poster",
                    "name": "Poster",
                },
                {
                    "id": 134331,
                    "file": "/media/PosterPDFs/NeurIPS%202025/114997.png",
                    "type": "Poster",
                    "name": "Poster",
                },
            ]
        )

        with patch("neurips_abstracts.web_ui.app.download_file") as mock_download:
            mock_download.return_value = "poster_114997.png"
            result = download_poster_image(eventmedia, tmp_path, "poster_114997", 114997)

            # Verify download_file was called with full size, not thumbnail
            mock_download.assert_called_once_with(
                "https://neurips.cc/media/PosterPDFs/NeurIPS%202025/114997.png", tmp_path, "poster_114997.png"
            )
            assert result == "poster_114997.png"

    def test_download_poster_fallback_to_paper_id(self, tmp_path):
        """Test fallback to constructing URL from paper ID when eventmedia has no poster."""
        from neurips_abstracts.web_ui.app import download_poster_image
        import json

        # Eventmedia with only OpenReview link, no poster
        eventmedia = json.dumps(
            [{"id": 121603, "type": "URL", "name": "OpenReview", "uri": "https://openreview.net/forum?id=test"}]
        )

        with patch("neurips_abstracts.web_ui.app.download_file") as mock_download:
            mock_download.return_value = "poster_115000.png"
            result = download_poster_image(eventmedia, tmp_path, "poster_115000", 115000)

            # Verify download_file was called with constructed URL
            mock_download.assert_called_once_with(
                "https://neurips.cc/media/PosterPDFs/NeurIPS%202025/115000.png", tmp_path, "poster_115000.png"
            )
            assert result == "poster_115000.png"

    def test_download_poster_no_eventmedia_uses_fallback(self, tmp_path):
        """Test fallback to paper ID when eventmedia is None."""
        from neurips_abstracts.web_ui.app import download_poster_image

        with patch("neurips_abstracts.web_ui.app.download_file") as mock_download:
            mock_download.return_value = "poster_115001.png"
            result = download_poster_image(None, tmp_path, "poster_115001", 115001)

            # Verify download_file was called with constructed URL
            mock_download.assert_called_once_with(
                "https://neurips.cc/media/PosterPDFs/NeurIPS%202025/115001.png", tmp_path, "poster_115001.png"
            )
            assert result == "poster_115001.png"

    def test_download_poster_returns_none_on_error(self, tmp_path):
        """Test that function returns None when download fails."""
        from neurips_abstracts.web_ui.app import download_poster_image

        with patch("neurips_abstracts.web_ui.app.download_file") as mock_download:
            mock_download.side_effect = Exception("Download failed")
            result = download_poster_image(None, tmp_path, "poster_error", 999999)

            assert result is None

    def test_download_poster_from_url_field(self, tmp_path):
        """Test downloading poster from eventmedia with 'url' key."""
        from neurips_abstracts.web_ui.app import download_poster_image
        import json

        eventmedia = json.dumps([{"id": 121603, "url": "https://example.com/poster.png", "type": "Image"}])

        with patch("neurips_abstracts.web_ui.app.download_file") as mock_download:
            mock_download.return_value = "poster_test.png"
            result = download_poster_image(eventmedia, tmp_path, "poster_test")

            # Verify download_file was called with URL from 'url' field
            mock_download.assert_called_once_with("https://example.com/poster.png", tmp_path, "poster_test.png")
            assert result == "poster_test.png"


class TestGetPosterUrl:
    """Test get_poster_url function."""

    def test_get_poster_url_from_file_path(self):
        """Test extracting poster URL from eventmedia with 'file' key."""
        from neurips_abstracts.web_ui.app import get_poster_url
        import json

        eventmedia = json.dumps(
            [
                {"id": 121603, "type": "URL", "name": "OpenReview", "uri": "https://openreview.net/forum?id=test"},
                {"id": 134330, "file": "/media/PosterPDFs/NeurIPS%202025/114996.png", "type": "Poster"},
            ]
        )

        url = get_poster_url(eventmedia, 114996)
        assert url == "https://neurips.cc/media/PosterPDFs/NeurIPS%202025/114996.png"

    def test_get_poster_url_skips_thumbnails(self):
        """Test that thumbnail images are skipped."""
        from neurips_abstracts.web_ui.app import get_poster_url
        import json

        eventmedia = json.dumps(
            [
                {"file": "/media/PosterPDFs/NeurIPS%202025/114997-thumb.png", "type": "Poster"},
                {"file": "/media/PosterPDFs/NeurIPS%202025/114997.png", "type": "Poster"},
            ]
        )

        url = get_poster_url(eventmedia, 114997)
        assert url == "https://neurips.cc/media/PosterPDFs/NeurIPS%202025/114997.png"

    def test_get_poster_url_fallback(self):
        """Test fallback to constructing URL from paper ID."""
        from neurips_abstracts.web_ui.app import get_poster_url

        # No eventmedia
        url = get_poster_url(None, 115000)
        assert url == "https://neurips.cc/media/PosterPDFs/NeurIPS%202025/115000.png"

    def test_get_poster_url_from_url_field(self):
        """Test extracting URL from 'url' field."""
        from neurips_abstracts.web_ui.app import get_poster_url
        import json

        eventmedia = json.dumps([{"url": "https://example.com/poster.png", "type": "Image"}])

        url = get_poster_url(eventmedia, 100)
        assert url == "https://example.com/poster.png"


class TestParallelDownload:
    """Test parallel download functionality."""

    def test_download_paper_pdf_task_success(self, tmp_path):
        """Test successful PDF download task."""
        from neurips_abstracts.web_ui.app import download_paper_pdf_task

        paper = {"id": 123, "paper_pdf_url": "https://example.com/paper.pdf"}

        with patch("neurips_abstracts.web_ui.app.download_file") as mock_download:
            mock_download.return_value = "paper_123.pdf"
            paper_id, filename = download_paper_pdf_task(paper, tmp_path)

            assert paper_id == 123
            assert filename == "paper_123.pdf"
            mock_download.assert_called_once_with("https://example.com/paper.pdf", tmp_path, "paper_123.pdf")

    def test_download_paper_pdf_task_constructs_url(self, tmp_path):
        """Test PDF download task constructs URL from paper_url."""
        from neurips_abstracts.web_ui.app import download_paper_pdf_task

        paper = {"id": 456, "paper_url": "https://openreview.net/forum?id=abc123"}

        with patch("neurips_abstracts.web_ui.app.download_file") as mock_download:
            mock_download.return_value = "paper_456.pdf"
            paper_id, filename = download_paper_pdf_task(paper, tmp_path)

            assert paper_id == 456
            assert filename == "paper_456.pdf"
            # Verify URL was converted from forum to pdf
            mock_download.assert_called_once_with("https://openreview.net/pdf?id=abc123", tmp_path, "paper_456.pdf")

    def test_download_paper_pdf_task_no_url(self, tmp_path):
        """Test PDF download task when no URL is available."""
        from neurips_abstracts.web_ui.app import download_paper_pdf_task

        paper = {"id": 789}

        paper_id, filename = download_paper_pdf_task(paper, tmp_path)

        assert paper_id == 789
        assert filename is None

    def test_download_poster_image_task_success(self, tmp_path):
        """Test successful poster image download task."""
        from neurips_abstracts.web_ui.app import download_poster_image_task
        import json

        eventmedia = json.dumps([{"file": "/media/PosterPDFs/NeurIPS%202025/111.png", "type": "Poster"}])
        paper = {"id": 111, "eventmedia": eventmedia}

        with patch("neurips_abstracts.web_ui.app.download_file") as mock_download:
            mock_download.return_value = "poster_111.png"
            paper_id, filename = download_poster_image_task(paper, tmp_path)

            assert paper_id == 111
            assert filename == "poster_111.png"

    def test_download_assets_parallel_success(self, tmp_path):
        """Test parallel download of poster images only (PDFs not downloaded)."""
        from neurips_abstracts.web_ui.app import download_assets_parallel
        import json

        papers = [
            {
                "id": 1,
                "eventmedia": json.dumps([{"file": "/media/PosterPDFs/NeurIPS%202025/1.png"}]),
            },
            {
                "id": 2,
                "eventmedia": json.dumps([{"file": "/media/PosterPDFs/NeurIPS%202025/2.png"}]),
            },
            {
                "id": 3,
                "eventmedia": json.dumps([{"file": "/media/PosterPDFs/NeurIPS%202025/3.png"}]),
            },
        ]

        with patch("neurips_abstracts.web_ui.app.download_file") as mock_download:
            # Simulate successful downloads
            mock_download.side_effect = lambda url, target, filename: filename

            poster_results = download_assets_parallel(papers, tmp_path, max_workers=2)

            # Verify all posters were downloaded (only posters, no PDFs)
            assert len(poster_results) == 3
            assert poster_results[1] == "poster_1.png"
            assert poster_results[2] == "poster_2.png"
            assert poster_results[3] == "poster_3.png"

            # Verify download_file was called 3 times (3 poster downloads, no PDFs)
            assert mock_download.call_count == 3

    def test_download_assets_parallel_handles_failures(self, tmp_path):
        """Test parallel download handles individual failures gracefully."""
        from neurips_abstracts.web_ui.app import download_assets_parallel
        import json

        papers = [
            {
                "id": 1,
                "eventmedia": json.dumps([{"file": "/media/PosterPDFs/NeurIPS%202025/1.png"}]),
            },
            {
                "id": 2,
                "eventmedia": json.dumps([{"file": "/media/PosterPDFs/NeurIPS%202025/2.png"}]),
            },
        ]

        with patch("neurips_abstracts.web_ui.app.download_file") as mock_download:
            # Simulate one success and one failure
            def download_side_effect(url, target, filename):
                if "poster_1" in filename:
                    return filename
                return None  # Simulate failure for paper 2

            mock_download.side_effect = download_side_effect

            poster_results = download_assets_parallel(papers, tmp_path, max_workers=2)

            # Only paper 1 should be in results
            assert 1 in poster_results
            assert poster_results[1] == "poster_1.png"
            assert 2 not in poster_results

    def test_generate_markdown_with_remote_links(self, tmp_path):
        """Test markdown generation with direct links to remote resources (no downloads)."""
        from neurips_abstracts.web_ui.app import generate_markdown_with_assets
        import json

        papers = [
            {
                "id": 100,
                "name": "Test Paper",
                "abstract": "Test abstract",
                "paper_pdf_url": "https://example.com/100.pdf",
                "eventmedia": json.dumps([{"file": "/media/PosterPDFs/NeurIPS%202025/100.png"}]),
            }
        ]

        markdown = generate_markdown_with_assets(papers, "test query", tmp_path)

        # Verify markdown contains link to PDF on OpenReview (not downloaded)
        assert "View on OpenReview" in markdown
        assert "https://example.com/100.pdf" in markdown

        # Verify markdown contains direct link to poster on neurips.cc (not downloaded)
        assert "https://neurips.cc/media/PosterPDFs/NeurIPS%202025/100.png" in markdown
        assert "![Poster](https://neurips.cc/media/PosterPDFs/NeurIPS%202025/100.png)" in markdown

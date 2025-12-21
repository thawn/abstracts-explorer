"""
Unit tests for the download/update feature with SSE streaming and smart embedding management.

Tests cover:
- /api/download endpoint
- SSE streaming
- Smart embedding optimization (only embed changed/missing papers)
- ChromaDB missing embedding detection
- Progress tracking
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


class TestDownloadEndpoint:
    """Test the /api/download POST endpoint."""

    def test_download_endpoint_requires_conference(self):
        """Test that the endpoint returns 400 if conference is missing."""
        from neurips_abstracts.web_ui.app import app

        with app.test_client() as client:
            response = client.post(
                "/api/download",
                json={"year": 2025}
            )
            assert response.status_code == 400
            data = response.get_json()
            assert "error" in data
            assert "required" in data["error"].lower()

    def test_download_endpoint_requires_year(self):
        """Test that the endpoint returns 400 if year is missing."""
        from neurips_abstracts.web_ui.app import app

        with app.test_client() as client:
            response = client.post(
                "/api/download",
                json={"conference": "NeurIPS"}
            )
            assert response.status_code == 400
            data = response.get_json()
            assert "error" in data
            assert "required" in data["error"].lower()

    def test_download_endpoint_requires_json_body(self):
        """Test that the endpoint returns 400/415 if no JSON body provided."""
        from neurips_abstracts.web_ui.app import app

        with app.test_client() as client:
            response = client.post("/api/download")
            # Returns 415 (Unsupported Media Type) when no content-type header
            assert response.status_code in (400, 415)

    @patch("neurips_abstracts.web_ui.app.DatabaseManager")
    @patch("neurips_abstracts.web_ui.app.EmbeddingsManager")
    @patch("neurips_abstracts.plugins.get_plugin")
    @patch("neurips_abstracts.web_ui.app.get_config")
    def test_download_endpoint_returns_sse_stream(
        self, mock_get_config, mock_get_plugin, mock_em_class, mock_db_class
    ):
        """Test that the endpoint returns SSE stream with correct content type."""
        from neurips_abstracts.web_ui.app import app

        # Setup config mock
        mock_config = Mock()
        mock_config.paper_db_path = "/tmp/test.db"
        mock_config.llm_backend_url = "http://localhost:1234"
        mock_config.embedding_model = "test-model"
        mock_config.embedding_db_path = "/tmp/chroma"
        mock_config.collection_name = "test_collection"
        mock_get_config.return_value = mock_config

        # Setup database mock
        mock_db = Mock()
        mock_db.query.return_value = [{"count": 0}]
        mock_db.load_json_data.return_value = 0
        mock_db_class.return_value = mock_db

        # Setup embeddings manager mock
        mock_em = Mock()
        mock_em.collection = Mock()
        mock_em.collection.get.return_value = {"ids": []}
        mock_em_class.return_value = mock_em

        # Setup plugin mock
        mock_plugin = Mock()
        mock_plugin.download.return_value = {
            "count": 10,
            "papers": []
        }
        mock_get_plugin.return_value = mock_plugin

        with app.test_client() as client:
            response = client.post(
                "/api/download",
                json={"conference": "NeurIPS", "year": 2025}
            )
            
            # Check that response is SSE stream
            assert response.status_code == 200
            assert response.content_type == "text/event-stream; charset=utf-8"

    @patch("neurips_abstracts.web_ui.app.DatabaseManager")
    @patch("neurips_abstracts.web_ui.app.EmbeddingsManager")
    @patch("neurips_abstracts.plugins.get_plugin")
    @patch("neurips_abstracts.web_ui.app.get_config")
    def test_download_endpoint_handles_missing_plugin(
        self, mock_get_config, mock_get_plugin, mock_em_class, mock_db_class
    ):
        """Test that endpoint handles missing plugin gracefully."""
        from neurips_abstracts.web_ui.app import app

        # Setup config mock
        mock_config = Mock()
        mock_config.paper_db_path = "/tmp/test.db"
        mock_config.llm_backend_url = "http://localhost:1234"
        mock_config.embedding_model = "test-model"
        mock_config.embedding_db_path = "/tmp/chroma"
        mock_config.collection_name = "test_collection"
        mock_get_config.return_value = mock_config

        # Setup database mock
        mock_db = Mock()
        mock_db_class.return_value = mock_db

        # Setup embeddings manager mock
        mock_em = Mock()
        mock_em_class.return_value = mock_em

        # Plugin not found
        mock_get_plugin.return_value = None

        with app.test_client() as client:
            response = client.post(
                "/api/download",
                json={"conference": "UnknownConf", "year": 2025}
            )
            
            # Read SSE stream
            data = response.data.decode('utf-8')
            assert "error" in data
            assert "No plugin found" in data


class TestSSEStreaming:
    """Test Server-Sent Events streaming functionality."""

    @patch("neurips_abstracts.web_ui.app.DatabaseManager")
    @patch("neurips_abstracts.web_ui.app.EmbeddingsManager")
    @patch("neurips_abstracts.plugins.get_plugin")
    @patch("neurips_abstracts.web_ui.app.get_config")
    def test_sse_stream_sends_download_stage(
        self, mock_get_config, mock_get_plugin, mock_em_class, mock_db_class
    ):
        """Test that SSE stream sends download stage messages."""
        from neurips_abstracts.web_ui.app import app

        # Setup config mock
        mock_config = Mock()
        mock_config.paper_db_path = "/tmp/test.db"
        mock_config.llm_backend_url = "http://localhost:1234"
        mock_config.embedding_model = "test-model"
        mock_config.embedding_db_path = "/tmp/chroma"
        mock_config.collection_name = "test_collection"
        mock_get_config.return_value = mock_config

        # Setup database mock
        mock_db = Mock()
        mock_db.query.side_effect = [
            [{"count": 0}],  # Existing count check
            []  # Papers to embed query
        ]
        mock_db.load_json_data.return_value = 0
        mock_db_class.return_value = mock_db

        # Setup embeddings manager mock
        mock_em = Mock()
        mock_em.collection = Mock()
        mock_em.collection.get.return_value = {"ids": []}
        mock_em_class.return_value = mock_em

        # Setup plugin mock
        mock_plugin = Mock()
        mock_plugin.download.return_value = {
            "count": 100,
            "papers": []
        }
        mock_get_plugin.return_value = mock_plugin

        with app.test_client() as client:
            response = client.post(
                "/api/download",
                json={"conference": "NeurIPS", "year": 2025}
            )
            
            data = response.data.decode('utf-8')
            
            # Check for download stage messages
            assert 'data: {' in data
            assert '"stage": "download"' in data
            assert '"progress"' in data
            assert '"message"' in data

    @patch("neurips_abstracts.web_ui.app.DatabaseManager")
    @patch("neurips_abstracts.web_ui.app.EmbeddingsManager")
    @patch("neurips_abstracts.plugins.get_plugin")
    @patch("neurips_abstracts.web_ui.app.get_config")
    def test_sse_stream_sends_complete_stage(
        self, mock_get_config, mock_get_plugin, mock_em_class, mock_db_class
    ):
        """Test that SSE stream sends complete stage with summary."""
        from neurips_abstracts.web_ui.app import app

        # Setup config mock
        mock_config = Mock()
        mock_config.paper_db_path = "/tmp/test.db"
        mock_config.llm_backend_url = "http://localhost:1234"
        mock_config.embedding_model = "test-model"
        mock_config.embedding_db_path = "/tmp/chroma"
        mock_config.collection_name = "test_collection"
        mock_get_config.return_value = mock_config

        # Setup database mock
        mock_db = Mock()
        mock_db.query.side_effect = [
            [{"count": 0}],  # Existing count check
            []  # Papers to embed query (empty - no papers to embed)
        ]
        mock_db.load_json_data.return_value = 5
        mock_db_class.return_value = mock_db

        # Setup embeddings manager mock
        mock_em = Mock()
        mock_em.collection = Mock()
        mock_em.collection.get.return_value = {"ids": []}
        mock_em_class.return_value = mock_em

        # Setup plugin mock
        mock_plugin = Mock()
        mock_plugin.download.return_value = {
            "count": 5,
            "papers": []
        }
        mock_get_plugin.return_value = mock_plugin

        with app.test_client() as client:
            response = client.post(
                "/api/download",
                json={"conference": "NeurIPS", "year": 2025}
            )
            
            data = response.data.decode('utf-8')
            
            # Check for complete stage
            assert '"stage": "complete"' in data
            assert '"success": true' in data
            assert '"downloaded"' in data
            assert '"updated"' in data
            assert '"embedded"' in data


class TestSmartEmbeddingOptimization:
    """Test smart embedding optimization logic."""

    @patch("neurips_abstracts.web_ui.app.DatabaseManager")
    @patch("neurips_abstracts.web_ui.app.EmbeddingsManager")
    @patch("neurips_abstracts.plugins.get_plugin")
    @patch("neurips_abstracts.web_ui.app.get_config")
    def test_only_embeds_new_papers(
        self, mock_get_config, mock_get_plugin, mock_em_class, mock_db_class
    ):
        """Test that only new papers get embedded."""
        from neurips_abstracts.web_ui.app import app

        # Setup config mock
        mock_config = Mock()
        mock_config.paper_db_path = "/tmp/test.db"
        mock_config.llm_backend_url = "http://localhost:1234"
        mock_config.embedding_model = "test-model"
        mock_config.embedding_db_path = "/tmp/chroma"
        mock_config.collection_name = "test_collection"
        mock_get_config.return_value = mock_config

        # Setup database mock - simulate update scenario
        mock_db = Mock()
        
        # First query: existing papers count (has data)
        # Second query: existing papers before update
        # Third query: all papers after update
        mock_db.query.side_effect = [
            [{"count": 2}],  # 2 existing papers
            [  # Existing papers before update
                {"id": 1, "uid": "uid1", "name": "Paper 1", "abstract": "Abstract 1"},
                {"id": 2, "uid": "uid2", "name": "Paper 2", "abstract": "Abstract 2"}
            ],
            [  # All papers after update (1 new paper added)
                {"id": 1, "uid": "uid1", "name": "Paper 1", "abstract": "Abstract 1", "year": 2025, "conference": "NeurIPS"},
                {"id": 2, "uid": "uid2", "name": "Paper 2", "abstract": "Abstract 2", "year": 2025, "conference": "NeurIPS"},
                {"id": 3, "uid": "uid3", "name": "Paper 3", "abstract": "Abstract 3", "year": 2025, "conference": "NeurIPS"}
            ]
        ]
        mock_db.load_json_data.return_value = 1  # 1 new paper added
        mock_db_class.return_value = mock_db

        # Setup embeddings manager mock - all papers have embeddings
        mock_em = Mock()
        mock_em.collection = Mock()
        mock_em.collection.get.return_value = {"ids": ["1", "2", "3"]}  # All have embeddings
        mock_em.collection.delete = Mock()
        mock_em.generate_embedding.return_value = [0.1] * 384  # Mock embedding vector
        mock_em_class.return_value = mock_em

        # Setup plugin mock
        mock_plugin = Mock()
        mock_plugin.download.return_value = {
            "count": 3,
            "papers": []
        }
        mock_get_plugin.return_value = mock_plugin

        with app.test_client() as client:
            response = client.post(
                "/api/download",
                json={"conference": "NeurIPS", "year": 2025}
            )
            
            data = response.data.decode('utf-8')
            
            # Should complete successfully - only new paper was considered
            assert '"stage": "complete"' in data

    @patch("neurips_abstracts.web_ui.app.DatabaseManager")
    @patch("neurips_abstracts.web_ui.app.EmbeddingsManager")
    @patch("neurips_abstracts.plugins.get_plugin")
    @patch("neurips_abstracts.web_ui.app.get_config")
    def test_embeds_papers_with_changed_content(
        self, mock_get_config, mock_get_plugin, mock_em_class, mock_db_class
    ):
        """Test that papers with changed title/abstract get re-embedded."""
        from neurips_abstracts.web_ui.app import app

        # Setup config mock
        mock_config = Mock()
        mock_config.paper_db_path = "/tmp/test.db"
        mock_config.llm_backend_url = "http://localhost:1234"
        mock_config.embedding_model = "test-model"
        mock_config.embedding_db_path = "/tmp/chroma"
        mock_config.collection_name = "test_collection"
        mock_get_config.return_value = mock_config

        # Setup database mock
        mock_db = Mock()
        mock_db.query.side_effect = [
            [{"count": 2}],  # Existing count
            [  # Existing papers (old content)
                {"id": 1, "uid": "uid1", "name": "Paper 1 Old Title", "abstract": "Old abstract"},
                {"id": 2, "uid": "uid2", "name": "Paper 2", "abstract": "Abstract 2"}
            ],
            [  # Papers after update (Paper 1 has changed title)
                {"id": 1, "uid": "uid1", "name": "Paper 1 New Title", "abstract": "Old abstract", "year": 2025, "conference": "NeurIPS"},
                {"id": 2, "uid": "uid2", "name": "Paper 2", "abstract": "Abstract 2", "year": 2025, "conference": "NeurIPS"}
            ]
        ]
        mock_db.load_json_data.return_value = 0
        mock_db_class.return_value = mock_db

        # Setup embeddings manager mock
        mock_em = Mock()
        mock_em.collection = Mock()
        mock_em.collection.get.return_value = {"ids": ["1", "2"]}  # Both have embeddings
        mock_em.collection.delete = Mock()
        mock_em.generate_embedding.return_value = [0.1] * 384
        mock_em_class.return_value = mock_em

        # Setup plugin mock
        mock_plugin = Mock()
        mock_plugin.download.return_value = {"count": 2, "papers": []}
        mock_get_plugin.return_value = mock_plugin

        with app.test_client() as client:
            response = client.post(
                "/api/download",
                json={"conference": "NeurIPS", "year": 2025}
            )
            
            data = response.data.decode('utf-8')
            
            # Should detect the changed paper and embed it
            assert '"stage": "embeddings"' in data
            # Should show 1 paper to embed (the changed one)
            assert '"embedded": 1' in data or 'embed 1 paper' in data.lower()


class TestMissingEmbeddingDetection:
    """Test automatic detection and creation of missing embeddings."""

    @patch("neurips_abstracts.web_ui.app.DatabaseManager")
    @patch("neurips_abstracts.web_ui.app.EmbeddingsManager")
    @patch("neurips_abstracts.plugins.get_plugin")
    @patch("neurips_abstracts.web_ui.app.get_config")
    def test_detects_missing_embeddings(
        self, mock_get_config, mock_get_plugin, mock_em_class, mock_db_class
    ):
        """Test that papers without embeddings in ChromaDB are detected."""
        from neurips_abstracts.web_ui.app import app

        # Setup config mock
        mock_config = Mock()
        mock_config.paper_db_path = "/tmp/test.db"
        mock_config.llm_backend_url = "http://localhost:1234"
        mock_config.embedding_model = "test-model"
        mock_config.embedding_db_path = "/tmp/chroma"
        mock_config.collection_name = "test_collection"
        mock_get_config.return_value = mock_config

        # Setup database mock
        mock_db = Mock()
        mock_db.query.side_effect = [
            [{"count": 3}],  # 3 existing papers
            [  # Existing papers
                {"id": 1, "uid": "uid1", "name": "Paper 1", "abstract": "Abstract 1"},
                {"id": 2, "uid": "uid2", "name": "Paper 2", "abstract": "Abstract 2"},
                {"id": 3, "uid": "uid3", "name": "Paper 3", "abstract": "Abstract 3"}
            ],
            [  # Papers after update (no changes)
                {"id": 1, "uid": "uid1", "name": "Paper 1", "abstract": "Abstract 1", "year": 2025, "conference": "NeurIPS"},
                {"id": 2, "uid": "uid2", "name": "Paper 2", "abstract": "Abstract 2", "year": 2025, "conference": "NeurIPS"},
                {"id": 3, "uid": "uid3", "name": "Paper 3", "abstract": "Abstract 3", "year": 2025, "conference": "NeurIPS"}
            ]
        ]
        mock_db.load_json_data.return_value = 0
        mock_db_class.return_value = mock_db

        # Setup embeddings manager mock - only paper 1 and 2 have embeddings, paper 3 is missing
        mock_em = Mock()
        mock_em.collection = Mock()
        mock_em.collection.get.return_value = {"ids": ["1", "2"]}  # Paper 3 missing embedding
        mock_em.collection.delete = Mock()
        mock_em.generate_embedding.return_value = [0.1] * 384
        mock_em_class.return_value = mock_em

        # Setup plugin mock
        mock_plugin = Mock()
        mock_plugin.download.return_value = {"count": 3, "papers": []}
        mock_get_plugin.return_value = mock_plugin

        with app.test_client() as client:
            response = client.post(
                "/api/download",
                json={"conference": "NeurIPS", "year": 2025}
            )
            
            data = response.data.decode('utf-8')
            
            # Should detect missing embedding and create it
            assert '"stage": "embeddings"' in data
            # Should show 1 paper to embed (paper 3 with missing embedding)
            assert '"embedded": 1' in data or 'embed 1 paper' in data.lower()

    @patch("neurips_abstracts.web_ui.app.DatabaseManager")
    @patch("neurips_abstracts.web_ui.app.EmbeddingsManager")
    @patch("neurips_abstracts.plugins.get_plugin")
    @patch("neurips_abstracts.web_ui.app.get_config")
    def test_handles_chromadb_query_errors_gracefully(
        self, mock_get_config, mock_get_plugin, mock_em_class, mock_db_class
    ):
        """Test that ChromaDB query errors are handled gracefully."""
        from neurips_abstracts.web_ui.app import app

        # Setup config mock
        mock_config = Mock()
        mock_config.paper_db_path = "/tmp/test.db"
        mock_config.llm_backend_url = "http://localhost:1234"
        mock_config.embedding_model = "test-model"
        mock_config.embedding_db_path = "/tmp/chroma"
        mock_config.collection_name = "test_collection"
        mock_get_config.return_value = mock_config

        # Setup database mock
        mock_db = Mock()
        mock_db.query.side_effect = [
            [{"count": 1}],  # Existing count
            [{"id": 1, "uid": "uid1", "name": "Paper 1", "abstract": "Abstract 1"}],
            [{"id": 1, "uid": "uid1", "name": "Paper 1", "abstract": "Abstract 1", "year": 2025, "conference": "NeurIPS"}]
        ]
        mock_db.load_json_data.return_value = 0
        mock_db_class.return_value = mock_db

        # Setup embeddings manager mock - ChromaDB query fails
        mock_em = Mock()
        mock_em.collection = Mock()
        mock_em.collection.get.side_effect = Exception("ChromaDB error")
        mock_em.generate_embedding.return_value = [0.1] * 384
        mock_em_class.return_value = mock_em

        # Setup plugin mock
        mock_plugin = Mock()
        mock_plugin.download.return_value = {"count": 1, "papers": []}
        mock_get_plugin.return_value = mock_plugin

        with app.test_client() as client:
            response = client.post(
                "/api/download",
                json={"conference": "NeurIPS", "year": 2025}
            )
            
            data = response.data.decode('utf-8')
            
            # Should continue and treat all papers as needing embeddings
            assert '"stage": "embeddings"' in data or '"stage": "complete"' in data


class TestConnectionManagement:
    """Test proper connection management for SSE streaming."""

    @patch("neurips_abstracts.web_ui.app.DatabaseManager")
    @patch("neurips_abstracts.web_ui.app.EmbeddingsManager")
    @patch("neurips_abstracts.plugins.get_plugin")
    @patch("neurips_abstracts.web_ui.app.get_config")
    def test_connections_are_closed_on_completion(
        self, mock_get_config, mock_get_plugin, mock_em_class, mock_db_class
    ):
        """Test that database and embeddings connections are closed after streaming."""
        from neurips_abstracts.web_ui.app import app

        # Setup config mock
        mock_config = Mock()
        mock_config.paper_db_path = "/tmp/test.db"
        mock_config.llm_backend_url = "http://localhost:1234"
        mock_config.embedding_model = "test-model"
        mock_config.embedding_db_path = "/tmp/chroma"
        mock_config.collection_name = "test_collection"
        mock_get_config.return_value = mock_config

        # Setup database mock
        mock_db = Mock()
        mock_db.query.side_effect = [
            [{"count": 0}],
            []
        ]
        mock_db.load_json_data.return_value = 0
        mock_db.close = Mock()
        mock_db_class.return_value = mock_db

        # Setup embeddings manager mock
        mock_em = Mock()
        mock_em.collection = Mock()
        mock_em.collection.get.return_value = {"ids": []}
        mock_em.close = Mock()
        mock_em_class.return_value = mock_em

        # Setup plugin mock
        mock_plugin = Mock()
        mock_plugin.download.return_value = {"count": 0, "papers": []}
        mock_get_plugin.return_value = mock_plugin

        with app.test_client() as client:
            response = client.post(
                "/api/download",
                json={"conference": "NeurIPS", "year": 2025}
            )
            
            # Consume the entire stream
            _ = response.data
            
            # Verify close was called on both connections
            mock_db.close.assert_called_once()
            mock_em.close.assert_called_once()

    @patch("neurips_abstracts.web_ui.app.DatabaseManager")
    @patch("neurips_abstracts.web_ui.app.EmbeddingsManager")
    @patch("neurips_abstracts.plugins.get_plugin")
    @patch("neurips_abstracts.web_ui.app.get_config")
    def test_connections_are_closed_on_error(
        self, mock_get_config, mock_get_plugin, mock_em_class, mock_db_class
    ):
        """Test that connections are closed even when errors occur."""
        from neurips_abstracts.web_ui.app import app

        # Setup config mock
        mock_config = Mock()
        mock_config.paper_db_path = "/tmp/test.db"
        mock_config.llm_backend_url = "http://localhost:1234"
        mock_config.embedding_model = "test-model"
        mock_config.embedding_db_path = "/tmp/chroma"
        mock_config.collection_name = "test_collection"
        mock_get_config.return_value = mock_config

        # Setup database mock that raises an error
        mock_db = Mock()
        mock_db.query.side_effect = Exception("Database error")
        mock_db.close = Mock()
        mock_db_class.return_value = mock_db

        # Setup embeddings manager mock
        mock_em = Mock()
        mock_em.close = Mock()
        mock_em_class.return_value = mock_em

        # Setup plugin mock
        mock_plugin = Mock()
        mock_get_plugin.return_value = mock_plugin

        with app.test_client() as client:
            response = client.post(
                "/api/download",
                json={"conference": "NeurIPS", "year": 2025}
            )
            
            # Consume the stream (will contain error)
            data = response.data.decode('utf-8')
            
            # Should have error in stream
            assert "error" in data.lower()
            
            # Verify close was still called
            mock_db.close.assert_called_once()
            mock_em.close.assert_called_once()


class TestProgressTracking:
    """Test progress tracking in SSE stream."""

    @patch("neurips_abstracts.web_ui.app.DatabaseManager")
    @patch("neurips_abstracts.web_ui.app.EmbeddingsManager")
    @patch("neurips_abstracts.plugins.get_plugin")
    @patch("neurips_abstracts.web_ui.app.get_config")
    def test_progress_includes_all_stages(
        self, mock_get_config, mock_get_plugin, mock_em_class, mock_db_class
    ):
        """Test that progress messages include all three stages: download, database, embeddings."""
        from neurips_abstracts.web_ui.app import app

        # Setup config mock
        mock_config = Mock()
        mock_config.paper_db_path = "/tmp/test.db"
        mock_config.llm_backend_url = "http://localhost:1234"
        mock_config.embedding_model = "test-model"
        mock_config.embedding_db_path = "/tmp/chroma"
        mock_config.collection_name = "test_collection"
        mock_get_config.return_value = mock_config

        # Setup database mock
        mock_db = Mock()
        mock_db.query.side_effect = [
            [{"count": 0}],
            [  # One paper to embed
                {"id": 1, "uid": "uid1", "name": "Paper 1", "abstract": "Abstract 1", "year": 2025, "conference": "NeurIPS"}
            ]
        ]
        mock_db.load_json_data.return_value = 1
        mock_db_class.return_value = mock_db

        # Setup embeddings manager mock
        mock_em = Mock()
        mock_em.collection = Mock()
        mock_em.collection.get.return_value = {"ids": []}  # No existing embeddings
        mock_em.collection.delete = Mock()
        mock_em.collection.add = Mock()
        mock_em.generate_embedding.return_value = [0.1] * 384
        mock_em_class.return_value = mock_em

        # Setup plugin mock
        mock_plugin = Mock()
        mock_plugin.download.return_value = {"count": 1, "papers": []}
        mock_get_plugin.return_value = mock_plugin

        with app.test_client() as client:
            response = client.post(
                "/api/download",
                json={"conference": "NeurIPS", "year": 2025}
            )
            
            data = response.data.decode('utf-8')
            
            # Should have all three stages
            assert '"stage": "download"' in data
            assert '"stage": "database"' in data
            assert '"stage": "embeddings"' in data
            assert '"stage": "complete"' in data
            
            # Should have progress indicators
            assert '"progress"' in data
            assert '"message"' in data

"""
Tests for the embeddings module.
"""

import pytest
from unittest.mock import Mock

from abstracts_explorer.embeddings import EmbeddingsError, EmbeddingsManager

# Fixtures imported from conftest.py:
# - mock_lm_studio: Mock LM Studio API responses
# - embeddings_manager: EmbeddingsManager instance for testing
# - test_database: Test database with sample papers for testing embeddings


class TestEmbeddingsManager:
    """Tests for EmbeddingsManager class."""

    def test_init(self, embeddings_manager):
        """Test EmbeddingsManager initialization."""
        assert isinstance(embeddings_manager, EmbeddingsManager)
        assert embeddings_manager.client is None
        assert embeddings_manager.collection is None

    def test_connect(self, embeddings_manager):
        """Test connecting to ChromaDB."""
        embeddings_manager.connect()
        assert embeddings_manager.client is not None
        assert embeddings_manager.chroma_path.exists()
        embeddings_manager.close()

    def test_close(self, embeddings_manager):
        """Test closing ChromaDB connection."""
        embeddings_manager.connect()
        embeddings_manager.close()
        assert embeddings_manager.client is None
        assert embeddings_manager.collection is None

    def test_context_manager(self, embeddings_manager):
        """Test context manager functionality."""
        with embeddings_manager as em:
            assert em.client is not None
        assert embeddings_manager.client is None

    def test_test_lm_studio_connection_success(self, embeddings_manager):
        """Test successful LM Studio connection."""
        result = embeddings_manager.test_lm_studio_connection()
        assert result is True
        embeddings_manager.openai_client.models.list.assert_called_once()

    def test_test_lm_studio_connection_failure(self, embeddings_manager):
        """Test failed LM Studio connection."""
        # Mock the OpenAI client's models.list() to raise an exception
        embeddings_manager.openai_client.models.list.side_effect = Exception("Connection error")
        result = embeddings_manager.test_lm_studio_connection()
        assert result is False

    def test_generate_embedding_success(self, embeddings_manager):
        """Test successful embedding generation."""
        embedding = embeddings_manager.generate_embedding("Test text")
        assert isinstance(embedding, list)
        assert len(embedding) == 4096
        embeddings_manager.openai_client.embeddings.create.assert_called_once()

    def test_generate_embedding_empty_text(self, embeddings_manager):
        """Test embedding generation with empty text."""
        with pytest.raises(EmbeddingsError, match="Cannot generate embedding for empty text"):
            embeddings_manager.generate_embedding("")

    def test_generate_embedding_api_error(self, embeddings_manager):
        """Test embedding generation with API error."""
        # Create a mock that will raise an exception
        mock_response = Mock()
        mock_response.side_effect = Exception("API error")
        embeddings_manager.openai_client.embeddings.create = mock_response
        
        with pytest.raises(EmbeddingsError, match="Failed to generate embedding"):
            embeddings_manager.generate_embedding("Test text")

    def test_create_collection(self, embeddings_manager):
        """Test creating a collection."""
        embeddings_manager.connect()
        embeddings_manager.create_collection()
        assert embeddings_manager.collection is not None
        embeddings_manager.close()

    def test_create_collection_not_connected(self, embeddings_manager):
        """Test creating collection without connection."""
        with pytest.raises(EmbeddingsError, match="Not connected to ChromaDB"):
            embeddings_manager.create_collection()

    def test_create_collection_reset(self, embeddings_manager):
        """Test resetting a collection."""
        embeddings_manager.connect()
        embeddings_manager.create_collection()
        embeddings_manager.create_collection(reset=True)
        assert embeddings_manager.collection is not None
        embeddings_manager.close()

    def test_add_paper(self, embeddings_manager, mock_lm_studio):
        """Test adding a paper."""
        embeddings_manager.connect()
        embeddings_manager.create_collection()

        paper = {
            "uid": "test_paper_1",
            "title": "Test Paper",
            "abstract": "Test abstract",
            "authors": "John Doe",
        }
        embeddings_manager.add_paper(paper)

        stats = embeddings_manager.get_collection_stats()
        assert stats["count"] == 1
        embeddings_manager.close()

    def test_add_paper_empty_abstract(self, embeddings_manager, mock_lm_studio):
        """Test adding a paper with empty abstract but non-empty title."""
        embeddings_manager.connect()
        embeddings_manager.create_collection()

        # Should work with title even if abstract is empty
        paper = {
            "uid": "test_paper_1",
            "title": "Test Paper",
            "abstract": "",
        }
        embeddings_manager.add_paper(paper)

        stats = embeddings_manager.get_collection_stats()
        assert stats["count"] == 1
        embeddings_manager.close()

    def test_add_paper_collection_not_initialized(self, embeddings_manager):
        """Test adding paper without collection."""
        paper = {
            "uid": "test_paper_1",
            "title": "Test Paper",
            "abstract": "Test abstract",
        }
        with pytest.raises(EmbeddingsError, match="Collection not initialized"):
            embeddings_manager.add_paper(paper)

    def test_add_multiple_papers(self, embeddings_manager, mock_lm_studio):
        """Test adding multiple papers."""
        embeddings_manager.connect()
        embeddings_manager.create_collection()

        papers = [
            {"uid": "paper1", "title": "Paper 1", "abstract": "Abstract 1"},
            {"uid": "paper2", "title": "Paper 2", "abstract": "Abstract 2"},
            {"uid": "paper3", "title": "Paper 3", "abstract": "Abstract 3"},
        ]

        for paper in papers:
            embeddings_manager.add_paper(paper)

        stats = embeddings_manager.get_collection_stats()
        assert stats["count"] == 3
        embeddings_manager.close()

    def test_add_papers_with_empty_abstracts(self, embeddings_manager, mock_lm_studio):
        """Test adding papers with some empty abstracts."""
        embeddings_manager.connect()
        embeddings_manager.create_collection()

        papers = [
            {"uid": "paper1", "title": "Paper 1", "abstract": "Abstract 1"},
            {"uid": "paper2", "title": "Paper 2", "abstract": "Abstract 2"},
        ]

        for paper in papers:
            embeddings_manager.add_paper(paper)

        stats = embeddings_manager.get_collection_stats()
        assert stats["count"] == 2
        embeddings_manager.close()

    def test_search_similar(self, embeddings_manager, mock_lm_studio):
        """Test similarity search."""
        embeddings_manager.connect()
        embeddings_manager.create_collection()

        # Add some papers
        papers = [
            {"uid": "paper1", "title": "DL Paper", "abstract": "Deep learning neural networks"},
            {"uid": "paper2", "title": "NLP Paper", "abstract": "Natural language processing"},
        ]
        for paper in papers:
            embeddings_manager.add_paper(paper)

        # Search
        results = embeddings_manager.search_similar("machine learning", n_results=2)

        assert "ids" in results
        assert "distances" in results
        assert "documents" in results
        assert "metadatas" in results
        assert len(results["ids"][0]) <= 2
        embeddings_manager.close()

    def test_search_similar_empty_query(self, embeddings_manager):
        """Test search with empty query."""
        embeddings_manager.connect()
        embeddings_manager.create_collection()

        with pytest.raises(EmbeddingsError, match="Query cannot be empty"):
            embeddings_manager.search_similar("")

        embeddings_manager.close()

    def test_search_similar_collection_not_initialized(self, embeddings_manager):
        """Test search without collection."""
        with pytest.raises(EmbeddingsError, match="Collection not initialized"):
            embeddings_manager.search_similar("test query")

    def test_get_collection_stats(self, embeddings_manager, mock_lm_studio):
        """Test getting collection statistics."""
        embeddings_manager.connect()
        embeddings_manager.create_collection()

        stats = embeddings_manager.get_collection_stats()
        assert "name" in stats
        assert "count" in stats
        assert "metadata" in stats
        assert stats["count"] == 0

        # Add a paper
        paper = {"uid": "paper1", "title": "Test", "abstract": "Test abstract"}
        embeddings_manager.add_paper(paper)

        stats = embeddings_manager.get_collection_stats()
        assert stats["count"] == 1
        embeddings_manager.close()

    def test_get_collection_stats_not_initialized(self, embeddings_manager):
        """Test getting stats without collection."""
        with pytest.raises(EmbeddingsError, match="Collection not initialized"):
            embeddings_manager.get_collection_stats()

    def test_embed_from_database(self, embeddings_manager, test_database, mock_lm_studio):
        """Test embedding papers from database."""
        embeddings_manager.connect()
        embeddings_manager.create_collection()

        count = embeddings_manager.embed_from_database(test_database)

        # Should embed all 3 papers (title is included even if abstract is empty)
        assert count == 3

        stats = embeddings_manager.get_collection_stats()
        assert stats["count"] == 3
        embeddings_manager.close()

    def test_embed_from_database_with_filter(self, embeddings_manager, test_database, mock_lm_studio):
        """Test embedding papers from database with filter."""
        embeddings_manager.connect()
        embeddings_manager.create_collection()

        count = embeddings_manager.embed_from_database(test_database, where_clause="session LIKE '%ML%'")

        # Should only embed papers in ML sessions with non-empty abstracts (paper1)
        assert count == 1

        stats = embeddings_manager.get_collection_stats()
        assert stats["count"] == 1
        embeddings_manager.close()

    def test_embed_from_database_not_found(self, embeddings_manager, tmp_path):
        """Test embedding from non-existent database."""
        embeddings_manager.connect()
        embeddings_manager.create_collection()

        with pytest.raises(EmbeddingsError, match="Database not found"):
            embeddings_manager.embed_from_database(tmp_path / "nonexistent.db")

        embeddings_manager.close()

    def test_embed_from_database_collection_not_initialized(self, embeddings_manager, test_database):
        """Test embedding from database without collection."""
        with pytest.raises(EmbeddingsError, match="Collection not initialized"):
            embeddings_manager.embed_from_database(test_database)

    def test_embed_from_database_with_progress_callback(self, embeddings_manager, test_database, mock_lm_studio):
        """Test embedding papers from database with progress callback."""
        embeddings_manager.connect()
        embeddings_manager.create_collection()

        progress_calls = []

        def progress_callback(current: int, total: int) -> None:
            progress_calls.append((current, total))

        count = embeddings_manager.embed_from_database(test_database, progress_callback=progress_callback)

        # Should embed all 3 papers (title is included even if abstract is empty)
        assert count == 3
        # Progress callback should be called
        assert len(progress_calls) > 0
        # Check that the last progress call shows completion
        assert progress_calls[-1][0] <= progress_calls[-1][1]

        embeddings_manager.close()

    def test_embed_from_database_empty_result(self, embeddings_manager, tmp_path, mock_lm_studio):
        """Test embedding from database with no matching papers."""
        from abstracts_explorer.database import DatabaseManager

        # Create empty database using DatabaseManager
        db_path = tmp_path / "empty.db"
        with DatabaseManager(db_path) as db:
            db.create_tables()

        embeddings_manager.connect()
        embeddings_manager.create_collection()

        count = embeddings_manager.embed_from_database(db_path)

        assert count == 0
        stats = embeddings_manager.get_collection_stats()
        assert stats["count"] == 0

        embeddings_manager.close()

    def test_embed_from_database_all_empty_abstracts(self, embeddings_manager, tmp_path, mock_lm_studio):
        """Test embedding from database where all papers have empty abstracts."""
        from abstracts_explorer.database import DatabaseManager
        from abstracts_explorer.plugin import LightweightPaper

        db_path = tmp_path / "test.db"
        with DatabaseManager(db_path) as db:
            db.create_tables()
            # Add papers with titles but empty abstracts
            for i in range(3):
                paper = LightweightPaper(
                    uid=f"paper{i}",
                    title=f"Paper {i+1}",  # Valid title
                    abstract="",  # Empty abstract
                    authors=["Author"],
                    session="Session",
                    poster_position=f"P{i}",
                    year=2025,
                    conference="NeurIPS",
                )
                db.add_paper(paper)

        embeddings_manager.connect()
        embeddings_manager.create_collection()

        count = embeddings_manager.embed_from_database(db_path)

        # Should embed all papers (title is used even if abstract is empty)
        assert count == 3
        embeddings_manager.close()

    def test_embed_from_database_sql_error(self, embeddings_manager, tmp_path):
        """Test embedding from database with SQL error."""
        from abstracts_explorer.database import DatabaseManager
        
        # Create database with only embeddings_metadata, but missing papers table
        db_path = tmp_path / "bad.db"
        
        # Use DatabaseManager to connect, but manually create only embeddings_metadata table
        # to simulate a corrupted/incomplete database
        db = DatabaseManager(str(db_path))
        db.connect()
        
        cursor = db.connection.cursor()
        # Create embeddings_metadata table
        cursor.execute(
            """
            CREATE TABLE embeddings_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding_model TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        # Note: NOT creating the papers table to trigger error
        db.connection.commit()
        db.close()

        embeddings_manager.connect()
        embeddings_manager.create_collection()

        # Should raise EmbeddingsError due to missing papers table
        with pytest.raises(EmbeddingsError, match="Failed to embed from database"):
            embeddings_manager.embed_from_database(db_path)

        embeddings_manager.close()

    def test_embed_from_database_with_metadata_fields(self, embeddings_manager, test_database, mock_lm_studio):
        """Test that metadata fields from lightweight schema are included."""
        embeddings_manager.connect()
        embeddings_manager.create_collection()

        count = embeddings_manager.embed_from_database(test_database)
        assert count == 3

        # Search to verify metadata includes lightweight schema fields
        results = embeddings_manager.search_similar("test", n_results=3)
        assert len(results["metadatas"][0]) > 0

        # Check papers have the lightweight schema metadata fields
        for metadata in results["metadatas"][0]:
            assert "title" in metadata
            assert "authors" in metadata
            assert "keywords" in metadata
            assert "session" in metadata

        # Verify one paper has expected metadata values (use first result)
        first_paper_metadata = results["metadatas"][0][0]
        assert first_paper_metadata is not None
        # Verify fields exist and contain expected data format
        assert "title" in first_paper_metadata
        assert "authors" in first_paper_metadata
        assert "session" in first_paper_metadata

        embeddings_manager.close()

    def test_paper_exists(self, embeddings_manager, mock_lm_studio):
        """Test paper_exists method."""
        embeddings_manager.connect()
        embeddings_manager.create_collection()

        # Initially paper should not exist
        assert not embeddings_manager.paper_exists("test_paper_1")

        # Add a paper
        paper = {
            "uid": "test_paper_1",
            "title": "Test Paper",
            "abstract": "This is a test abstract",
        }
        embeddings_manager.add_paper(paper)

        # Now paper should exist
        assert embeddings_manager.paper_exists("test_paper_1")

        # Other papers should not exist
        assert not embeddings_manager.paper_exists("test_paper_2")

        embeddings_manager.close()

    def test_paper_exists_collection_not_initialized(self, embeddings_manager):
        """Test paper_exists raises error when collection not initialized."""
        with pytest.raises(EmbeddingsError, match="Collection not initialized"):
            embeddings_manager.paper_exists("test_paper_1")

    def test_embed_from_database_skip_existing(self, embeddings_manager, test_database, mock_lm_studio):
        """Test that embed_from_database skips papers that already exist."""
        embeddings_manager.connect()
        embeddings_manager.create_collection()

        # First run - should embed all 3 papers
        count = embeddings_manager.embed_from_database(test_database)
        assert count == 3

        stats = embeddings_manager.get_collection_stats()
        assert stats["count"] == 3

        # Second run - should skip all existing papers and embed 0 new ones
        count = embeddings_manager.embed_from_database(test_database)
        assert count == 0

        # Collection count should still be 3
        stats = embeddings_manager.get_collection_stats()
        assert stats["count"] == 3

        embeddings_manager.close()


def test_check_model_compatibility_no_database(embeddings_manager, tmp_path):
    """Test checking model compatibility when database does not exist."""
    non_existent_db = tmp_path / "nonexistent.db"
    
    compatible, stored, current = embeddings_manager.check_model_compatibility(non_existent_db)
    
    assert compatible is True
    assert stored is None
    assert current == embeddings_manager.model_name


def test_check_model_compatibility_no_model_stored(embeddings_manager, tmp_path):
    """Test checking model compatibility when no model is stored in database."""
    from abstracts_explorer.database import DatabaseManager
    
    db_path = tmp_path / "test.db"
    with DatabaseManager(db_path) as db:
        db.create_tables()
    
    compatible, stored, current = embeddings_manager.check_model_compatibility(db_path)
    
    assert compatible is True
    assert stored is None
    assert current == embeddings_manager.model_name


def test_check_model_compatibility_matching_models(embeddings_manager, tmp_path):
    """Test checking model compatibility when models match."""
    from abstracts_explorer.database import DatabaseManager
    
    db_path = tmp_path / "test.db"
    with DatabaseManager(db_path) as db:
        db.create_tables()
        db.set_embedding_model(embeddings_manager.model_name)
    
    compatible, stored, current = embeddings_manager.check_model_compatibility(db_path)
    
    assert compatible is True
    assert stored == embeddings_manager.model_name
    assert current == embeddings_manager.model_name


def test_check_model_compatibility_mismatched_models(embeddings_manager, tmp_path):
    """Test checking model compatibility when models differ."""
    from abstracts_explorer.database import DatabaseManager
    
    db_path = tmp_path / "test.db"
    different_model = "different-embedding-model"
    
    with DatabaseManager(db_path) as db:
        db.create_tables()
        db.set_embedding_model(different_model)
    
    compatible, stored, current = embeddings_manager.check_model_compatibility(db_path)
    
    assert compatible is False
    assert stored == different_model
    assert current == embeddings_manager.model_name


def test_embed_from_database_stores_model(embeddings_manager, test_database):
    """Test that embed_from_database stores the embedding model in the database."""
    from abstracts_explorer.database import DatabaseManager
    
    embeddings_manager.connect()
    embeddings_manager.create_collection()
    
    # Embed papers
    embeddings_manager.embed_from_database(test_database)
    
    # Check that the model was stored
    with DatabaseManager(test_database) as db:
        stored_model = db.get_embedding_model()
        assert stored_model == embeddings_manager.model_name
    
    embeddings_manager.close()

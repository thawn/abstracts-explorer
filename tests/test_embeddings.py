"""
Tests for the embeddings module.
"""

import pytest
from unittest.mock import Mock, patch

from abstracts_explorer.embeddings import EmbeddingsError, EmbeddingsManager
from tests.conftest import set_test_db

# Fixtures imported from conftest.py:
# - mock_lm_studio: Mock LM Studio API responses
# - embeddings_manager: EmbeddingsManager instance for testing
# - test_database: Test database with sample papers for testing embeddings


class TestEmbeddingsManager:
    """Tests for EmbeddingsManager class."""

    def test_init(self, embeddings_manager):
        """Test EmbeddingsManager initialization."""
        assert isinstance(embeddings_manager, EmbeddingsManager)
        assert embeddings_manager._client is None
        assert embeddings_manager._collection is None

    def test_connect(self, embeddings_manager):
        """Test connecting to ChromaDB."""
        embeddings_manager.connect()
        assert embeddings_manager.client is not None
        # Check that embedding_db path exists (only for local paths)
        if not embeddings_manager.embedding_db.startswith("http"):
            from pathlib import Path

            assert Path(embeddings_manager.embedding_db).exists()
        embeddings_manager.close()

    def test_close(self, embeddings_manager):
        """Test closing ChromaDB connection."""
        embeddings_manager.connect()
        embeddings_manager.close()
        assert embeddings_manager._client is None
        assert embeddings_manager._collection is None

    def test_context_manager(self, embeddings_manager):
        """Test context manager functionality."""
        with embeddings_manager as em:
            assert em._client is not None
        assert embeddings_manager._client is None

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

    def test_create_collection_auto_connects(self, embeddings_manager):
        """Test that create_collection auto-connects to ChromaDB if not already connected."""
        assert embeddings_manager._client is None
        embeddings_manager.create_collection()
        assert embeddings_manager._client is not None
        assert embeddings_manager._collection is not None
        embeddings_manager.close()

    def test_create_collection_reset(self, embeddings_manager):
        """Test resetting a collection."""
        embeddings_manager.connect()
        embeddings_manager.create_collection()
        embeddings_manager.create_collection(reset=True)
        assert embeddings_manager.collection is not None
        embeddings_manager.close()

    def test_delete_embeddings_by_filter_no_filter_raises(self, embeddings_manager):
        """delete_embeddings_by_filter with no args raises ValueError."""
        embeddings_manager.connect()
        embeddings_manager.create_collection()
        with pytest.raises(ValueError):
            embeddings_manager.delete_embeddings_by_filter()
        embeddings_manager.close()

    def test_delete_embeddings_by_filter_conference(self, embeddings_manager, mock_lm_studio):
        """delete_embeddings_by_filter(conference=...) removes only matching papers."""
        embeddings_manager.connect()
        embeddings_manager.create_collection()

        papers = [
            {"uid": "p1", "title": "NeurIPS Paper 1", "abstract": "A", "conference": "NeurIPS", "year": 2024},
            {"uid": "p2", "title": "NeurIPS Paper 2", "abstract": "B", "conference": "NeurIPS", "year": 2024},
            {"uid": "p3", "title": "ICLR Paper", "abstract": "C", "conference": "ICLR", "year": 2024},
        ]
        for paper in papers:
            embeddings_manager.add_paper(paper)

        assert embeddings_manager.collection.count() == 3

        deleted = embeddings_manager.delete_embeddings_by_filter(conference="NeurIPS")
        assert deleted == 2
        assert embeddings_manager.collection.count() == 1

        # Remaining paper should be the ICLR one
        remaining = embeddings_manager.collection.get()
        assert remaining["ids"] == ["p3"]
        embeddings_manager.close()

    def test_delete_embeddings_by_filter_year(self, embeddings_manager, mock_lm_studio):
        """delete_embeddings_by_filter(year=...) removes only papers from that year."""
        embeddings_manager.connect()
        embeddings_manager.create_collection()

        papers = [
            {"uid": "p1", "title": "Paper 2024", "abstract": "A", "conference": "NeurIPS", "year": 2024},
            {"uid": "p2", "title": "Paper 2025", "abstract": "B", "conference": "NeurIPS", "year": 2025},
        ]
        for paper in papers:
            embeddings_manager.add_paper(paper)

        deleted = embeddings_manager.delete_embeddings_by_filter(year=2024)
        assert deleted == 1
        assert embeddings_manager.collection.count() == 1
        remaining = embeddings_manager.collection.get()
        assert remaining["ids"] == ["p2"]
        embeddings_manager.close()

    def test_delete_embeddings_by_filter_conference_and_year(self, embeddings_manager, mock_lm_studio):
        """delete_embeddings_by_filter(conference=..., year=...) scopes to exact slice."""
        embeddings_manager.connect()
        embeddings_manager.create_collection()

        papers = [
            {"uid": "p1", "title": "NeurIPS 2024", "abstract": "A", "conference": "NeurIPS", "year": 2024},
            {"uid": "p2", "title": "NeurIPS 2025", "abstract": "B", "conference": "NeurIPS", "year": 2025},
            {"uid": "p3", "title": "ICLR 2024", "abstract": "C", "conference": "ICLR", "year": 2024},
        ]
        for paper in papers:
            embeddings_manager.add_paper(paper)

        deleted = embeddings_manager.delete_embeddings_by_filter(conference="NeurIPS", year=2024)
        assert deleted == 1
        assert embeddings_manager.collection.count() == 2
        remaining_ids = set(embeddings_manager.collection.get()["ids"])
        assert "p1" not in remaining_ids
        assert {"p2", "p3"} == remaining_ids
        embeddings_manager.close()

    def test_delete_embeddings_by_filter_nothing_to_delete(self, embeddings_manager, mock_lm_studio):
        """delete_embeddings_by_filter returns 0 when no matching embeddings exist."""
        embeddings_manager.connect()
        embeddings_manager.create_collection()

        papers = [
            {"uid": "p1", "title": "NeurIPS Paper", "abstract": "A", "conference": "NeurIPS", "year": 2024},
        ]
        for paper in papers:
            embeddings_manager.add_paper(paper)

        deleted = embeddings_manager.delete_embeddings_by_filter(conference="ICLR")
        assert deleted == 0
        assert embeddings_manager.collection.count() == 1
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

    def test_add_paper_auto_initializes(self, embeddings_manager):
        """Test that add_paper auto-initializes client and collection if needed."""
        paper = {
            "uid": "test_paper_1",
            "title": "Test Paper",
            "abstract": "Test abstract",
        }
        # Should not raise - auto-initializes ChromaDB connection and collection
        embeddings_manager.add_paper(paper)
        assert embeddings_manager._client is not None
        assert embeddings_manager._collection is not None
        embeddings_manager.close()

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

        # Add some papers (all required LightweightPaper fields needed for metadata parsing)
        papers = [
            {
                "uid": "paper1",
                "title": "DL Paper",
                "abstract": "Deep learning neural networks",
                "authors": "Alice;Bob",
                "session": "ML Track",
                "poster_position": "1",
                "year": "2024",
                "conference": "NeurIPS",
            },
            {
                "uid": "paper2",
                "title": "NLP Paper",
                "abstract": "Natural language processing",
                "authors": "Charlie",
                "session": "NLP Track",
                "poster_position": "2",
                "year": "2024",
                "conference": "NeurIPS",
            },
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

    def test_search_similar_auto_initializes(self, embeddings_manager, mock_lm_studio):
        """Test that search_similar auto-initializes client and collection if needed."""
        # Should not raise - auto-initializes (collection is empty so 0 results)
        results = embeddings_manager.search_similar("test query", n_results=1)
        assert embeddings_manager._client is not None
        assert embeddings_manager._collection is not None
        assert "ids" in results
        embeddings_manager.close()

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

    def test_get_collection_stats_auto_initializes(self, embeddings_manager):
        """Test that get_collection_stats auto-initializes client and collection if needed."""
        stats = embeddings_manager.get_collection_stats()
        assert embeddings_manager._client is not None
        assert embeddings_manager._collection is not None
        assert "name" in stats
        assert "count" in stats
        assert stats["count"] == 0
        embeddings_manager.close()

    def test_embed_from_database(self, embeddings_manager, test_database, mock_lm_studio):
        """Test embedding papers from database."""
        embeddings_manager.connect()
        embeddings_manager.create_collection()

        count = embeddings_manager.embed_from_database()

        # Should embed all 3 papers (title is included even if abstract is empty)
        assert count == 3

        stats = embeddings_manager.get_collection_stats()
        assert stats["count"] == 3
        embeddings_manager.close()

    def test_embed_from_database_with_filter(self, embeddings_manager, test_database, mock_lm_studio):
        """Test embedding papers from database with filter."""
        embeddings_manager.connect()
        embeddings_manager.create_collection()

        count = embeddings_manager.embed_from_database(where_clause="session LIKE '%ML%'")

        # Should only embed papers in ML sessions with non-empty abstracts (paper1)
        assert count == 1

        stats = embeddings_manager.get_collection_stats()
        assert stats["count"] == 1
        embeddings_manager.close()

    def test_embed_from_database_not_found(self, embeddings_manager, tmp_path):
        """Test embedding from non-existent database."""

        # Set PAPER_DB to a nonexistent database
        nonexistent_db = tmp_path / "nonexistent.db"
        set_test_db(nonexistent_db)

        embeddings_manager.connect()
        embeddings_manager.create_collection()

        # The error message will now be about SQL syntax, not "database not found"
        # because the database_url is constructed and the error happens during query
        with pytest.raises(EmbeddingsError, match="Failed to embed from database"):
            embeddings_manager.embed_from_database()

        embeddings_manager.close()

    def test_embed_from_database_auto_initializes(self, embeddings_manager, test_database):
        """Test that embed_from_database auto-initializes client and collection if needed."""
        # Should not raise - auto-initializes ChromaDB connection and collection
        count = embeddings_manager.embed_from_database()
        assert embeddings_manager._client is not None
        assert embeddings_manager._collection is not None
        assert count == 3
        embeddings_manager.close()

    def test_embed_from_database_with_progress_callback(self, embeddings_manager, test_database, mock_lm_studio):
        """Test embedding papers from database with progress callback."""
        embeddings_manager.connect()
        embeddings_manager.create_collection()

        progress_calls = []

        def progress_callback(current: int, total: int) -> None:
            progress_calls.append((current, total))

        count = embeddings_manager.embed_from_database(progress_callback=progress_callback)

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
        set_test_db(db_path)
        with DatabaseManager() as db:
            db.create_tables()

        embeddings_manager.connect()
        embeddings_manager.create_collection()

        count = embeddings_manager.embed_from_database()

        assert count == 0
        stats = embeddings_manager.get_collection_stats()
        assert stats["count"] == 0

        embeddings_manager.close()

    def test_embed_from_database_all_empty_abstracts(self, embeddings_manager, tmp_path, mock_lm_studio):
        """Test embedding from database where all papers have empty abstracts.

        Empty-abstract papers are rejected by LightweightPaper validation but may
        still exist in legacy databases before the purge script has been run.  The
        embeddings manager must handle them gracefully (title is used as fallback).
        """
        from abstracts_explorer.database import DatabaseManager
        from abstracts_explorer.db_models import Paper
        from sqlalchemy import insert as sa_insert

        db_path = tmp_path / "test.db"
        set_test_db(db_path)
        with DatabaseManager() as db:
            db.create_tables()
            # Directly insert papers with empty abstracts (bypassing LightweightPaper
            # validation to simulate legacy data that predates the abstract check).
            for i in range(3):
                db._session.execute(
                    sa_insert(Paper).values(
                        uid=f"emptyabs{i:04d}",
                        title=f"Paper {i + 1}",
                        abstract="",
                        authors="Author",
                        session="Session",
                        poster_position=f"P{i}",
                        year=2025,
                        conference="NeurIPS",
                    )
                )
            db._session.commit()

        embeddings_manager.connect()
        embeddings_manager.create_collection()

        count = embeddings_manager.embed_from_database()

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
        set_test_db(db_path)
        db = DatabaseManager()
        db.connect()

        cursor = db.connection.cursor()
        # Create embeddings_metadata table
        cursor.execute("""
            CREATE TABLE embeddings_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding_model TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Note: NOT creating the papers table to trigger error
        db.connection.commit()
        db.close()

        embeddings_manager.connect()
        embeddings_manager.create_collection()

        # Should raise EmbeddingsError due to missing papers table
        with pytest.raises(EmbeddingsError, match="Failed to embed from database"):
            embeddings_manager.embed_from_database()

        embeddings_manager.close()

    def test_embed_from_database_with_metadata_fields(self, embeddings_manager, test_database, mock_lm_studio):
        """Test that metadata fields from lightweight schema are included."""
        embeddings_manager.connect()
        embeddings_manager.create_collection()

        count = embeddings_manager.embed_from_database()
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

    def test_paper_exists_auto_initializes(self, embeddings_manager):
        """Test that paper_exists auto-initializes client and collection if needed."""
        # Should not raise - auto-initializes (paper doesn't exist yet)
        result = embeddings_manager.paper_exists("test_paper_1")
        assert result is False
        assert embeddings_manager._client is not None
        assert embeddings_manager._collection is not None
        embeddings_manager.close()

    def test_embed_from_database_skip_existing(self, embeddings_manager, test_database, mock_lm_studio):
        """Test that embed_from_database skips papers that already exist."""
        embeddings_manager.connect()
        embeddings_manager.create_collection()

        # First run - should embed all 3 papers
        count = embeddings_manager.embed_from_database()
        assert count == 3

        stats = embeddings_manager.get_collection_stats()
        assert stats["count"] == 3

        # Second run - should skip all existing papers and embed 0 new ones
        count = embeddings_manager.embed_from_database()
        assert count == 0

        # Collection count should still be 3
        stats = embeddings_manager.get_collection_stats()
        assert stats["count"] == 3

        embeddings_manager.close()

    def test_paper_needs_update_returns_true_on_error(self, embeddings_manager, mock_lm_studio):
        """Test that paper_needs_update returns True when an error occurs.

        When paper_needs_update encounters an exception it should return True
        (assume update needed) rather than False, so the paper is re-embedded
        instead of silently skipped.
        """
        from unittest.mock import patch

        embeddings_manager.connect()
        embeddings_manager.create_collection()

        paper = {
            "uid": "test_paper_err",
            "title": "Test Paper",
            "abstract": "Test abstract",
        }

        # Simulate an exception during the collection.get() call
        with patch.object(embeddings_manager.collection, "get", side_effect=RuntimeError("db error")):
            result = embeddings_manager.paper_needs_update(paper)

        # Should return True so the paper is embedded rather than skipped
        assert result is True

        embeddings_manager.close()


def test_check_model_compatibility_no_database(embeddings_manager, tmp_path):
    """Test checking model compatibility when database does not exist."""

    non_existent_db = tmp_path / "nonexistent.db"
    set_test_db(non_existent_db)

    # Since the database doesn't exist, this will raise an error when trying to connect
    # The behavior has changed - we no longer check if db exists before connecting
    from abstracts_explorer.embeddings import EmbeddingsError

    with pytest.raises(EmbeddingsError, match="Failed to check model compatibility"):
        embeddings_manager.check_model_compatibility()


def test_check_model_compatibility_no_model_stored(embeddings_manager, tmp_path):
    """Test checking model compatibility when no model is stored in database."""
    from abstracts_explorer.database import DatabaseManager

    db_path = tmp_path / "test.db"
    set_test_db(db_path)
    with DatabaseManager() as db:
        db.create_tables()

    compatible, stored, current = embeddings_manager.check_model_compatibility()

    assert compatible is True
    assert stored is None
    assert current == embeddings_manager.model_name


def test_check_model_compatibility_matching_models(embeddings_manager, tmp_path):
    """Test checking model compatibility when models match."""
    from abstracts_explorer.database import DatabaseManager

    db_path = tmp_path / "test.db"
    set_test_db(db_path)
    with DatabaseManager() as db:
        db.create_tables()
        db.set_embedding_model(embeddings_manager.model_name)

    compatible, stored, current = embeddings_manager.check_model_compatibility()

    assert compatible is True
    assert stored == embeddings_manager.model_name
    assert current == embeddings_manager.model_name


def test_check_model_compatibility_mismatched_models(embeddings_manager, tmp_path):
    """Test checking model compatibility when models differ."""
    from abstracts_explorer.database import DatabaseManager

    db_path = tmp_path / "test.db"
    different_model = "different-embedding-model"

    set_test_db(db_path)
    with DatabaseManager() as db:
        db.create_tables()
        db.set_embedding_model(different_model)

    compatible, stored, current = embeddings_manager.check_model_compatibility()

    assert compatible is False
    assert stored == different_model
    assert current == embeddings_manager.model_name


def test_check_model_compatibility_alias_prefix_ignored(embeddings_manager, tmp_path):
    """Test that alias- prefix is ignored when checking model compatibility."""
    from abstracts_explorer.database import DatabaseManager, normalize_model_name

    db_path = tmp_path / "test.db"
    # Store the model name without alias- prefix; embeddings_manager may use one with it
    base_model = normalize_model_name(embeddings_manager.model_name)
    # Ensure stored name differs from current name only by alias- prefix
    if embeddings_manager.model_name.lower().startswith("alias-"):
        stored_model = base_model  # strip alias-
    else:
        stored_model = f"alias-{base_model}"  # add alias-
    assert stored_model != embeddings_manager.model_name  # names differ textually

    set_test_db(db_path)
    with DatabaseManager() as db:
        db.create_tables()
        db.set_embedding_model(stored_model)

    compatible, stored, current = embeddings_manager.check_model_compatibility()

    assert compatible is True
    assert stored == stored_model
    assert current == embeddings_manager.model_name


def test_embed_from_database_stores_model(embeddings_manager, test_database):
    """Test that embed_from_database stores the embedding model in the database."""
    from abstracts_explorer.database import DatabaseManager

    # Set env var for test_database fixture
    set_test_db(test_database)

    embeddings_manager.connect()
    embeddings_manager.create_collection()

    # Embed papers
    embeddings_manager.embed_from_database()

    # Check that the model was stored
    with DatabaseManager() as db:
        stored_model = db.get_embedding_model()
        assert stored_model == embeddings_manager.model_name

    embeddings_manager.close()


def test_search_papers_semantic_with_year_filter(embeddings_manager, tmp_path, mock_lm_studio):
    """Test that search_papers_semantic correctly filters by year.

    This is a regression test for the bug where year filters were being
    converted to integers but ChromaDB stores metadata as strings, causing
    filters to return 0 results.
    """
    from abstracts_explorer.database import DatabaseManager
    from abstracts_explorer.plugin import LightweightPaper

    # Create a database with papers from different years
    db_path = tmp_path / "test_year_filter.db"
    set_test_db(db_path)

    with DatabaseManager() as db:
        db.create_tables()

        # Add papers with different years
        papers = [
            LightweightPaper(
                uid="paper2024",
                title="Paper from 2024",
                abstract="This is a paper from 2024.",
                authors=["Author A"],
                session="Session 1",
                poster_position="A1",
                year=2024,
                conference="NeurIPS",
            ),
            LightweightPaper(
                uid="paper2025",
                title="Paper from 2025",
                abstract="This is a paper from 2025.",
                authors=["Author B"],
                session="Session 2",
                poster_position="A2",
                year=2025,
                conference="NeurIPS",
            ),
        ]
        for paper in papers:
            db.add_paper(paper)

    # Embed papers
    embeddings_manager.connect()
    embeddings_manager.create_collection()
    embeddings_manager.embed_from_database()

    # Test search without filter - should return both papers
    with DatabaseManager() as db:
        results_all = embeddings_manager.search_papers_semantic("paper", database=db, limit=10)
        assert len(results_all) == 2, f"Expected 2 papers without filter, got {len(results_all)}"

    # Test search with year filter [2024] - should return only 2024 paper
    with DatabaseManager() as db:
        results_2024 = embeddings_manager.search_papers_semantic("paper", database=db, limit=10, years=[2024])
        assert len(results_2024) == 1, f"Expected 1 paper with year=2024 filter, got {len(results_2024)}"
        assert results_2024[0]["year"] == 2024, f"Expected year 2024, got {results_2024[0]['year']}"

    # Test search with year filter [2025] - should return only 2025 paper
    with DatabaseManager() as db:
        results_2025 = embeddings_manager.search_papers_semantic("paper", database=db, limit=10, years=[2025])
        assert len(results_2025) == 1, f"Expected 1 paper with year=2025 filter, got {len(results_2025)}"
        assert results_2025[0]["year"] == 2025, f"Expected year 2025, got {results_2025[0]['year']}"

    # Test search with multiple year filters - should return both papers
    with DatabaseManager() as db:
        results_both = embeddings_manager.search_papers_semantic("paper", database=db, limit=10, years=[2024, 2025])
        assert len(results_both) == 2, f"Expected 2 papers with years=[2024, 2025] filter, got {len(results_both)}"

    embeddings_manager.close()


def test_search_papers_semantic_field_filter_partial_match(embeddings_manager, tmp_path, mock_lm_studio):
    """Regression test: field:"value" filters in semantic search must use partial matching.

    ChromaDB does not support a $contains operator on metadata fields.  An earlier
    implementation tried to use ``{field: {"$contains": value}}`` as a ChromaDB
    where-clause which raised::

        Expected where operator to be one of $gt, $gte, $lt, $lte, $ne, $eq,
        $in, $nin, got $contains in query.

    The fix pre-filters using the SQL database (ILIKE substring match) and passes
    the resulting paper UIDs to ChromaDB as a ``{"uid": {"$in": [...]}}`` condition so:
    * partial author name matches work  (e.g. ``author:"Vaswani"`` matches the
      paper whose authors string contains "Vaswani")
    * the search does not raise an exception
    """
    from abstracts_explorer.database import DatabaseManager
    from abstracts_explorer.plugin import LightweightPaper

    db_path = tmp_path / "test_field_filter.db"
    set_test_db(db_path)

    with DatabaseManager() as db:
        db.create_tables()
        papers = [
            LightweightPaper(
                uid="paper_vaswani",
                title="Attention is All You Need",
                abstract="Transformer architecture paper.",
                authors=["Vaswani", "Shazeer"],
                session="Session 1",
                poster_position="A1",
                year=2017,
                conference="NeurIPS",
            ),
            LightweightPaper(
                uid="paper_devlin",
                title="BERT Paper",
                abstract="BERT language model paper.",
                authors=["Devlin", "Chang"],
                session="Session 2",
                poster_position="A2",
                year=2019,
                conference="NeurIPS",
            ),
        ]
        for paper in papers:
            db.add_paper(paper)

    embeddings_manager.connect()
    embeddings_manager.create_collection()
    embeddings_manager.embed_from_database()

    # This query previously triggered:
    #   "Expected where operator … got $contains in query."
    with DatabaseManager() as db:
        results = embeddings_manager.search_papers_semantic(
            'authors:"Vaswani" attention',
            database=db,
            limit=10,
        )

    # Should match exactly the Vaswani paper (partial author match)
    assert len(results) == 1, f"Expected 1 result, got {len(results)}: {[r.get('title') for r in results]}"
    assert results[0]["title"] == "Attention is All You Need"

    # Confirm the non-matching paper is excluded
    with DatabaseManager() as db:
        no_results = embeddings_manager.search_papers_semantic(
            'authors:"Lecun" attention',
            database=db,
            limit=10,
        )
    assert len(no_results) == 0, f"Expected 0 results for unknown author, got {len(no_results)}"

    embeddings_manager.close()


def test_search_papers_semantic_field_filter_only(embeddings_manager, tmp_path, mock_lm_studio):
    """Test that field-filter-only queries bypass semantic search and return DB results.

    When the query is e.g. ``authors:"Vaswani"`` (no remaining keywords after
    extracting the field filter), the method should return matching papers from
    the SQL database directly without generating an embedding or querying ChromaDB.
    """
    from abstracts_explorer.database import DatabaseManager
    from abstracts_explorer.plugin import LightweightPaper

    db_path = tmp_path / "test_field_only.db"
    set_test_db(db_path)

    with DatabaseManager() as db:
        db.create_tables()
        papers = [
            LightweightPaper(
                uid="paper_vaswani",
                title="Attention is All You Need",
                abstract="Transformer architecture paper.",
                authors=["Vaswani", "Shazeer"],
                session="Session 1",
                poster_position="A1",
                year=2017,
                conference="NeurIPS",
            ),
            LightweightPaper(
                uid="paper_devlin",
                title="BERT Paper",
                abstract="BERT language model paper.",
                authors=["Devlin", "Chang"],
                session="Session 2",
                poster_position="A2",
                year=2019,
                conference="NeurIPS",
            ),
        ]
        for paper in papers:
            db.add_paper(paper)

    # Connect embeddings (but note: the search should NOT need to generate embeddings)
    embeddings_manager.connect()
    embeddings_manager.create_collection()
    # Don't embed papers - field-filter-only queries should not need ChromaDB

    # Query with only a field filter (no remaining keywords)
    with DatabaseManager() as db:
        results = embeddings_manager.search_papers_semantic(
            'authors:"Vaswani"',
            database=db,
            limit=10,
        )

    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    assert results[0]["title"] == "Attention is All You Need"
    # Authors should be parsed into a list
    assert isinstance(results[0]["authors"], list)
    assert "Vaswani" in results[0]["authors"]

    # Test with the 'author' alias
    with DatabaseManager() as db:
        results = embeddings_manager.search_papers_semantic(
            'author:"Vaswani"',
            database=db,
            limit=10,
        )

    assert len(results) == 1
    assert results[0]["title"] == "Attention is All You Need"

    # Test with no matching results
    with DatabaseManager() as db:
        results = embeddings_manager.search_papers_semantic(
            'authors:"Nonexistent"',
            database=db,
            limit=10,
        )
    assert len(results) == 0

    embeddings_manager.close()


def test_search_papers_semantic_field_filter_only_with_filters(embeddings_manager, tmp_path, mock_lm_studio):
    """Test field-filter-only queries respect additional session/year/conference filters."""
    from abstracts_explorer.database import DatabaseManager
    from abstracts_explorer.plugin import LightweightPaper

    db_path = tmp_path / "test_field_only_filters.db"
    set_test_db(db_path)

    with DatabaseManager() as db:
        db.create_tables()
        papers = [
            LightweightPaper(
                uid="paper_a",
                title="Paper A",
                abstract="Paper A abstract.",
                authors=["John Smith"],
                session="Session 1",
                poster_position="A1",
                year=2024,
                conference="NeurIPS",
            ),
            LightweightPaper(
                uid="paper_b",
                title="Paper B",
                abstract="Paper B abstract.",
                authors=["John Smith"],
                session="Session 2",
                poster_position="A2",
                year=2025,
                conference="ICLR",
            ),
        ]
        for paper in papers:
            db.add_paper(paper)

    embeddings_manager.connect()
    embeddings_manager.create_collection()

    # Both papers by John Smith
    with DatabaseManager() as db:
        results = embeddings_manager.search_papers_semantic(
            'authors:"John Smith"',
            database=db,
            limit=10,
        )
    assert len(results) == 2

    # Filter by year
    with DatabaseManager() as db:
        results = embeddings_manager.search_papers_semantic(
            'authors:"John Smith"',
            database=db,
            limit=10,
            years=[2024],
        )
    assert len(results) == 1
    assert results[0]["year"] == 2024

    # Filter by conference
    with DatabaseManager() as db:
        results = embeddings_manager.search_papers_semantic(
            'authors:"John Smith"',
            database=db,
            limit=10,
            conferences=["ICLR"],
        )
    assert len(results) == 1
    assert results[0]["conference"] == "ICLR"

    embeddings_manager.close()


def test_search_papers_semantic_implicit_author_match(embeddings_manager, tmp_path, mock_lm_studio):
    """Test that semantic search always checks for author matches without explicit author: filter.

    When the query text matches an author name in the database (even without using
    the ``author:"Name"`` syntax), those papers should appear first in the results.
    """
    from abstracts_explorer.database import DatabaseManager
    from abstracts_explorer.plugin import LightweightPaper

    db_path = tmp_path / "test_implicit_author.db"
    set_test_db(db_path)

    with DatabaseManager() as db:
        db.create_tables()
        papers = [
            LightweightPaper(
                uid="paper_lecun",
                title="LeCun Vision Paper",
                abstract="Convolutional neural networks for image recognition.",
                authors=["Yann LeCun", "Bengio"],
                session="Session 1",
                poster_position="A1",
                year=2020,
                conference="NeurIPS",
            ),
            LightweightPaper(
                uid="paper_attention",
                title="Attention Mechanism Paper",
                abstract="Self-attention and transformer architectures.",
                authors=["Vaswani", "Shazeer"],
                session="Session 2",
                poster_position="A2",
                year=2017,
                conference="NeurIPS",
            ),
        ]
        for paper in papers:
            db.add_paper(paper)

    embeddings_manager.connect()
    embeddings_manager.create_collection()
    embeddings_manager.embed_from_database()

    # Search for "LeCun" without the author: field prefix — should still find
    # the LeCun paper first (implicit author match).
    with DatabaseManager() as db:
        results = embeddings_manager.search_papers_semantic(
            "LeCun",
            database=db,
            limit=10,
        )

    assert len(results) >= 1, f"Expected at least 1 result, got {len(results)}"
    # The LeCun paper must appear first (author match takes priority)
    assert results[0]["title"] == "LeCun Vision Paper", f"Expected LeCun paper first, got {results[0]['title']!r}"
    assert isinstance(results[0]["authors"], list)
    assert any("LeCun" in a for a in results[0]["authors"])

    embeddings_manager.close()


def test_search_papers_semantic_implicit_author_match_no_duplicate(embeddings_manager, tmp_path, mock_lm_studio):
    """Test that implicit author matches are not duplicated in the semantic results."""
    from abstracts_explorer.database import DatabaseManager
    from abstracts_explorer.plugin import LightweightPaper

    db_path = tmp_path / "test_implicit_author_dedup.db"
    set_test_db(db_path)

    with DatabaseManager() as db:
        db.create_tables()
        papers = [
            LightweightPaper(
                uid="paper_lecun",
                title="LeCun Vision Paper",
                abstract="Convolutional neural networks for image recognition.",
                authors=["Yann LeCun"],
                session="Session 1",
                poster_position="A1",
                year=2020,
                conference="NeurIPS",
            ),
            LightweightPaper(
                uid="paper_other",
                title="Other Paper",
                abstract="Unrelated topic.",
                authors=["Someone Else"],
                session="Session 2",
                poster_position="A2",
                year=2021,
                conference="NeurIPS",
            ),
        ]
        for paper in papers:
            db.add_paper(paper)

    embeddings_manager.connect()
    embeddings_manager.create_collection()
    embeddings_manager.embed_from_database()

    with DatabaseManager() as db:
        results = embeddings_manager.search_papers_semantic(
            "LeCun",
            database=db,
            limit=10,
        )

    # No duplicate UIDs
    uids = [p["uid"] for p in results]
    assert len(uids) == len(set(uids)), f"Duplicate UIDs in results: {uids}"
    # LeCun paper must appear first
    assert results[0]["title"] == "LeCun Vision Paper"

    embeddings_manager.close()


def test_search_papers_semantic_no_author_match_falls_back_to_semantic(embeddings_manager, tmp_path, mock_lm_studio):
    """Test that when query doesn't match any author, semantic results are returned normally."""
    from abstracts_explorer.database import DatabaseManager
    from abstracts_explorer.plugin import LightweightPaper

    db_path = tmp_path / "test_no_author_fallback.db"
    set_test_db(db_path)

    with DatabaseManager() as db:
        db.create_tables()
        papers = [
            LightweightPaper(
                uid="paper_transformer",
                title="Transformer Paper",
                abstract="Self-attention and transformer architectures.",
                authors=["Vaswani", "Shazeer"],
                session="Session 1",
                poster_position="A1",
                year=2017,
                conference="NeurIPS",
            ),
        ]
        for paper in papers:
            db.add_paper(paper)

    embeddings_manager.connect()
    embeddings_manager.create_collection()
    embeddings_manager.embed_from_database()

    # Query that does NOT match any author name
    with DatabaseManager() as db:
        results = embeddings_manager.search_papers_semantic(
            "transformer architecture",
            database=db,
            limit=10,
        )

    # Should return the paper via semantic search (no author match, but semantically similar)
    assert len(results) >= 1
    assert results[0]["title"] == "Transformer Paper"

    embeddings_manager.close()


def test_search_papers_semantic_distance_threshold(embeddings_manager, tmp_path, mock_lm_studio):
    """Test that search_papers_semantic filters results by distance_threshold.

    Papers with distance greater than the threshold must be excluded from the
    results, matching the behaviour of count_papers_within_distance.
    """
    from unittest.mock import patch as mock_patch
    from abstracts_explorer.database import DatabaseManager
    from abstracts_explorer.plugin import LightweightPaper

    db_path = tmp_path / "test_distance_threshold.db"
    set_test_db(db_path)

    # Pre-compute the UIDs that the database will assign.
    # LightweightPaper has no 'uid' field; original_id defaults to None.
    close_uid = DatabaseManager.compute_uid("Close Paper", None, "NeurIPS", 2024)
    far_uid = DatabaseManager.compute_uid("Far Paper", None, "NeurIPS", 2024)

    with DatabaseManager() as db:
        db.create_tables()
        papers = [
            LightweightPaper(
                title="Close Paper",
                abstract="Very similar to the query.",
                authors=["Author A"],
                session="Session 1",
                poster_position="A1",
                year=2024,
                conference="NeurIPS",
            ),
            LightweightPaper(
                title="Far Paper",
                abstract="Not similar to the query at all.",
                authors=["Author B"],
                session="Session 2",
                poster_position="A2",
                year=2024,
                conference="NeurIPS",
            ),
        ]
        for paper in papers:
            db.add_paper(paper)

    embeddings_manager.connect()
    embeddings_manager.create_collection()
    embeddings_manager.embed_from_database()

    # Mock search_similar to return controlled distances:
    # close_paper has distance 0.5 (within threshold), far_paper has distance 1.5 (outside)
    mock_results = {
        "ids": [[close_uid, far_uid]],
        "distances": [[0.5, 1.5]],
        "documents": [["doc1", "doc2"]],
        "metadatas": [[{"uid": close_uid}, {"uid": far_uid}]],
    }

    with mock_patch.object(embeddings_manager, "search_similar", return_value=mock_results):
        with DatabaseManager() as db:
            # With default threshold (1.1): only the close paper should be returned
            results = embeddings_manager.search_papers_semantic("query", database=db, limit=10)
            titles = [r["title"] for r in results]
            assert "Close Paper" in titles, "Close paper (distance 0.5) should be within threshold 1.1"
            assert "Far Paper" not in titles, "Far paper (distance 1.5) should be outside threshold 1.1"

        with DatabaseManager() as db:
            # With higher threshold (2.0): both papers should be returned
            results_high = embeddings_manager.search_papers_semantic(
                "query", database=db, limit=10, distance_threshold=2.0
            )
            titles_high = [r["title"] for r in results_high]
            assert "Close Paper" in titles_high, "Close paper should be returned with threshold 2.0"
            assert "Far Paper" in titles_high, "Far paper (distance 1.5) should be within threshold 2.0"

        with DatabaseManager() as db:
            # With very low threshold (0.1): no papers should be returned
            results_none = embeddings_manager.search_papers_semantic(
                "query", database=db, limit=10, distance_threshold=0.1
            )
            assert len(results_none) == 0, "No papers should be returned with threshold 0.1"

    embeddings_manager.close()


class TestParseChromaDBMetadata:
    """Tests for EmbeddingsManager.parse_chromadb_metadata."""

    def _make_raw_metadata(self, **overrides):
        """Create a complete raw ChromaDB metadata dict with all required fields."""
        base = {
            "title": "Test Paper",
            "authors": "Alice;Bob",
            "abstract": "An abstract",
            "session": "ML Track",
            "poster_position": "1",
            "year": "2024",
            "conference": "NeurIPS",
        }
        base.update(overrides)
        return base

    def test_converts_year_string_to_int(self):
        """Test that year is converted from string to int."""
        raw = self._make_raw_metadata(year="2024")
        parsed = EmbeddingsManager.parse_chromadb_metadata(raw)
        assert parsed["year"] == 2024
        assert isinstance(parsed["year"], int)

    def test_converts_original_id_string_to_int(self):
        """Test that original_id is converted from string to int."""
        raw = self._make_raw_metadata(original_id="42")
        parsed = EmbeddingsManager.parse_chromadb_metadata(raw)
        assert parsed["original_id"] == 42
        assert isinstance(parsed["original_id"], int)

    def test_authors_parsed_to_list(self):
        """Test that semicolon-separated authors are parsed to a list."""
        raw = self._make_raw_metadata(authors="Alice;Bob;Charlie")
        parsed = EmbeddingsManager.parse_chromadb_metadata(raw)
        assert parsed["authors"] == ["Alice", "Bob", "Charlie"]

    def test_keywords_parsed_to_list(self):
        """Test that comma-separated keywords are parsed to a list."""
        raw = self._make_raw_metadata(keywords="ml,ai,deep learning")
        parsed = EmbeddingsManager.parse_chromadb_metadata(raw)
        assert parsed["keywords"] == ["ml", "ai", "deep learning"]

    def test_string_fields_preserved(self):
        """Test that string fields are preserved correctly."""
        raw = self._make_raw_metadata()
        parsed = EmbeddingsManager.parse_chromadb_metadata(raw)
        assert parsed["title"] == "Test Paper"
        assert parsed["session"] == "ML Track"
        assert parsed["conference"] == "NeurIPS"

    def test_integer_year_stays_int(self):
        """Test that integer year value passes through unchanged."""
        raw = self._make_raw_metadata(year=2024)
        parsed = EmbeddingsManager.parse_chromadb_metadata(raw)
        assert parsed["year"] == 2024
        assert isinstance(parsed["year"], int)


class TestSerializeMetadataForChromaDB:
    """Tests for EmbeddingsManager._serialize_metadata_for_chromadb."""

    def test_authors_list_joined_with_semicolon_and_space(self):
        """Test that authors list is joined with '; ' matching the DB format."""
        metadata = {"authors": ["Alice", "Bob", "Charlie"], "year": 2024, "title": "Test"}
        result = EmbeddingsManager._serialize_metadata_for_chromadb(metadata)
        assert result["authors"] == "Alice; Bob; Charlie"

    def test_keywords_list_joined_with_comma_and_space(self):
        """Test that keywords list is joined with ', ' matching the DB format."""
        metadata = {"keywords": ["ml", "ai", "deep learning"], "authors": ["Alice"], "year": 2024}
        result = EmbeddingsManager._serialize_metadata_for_chromadb(metadata)
        assert result["keywords"] == "ml, ai, deep learning"

    def test_none_values_converted_to_empty_string(self):
        """Test that None values are converted to empty strings."""
        metadata = {"title": None, "authors": ["Alice"], "year": 2024}
        result = EmbeddingsManager._serialize_metadata_for_chromadb(metadata)
        assert result["title"] == ""

    def test_numeric_values_converted_to_string(self):
        """Test that numeric values are converted to strings for consistent ChromaDB filters."""
        metadata = {"year": 2024, "authors": ["Alice"]}
        result = EmbeddingsManager._serialize_metadata_for_chromadb(metadata)
        assert result["year"] == "2024"
        assert isinstance(result["year"], str)

    def test_string_values_preserved(self):
        """Test that existing string values are not changed."""
        metadata = {"title": "My Paper", "conference": "NeurIPS", "authors": "Alice; Bob", "year": "2024"}
        result = EmbeddingsManager._serialize_metadata_for_chromadb(metadata)
        assert result["title"] == "My Paper"
        assert result["conference"] == "NeurIPS"
        assert result["authors"] == "Alice; Bob"
        assert result["year"] == "2024"
        assert isinstance(result["year"], str)

    def test_authors_list_whitespace_stripped(self):
        """Test that whitespace is stripped from each author name before joining."""
        metadata = {"authors": [" Alice ", "Bob", "  Charlie  "], "year": 2024}
        result = EmbeddingsManager._serialize_metadata_for_chromadb(metadata)
        assert result["authors"] == "Alice; Bob; Charlie"

    def test_keywords_list_whitespace_stripped(self):
        """Test that whitespace is stripped from each keyword before joining."""
        metadata = {"keywords": [" ml ", "ai", "  deep learning  "], "authors": ["Alice"]}
        result = EmbeddingsManager._serialize_metadata_for_chromadb(metadata)
        assert result["keywords"] == "ml, ai, deep learning"


class TestImportEmbeddings:
    """Tests for EmbeddingsManager.import_embeddings."""

    def test_import_embeddings_with_list_metadata(self, embeddings_manager):
        """Test that import_embeddings handles list metadata values (authors, keywords).

        Regression test for the registry import bug where exported metadata
        contains Python lists (after being parsed through LightweightPaper) but
        ChromaDB only accepts scalar values.
        """
        embeddings_manager.connect()
        embeddings_manager.create_collection()

        # Simulate data as it would appear after export_embeddings() + JSON round-trip:
        # authors and keywords are lists, year is an int.
        data = {
            "ids": ["paper1"],
            "documents": ["Title: Test Paper\nAbstract: Test abstract"],
            "metadatas": [
                {
                    "title": "Test Paper",
                    "abstract": "Test abstract",
                    "authors": ["Alice", "Bob", "Charlie"],  # list, not string
                    "keywords": ["ml", "ai"],  # list, not string
                    "year": 2018,  # int, not string
                    "conference": "CHI",
                    "session": "Session A",
                    "poster_position": "1",
                }
            ],
            "embeddings": [[0.1, 0.2, 0.3]],
        }

        # Should not raise an error
        count = embeddings_manager.import_embeddings(data, "CHI", 2018)
        assert count == 1

        # Verify the paper was actually stored
        stats = embeddings_manager.get_collection_stats()
        assert stats["count"] == 1

        embeddings_manager.close()

    def test_import_embeddings_authors_stored_with_consistent_separator(self, embeddings_manager):
        """Test that after importing with list authors, they are stored consistently with the DB format."""
        embeddings_manager.connect()
        embeddings_manager.create_collection()

        data = {
            "ids": ["paper1"],
            "documents": ["Title: Test Paper\nAbstract: Test abstract"],
            "metadatas": [
                {
                    "title": "Test Paper",
                    "abstract": "Test abstract",
                    "authors": ["Alice", "Bob", "Charlie"],
                    "year": 2024,
                    "conference": "NeurIPS",
                    "session": "Session A",
                    "poster_position": "1",
                }
            ],
            "embeddings": [[0.1, 0.2, 0.3]],
        }

        embeddings_manager.import_embeddings(data, "NeurIPS", 2024)

        # Retrieve raw metadata from ChromaDB and verify authors are stored as a
        # semicolon-and-space-separated string (same format as the SQL database).
        raw = embeddings_manager.collection.get(ids=["paper1"], include=["metadatas"])
        stored_authors = raw["metadatas"][0]["authors"]
        assert isinstance(stored_authors, str)
        assert stored_authors == "Alice; Bob; Charlie"

        embeddings_manager.close()


class TestExportImportRoundTrip:
    """Integration tests for the export→import round-trip (no registry involved)."""

    def test_export_import_round_trip_preserves_data(self, embeddings_manager):
        """Test that papers survive an export→import round-trip with their metadata intact.

        This tests the full cycle: add a paper with list authors → export to dict →
        import back from that dict → verify authors are stored correctly.
        """
        embeddings_manager.connect()
        embeddings_manager.create_collection()

        # Add a paper via add_paper (as would happen from embed_from_database)
        paper = {
            "uid": "roundtrip1",
            "title": "Round Trip Test Paper",
            "abstract": "Testing the full export-import cycle.",
            "authors": "Cynthia Putnam; Mary Bungum; Dan Spinner",  # already string (from DB)
            "keywords": "testing, round trip, embeddings",
            "year": "2018",
            "conference": "CHI",
            "session": "Session A",
            "poster_position": "1",
        }
        embeddings_manager.add_paper(paper)

        # Export embeddings to a local dict (no registry)
        exported = embeddings_manager.export_embeddings("CHI", 2018)

        assert exported["ids"] == ["roundtrip1"]
        assert len(exported["metadatas"]) == 1

        # After export, authors should be a list (parse_chromadb_metadata converts to list)
        exported_authors = exported["metadatas"][0]["authors"]
        assert isinstance(exported_authors, list)
        assert exported_authors == ["Cynthia Putnam", "Mary Bungum", "Dan Spinner"]

        # Clear the collection so we can re-import
        embeddings_manager.collection.delete(ids=["roundtrip1"])
        assert embeddings_manager.get_collection_stats()["count"] == 0

        # Import from the exported dict (simulating registry import)
        count = embeddings_manager.import_embeddings(exported, "CHI", 2018)
        assert count == 1

        # Retrieve raw metadata and verify authors stored as string with '; ' separator
        raw = embeddings_manager.collection.get(ids=["roundtrip1"], include=["metadatas"])
        stored_meta = raw["metadatas"][0]
        assert isinstance(stored_meta["authors"], str)
        assert stored_meta["authors"] == "Cynthia Putnam; Mary Bungum; Dan Spinner"

        # Parse back through LightweightPaper to verify round-trip fidelity
        reparsed = embeddings_manager.parse_chromadb_metadata(stored_meta)

        # The full dictionary must be identical to what was exported before the delete+re-import.
        # This covers all fields: authors (list), keywords (list), year (int), title, conference,
        # session, poster_position, abstract, and any other metadata that was stored.
        assert reparsed == exported["metadatas"][0]

        embeddings_manager.close()

    def test_export_import_round_trip_with_keywords(self, embeddings_manager):
        """Test that keywords survive an export→import round-trip."""
        embeddings_manager.connect()
        embeddings_manager.create_collection()

        paper = {
            "uid": "kw_roundtrip",
            "title": "Keyword Round Trip Paper",
            "abstract": "Tests keyword export-import.",
            "authors": "Alice Author",
            "keywords": "machine learning, deep learning, neural networks",
            "year": "2024",
            "conference": "NeurIPS",
            "session": "ML Track",
            "poster_position": "P1",
        }
        embeddings_manager.add_paper(paper)

        exported = embeddings_manager.export_embeddings("NeurIPS", 2024)

        # After export, keywords should be a list
        exported_keywords = exported["metadatas"][0]["keywords"]
        assert isinstance(exported_keywords, list)
        assert exported_keywords == ["machine learning", "deep learning", "neural networks"]

        # Import and verify full dictionary equality
        embeddings_manager.collection.delete(ids=["kw_roundtrip"])
        embeddings_manager.import_embeddings(exported, "NeurIPS", 2024)

        raw = embeddings_manager.collection.get(ids=["kw_roundtrip"], include=["metadatas"])
        reparsed = embeddings_manager.parse_chromadb_metadata(raw["metadatas"][0])

        # The full dictionary must equal what was exported: keywords as a list,
        # authors as a list, year as int, and all other metadata fields intact.
        assert reparsed == exported["metadatas"][0]

        embeddings_manager.close()


class TestEmbeddingsMetadataRoundTrip:
    """Integration tests for the SQL DB → ChromaDB metadata round-trip."""

    def test_sql_to_chromadb_round_trip(self, embeddings_manager, tmp_path):
        """Test that paper metadata is preserved when going SQL DB → embed → ChromaDB → retrieve.

        Verifies that:
        1. A LightweightPaper stored in the SQL DB is embedded into ChromaDB.
        2. The raw ChromaDB metadata can be parsed back via parse_chromadb_metadata.
        3. The parsed metadata matches the original LightweightPaper.
        """
        from abstracts_explorer.database import DatabaseManager
        from abstracts_explorer.plugin import LightweightPaper
        from tests.conftest import set_test_db

        # Build an isolated DB
        db_path = tmp_path / "roundtrip_test.db"
        set_test_db(db_path)

        original_paper = LightweightPaper(
            title="SQL to ChromaDB Round Trip Paper",
            abstract="This paper tests the full round-trip from SQL DB to ChromaDB.",
            authors=["Cynthia Putnam", "Mary Bungum", "Dan Spinner"],
            keywords=["testing", "round trip", "embeddings"],
            session="Integration Session",
            poster_position="RT1",
            year=2018,
            conference="CHI",
        )

        with DatabaseManager() as db:
            db.create_tables()
            uid = db.add_paper(original_paper)

        assert uid is not None

        # Embed the paper from the database (embedding API call is mocked in the fixture)
        embeddings_manager.connect()
        embeddings_manager.create_collection()
        embedded_count = embeddings_manager.embed_from_database()
        assert embedded_count == 1

        # Retrieve the raw metadata directly from ChromaDB
        raw_results = embeddings_manager.collection.get(ids=[uid], include=["metadatas"])
        assert len(raw_results["metadatas"]) == 1
        raw_meta = raw_results["metadatas"][0]

        # Raw metadata must not contain list values (ChromaDB rejects lists)
        for value in raw_meta.values():
            assert not isinstance(value, list), f"ChromaDB metadata contains a list: {value!r}"

        # Authors and keywords should be stored as strings with the canonical separators
        assert raw_meta["authors"] == "Cynthia Putnam; Mary Bungum; Dan Spinner"
        assert raw_meta["keywords"] == "testing, round trip, embeddings"

        # Parse the raw metadata back through LightweightPaper
        parsed = embeddings_manager.parse_chromadb_metadata(raw_meta)

        # All fields must match the original LightweightPaper.  Optional fields
        # that were None in the original are excluded from both dicts via
        # model_dump(exclude_none=True), so equality holds end-to-end.
        assert parsed == original_paper.model_dump(exclude_none=True)

        embeddings_manager.close()

    def test_sql_to_chromadb_round_trip_no_keywords(self, embeddings_manager, tmp_path):
        """Test that papers without keywords are handled correctly in the SQL → ChromaDB round-trip."""
        from abstracts_explorer.database import DatabaseManager
        from abstracts_explorer.plugin import LightweightPaper
        from tests.conftest import set_test_db

        db_path = tmp_path / "roundtrip_nokw.db"
        set_test_db(db_path)

        original_paper = LightweightPaper(
            title="Paper Without Keywords",
            abstract="This paper has no keywords.",
            authors=["Solo Author"],
            keywords=None,  # no keywords
            session="Solo Session",
            poster_position="S1",
            year=2020,
            conference="CHI",
        )

        with DatabaseManager() as db:
            db.create_tables()
            uid = db.add_paper(original_paper)

        assert uid is not None

        embeddings_manager.connect()
        embeddings_manager.create_collection()
        embedded_count = embeddings_manager.embed_from_database()
        assert embedded_count == 1

        raw_results = embeddings_manager.collection.get(ids=[uid], include=["metadatas"])
        raw_meta = raw_results["metadatas"][0]

        # keywords should be stored as an empty string (not None, not a list)
        assert isinstance(raw_meta.get("keywords", ""), str)

        # No list values should appear in metadata
        for value in raw_meta.values():
            assert not isinstance(value, list), f"ChromaDB metadata contains a list: {value!r}"

        # Parsing back must reproduce all fields of the original paper exactly.
        # Optional fields that were None on the original are absent from both
        # model_dump(exclude_none=True) dicts (including keywords=None).
        parsed = embeddings_manager.parse_chromadb_metadata(raw_meta)
        assert parsed == original_paper.model_dump(exclude_none=True)

        embeddings_manager.close()


class TestRateLimiting:
    """Tests for rate limiting in EmbeddingsManager."""

    def test_default_requests_per_minute_from_config(self, embeddings_manager):
        """Test that requests_per_minute is initialized from config (default 60)."""
        # The fixture does not pass requests_per_minute explicitly, so it uses config default
        assert isinstance(embeddings_manager.requests_per_minute, int)
        assert embeddings_manager.requests_per_minute >= 0

    def test_requests_per_minute_explicit(self, tmp_path, monkeypatch):
        """Test that requests_per_minute can be set explicitly."""
        from tests.conftest import get_env_test_path
        from abstracts_explorer.config import get_config

        monkeypatch.setenv("EMBEDDING_DB", str(tmp_path / "chroma"))
        get_config(reload=True, env_path=get_env_test_path())

        em = EmbeddingsManager(requests_per_minute=120)
        assert em.requests_per_minute == 120

    def test_requests_per_minute_zero_disables_limiting(self, tmp_path, monkeypatch):
        """Test that setting requests_per_minute=0 disables rate limiting."""
        from tests.conftest import get_env_test_path
        from abstracts_explorer.config import get_config

        monkeypatch.setenv("EMBEDDING_DB", str(tmp_path / "chroma"))
        get_config(reload=True, env_path=get_env_test_path())

        em = EmbeddingsManager(requests_per_minute=0)
        assert em.requests_per_minute == 0

    def test_rate_limited_transport_attached_when_rpm_gt_zero(self, tmp_path, monkeypatch):
        """Test that a RateLimitedTransport http_client is used when requests_per_minute > 0."""
        from tests.conftest import get_env_test_path
        from abstracts_explorer.config import get_config
        from abstracts_explorer.embeddings import RateLimitedTransport

        monkeypatch.setenv("EMBEDDING_DB", str(tmp_path / "chroma"))
        get_config(reload=True, env_path=get_env_test_path())

        em = EmbeddingsManager(requests_per_minute=30)
        # Trigger lazy initialization of the OpenAI client
        client = em.openai_client
        # The underlying httpx client should use our rate-limited transport
        assert isinstance(client._client._transport, RateLimitedTransport)
        assert client._client._transport._min_interval == pytest.approx(2.0)  # 60/30

    def test_no_rate_limited_transport_when_rpm_zero(self, tmp_path, monkeypatch):
        """Test that no RateLimitedTransport is used when requests_per_minute=0."""
        from tests.conftest import get_env_test_path
        from abstracts_explorer.config import get_config
        from abstracts_explorer.embeddings import RateLimitedTransport

        monkeypatch.setenv("EMBEDDING_DB", str(tmp_path / "chroma"))
        get_config(reload=True, env_path=get_env_test_path())

        em = EmbeddingsManager(requests_per_minute=0)
        client = em.openai_client
        # When rate limiting is disabled, the http_client is not a custom one wrapping RateLimitedTransport
        # The OpenAI client uses its own default transport
        assert not isinstance(getattr(getattr(client, "_client", None), "_transport", None), RateLimitedTransport)

    def test_rate_limited_transport_sleeps_between_requests(self):
        """Test that RateLimitedTransport sleeps the right amount between rapid requests."""
        import httpx
        from abstracts_explorer.embeddings import RateLimitedTransport

        mock_inner = Mock()
        mock_inner.handle_request.return_value = Mock(spec=httpx.Response)

        transport = RateLimitedTransport(mock_inner, requests_per_minute=60)  # 1 s interval
        transport._last_request_time = 0.0

        sleep_calls = []

        with patch("abstracts_explorer.embeddings.time.sleep", side_effect=lambda d: sleep_calls.append(d)):
            with patch("abstracts_explorer.embeddings.time.monotonic", return_value=0.3):
                # elapsed = 0.3 - 0.0 = 0.3 < 1.0 → sleep(0.7)
                transport.handle_request(Mock(spec=httpx.Request))

        assert len(sleep_calls) == 1
        assert sleep_calls[0] == pytest.approx(0.7)

    def test_rate_limited_transport_no_sleep_when_enough_time_elapsed(self):
        """Test that RateLimitedTransport does not sleep when the interval is already met."""
        import httpx
        from abstracts_explorer.embeddings import RateLimitedTransport

        mock_inner = Mock()
        mock_inner.handle_request.return_value = Mock(spec=httpx.Response)

        transport = RateLimitedTransport(mock_inner, requests_per_minute=60)  # 1 s interval
        transport._last_request_time = 0.0

        sleep_calls = []

        with patch("abstracts_explorer.embeddings.time.sleep", side_effect=lambda d: sleep_calls.append(d)):
            with patch("abstracts_explorer.embeddings.time.monotonic", return_value=2.0):
                # elapsed = 2.0 > 1.0 → no sleep
                transport.handle_request(Mock(spec=httpx.Request))

        assert len(sleep_calls) == 0

    def test_rate_limited_transport_updates_last_request_time(self):
        """Test that RateLimitedTransport updates _last_request_time after the request."""
        import httpx
        from abstracts_explorer.embeddings import RateLimitedTransport

        mock_inner = Mock()
        mock_inner.handle_request.return_value = Mock(spec=httpx.Response)

        transport = RateLimitedTransport(mock_inner, requests_per_minute=60)
        transport._last_request_time = 0.0

        monotonic_values = iter([5.0, 5.5])  # check, post-call update

        with patch("abstracts_explorer.embeddings.time.monotonic", side_effect=lambda: next(monotonic_values)):
            transport.handle_request(Mock(spec=httpx.Request))

        assert transport._last_request_time == 5.5

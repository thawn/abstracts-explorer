"""
Tests for the embeddings module.
"""

import logging
import threading
import time
from pathlib import Path

import httpx2
import pytest
from filelock import ReadWriteLock
from unittest.mock import Mock

from abstracts_explorer.embeddings import (
    EmbeddingsError,
    EmbeddingsManager,
    QueryEmbedding,
    hash_text,
    is_content_hash,
)
from abstracts_explorer.embeddings.tokens import count_tokens as _count_tokens
from tests.conftest import set_test_db
from tests.helpers import FakeTime


def _search_semantic(db, query, **kwargs):
    """Run a semantic search through the unified DatabaseManager.search()."""
    papers, _ = db.search(query, use_embeddings=True, **kwargs)
    return papers


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
        """Test connecting to ChromaDB via the lazy client property."""
        assert embeddings_manager.client is not None
        # Check that embedding_db path exists (only for local paths)
        if not embeddings_manager.embedding_db.startswith("http"):
            from pathlib import Path

            assert Path(embeddings_manager.embedding_db).exists()
        embeddings_manager.close()

    def test_close(self, embeddings_manager):
        """Test closing ChromaDB connection."""
        embeddings_manager.client
        embeddings_manager.close()
        assert embeddings_manager._client is None
        assert embeddings_manager._collection is None

    def test_context_manager(self, embeddings_manager):
        """Test context manager functionality."""
        with embeddings_manager as em:
            assert em._client is not None
        assert embeddings_manager._client is None

    def test_create_collection_connects_to_chromadb(self, embeddings_manager):
        """Test that create_collection connects the client on first use."""
        embeddings_manager.create_collection()

        assert embeddings_manager._client is not None
        assert embeddings_manager._collection is not None
        embeddings_manager.close()

    def test_test_api_connection_success(self, embeddings_manager):
        """Test successful API connection."""
        result = embeddings_manager.test_api_connection()
        assert result is True
        embeddings_manager.openai_client.models.list.assert_called_once()

    def test_test_api_connection_failure(self, embeddings_manager):
        """Test failed API connection."""
        embeddings_manager.openai_client.models.list.side_effect = Exception("Connection error")
        result = embeddings_manager.test_api_connection()
        assert result is False

    def test_test_api_connection_verbose_success(self, embeddings_manager, capsys):
        """Test verbose connection test prints success messages."""
        result = embeddings_manager.test_api_connection(verbose=True)

        assert result is True
        captured = capsys.readouterr()
        assert "Successfully connected to OpenAI API" in captured.out
        assert captured.err == ""

    def test_test_api_connection_verbose_failure(self, embeddings_manager, capsys):
        """Test verbose connection test prints failure details."""
        embeddings_manager.openai_client.models.list.side_effect = Exception("Connection error")

        result = embeddings_manager.test_api_connection(verbose=True)

        assert result is False
        captured = capsys.readouterr()
        assert "Failed to connect to OpenAI API" in captured.err
        assert embeddings_manager.backend_url in captured.err
        assert embeddings_manager.model_name in captured.err

    def test_test_api_connection_model_hint(self, embeddings_manager, capsys):
        """Test verbose connection test uses a custom model hint."""
        embeddings_manager.openai_client.models.list.side_effect = Exception("Connection error")

        result = embeddings_manager.test_api_connection(verbose=True, model_hint="A language model is loaded")

        assert result is False
        captured = capsys.readouterr()
        assert "A language model is loaded" in captured.err

    def test_close_closes_openai_client(self, embeddings_manager):
        """Test that close() closes and resets the OpenAI client."""
        embeddings_manager.client
        openai_client = embeddings_manager.openai_client
        embeddings_manager.close()
        assert embeddings_manager._openai_client is None
        openai_client.close.assert_called_once()

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

    def test_test_api_connection_timeout_is_logged_as_error(self, embeddings_manager, caplog):
        """A backend connection timeout is logged as an error."""
        embeddings_manager.openai_client.models.list.side_effect = httpx2.ReadTimeout("timed out")

        with caplog.at_level(logging.ERROR, logger="abstracts_explorer.embeddings.models"):
            result = embeddings_manager.test_api_connection()

        assert result is False
        assert any(
            record.levelno == logging.ERROR and "timed out" in record.getMessage() for record in caplog.records
        )

    def test_generate_embeddings_batch_timeout_logs_warning(self, embeddings_manager, caplog):
        """A recovered batch timeout is logged as a warning, not an error."""
        embeddings_manager.openai_client.embeddings.create.side_effect = [httpx2.ReadTimeout("timed out")] + [
            self._single_response() for _ in range(2)
        ]

        with caplog.at_level(logging.WARNING, logger="abstracts_explorer.embeddings.generation"):
            vectors = embeddings_manager.generate_embeddings(["one", "two"])

        assert len(vectors) == 2
        assert any(
            record.levelno == logging.WARNING and "timed out" in record.getMessage() for record in caplog.records
        )
        assert not any(record.levelno == logging.ERROR for record in caplog.records)

    def test_generate_embeddings_batches_inputs(self, embeddings_manager):
        """A batch request returns one vector per input."""
        vectors = embeddings_manager.generate_embeddings(["one", "two", "three"])
        assert len(vectors) == 3
        assert all(len(vector) == 4096 for vector in vectors)
        assert embeddings_manager.openai_client.embeddings.create.call_args.kwargs["input"] == [
            "one",
            "two",
            "three",
        ]

    def test_generate_embeddings_empty_list(self, embeddings_manager):
        """An empty batch makes no API call."""
        assert embeddings_manager.generate_embeddings([]) == []
        embeddings_manager.openai_client.embeddings.create.assert_not_called()

    def test_generate_embeddings_rejects_empty_text(self, embeddings_manager):
        """Every text in a batch must be non-empty."""
        with pytest.raises(EmbeddingsError, match="Cannot generate embedding for empty text"):
            embeddings_manager.generate_embeddings(["ok", "  "])

    def test_generate_embeddings_falls_back_on_context_error(self, embeddings_manager):
        """A context-length batch error retries each text individually."""
        context_error = Exception("maximum context length is 10 tokens")
        embeddings_manager.openai_client.embeddings.create.side_effect = [context_error] + [
            self._single_response() for _ in range(2)
        ]

        vectors = embeddings_manager.generate_embeddings(["one", "two"])

        assert len(vectors) == 2
        assert embeddings_manager.openai_client.embeddings.create.call_count == 3

    def test_generate_embeddings_falls_back_on_count_mismatch(self, embeddings_manager):
        """A response with the wrong number of vectors retries individually."""
        embeddings_manager.openai_client.embeddings.create.side_effect = [
            self._single_response(),
            self._single_response(),
            self._single_response(),
        ]

        vectors = embeddings_manager.generate_embeddings(["one", "two"])

        assert len(vectors) == 2
        assert embeddings_manager.openai_client.embeddings.create.call_count == 3

    def test_generate_embeddings_rate_limit_does_not_fan_out(self, embeddings_manager):
        """A rate-limited batch is not retried as individual requests."""

        class RateLimitError(Exception):
            status_code = 429

        embeddings_manager.openai_client.embeddings.create.side_effect = RateLimitError("rate limited")

        with pytest.raises(EmbeddingsError, match="rate-limited"):
            embeddings_manager.generate_embeddings(["one", "two"])

        assert embeddings_manager.openai_client.embeddings.create.call_count == 1

    @staticmethod
    def _single_response():
        item = Mock()
        item.embedding = [0.1] * 4096
        response = Mock()
        response.data = [item]
        return response

    def test_create_collection(self, embeddings_manager):
        """Test creating a collection."""
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
        embeddings_manager.create_collection()
        embeddings_manager.create_collection(reset=True)
        assert embeddings_manager.collection is not None
        embeddings_manager.close()

    def test_delete_embeddings_by_ids_empty(self, embeddings_manager):
        """delete_embeddings_by_ids returns 0 for an empty list."""
        embeddings_manager.create_collection()
        assert embeddings_manager.delete_embeddings_by_ids([]) == 0
        embeddings_manager.close()

    def test_delete_embeddings_by_ids(self, embeddings_manager, mock_lm_studio):
        """delete_embeddings_by_ids removes only the supplied UIDs."""
        embeddings_manager.create_collection()

        papers = [
            {"uid": "p1", "title": "Paper 1", "abstract": "A"},
            {"uid": "p2", "title": "Paper 2", "abstract": "B"},
            {"uid": "p3", "title": "Paper 3", "abstract": "C"},
        ]
        for paper in papers:
            embeddings_manager.add_paper(paper)

        assert embeddings_manager.collection.count() == 3
        deleted = embeddings_manager.delete_embeddings_by_ids(["p1", "p3"])
        assert deleted == 2
        assert embeddings_manager.collection.count() == 1
        assert embeddings_manager.collection.get()["ids"] == ["p2"]
        embeddings_manager.close()

    def test_delete_embeddings_by_ids_unknown_ids(self, embeddings_manager, mock_lm_studio):
        """delete_embeddings_by_ids ignores unknown UIDs."""
        embeddings_manager.create_collection()

        embeddings_manager.add_paper({"uid": "p1", "title": "Paper 1", "abstract": "A"})
        deleted = embeddings_manager.delete_embeddings_by_ids(["missing"])
        assert deleted == 1
        assert embeddings_manager.collection.count() == 1
        embeddings_manager.close()

    def test_add_paper(self, embeddings_manager, mock_lm_studio):
        """Test adding a paper."""
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

    def test_add_paper_updates_existing(self, embeddings_manager, mock_lm_studio):
        """Test that add_paper replaces the embedding when a paper's content changes."""
        embeddings_manager.create_collection()

        uid = "update_paper"
        embeddings_manager.add_paper({"uid": uid, "title": "Title", "abstract": "Old abstract"})
        old_hash = embeddings_manager.collection.get(ids=[uid])["documents"][0]

        embeddings_manager.add_paper({"uid": uid, "title": "Title", "abstract": "New abstract"})

        stored = embeddings_manager.collection.get(ids=[uid])
        assert embeddings_manager.collection.count() == 1
        assert stored["documents"][0] != old_hash
        assert stored["documents"][0] == hash_text("Title\n\nNew abstract")
        embeddings_manager.close()

    def test_add_paper_empty_abstract(self, embeddings_manager, mock_lm_studio):
        """Test adding a paper with empty abstract but non-empty title."""
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
        query = QueryEmbedding("machine learning", embeddings_manager)
        results = embeddings_manager.search_similar(query, limit=2)

        assert len(results) <= 2
        assert all(isinstance(uid, str) and isinstance(distance, float) for uid, distance in results)
        embeddings_manager.close()

    def test_search_similar_empty_query(self, embeddings_manager):
        """Test search with empty query."""
        embeddings_manager.create_collection()

        with pytest.raises(EmbeddingsError, match="Query cannot be empty"):
            embeddings_manager.search_similar(QueryEmbedding("", embeddings_manager))

        embeddings_manager.close()

    def test_search_similar_auto_initializes(self, embeddings_manager, mock_lm_studio):
        """Test that search_similar auto-initializes client and collection if needed."""
        # Should not raise - auto-initializes (collection is empty so 0 results)
        results = embeddings_manager.search_similar(QueryEmbedding("test query", embeddings_manager), limit=1)
        assert embeddings_manager._client is not None
        assert embeddings_manager._collection is not None
        assert results == []
        embeddings_manager.close()

    def test_get_collection_stats(self, embeddings_manager, mock_lm_studio):
        """Test getting collection statistics."""
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
        embeddings_manager.create_collection()

        count = embeddings_manager.embed_from_database()

        # Should embed all 3 papers (title is included even if abstract is empty)
        assert count == 3

        stats = embeddings_manager.get_collection_stats()
        assert stats["count"] == 3
        embeddings_manager.close()

    def test_embed_from_database_with_filter(self, embeddings_manager, test_database, mock_lm_studio):
        """Test embedding papers from database with filter."""
        embeddings_manager.create_collection()

        count = embeddings_manager.embed_from_database(where_filters={"session": {"$contains": "ML"}})

        # Should only embed papers in ML sessions with non-empty abstracts (paper1)
        assert count == 1

        stats = embeddings_manager.get_collection_stats()
        assert stats["count"] == 1
        embeddings_manager.close()

    def test_embed_from_database_creates_empty_database(self, embeddings_manager, tmp_path):
        """Embedding from a non-existent database creates it and embeds nothing."""

        # Set PAPER_DB to a nonexistent database
        nonexistent_db = tmp_path / "nonexistent.db"
        set_test_db(nonexistent_db)

        embeddings_manager.create_collection()

        count = embeddings_manager.embed_from_database()

        assert count == 0
        assert nonexistent_db.exists()

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

        embeddings_manager.create_collection()

        count = embeddings_manager.embed_from_database()

        # Should embed all papers (title is used even if abstract is empty)
        assert count == 3
        embeddings_manager.close()

    def test_embed_from_database_sql_error(self, embeddings_manager, tmp_path, monkeypatch):
        """Test embedding from database with a SQL error."""
        from abstracts_explorer.database import DatabaseManager

        db_path = tmp_path / "bad.db"
        set_test_db(db_path)

        def fail_query(self, *args, **kwargs):
            raise RuntimeError("simulated query failure")

        monkeypatch.setattr(DatabaseManager, "get_paper", fail_query)

        embeddings_manager.create_collection()

        with pytest.raises(EmbeddingsError, match="Failed to embed from database"):
            embeddings_manager.embed_from_database()

        embeddings_manager.close()

    def test_embed_from_database_without_metadata(self, embeddings_manager, test_database, mock_lm_studio):
        """Embedding a database stores hashes and no metadata."""
        embeddings_manager.create_collection()

        count = embeddings_manager.embed_from_database()
        assert count == 3

        stored = embeddings_manager.collection.get(include=["metadatas", "documents"])
        assert all(metadata is None for metadata in stored["metadatas"])
        assert all(is_content_hash(document) for document in stored["documents"])

        embeddings_manager.close()

    def test_embed_from_database_skip_existing(self, embeddings_manager, test_database, mock_lm_studio):
        """Test that embed_from_database skips papers that already exist."""
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

    def test_embed_from_database_batches_requests(self, embeddings_manager, test_database):
        """Papers are embedded with a single batched API request."""
        embeddings_manager.create_collection()

        count = embeddings_manager.embed_from_database()

        assert count == 3
        create = embeddings_manager.openai_client.embeddings.create
        assert create.call_count == 1
        assert isinstance(create.call_args.kwargs["input"], list)
        embeddings_manager.close()

    def test_embed_from_database_retries_batch_failures_individually(self, embeddings_manager, test_database):
        """A failed batch falls back to individual paper embeddings."""
        embeddings_manager.create_collection()

        calls = {"count": 0}

        def create(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("batch failed")
            item = Mock()
            item.embedding = [0.1] * 4096
            response = Mock()
            response.data = [item]
            return response

        embeddings_manager.openai_client.embeddings.create.side_effect = create

        count = embeddings_manager.embed_from_database()

        assert count == 3
        assert calls["count"] == 4
        embeddings_manager.close()


def test_check_model_compatibility_no_database(embeddings_manager, tmp_path):
    """Checking model compatibility creates an empty database and reports no stored model."""

    non_existent_db = tmp_path / "nonexistent.db"
    set_test_db(non_existent_db)

    compatible, stored_model, current_model = embeddings_manager.check_model_compatibility()

    assert compatible is True
    assert stored_model is None
    assert current_model == embeddings_manager.model_name
    assert non_existent_db.exists()


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
    embeddings_manager.create_collection()
    embeddings_manager.embed_from_database()

    # Test search without filter - should return both papers
    with DatabaseManager() as db:
        results_all = _search_semantic(db, "paper", limit=10)
        assert len(results_all) == 2, f"Expected 2 papers without filter, got {len(results_all)}"

    # Test search with year filter [2024] - should return only 2024 paper
    with DatabaseManager() as db:
        results_2024 = _search_semantic(db, "paper", limit=10, years=[2024])
        assert len(results_2024) == 1, f"Expected 1 paper with year=2024 filter, got {len(results_2024)}"
        assert results_2024[0]["year"] == 2024, f"Expected year 2024, got {results_2024[0]['year']}"

    # Test search with year filter [2025] - should return only 2025 paper
    with DatabaseManager() as db:
        results_2025 = _search_semantic(db, "paper", limit=10, years=[2025])
        assert len(results_2025) == 1, f"Expected 1 paper with year=2025 filter, got {len(results_2025)}"
        assert results_2025[0]["year"] == 2025, f"Expected year 2025, got {results_2025[0]['year']}"

    # Test search with multiple year filters - should return both papers
    with DatabaseManager() as db:
        results_both = _search_semantic(db, "paper", limit=10, years=[2024, 2025])
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

    embeddings_manager.create_collection()
    embeddings_manager.embed_from_database()

    # This query previously triggered:
    #   "Expected where operator … got $contains in query."
    with DatabaseManager() as db:
        results = _search_semantic(
            db,
            'authors:"Vaswani" attention',
            limit=10,
        )

    # Should match exactly the Vaswani paper (partial author match)
    assert len(results) == 1, f"Expected 1 result, got {len(results)}: {[r.get('title') for r in results]}"
    assert results[0]["title"] == "Attention is All You Need"

    # Confirm the non-matching paper is excluded
    with DatabaseManager() as db:
        no_results = _search_semantic(
            db,
            'authors:"Lecun" attention',
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
    embeddings_manager.create_collection()
    # Don't embed papers - field-filter-only queries should not need ChromaDB

    # Query with only a field filter (no remaining keywords)
    with DatabaseManager() as db:
        results = _search_semantic(
            db,
            'authors:"Vaswani"',
            limit=10,
        )

    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    assert results[0]["title"] == "Attention is All You Need"
    # Authors should be parsed into a list
    assert isinstance(results[0]["authors"], list)
    assert "Vaswani" in results[0]["authors"]

    # Test with the 'author' alias
    with DatabaseManager() as db:
        results = _search_semantic(
            db,
            'author:"Vaswani"',
            limit=10,
        )

    assert len(results) == 1
    assert results[0]["title"] == "Attention is All You Need"

    # Test with no matching results
    with DatabaseManager() as db:
        results = _search_semantic(
            db,
            'authors:"Nonexistent"',
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

    embeddings_manager.create_collection()

    # Both papers by John Smith
    with DatabaseManager() as db:
        results = _search_semantic(
            db,
            'authors:"John Smith"',
            limit=10,
        )
    assert len(results) == 2

    # Filter by year
    with DatabaseManager() as db:
        results = _search_semantic(
            db,
            'authors:"John Smith"',
            limit=10,
            years=[2024],
        )
    assert len(results) == 1
    assert results[0]["year"] == 2024

    # Filter by conference
    with DatabaseManager() as db:
        results = _search_semantic(
            db,
            'authors:"John Smith"',
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

    embeddings_manager.create_collection()
    embeddings_manager.embed_from_database()

    # Search for "LeCun" without the author: field prefix — should still find
    # the LeCun paper first (implicit author match).
    with DatabaseManager() as db:
        results = _search_semantic(
            db,
            "LeCun",
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

    embeddings_manager.create_collection()
    embeddings_manager.embed_from_database()

    with DatabaseManager() as db:
        results = _search_semantic(
            db,
            "LeCun",
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

    embeddings_manager.create_collection()
    embeddings_manager.embed_from_database()

    # Query that does NOT match any author name
    with DatabaseManager() as db:
        results = _search_semantic(
            db,
            "transformer architecture",
            limit=10,
        )

    # Should return the paper via semantic search (no author match, but semantically similar)
    assert len(results) >= 1
    assert results[0]["title"] == "Transformer Paper"

    embeddings_manager.close()


def test_search_papers_semantic_distance_threshold(embeddings_manager, tmp_path, mock_lm_studio):
    """Test that search_papers_semantic filters results by distance_threshold.

    Papers with distance greater than the threshold must be excluded from the
    results, matching the behaviour of query_ids_within_distance.
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

    embeddings_manager.create_collection()
    embeddings_manager.embed_from_database()

    # Mock search_similar to return controlled distances:
    # close_paper has distance 0.5 (within threshold), far_paper has distance 1.5 (outside)
    mock_results = [(close_uid, 0.5), (far_uid, 1.5)]

    def fake_query_ids(query, distance_threshold=1.1, ids=None, limit=None):
        return [(uid, distance) for uid, distance in mock_results if distance <= distance_threshold]

    with mock_patch.object(embeddings_manager, "query_ids_within_distance", side_effect=fake_query_ids):
        with DatabaseManager() as db:
            # With default threshold (1.1): only the close paper should be returned
            results = _search_semantic(db, "query", limit=10, embeddings_manager=embeddings_manager)
            titles = [r["title"] for r in results]
            assert "Close Paper" in titles, "Close paper (distance 0.5) should be within threshold 1.1"
            assert "Far Paper" not in titles, "Far paper (distance 1.5) should be outside threshold 1.1"

        with DatabaseManager() as db:
            # With higher threshold (2.0): both papers should be returned
            results_high = _search_semantic(
                db, "query", limit=10, distance_threshold=2.0, embeddings_manager=embeddings_manager
            )
            titles_high = [r["title"] for r in results_high]
            assert "Close Paper" in titles_high, "Close paper should be returned with threshold 2.0"
            assert "Far Paper" in titles_high, "Far paper (distance 1.5) should be within threshold 2.0"

        with DatabaseManager() as db:
            # With very low threshold (0.1): no papers should be returned
            results_none = _search_semantic(
                db, "query", limit=10, distance_threshold=0.1, embeddings_manager=embeddings_manager
            )
            assert len(results_none) == 0, "No papers should be returned with threshold 0.1"

    embeddings_manager.close()


class TestContentHashing:
    """Tests for ChromaDB content hashing."""

    def test_hash_text_format(self):
        digest = hash_text("hello")
        assert digest.startswith("sha256:")
        assert len(digest) == 71
        assert is_content_hash(digest)
        assert not is_content_hash("hello")
        assert not is_content_hash(None)

    def test_paper_text_hash(self):
        paper = {"uid": "p1", "title": "Title", "abstract": "Abstract"}
        text = EmbeddingsManager.embedding_text_from_paper(paper)
        assert hash_text(text) == hash_text("Title\n\nAbstract")


class TestImportExport:
    """Tests for the v1 embeddings artifact format."""

    def test_export_import_round_trip(self, embeddings_manager, mock_lm_studio):
        embeddings_manager.create_collection()
        papers = [
            {"uid": "p1", "title": "Paper 1", "abstract": "Abstract 1"},
            {"uid": "p2", "title": "Paper 2", "abstract": "Abstract 2"},
        ]
        for paper in papers:
            embeddings_manager.add_paper(paper)

        exported = embeddings_manager.export_embeddings(["p1", "p2"])
        assert exported["format_version"] == 1
        assert exported["ids"] == ["p1", "p2"]
        assert all(is_content_hash(doc) for doc in exported["documents"])
        assert "metadatas" not in exported

        embeddings_manager.delete_embeddings_by_ids(["p1", "p2"])
        imported = embeddings_manager.import_embeddings(exported)
        assert imported == 2
        assert embeddings_manager.collection.count() == 2
        embeddings_manager.close()

    def test_export_includes_documents(self, embeddings_manager):
        embeddings_manager.create_collection()
        embeddings_manager.collection.add(
            ids=["p1"],
            embeddings=[[0.1] * 4096],
            documents=["full text"],
        )
        exported = embeddings_manager.export_embeddings(["p1"])
        assert exported["ids"] == ["p1"]
        assert exported["documents"] == ["full text"]
        embeddings_manager.close()

    def test_import_rejects_legacy_format(self, embeddings_manager):
        embeddings_manager.create_collection()
        with pytest.raises(EmbeddingsError, match="Unsupported"):
            embeddings_manager.import_embeddings({"ids": ["p1"], "documents": ["x"], "embeddings": [[0.1]]})
        embeddings_manager.close()

    def test_import_replaces_ids(self, embeddings_manager, mock_lm_studio):
        embeddings_manager.create_collection()
        embeddings_manager.add_paper({"uid": "old", "title": "Old", "abstract": "Old abstract"})
        data = {
            "format_version": 1,
            "ids": ["new"],
            "documents": [hash_text("New\n\nNew abstract")],
            "embeddings": [[0.1] * 4096],
        }
        imported = embeddings_manager.import_embeddings(data, replace_ids=["old"])
        assert imported == 1
        assert embeddings_manager.collection.count() == 1
        assert embeddings_manager.collection.get()["ids"] == ["new"]
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
        from abstracts_explorer.rate_limiter import CooldownAwareTransport, RateLimitedTransport
        from abstracts_explorer.rate_limiter import (
            TokenBucketRateLimiter,
            set_global_rate_limiter,
        )

        set_global_rate_limiter(None)

        monkeypatch.setenv("EMBEDDING_DB", str(tmp_path / "chroma"))
        config = get_config(reload=True, env_path=get_env_test_path())

        em = EmbeddingsManager(requests_per_minute=30)
        # Trigger lazy initialization of the OpenAI client
        client = em.openai_client
        transport = client._client._transport
        assert isinstance(transport, CooldownAwareTransport)
        assert isinstance(transport._transport, RateLimitedTransport)
        assert isinstance(transport._transport._rate_limiter, TokenBucketRateLimiter)
        assert transport._transport._rate_limiter.requests_per_minute == 30
        assert client.max_retries == config.llm_max_attempts - 1
        assert client._client.timeout.connect == 10
        assert client._client.timeout.read == 60

    def test_no_rate_limited_transport_when_rpm_zero(self, tmp_path, monkeypatch):
        """Test that no RateLimitedTransport is used when requests_per_minute=0."""
        from tests.conftest import get_env_test_path
        from abstracts_explorer.config import get_config
        from abstracts_explorer.rate_limiter import CooldownAwareTransport, RateLimitedTransport

        monkeypatch.setenv("EMBEDDING_DB", str(tmp_path / "chroma"))
        get_config(reload=True, env_path=get_env_test_path())

        em = EmbeddingsManager(requests_per_minute=0)
        client = em.openai_client
        # The cooldown wrapper is always present, but rate limiting is disabled.
        transport = getattr(getattr(client, "_client", None), "_transport", None)
        assert isinstance(transport, CooldownAwareTransport)
        assert not isinstance(transport._transport, RateLimitedTransport)
        assert client._client.timeout.connect == 10
        assert client._client.timeout.read == 60

    def test_rate_limited_transport_sleeps_when_tokens_exhausted(self, monkeypatch):
        """Test that RateLimitedTransport blocks when token bucket is empty."""
        import httpx2
        from abstracts_explorer.rate_limiter import RateLimitedTransport
        from abstracts_explorer.rate_limiter import set_global_rate_limiter

        clock = FakeTime()
        monkeypatch.setattr("abstracts_explorer.rate_limiter.time", clock)

        set_global_rate_limiter(None)

        mock_inner = Mock(spec=httpx2.BaseTransport)
        mock_inner.handle_request.return_value = Mock(spec=httpx2.Response)

        transport = RateLimitedTransport(mock_inner, requests_per_minute=60)
        transport._rate_limiter._tokens = 0.0
        transport._rate_limiter._last_update = clock.monotonic()

        transport.handle_request(Mock(spec=httpx2.Request))

        assert sum(clock.sleeps) >= 0.8
        mock_inner.handle_request.assert_called_once()

    def test_rate_limited_transport_no_wait_when_tokens_available(self):
        """Test that RateLimitedTransport does not wait when tokens are available."""
        import httpx2
        from abstracts_explorer.rate_limiter import RateLimitedTransport
        from abstracts_explorer.rate_limiter import set_global_rate_limiter

        set_global_rate_limiter(None)

        mock_inner = Mock(spec=httpx2.BaseTransport)
        mock_inner.handle_request.return_value = Mock(spec=httpx2.Response)

        transport = RateLimitedTransport(mock_inner, requests_per_minute=60)
        transport._rate_limiter._tokens = 60.0

        start = time.monotonic()
        transport.handle_request(Mock(spec=httpx2.Request))
        elapsed = time.monotonic() - start

        assert elapsed < 0.1
        mock_inner.handle_request.assert_called_once()

    def test_rate_limited_transport_forwards_response(self):
        """Test that RateLimitedTransport forwards the response from inner transport."""
        import httpx2
        from abstracts_explorer.rate_limiter import RateLimitedTransport
        from abstracts_explorer.rate_limiter import set_global_rate_limiter

        set_global_rate_limiter(None)

        mock_inner = Mock(spec=httpx2.BaseTransport)
        mock_response = Mock(spec=httpx2.Response)
        mock_inner.handle_request.return_value = mock_response

        transport = RateLimitedTransport(mock_inner, requests_per_minute=60)
        transport._rate_limiter._tokens = 60.0

        result = transport.handle_request(Mock(spec=httpx2.Request))

        assert result is mock_response
        mock_inner.handle_request.assert_called_once()


class TestQueryEmbedding:
    """Tests for lazy query embeddings."""

    def test_embedding_is_generated_once(self):
        manager = Mock()
        manager.generate_embedding.return_value = [0.1, 0.2]
        query = QueryEmbedding("test query", manager)

        assert query.embedding == [0.1, 0.2]
        assert query.embedding == [0.1, 0.2]
        manager.generate_embedding.assert_called_once_with("test query")

    def test_embedding_retries_after_failure(self):
        manager = Mock()
        manager.generate_embedding.side_effect = [EmbeddingsError("boom"), [0.3]]
        query = QueryEmbedding("test query", manager)

        with pytest.raises(EmbeddingsError, match="boom"):
            _ = query.embedding

        assert query.embedding == [0.3]
        assert manager.generate_embedding.call_count == 2


class TestConcurrentClientCreation:
    """Tests for concurrent ChromaDB client creation."""

    def test_client_creation_is_serialized_across_managers(self, embeddings_manager, monkeypatch):
        """Concurrent managers must not create ChromaDB clients at the same time."""
        active = 0
        max_active = 0
        counter_lock = threading.Lock()
        barrier = threading.Barrier(2)

        def fake_persistent_client(*args, **kwargs):
            nonlocal active, max_active
            with counter_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with counter_lock:
                active -= 1
            return Mock()

        monkeypatch.setattr(
            "abstracts_explorer.embeddings.chroma.chromadb.PersistentClient",
            fake_persistent_client,
        )

        managers = [
            EmbeddingsManager(collection_name="test_collection"),
            EmbeddingsManager(collection_name="test_collection"),
        ]
        errors = []

        def connect(manager):
            barrier.wait(timeout=5)
            try:
                _ = manager.client
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=connect, args=(manager,)) for manager in managers]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        assert all(not thread.is_alive() for thread in threads)
        assert errors == []
        assert max_active == 1


class TestLocalChromaLock:
    """Tests for the exclusive process-lifetime lock on local ChromaDB."""

    @staticmethod
    def _foreign_lock(manager) -> ReadWriteLock:
        lock_path = Path(manager.embedding_db) / ".chroma.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock = ReadWriteLock(str(lock_path), timeout=0, blocking=False, is_singleton=False)
        lock.acquire_write(timeout=0)
        return lock

    def test_foreign_holder_is_rejected(self, embeddings_manager):
        """A local client fails fast when another process holds the lock."""
        foreign = self._foreign_lock(embeddings_manager)
        try:
            with pytest.raises(EmbeddingsError, match="(?i)another process"):
                _ = embeddings_manager.client
        finally:
            foreign.release()

    def test_lock_is_released_on_close(self, embeddings_manager):
        """Closing the manager releases the lock so another process can take it."""
        _ = embeddings_manager.client
        embeddings_manager.close()

        foreign = self._foreign_lock(embeddings_manager)
        foreign.release()

    def test_same_process_managers_share_the_lock(self, tmp_path, monkeypatch):
        """Multiple managers in one process may connect to the same local path."""
        from abstracts_explorer.config import get_config

        chroma_path = tmp_path / "shared_chroma"
        monkeypatch.setenv("EMBEDDING_DB", str(chroma_path))
        get_config(reload=True)

        first = EmbeddingsManager()
        second = EmbeddingsManager()
        try:
            assert first.client is not None
            assert second.client is not None
        finally:
            first.close()
            second.close()

    def test_http_mode_does_not_take_the_local_lock(self, monkeypatch):
        """HTTP backends rely on the server for concurrency and need no lock."""
        import abstracts_explorer.embeddings.chroma as chroma_module

        monkeypatch.setattr(chroma_module.chromadb, "HttpClient", Mock(return_value=Mock()))

        manager = EmbeddingsManager.__new__(EmbeddingsManager)
        manager.embedding_db = "http://localhost:8000"
        manager._client = None
        manager._collection = None
        manager._chroma_lock_path = None
        manager._init_lock = threading.RLock()

        manager._connect()

        assert manager._client is not None
        assert manager._chroma_lock_path is None


class TestDiskSpacePreflight:
    """Tests for the disk-space check before embedding writes."""

    def test_embed_from_database_rejects_low_disk(
        self, embeddings_manager, test_database, mock_lm_studio, monkeypatch
    ):
        """Embedding writes fail with a clear error instead of risking corruption."""
        from abstracts_explorer.utils.disk_utils import InsufficientDiskSpaceError

        def _raise(*args, **kwargs):
            raise InsufficientDiskSpaceError("disk full")

        monkeypatch.setattr("abstracts_explorer.embeddings.ingest.ensure_free_space", _raise)

        with pytest.raises(EmbeddingsError, match="disk full"):
            embeddings_manager.embed_from_database()

        assert embeddings_manager.get_collection_stats()["count"] == 0
        embeddings_manager.close()


class TestDistanceHelpers:
    """Tests for ID-based distance helpers."""

    def _make_manager(self):
        manager = object.__new__(EmbeddingsManager)
        manager.generate_embedding = Mock(return_value=[0.1])
        manager._collection = Mock()
        manager._collection.count.return_value = 10
        manager._collection.query.return_value = {
            "ids": [["paper-a", "paper-b", "paper-c"]],
            "distances": [[0.1, 0.2, 2.0]],
        }
        return manager

    def test_query_ids_within_distance(self):
        manager = self._make_manager()
        result = manager.query_ids_within_distance(
            QueryEmbedding("query", manager),
            distance_threshold=1.0,
        )
        assert result == [("paper-a", 0.1), ("paper-b", 0.2)]
        assert manager._collection.query.call_args.kwargs["include"] == ["distances"]

    def test_query_ids_within_distance_overfetches_for_limit(self):
        """limit applies after threshold filtering, so Chroma gets limit * 2 candidates."""
        manager = self._make_manager()
        manager._collection.count.return_value = 100
        manager._collection.query.return_value = {
            "ids": [["paper-a", "paper-b", "paper-c", "paper-d"]],
            "distances": [[0.1, 0.2, 0.3, 0.9]],
        }

        result = manager.query_ids_within_distance(
            QueryEmbedding("query", manager),
            distance_threshold=0.5,
            limit=2,
        )

        assert result == [("paper-a", 0.1), ("paper-b", 0.2)]
        assert manager._collection.query.call_args.kwargs["n_results"] == 4

    def test_query_ids_within_distance_reuses_query_embedding(self):
        manager = self._make_manager()
        query = QueryEmbedding("query", manager)
        query._vector = [0.5, 0.6]
        manager.query_ids_within_distance(query, distance_threshold=1.0)
        manager.generate_embedding.assert_not_called()
        assert manager._collection.query.call_args.kwargs["query_embeddings"] == [[0.5, 0.6]]

    def test_query_ids_within_distance_with_ids(self):
        manager = self._make_manager()
        manager.query_ids_within_distance(
            QueryEmbedding("query", manager),
            distance_threshold=1.0,
            ids=["paper-a", "paper-b"],
        )
        assert manager._collection.query.call_args.kwargs["ids"] == ["paper-a", "paper-b"]

    def test_large_id_lists_are_queried_in_batches(self):
        manager = self._make_manager()
        ids = [f"paper-{index}" for index in range(600)]

        def fake_query(**kwargs):
            batch = kwargs["ids"]
            return {"ids": [batch], "distances": [[0.1] * len(batch)]}

        manager._collection.query.side_effect = fake_query

        result = manager.query_ids_within_distance(
            QueryEmbedding("query", manager),
            distance_threshold=1.0,
            ids=ids,
        )

        assert manager._collection.query.call_count == 2
        assert len(result) == 600

    def test_empty_ids_returns_empty(self):
        manager = self._make_manager()
        assert manager.query_ids_within_distance(QueryEmbedding("query", manager), ids=[]) == []

    def test_query_ids_reject_empty_query(self):
        manager = self._make_manager()
        with pytest.raises(EmbeddingsError, match="Query cannot be empty"):
            manager.query_ids_within_distance(QueryEmbedding("  ", manager), distance_threshold=1.0)

    def test_query_ids_returns_empty_when_no_results(self):
        manager = self._make_manager()
        manager._collection.query.return_value = {"ids": [[]], "distances": [[]]}
        assert manager.query_ids_within_distance(QueryEmbedding("query", manager)) == []


class TestEmbeddingInputTruncation:
    """Tests for truncating inputs that exceed the embedding context window."""

    @staticmethod
    def _long_text() -> str:
        return " ".join(f"token{i}" for i in range(5000))

    def test_short_text_is_not_truncated(self, embeddings_manager):
        """Text within the budget is sent unchanged."""
        em = embeddings_manager
        em.max_input_tokens = 128
        text = "A short abstract about machine learning."

        em.generate_embedding(text)

        sent = em.openai_client.embeddings.create.call_args.kwargs["input"]
        assert sent == text

    def test_long_text_is_truncated_before_api_call(self, embeddings_manager):
        """Text above the budget is shortened before the API call."""
        em = embeddings_manager
        em.max_input_tokens = 100
        text = self._long_text()

        em.generate_embedding(text)

        sent = em.openai_client.embeddings.create.call_args.kwargs["input"]
        assert sent != text
        token_count = _count_tokens(sent)
        if token_count is not None:
            assert token_count <= em._effective_token_budget()

    def test_truncation_can_be_disabled(self, embeddings_manager):
        """A zero token budget disables proactive truncation."""
        em = embeddings_manager
        em.max_input_tokens = 0
        text = self._long_text()

        em.generate_embedding(text)

        sent = em.openai_client.embeddings.create.call_args.kwargs["input"]
        assert sent == text

    def test_add_paper_stores_untruncated_document(self, embeddings_manager):
        """The full abstract is stored while only the embedding input is truncated."""
        em = embeddings_manager
        em.max_input_tokens = 100
        abstract = self._long_text()
        paper = {"uid": "long_paper", "title": "Long Paper", "abstract": abstract}

        em.create_collection()
        em.add_paper(paper)

        stored = em.collection.get(ids=["long_paper"])
        assert stored["documents"][0] == hash_text(f"Long Paper\n\n{abstract}".strip())
        assert stored["metadatas"][0] is None

        sent = em.openai_client.embeddings.create.call_args.kwargs["input"]
        assert len(sent) < len(abstract)
        em.close()

    def test_context_length_error_retries_with_shorter_input(self, embeddings_manager):
        """A backend context-length error triggers a retry with a smaller input."""
        em = embeddings_manager
        em.max_input_tokens = 0
        text = self._long_text()
        response = Mock()
        item = Mock()
        item.embedding = [0.1] * 4096
        response.data = [item]
        em.openai_client.embeddings.create.side_effect = [
            Exception(
                "Error code: 400 - This model's maximum context length is 100 tokens. "
                "However, your prompt contains at least 101 input tokens."
            ),
            response,
        ]

        em.generate_embedding(text)

        assert em.openai_client.embeddings.create.call_count == 2
        first = em.openai_client.embeddings.create.call_args_list[0].kwargs["input"]
        second = em.openai_client.embeddings.create.call_args_list[1].kwargs["input"]
        assert len(second) < len(first)
        assert _count_tokens("probe") is None or em._learned_token_budget is not None

    def test_non_context_error_is_not_retried(self, embeddings_manager):
        """Unrelated API errors fail on the first attempt."""
        em = embeddings_manager
        em.openai_client.embeddings.create.side_effect = Exception("API error")

        with pytest.raises(EmbeddingsError, match="Failed to generate embedding"):
            em.generate_embedding("Some text")

        assert em.openai_client.embeddings.create.call_count == 1


class TestCentralLLMRetries:
    """The SDK owns retries; attempts come from config and the cooldown wraps it."""

    def test_sdk_retries_derived_from_max_attempts(self, embeddings_manager, monkeypatch):
        """The OpenAI client gets ``LLM_MAX_ATTEMPTS - 1`` retries."""
        created = {}

        def fake_openai(**kwargs):
            created.update(kwargs)
            return Mock()

        monkeypatch.setattr("abstracts_explorer.embeddings.models.OpenAI", fake_openai)
        monkeypatch.setattr("abstracts_explorer.embeddings.models.httpx2.Client", Mock())
        embeddings_manager.requests_per_minute = 0
        embeddings_manager.llm_max_attempts = 6

        embeddings_manager._create_openai_client()

        assert created["max_retries"] == 5

    def test_cooldown_transport_always_wraps_client(self, embeddings_manager, monkeypatch):
        """Every client has the shared-cooldown transport, even without rate limiting."""
        from abstracts_explorer.rate_limiter import CooldownAwareTransport, RateLimitedTransport

        captured = {}

        def fake_client(**kwargs):
            captured.update(kwargs)
            return Mock()

        monkeypatch.setattr("abstracts_explorer.embeddings.models.OpenAI", Mock())
        monkeypatch.setattr("abstracts_explorer.embeddings.models.httpx2.Client", fake_client)
        embeddings_manager.requests_per_minute = 0

        embeddings_manager._create_openai_client()

        transport = captured["transport"]
        assert isinstance(transport, CooldownAwareTransport)
        assert not isinstance(transport._transport, RateLimitedTransport)

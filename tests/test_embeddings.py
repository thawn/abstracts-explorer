"""
Tests for the embeddings module.
"""

import json
import pytest
import sqlite3
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from neurips_abstracts.embeddings import EmbeddingsManager, EmbeddingsError

# Fixtures imported from conftest.py:
# - mock_lm_studio: Mock LM Studio API responses
# - embeddings_manager: EmbeddingsManager instance for testing


@pytest.fixture
def test_database(tmp_path):
    """Create a test database with sample papers."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Create papers table
    cursor.execute(
        """
        CREATE TABLE papers (
            id INTEGER PRIMARY KEY,
            name TEXT,
            abstract TEXT,
            authors TEXT,
            topic TEXT,
            keywords TEXT,
            decision TEXT,
            paper_url TEXT,
            poster_position TEXT
        )
    """
    )

    # Insert test data
    test_papers = [
        (
            1,
            "Deep Learning Paper",
            "This paper presents a novel deep learning approach.",
            "John Doe, Jane Smith",
            "Machine Learning",
            "deep learning, neural networks",
            "Accept",
            "https://papers.nips.cc/paper/1",
            "A12",
        ),
        (
            2,
            "NLP Paper",
            "We introduce a new natural language processing method.",
            "Alice Johnson",
            "Natural Language Processing",
            "NLP, transformers",
            "Accept",
            "https://papers.nips.cc/paper/2",
            "B05",
        ),
        (
            3,
            "Computer Vision Paper",
            "",  # Empty abstract
            "Bob Wilson",
            "Computer Vision",
            "vision, CNN",
            "Reject",
            "",
            "",
        ),
    ]

    cursor.executemany(
        """
        INSERT INTO papers (id, name, abstract, authors, topic, keywords, decision, paper_url, poster_position)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        test_papers,
    )

    conn.commit()
    conn.close()
    return db_path


# Note: The above test_database fixture is kept here as it has a slightly different
# schema than the one in conftest.py. This fixture includes specific fields
# (paper_url, poster_position) that are needed for embeddings tests.


class TestEmbeddingsManager:
    """Tests for EmbeddingsManager class."""

    def test_init(self, embeddings_manager):
        """Test EmbeddingsManager initialization."""
        assert embeddings_manager.lm_studio_url == "http://localhost:1234"
        assert embeddings_manager.model_name == "text-embedding-qwen3-embedding-4b"
        assert embeddings_manager.collection_name == "test_collection"
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

    def test_test_lm_studio_connection_success(self, embeddings_manager, mock_lm_studio):
        """Test successful LM Studio connection."""
        result = embeddings_manager.test_lm_studio_connection()
        assert result is True
        mock_lm_studio.get.assert_called_once()

    def test_test_lm_studio_connection_failure(self, embeddings_manager):
        """Test failed LM Studio connection."""
        with patch("neurips_abstracts.embeddings.requests.get") as mock_get:
            import requests

            mock_get.side_effect = requests.exceptions.RequestException("Connection error")
            result = embeddings_manager.test_lm_studio_connection()
            assert result is False

    def test_generate_embedding_success(self, embeddings_manager, mock_lm_studio):
        """Test successful embedding generation."""
        embedding = embeddings_manager.generate_embedding("Test text")
        assert isinstance(embedding, list)
        assert len(embedding) == 4096
        mock_lm_studio.post.assert_called_once()

    def test_generate_embedding_empty_text(self, embeddings_manager):
        """Test embedding generation with empty text."""
        with pytest.raises(EmbeddingsError, match="Cannot generate embedding for empty text"):
            embeddings_manager.generate_embedding("")

    def test_generate_embedding_api_error(self, embeddings_manager):
        """Test embedding generation with API error."""
        with patch("neurips_abstracts.embeddings.requests.post") as mock_post:
            import requests

            mock_post.side_effect = requests.exceptions.RequestException("API error")
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

        embeddings_manager.add_paper(
            paper_id=1,
            abstract="Test abstract",
            metadata={"title": "Test Paper", "authors": "John Doe"},
        )

        stats = embeddings_manager.get_collection_stats()
        assert stats["count"] == 1
        embeddings_manager.close()

    def test_add_paper_with_embedding(self, embeddings_manager):
        """Test adding a paper with pre-computed embedding."""
        embeddings_manager.connect()
        embeddings_manager.create_collection()

        embedding = [0.1] * 4096
        embeddings_manager.add_paper(
            paper_id=1,
            abstract="Test abstract",
            metadata={"title": "Test Paper"},
            embedding=embedding,
        )

        stats = embeddings_manager.get_collection_stats()
        assert stats["count"] == 1
        embeddings_manager.close()

    def test_add_paper_empty_abstract(self, embeddings_manager):
        """Test adding a paper with empty abstract."""
        embeddings_manager.connect()
        embeddings_manager.create_collection()

        # Should log warning but not raise error
        embeddings_manager.add_paper(paper_id=1, abstract="")

        stats = embeddings_manager.get_collection_stats()
        assert stats["count"] == 0
        embeddings_manager.close()

    def test_add_paper_collection_not_initialized(self, embeddings_manager):
        """Test adding paper without collection."""
        with pytest.raises(EmbeddingsError, match="Collection not initialized"):
            embeddings_manager.add_paper(paper_id=1, abstract="Test")

    def test_add_papers_batch(self, embeddings_manager, mock_lm_studio):
        """Test batch adding papers."""
        embeddings_manager.connect()
        embeddings_manager.create_collection()

        papers = [
            (1, "Abstract 1", {"title": "Paper 1"}),
            (2, "Abstract 2", {"title": "Paper 2"}),
            (3, "Abstract 3", {"title": "Paper 3"}),
        ]

        embeddings_manager.add_papers_batch(papers, batch_size=2)

        stats = embeddings_manager.get_collection_stats()
        assert stats["count"] == 3
        embeddings_manager.close()

    def test_add_papers_batch_with_empty_abstracts(self, embeddings_manager, mock_lm_studio):
        """Test batch adding papers with some empty abstracts."""
        embeddings_manager.connect()
        embeddings_manager.create_collection()

        papers = [
            (1, "Abstract 1", {"title": "Paper 1"}),
            (2, "", {"title": "Paper 2"}),  # Empty abstract
            (3, "Abstract 3", {"title": "Paper 3"}),
        ]

        embeddings_manager.add_papers_batch(papers)

        stats = embeddings_manager.get_collection_stats()
        assert stats["count"] == 2  # Only 2 papers should be added
        embeddings_manager.close()

    def test_search_similar(self, embeddings_manager, mock_lm_studio):
        """Test similarity search."""
        embeddings_manager.connect()
        embeddings_manager.create_collection()

        # Add some papers
        papers = [
            (1, "Deep learning neural networks", {"title": "DL Paper"}),
            (2, "Natural language processing", {"title": "NLP Paper"}),
        ]
        embeddings_manager.add_papers_batch(papers)

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
        embeddings_manager.add_paper(1, "Test abstract", {"title": "Test"})

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

        # Should embed 2 papers (3rd has empty abstract)
        assert count == 2

        stats = embeddings_manager.get_collection_stats()
        assert stats["count"] == 2
        embeddings_manager.close()

    def test_embed_from_database_with_filter(self, embeddings_manager, test_database, mock_lm_studio):
        """Test embedding papers from database with filter."""
        embeddings_manager.connect()
        embeddings_manager.create_collection()

        count = embeddings_manager.embed_from_database(test_database, where_clause="decision = 'Accept'")

        # Should only embed accepted papers with non-empty abstracts
        assert count == 2

        stats = embeddings_manager.get_collection_stats()
        assert stats["count"] == 2
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

        def progress_callback(n):
            progress_calls.append(n)

        count = embeddings_manager.embed_from_database(
            test_database, batch_size=1, progress_callback=progress_callback
        )

        # Should embed 2 papers (one has empty abstract)
        assert count == 2
        # Progress callback should be called with batch updates
        assert len(progress_calls) > 0
        assert sum(progress_calls) == 2  # Total progress equals embedded count

        embeddings_manager.close()

    def test_embed_from_database_empty_result(self, embeddings_manager, tmp_path, mock_lm_studio):
        """Test embedding from database with no matching papers."""
        # Create empty database
        db_path = tmp_path / "empty.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE papers (
                id INTEGER PRIMARY KEY,
                name TEXT,
                abstract TEXT,
                authors TEXT,
                topic TEXT,
                keywords TEXT,
                decision TEXT,
                paper_url TEXT,
                poster_position TEXT
            )
        """
        )
        conn.commit()
        conn.close()

        embeddings_manager.connect()
        embeddings_manager.create_collection()

        count = embeddings_manager.embed_from_database(db_path)

        assert count == 0
        stats = embeddings_manager.get_collection_stats()
        assert stats["count"] == 0

        embeddings_manager.close()

    def test_embed_from_database_all_empty_abstracts(self, embeddings_manager, tmp_path, mock_lm_studio):
        """Test embedding from database where all papers have empty abstracts."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE papers (
                id INTEGER PRIMARY KEY,
                name TEXT,
                abstract TEXT,
                authors TEXT,
                topic TEXT,
                keywords TEXT,
                decision TEXT,
                paper_url TEXT,
                poster_position TEXT
            )
        """
        )
        # Insert papers with empty or None abstracts
        cursor.executemany(
            "INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (1, "Paper 1", "", "Author", "Topic", "kw", "Accept", "url", "pos"),
                (2, "Paper 2", None, "Author", "Topic", "kw", "Accept", "url", "pos"),
                (3, "Paper 3", "   ", "Author", "Topic", "kw", "Accept", "url", "pos"),
            ],
        )
        conn.commit()
        conn.close()

        embeddings_manager.connect()
        embeddings_manager.create_collection()

        count = embeddings_manager.embed_from_database(db_path)

        # Should skip all papers with empty abstracts
        assert count == 0
        embeddings_manager.close()

    def test_embed_from_database_sql_error(self, embeddings_manager, tmp_path):
        """Test embedding from database with SQL error."""
        # Create database without proper schema
        db_path = tmp_path / "bad.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE papers (id INTEGER PRIMARY KEY)")  # Missing columns
        conn.commit()
        conn.close()

        embeddings_manager.connect()
        embeddings_manager.create_collection()

        with pytest.raises(EmbeddingsError, match="Database error"):
            embeddings_manager.embed_from_database(db_path)

        embeddings_manager.close()

    def test_embed_from_database_with_metadata_fields(self, embeddings_manager, test_database, mock_lm_studio):
        """Test that paper_url and poster_position are included in metadata."""
        embeddings_manager.connect()
        embeddings_manager.create_collection()

        count = embeddings_manager.embed_from_database(test_database)
        assert count == 2

        # Search to verify metadata includes new fields
        results = embeddings_manager.search_similar("test", n_results=2)
        assert len(results["metadatas"][0]) > 0

        # Check both papers have the new metadata fields
        for metadata in results["metadatas"][0]:
            assert "paper_url" in metadata
            assert "poster_position" in metadata

        # Find paper 1's metadata
        paper_1_metadata = None
        for i, paper_id in enumerate(results["ids"][0]):
            if paper_id == "1":
                paper_1_metadata = results["metadatas"][0][i]
                break

        assert paper_1_metadata is not None
        assert paper_1_metadata["paper_url"] == "https://papers.nips.cc/paper/1"
        assert paper_1_metadata["poster_position"] == "A12"

        embeddings_manager.close()

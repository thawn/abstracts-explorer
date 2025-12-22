"""
Shared pytest fixtures for all test modules.

This module contains common fixtures used across multiple test files to reduce
code duplication and ensure consistency in test setup.
"""

import pytest
import sqlite3
from unittest.mock import Mock

from neurips_abstracts.database import DatabaseManager
from neurips_abstracts.embeddings import EmbeddingsManager


@pytest.fixture
def db_manager(tmp_path):
    """
    Create a DatabaseManager instance with a temporary database.

    Parameters
    ----------
    tmp_path : Path
        Pytest's temporary path fixture

    Returns
    -------
    DatabaseManager
        Database manager instance with temporary database
    """
    db_path = tmp_path / "test.db"
    return DatabaseManager(db_path)


@pytest.fixture
def connected_db(db_manager):
    """
    Create and connect to a database with tables created.

    Parameters
    ----------
    db_manager : DatabaseManager
        Database manager fixture

    Yields
    ------
    DatabaseManager
        Connected database manager with tables created

    Notes
    -----
    Automatically closes the database connection after the test.
    """
    db_manager.connect()
    db_manager.create_tables()
    yield db_manager
    db_manager.close()


@pytest.fixture
def sample_neurips_data():
    """
    Sample data with lightweight schema for testing.

    Returns
    -------
    list
        List of paper dictionaries with lightweight schema

    Notes
    -----
    This data uses the lightweight schema with comma-separated authors.
    Includes two papers for testing purposes.
    """
    return [
        {
            "id": 123456,
            "uid": "abc123",
            "title": "Deep Learning with Neural Networks",
            "abstract": "This paper explores deep neural networks",
            "authors": ["Miaomiao Huang", "John Smith"],
            "keywords": ["deep learning", "neural networks"],
            "session": "Session A",
            "poster_position": "A-1",
            "room_name": "Hall A",
            "url": "https://openreview.net/forum?id=abc123",
            "paper_pdf_url": "https://openreview.net/pdf?id=abc123",
            "starttime": "2025-12-10T10:00:00",
            "endtime": "2025-12-10T12:00:00",
            "year": 2025,
            "conference": "NeurIPS",
        },
        {
            "id": 123457,
            "uid": "def456",
            "title": "Advances in Computer Vision",
            "abstract": "This paper discusses computer vision techniques",
            "authors": ["John Smith", "Jane Doe"],
            "keywords": ["computer vision", "image processing"],
            "session": "Session B",
            "poster_position": "B-2",
            "room_name": "Hall B",
            "url": "https://openreview.net/forum?id=def456",
            "paper_pdf_url": "https://openreview.net/pdf?id=def456",
            "starttime": "2025-12-10T14:00:00",
            "endtime": "2025-12-10T16:00:00",
            "award": "Best Paper Award",
            "year": 2025,
            "conference": "NeurIPS",
        },
    ]


@pytest.fixture
def test_database(tmp_path):
    """
    Create a test database with sample papers for testing.

    Parameters
    ----------
    tmp_path : Path
        Pytest's temporary path fixture

    Returns
    -------
    Path
        Path to the test database file

    Notes
    -----
    Creates a database with 3 papers and 3 authors for testing search and
    retrieval functionality.
    """
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Create papers table
    cursor.execute(
        """
        CREATE TABLE papers (
            id INTEGER PRIMARY KEY,
            uid TEXT,
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
            "paper1",
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
            "paper2",
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
            "paper3",
            "Computer Vision Paper",
            "",  # Empty abstract to test edge case
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
        INSERT INTO papers 
        (id, uid, name, abstract, authors, topic, keywords, decision, paper_url, poster_position)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        test_papers,
    )

    conn.commit()
    conn.close()

    return db_path


@pytest.fixture
def mock_lm_studio():
    """
    Mock LM Studio API responses for testing embeddings.

    Yields
    ------
    Mock
        Mock requests module with configured responses

    Notes
    -----
    Mocks both embedding and chat completion endpoints with typical responses.
    """
    from unittest.mock import patch

    with patch("neurips_abstracts.embeddings.requests") as mock_requests:
        # Mock successful embedding response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1] * 4096}],
            "model": "text-embedding-qwen3-embedding-4b",
        }
        mock_response.raise_for_status = Mock()
        mock_requests.post.return_value = mock_response
        mock_requests.get.return_value = mock_response
        yield mock_requests


@pytest.fixture
def embeddings_manager(tmp_path):
    """
    Create an EmbeddingsManager instance for testing.

    Parameters
    ----------
    tmp_path : Path
        Pytest's temporary path fixture

    Returns
    -------
    EmbeddingsManager
        Embeddings manager with temporary ChromaDB path

    Notes
    -----
    Uses a test collection name and temporary ChromaDB storage path.
    """
    chroma_path = tmp_path / "test_chroma"
    return EmbeddingsManager(
        lm_studio_url="http://localhost:1234",
        chroma_path=chroma_path,
        collection_name="test_collection",
    )


@pytest.fixture
def mock_embeddings_manager():
    """
    Create a mock embeddings manager for RAG testing.

    Returns
    -------
    Mock
        Mock EmbeddingsManager with configured search results

    Notes
    -----
    Returns mock search results with 3 papers about transformers and language models.
    Uses integer IDs as required by the database.
    """
    mock_em = Mock(spec=EmbeddingsManager)

    # Mock successful search results with INTEGER IDs
    mock_em.search_similar.return_value = {
        "ids": [[1, 2, 3]],  # Use integer IDs instead of strings
        "distances": [[0.1, 0.2, 0.3]],
        "metadatas": [
            [
                {
                    "title": "Attention Is All You Need",
                    "authors": "Vaswani et al.",
                    "topic": "Deep Learning",
                    "decision": "Accept (oral)",
                    "keywords": "transformers, attention",
                },
                {
                    "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                    "authors": "Devlin et al.",
                    "topic": "Natural Language Processing",
                    "decision": "Accept (poster)",
                    "keywords": "language models, pretraining",
                },
                {
                    "title": "GPT-3: Language Models are Few-Shot Learners",
                    "authors": "Brown et al.",
                    "topic": "Language Models",
                    "decision": "Accept (oral)",
                    "keywords": "large language models, in-context learning",
                },
            ]
        ],
        "documents": [
            [
                "We propose the Transformer, a model architecture...",
                "We introduce BERT, a new language representation model...",
                "We train GPT-3, an autoregressive language model with 175B parameters...",
            ]
        ],
    }

    return mock_em


@pytest.fixture
def mock_response(sample_neurips_data):
    """
    Mock HTTP response for downloader tests.

    Parameters
    ----------
    sample_neurips_data : list
        Sample NeurIPS data fixture

    Returns
    -------
    Mock
        Mock response object with JSON data

    Notes
    -----
    Simulates a successful HTTP response from the NeurIPS API.
    """
    mock = Mock()
    mock.status_code = 200
    mock.json.return_value = sample_neurips_data
    mock.raise_for_status = Mock()
    return mock

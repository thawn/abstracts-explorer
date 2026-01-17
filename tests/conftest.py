"""
Shared pytest fixtures for all test modules.

This module contains common fixtures used across multiple test files to reduce
code duplication and ensure consistency in test setup.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import Mock

from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.embeddings import EmbeddingsManager
from abstracts_explorer.plugin import LightweightPaper
from abstracts_explorer.config import load_env_file


@pytest.fixture(scope="session", autouse=True)
def test_config():
    """
    Ensure tests use predictable configuration from .env.example.
    
    This fixture sets environment variables from .env.example to provide
    consistent defaults for all tests, preventing tests from being affected
    by the user's custom .env file. However, it only sets variables that
    are not already in the environment, allowing individual tests to override.
    
    The fixture runs automatically for all tests (autouse=True) at session scope.
    """
    from abstracts_explorer.config import get_config
    
    # Find .env.example file
    repo_root = Path(__file__).parent.parent
    env_example = repo_root / ".env.example"
    
    if not env_example.exists():
        # If .env.example doesn't exist, skip this fixture
        yield
        return
    
    # Load values from .env.example
    example_vars = load_env_file(env_example)
    
    # Store which environment variables we added (not ones that were already there)
    added_vars = []
    
    # Only set environment variables that are NOT already set
    # This allows individual tests to override and prevents interference
    for key, value in example_vars.items():
        if key not in os.environ:
            os.environ[key] = value
            added_vars.append(key)
    
    # Force config reload to pick up test environment variables
    get_config(reload=True)
    
    yield
    
    # Remove only the environment variables we added
    for key in added_vars:
        if key in os.environ:
            del os.environ[key]
    
    # Reload config to restore user's configuration
    get_config(reload=True)


@pytest.fixture
def db_manager(tmp_path, monkeypatch):
    """
    Create a DatabaseManager instance with a temporary database.

    Parameters
    ----------
    tmp_path : Path
        Pytest's temporary path fixture
    monkeypatch : MonkeyPatch
        Pytest's monkeypatch fixture for setting environment variables

    Returns
    -------
    DatabaseManager
        Database manager instance with temporary database

    Notes
    -----
    Sets PAPER_DB environment variable to point to a temporary database
    and reloads config to pick up the change.
    """
    from abstracts_explorer.config import get_config
    
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("PAPER_DB", str(db_path))
    get_config(reload=True)  # Reload config to pick up the new PAPER_DB
    
    return DatabaseManager()


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
    This data uses the lightweight schema with semicolon-separated authors.
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
def test_database(tmp_path, monkeypatch):
    """
    Create a test database with sample papers for testing.

    Parameters
    ----------
    tmp_path : Path
        Pytest's temporary path fixture
    monkeypatch : MonkeyPatch
        Pytest's monkeypatch fixture for setting environment variables

    Returns
    -------
    Path
        Path to the test database file

    Notes
    -----
    Creates a database with 3 papers using the lightweight schema
    via DatabaseManager for testing search and retrieval functionality.
    Sets PAPER_DB environment variable to point to the test database.
    """
    from abstracts_explorer.database import DatabaseManager
    from abstracts_explorer.plugin import LightweightPaper
    from abstracts_explorer.config import get_config
    
    db_path = tmp_path / "test.db"
    
    # Set PAPER_DB to point to our test database
    monkeypatch.setenv("PAPER_DB", str(db_path))
    get_config(reload=True)  # Reload config to pick up the new PAPER_DB
    
    # Use DatabaseManager to create the database with proper schema
    with DatabaseManager() as db:
        db.create_tables()
        
        # Create sample papers using LightweightPaper model
        papers = [
            LightweightPaper(
                original_id=1,
                title="Deep Learning Paper",
                abstract="This paper presents a novel deep learning approach.",
                authors=["John Doe", "Jane Smith"],
                keywords=["deep learning", "neural networks"],
                session="ML Session 1",
                poster_position="A12",
                paper_pdf_url="https://papers.nips.cc/paper/1",
                year=2025,
                conference="NeurIPS",
            ),
            LightweightPaper(
                original_id=2,
                title="NLP Paper",
                abstract="We introduce a new natural language processing method.",
                authors=["Alice Johnson"],
                keywords=["NLP", "transformers"],
                session="NLP Session 2",
                poster_position="B05",
                paper_pdf_url="https://papers.nips.cc/paper/2",
                year=2025,
                conference="NeurIPS",
            ),
            LightweightPaper(
                original_id=3,
                title="Computer Vision Paper",
                abstract="",  # Empty abstract to test edge case
                authors=["Bob Wilson"],
                keywords=["vision", "CNN"],
                session="CV Session 3",
                poster_position="",
                paper_pdf_url="",
                year=2025,
                conference="NeurIPS",
            ),
        ]
        
        # Add papers to database
        for paper in papers:
            db.add_paper(paper)
    
    return db_path


@pytest.fixture
def mock_lm_studio():
    """
    Mock LM Studio API responses for testing embeddings.

    Yields
    ------
    Mock
        Mock OpenAI client with configured responses

    Notes
    -----
    Mocks both embedding and chat completion endpoints with typical responses.
    """
    from unittest.mock import patch, Mock

    with patch("abstracts_explorer.embeddings.OpenAI") as mock_openai_class:
        # Create mock OpenAI client instance
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock models.list() for connection test
        mock_models = Mock()
        mock_client.models.list.return_value = mock_models
        
        # Mock embeddings.create() for embedding generation
        mock_embedding_response = Mock()
        mock_embedding_data = Mock()
        mock_embedding_data.embedding = [0.1] * 4096
        mock_embedding_response.data = [mock_embedding_data]
        mock_client.embeddings.create.return_value = mock_embedding_response
        
        yield mock_client


@pytest.fixture
def mock_rag_openai():
    """
    Mock OpenAI client for RAG testing.

    Yields
    ------
    Mock
        Mock OpenAI client with configured chat completion responses

    Notes
    -----
    Mocks chat.completions.create() for RAG query testing.
    """
    from unittest.mock import patch, Mock

    with patch("abstracts_explorer.rag.OpenAI") as mock_openai_class:
        # Create mock OpenAI client instance
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock chat.completions.create() for chat generation
        mock_chat_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Based on Paper 1 and Paper 2, attention mechanisms allow models to focus on relevant parts of the input."
        mock_choice.message = mock_message
        mock_chat_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_chat_response
        
        yield mock_client


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
    Mocks the OpenAI client to avoid real API calls.
    """
    from unittest.mock import Mock
    
    chroma_path = tmp_path / "test_chroma"
    
    # Create the manager
    em = EmbeddingsManager(
        lm_studio_url="http://localhost:1234",
        chroma_path=chroma_path,
        collection_name="test_collection",
    )
    
    # Replace the OpenAI client with a mock
    mock_client = Mock()
    
    # Mock embeddings.create()
    mock_embedding_response = Mock()
    mock_embedding_data = Mock()
    mock_embedding_data.embedding = [0.1] * 4096
    mock_embedding_response.data = [mock_embedding_data]
    mock_client.embeddings.create.return_value = mock_embedding_response
    
    # Mock models.list()
    mock_client.models.list.return_value = Mock()
    
    # Set the private attribute directly to avoid triggering lazy loading
    em._openai_client = mock_client
    
    return em


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
    Uses string UIDs as required by the lightweight database schema.
    """
    mock_em = Mock(spec=EmbeddingsManager)

    # Mock successful search results with STRING UIDs (lightweight schema)
    mock_em.search_similar.return_value = {
        "ids": [["1", "2", "3"]],  # Use string UIDs (lightweight schema)
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


@pytest.fixture(scope="module")
def web_test_papers():
    """
    Create a list of test papers for web interface testing.

    Returns
    -------
    list of LightweightPaper
        Three test papers with diverse content for web testing

    Notes
    -----
    This fixture provides consistent test data for both test_web.py and
    test_web_integration.py, reducing code duplication.
    """
    return [
        LightweightPaper(
            title="Attention is All You Need",
            authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
            abstract="We propose the Transformer, a model architecture based solely on attention mechanisms.",
            session="Oral Session 1",
            poster_position="O1",
            year=2017,
            conference="NeurIPS",
            keywords=["attention", "transformer", "neural networks"],
        ),
        LightweightPaper(
            title="BERT: Pre-training of Deep Bidirectional Transformers",
            authors=["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee"],
            abstract="We introduce BERT, which stands for Bidirectional Encoder Representations from Transformers.",
            session="Poster Session 2",
            poster_position="P42",
            year=2019,
            conference="NeurIPS",
            keywords=["bert", "pretraining", "transformers"],
        ),
        LightweightPaper(
            title="Deep Residual Learning for Image Recognition",
            authors=["Kaiming He", "Xiangyu Zhang", "Shaoqing Ren"],
            abstract="We present a residual learning framework to ease the training of networks.",
            session="Oral Session 3",
            poster_position="O2",
            year=2016,
            conference="NeurIPS",
            keywords=["resnet", "computer vision", "deep learning"],
        ),
    ]

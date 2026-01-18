"""
Tests for ChromaDB metadata filtering functionality.

Tests verify that lightweight schema metadata fields (session, year, conference, etc.)
are correctly stored in ChromaDB and can be used for filtering search results.

Note: This file uses specialized fixtures (chroma_embeddings_manager, chroma_collection)
that are scoped specifically for ChromaDB metadata testing. The fixture names have been
chosen to avoid conflicts with the general embeddings fixtures in conftest.py.
"""

import pytest
from unittest.mock import Mock
from abstracts_explorer.plugin import validate_lightweight_paper, prepare_chroma_db_paper_data
from abstracts_explorer.embeddings import EmbeddingsManager


# Test embedding dimension for mock embeddings
MOCK_EMBEDDING_DIMENSION = 4096


@pytest.fixture(scope="module")
def test_chroma_collection(tmp_path_factory, monkeypatch_session):
    """
    Create a test ChromaDB collection with sample paper data.

    This fixture creates an isolated ChromaDB instance in a temporary directory
    and populates it with test paper data. All tests use this fixture instead of
    relying on real data files.

    Returns
    -------
    tuple
        Tuple of (collection, mock_client, embeddings_manager)
    """
    import uuid
    import time
    import chromadb.api.shared_system_client
    import os

    # Clear ChromaDB's global client registry to avoid conflicts
    chromadb.api.shared_system_client.SharedSystemClient._identifier_to_system.clear()

    # Create unique collection name to avoid conflicts
    unique_id = f"{int(time.time())}_{str(uuid.uuid4())[:8]}"
    tmp_dir = tmp_path_factory.mktemp("chroma_test")
    chroma_path = tmp_dir / f"chroma_{unique_id}"
    collection_name = f"test_papers_{unique_id}"

    # Set environment variable for EMBEDDING_DB
    os.environ["EMBEDDING_DB"] = str(chroma_path)
    
    # Force config reload to pick up the environment variable
    from abstracts_explorer.config import get_config
    config = get_config(reload=True)

    # Create mock OpenAI client
    mock_client = Mock()
    mock_models = Mock()
    mock_client.models.list.return_value = mock_models

    # Mock embeddings with consistent dimensions
    mock_embedding_response = Mock()
    mock_embedding_data = Mock()
    mock_embedding_data.embedding = [0.1] * MOCK_EMBEDDING_DIMENSION
    mock_embedding_response.data = [mock_embedding_data]
    mock_client.embeddings.create.return_value = mock_embedding_response

    # Initialize embeddings manager
    em = EmbeddingsManager(collection_name=collection_name)
    em._openai_client = mock_client  # Inject mock
    em.connect()
    em.create_collection(reset=True)

    # Add test papers with diverse metadata
    test_papers = [
        {
            "uid": "paper1",
            "title": "Deep Learning for Computer Vision",
            "authors": "Jane Smith; John Doe",
            "abstract": "We present a novel deep learning approach for computer vision tasks.",
            "session": "Oral Session 1",
            "poster_position": "A1",
            "keywords": "deep learning, computer vision",
            "year": 2025,
            "conference": "NeurIPS",
        },
        {
            "uid": "paper2",
            "title": "Reinforcement Learning in Robotics",
            "authors": "Alice Johnson; Bob Wilson",
            "abstract": "This paper explores reinforcement learning applications in robotics.",
            "session": "Poster Session A",
            "poster_position": "P1",
            "keywords": "reinforcement learning, robotics",
            "year": 2025,
            "conference": "NeurIPS",
        },
        {
            "uid": "paper3",
            "title": "Natural Language Processing Advances",
            "authors": "Carol White; David Brown",
            "abstract": "Recent advances in natural language processing are discussed.",
            "session": "Oral Session 1",
            "poster_position": "A2",
            "keywords": "nlp, transformers",
            "year": 2025,
            "conference": "NeurIPS",
        },
        {
            "uid": "paper4",
            "title": "Graph Neural Networks",
            "authors": "Eve Green; Frank Black",
            "abstract": "We introduce new architectures for graph neural networks.",
            "session": "Poster Session B",
            "poster_position": "P2",
            "keywords": "graph neural networks, gnn",
            "year": 2024,
            "conference": "NeurIPS",
        },
        {
            "uid": "paper5",
            "title": "Generative Models for Images",
            "authors": "Grace Lee; Henry Chen",
            "abstract": "Novel generative models for image synthesis are presented.",
            "session": "Spotlight Session",
            "poster_position": "S1",
            "keywords": "generative models, images",
            "year": 2024,
            "conference": "ICML",
        },
    ]

    # Add papers to collection
    for paper in test_papers:
        em.add_paper(paper)

    collection = em.collection

    yield (collection, mock_client, em)

    # Cleanup happens automatically


@pytest.fixture
def chroma_collection(test_chroma_collection):
    """
    Get the ChromaDB collection for testing.

    This fixture wraps the module-scoped test_chroma_collection fixture
    to provide just the collection object.

    Returns
    -------
    chromadb.Collection
        The ChromaDB collection containing paper embeddings.
    """
    collection, _, _ = test_chroma_collection
    return collection


@pytest.fixture
def chroma_embeddings_manager(test_chroma_collection):
    """
    Get the EmbeddingsManager for ChromaDB semantic search tests.

    This fixture provides an EmbeddingsManager instance with a mock OpenAI client
    that returns consistent embeddings. Renamed to avoid conflict with the general
    embeddings_manager fixture in conftest.py.

    Returns
    -------
    EmbeddingsManager
        Configured embeddings manager instance for ChromaDB tests.
    """
    _, mock_client, em = test_chroma_collection
    # Ensure mock client is still injected
    em._openai_client = mock_client
    return em


class TestChromaDBMetadata:
    """Test suite for ChromaDB metadata filtering."""

    def test_collection_has_documents(self, chroma_collection):
        """
        Test that the ChromaDB collection contains documents.

        Verifies basic collection functionality and that embeddings have been created.
        """
        count = chroma_collection.count()
        assert count > 0, "ChromaDB collection should contain documents"

    def test_metadata_has_required_fields(self, chroma_collection):
        """
        Test that documents have required metadata fields.

        Verifies that session, title, authors, and keywords fields exist in
        document metadata. Note that ChromaDB embeddings only store a subset
        of the lightweight schema fields for filtering purposes.
        """
        # Get a few documents with metadata
        results = chroma_collection.get(limit=5, include=["metadatas"])

        assert len(results["ids"]) > 0, "Should retrieve at least one document"

        # Check first document has validates successfully using plugin.validate_lightweight_paper()
        metadata = prepare_chroma_db_paper_data(results["metadatas"][0])
        assert validate_lightweight_paper(metadata), "Metadata validation failed"

    def test_filter_by_session(self, chroma_collection):
        """
        Test filtering documents by session.

        Verifies that the session metadata field can be used to filter search results.
        """
        # Get a session value to test with
        sample_results = chroma_collection.get(limit=1, include=["metadatas"])
        if not sample_results["metadatas"][0].get("session"):
            pytest.skip("No session metadata available")

        test_session = sample_results["metadatas"][0]["session"]

        # Filter by this session
        where_filter = {"session": {"$in": [test_session]}}
        results = chroma_collection.get(where=where_filter, limit=10, include=["metadatas"])

        assert len(results["ids"]) > 0, f"Should find documents with session '{test_session}'"

        # Verify all results have the correct session
        for metadata in results["metadatas"]:
            assert metadata.get("session") == test_session, "All results should have the filtered session"

    def test_filter_by_keywords(self, chroma_collection):
        """
        Test filtering documents by keywords.

        Verifies that the keywords metadata field can be used to filter search results.
        """
        # Get a document with keywords to test with
        sample_results = chroma_collection.get(limit=10, include=["metadatas"])
        keywords_found = [m.get("keywords") for m in sample_results["metadatas"] if m.get("keywords")]

        if not keywords_found:
            pytest.skip("No keywords metadata available")

        test_keywords = keywords_found[0]

        # Filter by these keywords
        where_filter = {"keywords": {"$in": [test_keywords]}}
        results = chroma_collection.get(where=where_filter, limit=10, include=["metadatas"])

        assert len(results["ids"]) > 0, f"Should find documents with keywords '{test_keywords}'"

        # Verify all results have the correct keywords
        for metadata in results["metadatas"]:
            assert metadata.get("keywords") == test_keywords, "All results should have the filtered keywords"

    def test_filter_by_title(self, chroma_collection):
        """
        Test filtering documents by title.

        Verifies that the title metadata field can be used to filter search results
        using exact matching.
        """
        # Get a title value to test with
        sample_results = chroma_collection.get(limit=10, include=["metadatas"])
        titles_found = [m.get("title") for m in sample_results["metadatas"] if m.get("title")]

        if not titles_found:
            pytest.skip("No title metadata available")

        test_title = titles_found[0]

        # Filter by this exact title
        where_filter = {"title": {"$in": [test_title]}}
        results = chroma_collection.get(where=where_filter, limit=10, include=["metadatas"])

        assert len(results["ids"]) > 0, f"Should find documents with title '{test_title}'"

        # Verify all results have the correct title
        for metadata in results["metadatas"]:
            assert metadata.get("title") == test_title, "All results should have the filtered title"

    def test_filter_with_or_operator(self, chroma_collection):
        """
        Test combining multiple filters with $or operator.

        Verifies that ChromaDB supports $or queries for combining multiple
        filter conditions.
        """
        # Get sample values for two different sessions
        sample_results = chroma_collection.get(limit=20, include=["metadatas"])
        sessions = [m.get("session") for m in sample_results["metadatas"] if m.get("session")]

        # Find unique sessions
        unique_sessions = list(set(sessions))

        if len(unique_sessions) < 2:
            pytest.skip("Need at least 2 different sessions for $or test")

        # Use first two unique sessions
        unique_sessions = unique_sessions[:2]

        # Create $or filter
        where_filter = {
            "$or": [
                {"session": {"$in": [unique_sessions[0]]}},
                {"session": {"$in": [unique_sessions[1]]}},
            ]
        }

        results = chroma_collection.get(where=where_filter, limit=10, include=["metadatas"])

        assert len(results["ids"]) > 0, "Should find documents matching $or filter"

        # Verify all results match one of the sessions
        for metadata in results["metadatas"]:
            session = metadata.get("session")
            assert session in unique_sessions, f"Result session '{session}' should match one of {unique_sessions}"

    def test_filter_with_and_operator(self, chroma_collection):
        """
        Test combining multiple filters with $and operator.

        Verifies that ChromaDB supports $and queries for combining multiple
        filter conditions.
        """
        # Get a sample document with both session and keywords
        sample_results = chroma_collection.get(limit=10, include=["metadatas"])

        # Find a document with both fields
        test_doc = None
        for metadata in sample_results["metadatas"]:
            if metadata.get("session") and metadata.get("keywords"):
                test_doc = metadata
                break

        if not test_doc:
            pytest.skip("Need document with both session and keywords")

        # Create $and filter
        where_filter = {
            "$and": [
                {"session": {"$in": [test_doc["session"]]}},
                {"keywords": {"$in": [test_doc["keywords"]]}},
            ]
        }

        results = chroma_collection.get(where=where_filter, limit=10, include=["metadatas"])

        assert len(results["ids"]) > 0, "Should find documents matching $and filter"

        # Verify all results match both conditions
        for metadata in results["metadatas"]:
            assert metadata.get("session") == test_doc["session"], "All results should have the filtered session"
            assert metadata.get("keywords") == test_doc["keywords"], "All results should have the filtered keywords"

    def test_no_invalid_session_values(self, chroma_collection):
        """
        Test that session field contains only valid values.

        Verifies that session names are properly formatted strings.
        """
        # Get all documents and check session values
        results = chroma_collection.get(limit=100, include=["metadatas"])

        for metadata in results["metadatas"]:
            session = metadata.get("session", "")
            if session:  # If session is present
                assert isinstance(session, str), "Session should be a string"
                assert len(session) > 0, "Session should not be empty string if present"

    def test_session_format(self, chroma_collection):
        """
        Test that session field has expected format.

        Verifies that session values are properly formatted strings.
        """
        results = chroma_collection.get(limit=10, include=["metadatas"])

        for metadata in results["metadatas"]:
            session = metadata.get("session", "")
            if session:  # If session is present
                assert isinstance(session, str), "Session should be a string"
                assert len(session) > 0, "Session should not be empty string if present"

    def test_all_documents_have_metadata(self, chroma_collection):
        """
        Test that all documents in the collection have metadata.

        This is a comprehensive test that checks all documents have the required
        metadata fields stored in ChromaDB. Marked as slow since it may
        process many documents.
        """
        # Get total count
        total_count = chroma_collection.count()

        # Sample a reasonable number of documents
        sample_size = min(100, total_count)
        results = chroma_collection.get(limit=sample_size, include=["metadatas"])

        missing_session = 0
        missing_title = 0
        missing_authors = 0

        for metadata in results["metadatas"]:
            if not metadata.get("session"):
                missing_session += 1
            if not metadata.get("title"):
                missing_title += 1
            if not metadata.get("authors"):
                missing_authors += 1

        # Allow some missing values but most should be present
        assert (
            missing_session < sample_size * 0.1
        ), f"Too many documents missing session metadata: {missing_session}/{sample_size}"
        assert (
            missing_title < sample_size * 0.1
        ), f"Too many documents missing title metadata: {missing_title}/{sample_size}"
        assert (
            missing_authors < sample_size * 0.5
        ), f"Too many documents missing authors metadata: {missing_authors}/{sample_size}"

    def test_semantic_search_with_filter(self, chroma_embeddings_manager, chroma_collection):
        """
        Test semantic search combined with metadata filtering.

        Verifies that ChromaDB can perform vector similarity search while
        simultaneously filtering by metadata fields.

        Note: Marked as slow because it generates embeddings for search queries.
        """
        # Get a session to filter by
        sample_results = chroma_collection.get(limit=5, include=["metadatas"])
        sessions_found = [m.get("session") for m in sample_results["metadatas"] if m.get("session")]

        if not sessions_found:
            pytest.skip("No session metadata available for semantic search test")

        test_session = sessions_found[0]
        where_filter = {"session": {"$in": [test_session]}}

        # Use EmbeddingsManager to perform semantic search with filter
        query = "neural networks machine learning"
        results = chroma_embeddings_manager.search_similar(query, n_results=5, where=where_filter)

        # Verify results structure
        assert "ids" in results, "Results should contain 'ids' key"
        assert "metadatas" in results, "Results should contain 'metadatas' key"
        assert "distances" in results, "Results should contain 'distances' key"

        # Verify we got results
        assert len(results["ids"]) > 0, "Should return result lists"
        assert len(results["ids"][0]) > 0, "Should have at least one result"

        # Verify all results match the filter
        for metadata in results["metadatas"][0]:
            assert (
                metadata.get("session") == test_session
            ), f"All results should have session '{test_session}', got '{metadata.get('session')}'"

        # Verify distances are present and reasonable
        assert len(results["distances"][0]) > 0, "Should return distance scores"
        for distance in results["distances"][0]:
            assert isinstance(distance, (int, float)), "Distance should be numeric"
            assert distance >= 0, "Distance should be non-negative"

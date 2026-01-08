"""
Tests for ChromaDB metadata filtering functionality.

Tests verify that lightweight schema metadata fields (session, year, conference, etc.)
are correctly stored in ChromaDB and can be used for filtering search results.
"""

import pytest
from pathlib import Path
import chromadb
from neurips_abstracts.plugin import validate_lightweight_paper, prepare_chroma_db_paper_data
from neurips_abstracts.config import Config
from neurips_abstracts.embeddings import EmbeddingsManager


@pytest.fixture
def chroma_collection():
    """
    Get the ChromaDB collection for testing.

    This fixture connects to the actual ChromaDB database used by the application.
    Tests are marked as integration tests since they depend on real data.

    Returns
    -------
    chromadb.Collection
        The ChromaDB collection containing paper embeddings.
    """
    config = Config()
    chroma_path = Path(config.embedding_db_path)

    if not chroma_path.exists():
        pytest.skip(f"ChromaDB not found at {chroma_path}")

    client = chromadb.PersistentClient(path=str(chroma_path))
    try:
        collection = client.get_collection(name="neurips_papers")
        return collection
    except Exception as e:
        pytest.skip(f"Could not load ChromaDB collection: {e}")


@pytest.fixture
def embeddings_manager(chroma_collection):
    """
    Get the EmbeddingsManager configured with OpenAI API.

    This fixture creates an EmbeddingsManager instance using the configuration
    from the environment (.env file). It reuses the ChromaDB client from the
    chroma_collection fixture to avoid conflicts. Tests using this fixture will
    be skipped if OpenAI API is not available.

    Parameters
    ----------
    chroma_collection : chromadb.Collection
        The ChromaDB collection (from fixture).

    Returns
    -------
    EmbeddingsManager
        Configured embeddings manager instance.
    """
    config = Config()

    try:
        # Create embeddings manager with config from .env
        # Note: We don't call connect() to avoid creating a second client
        em = EmbeddingsManager(
            lm_studio_url=config.llm_backend_url,
            model_name=config.embedding_model,
            chroma_path=Path(config.embedding_db_path),
            collection_name=config.collection_name,
        )

        # Reuse the existing collection from the fixture
        em.collection = chroma_collection
        em.client = chroma_collection._client

        # Test OpenAI API connection
        if not em.test_lm_studio_connection():
            pytest.skip("OpenAI API not available or model not loaded")

        return em
    except Exception as e:
        pytest.skip(f"Could not initialize EmbeddingsManager: {e}")


@pytest.mark.integration
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

    @pytest.mark.slow
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

    @pytest.mark.integration
    @pytest.mark.slow
    def test_semantic_search_with_filter(self, embeddings_manager, chroma_collection):
        """
        Test semantic search combined with metadata filtering.

        Verifies that ChromaDB can perform vector similarity search while
        simultaneously filtering by metadata fields. Uses the configured OpenAI API
        embedding model from .env file.

        This test requires:
        - OpenAI API running at configured URL
        - Correct embedding model loaded in OpenAI API
        - ChromaDB collection with embeddings created using that model

        Note: Marked as slow because it requires OpenAI API connection and may
        take time to generate embeddings.
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
        results = embeddings_manager.search_similar(query, n_results=5, where=where_filter)

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

"""
Tests for the RAG (Retrieval Augmented Generation) module.

This module tests the RAGChat functionality with both mocked and real LM Studio backends.
Tests that require a running LM Studio instance are skipped if it's not available.
"""

import json
import pytest
import requests
from unittest.mock import Mock, patch

from neurips_abstracts.rag import RAGChat, RAGError
from neurips_abstracts.embeddings import EmbeddingsManager
from neurips_abstracts.config import get_config
from tests.test_helpers import requires_lm_studio

# Fixtures imported from conftest.py:
# - mock_embeddings_manager: Mock embeddings manager with predefined search results
#
# Helper functions imported from test_helpers:
# - check_lm_studio_available(): Check if LM Studio is running
# - requires_lm_studio: Skip marker for tests requiring LM Studio


@pytest.fixture
def mock_database():
    """Create a mock database manager that returns papers for UIDs "1", "2", "3"."""
    mock_db = Mock()

    # Set up mock to return papers based on UID (string)
    def mock_query_side_effect(sql, params):
        paper_uid = params[0] if params else None
        papers_map = {
            "1": {
                "uid": "1",
                "title": "Attention Is All You Need",
                "abstract": "We propose the Transformer...",
                "authors": "Vaswani et al.",  # Stored as semicolon-separated string in DB
                "session": "Oral",
                "year": 2017,
                "conference": "NeurIPS",
            },
            "2": {
                "uid": "2",
                "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "abstract": "We introduce BERT...",
                "authors": "Devlin et al.",
                "session": "Poster",
                "year": 2019,
                "conference": "NeurIPS",
            },
            "3": {
                "uid": "3",
                "title": "GPT-3: Language Models are Few-Shot Learners",
                "abstract": "We train GPT-3...",
                "authors": "Brown et al.",
                "session": "Oral",
                "year": 2020,
                "conference": "NeurIPS",
            },
        }
        paper = papers_map.get(paper_uid)
        return [paper] if paper else []

    mock_db.query.side_effect = mock_query_side_effect

    return mock_db


@pytest.fixture
def mock_embeddings_manager_empty():
    """Create a mock embeddings manager that returns no results."""
    mock_em = Mock(spec=EmbeddingsManager)
    mock_em.search_similar.return_value = {
        "ids": [[]],
        "distances": [[]],
        "metadatas": [[]],
        "documents": [[]],
    }
    return mock_em


@pytest.fixture
def mock_lm_studio_response():
    """Mock LM Studio API response."""
    with patch("neurips_abstracts.rag.requests.post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Based on Paper 1 and Paper 2, attention mechanisms allow models to focus on relevant parts of the input."
                    }
                }
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        yield mock_post


class TestRAGChatInit:
    """Test RAGChat initialization."""

    def test_init_with_defaults(self, mock_embeddings_manager, mock_database):
        """Test initialization with default parameters."""
        chat = RAGChat(mock_embeddings_manager, mock_database)

        assert chat.embeddings_manager == mock_embeddings_manager
        assert chat.database == mock_database
        assert "localhost" in chat.lm_studio_url or "127.0.0.1" in chat.lm_studio_url
        assert chat.model is not None
        assert chat.max_context_papers > 0
        assert 0 <= chat.temperature <= 1
        assert isinstance(chat.conversation_history, list)
        assert len(chat.conversation_history) == 0

    def test_init_with_custom_params(self, mock_embeddings_manager, mock_database):
        """Test initialization with custom parameters."""
        chat = RAGChat(
            mock_embeddings_manager,
            mock_database,
            lm_studio_url="http://custom:8080",
            model="custom-model",
            max_context_papers=10,
            temperature=0.9,
        )

        assert chat.lm_studio_url == "http://custom:8080"
        assert chat.model == "custom-model"
        assert chat.max_context_papers == 10
        assert chat.temperature == 0.9

    def test_init_url_trailing_slash(self, mock_embeddings_manager, mock_database):
        """Test that trailing slash is removed from URL."""
        chat = RAGChat(
            mock_embeddings_manager,
            mock_database,
            lm_studio_url="http://localhost:1234/",
        )

        assert chat.lm_studio_url == "http://localhost:1234"

    def test_init_without_embeddings_manager(self, mock_database):
        """Test that missing embeddings manager raises error."""
        with pytest.raises(RAGError, match="embeddings_manager is required"):
            RAGChat(None, mock_database)

    def test_init_without_database(self, mock_embeddings_manager):
        """Test that missing database raises error."""
        with pytest.raises(RAGError, match="database is required"):
            RAGChat(mock_embeddings_manager, None)


class TestRAGChatQuery:
    """Test RAGChat query method."""

    def test_query_success(self, mock_embeddings_manager, mock_database, mock_lm_studio_response):
        """Test successful query with papers."""
        chat = RAGChat(mock_embeddings_manager, mock_database)

        result = chat.query("What is attention mechanism?")

        assert "response" in result
        assert "papers" in result
        assert "metadata" in result
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0
        assert len(result["papers"]) == 3
        assert result["metadata"]["n_papers"] == 3

        # Check that search was called
        mock_embeddings_manager.search_similar.assert_called_once()

        # Check conversation history
        assert len(chat.conversation_history) == 2
        assert chat.conversation_history[0]["role"] == "user"
        assert chat.conversation_history[1]["role"] == "assistant"

    def test_query_with_n_results(self, mock_embeddings_manager, mock_database, mock_lm_studio_response):
        """Test query with custom n_results."""
        chat = RAGChat(mock_embeddings_manager, mock_database)

        chat.query("What is deep learning?", n_results=2)

        # Check that n_results was passed
        call_args = mock_embeddings_manager.search_similar.call_args
        assert call_args[1]["n_results"] == 2

    def test_query_with_metadata_filter(self, mock_embeddings_manager, mock_database, mock_lm_studio_response):
        """Test query with metadata filter."""
        chat = RAGChat(mock_embeddings_manager, mock_database)

        metadata_filter = {"decision": "Accept (oral)"}
        chat.query("What are oral presentations?", metadata_filter=metadata_filter)

        # Check that filter was passed
        call_args = mock_embeddings_manager.search_similar.call_args
        assert call_args[1]["where"] == metadata_filter

    def test_query_with_system_prompt(self, mock_embeddings_manager, mock_database, mock_lm_studio_response):
        """Test query with custom system prompt."""
        chat = RAGChat(mock_embeddings_manager, mock_database)
        custom_prompt = "You are a helpful assistant specializing in machine learning."

        chat.query("Explain transformers", system_prompt=custom_prompt)

        # Check that the custom prompt was used in the API call
        call_args = mock_lm_studio_response.call_args
        messages = call_args[1]["json"]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == custom_prompt

    def test_query_no_results(self, mock_embeddings_manager_empty, mock_database, mock_lm_studio_response):
        """Test query when no papers are found."""
        chat = RAGChat(mock_embeddings_manager_empty, mock_database)

        result = chat.query("Unknown topic")

        assert "response" in result
        assert "couldn't find any relevant papers" in result["response"].lower()
        assert result["papers"] == []
        assert result["metadata"]["n_papers"] == 0

    def test_query_api_timeout(self, mock_embeddings_manager, mock_database):
        """Test query with API timeout."""
        with patch("neurips_abstracts.rag.requests.post") as mock_post:
            mock_post.side_effect = requests.exceptions.Timeout("Timeout")

            chat = RAGChat(mock_embeddings_manager, mock_database)

            with pytest.raises(RAGError, match="Request to LM Studio timed out"):
                chat.query("What is machine learning?")

    def test_query_api_http_error(self, mock_embeddings_manager, mock_database):
        """Test query with HTTP error."""
        with patch("neurips_abstracts.rag.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
            mock_post.return_value = mock_response

            chat = RAGChat(mock_embeddings_manager, mock_database)

            with pytest.raises(RAGError, match="LM Studio API error"):
                chat.query("What is deep learning?")

    def test_query_invalid_response(self, mock_embeddings_manager, mock_database):
        """Test query with invalid API response format."""
        with patch("neurips_abstracts.rag.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"invalid": "response"}
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            chat = RAGChat(mock_embeddings_manager, mock_database)

            with pytest.raises(RAGError, match="Invalid response from LM Studio API"):
                chat.query("What is AI?")

    def test_query_general_exception(self, mock_embeddings_manager, mock_database):
        """Test query with general exception."""
        mock_embeddings_manager.search_similar.side_effect = Exception("Unexpected error")

        chat = RAGChat(mock_embeddings_manager, mock_database)

        with pytest.raises(RAGError, match="Query failed"):
            chat.query("What is NLP?")


class TestRAGChatChat:
    """Test RAGChat chat method."""

    def test_chat_with_context(self, mock_embeddings_manager, mock_database, mock_lm_studio_response):
        """Test chat with context retrieval."""
        chat = RAGChat(mock_embeddings_manager, mock_database)

        result = chat.chat("Tell me about transformers")

        assert "response" in result
        assert len(result["papers"]) > 0
        mock_embeddings_manager.search_similar.assert_called_once()

    def test_chat_without_context(self, mock_embeddings_manager, mock_database, mock_lm_studio_response):
        """Test chat without context retrieval."""
        chat = RAGChat(mock_embeddings_manager, mock_database)

        result = chat.chat("Hello, how are you?", use_context=False)

        assert "response" in result
        assert result["papers"] == []
        assert result["metadata"]["n_papers"] == 0
        # Search should not be called when use_context=False
        mock_embeddings_manager.search_similar.assert_not_called()

    def test_chat_custom_n_results(self, mock_embeddings_manager, mock_database, mock_lm_studio_response):
        """Test chat with custom n_results."""
        chat = RAGChat(mock_embeddings_manager, mock_database)

        chat.chat("What is AI?", n_results=7)

        call_args = mock_embeddings_manager.search_similar.call_args
        assert call_args[1]["n_results"] == 7


class TestRAGChatConversation:
    """Test conversation history management."""

    def test_reset_conversation(self, mock_embeddings_manager, mock_database, mock_lm_studio_response):
        """Test resetting conversation history."""
        chat = RAGChat(mock_embeddings_manager, mock_database)

        # Have some conversation
        chat.query("First question")
        chat.query("Second question")

        assert len(chat.conversation_history) == 4  # 2 questions + 2 responses

        # Reset
        chat.reset_conversation()

        assert len(chat.conversation_history) == 0

    def test_conversation_history_accumulates(self, mock_embeddings_manager, mock_database, mock_lm_studio_response):
        """Test that conversation history accumulates."""
        chat = RAGChat(mock_embeddings_manager, mock_database)

        chat.query("First question")
        assert len(chat.conversation_history) == 2

        chat.query("Second question")
        assert len(chat.conversation_history) == 4

        chat.chat("Third message", use_context=False)
        assert len(chat.conversation_history) == 6

    def test_conversation_history_in_api_call(self, mock_embeddings_manager, mock_database, mock_lm_studio_response):
        """Test that conversation history is included in API calls."""
        chat = RAGChat(mock_embeddings_manager, mock_database)

        # First query
        chat.query("What is attention?")

        # Second query - should include history
        chat.query("Tell me more")

        # Check the API was called with history
        call_args = mock_lm_studio_response.call_args
        messages = call_args[1]["json"]["messages"]

        # Should have system prompt + history + current message
        assert len(messages) > 2
        assert messages[0]["role"] == "system"


# NOTE: Paper formatting tests have been moved to test_paper_utils.py
# The _format_papers and _build_context methods are now shared utilities
# in the paper_utils module and are tested there.


class TestRAGChatExport:
    """Test conversation export functionality."""

    def test_export_conversation(self, mock_embeddings_manager, mock_database, mock_lm_studio_response, tmp_path):
        """Test exporting conversation to JSON."""
        chat = RAGChat(mock_embeddings_manager, mock_database)

        # Have some conversation
        chat.query("First question")
        chat.query("Second question")

        # Export
        output_path = tmp_path / "conversation.json"
        chat.export_conversation(output_path)

        # Verify file exists and content
        assert output_path.exists()

        with open(output_path, "r") as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) == 4  # 2 questions + 2 responses
        assert data[0]["role"] == "user"
        assert data[1]["role"] == "assistant"

    def test_export_empty_conversation(self, mock_embeddings_manager, tmp_path):
        """Test exporting empty conversation."""
        chat = RAGChat(mock_embeddings_manager, mock_database)

        output_path = tmp_path / "empty_conversation.json"
        chat.export_conversation(output_path)

        assert output_path.exists()

        with open(output_path, "r") as f:
            data = json.load(f)

        assert data == []


# Integration tests that require actual LM Studio
class TestRAGChatIntegration:
    """Integration tests requiring a running LM Studio instance."""

    @requires_lm_studio
    def test_real_query(self, tmp_path):
        """Test real query with actual LM Studio backend using configured model."""
        # This test requires LM Studio to be running with the configured chat model
        # Create real embeddings manager
        from neurips_abstracts.embeddings import EmbeddingsManager

        config = get_config()

        chroma_path = tmp_path / "chroma_integration"
        em = EmbeddingsManager(chroma_path=chroma_path)
        em.connect()
        em.create_collection(reset=True)

        # Add a test paper (use string uid to match lightweight schema)
        em.add_paper(
            paper_id="1",
            abstract="This paper discusses attention mechanisms in neural networks.",
            metadata={
                "title": "Attention Mechanisms",
                "authors": "Test Author",
                "session": "Test Session",
                "keywords": "attention, neural networks",
            },
        )

        # Create database with test data
        from neurips_abstracts.database import DatabaseManager

        db_path = tmp_path / "test.db"
        db = DatabaseManager(str(db_path))
        db.connect()
        db.create_tables()

        # Insert paper into database (lightweight schema)
        cursor = db.connection.cursor()
        cursor.execute(
            """
            INSERT INTO papers (uid, title, abstract, authors, session, poster_position, keywords, year, conference)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "1",  # uid is TEXT in lightweight schema
                "Attention Mechanisms",
                "This paper discusses attention mechanisms in neural networks.",
                "Test Author",  # authors stored as comma-separated string
                "Test Session",  # session is required
                "",  # poster_position
                "attention, neural networks",
                2025,  # year is required
                "NeurIPS",  # conference is required
            ),
        )

        db.connection.commit()

        # Create RAG chat with configured settings
        chat = RAGChat(
            em,
            db,
            lm_studio_url=config.llm_backend_url,
            model=config.chat_model,
        )

        # Query
        result = chat.query("What is attention mechanism?")

        # Verify response
        assert "response" in result
        assert len(result["response"]) > 0
        assert len(result["papers"]) > 0
        assert "attention" in result["response"].lower() or "Attention" in result["response"]

        em.close()
        db.close()

    @requires_lm_studio
    def test_real_conversation(self, tmp_path):
        """Test real conversation with actual LM Studio backend using configured model."""
        from neurips_abstracts.embeddings import EmbeddingsManager

        config = get_config()

        chroma_path = tmp_path / "chroma_conversation"
        em = EmbeddingsManager(chroma_path=chroma_path)
        em.connect()
        em.create_collection(reset=True)

        # Add test papers (use string uid to match lightweight schema)
        em.add_paper(
            paper_id="1",
            abstract="Transformers are a deep learning architecture based on attention.",
            metadata={
                "title": "Transformers",
                "authors": "Vaswani et al.",
                "session": "Test Session",
                "keywords": "transformers, attention",
            },
        )

        # Create database with test data
        from neurips_abstracts.database import DatabaseManager

        db_path = tmp_path / "test.db"
        db = DatabaseManager(str(db_path))
        db.connect()
        db.create_tables()

        # Insert paper into database (lightweight schema)
        cursor = db.connection.cursor()
        cursor.execute(
            """
            INSERT INTO papers (uid, title, abstract, authors, session, poster_position, keywords, year, conference)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "1",  # uid is TEXT in lightweight schema
                "Transformers",
                "Transformers are a deep learning architecture based on attention.",
                "Vaswani et al.",  # authors stored as comma-separated string
                "Test Session",  # session is required
                "",  # poster_position
                "transformers, attention",
                2025,  # year is required
                "NeurIPS",  # conference is required
            ),
        )

        db.connection.commit()

        chat = RAGChat(
            em,
            db,
            lm_studio_url=config.llm_backend_url,
            model=config.chat_model,
        )

        # First query
        result1 = chat.query("What are transformers?")
        assert len(result1["response"]) > 0

        # Follow-up query
        result2 = chat.chat("Tell me more about their architecture")
        assert len(result2["response"]) > 0

        # Check history
        assert len(chat.conversation_history) == 4

        em.close()
        db.close()

    @requires_lm_studio
    def test_real_export(self, tmp_path):
        """Test exporting real conversation with configured model."""
        from neurips_abstracts.embeddings import EmbeddingsManager

        config = get_config()

        chroma_path = tmp_path / "chroma_export"
        em = EmbeddingsManager(chroma_path=chroma_path)
        em.connect()
        em.create_collection(reset=True)

        # Add test paper (use string uid to match lightweight schema)
        em.add_paper(
            paper_id="1",
            abstract="Test abstract about machine learning.",
            metadata={
                "title": "ML Paper",
                "authors": "Author",
                "session": "Test Session",
                "keywords": "machine learning",
            },
        )

        # Create database with test data
        from neurips_abstracts.database import DatabaseManager

        db_path = tmp_path / "test.db"
        db = DatabaseManager(str(db_path))
        db.connect()
        db.create_tables()

        # Insert paper into database (lightweight schema)
        cursor = db.connection.cursor()
        cursor.execute(
            """
            INSERT INTO papers (uid, title, abstract, authors, session, poster_position, keywords, year, conference)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "1",  # uid is TEXT in lightweight schema
                "ML Paper",
                "Test abstract about machine learning.",
                "Author",  # authors stored as comma-separated string
                "Test Session",  # session is required
                "",  # poster_position
                "machine learning",
                2025,  # year is required
                "NeurIPS",  # conference is required
            ),
        )

        db.connection.commit()

        chat = RAGChat(
            em,
            db,
            lm_studio_url=config.llm_backend_url,
            model=config.chat_model,
        )
        chat.query("What is machine learning?")

        # Export
        export_path = tmp_path / "real_conversation.json"
        chat.export_conversation(export_path)

        assert export_path.exists()

        with open(export_path, "r") as f:
            data = json.load(f)

        assert len(data) > 0
        assert all("role" in msg and "content" in msg for msg in data)

        em.close()
        db.close()


class TestRAGChatQueryRewriting:
    """Test RAGChat query rewriting functionality."""

    def test_rewrite_query_success(self, mock_embeddings_manager, mock_database):
        """Test successful query rewriting."""
        with patch("neurips_abstracts.rag.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "transformer attention mechanism neural networks"}}]
            }
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            chat = RAGChat(mock_embeddings_manager, mock_database)
            rewritten = chat._rewrite_query("What about transformers?")

            assert isinstance(rewritten, str)
            assert len(rewritten) > 0
            assert rewritten == "transformer attention mechanism neural networks"
            mock_post.assert_called_once()

    def test_rewrite_query_with_conversation_history(self, mock_embeddings_manager, mock_database):
        """Test query rewriting with conversation history."""
        with patch("neurips_abstracts.rag.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "transformer attention mechanism applications"}}]
            }
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            chat = RAGChat(mock_embeddings_manager, mock_database)

            # Add conversation history
            chat.conversation_history = [
                {"role": "user", "content": "Tell me about transformers"},
                {"role": "assistant", "content": "Transformers use attention..."},
            ]

            chat._rewrite_query("What are the applications?")

            # Check that conversation history was included
            call_args = mock_post.call_args
            messages = call_args[1]["json"]["messages"]
            assert len(messages) > 2  # System prompt + history + current query
            assert any(msg["content"] == "Tell me about transformers" for msg in messages)

    def test_rewrite_query_timeout_fallback(self, mock_embeddings_manager, mock_database):
        """Test query rewriting falls back to original on timeout."""
        with patch("neurips_abstracts.rag.requests.post") as mock_post:
            mock_post.side_effect = requests.exceptions.Timeout("Timeout")

            chat = RAGChat(mock_embeddings_manager, mock_database)
            original_query = "What is deep learning?"
            rewritten = chat._rewrite_query(original_query)

            # Should return original query on timeout
            assert rewritten == original_query

    def test_rewrite_query_http_error_fallback(self, mock_embeddings_manager, mock_database):
        """Test query rewriting falls back to original on HTTP error."""
        with patch("neurips_abstracts.rag.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
            mock_post.return_value = mock_response

            chat = RAGChat(mock_embeddings_manager, mock_database)
            original_query = "What is neural network?"
            rewritten = chat._rewrite_query(original_query)

            # Should return original query on error
            assert rewritten == original_query

    def test_rewrite_query_invalid_response_fallback(self, mock_embeddings_manager, mock_database):
        """Test query rewriting falls back to original on invalid response."""
        with patch("neurips_abstracts.rag.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"invalid": "response"}
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            chat = RAGChat(mock_embeddings_manager, mock_database)
            original_query = "What is AI?"
            rewritten = chat._rewrite_query(original_query)

            # Should return original query on invalid response
            assert rewritten == original_query

    def test_should_retrieve_papers_first_query(self, mock_embeddings_manager, mock_database):
        """Test that first query always retrieves papers."""
        chat = RAGChat(mock_embeddings_manager, mock_database)

        should_retrieve = chat._should_retrieve_papers("deep learning neural networks")

        assert should_retrieve is True

    def test_should_retrieve_papers_similar_queries(self, mock_embeddings_manager, mock_database):
        """Test that similar queries reuse cached papers."""
        chat = RAGChat(mock_embeddings_manager, mock_database)
        chat.last_search_query = "deep learning neural networks"

        # Very similar query (high Jaccard similarity)
        should_retrieve = chat._should_retrieve_papers("deep learning networks")

        # Should NOT retrieve (similarity is high)
        assert should_retrieve is False

    def test_should_retrieve_papers_different_queries(self, mock_embeddings_manager, mock_database):
        """Test that different queries retrieve new papers."""
        chat = RAGChat(mock_embeddings_manager, mock_database)
        chat.last_search_query = "deep learning neural networks"

        # Very different query
        should_retrieve = chat._should_retrieve_papers("natural language processing transformers")

        # Should retrieve (similarity is low)
        assert should_retrieve is True

    def test_query_with_rewriting_enabled(self, mock_embeddings_manager, mock_database):
        """Test query with query rewriting enabled."""
        with patch("neurips_abstracts.rag.requests.post") as mock_post:
            # Mock both query rewriting and response generation
            def mock_post_side_effect(*args, **kwargs):
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.raise_for_status = Mock()

                # Check if it's a rewriting request (shorter max_tokens)
                if kwargs.get("json", {}).get("max_tokens", 1000) == 100:
                    mock_response.json.return_value = {
                        "choices": [{"message": {"content": "attention mechanism transformers"}}]
                    }
                else:
                    mock_response.json.return_value = {
                        "choices": [{"message": {"content": "Response about attention"}}]
                    }
                return mock_response

            mock_post.side_effect = mock_post_side_effect

            chat = RAGChat(mock_embeddings_manager, mock_database)
            chat.enable_query_rewriting = True

            result = chat.query("What about attention?")

            assert "response" in result
            assert "metadata" in result
            assert "rewritten_query" in result["metadata"]
            assert result["metadata"]["rewritten_query"] == "attention mechanism transformers"

    def test_query_with_rewriting_disabled(self, mock_embeddings_manager, mock_database, mock_lm_studio_response):
        """Test query with query rewriting disabled."""
        chat = RAGChat(mock_embeddings_manager, mock_database)
        chat.enable_query_rewriting = False

        result = chat.query("What is attention mechanism?")

        assert "response" in result
        # Should use original query
        call_args = mock_embeddings_manager.search_similar.call_args
        assert call_args[0][0] == "What is attention mechanism?"

    def test_query_caching_similar_queries(self, mock_embeddings_manager, mock_database):
        """Test that similar follow-up queries reuse cached papers."""
        with patch("neurips_abstracts.rag.requests.post") as mock_post:

            def mock_post_side_effect(*args, **kwargs):
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.raise_for_status = Mock()

                # Check if it's a rewriting request
                if kwargs.get("json", {}).get("max_tokens", 1000) == 100:
                    mock_response.json.return_value = {
                        "choices": [{"message": {"content": "deep learning networks"}}]
                    }
                else:
                    mock_response.json.return_value = {"choices": [{"message": {"content": "Response"}}]}
                return mock_response

            mock_post.side_effect = mock_post_side_effect

            chat = RAGChat(mock_embeddings_manager, mock_database)
            chat.enable_query_rewriting = True

            # First query
            chat.query("Tell me about deep learning")
            search_call_count_1 = mock_embeddings_manager.search_similar.call_count

            # Similar follow-up query
            result2 = chat.query("What about deep learning?")
            search_call_count_2 = mock_embeddings_manager.search_similar.call_count

            # Should only have called search once (cached second time)
            assert search_call_count_1 == 1
            assert search_call_count_2 == 1  # No new search
            assert result2["metadata"]["retrieved_new_papers"] is False

    def test_reset_conversation_clears_cache(self, mock_embeddings_manager, mock_database):
        """Test that reset_conversation clears query cache."""
        chat = RAGChat(mock_embeddings_manager, mock_database)

        # Set up some state
        chat.last_search_query = "some query"
        chat._cached_papers = [{"id": 1}]
        chat._cached_context = "some context"
        chat.conversation_history = [{"role": "user", "content": "test"}]

        # Reset
        chat.reset_conversation()

        # Check everything is cleared
        assert chat.last_search_query is None
        assert chat._cached_papers == []
        assert chat._cached_context == ""
        assert chat.conversation_history == []

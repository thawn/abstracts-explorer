"""
Tests for the RAG (Retrieval Augmented Generation) module.

This module tests the RAGChat functionality with both mocked and real LM Studio backends.
Tests that require a running LM Studio instance are skipped if it's not available.
"""

import json
import pytest
import requests
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from neurips_abstracts.rag import RAGChat, RAGError
from neurips_abstracts.embeddings import EmbeddingsManager
from neurips_abstracts.config import get_config
from tests.test_helpers import check_lm_studio_available, requires_lm_studio

# Fixtures imported from conftest.py:
# - mock_embeddings_manager: Mock embeddings manager with predefined search results
#
# Helper functions imported from test_helpers:
# - check_lm_studio_available(): Check if LM Studio is running
# - requires_lm_studio: Skip marker for tests requiring LM Studio


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

    def test_init_with_defaults(self, mock_embeddings_manager):
        """Test initialization with default parameters."""
        chat = RAGChat(mock_embeddings_manager)

        assert chat.embeddings_manager == mock_embeddings_manager
        assert "localhost" in chat.lm_studio_url or "127.0.0.1" in chat.lm_studio_url
        assert chat.model is not None
        assert chat.max_context_papers > 0
        assert 0 <= chat.temperature <= 1
        assert isinstance(chat.conversation_history, list)
        assert len(chat.conversation_history) == 0

    def test_init_with_custom_params(self, mock_embeddings_manager):
        """Test initialization with custom parameters."""
        chat = RAGChat(
            mock_embeddings_manager,
            lm_studio_url="http://custom:8080",
            model="custom-model",
            max_context_papers=10,
            temperature=0.9,
        )

        assert chat.lm_studio_url == "http://custom:8080"
        assert chat.model == "custom-model"
        assert chat.max_context_papers == 10
        assert chat.temperature == 0.9

    def test_init_url_trailing_slash(self, mock_embeddings_manager):
        """Test that trailing slash is removed from URL."""
        chat = RAGChat(
            mock_embeddings_manager,
            lm_studio_url="http://localhost:1234/",
        )

        assert chat.lm_studio_url == "http://localhost:1234"


class TestRAGChatQuery:
    """Test RAGChat query method."""

    def test_query_success(self, mock_embeddings_manager, mock_lm_studio_response):
        """Test successful query with papers."""
        chat = RAGChat(mock_embeddings_manager)

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

    def test_query_with_n_results(self, mock_embeddings_manager, mock_lm_studio_response):
        """Test query with custom n_results."""
        chat = RAGChat(mock_embeddings_manager)

        result = chat.query("What is deep learning?", n_results=2)

        # Check that n_results was passed
        call_args = mock_embeddings_manager.search_similar.call_args
        assert call_args[1]["n_results"] == 2

    def test_query_with_metadata_filter(self, mock_embeddings_manager, mock_lm_studio_response):
        """Test query with metadata filter."""
        chat = RAGChat(mock_embeddings_manager)

        metadata_filter = {"decision": "Accept (oral)"}
        result = chat.query("What are oral presentations?", metadata_filter=metadata_filter)

        # Check that filter was passed
        call_args = mock_embeddings_manager.search_similar.call_args
        assert call_args[1]["where"] == metadata_filter

    def test_query_with_system_prompt(self, mock_embeddings_manager, mock_lm_studio_response):
        """Test query with custom system prompt."""
        chat = RAGChat(mock_embeddings_manager)
        custom_prompt = "You are a helpful assistant specializing in machine learning."

        result = chat.query("Explain transformers", system_prompt=custom_prompt)

        # Check that the custom prompt was used in the API call
        call_args = mock_lm_studio_response.call_args
        messages = call_args[1]["json"]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == custom_prompt

    def test_query_no_results(self, mock_embeddings_manager_empty, mock_lm_studio_response):
        """Test query when no papers are found."""
        chat = RAGChat(mock_embeddings_manager_empty)

        result = chat.query("Unknown topic")

        assert "response" in result
        assert "couldn't find any relevant papers" in result["response"].lower()
        assert result["papers"] == []
        assert result["metadata"]["n_papers"] == 0

    def test_query_api_timeout(self, mock_embeddings_manager):
        """Test query with API timeout."""
        with patch("neurips_abstracts.rag.requests.post") as mock_post:
            mock_post.side_effect = requests.exceptions.Timeout("Timeout")

            chat = RAGChat(mock_embeddings_manager)

            with pytest.raises(RAGError, match="Request to LM Studio timed out"):
                chat.query("What is machine learning?")

    def test_query_api_http_error(self, mock_embeddings_manager):
        """Test query with HTTP error."""
        with patch("neurips_abstracts.rag.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
            mock_post.return_value = mock_response

            chat = RAGChat(mock_embeddings_manager)

            with pytest.raises(RAGError, match="LM Studio API error"):
                chat.query("What is deep learning?")

    def test_query_invalid_response(self, mock_embeddings_manager):
        """Test query with invalid API response format."""
        with patch("neurips_abstracts.rag.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"invalid": "response"}
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            chat = RAGChat(mock_embeddings_manager)

            with pytest.raises(RAGError, match="Invalid response from LM Studio API"):
                chat.query("What is AI?")

    def test_query_general_exception(self, mock_embeddings_manager):
        """Test query with general exception."""
        mock_embeddings_manager.search_similar.side_effect = Exception("Unexpected error")

        chat = RAGChat(mock_embeddings_manager)

        with pytest.raises(RAGError, match="Query failed"):
            chat.query("What is NLP?")


class TestRAGChatChat:
    """Test RAGChat chat method."""

    def test_chat_with_context(self, mock_embeddings_manager, mock_lm_studio_response):
        """Test chat with context retrieval."""
        chat = RAGChat(mock_embeddings_manager)

        result = chat.chat("Tell me about transformers")

        assert "response" in result
        assert len(result["papers"]) > 0
        mock_embeddings_manager.search_similar.assert_called_once()

    def test_chat_without_context(self, mock_embeddings_manager, mock_lm_studio_response):
        """Test chat without context retrieval."""
        chat = RAGChat(mock_embeddings_manager)

        result = chat.chat("Hello, how are you?", use_context=False)

        assert "response" in result
        assert result["papers"] == []
        assert result["metadata"]["n_papers"] == 0
        # Search should not be called when use_context=False
        mock_embeddings_manager.search_similar.assert_not_called()

    def test_chat_custom_n_results(self, mock_embeddings_manager, mock_lm_studio_response):
        """Test chat with custom n_results."""
        chat = RAGChat(mock_embeddings_manager)

        result = chat.chat("What is AI?", n_results=7)

        call_args = mock_embeddings_manager.search_similar.call_args
        assert call_args[1]["n_results"] == 7


class TestRAGChatConversation:
    """Test conversation history management."""

    def test_reset_conversation(self, mock_embeddings_manager, mock_lm_studio_response):
        """Test resetting conversation history."""
        chat = RAGChat(mock_embeddings_manager)

        # Have some conversation
        chat.query("First question")
        chat.query("Second question")

        assert len(chat.conversation_history) == 4  # 2 questions + 2 responses

        # Reset
        chat.reset_conversation()

        assert len(chat.conversation_history) == 0

    def test_conversation_history_accumulates(self, mock_embeddings_manager, mock_lm_studio_response):
        """Test that conversation history accumulates."""
        chat = RAGChat(mock_embeddings_manager)

        chat.query("First question")
        assert len(chat.conversation_history) == 2

        chat.query("Second question")
        assert len(chat.conversation_history) == 4

        chat.chat("Third message", use_context=False)
        assert len(chat.conversation_history) == 6

    def test_conversation_history_in_api_call(self, mock_embeddings_manager, mock_lm_studio_response):
        """Test that conversation history is included in API calls."""
        chat = RAGChat(mock_embeddings_manager)

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


class TestRAGChatFormatting:
    """Test internal formatting methods."""

    def test_format_papers(self, mock_embeddings_manager):
        """Test paper formatting from search results."""
        chat = RAGChat(mock_embeddings_manager)

        search_results = {
            "ids": [["paper1", "paper2"]],
            "distances": [[0.1, 0.3]],
            "metadatas": [
                [
                    {
                        "title": "Test Paper 1",
                        "authors": "Author A",
                        "topic": "Topic A",
                        "decision": "Accept",
                    },
                    {
                        "title": "Test Paper 2",
                        "authors": "Author B",
                        "topic": "Topic B",
                        "decision": "Reject",
                    },
                ]
            ],
            "documents": [["Abstract 1", "Abstract 2"]],
        }

        papers = chat._format_papers(search_results)

        assert len(papers) == 2
        assert papers[0]["id"] == "paper1"
        assert papers[0]["title"] == "Test Paper 1"
        assert papers[0]["authors"] == "Author A"
        assert papers[0]["abstract"] == "Abstract 1"
        assert papers[0]["similarity"] == pytest.approx(0.9, abs=0.01)

        assert papers[1]["id"] == "paper2"
        assert papers[1]["similarity"] == pytest.approx(0.7, abs=0.01)

    def test_format_papers_missing_metadata(self, mock_embeddings_manager):
        """Test paper formatting with missing metadata fields."""
        chat = RAGChat(mock_embeddings_manager)

        search_results = {
            "ids": [["paper1"]],
            "distances": [[0.2]],
            "metadatas": [[{}]],  # Empty metadata
            "documents": [["Abstract text"]],
        }

        papers = chat._format_papers(search_results)

        assert len(papers) == 1
        assert papers[0]["title"] == "N/A"
        assert papers[0]["authors"] == "N/A"
        assert papers[0]["decision"] == "N/A"
        assert papers[0]["topic"] == "N/A"

    def test_build_context(self, mock_embeddings_manager):
        """Test context building from papers."""
        chat = RAGChat(mock_embeddings_manager)

        papers = [
            {
                "title": "Test Paper",
                "authors": "Author A, Author B",
                "topic": "Machine Learning",
                "decision": "Accept",
                "abstract": "This is a test abstract.",
            }
        ]

        context = chat._build_context(papers)

        assert "Paper 1:" in context
        assert "Test Paper" in context
        assert "Author A, Author B" in context
        assert "Machine Learning" in context
        assert "Accept" in context
        assert "This is a test abstract." in context

    def test_build_context_multiple_papers(self, mock_embeddings_manager):
        """Test context building with multiple papers."""
        chat = RAGChat(mock_embeddings_manager)

        papers = [
            {
                "title": "Paper 1",
                "authors": "Author 1",
                "topic": "Topic 1",
                "decision": "Accept",
                "abstract": "Abstract 1",
            },
            {
                "title": "Paper 2",
                "authors": "Author 2",
                "topic": "Topic 2",
                "decision": "Reject",
                "abstract": "Abstract 2",
            },
        ]

        context = chat._build_context(papers)

        assert "Paper 1:" in context
        assert "Paper 2:" in context
        assert "Paper 1" in context
        assert "Paper 2" in context


class TestRAGChatExport:
    """Test conversation export functionality."""

    def test_export_conversation(self, mock_embeddings_manager, mock_lm_studio_response, tmp_path):
        """Test exporting conversation to JSON."""
        chat = RAGChat(mock_embeddings_manager)

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
        chat = RAGChat(mock_embeddings_manager)

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

        # Add a test paper
        em.add_paper(
            paper_id=1,
            abstract="This paper discusses attention mechanisms in neural networks.",
            metadata={
                "title": "Attention Mechanisms",
                "authors": "Test Author",
                "topic": "Deep Learning",
                "decision": "Accept",
                "keywords": "attention, neural networks",
            },
        )

        # Create RAG chat with configured settings
        chat = RAGChat(
            em,
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

    @requires_lm_studio
    def test_real_conversation(self, tmp_path):
        """Test real conversation with actual LM Studio backend using configured model."""
        from neurips_abstracts.embeddings import EmbeddingsManager

        config = get_config()

        chroma_path = tmp_path / "chroma_conversation"
        em = EmbeddingsManager(chroma_path=chroma_path)
        em.connect()
        em.create_collection(reset=True)

        # Add test papers
        em.add_paper(
            paper_id=1,
            abstract="Transformers are a deep learning architecture based on attention.",
            metadata={
                "title": "Transformers",
                "authors": "Vaswani et al.",
                "topic": "Deep Learning",
                "decision": "Accept (oral)",
                "keywords": "transformers, attention",
            },
        )

        chat = RAGChat(
            em,
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

    @requires_lm_studio
    def test_real_export(self, tmp_path):
        """Test exporting real conversation with configured model."""
        from neurips_abstracts.embeddings import EmbeddingsManager

        config = get_config()

        chroma_path = tmp_path / "chroma_export"
        em = EmbeddingsManager(chroma_path=chroma_path)
        em.connect()
        em.create_collection(reset=True)

        em.add_paper(
            paper_id=1,
            abstract="Test abstract about machine learning.",
            metadata={
                "title": "ML Paper",
                "authors": "Author",
                "topic": "ML",
                "decision": "Accept",
                "keywords": "machine learning",
            },
        )

        chat = RAGChat(
            em,
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

"""
Tests for the RAG (Retrieval Augmented Generation) module.

This module tests the RAGChat functionality with both mocked and real LM Studio backends.
Tests that require a running LM Studio instance are skipped if it's not available.
"""

import json
import pytest
from unittest.mock import Mock, patch

from abstracts_explorer.rag import RAGChat, RAGError
from abstracts_explorer.embeddings import EmbeddingsManager
from abstracts_explorer.config import get_config
from tests.conftest import set_test_embedding_db
from tests.helpers import requires_lm_studio, create_test_db_with_paper

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
    with patch("abstracts_explorer.rag.OpenAI") as mock_openai_class:
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock chat.completions.create()
        mock_chat_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Based on Paper 1 and Paper 2, attention mechanisms allow models to focus on relevant parts of the input."
        mock_message.tool_calls = None  # No tool calls for standard RAG
        mock_choice.message = mock_message
        mock_chat_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_chat_response
        
        yield mock_openai_class


@pytest.fixture
def mock_mcp_tool_execution():
    """Mock MCP tool execution to return papers."""
    with patch("abstracts_explorer.rag.execute_mcp_tool") as mock_execute:
        # Return papers in the format that get_recent_developments would return
        mock_execute.return_value = json.dumps({
            "topic": "test query",
            "papers_found": 3,
            "papers": [
                {
                    "id": "1",
                    "title": "Attention Is All You Need",
                    "abstract": "The dominant sequence transduction models...",
                    "year": 2017,
                    "conference": "NeurIPS",
                    "session": "Oral",
                    "relevance_score": 0.95,
                },
                {
                    "id": "2",
                    "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                    "abstract": "We introduce BERT...",
                    "year": 2019,
                    "conference": "NeurIPS",
                    "session": "Poster",
                    "relevance_score": 0.92,
                },
                {
                    "id": "3",
                    "title": "GPT-3: Language Models are Few-Shot Learners",
                    "abstract": "We train GPT-3...",
                    "year": 2020,
                    "conference": "NeurIPS",
                    "session": "Oral",
                    "relevance_score": 0.88,
                },
            ],
        }, indent=2)
        yield mock_execute


class TestRAGChatInit:
    """Test RAGChat initialization."""

    def test_init_with_defaults(self, mock_embeddings_manager, mock_database):
        """Test initialization with default parameters."""
        chat = RAGChat(mock_embeddings_manager, mock_database)

        assert chat.embeddings_manager == mock_embeddings_manager
        assert chat.database == mock_database
        assert chat.lm_studio_url == "https://api.helmholtz-blablador.fz-juelich.de"
        assert chat.model == "alias-fast"
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

    def test_query_success(self, mock_embeddings_manager, mock_database, mock_lm_studio_response, mock_mcp_tool_execution):
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

        # Check that MCP tool was called
        mock_mcp_tool_execution.assert_called_once()

        # Check conversation history
        assert len(chat.conversation_history) == 2
        assert chat.conversation_history[0]["role"] == "user"
        assert chat.conversation_history[1]["role"] == "assistant"

    def test_query_with_n_results(self, mock_embeddings_manager, mock_database, mock_lm_studio_response, mock_mcp_tool_execution):
        """Test query with custom n_results."""
        chat = RAGChat(mock_embeddings_manager, mock_database)

        chat.query("What is deep learning?", n_results=2)

        # Check that n_results was passed to the MCP tool
        call_args = mock_mcp_tool_execution.call_args
        assert call_args is not None
        # Check the arguments dict passed to the tool
        tool_name, tool_args = call_args[0]
        assert tool_args.get("n_results") == 2

    def test_query_with_metadata_filter(self, mock_embeddings_manager, mock_database, mock_lm_studio_response, mock_mcp_tool_execution):
        """Test query with metadata filter."""
        chat = RAGChat(mock_embeddings_manager, mock_database)

        metadata_filter = {"decision": "Accept (oral)"}
        chat.query("What are oral presentations?", metadata_filter=metadata_filter)

        # Check that MCP tool was called (metadata filters are passed to tools internally)
        mock_mcp_tool_execution.assert_called_once()

    def test_query_with_system_prompt(self, mock_embeddings_manager, mock_database, mock_mcp_tool_execution):
        """Test query with custom system prompt."""
        with patch("abstracts_explorer.rag.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            
            # Mock routing response
            mock_route_response = Mock()
            mock_route_choice = Mock()
            mock_route_message = Mock()
            mock_route_message.content = '{"name": "get_recent_developments", "arguments": {"topic_keywords": "transformers", "n_results": 5}}'
            mock_route_choice.message = mock_route_message
            mock_route_response.choices = [mock_route_choice]
            
            # Mock final response
            mock_final_response = Mock()
            mock_final_choice = Mock()
            mock_final_message = Mock()
            mock_final_message.content = "Based on the papers, transformers are..."
            mock_final_choice.message = mock_final_message
            mock_final_response.choices = [mock_final_choice]
            
            mock_client.chat.completions.create.side_effect = [
                mock_route_response,
                mock_final_response
            ]
            
            chat = RAGChat(mock_embeddings_manager, mock_database)
            custom_prompt = "You are a helpful assistant specializing in machine learning."

            chat.query("Explain transformers", system_prompt=custom_prompt)

            # Check that the custom prompt was used in the final API call
            # The routing call uses a different prompt
            assert mock_client.chat.completions.create.call_count == 2
            final_call_args = mock_client.chat.completions.create.call_args_list[1]
            messages = final_call_args.kwargs["messages"]
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == custom_prompt

    def test_query_no_results(self, mock_embeddings_manager_empty, mock_database):
        """Test query when no papers are found."""
        with patch("abstracts_explorer.rag.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            
            # Mock MCP tool execution to return empty results
            with patch("abstracts_explorer.rag.execute_mcp_tool") as mock_execute:
                mock_execute.return_value = json.dumps({
                    "topic": "Unknown topic",
                    "papers_found": 0,
                    "papers": [],
                }, indent=2)
                
                # Mock responses that check context for empty results
                def mock_create(**kwargs):
                    messages = kwargs.get("messages", [])
                    # Check if context mentions "No relevant papers found" or empty papers list
                    for msg in messages:
                        content = msg.get("content", "")
                        if "No relevant papers found" in content or '"papers": []' in content:
                            mock_message = Mock()
                            mock_message.content = "I couldn't find any relevant papers for this topic."
                            mock_message.tool_calls = None
                            mock_choice = Mock()
                            mock_choice.message = mock_message
                            mock_response = Mock()
                            mock_response.choices = [mock_choice]
                            return mock_response
                    
                    # Default response
                    mock_message = Mock()
                    mock_message.content = "Based on the available information."
                    mock_message.tool_calls = None
                    mock_choice = Mock()
                    mock_choice.message = mock_message
                    mock_response = Mock()
                    mock_response.choices = [mock_choice]
                    return mock_response
                
                mock_client.chat.completions.create.side_effect = mock_create
                
                chat = RAGChat(mock_embeddings_manager_empty, mock_database)
                result = chat.query("Unknown topic")

                assert "response" in result
                assert "couldn't find any relevant papers" in result["response"].lower()
                assert result["papers"] == []
                assert result["metadata"]["n_papers"] == 0

    def test_query_api_timeout(self, mock_embeddings_manager, mock_database):
        """Test query with API timeout."""
        with patch("abstracts_explorer.rag.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.side_effect = Exception("Timeout")

            chat = RAGChat(mock_embeddings_manager, mock_database)

            with pytest.raises(RAGError, match="Failed to generate response"):
                chat.query("What is machine learning?")

    def test_query_api_http_error(self, mock_embeddings_manager, mock_database):
        """Test query with HTTP error."""
        with patch("abstracts_explorer.rag.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.side_effect = Exception("API error")

            chat = RAGChat(mock_embeddings_manager, mock_database)

            with pytest.raises(RAGError, match="Failed to generate response"):
                chat.query("What is deep learning?")

    def test_query_invalid_response(self, mock_embeddings_manager, mock_database):
        """Test query with invalid API response format."""
        with patch("abstracts_explorer.rag.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            # Mock an invalid response structure
            mock_response = Mock()
            mock_response.choices = []
            mock_client.chat.completions.create.return_value = mock_response

            chat = RAGChat(mock_embeddings_manager, mock_database)

            with pytest.raises(RAGError, match="Failed to generate response"):
                chat.query("What is AI?")

    def test_query_general_exception(self, mock_embeddings_manager, mock_database):
        """Test query with general exception."""
        # Mock MCP tool execution to raise exception
        with patch("abstracts_explorer.rag.execute_mcp_tool") as mock_execute:
            mock_execute.side_effect = Exception("Unexpected error")

            chat = RAGChat(mock_embeddings_manager, mock_database)

            with pytest.raises(RAGError, match="Query failed"):
                chat.query("What is NLP?")


class TestRAGChatChat:
    """Test RAGChat chat method."""

    def test_chat_with_context(self, mock_embeddings_manager, mock_database, mock_lm_studio_response, mock_mcp_tool_execution):
        """Test chat with context retrieval."""
        chat = RAGChat(mock_embeddings_manager, mock_database)

        result = chat.chat("Tell me about transformers")

        assert "response" in result
        assert len(result["papers"]) > 0
        # With MCP tools, execute_mcp_tool is called instead of search_similar
        mock_mcp_tool_execution.assert_called_once()

    def test_chat_without_context(self, mock_embeddings_manager, mock_database, mock_lm_studio_response):
        """Test chat without context retrieval."""
        chat = RAGChat(mock_embeddings_manager, mock_database)

        result = chat.chat("Hello, how are you?", use_context=False)

        assert "response" in result
        assert result["papers"] == []
        assert result["metadata"]["n_papers"] == 0
        # Search should not be called when use_context=False
        mock_embeddings_manager.search_similar.assert_not_called()

    def test_chat_custom_n_results(self, mock_embeddings_manager, mock_database, mock_lm_studio_response, mock_mcp_tool_execution):
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

    def test_conversation_history_in_api_call(self, mock_embeddings_manager, mock_database, mock_lm_studio_response, mock_mcp_tool_execution):
        """Test that conversation history is included in API calls."""
        chat = RAGChat(mock_embeddings_manager, mock_database)

        # First query
        chat.query("What is attention?")

        # Second query - should include history
        chat.query("Tell me more")

        # Check the API was called with history
        call_args = mock_lm_studio_response.chat.completions.create.call_args
        assert call_args is not None
        messages = call_args.kwargs["messages"]

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
    """
    Integration tests requiring a running LM Studio instance.
    
    These tests verify end-to-end functionality with real LM Studio API.
    Mocked versions of these tests exist in other test classes:
    - test_real_query: See TestRAGChatQuery.test_query_success
    - test_real_conversation: See TestRAGChatChat.test_chat_with_context and 
      TestRAGChatConversation.test_conversation_history_accumulates
    - test_real_export: See TestRAGChatExport.test_export_conversation
    """

    @requires_lm_studio
    def test_real_query(self, tmp_path):
        """
        Test real query with actual LM Studio backend using configured model.
        
        This integration test verifies the complete RAG query workflow with real API.
        For unit testing without LM Studio, see TestRAGChatQuery.test_query_success.
        """
        # This test requires LM Studio to be running with the configured chat model
        # Create real embeddings manager
        from abstracts_explorer.embeddings import EmbeddingsManager

        config = get_config()

        # Create database with test data FIRST to get the generated UID
        db_path = tmp_path / "test.db"
        db = create_test_db_with_paper(
            db_path,
            {
                "uid": "1",
                "title": "Attention Mechanisms",
                "abstract": "This paper discusses attention mechanisms in neural networks.",
                "authors": "Test Author",
                "session": "Test Session",
                "keywords": "attention, neural networks",
                "year": 2025,
                "conference": "NeurIPS",
            }
        )
        
        # Get the generated UID from the database
        papers = db.query("SELECT uid FROM papers LIMIT 1")
        generated_uid = papers[0]["uid"]

        chroma_path = tmp_path / "chroma_integration"
        set_test_embedding_db(chroma_path)  # Ensure chroma path is clean before test
        em = EmbeddingsManager()
        em.connect()
        em.create_collection(reset=True)

        # Add test paper to ChromaDB using the generated UID from database
        em.add_paper(
            {
                "uid": generated_uid,
                "abstract": "This paper discusses attention mechanisms in neural networks.",
                "title": "Attention Mechanisms",
                "authors": "Test Author",
                "session": "Test Session",
                "keywords": "attention, neural networks",
            }
        )

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
        """
        Test real conversation with actual LM Studio backend using configured model.
        
        This integration test verifies multi-turn conversation with real API.
        For unit testing without LM Studio, see TestRAGChatChat.test_chat_with_context
        and TestRAGChatConversation.test_conversation_history_accumulates.
        """
        from abstracts_explorer.embeddings import EmbeddingsManager

        config = get_config()

        # Create database with test data FIRST to get the generated UID
        db_path = tmp_path / "test.db"
        db = create_test_db_with_paper(
            db_path,
            {
                "uid": "1",
                "title": "Transformers",
                "abstract": "Transformers are a deep learning architecture based on attention.",
                "authors": "Vaswani et al.",
                "session": "Test Session",
                "keywords": "transformers, attention",
                "year": 2025,
                "conference": "NeurIPS",
            }
        )
        
        # Get the generated UID from the database
        papers = db.query("SELECT uid FROM papers LIMIT 1")
        generated_uid = papers[0]["uid"]

        chroma_path = tmp_path / "chroma_conversation"
        set_test_embedding_db(chroma_path)  # Ensure chroma path is clean before test
        em = EmbeddingsManager()
        em.connect()
        em.create_collection(reset=True)

        # Add test paper to ChromaDB using the generated UID from database
        em.add_paper(
            {
                "uid": generated_uid,
                "abstract": "Transformers are a deep learning architecture based on attention.",
                "title": "Transformers",
                "authors": "Vaswani et al.",
                "session": "Test Session",
                "keywords": "transformers, attention",
            }
        )

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
        """
        Test exporting real conversation with configured model.
        
        This integration test verifies conversation export with real API.
        For unit testing without LM Studio, see TestRAGChatExport.test_export_conversation.
        """
        from abstracts_explorer.embeddings import EmbeddingsManager

        config = get_config()

        # Create database with test data FIRST to get the generated UID
        db_path = tmp_path / "test.db"
        db = create_test_db_with_paper(
            db_path,
            {
                "uid": "1",
                "title": "ML Paper",
                "abstract": "Test abstract about machine learning.",
                "authors": "Author",
                "session": "Test Session",
                "keywords": "machine learning",
                "year": 2025,
                "conference": "NeurIPS",
            }
        )
        
        # Get the generated UID from the database
        papers = db.query("SELECT uid FROM papers LIMIT 1")
        generated_uid = papers[0]["uid"]

        chroma_path = tmp_path / "chroma_export"
        set_test_embedding_db(chroma_path)  # Ensure chroma path is clean before test
        em = EmbeddingsManager()
        em.connect()
        em.create_collection(reset=True)

        # Add test paper to ChromaDB using the generated UID from database
        em.add_paper(
            {
                "uid": generated_uid,
                "abstract": "Test abstract about machine learning.",
                "title": "ML Paper",
                "authors": "Author",
                "session": "Test Session",
                "keywords": "machine learning",
            }
        )

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
        with patch("abstracts_explorer.rag.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            
            # Mock routing response with JSON tool call
            mock_route_response = Mock()
            mock_route_choice = Mock()
            mock_route_message = Mock()
            mock_route_message.content = '{"name": "get_recent_developments", "arguments": {"topic_keywords": "attention mechanism transformers", "n_results": 5}}'
            mock_route_choice.message = mock_route_message
            mock_route_response.choices = [mock_route_choice]
            
            # Mock final response
            mock_final_response = Mock()
            mock_final_choice = Mock()
            mock_final_message = Mock()
            mock_final_message.content = "Response about attention"
            mock_final_choice.message = mock_final_message
            mock_final_response.choices = [mock_final_choice]
            
            mock_client.chat.completions.create.side_effect = [
                mock_route_response,
                mock_final_response
            ]

            chat = RAGChat(mock_embeddings_manager, mock_database)
            chat.enable_query_rewriting = True

            result = chat.query("What about attention?")

            assert "response" in result
            assert "metadata" in result
            # In the new architecture, query rewriting happens through tools
            assert result["metadata"]["used_tools"] is True

    def test_query_with_rewriting_disabled(self, mock_embeddings_manager, mock_database, mock_lm_studio_response, mock_mcp_tool_execution):
        """Test query with query rewriting disabled."""
        chat = RAGChat(mock_embeddings_manager, mock_database)
        chat.enable_query_rewriting = False

        result = chat.query("What is attention mechanism?")

        assert "response" in result
        # With MCP tools, the query goes through get_recent_developments
        mock_mcp_tool_execution.assert_called_once()

    def test_query_caching_similar_queries(self, mock_embeddings_manager, mock_database):
        """Test that similar follow-up queries reuse cached papers."""
        with patch("abstracts_explorer.rag.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            # Mock tool execution to return papers
            with patch("abstracts_explorer.rag.execute_mcp_tool") as mock_execute:
                mock_execute.return_value = json.dumps({
                    "topic": "deep learning",
                    "papers_found": 3,
                    "papers": [
                        {
                            "id": "1",
                            "title": "Deep Learning Paper",
                            "abstract": "About deep learning...",
                            "year": 2023,
                            "conference": "NeurIPS",
                            "session": "Oral",
                            "relevance_score": 0.95,
                        },
                    ],
                }, indent=2)

                # Mock routing responses
                def mock_create_side_effect(*args, **kwargs):
                    mock_response = Mock()
                    mock_choice = Mock()
                    mock_message = Mock()

                    # Check if it's a routing request (shorter timeout)
                    if kwargs.get("timeout", 180) == 30:
                        mock_message.content = '{"name": "get_recent_developments", "arguments": {"topic_keywords": "deep learning networks", "n_results": 5}}'
                    else:
                        mock_message.content = "Response"
                    
                    mock_message.tool_calls = None  # No tool calls
                    mock_choice.message = mock_message
                    mock_response.choices = [mock_choice]
                    return mock_response

                mock_client.chat.completions.create.side_effect = mock_create_side_effect

                chat = RAGChat(mock_embeddings_manager, mock_database)
                chat.enable_query_rewriting = True

                # First query
                chat.query("Tell me about deep learning")
                tool_call_count_1 = mock_execute.call_count

                # Similar follow-up query
                result2 = chat.query("What about deep learning?")
                tool_call_count_2 = mock_execute.call_count

                # Should only have called tool once (cached second time)
                assert tool_call_count_1 == 1
                assert tool_call_count_2 == 1  # No new tool execution
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


class TestRAGChatMCPTools:
    """Test RAGChat integration with MCP clustering tools."""

    def test_init_with_mcp_tools_enabled(self, mock_embeddings_manager, mock_database):
        """Test that MCP tools can be enabled during initialization."""
        chat = RAGChat(mock_embeddings_manager, mock_database, enable_mcp_tools=True)
        
        assert chat.enable_mcp_tools is True

    def test_init_with_mcp_tools_disabled(self, mock_embeddings_manager, mock_database):
        """Test that MCP tools can be disabled during initialization."""
        chat = RAGChat(mock_embeddings_manager, mock_database, enable_mcp_tools=False)
        
        assert chat.enable_mcp_tools is False

    def test_query_routing_for_tool_call(self, mock_embeddings_manager, mock_database):
        """Test that queries requiring tools are routed to tool execution."""
        with patch("abstracts_explorer.rag.OpenAI") as mock_openai_class:
            with patch("abstracts_explorer.rag.execute_mcp_tool") as mock_execute:
                with patch("abstracts_explorer.rag.format_tool_result_for_llm") as mock_format:
                    mock_client = Mock()
                    mock_openai_class.return_value = mock_client
                    
                    # Mock routing response - returns JSON tool call
                    mock_route_response = Mock()
                    mock_route_choice = Mock()
                    mock_route_message = Mock()
                    mock_route_message.content = '{"name": "get_cluster_topics", "arguments": {"n_clusters": 8}}'
                    mock_route_choice.message = mock_route_message
                    mock_route_response.choices = [mock_route_choice]
                    
                    # Mock final response generation
                    mock_final_response = Mock()
                    mock_final_choice = Mock()
                    mock_final_message = Mock()
                    mock_final_message.content = "The main topics are transformers, diffusion models, and reinforcement learning."
                    mock_final_choice.message = mock_final_message
                    mock_final_response.choices = [mock_final_choice]
                    
                    mock_client.chat.completions.create.side_effect = [
                        mock_route_response,  # Routing/analysis call
                        mock_final_response   # Response generation call
                    ]
                    
                    # Mock tool execution
                    mock_execute.return_value = '{"topics": ["transformers", "diffusion", "RL"]}'
                    mock_format.return_value = "Topics: transformers, diffusion, RL"
                    
                    chat = RAGChat(mock_embeddings_manager, mock_database, enable_mcp_tools=True)
                    result = chat.query("What are the main topics?")
                    
                    # Verify tool was executed
                    mock_execute.assert_called_once()
                    assert mock_execute.call_args[0][0] == "get_cluster_topics"
                    
                    # Verify response metadata
                    assert result["metadata"]["used_tools"] is True
                    assert "get_cluster_topics" in result["metadata"]["tools_executed"]
                    assert result["metadata"]["n_papers"] == 0  # No papers for tool queries

    def test_query_routing_for_paper_search(self, mock_embeddings_manager, mock_database):
        """Test that queries requiring papers are routed to paper search tool."""
        with patch("abstracts_explorer.rag.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            
            # Mock tool execution to return papers
            with patch("abstracts_explorer.rag.execute_mcp_tool") as mock_execute:
                mock_execute.return_value = json.dumps({
                    "topic": "attention mechanism",
                    "papers_found": 3,
                    "papers": [
                        {
                            "id": "1",
                            "title": "Attention Mechanisms",
                            "abstract": "About attention...",
                            "year": 2023,
                            "conference": "NeurIPS",
                            "session": "Oral",
                            "relevance_score": 0.95,
                        },
                    ],
                }, indent=2)
                
                # Mock routing response - returns JSON tool call for get_recent_developments
                mock_route_response = Mock()
                mock_route_choice = Mock()
                mock_route_message = Mock()
                mock_route_message.content = '{"name": "get_recent_developments", "arguments": {"topic_keywords": "attention mechanism neural networks", "n_results": 5}}'
                mock_route_choice.message = mock_route_message
                mock_route_response.choices = [mock_route_choice]
                
                # Mock final response
                mock_final_response = Mock()
                mock_final_choice = Mock()
                mock_final_message = Mock()
                mock_final_message.content = "Attention mechanisms allow models to focus..."
                mock_final_choice.message = mock_final_message
                mock_final_response.choices = [mock_final_choice]
                
                mock_client.chat.completions.create.side_effect = [
                    mock_route_response,  # Routing/analysis call
                    mock_final_response   # Response generation call
                ]
            
            # Mock paper search
            mock_embeddings_manager.search_similar.return_value = {
                "ids": [["1", "2", "3"]],
                "distances": [[0.1, 0.2, 0.3]],
                "metadatas": [[
                    {"openreview_id": "1", "title": "Paper 1", "year": 2024},
                    {"openreview_id": "2", "title": "Paper 2", "year": 2024},
                    {"openreview_id": "3", "title": "Paper 3", "year": 2024},
                ]],
                "documents": [["Abstract 1", "Abstract 2", "Abstract 3"]],
            }
            
            mock_database.get_paper_by_openreview_id.side_effect = [
                {"openreview_id": "1", "title": "Paper 1", "abstract": "Abstract 1"},
                {"openreview_id": "2", "title": "Paper 2", "abstract": "Abstract 2"},
                {"openreview_id": "3", "title": "Paper 3", "abstract": "Abstract 3"},
            ]
            
            chat = RAGChat(mock_embeddings_manager, mock_database, enable_mcp_tools=True)
            result = chat.query("Explain attention mechanism")
            
            # Verify get_recent_developments tool was executed
            mock_execute.assert_called_once()
            call_args = mock_execute.call_args[0]
            assert call_args[0] == "get_recent_developments"
            
            # Verify papers were returned
            assert len(result["papers"]) > 0
            
            # Verify metadata shows tools were used
            assert result["metadata"]["used_tools"] is True
            assert "get_recent_developments" in result["metadata"]["tools_executed"]
            assert result["metadata"]["n_papers"] == 3
            assert len(result["papers"]) == 3


class TestRAGChatMCPToolsE2E:
    """E2E tests for RAG chat with MCP tools - both mocked and real LLM."""

    def test_mocked_query_triggers_cluster_topics(self, mock_embeddings_manager, mock_database):
        """Test that a query about main topics triggers get_cluster_topics (mocked)."""
        with patch("abstracts_explorer.rag.OpenAI") as mock_openai_class:
            with patch("abstracts_explorer.rag.execute_mcp_tool") as mock_execute:
                mock_client = Mock()
                mock_openai_class.return_value = mock_client
                
                # Mock routing response with JSON tool call
                mock_route_response = Mock()
                mock_route_choice = Mock()
                mock_route_message = Mock()
                mock_route_message.content = '{"name": "get_cluster_topics", "arguments": {"n_clusters": 8}}'
                mock_route_choice.message = mock_route_message
                mock_route_response.choices = [mock_route_choice]
                
                # Mock tool execution result
                mock_execute.return_value = json.dumps({
                    "statistics": {"n_clusters": 8, "total_papers": 100},
                    "clusters": [
                        {
                            "cluster_id": 0,
                            "paper_count": 20,
                            "keywords": [{"keyword": "transformers", "count": 15}]
                        }
                    ]
                })
                
                # Mock final response after tool execution
                mock_final_response = Mock()
                mock_final_choice = Mock()
                mock_final_message = Mock()
                mock_final_message.content = "The main topics include transformers and attention mechanisms."
                mock_final_choice.message = mock_final_message
                mock_final_response.choices = [mock_final_choice]
                
                # Set up create to return routing response first, then final
                mock_client.chat.completions.create.side_effect = [
                    mock_route_response,
                    mock_final_response
                ]
                
                chat = RAGChat(mock_embeddings_manager, mock_database, enable_mcp_tools=True)
                result = chat.query("What are the main research topics at this conference?")
                
                # Verify tool was executed
                mock_execute.assert_called_once_with("get_cluster_topics", {"n_clusters": 8})
                
                # Verify final response was returned
                assert "main topics" in result["response"].lower() or "transformers" in result["response"].lower()

    def test_mocked_query_triggers_topic_evolution(self, mock_embeddings_manager, mock_database):
        """Test that a query about trends triggers get_topic_evolution (mocked)."""
        with patch("abstracts_explorer.rag.OpenAI") as mock_openai_class:
            with patch("abstracts_explorer.rag.execute_mcp_tool") as mock_execute:
                mock_client = Mock()
                mock_openai_class.return_value = mock_client
                
                # Mock routing response with JSON tool call
                mock_route_response = Mock()
                mock_route_choice = Mock()
                mock_route_message = Mock()
                mock_route_message.content = '{"name": "get_topic_evolution", "arguments": {"topic_keywords": "transformers", "conference": "neurips"}}'
                mock_route_choice.message = mock_route_message
                mock_route_response.choices = [mock_route_choice]
                
                # Mock tool execution result
                mock_execute.return_value = json.dumps({
                    "topic": "transformers",
                    "year_counts": {"2020": 10, "2021": 15, "2022": 20},
                    "total_papers": 45
                })
                
                # Mock final response after tool execution
                mock_final_response = Mock()
                mock_final_choice = Mock()
                mock_final_message = Mock()
                mock_final_message.content = "Transformers have grown from 10 papers in 2020 to 20 papers in 2022."
                mock_final_choice.message = mock_final_message
                mock_final_response.choices = [mock_final_choice]
                
                # Set up create to return routing response first, then final
                mock_client.chat.completions.create.side_effect = [
                    mock_route_response,
                    mock_final_response
                ]
                
                chat = RAGChat(mock_embeddings_manager, mock_database, enable_mcp_tools=True)
                result = chat.query("How have transformers evolved at NeurIPS over the years?")
                
                # Verify tool was executed with correct parameters
                mock_execute.assert_called_once_with(
                    "get_topic_evolution",
                    {"topic_keywords": "transformers", "conference": "neurips"}
                )
                
                # Verify final response was returned
                assert "2020" in result["response"] or "evolved" in result["response"].lower()

    def test_mocked_query_triggers_recent_developments(self, mock_embeddings_manager, mock_database):
        """Test that a query about recent papers triggers get_recent_developments (mocked)."""
        with patch("abstracts_explorer.rag.OpenAI") as mock_openai_class:
            with patch("abstracts_explorer.rag.execute_mcp_tool") as mock_execute:
                mock_client = Mock()
                mock_openai_class.return_value = mock_client
                
                # Mock routing response with JSON tool call
                mock_route_response = Mock()
                mock_route_choice = Mock()
                mock_route_message = Mock()
                mock_route_message.content = '{"name": "get_recent_developments", "arguments": {"topic_keywords": "large language models", "n_years": 2}}'
                mock_route_choice.message = mock_route_message
                mock_route_response.choices = [mock_route_choice]
                
                # Mock tool execution result
                mock_execute.return_value = json.dumps({
                    "topic": "large language models",
                    "papers_found": 5,
                    "papers": [
                        {"title": "GPT-4 Architecture", "year": 2024},
                        {"title": "Scaling Laws", "year": 2024}
                    ]
                })
                
                # Mock final response after tool execution
                mock_final_response = Mock()
                mock_final_choice = Mock()
                mock_final_message = Mock()
                mock_final_message.content = "Recent papers include GPT-4 Architecture and Scaling Laws."
                mock_final_choice.message = mock_final_message
                mock_final_response.choices = [mock_final_choice]
                
                # Set up create to return routing response first, then final
                mock_client.chat.completions.create.side_effect = [
                    mock_route_response,
                    mock_final_response
                ]
                
                chat = RAGChat(mock_embeddings_manager, mock_database, enable_mcp_tools=True)
                result = chat.query("What are the latest papers on large language models?")
                
                # Verify tool was executed
                mock_execute.assert_called_once_with(
                    "get_recent_developments",
                    {"topic_keywords": "large language models", "n_years": 2}
                )
                
                # Verify final response was returned
                assert "GPT-4" in result["response"] or "recent" in result["response"].lower()

    @requires_lm_studio
    def test_real_llm_query_triggers_tools(self, mock_embeddings_manager, mock_database):
        """Test with real LLM that it can decide to use MCP tools (requires LM Studio)."""
        # This test requires a real LLM backend running
        # It will be skipped if LM Studio is not available
        
        chat = RAGChat(mock_embeddings_manager, mock_database, enable_mcp_tools=True)
        
        # Query that should trigger clustering tools
        result = chat.query("What are the main research topics covered in the conference?")
        
        # Verify we got a response
        assert "response" in result
        assert len(result["response"]) > 0
        
        # Note: We can't assert specific tool calls without introspecting the LLM's behavior,
        # but this test ensures the integration works end-to-end with a real LLM


class TestJSONToolCalls:
    """Tests for JSON-based tool call detection and handling."""
    
    def test_parse_json_tool_call_single(self):
        """Test parsing a single JSON tool call."""
        from abstracts_explorer.rag import parse_json_tool_call
        
        json_response = '{"name": "analyze_topic_relevance", "arguments": {"query": "transformers", "distance_threshold": 1.1}}'
        tool_calls = parse_json_tool_call(json_response)
        
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0]['name'] == 'analyze_topic_relevance'
        assert tool_calls[0]['arguments']['query'] == 'transformers'
        assert tool_calls[0]['arguments']['distance_threshold'] == 1.1
    
    def test_parse_json_tool_call_array(self):
        """Test parsing multiple JSON tool calls in array format."""
        from abstracts_explorer.rag import parse_json_tool_call
        
        json_response = '''[
            {"name": "get_cluster_topics", "arguments": {"n_clusters": 8}},
            {"name": "get_recent_developments", "arguments": {"topic_keywords": "transformers"}}
        ]'''
        tool_calls = parse_json_tool_call(json_response)
        
        assert tool_calls is not None
        assert len(tool_calls) == 2
        assert tool_calls[0]['name'] == 'get_cluster_topics'
        assert tool_calls[0]['arguments']['n_clusters'] == 8
        assert tool_calls[1]['name'] == 'get_recent_developments'
        assert tool_calls[1]['arguments']['topic_keywords'] == 'transformers'
    
    def test_parse_json_tool_call_openai_format(self):
        """Test parsing OpenAI-like JSON format."""
        from abstracts_explorer.rag import parse_json_tool_call
        
        json_response = '''{
            "tool_calls": [
                {
                    "function": {
                        "name": "analyze_topic_relevance",
                        "arguments": {"query": "transformers"}
                    }
                }
            ]
        }'''
        tool_calls = parse_json_tool_call(json_response)
        
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0]['name'] == 'analyze_topic_relevance'
        assert tool_calls[0]['arguments']['query'] == 'transformers'
    
    def test_parse_json_tool_call_function_format(self):
        """Test parsing function-only JSON format."""
        from abstracts_explorer.rag import parse_json_tool_call
        
        json_response = '{"function": {"name": "get_cluster_topics", "arguments": {"n_clusters": 5}}}'
        tool_calls = parse_json_tool_call(json_response)
        
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0]['name'] == 'get_cluster_topics'
        assert tool_calls[0]['arguments']['n_clusters'] == 5
    
    def test_parse_json_tool_call_not_json(self):
        """Test that non-JSON responses return None."""
        from abstracts_explorer.rag import parse_json_tool_call
        
        text_response = "This is a regular text response, not JSON."
        tool_calls = parse_json_tool_call(text_response)
        
        assert tool_calls is None
    
    def test_parse_json_tool_call_invalid_format(self):
        """Test that JSON without tool call structure returns None."""
        from abstracts_explorer.rag import parse_json_tool_call
        
        json_response = '{"message": "This is just a JSON message", "value": 123}'
        tool_calls = parse_json_tool_call(json_response)
        
        assert tool_calls is None
    
    def test_parse_json_tool_call_empty(self):
        """Test that empty input returns None."""
        from abstracts_explorer.rag import parse_json_tool_call
        
        assert parse_json_tool_call("") is None
        assert parse_json_tool_call("  ") is None
        assert parse_json_tool_call(None) is None
    
    def test_handle_json_tool_calls_integration(self, mock_embeddings_manager, mock_database):
        """Test that JSON tool calls are executed and result in proper response."""
        with patch("abstracts_explorer.rag.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            
            # First call: Query routing - returns JSON tool call
            mock_route_response = Mock()
            mock_route_choice = Mock()
            mock_route_message = Mock()
            mock_route_message.content = '{"name": "analyze_topic_relevance", "arguments": {"query": "transformers", "distance_threshold": 1.1}}'
            mock_route_choice.message = mock_route_message
            mock_route_response.choices = [mock_route_choice]
            
            # Second call: Model generates final answer based on tool results
            mock_final_response = Mock()
            mock_final_choice = Mock()
            mock_final_message = Mock()
            mock_final_message.content = "Based on the tool results, there are 42 papers about transformers."
            mock_final_choice.message = mock_final_message
            mock_final_response.choices = [mock_final_choice]
            
            # Set up side_effect for sequential calls
            mock_client.chat.completions.create.side_effect = [
                mock_route_response,  # Query routing
                mock_final_response   # Final answer
            ]
            
            # Mock the tool execution and formatting
            with patch("abstracts_explorer.rag.execute_mcp_tool") as mock_execute, \
                 patch("abstracts_explorer.rag.format_tool_result_for_llm") as mock_format:
                
                mock_execute.return_value = json.dumps({
                    "query": "transformers",
                    "total_papers": 42,
                    "relevance_score": 85
                })
                mock_format.return_value = "Found 42 papers about transformers with relevance score 85"
                
                chat = RAGChat(mock_embeddings_manager, mock_database, enable_mcp_tools=True)
                result = chat.query("How many papers about transformers?")
                
                # Verify tool was executed
                mock_execute.assert_called_once_with(
                    'analyze_topic_relevance',
                    {'query': 'transformers', 'distance_threshold': 1.1}
                )
                
                # Verify final response contains the answer
                assert "42 papers" in result["response"]
                # Verify metadata shows tools were used
                assert result["metadata"]["used_tools"] is True
    
    def test_handle_json_tool_calls_array(self, mock_embeddings_manager, mock_database):
        """Test handling multiple JSON tool calls in array format."""
        with patch("abstracts_explorer.rag.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            
            # First call: Query routing - returns JSON array of tool calls
            mock_route_response = Mock()
            mock_route_choice = Mock()
            mock_route_message = Mock()
            mock_route_message.content = '''[
                {"name": "get_cluster_topics", "arguments": {"n_clusters": 5}},
                {"name": "get_recent_developments", "arguments": {"topic_keywords": "transformers"}}
            ]'''
            mock_route_choice.message = mock_route_message
            mock_route_response.choices = [mock_route_choice]
            
            
            # Second call: Final answer
            mock_final_response = Mock()
            mock_final_choice = Mock()
            mock_final_message = Mock()
            mock_final_message.content = "Based on clustering and recent papers, transformers are a major topic."
            mock_final_choice.message = mock_final_message
            mock_final_response.choices = [mock_final_choice]
            
            mock_client.chat.completions.create.side_effect = [
                mock_route_response,
                mock_final_response
            ]
            
            # Mock the tool execution
            with patch("abstracts_explorer.rag.execute_mcp_tool") as mock_execute, \
                 patch("abstracts_explorer.rag.format_tool_result_for_llm") as mock_format:
                
                mock_execute.side_effect = [
                    json.dumps({"clusters": 5, "total_papers": 100}),
                    json.dumps({"papers": [{"title": "New Transformer"}]})
                ]
                mock_format.side_effect = [
                    "Found 5 clusters with 100 total papers",
                    "Recent papers include: New Transformer"
                ]
                
                chat = RAGChat(mock_embeddings_manager, mock_database, enable_mcp_tools=True)
                result = chat.query("What's trending?")
                
                # Verify both tools were executed
                assert mock_execute.call_count == 2
                assert mock_execute.call_args_list[0][0] == ('get_cluster_topics', {'n_clusters': 5})
                assert mock_execute.call_args_list[1][0] == ('get_recent_developments', {'topic_keywords': 'transformers'})
                
                # Verify final response
                assert "transformers" in result["response"].lower()
                # Verify metadata shows tools were used
                assert result["metadata"]["used_tools"] is True
    
    def test_json_tool_calls_disabled_when_mcp_disabled(self, mock_embeddings_manager, mock_database):
        """Test that JSON tool calls are handled when MCP tools are disabled."""
        with patch("abstracts_explorer.rag.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            
            # Query routing returns JSON for paper search even when MCP tools disabled
            mock_route_response = Mock()
            mock_route_choice = Mock()
            mock_route_message = Mock()
            mock_route_message.content = '{"name": "get_recent_developments", "arguments": {"topic_keywords": "test query", "n_results": 5}}'
            mock_route_choice.message = mock_route_message
            mock_route_response.choices = [mock_route_choice]
            
            # Second call: Final response generation
            mock_response = Mock()
            mock_choice = Mock()
            mock_message = Mock()
            mock_message.content = "Here are some relevant papers about test query..."
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            
            # Mock paper search
            mock_embeddings_manager.search_similar.return_value = {
                "ids": [["1", "2", "3"]],
                "distances": [[0.1, 0.2, 0.3]],
                "metadatas": [[
                    {"openreview_id": "1", "title": "Paper 1", "year": 2024},
                    {"openreview_id": "2", "title": "Paper 2", "year": 2024},
                    {"openreview_id": "3", "title": "Paper 3", "year": 2024},
                ]],
                "documents": [["Abstract 1", "Abstract 2", "Abstract 3"]],
            }
            
            mock_database.get_paper_by_openreview_id.side_effect = [
                {"openreview_id": "1", "title": "Paper 1", "abstract": "Abstract 1"},
                {"openreview_id": "2", "title": "Paper 2", "abstract": "Abstract 2"},
                {"openreview_id": "3", "title": "Paper 3", "abstract": "Abstract 3"},
            ]
            
            mock_client.chat.completions.create.side_effect = [
                mock_route_response,
                mock_response
            ]
            
            chat = RAGChat(mock_embeddings_manager, mock_database, enable_mcp_tools=False)
            result = chat.query("Test query")
            
            # In the new architecture, all queries go through tools (including paper search)
            # So used_tools is True even when MCP clustering tools are disabled
            assert result["metadata"]["used_tools"] is True
            assert result["metadata"]["n_papers"] == 3
            
            # Should have made two API calls (routing + generation)
            assert mock_client.chat.completions.create.call_count == 2

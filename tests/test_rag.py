"""
Tests for the RAG (Retrieval Augmented Generation) module.

This module tests the RAGChat functionality using Pydantic AI's TestModel and
FunctionModel for deterministic unit testing.
Tests that require a running LM Studio instance are skipped if it's not available.
"""

import json
import pytest
from unittest.mock import Mock, patch

from pydantic_ai.models.test import TestModel
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.messages import (
    ModelResponse,
    TextPart,
    ToolCallPart,
)

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
                "authors": "Vaswani et al.",
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
    mock_db.get_conferences.return_value = ["NeurIPS"]
    mock_db.get_years.return_value = [2025, 2024]
    mock_db.get_sessions.return_value = []

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
def mock_search_papers():
    """Mock mcp_server.search_papers to return test papers."""
    with patch("abstracts_explorer.rag.mcp_search_papers") as mock_sp:
        mock_sp.return_value = json.dumps(
            {
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
            },
            indent=2,
        )
        yield mock_sp


@pytest.fixture
def mock_cluster_topics():
    """Mock mcp_server.get_conference_topics to return test data."""
    with patch("abstracts_explorer.rag.mcp_get_conference_topics") as mock_ct:
        mock_ct.return_value = json.dumps(
            {
                "n_topics": 8,
                "total_papers": 100,
                "topics": [
                    {
                        "topic": "Transformers",
                        "paper_count": 20,
                        "keywords": ["transformers", "attention"],
                    }
                ],
            }
        )
        yield mock_ct


@pytest.fixture
def mock_topic_evolution():
    """Mock mcp_server.get_topic_evolution to return test data."""
    with patch("abstracts_explorer.rag.mcp_get_topic_evolution") as mock_te:
        mock_te.return_value = json.dumps(
            {
                "topic": "transformers",
                "conferences": ["NeurIPS"],
                "conference_data": {
                    "NeurIPS": {
                        "year_counts": {"2020": 10, "2021": 15, "2022": 20},
                        "year_relative": {"2020": 5.0, "2021": 6.0, "2022": 7.0},
                        "year_totals": {"2020": 200, "2021": 250, "2022": 286},
                    }
                },
                "total_papers": 45,
            }
        )
        yield mock_te


@pytest.fixture
def mock_analyze_topic():
    """Mock mcp_server.analyze_topic_relevance to return test data."""
    with patch("abstracts_explorer.rag.mcp_analyze_topic_relevance") as mock_at:
        mock_at.return_value = json.dumps(
            {
                "topic": "uncertainty quantification",
                "total_papers": 42,
                "relevance_score": 85.0,
                "distance_threshold": 1.1,
                "filters": {"conferences": None, "years": [2025]},
                "conferences": {"NeurIPS": 42},
                "years": {"2025": 42},
                "sample_papers": [
                    {
                        "title": "Uncertainty in Deep Learning",
                        "year": 2025,
                        "conference": "NeurIPS",
                        "distance": 0.5,
                    }
                ],
                "closest_distance": 0.5,
            }
        )
        yield mock_at


@pytest.fixture
def mock_all_tools(mock_search_papers, mock_cluster_topics, mock_topic_evolution, mock_analyze_topic):
    """Mock all MCP server tools."""
    with patch("abstracts_explorer.rag.mcp_get_cluster_visualization") as mock_viz:
        mock_viz.return_value = json.dumps(
            {
                "n_dimensions": 2,
                "n_points": 100,
                "statistics": {"n_clusters": 5},
                "points": [],
                "visualization_saved": False,
            }
        )
        yield {
            "search_papers": mock_search_papers,
            "cluster_topics": mock_cluster_topics,
            "topic_evolution": mock_topic_evolution,
            "analyze_topic": mock_analyze_topic,
            "visualization": mock_viz,
        }


class TestRAGChatInit:
    """Test RAGChat initialization."""

    def test_init_with_defaults(self, mock_embeddings_manager, mock_database):
        """Test initialization with default parameters."""
        chat = RAGChat(mock_embeddings_manager, mock_database)

        assert chat.embeddings_manager == mock_embeddings_manager
        assert chat.database == mock_database
        assert chat.lm_studio_url == "https://api.helmholtz-blablador.fz-juelich.de"
        assert chat.model == "alias-code"
        assert chat.max_context_papers > 0
        assert 0 <= chat.temperature <= 1
        assert chat.conversation_history == []
        assert chat.agent is not None

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

    def test_init_with_mcp_tools_enabled(self, mock_embeddings_manager, mock_database):
        """Test that MCP tools can be enabled during initialization."""
        chat = RAGChat(mock_embeddings_manager, mock_database, enable_mcp_tools=True)
        assert chat.enable_mcp_tools is True

    def test_init_with_mcp_tools_disabled(self, mock_embeddings_manager, mock_database):
        """Test that MCP tools can be disabled during initialization."""
        chat = RAGChat(mock_embeddings_manager, mock_database, enable_mcp_tools=False)
        assert chat.enable_mcp_tools is False


class TestRAGChatInstructions:
    """Test RAGChat instruction building with conference context."""

    def test_build_base_instructions_contains_defaults(self, mock_embeddings_manager, mock_database):
        """Test that base instructions contain the standard content."""
        chat = RAGChat(mock_embeddings_manager, mock_database)
        instructions = chat._build_base_instructions()

        assert "AI assistant" in instructions
        assert "conference data" in instructions
        assert "Today's date" in instructions

    def test_build_instructions_with_conference_and_year(self, mock_embeddings_manager, mock_database):
        """Test instructions include selected conference and year."""
        chat = RAGChat(mock_embeddings_manager, mock_database)
        instructions = chat._build_instructions(
            conferences=["NeurIPS"],
            years=[2025],
        )

        assert "NeurIPS" in instructions
        assert "2025" in instructions
        assert "default conference" in instructions
        assert "default year" in instructions

    def test_build_instructions_with_conference_only(self, mock_embeddings_manager, mock_database):
        """Test instructions include conference when no year selected."""
        chat = RAGChat(mock_embeddings_manager, mock_database)
        instructions = chat._build_instructions(conferences=["ICLR"])

        assert "ICLR" in instructions
        assert "default conference" in instructions
        assert "default year" not in instructions

    def test_build_instructions_with_year_only(self, mock_embeddings_manager, mock_database):
        """Test instructions include year when no conference selected."""
        chat = RAGChat(mock_embeddings_manager, mock_database)
        instructions = chat._build_instructions(years=[2024])

        assert "2024" in instructions
        assert "default year" in instructions
        assert "default conference" not in instructions

    def test_build_instructions_with_available_conferences(self, mock_embeddings_manager, mock_database):
        """Test instructions include available conferences list."""
        chat = RAGChat(mock_embeddings_manager, mock_database)
        instructions = chat._build_instructions(
            available_conferences=["NeurIPS", "ICLR", "ICML"],
        )

        assert "NeurIPS" in instructions
        assert "ICLR" in instructions
        assert "ICML" in instructions
        assert "available conferences" in instructions

    def test_build_instructions_no_context(self, mock_embeddings_manager, mock_database):
        """Test instructions without any context match base instructions."""
        chat = RAGChat(mock_embeddings_manager, mock_database)
        base = chat._build_base_instructions()
        full = chat._build_instructions()

        assert full == base

    def test_build_instructions_multiple_conferences(self, mock_embeddings_manager, mock_database):
        """Test instructions with multiple selected conferences."""
        chat = RAGChat(mock_embeddings_manager, mock_database)
        instructions = chat._build_instructions(
            conferences=["NeurIPS", "ICLR"],
            years=[2024, 2025],
        )

        assert "NeurIPS" in instructions
        assert "ICLR" in instructions
        assert "2024" in instructions
        assert "2025" in instructions

    def test_query_with_conference_context(self, mock_embeddings_manager, mock_database, mock_search_papers):
        """Test that query passes conference context to instructions."""
        chat = RAGChat(mock_embeddings_manager, mock_database)

        with chat.agent.override(model=TestModel()):
            result = chat.query(
                "What are the main topics?",
                conferences=["NeurIPS"],
                years=[2025],
                available_conferences=["NeurIPS", "ICLR", "ICML"],
            )

        assert "response" in result

    def test_query_system_prompt_overrides_conference_context(
        self, mock_embeddings_manager, mock_database, mock_search_papers
    ):
        """Test that explicit system_prompt overrides conference context instructions."""
        chat = RAGChat(mock_embeddings_manager, mock_database)

        custom_prompt = "Custom system prompt."
        with chat.agent.override(model=TestModel()):
            result = chat.query(
                "Test question",
                conferences=["NeurIPS"],
                years=[2025],
                system_prompt=custom_prompt,
            )

        assert "response" in result


class TestRAGChatQuery:
    """Test RAGChat query method."""

    def test_query_success(self, mock_embeddings_manager, mock_database, mock_search_papers):
        """Test successful query with papers."""
        chat = RAGChat(mock_embeddings_manager, mock_database)

        with chat.agent.override(model=TestModel()):
            result = chat.query("What is attention mechanism?")

        assert "response" in result
        assert "papers" in result
        assert "visualizations" in result
        assert "metadata" in result
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0

        # Check conversation history
        history = chat.conversation_history
        assert len(history) >= 2
        assert history[0]["role"] == "user"

    def test_query_returns_papers_from_search(self, mock_embeddings_manager, mock_database, mock_search_papers):
        """Test that query returns papers extracted from search_papers tool."""
        chat = RAGChat(mock_embeddings_manager, mock_database)

        with chat.agent.override(model=TestModel()):
            result = chat.query("What is attention mechanism?")

        assert len(result["papers"]) == 3
        assert result["metadata"]["n_papers"] == 3

    def test_query_with_system_prompt(self, mock_embeddings_manager, mock_database, mock_search_papers):
        """Test query with custom system prompt."""
        chat = RAGChat(mock_embeddings_manager, mock_database)

        custom_prompt = "You are a helpful assistant specializing in machine learning."
        with chat.agent.override(model=TestModel()):
            result = chat.query("Explain transformers", system_prompt=custom_prompt)

        assert "response" in result
        assert isinstance(result["response"], str)

    def test_query_no_results(self, mock_embeddings_manager_empty, mock_database):
        """Test query when no papers are found."""
        with patch("abstracts_explorer.rag.mcp_search_papers") as mock_sp:
            mock_sp.return_value = json.dumps(
                {"topic": "Unknown topic", "papers_found": 0, "papers": []},
                indent=2,
            )

            chat = RAGChat(mock_embeddings_manager_empty, mock_database)

            with chat.agent.override(model=TestModel()):
                result = chat.query("Unknown topic")

            assert "response" in result
            assert result["papers"] == []
            assert result["metadata"]["n_papers"] == 0

    def test_query_api_error_raises_rag_error(self, mock_embeddings_manager, mock_database):
        """Test query with API error raises RAGError."""
        chat = RAGChat(mock_embeddings_manager, mock_database)

        # FunctionModel that raises an exception
        def failing_model(messages, info):
            raise Exception("API connection failed")

        with chat.agent.override(model=FunctionModel(failing_model)):
            with pytest.raises(RAGError, match="Query failed"):
                chat.query("What is deep learning?")

    def test_query_general_exception(self, mock_embeddings_manager, mock_database):
        """Test query with general exception."""
        # Mock the search_papers tool to raise exception
        with patch("abstracts_explorer.rag.mcp_search_papers") as mock_sp:
            mock_sp.side_effect = Exception("Unexpected error")

            chat = RAGChat(mock_embeddings_manager, mock_database)

            with chat.agent.override(model=TestModel()):
                with pytest.raises(RAGError, match="Query failed"):
                    chat.query("What is NLP?")


class TestRAGChatChat:
    """Test RAGChat chat method."""

    def test_chat_with_context(self, mock_embeddings_manager, mock_database, mock_search_papers):
        """Test chat with context retrieval."""
        chat = RAGChat(mock_embeddings_manager, mock_database)

        with chat.agent.override(model=TestModel()):
            result = chat.chat("Tell me about transformers")

        assert "response" in result
        assert len(result["papers"]) > 0
        mock_search_papers.assert_called()

    def test_chat_without_context(self, mock_embeddings_manager, mock_database):
        """Test chat without context retrieval."""
        chat = RAGChat(mock_embeddings_manager, mock_database)

        with chat.agent.override(model=TestModel()):
            result = chat.chat("Hello, how are you?", use_context=False)

        assert "response" in result
        assert result["papers"] == []
        assert result["metadata"]["n_papers"] == 0

    def test_chat_custom_n_results(self, mock_embeddings_manager, mock_database, mock_search_papers):
        """Test chat with custom n_results."""
        chat = RAGChat(mock_embeddings_manager, mock_database)

        with chat.agent.override(model=TestModel()):
            chat.chat("What is AI?", n_results=7)

        # Verify search was called
        mock_search_papers.assert_called()


class TestRAGChatConversation:
    """Test conversation history management."""

    def test_reset_conversation(self, mock_embeddings_manager, mock_database, mock_search_papers):
        """Test resetting conversation history."""
        chat = RAGChat(mock_embeddings_manager, mock_database)

        with chat.agent.override(model=TestModel()):
            chat.query("First question")
            chat.query("Second question")

        assert len(chat.conversation_history) > 0

        # Reset
        chat.reset_conversation()

        assert len(chat.conversation_history) == 0

    def test_conversation_history_accumulates(self, mock_embeddings_manager, mock_database, mock_search_papers):
        """Test that conversation history accumulates across queries."""
        chat = RAGChat(mock_embeddings_manager, mock_database)

        with chat.agent.override(model=TestModel()):
            chat.query("First question")
            history_1 = len(chat.conversation_history)

            chat.query("Second question")
            history_2 = len(chat.conversation_history)

        assert history_2 > history_1

    def test_conversation_history_format(self, mock_embeddings_manager, mock_database, mock_search_papers):
        """Test that conversation_history returns proper role/content dicts."""
        chat = RAGChat(mock_embeddings_manager, mock_database)

        with chat.agent.override(model=TestModel()):
            chat.query("What is attention?")

        history = chat.conversation_history
        assert len(history) >= 2

        # Check format
        user_msgs = [m for m in history if m["role"] == "user"]
        assistant_msgs = [m for m in history if m["role"] == "assistant"]

        assert len(user_msgs) >= 1
        assert len(assistant_msgs) >= 1
        assert "content" in user_msgs[0]
        assert "content" in assistant_msgs[0]


class TestRAGChatExport:
    """Test conversation export functionality."""

    def test_export_conversation(self, mock_embeddings_manager, mock_database, mock_search_papers, tmp_path):
        """Test exporting conversation to JSON."""
        chat = RAGChat(mock_embeddings_manager, mock_database)

        with chat.agent.override(model=TestModel()):
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
        assert len(data) >= 4  # At least 2 questions + 2 responses
        assert data[0]["role"] == "user"

    def test_export_empty_conversation(self, mock_embeddings_manager, mock_database, tmp_path):
        """Test exporting empty conversation."""
        chat = RAGChat(mock_embeddings_manager, mock_database)

        output_path = tmp_path / "empty_conversation.json"
        chat.export_conversation(output_path)

        assert output_path.exists()

        with open(output_path, "r") as f:
            data = json.load(f)

        assert data == []


class TestRAGChatMCPTools:
    """Test RAGChat integration with MCP clustering tools."""

    def test_query_calls_search_papers(self, mock_embeddings_manager, mock_database, mock_search_papers):
        """Test that search_papers tool is called during query."""
        chat = RAGChat(mock_embeddings_manager, mock_database)

        with chat.agent.override(model=TestModel()):
            result = chat.query("Explain attention mechanism")

        # Verify search_papers was called
        mock_search_papers.assert_called()

        # Verify papers were returned
        assert len(result["papers"]) > 0
        assert result["metadata"]["used_tools"] is True
        assert "search_papers" in result["metadata"]["tools_executed"]

    def test_query_calls_cluster_topics(self, mock_embeddings_manager, mock_database, mock_cluster_topics):
        """Test that get_conference_topics tool can be called."""

        def model_fn(messages, info):
            """Model that calls get_conference_topics on first call, then responds."""
            # Check if tool results already exist (tool was already called)
            for msg in messages:
                for part in msg.parts:
                    if getattr(part, "part_kind", None) == "tool-return":
                        return ModelResponse(
                            parts=[TextPart(content="The main topics are transformers and diffusion models.")]
                        )
            # First call: request the tool
            return ModelResponse(parts=[ToolCallPart(tool_name="get_conference_topics", args={})])

        chat = RAGChat(mock_embeddings_manager, mock_database, enable_mcp_tools=True)

        with chat.agent.override(model=FunctionModel(model_fn)):
            result = chat.query("What are the main research topics?")

        mock_cluster_topics.assert_called_once()
        assert result["metadata"]["used_tools"] is True
        assert "get_conference_topics" in result["metadata"]["tools_executed"]

    def test_query_calls_topic_evolution(self, mock_embeddings_manager, mock_database, mock_topic_evolution):
        """Test that get_topic_evolution tool can be called."""

        def model_fn(messages, info):
            for msg in messages:
                for part in msg.parts:
                    if getattr(part, "part_kind", None) == "tool-return":
                        return ModelResponse(
                            parts=[TextPart(content="Transformers have grown from 10 papers in 2020 to 20 in 2022.")]
                        )
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="get_topic_evolution",
                        args={"topic_keywords": "transformers", "conferences": ["neurips"]},
                    )
                ]
            )

        chat = RAGChat(mock_embeddings_manager, mock_database, enable_mcp_tools=True)

        with chat.agent.override(model=FunctionModel(model_fn)):
            result = chat.query("How have transformers evolved at NeurIPS?")

        mock_topic_evolution.assert_called_once()
        assert "get_topic_evolution" in result["metadata"]["tools_executed"]

    def test_query_calls_analyze_topic_relevance(self, mock_embeddings_manager, mock_database, mock_analyze_topic):
        """Test that analyze_topic_relevance tool can be called."""

        def model_fn(messages, info):
            for msg in messages:
                for part in msg.parts:
                    if getattr(part, "part_kind", None) == "tool-return":
                        return ModelResponse(
                            parts=[TextPart(content="There were 42 papers about uncertainty quantification.")]
                        )
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="analyze_topic_relevance",
                        args={"topic": "uncertainty quantification", "years": [2025]},
                    )
                ]
            )

        chat = RAGChat(mock_embeddings_manager, mock_database, enable_mcp_tools=True)

        with chat.agent.override(model=FunctionModel(model_fn)):
            result = chat.query("How many papers about uncertainty quantification?")

        mock_analyze_topic.assert_called_once()
        assert "analyze_topic_relevance" in result["metadata"]["tools_executed"]

    def test_mcp_tools_disabled_only_has_search(self, mock_embeddings_manager, mock_database, mock_search_papers):
        """Test that only search_papers is available when MCP tools disabled."""
        chat = RAGChat(mock_embeddings_manager, mock_database, enable_mcp_tools=False)

        with chat.agent.override(model=TestModel()):
            result = chat.query("Test query")

        # Only search_papers should be available
        assert result["metadata"]["used_tools"] is True
        assert "search_papers" in result["metadata"]["tools_executed"]

    def test_multiple_tool_calls(self, mock_embeddings_manager, mock_database, mock_all_tools):
        """Test that multiple tools can be called in a single query."""
        call_count = [0]

        def model_fn(messages, info):
            call_count[0] += 1
            # Check if any tool results have come back
            has_tool_returns = any(
                getattr(part, "part_kind", None) == "tool-return" for msg in messages for part in msg.parts
            )
            if has_tool_returns:
                return ModelResponse(parts=[TextPart(content="The main topics include transformers.")])
            # First call: return two tool calls
            return ModelResponse(
                parts=[
                    ToolCallPart(tool_name="get_conference_topics", args={}),
                    ToolCallPart(
                        tool_name="search_papers",
                        args={"topic_keywords": "transformers", "n_results": 5},
                    ),
                ]
            )

        chat = RAGChat(mock_embeddings_manager, mock_database, enable_mcp_tools=True)

        with chat.agent.override(model=FunctionModel(model_fn)):
            result = chat.query("What are the main topics and show me papers about transformers?")

        # Verify both tools were executed
        assert "get_conference_topics" in result["metadata"]["tools_executed"]
        assert "search_papers" in result["metadata"]["tools_executed"]
        assert len(result["metadata"]["tools_executed"]) == 2


class TestRAGChatPaperExtraction:
    """Test paper extraction from tool results."""

    def test_extract_papers_from_search(self):
        """Test extracting papers from search_papers results."""
        tool_results = [
            {
                "name": "search_papers",
                "raw_result": json.dumps(
                    {
                        "topic": "test",
                        "papers_found": 2,
                        "papers": [
                            {"id": "1", "title": "Paper 1"},
                            {"id": "2", "title": "Paper 2"},
                        ],
                    }
                ),
            }
        ]

        papers = RAGChat._extract_papers(tool_results)
        assert len(papers) == 2
        assert papers[0]["title"] == "Paper 1"

    def test_extract_papers_from_analyze_topic(self):
        """Test extracting papers from analyze_topic_relevance results."""
        tool_results = [
            {
                "name": "analyze_topic_relevance",
                "raw_result": json.dumps(
                    {
                        "topic": "test",
                        "papers": [
                            {"title": "Relevant Paper", "distance": 0.5},
                        ],
                    }
                ),
            }
        ]

        papers = RAGChat._extract_papers(tool_results)
        assert len(papers) == 1

    def test_extract_papers_from_non_paper_tools(self):
        """Test that non-paper tools return no papers."""
        tool_results = [
            {
                "name": "get_conference_topics",
                "raw_result": json.dumps({"topics": []}),
            }
        ]

        papers = RAGChat._extract_papers(tool_results)
        assert len(papers) == 0

    def test_extract_papers_handles_invalid_json(self):
        """Test that invalid JSON is handled gracefully."""
        tool_results = [
            {
                "name": "search_papers",
                "raw_result": "not valid json",
            }
        ]

        papers = RAGChat._extract_papers(tool_results)
        assert len(papers) == 0


class TestRAGChatVisualizationExtraction:
    """Test visualization extraction from tool results."""

    def test_extract_topic_evolution_visualization(self):
        """Test extracting topic evolution chart data."""
        tool_results = [
            {
                "name": "get_topic_evolution",
                "raw_result": json.dumps(
                    {
                        "topic": "transformers",
                        "conferences": ["NeurIPS"],
                        "total_papers": 30,
                        "conference_data": {
                            "NeurIPS": {
                                "year_counts": {"2022": 5, "2023": 10, "2024": 15},
                                "year_relative": {"2022": 2.5, "2023": 4.0, "2024": 5.0},
                                "year_totals": {"2022": 200, "2023": 250, "2024": 300},
                            }
                        },
                    }
                ),
            }
        ]

        visualizations = RAGChat._extract_visualizations(tool_results)
        assert len(visualizations) == 1
        viz = visualizations[0]
        assert viz["type"] == "topic_evolution"
        assert viz["topics"] == ["transformers"]
        assert "transformers" in viz["conference_data"]
        assert "NeurIPS" in viz["conference_data"]["transformers"]

    def test_extract_cluster_visualization(self):
        """Test extracting cluster visualization scatter data."""
        tool_results = [
            {
                "name": "get_cluster_visualization",
                "raw_result": json.dumps(
                    {
                        "n_dimensions": 2,
                        "n_points": 3,
                        "statistics": {"n_clusters": 2, "total_papers": 3},
                        "points": [
                            {"x": 1.0, "y": 2.0, "cluster": 0, "title": "Paper A"},
                            {"x": 3.0, "y": 4.0, "cluster": 1, "title": "Paper B"},
                            {"x": 1.5, "y": 2.5, "cluster": 0, "title": "Paper C"},
                        ],
                    }
                ),
            }
        ]

        visualizations = RAGChat._extract_visualizations(tool_results)
        assert len(visualizations) == 1
        viz = visualizations[0]
        assert viz["type"] == "cluster_visualization"
        assert len(viz["points"]) == 3
        assert viz["statistics"]["n_clusters"] == 2

    def test_extract_visualizations_skips_errors(self):
        """Test that tool results with errors are skipped."""
        tool_results = [
            {
                "name": "get_topic_evolution",
                "raw_result": json.dumps({"error": "No conference specified"}),
            }
        ]

        visualizations = RAGChat._extract_visualizations(tool_results)
        assert len(visualizations) == 0

    def test_extract_visualizations_skips_invalid_json(self):
        """Test that invalid JSON is handled gracefully."""
        tool_results = [
            {
                "name": "get_topic_evolution",
                "raw_result": "not valid json",
            }
        ]

        visualizations = RAGChat._extract_visualizations(tool_results)
        assert len(visualizations) == 0

    def test_extract_visualizations_skips_non_dict_json(self):
        """Test that non-dict JSON values (list, string, null) are skipped."""
        tool_results = [
            {
                "name": "get_topic_evolution",
                "raw_result": json.dumps(["a", "list"]),
            },
            {
                "name": "get_cluster_visualization",
                "raw_result": json.dumps("just a string"),
            },
            {
                "name": "get_topic_evolution",
                "raw_result": json.dumps(None),
            },
        ]

        visualizations = RAGChat._extract_visualizations(tool_results)
        assert len(visualizations) == 0

    def test_extract_visualizations_skips_empty_data(self):
        """Test that results with empty data are skipped."""
        tool_results = [
            {
                "name": "get_topic_evolution",
                "raw_result": json.dumps(
                    {
                        "topic": "nothing",
                        "conferences": ["NeurIPS"],
                        "conference_data": {},
                    }
                ),
            },
            {
                "name": "get_cluster_visualization",
                "raw_result": json.dumps(
                    {
                        "n_dimensions": 2,
                        "points": [],
                        "statistics": {},
                    }
                ),
            },
        ]

        visualizations = RAGChat._extract_visualizations(tool_results)
        assert len(visualizations) == 0

    def test_extract_visualizations_ignores_non_viz_tools(self):
        """Test that non-visualization tools are ignored."""
        tool_results = [
            {
                "name": "search_papers",
                "raw_result": json.dumps({"papers": [{"title": "Paper"}]}),
            },
            {
                "name": "get_conference_topics",
                "raw_result": json.dumps({"topics": []}),
            },
        ]

        visualizations = RAGChat._extract_visualizations(tool_results)
        assert len(visualizations) == 0

    def test_extract_multiple_visualizations(self):
        """Test extracting multiple visualizations from different tools."""
        tool_results = [
            {
                "name": "get_topic_evolution",
                "raw_result": json.dumps(
                    {
                        "topic": "attention",
                        "conferences": ["ICLR"],
                        "conference_data": {
                            "ICLR": {
                                "year_counts": {"2023": 8},
                                "year_relative": {"2023": 3.0},
                            }
                        },
                    }
                ),
            },
            {
                "name": "get_cluster_visualization",
                "raw_result": json.dumps(
                    {
                        "n_dimensions": 2,
                        "points": [{"x": 1, "y": 2, "cluster": 0}],
                        "statistics": {"n_clusters": 1, "total_papers": 1},
                    }
                ),
            },
        ]

        visualizations = RAGChat._extract_visualizations(tool_results)
        assert len(visualizations) == 2
        assert visualizations[0]["type"] == "topic_evolution"
        assert visualizations[1]["type"] == "cluster_visualization"

    def test_extract_multiple_topic_evolutions_merged(self):
        """Test that multiple get_topic_evolution results are merged into one visualization."""
        tool_results = [
            {
                "name": "get_topic_evolution",
                "raw_result": json.dumps(
                    {
                        "topic": "transformers",
                        "conferences": ["NeurIPS"],
                        "conference_data": {
                            "NeurIPS": {
                                "year_counts": {"2022": 5, "2023": 10},
                                "year_relative": {"2022": 2.5, "2023": 4.0},
                            }
                        },
                    }
                ),
            },
            {
                "name": "get_topic_evolution",
                "raw_result": json.dumps(
                    {
                        "topic": "reinforcement learning",
                        "conferences": ["NeurIPS"],
                        "conference_data": {
                            "NeurIPS": {
                                "year_counts": {"2022": 12, "2023": 8},
                                "year_relative": {"2022": 6.0, "2023": 3.2},
                            }
                        },
                    }
                ),
            },
        ]

        visualizations = RAGChat._extract_visualizations(tool_results)
        assert len(visualizations) == 1
        viz = visualizations[0]
        assert viz["type"] == "topic_evolution"
        assert viz["topics"] == ["transformers", "reinforcement learning"]
        assert "transformers" in viz["conference_data"]
        assert "reinforcement learning" in viz["conference_data"]
        assert "NeurIPS" in viz["conference_data"]["transformers"]
        assert "NeurIPS" in viz["conference_data"]["reinforcement learning"]

    def test_extract_multiple_topics_multiple_conferences_merged(self):
        """Test merging topic evolutions across different conferences."""
        tool_results = [
            {
                "name": "get_topic_evolution",
                "raw_result": json.dumps(
                    {
                        "topic": "transformers",
                        "conferences": ["NeurIPS", "ICLR"],
                        "conference_data": {
                            "NeurIPS": {
                                "year_counts": {"2022": 5},
                                "year_relative": {"2022": 2.5},
                            },
                            "ICLR": {
                                "year_counts": {"2022": 3},
                                "year_relative": {"2022": 1.8},
                            },
                        },
                    }
                ),
            },
            {
                "name": "get_topic_evolution",
                "raw_result": json.dumps(
                    {
                        "topic": "reinforcement learning",
                        "conferences": ["NeurIPS", "ICLR"],
                        "conference_data": {
                            "NeurIPS": {
                                "year_counts": {"2022": 12},
                                "year_relative": {"2022": 6.0},
                            },
                            "ICLR": {
                                "year_counts": {"2022": 8},
                                "year_relative": {"2022": 5.0},
                            },
                        },
                    }
                ),
            },
        ]

        visualizations = RAGChat._extract_visualizations(tool_results)
        assert len(visualizations) == 1
        viz = visualizations[0]
        assert viz["type"] == "topic_evolution"
        assert viz["topics"] == ["transformers", "reinforcement learning"]
        assert "NeurIPS" in viz["conference_data"]["transformers"]
        assert "ICLR" in viz["conference_data"]["transformers"]
        assert "NeurIPS" in viz["conference_data"]["reinforcement learning"]
        assert "ICLR" in viz["conference_data"]["reinforcement learning"]


class TestRAGChatIntegration:
    """
    Integration tests requiring a running LM Studio instance.

    These tests verify end-to-end functionality with real LM Studio API.
    """

    @requires_lm_studio
    def test_real_query(self, tmp_path):
        """
        Test real query with actual LM Studio backend using configured model.

        This integration test verifies the complete RAG query workflow with real API.
        """
        from abstracts_explorer.embeddings import EmbeddingsManager

        config = get_config()

        # Create database with test data
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
            },
        )

        # Get the generated UID from the database
        papers = db.query("SELECT uid FROM papers LIMIT 1")
        generated_uid = papers[0]["uid"]

        chroma_path = tmp_path / "chroma_integration"
        set_test_embedding_db(chroma_path)
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

        # Verify response structure
        assert "response" in result
        assert len(result["response"]) > 0
        assert isinstance(result["papers"], list)
        assert isinstance(result["metadata"], dict)

        em.close()
        db.close()

    @requires_lm_studio
    def test_real_conversation(self, tmp_path):
        """
        Test real conversation with actual LM Studio backend.
        """
        from abstracts_explorer.embeddings import EmbeddingsManager

        config = get_config()

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
            },
        )

        papers = db.query("SELECT uid FROM papers LIMIT 1")
        generated_uid = papers[0]["uid"]

        chroma_path = tmp_path / "chroma_conversation"
        set_test_embedding_db(chroma_path)
        em = EmbeddingsManager()
        em.connect()
        em.create_collection(reset=True)

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
        assert len(chat.conversation_history) >= 4

        em.close()
        db.close()

    @requires_lm_studio
    def test_real_export(self, tmp_path):
        """
        Test exporting real conversation with configured model.
        """
        from abstracts_explorer.embeddings import EmbeddingsManager

        config = get_config()

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
            },
        )

        papers = db.query("SELECT uid FROM papers LIMIT 1")
        generated_uid = papers[0]["uid"]

        chroma_path = tmp_path / "chroma_export"
        set_test_embedding_db(chroma_path)
        em = EmbeddingsManager()
        em.connect()
        em.create_collection(reset=True)

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


class TestRAGToolLogging:
    """Test that RAG tool wrappers log tool calls and results."""

    def test_search_papers_logs_call_and_result(self, caplog, mock_search_papers):
        """_tool_search_papers logs the call arguments and abbreviated result."""
        from abstracts_explorer.rag import _tool_search_papers, RAGDeps

        ctx = Mock()
        ctx.deps = RAGDeps()
        ctx.deps.conferences = ["NeurIPS"]

        with caplog.at_level("INFO", logger="abstracts_explorer.rag"):
            _tool_search_papers(ctx, topic_keywords="transformers", n_results=5)

        assert "Tool call: search_papers" in caplog.text
        assert "Tool result: search_papers" in caplog.text

    def test_get_conference_topics_logs_call_and_result(self, caplog, mock_cluster_topics):
        """_tool_get_conference_topics logs the call arguments and abbreviated result."""
        from abstracts_explorer.rag import _tool_get_conference_topics, RAGDeps

        ctx = Mock()
        ctx.deps = RAGDeps()
        ctx.deps.conferences = ["NeurIPS"]

        with caplog.at_level("INFO", logger="abstracts_explorer.rag"):
            _tool_get_conference_topics(ctx, conference="NeurIPS")

        assert "Tool call: get_conference_topics" in caplog.text
        assert "Tool result: get_conference_topics" in caplog.text

    def test_get_topic_evolution_logs_call_and_result(self, caplog, mock_topic_evolution):
        """_tool_get_topic_evolution logs the call arguments and abbreviated result."""
        from abstracts_explorer.rag import _tool_get_topic_evolution, RAGDeps

        ctx = Mock()
        ctx.deps = RAGDeps()

        with caplog.at_level("INFO", logger="abstracts_explorer.rag"):
            _tool_get_topic_evolution(ctx, topic_keywords="attention", conferences=["NeurIPS"])

        assert "Tool call: get_topic_evolution" in caplog.text
        assert "Tool result: get_topic_evolution" in caplog.text

    def test_analyze_topic_relevance_logs_call_and_result(self, caplog, mock_analyze_topic):
        """_tool_analyze_topic_relevance logs the call arguments and abbreviated result."""
        from abstracts_explorer.rag import _tool_analyze_topic_relevance, RAGDeps

        ctx = Mock()
        ctx.deps = RAGDeps()

        with caplog.at_level("INFO", logger="abstracts_explorer.rag"):
            _tool_analyze_topic_relevance(ctx, topic="diffusion models", conference="NeurIPS")

        assert "Tool call: analyze_topic_relevance" in caplog.text
        assert "Tool result: analyze_topic_relevance" in caplog.text

    def test_get_cluster_visualization_logs_call_and_result(self, caplog):
        """_tool_get_cluster_visualization logs the call arguments and abbreviated result."""
        from abstracts_explorer.rag import _tool_get_cluster_visualization, RAGDeps

        with patch("abstracts_explorer.rag.mcp_get_cluster_visualization") as mock_viz:
            mock_viz.return_value = json.dumps({"points": [], "n_points": 0})

            ctx = Mock()
            ctx.deps = RAGDeps()
            ctx.deps.conferences = ["ICLR"]

            with caplog.at_level("INFO", logger="abstracts_explorer.rag"):
                _tool_get_cluster_visualization(ctx, conference="ICLR")

        assert "Tool call: get_cluster_visualization" in caplog.text
        assert "Tool result: get_cluster_visualization" in caplog.text


class TestRAGToolConferenceDefaults:
    """Test that RAG tool wrappers use ctx.deps.conferences and ctx.deps.years as defaults."""

    def test_search_papers_uses_deps_conference_when_not_specified(self, mock_search_papers):
        """_tool_search_papers uses ctx.deps.conferences[0] when no conference provided."""
        from abstracts_explorer.rag import _tool_search_papers, RAGDeps

        ctx = Mock()
        ctx.deps = RAGDeps()
        ctx.deps.conferences = ["NeurIPS"]
        ctx.deps.years = []

        _tool_search_papers(ctx, topic_keywords="transformers")

        call_kwargs = mock_search_papers.call_args[1]
        assert call_kwargs.get("conference") == "NeurIPS"

    def test_search_papers_explicit_conference_overrides_deps(self, mock_search_papers):
        """_tool_search_papers uses the explicitly provided conference over deps."""
        from abstracts_explorer.rag import _tool_search_papers, RAGDeps

        ctx = Mock()
        ctx.deps = RAGDeps()
        ctx.deps.conferences = ["NeurIPS"]
        ctx.deps.years = []

        _tool_search_papers(ctx, topic_keywords="transformers", conference="ICLR")

        call_kwargs = mock_search_papers.call_args[1]
        assert call_kwargs.get("conference") == "ICLR"

    def test_search_papers_uses_deps_years_when_not_specified(self, mock_search_papers):
        """_tool_search_papers uses ctx.deps.years when no years provided."""
        from abstracts_explorer.rag import _tool_search_papers, RAGDeps

        ctx = Mock()
        ctx.deps = RAGDeps()
        ctx.deps.conferences = ["NeurIPS"]
        ctx.deps.years = [2024]

        _tool_search_papers(ctx, topic_keywords="transformers")

        call_kwargs = mock_search_papers.call_args[1]
        assert call_kwargs.get("years") == [2024]

    def test_search_papers_explicit_years_overrides_deps(self, mock_search_papers):
        """_tool_search_papers uses explicitly provided years over deps."""
        from abstracts_explorer.rag import _tool_search_papers, RAGDeps

        ctx = Mock()
        ctx.deps = RAGDeps()
        ctx.deps.conferences = ["NeurIPS"]
        ctx.deps.years = [2024]

        _tool_search_papers(ctx, topic_keywords="transformers", years=[2025])

        call_kwargs = mock_search_papers.call_args[1]
        assert call_kwargs.get("years") == [2025]

    def test_search_papers_no_years_when_deps_empty(self, mock_search_papers):
        """_tool_search_papers does not filter by years when deps.years is empty."""
        from abstracts_explorer.rag import _tool_search_papers, RAGDeps

        ctx = Mock()
        ctx.deps = RAGDeps()
        ctx.deps.conferences = ["NeurIPS"]
        ctx.deps.years = []

        _tool_search_papers(ctx, topic_keywords="transformers")

        call_kwargs = mock_search_papers.call_args[1]
        assert "years" not in call_kwargs

    def test_analyze_topic_relevance_uses_deps_conference_when_not_specified(self, mock_analyze_topic):
        """_tool_analyze_topic_relevance uses ctx.deps.conferences when no conference provided."""
        from abstracts_explorer.rag import _tool_analyze_topic_relevance, RAGDeps

        ctx = Mock()
        ctx.deps = RAGDeps()
        ctx.deps.conferences = ["ICLR"]
        ctx.deps.years = []

        _tool_analyze_topic_relevance(ctx, topic="diffusion models")

        call_kwargs = mock_analyze_topic.call_args[1]
        assert call_kwargs.get("conferences") == ["ICLR"]

    def test_analyze_topic_relevance_explicit_conference_overrides_deps(self, mock_analyze_topic):
        """_tool_analyze_topic_relevance uses explicitly provided conference over deps."""
        from abstracts_explorer.rag import _tool_analyze_topic_relevance, RAGDeps

        ctx = Mock()
        ctx.deps = RAGDeps()
        ctx.deps.conferences = ["ICLR"]
        ctx.deps.years = []

        _tool_analyze_topic_relevance(ctx, topic="diffusion models", conference="NeurIPS")

        call_kwargs = mock_analyze_topic.call_args[1]
        assert call_kwargs.get("conferences") == ["NeurIPS"]

    def test_analyze_topic_relevance_uses_deps_years_when_not_specified(self, mock_analyze_topic):
        """_tool_analyze_topic_relevance uses ctx.deps.years when no years provided."""
        from abstracts_explorer.rag import _tool_analyze_topic_relevance, RAGDeps

        ctx = Mock()
        ctx.deps = RAGDeps()
        ctx.deps.conferences = ["NeurIPS"]
        ctx.deps.years = [2024]

        _tool_analyze_topic_relevance(ctx, topic="diffusion models")

        call_kwargs = mock_analyze_topic.call_args[1]
        assert call_kwargs.get("years") == [2024]

    def test_analyze_topic_relevance_explicit_years_overrides_deps(self, mock_analyze_topic):
        """_tool_analyze_topic_relevance uses explicitly provided years over deps."""
        from abstracts_explorer.rag import _tool_analyze_topic_relevance, RAGDeps

        ctx = Mock()
        ctx.deps = RAGDeps()
        ctx.deps.conferences = ["NeurIPS"]
        ctx.deps.years = [2024]

        _tool_analyze_topic_relevance(ctx, topic="diffusion models", years=[2025])

        call_kwargs = mock_analyze_topic.call_args[1]
        assert call_kwargs.get("years") == [2025]

    def test_analyze_topic_relevance_no_years_when_deps_empty(self, mock_analyze_topic):
        """_tool_analyze_topic_relevance does not filter by years when deps.years is empty."""
        from abstracts_explorer.rag import _tool_analyze_topic_relevance, RAGDeps

        ctx = Mock()
        ctx.deps = RAGDeps()
        ctx.deps.conferences = ["NeurIPS"]
        ctx.deps.years = []

        _tool_analyze_topic_relevance(ctx, topic="diffusion models")

        call_kwargs = mock_analyze_topic.call_args[1]
        assert "years" not in call_kwargs


class TestRAGToolGetPaperDetails:
    """Test the _tool_get_paper_details wrapper and its conference/year defaults."""

    @pytest.fixture
    def mock_paper_details(self):
        """Mock mcp_server.get_paper_details to return test data."""
        with patch("abstracts_explorer.rag.mcp_get_paper_details") as mock_gpd:
            mock_gpd.return_value = json.dumps(
                {
                    "papers_found": 1,
                    "papers": [
                        {
                            "uid": "abc123",
                            "title": "Attention Is All You Need",
                            "authors": ["Vaswani et al."],
                            "abstract": "We propose the Transformer...",
                            "year": 2024,
                            "conference": "NeurIPS",
                            "url": "https://example.com/paper",
                        }
                    ],
                },
                indent=2,
            )
            yield mock_gpd

    def test_get_paper_details_uses_deps_conference_when_not_specified(self, mock_paper_details):
        """_tool_get_paper_details uses ctx.deps.conferences[0] when no conference provided."""
        from abstracts_explorer.rag import _tool_get_paper_details, RAGDeps

        ctx = Mock()
        ctx.deps = RAGDeps()
        ctx.deps.conferences = ["NeurIPS"]
        ctx.deps.years = []

        _tool_get_paper_details(ctx, title="Attention")

        call_kwargs = mock_paper_details.call_args[1]
        assert call_kwargs.get("conference") == "NeurIPS"

    def test_get_paper_details_explicit_conference_overrides_deps(self, mock_paper_details):
        """_tool_get_paper_details uses explicitly provided conference over deps."""
        from abstracts_explorer.rag import _tool_get_paper_details, RAGDeps

        ctx = Mock()
        ctx.deps = RAGDeps()
        ctx.deps.conferences = ["NeurIPS"]
        ctx.deps.years = []

        _tool_get_paper_details(ctx, title="Attention", conference="ICLR")

        call_kwargs = mock_paper_details.call_args[1]
        assert call_kwargs.get("conference") == "ICLR"

    def test_get_paper_details_uses_deps_year_when_not_specified(self, mock_paper_details):
        """_tool_get_paper_details uses ctx.deps.years[0] when no year provided."""
        from abstracts_explorer.rag import _tool_get_paper_details, RAGDeps

        ctx = Mock()
        ctx.deps = RAGDeps()
        ctx.deps.conferences = ["NeurIPS"]
        ctx.deps.years = [2024]

        _tool_get_paper_details(ctx, title="Attention")

        call_kwargs = mock_paper_details.call_args[1]
        assert call_kwargs.get("year") == 2024

    def test_get_paper_details_explicit_year_overrides_deps(self, mock_paper_details):
        """_tool_get_paper_details uses explicitly provided year over deps."""
        from abstracts_explorer.rag import _tool_get_paper_details, RAGDeps

        ctx = Mock()
        ctx.deps = RAGDeps()
        ctx.deps.conferences = ["NeurIPS"]
        ctx.deps.years = [2024]

        _tool_get_paper_details(ctx, title="Attention", year=2025)

        call_kwargs = mock_paper_details.call_args[1]
        assert call_kwargs.get("year") == 2025

    def test_get_paper_details_no_year_when_deps_empty(self, mock_paper_details):
        """_tool_get_paper_details does not filter by year when deps.years is empty."""
        from abstracts_explorer.rag import _tool_get_paper_details, RAGDeps

        ctx = Mock()
        ctx.deps = RAGDeps()
        ctx.deps.conferences = ["NeurIPS"]
        ctx.deps.years = []

        _tool_get_paper_details(ctx, title="Attention")

        call_kwargs = mock_paper_details.call_args[1]
        assert "year" not in call_kwargs

    def test_get_paper_details_no_conference_when_deps_empty(self, mock_paper_details):
        """_tool_get_paper_details does not filter by conference when deps.conferences is empty."""
        from abstracts_explorer.rag import _tool_get_paper_details, RAGDeps

        ctx = Mock()
        ctx.deps = RAGDeps()
        ctx.deps.conferences = []
        ctx.deps.years = []

        _tool_get_paper_details(ctx, title="Attention")

        call_kwargs = mock_paper_details.call_args[1]
        assert "conference" not in call_kwargs

    def test_get_paper_details_stores_tool_result(self, mock_paper_details):
        """_tool_get_paper_details appends result to ctx.deps.tool_results."""
        from abstracts_explorer.rag import _tool_get_paper_details, RAGDeps

        ctx = Mock()
        ctx.deps = RAGDeps()
        ctx.deps.conferences = ["NeurIPS"]
        ctx.deps.years = []

        _tool_get_paper_details(ctx, title="Attention")

        assert len(ctx.deps.tool_results) == 1
        assert ctx.deps.tool_results[0]["name"] == "get_paper_details"

    def test_get_paper_details_in_agent_tools(self, mock_embeddings_manager, mock_database):
        """_tool_get_paper_details is registered in the Pydantic AI agent."""
        chat = RAGChat(mock_embeddings_manager, mock_database)
        tool_names = list(chat.agent._function_toolset.tools.keys())
        assert "get_paper_details" in tool_names

    def test_get_paper_details_in_agent_tools_with_mcp_disabled(self, mock_embeddings_manager, mock_database):
        """_tool_get_paper_details is available even when MCP tools are disabled."""
        chat = RAGChat(mock_embeddings_manager, mock_database, enable_mcp_tools=False)
        tool_names = list(chat.agent._function_toolset.tools.keys())
        assert "get_paper_details" in tool_names

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
    mock_db.get_filter_options.return_value = {
        "sessions": [],
        "years": [2025, 2024],
        "conferences": ["NeurIPS"],
    }

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
    """Mock mcp_server.get_cluster_topics to return test data."""
    with patch("abstracts_explorer.rag.mcp_get_cluster_topics") as mock_ct:
        mock_ct.return_value = json.dumps(
            {
                "statistics": {"n_clusters": 8, "total_papers": 100},
                "clusters": [
                    {
                        "cluster_id": 0,
                        "paper_count": 20,
                        "keywords": [{"keyword": "transformers", "count": 15}],
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
            {"topic": "transformers", "year_counts": {"2020": 10, "2021": 15, "2022": 20}, "total_papers": 45}
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
        """Test that get_cluster_topics tool can be called."""

        def model_fn(messages, info):
            """Model that calls get_cluster_topics on first call, then responds."""
            # Check if tool results already exist (tool was already called)
            for msg in messages:
                for part in msg.parts:
                    if getattr(part, "part_kind", None) == "tool-return":
                        return ModelResponse(
                            parts=[TextPart(content="The main topics are transformers and diffusion models.")]
                        )
            # First call: request the tool
            return ModelResponse(parts=[ToolCallPart(tool_name="get_cluster_topics", args={})])

        chat = RAGChat(mock_embeddings_manager, mock_database, enable_mcp_tools=True)

        with chat.agent.override(model=FunctionModel(model_fn)):
            result = chat.query("What are the main research topics?")

        mock_cluster_topics.assert_called_once()
        assert result["metadata"]["used_tools"] is True
        assert "get_cluster_topics" in result["metadata"]["tools_executed"]

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
                        args={"topic_keywords": "transformers", "conference": "neurips"},
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
                    ToolCallPart(tool_name="get_cluster_topics", args={}),
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
        assert "get_cluster_topics" in result["metadata"]["tools_executed"]
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
                "name": "get_cluster_topics",
                "raw_result": json.dumps({"statistics": {}, "clusters": []}),
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
                        "conference": "NeurIPS",
                        "total_papers": 30,
                        "year_counts": {"2022": 5, "2023": 10, "2024": 15},
                    }
                ),
            }
        ]

        visualizations = RAGChat._extract_visualizations(tool_results)
        assert len(visualizations) == 1
        viz = visualizations[0]
        assert viz["type"] == "topic_evolution"
        assert viz["topic"] == "transformers"
        assert viz["conference"] == "NeurIPS"
        assert viz["year_counts"] == {"2022": 5, "2023": 10, "2024": 15}

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
                        "conference": "NeurIPS",
                        "year_counts": {},
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
                "name": "get_cluster_topics",
                "raw_result": json.dumps({"clusters": []}),
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
                        "conference": "ICLR",
                        "year_counts": {"2023": 8},
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

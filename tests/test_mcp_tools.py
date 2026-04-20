"""
Tests for MCP Tools Integration Module.

Tests the integration between MCP clustering tools and RAG chat,
including tool schema generation, execution, and result formatting.
"""

import json
from typing import Any
from unittest.mock import patch, Mock

from abstracts_explorer.mcp_tools import (
    get_mcp_tools_schema,
    execute_mcp_tool,
    format_tool_result_for_llm,
    _abbreviate_result,
    _format_conference_topics_result,
    _format_topic_evolution_result,
    _format_search_papers_result,
    _format_visualization_result,
    _format_paper_details_result,
    _normalize_search_papers_args,
    _normalize_get_topic_evolution_args,
    _normalize_analyze_topic_relevance_args,
    _normalize_get_paper_details_args,
    _filter_unknown_kwargs,
)


def test_get_mcp_tools_schema():
    """Test that MCP tools schema is in correct OpenAI format."""
    schema = get_mcp_tools_schema()

    # Should return a list of tool definitions
    assert isinstance(schema, list)
    assert len(schema) == 6  # 6 MCP tools (including get_paper_details)

    # Check structure of first tool
    tool = schema[0]
    assert tool["type"] == "function"
    assert "function" in tool
    assert "name" in tool["function"]
    assert "description" in tool["function"]
    assert "parameters" in tool["function"]

    # Verify all expected tools are present
    tool_names = [t["function"]["name"] for t in schema]
    assert "analyze_topic_relevance" in tool_names
    assert "get_conference_topics" in tool_names
    assert "get_topic_evolution" in tool_names
    assert "search_papers" in tool_names
    assert "get_cluster_visualization" in tool_names
    assert "get_paper_details" in tool_names


def test_execute_mcp_tool_cluster_topics():
    """Test executing get_conference_topics tool."""
    with patch("abstracts_explorer.mcp_tools.get_conference_topics") as mock_tool:
        mock_result = json.dumps({"n_topics": 5, "total_papers": 100, "topics": []})
        mock_tool.return_value = mock_result

        result = execute_mcp_tool("get_conference_topics", {"conferences": ["NeurIPS"]})

        assert result == mock_result
        mock_tool.assert_called_once_with(conferences=["NeurIPS"])


def test_execute_mcp_tool_topic_evolution():
    """Test executing get_topic_evolution tool."""
    with patch("abstracts_explorer.mcp_tools.get_topic_evolution") as mock_tool:
        mock_result = json.dumps(
            {"topic": "transformers", "conference_data": {"NeurIPS": {"year_counts": {"2023": 10}}}}
        )
        mock_tool.return_value = mock_result

        result = execute_mcp_tool(
            "get_topic_evolution", {"topic_keywords": "transformers", "conferences": ["NeurIPS"]}
        )

        assert result == mock_result
        mock_tool.assert_called_once_with(topic_keywords="transformers", conferences=["NeurIPS"])


def test_execute_mcp_tool_recent_developments():
    """Test executing search_papers tool."""
    with patch("abstracts_explorer.mcp_tools.search_papers") as mock_tool:
        mock_result = json.dumps({"topic": "LLMs", "papers": [{"title": "Paper 1", "year": 2024}]})
        mock_tool.return_value = mock_result

        result = execute_mcp_tool("search_papers", {"topic_keywords": "LLMs", "n_years": 2})

        assert result == mock_result
        mock_tool.assert_called_once_with(topic_keywords="LLMs", n_years=2)


def test_execute_mcp_tool_unknown():
    """Test executing unknown tool raises error."""
    result = execute_mcp_tool("unknown_tool", {})

    # Should return error JSON
    result_data = json.loads(result)
    assert "error" in result_data
    assert "Unknown MCP tool" in result_data["error"]


def test_execute_mcp_tool_exception_handling():
    """Test that tool execution exceptions are caught and returned as JSON."""
    with patch("abstracts_explorer.mcp_tools.get_conference_topics") as mock_tool:
        mock_tool.side_effect = Exception("Database connection failed")

        result = execute_mcp_tool("get_conference_topics", {})

        result_data = json.loads(result)
        assert "error" in result_data
        assert "Tool execution failed" in result_data["error"]


def test_format_conference_topics_result():
    """Test formatting conference topics result for LLM."""
    data = {
        "conference": "NeurIPS",
        "n_topics": 3,
        "total_papers": 150,
        "topics": [
            {
                "topic": "Transformers",
                "paper_count": 60,
                "keywords": ["reinforcement learning", "policy"],
            },
            {
                "topic": "Attention Mechanisms",
                "paper_count": 50,
                "keywords": ["transformer", "attention"],
            },
        ],
    }

    result = _format_conference_topics_result(data)

    assert "Conference Topics for NeurIPS" in result
    assert "150 papers" in result
    assert "3 topics" in result
    assert "Transformers" in result
    assert "60 papers" in result
    assert "reinforcement learning" in result
    assert "Attention Mechanisms" in result
    assert "50 papers" in result


def test_format_topic_evolution_result():
    """Test formatting topic evolution result for LLM."""
    data = {
        "topic": "transformers",
        "conference_data": {
            "NeurIPS": {
                "year_counts": {"2022": 10, "2023": 15, "2024": 20},
                "year_relative": {"2022": 5.0, "2023": 6.0, "2024": 7.5},
            }
        },
        "total_papers": 45,
    }

    result = _format_topic_evolution_result(data)

    assert "Topic Evolution Analysis" in result
    assert "transformers" in result
    assert "NeurIPS" in result
    assert "2022: 10 papers (5.0%)" in result
    assert "2023: 15 papers (6.0%)" in result
    assert "2024: 20 papers (7.5%)" in result
    assert "Total papers found: 45" in result


def test_format_search_papers_result():
    """Test formatting recent developments result for LLM."""
    data = {
        "topic": "LLMs",
        "papers": [
            {
                "title": "GPT-4 Architecture",
                "year": 2024,
                "abstract": "This paper presents the architecture of GPT-4...",
            },
            {"title": "Scaling Laws for LLMs", "year": 2024, "abstract": "We study scaling laws..."},
        ],
    }

    result = _format_search_papers_result(data)

    assert "Search Results" in result
    assert "LLMs" in result
    assert "GPT-4 Architecture (2024)" in result
    assert "Scaling Laws for LLMs (2024)" in result
    assert "This paper presents" in result


def test_format_visualization_result():
    """Test formatting visualization result for LLM."""
    data = {
        "n_dimensions": 2,
        "n_points": 500,
        "statistics": {"n_clusters": 8},
        "visualization_saved": True,
        "output_path": "/tmp/clusters.json",
    }

    result = _format_visualization_result(data)

    assert "Cluster Visualization Data Generated" in result
    assert "2D visualization" in result
    assert "500 points" in result
    assert "Clusters: 8" in result
    assert "Saved to: /tmp/clusters.json" in result


def test_format_tool_result_with_error():
    """Test formatting tool result when error is present."""
    error_result = json.dumps({"error": "Database not found"})

    result = format_tool_result_for_llm("get_conference_topics", error_result)

    assert "Tool execution failed" in result
    assert "Database not found" in result


def test_format_tool_result_invalid_json():
    """Test formatting tool result with invalid JSON."""
    invalid_result = "This is not JSON"

    result = format_tool_result_for_llm("get_conference_topics", invalid_result)

    # Should return the original result
    assert result == invalid_result


def test_format_tool_result_unknown_tool():
    """Test formatting result for unknown tool."""
    result_json = json.dumps({"some": "data"})

    result = format_tool_result_for_llm("unknown_tool", result_json)

    # Should return raw result for unknown tools
    assert result == result_json


def test_mcp_tools_schema_parameters():
    """Test that tool schemas have correct parameter definitions."""
    schema = get_mcp_tools_schema()

    # Check get_conference_topics parameters
    conference_topics = next(t for t in schema if t["function"]["name"] == "get_conference_topics")
    params = conference_topics["function"]["parameters"]["properties"]
    assert "collection_name" in params

    # Check get_topic_evolution parameters
    topic_evolution = next(t for t in schema if t["function"]["name"] == "get_topic_evolution")
    params = topic_evolution["function"]["parameters"]
    assert "topic_keywords" in params["properties"]
    assert "conferences" in params["properties"]
    assert "topic_keywords" in params["required"]
    assert "conferences" in params["required"]

    # Check search_papers parameters
    recent_dev = next(t for t in schema if t["function"]["name"] == "search_papers")
    params = recent_dev["function"]["parameters"]
    assert "topic_keywords" in params["properties"]
    assert "years" in params["properties"]
    assert "topic_keywords" in params["required"]

    # Check get_cluster_visualization parameters (simplified)
    viz = next(t for t in schema if t["function"]["name"] == "get_cluster_visualization")
    params = viz["function"]["parameters"]["properties"]
    assert "output_path" in params
    assert "collection_name" in params


def test_get_mcp_tools_schema_with_conferences_enum():
    """Test that passing conferences injects enum into the schema."""
    conferences = ["NeurIPS", "ICLR", "ICML"]
    schema = get_mcp_tools_schema(conferences=conferences)

    # analyze_topic_relevance has an array 'conferences' field
    atr = next(t for t in schema if t["function"]["name"] == "analyze_topic_relevance")
    items = atr["function"]["parameters"]["properties"]["conferences"]["items"]
    assert items["enum"] == conferences

    # search_papers has a single-string 'conference' field
    sp = next(t for t in schema if t["function"]["name"] == "search_papers")
    assert sp["function"]["parameters"]["properties"]["conference"]["enum"] == conferences

    # get_topic_evolution now has an array 'conferences' field
    te = next(t for t in schema if t["function"]["name"] == "get_topic_evolution")
    items = te["function"]["parameters"]["properties"]["conferences"]["items"]
    assert items["enum"] == conferences


def test_get_mcp_tools_schema_with_years_enum():
    """Test that passing years injects enum into the schema."""
    years = [2023, 2024, 2025]
    schema = get_mcp_tools_schema(years=years)

    # analyze_topic_relevance has array 'years' field
    atr = next(t for t in schema if t["function"]["name"] == "analyze_topic_relevance")
    items = atr["function"]["parameters"]["properties"]["years"]["items"]
    assert items["enum"] == years

    # search_papers has array 'years' field
    sp = next(t for t in schema if t["function"]["name"] == "search_papers")
    items = sp["function"]["parameters"]["properties"]["years"]["items"]
    assert items["enum"] == years

    # get_topic_evolution has start_year/end_year integer fields
    te = next(t for t in schema if t["function"]["name"] == "get_topic_evolution")
    props = te["function"]["parameters"]["properties"]
    assert props["start_year"]["enum"] == years
    assert props["end_year"]["enum"] == years


def test_get_mcp_tools_schema_without_args_has_no_enum():
    """Test that calling without arguments produces no enum constraints."""
    schema = get_mcp_tools_schema()

    atr = next(t for t in schema if t["function"]["name"] == "analyze_topic_relevance")
    items = atr["function"]["parameters"]["properties"]["conferences"]["items"]
    assert "enum" not in items

    sp = next(t for t in schema if t["function"]["name"] == "search_papers")
    assert "enum" not in sp["function"]["parameters"]["properties"]["conference"]


def test_get_mcp_tools_schema_does_not_mutate_base():
    """Test that dynamic schema generation does not mutate the base schema."""
    # Call once with enums to verify it doesn't leak into a subsequent call
    get_mcp_tools_schema(conferences=["NeurIPS"])
    schema2 = get_mcp_tools_schema()

    # schema2 should not have the enum from schema1
    atr = next(t for t in schema2 if t["function"]["name"] == "analyze_topic_relevance")
    items = atr["function"]["parameters"]["properties"]["conferences"]["items"]
    assert "enum" not in items


# ---------------------------------------------------------------------------
# Argument normalization unit tests
# ---------------------------------------------------------------------------


class TestNormalizeSearchPapersArgs:
    """Tests for _normalize_search_papers_args()."""

    def test_year_singular_int(self):
        """'year' as int is converted to 'years' list."""
        result = _normalize_search_papers_args({"topic_keywords": "RL", "year": 2025})
        assert result["years"] == [2025]
        assert "year" not in result

    def test_year_singular_list(self):
        """'year' as list is renamed to 'years'."""
        result = _normalize_search_papers_args({"topic_keywords": "RL", "year": [2024, 2025]})
        assert result["years"] == [2024, 2025]
        assert "year" not in result

    def test_year_and_years_coexist_drops_year(self):
        """When both 'year' and 'years' are present, 'year' is dropped."""
        result = _normalize_search_papers_args({"topic_keywords": "RL", "year": 2023, "years": [2024]})
        assert result["years"] == [2024]
        assert "year" not in result

    def test_topic_keywords_list_joined(self):
        """topic_keywords as list is joined into a space-separated string."""
        result = _normalize_search_papers_args(
            {"topic_keywords": ["reinforcement learning", "autoregressive", "image editing"]}
        )
        assert result["topic_keywords"] == "reinforcement learning autoregressive image editing"

    def test_conference_list_first_element(self):
        """conference as list uses the first element."""
        result = _normalize_search_papers_args({"topic_keywords": "RL", "conference": ["NeurIPS", "ICLR"]})
        assert result["conference"] == "NeurIPS"

    def test_conference_empty_list_becomes_none(self):
        """conference as empty list becomes None."""
        result = _normalize_search_papers_args({"topic_keywords": "RL", "conference": []})
        assert result["conference"] is None

    def test_conferences_wrong_field_name(self):
        """'conferences' (wrong plural field) is mapped to 'conference'."""
        result = _normalize_search_papers_args({"topic_keywords": "RL", "conferences": ["NeurIPS"]})
        assert result["conference"] == "NeurIPS"
        assert "conferences" not in result

    def test_valid_args_unchanged(self):
        """Already-valid args are passed through unchanged."""
        args = {"topic_keywords": "deep learning", "years": [2024, 2025], "n_results": 5}
        result = _normalize_search_papers_args(args)
        assert result == args

    def test_full_llm_malformed_args(self):
        """Full example matching the reported error case."""
        raw = {
            "topic_keywords": ["reinforcement learning", "autoregressive", "image editing"],
            "conference": ["NeurIPS"],
            "year": [2025],
            "n_results": 2,
        }
        result = _normalize_search_papers_args(raw)
        assert result["topic_keywords"] == "reinforcement learning autoregressive image editing"
        assert result["conference"] == "NeurIPS"
        assert result["years"] == [2025]
        assert result["n_results"] == 2
        assert "year" not in result


class TestNormalizeGetTopicEvolutionArgs:
    """Tests for _normalize_get_topic_evolution_args()."""

    def test_topic_keywords_list_joined(self):
        """topic_keywords list is joined into a string."""
        result = _normalize_get_topic_evolution_args({"topic_keywords": ["transformers", "attention"]})
        assert result["topic_keywords"] == "transformers attention"

    def test_start_end_year_list(self):
        """start_year and end_year as list use first element."""
        result = _normalize_get_topic_evolution_args(
            {"topic_keywords": "RL", "start_year": [2022], "end_year": [2025]}
        )
        assert result["start_year"] == 2022
        assert result["end_year"] == 2025

    def test_valid_args_unchanged(self):
        """Already-valid args pass through unchanged."""
        args = {"topic_keywords": "transformers", "conferences": ["NeurIPS"], "start_year": 2022, "end_year": 2025}
        assert _normalize_get_topic_evolution_args(args) == args


class TestNormalizeAnalyzeTopicRelevanceArgs:
    """Tests for _normalize_analyze_topic_relevance_args()."""

    def test_topic_list_joined(self):
        """topic as list is joined into a string."""
        result = _normalize_analyze_topic_relevance_args({"topic": ["deep learning", "transformers"]})
        assert result["topic"] == "deep learning transformers"

    def test_conference_to_conferences(self):
        """conference (singular) is renamed to conferences (list)."""
        result = _normalize_analyze_topic_relevance_args({"topic": "RL", "conference": "NeurIPS"})
        assert result["conferences"] == ["NeurIPS"]
        assert "conference" not in result

    def test_conference_list_to_conferences(self):
        """conference as list becomes conferences."""
        result = _normalize_analyze_topic_relevance_args({"topic": "RL", "conference": ["NeurIPS", "ICLR"]})
        assert result["conferences"] == ["NeurIPS", "ICLR"]
        assert "conference" not in result

    def test_valid_args_unchanged(self):
        """Already-valid args pass through unchanged."""
        args = {"topic": "deep learning", "conferences": ["NeurIPS"], "years": [2024]}
        assert _normalize_analyze_topic_relevance_args(args) == args


# ---------------------------------------------------------------------------
# _filter_unknown_kwargs tests
# ---------------------------------------------------------------------------


class TestFilterUnknownKwargs:
    """Tests for _filter_unknown_kwargs()."""

    def test_valid_kwargs_pass_through(self):
        """All valid kwargs are retained unchanged."""

        def func(a: int, b: str = "hello") -> str:
            return f"{a} {b}"

        result = _filter_unknown_kwargs(func, {"a": 1, "b": "world"})
        assert result == {"a": 1, "b": "world"}

    def test_unknown_key_is_dropped(self, caplog):
        """Unknown kwargs are dropped with a warning."""

        def func(a: int) -> int:
            return a

        with caplog.at_level("WARNING", logger="abstracts_explorer.mcp_tools"):
            result = _filter_unknown_kwargs(func, {"a": 1, "unknown_key": "ignored"})

        assert result == {"a": 1}
        assert "unknown_key" in caplog.text

    def test_all_unknown_drops_all(self, caplog):
        """When all kwargs are unknown, result is empty dict with warnings."""

        def func() -> None:
            pass

        with caplog.at_level("WARNING", logger="abstracts_explorer.mcp_tools"):
            result = _filter_unknown_kwargs(func, {"x": 1, "y": 2})

        assert result == {}
        assert "x" in caplog.text
        assert "y" in caplog.text

    def test_function_with_var_keyword_passes_all(self):
        """Functions that accept **kwargs receive all arguments."""

        def func(a: int, **kwargs: Any) -> None:
            pass

        args = {"a": 1, "extra": "allowed"}
        result = _filter_unknown_kwargs(func, args)
        assert result == args

    def test_real_search_papers_unknown_key_dropped(self, caplog):
        """Unknown key 'n_years' is silently dropped for search_papers."""
        from abstracts_explorer.mcp_server import search_papers

        with caplog.at_level("WARNING", logger="abstracts_explorer.mcp_tools"):
            result = _filter_unknown_kwargs(search_papers, {"topic_keywords": "RL", "n_years": 5, "n_results": 3})

        assert "topic_keywords" in result
        assert "n_results" in result
        assert "n_years" not in result
        assert "n_years" in caplog.text


# ---------------------------------------------------------------------------
# End-to-end tests for execute_mcp_tool() — each MCP tool exercised through
# the full dispatch path with mocked backends.
# ---------------------------------------------------------------------------


class TestExecuteMCPToolE2E:
    """End-to-end tests for execute_mcp_tool() through the full dispatch path."""

    # ------------------------------------------------------------------
    # search_papers
    # ------------------------------------------------------------------

    def test_search_papers_basic(self):
        """search_papers executes and returns valid JSON."""
        mock_result = json.dumps(
            {
                "topic": "deep learning",
                "papers": [{"title": "DL Paper", "year": 2025}],
                "papers_found": 1,
            }
        )
        with patch("abstracts_explorer.mcp_tools.search_papers", return_value=mock_result) as mock_fn:
            result = execute_mcp_tool("search_papers", {"topic_keywords": "deep learning", "n_results": 3})

        assert json.loads(result)["topic"] == "deep learning"
        mock_fn.assert_called_once_with(topic_keywords="deep learning", n_results=3)

    def test_search_papers_year_singular_normalized(self):
        """search_papers normalizes 'year' (singular) to 'years'."""
        mock_result = json.dumps({"topic": "RL", "papers": [], "papers_found": 0})
        with patch("abstracts_explorer.mcp_tools.search_papers", return_value=mock_result) as mock_fn:
            execute_mcp_tool(
                "search_papers",
                {"topic_keywords": "RL", "year": 2025, "conference": "NeurIPS"},
            )

        kwargs = mock_fn.call_args[1]
        assert kwargs["years"] == [2025]
        assert "year" not in kwargs

    def test_search_papers_full_llm_malformed(self):
        """search_papers handles the exact malformed args from reported error."""
        mock_result = json.dumps({"topic": "RL", "papers": [], "papers_found": 0})
        with patch("abstracts_explorer.mcp_tools.search_papers", return_value=mock_result) as mock_fn:
            result = execute_mcp_tool(
                "search_papers",
                {
                    "topic_keywords": ["reinforcement learning", "autoregressive", "image editing"],
                    "conference": ["NeurIPS"],
                    "year": [2025],
                    "n_results": 2,
                },
            )

        # Must not return an error
        result_data = json.loads(result)
        assert "error" not in result_data

        # Verify search_papers was called with normalized args
        mock_fn.assert_called_once()
        kwargs = mock_fn.call_args[1]
        assert kwargs["topic_keywords"] == "reinforcement learning autoregressive image editing"
        assert kwargs["conference"] == "NeurIPS"
        assert kwargs["years"] == [2025]
        assert kwargs["n_results"] == 2

    def test_search_papers_query_alias(self):
        """search_papers maps 'query' to 'topic_keywords' (legacy alias)."""
        mock_result = json.dumps({"topic": "llm", "papers": [], "papers_found": 0})
        with patch("abstracts_explorer.mcp_tools.search_papers", return_value=mock_result) as mock_fn:
            execute_mcp_tool("search_papers", {"query": "llm trends"})

        kwargs = mock_fn.call_args[1]
        assert kwargs["topic_keywords"] == "llm trends"
        assert "query" not in kwargs

    def test_search_papers_unknown_kwarg_does_not_raise(self):
        """search_papers does not raise TypeError for unknown kwargs; result has no error."""
        mock_result = json.dumps({"topic": "RL", "papers": [], "papers_found": 0})
        with patch("abstracts_explorer.mcp_tools.search_papers", return_value=mock_result):
            result = execute_mcp_tool(
                "search_papers",
                {"topic_keywords": "RL", "n_results": 3, "totally_unknown_param": "should be ignored"},
            )

        data = json.loads(result)
        assert "error" not in data

    def test_get_conference_topics_unknown_kwarg_does_not_raise(self):
        """get_conference_topics does not raise TypeError for unknown kwargs."""
        mock_result = json.dumps({"n_topics": 2, "topics": []})
        with patch("abstracts_explorer.mcp_tools.get_conference_topics", return_value=mock_result):
            result = execute_mcp_tool(
                "get_conference_topics",
                {"conferences": ["NeurIPS"], "unknown_extra": "ignored"},
            )

        data = json.loads(result)
        assert "error" not in data

    def test_search_papers_real_execution(self):
        """search_papers executes end-to-end with mocked EmbeddingsManager and DatabaseManager."""
        mock_em = Mock()
        mock_em.search_similar.return_value = {
            "ids": [["p1", "p2"]],
            "metadatas": [
                [
                    {"title": "Paper 1", "year": 2025, "conference": "NeurIPS", "session": "ML"},
                    {"title": "Paper 2", "year": 2025, "conference": "NeurIPS", "session": "DL"},
                ]
            ],
            "distances": [[0.1, 0.3]],
            "documents": [["Abstract 1", "Abstract 2"]],
        }

        mock_db = Mock()

        with (
            patch("abstracts_explorer.mcp_server.EmbeddingsManager", return_value=mock_em),
            patch("abstracts_explorer.mcp_server.DatabaseManager", return_value=mock_db),
            patch("abstracts_explorer.mcp_server.get_config") as mock_cfg,
        ):
            mock_cfg.return_value = Mock(collection_name="papers")
            result = execute_mcp_tool(
                "search_papers", {"topic_keywords": "deep learning", "conference": "NeurIPS", "n_results": 2}
            )

        data = json.loads(result)
        assert "error" not in data
        assert "papers" in data
        assert len(data["papers"]) == 2
        assert data["papers"][0]["title"] == "Paper 1"

    # ------------------------------------------------------------------
    # get_conference_topics
    # ------------------------------------------------------------------

    def test_get_conference_topics_real_execution(self):
        """get_conference_topics executes end-to-end with mocked clustering stack."""
        import numpy as np

        mock_cm = Mock()
        mock_cm.load_embeddings.return_value = 10
        mock_cm.get_cluster_statistics.return_value = {
            "n_clusters": 2,
            "n_noise": 0,
            "cluster_sizes": {0: 3, 1: 2},
            "total_papers": 5,
        }
        mock_cm.paper_ids = ["p1", "p2", "p3", "p4", "p5"]
        mock_cm.cluster_labels = np.array([0, 0, 1, 1, 0])
        mock_cm.cluster_label_names = None
        mock_cm.cluster_keywords = None
        mock_cm.metadatas = [
            {"title": "A", "keywords": ["ml"], "session": "S1", "year": 2025},
            {"title": "B", "keywords": ["dl"], "session": "S1", "year": 2025},
            {"title": "C", "keywords": ["nlp"], "session": "S2", "year": 2025},
            {"title": "D", "keywords": ["nlp"], "session": "S2", "year": 2025},
            {"title": "E", "keywords": ["ml"], "session": "S1", "year": 2025},
        ]
        mock_cm.embeddings_manager = Mock()
        mock_db = Mock()

        # Mock db.get_clustering_cache to return cached results with cluster names
        mock_db.get_clustering_cache.return_value = {
            "points": [
                {"id": "p1", "cluster": 0, "x": 0.0, "y": 0.0},
                {"id": "p2", "cluster": 0, "x": 0.1, "y": 0.1},
                {"id": "p3", "cluster": 1, "x": 1.0, "y": 1.0},
                {"id": "p4", "cluster": 1, "x": 1.1, "y": 1.1},
                {"id": "p5", "cluster": 0, "x": 0.2, "y": 0.2},
            ],
            "statistics": {
                "n_clusters": 2,
                "n_noise": 0,
                "cluster_sizes": {0: 3, 1: 2},
                "total_papers": 5,
            },
            "cluster_labels": {"0": "Machine Learning", "1": "NLP"},
            "cluster_keywords": {"0": ["ml", "deep"], "1": ["nlp", "bert"]},
        }

        with (
            patch("abstracts_explorer.mcp_server.load_clustering_data", return_value=(mock_cm, mock_db)),
            patch("abstracts_explorer.mcp_server.get_config") as mock_cfg,
        ):
            mock_cfg.return_value = Mock(collection_name="papers", embedding_model="test-model")
            result = execute_mcp_tool("get_conference_topics", {"conferences": ["NeurIPS"]})

        data = json.loads(result)
        assert "error" not in data
        assert "topics" in data
        assert data["n_topics"] == 2
        # topic_sizes should use topic names sorted by size descending
        assert list(data["topic_sizes"].keys()) == ["Machine Learning", "NLP"]
        assert data["topic_sizes"]["Machine Learning"] == 3
        assert data["topic_sizes"]["NLP"] == 2
        # Verify topics are returned, sorted by paper_count desc
        assert data["topics"][0]["topic"] == "Machine Learning"
        assert data["topics"][0]["keywords"] == ["ml", "deep"]
        assert data["topics"][1]["topic"] == "NLP"
        assert data["topics"][1]["keywords"] == ["nlp", "bert"]

    # ------------------------------------------------------------------
    # get_topic_evolution
    # ------------------------------------------------------------------

    def test_get_topic_evolution_real_execution(self):
        """get_topic_evolution executes end-to-end with mocked EmbeddingsManager."""
        mock_em = Mock()

        def _find_papers(database, query, distance_threshold, conferences=None, years=None, query_embedding=None):
            year = years[0] if years else None
            if year == 2022:
                return {
                    "count": 1,
                    "papers": [{"title": "Paper 2022", "session": "ML", "distance": 0.1}],
                    "total_considered": 30,
                }
            elif year == 2023:
                return {
                    "count": 1,
                    "papers": [{"title": "Paper 2023", "session": "ML", "distance": 0.2}],
                    "total_considered": 30,
                }
            elif year == 2024:
                return {
                    "count": 1,
                    "papers": [{"title": "Paper 2024", "session": "ML", "distance": 0.3}],
                    "total_considered": 30,
                }
            return {"count": 0, "papers": [], "total_considered": 0}

        mock_em.find_papers_within_distance.side_effect = _find_papers
        mock_db = Mock()
        mock_db.get_years_for_conference.return_value = [2022, 2023, 2024]
        mock_db.get_stats.side_effect = lambda year, conference: {"total_papers": 100}

        with (
            patch("abstracts_explorer.mcp_server.EmbeddingsManager", return_value=mock_em),
            patch("abstracts_explorer.mcp_server.DatabaseManager", return_value=mock_db),
            patch("abstracts_explorer.mcp_server.get_config") as mock_cfg,
        ):
            mock_cfg.return_value = Mock(collection_name="papers")
            result = execute_mcp_tool(
                "get_topic_evolution", {"topic_keywords": "transformers", "conferences": ["NeurIPS"]}
            )

        data = json.loads(result)
        assert "error" not in data
        assert data["topic"] == "transformers"
        assert data["total_papers"] == 3
        assert "conference_data" in data

    # ------------------------------------------------------------------
    # analyze_topic_relevance
    # ------------------------------------------------------------------

    def test_analyze_topic_relevance_real_execution(self):
        """analyze_topic_relevance executes end-to-end with mocked EmbeddingsManager."""
        mock_em = Mock()
        mock_em.find_papers_within_distance.return_value = {
            "papers": [
                {"title": "Paper 1", "year": 2025, "conference": "NeurIPS", "distance": 0.5},
                {"title": "Paper 2", "year": 2025, "conference": "NeurIPS", "distance": 1.0},
            ],
            "total_considered": 100,
        }

        with (
            patch("abstracts_explorer.mcp_server.EmbeddingsManager", return_value=mock_em),
            patch("abstracts_explorer.mcp_server.DatabaseManager", return_value=Mock()),
            patch("abstracts_explorer.mcp_server.get_config") as mock_cfg,
        ):
            mock_cfg.return_value = Mock(collection_name="papers")
            result = execute_mcp_tool(
                "analyze_topic_relevance",
                {"topic": "deep learning", "distance_threshold": 1.1, "conferences": ["NeurIPS"]},
            )

        data = json.loads(result)
        assert "error" not in data
        assert "topic" in data
        assert data["topic"] == "deep learning"
        assert data["total_papers"] == 2

    def test_analyze_topic_relevance_conference_singular_normalized(self):
        """analyze_topic_relevance maps 'conference' (singular) to 'conferences'."""
        mock_em = Mock()
        mock_em.find_papers_within_distance.return_value = {
            "papers": [{"title": "P", "year": 2025, "conference": "NeurIPS", "distance": 0.5}],
            "total_considered": 50,
        }

        with (
            patch("abstracts_explorer.mcp_server.EmbeddingsManager", return_value=mock_em),
            patch("abstracts_explorer.mcp_server.DatabaseManager", return_value=Mock()),
            patch("abstracts_explorer.mcp_server.get_config") as mock_cfg,
        ):
            mock_cfg.return_value = Mock(collection_name="papers")
            # 'conference' (singular) should not crash
            result = execute_mcp_tool(
                "analyze_topic_relevance",
                {"topic": "RL", "conference": "NeurIPS"},
            )

        data = json.loads(result)
        assert "error" not in data

    # ------------------------------------------------------------------
    # get_cluster_visualization
    # ------------------------------------------------------------------

    def test_get_cluster_visualization_real_execution(self):
        """get_cluster_visualization executes end-to-end with mocked perform_clustering."""
        mock_viz_result = {
            "n_dimensions": 2,
            "n_points": 10,
            "statistics": {"n_clusters": 3},
            "visualization_saved": False,
            "points": [],
        }

        with patch(
            "abstracts_explorer.mcp_tools.get_cluster_visualization",
            return_value=json.dumps(mock_viz_result),
        ) as mock_fn:
            result = execute_mcp_tool("get_cluster_visualization", {"n_clusters": 3})

        data = json.loads(result)
        assert "error" not in data
        assert data["statistics"]["n_clusters"] == 3
        mock_fn.assert_called_once_with(n_clusters=3)

    # ------------------------------------------------------------------
    # Unknown tool
    # ------------------------------------------------------------------

    def test_unknown_tool_returns_error_json(self):
        """Unknown tool name returns error JSON without raising."""
        result = execute_mcp_tool("nonexistent_tool", {"x": 1})
        data = json.loads(result)
        assert "error" in data
        assert "Unknown MCP tool" in data["error"]

    # ------------------------------------------------------------------
    # get_paper_details
    # ------------------------------------------------------------------

    def test_get_paper_details_by_title(self):
        """get_paper_details searches by title and returns paper list."""
        mock_result = json.dumps(
            {
                "papers_found": 1,
                "papers": [
                    {
                        "uid": "abc123",
                        "title": "Attention Is All You Need",
                        "authors": ["Vaswani, Ashish", "Shazeer, Noam"],
                        "year": 2017,
                        "conference": "NeurIPS",
                        "url": "https://example.com/paper",
                        "paper_pdf_url": "https://example.com/paper.pdf",
                    }
                ],
            }
        )
        with patch("abstracts_explorer.mcp_tools.get_paper_details", return_value=mock_result) as mock_fn:
            result = execute_mcp_tool("get_paper_details", {"title": "Attention Is All You Need"})

        assert json.loads(result)["papers_found"] == 1
        mock_fn.assert_called_once_with(title="Attention Is All You Need")

    def test_get_paper_details_by_id(self):
        """get_paper_details looks up by paper_id when provided."""
        mock_result = json.dumps({"papers_found": 1, "papers": [{"uid": "abc123", "title": "A Paper"}]})
        with patch("abstracts_explorer.mcp_tools.get_paper_details", return_value=mock_result) as mock_fn:
            result = execute_mcp_tool("get_paper_details", {"paper_id": "abc123"})

        mock_fn.assert_called_once_with(paper_id="abc123")
        assert json.loads(result)["papers_found"] == 1

    def test_get_paper_details_real_execution_by_title(self):
        """get_paper_details executes end-to-end with mocked DatabaseManager."""
        mock_db = Mock()
        mock_db.search_papers.return_value = [
            {
                "uid": "abc123",
                "original_id": "neurips2023/abc",
                "title": "A Test Paper",
                "authors": ["Smith, John", "Doe, Jane"],
                "abstract": "This is the abstract.",
                "session": "Poster Session 1",
                "poster_position": "P01",
                "paper_pdf_url": "https://example.com/paper.pdf",
                "poster_image_url": None,
                "url": "https://example.com/paper",
                "room_name": "Hall A",
                "keywords": "deep learning, transformers",
                "starttime": "09:00",
                "endtime": "11:00",
                "award": None,
                "year": 2023,
                "conference": "NeurIPS",
                "created_at": "2024-01-01",
            }
        ]

        with (patch("abstracts_explorer.mcp_server.DatabaseManager", return_value=mock_db),):
            result = execute_mcp_tool("get_paper_details", {"title": "Test Paper", "conference": "NeurIPS"})

        data = json.loads(result)
        assert "error" not in data
        assert data["papers_found"] == 1
        paper = data["papers"][0]
        assert paper["title"] == "A Test Paper"
        assert paper["authors"] == ["Smith, John", "Doe, Jane"]
        assert paper["paper_pdf_url"] == "https://example.com/paper.pdf"
        assert paper["keywords"] == "deep learning, transformers"

    def test_get_paper_details_real_execution_by_id(self):
        """get_paper_details executes exact UID/original_id lookup end-to-end."""
        mock_db = Mock()
        mock_db.get_paper_by_original_id_or_uid.return_value = {
            "uid": "abc123",
            "original_id": "neurips2023/abc",
            "title": "Exact Paper",
            "authors": ["Smith, John"],
            "abstract": "Abstract here.",
            "session": "Oral",
            "poster_position": None,
            "paper_pdf_url": "https://example.com/pdf",
            "poster_image_url": None,
            "url": "https://example.com",
            "room_name": "Room 1",
            "keywords": "ml",
            "starttime": None,
            "endtime": None,
            "award": "Best Paper",
            "year": 2023,
            "conference": "NeurIPS",
            "created_at": "2024-01-01",
        }

        with (patch("abstracts_explorer.mcp_server.DatabaseManager", return_value=mock_db),):
            result = execute_mcp_tool("get_paper_details", {"paper_id": "abc123"})

        data = json.loads(result)
        assert "error" not in data
        assert data["papers_found"] == 1
        paper = data["papers"][0]
        assert paper["uid"] == "abc123"
        assert paper["award"] == "Best Paper"
        assert paper["authors"] == ["Smith, John"]

    def test_get_paper_details_no_args_returns_error(self):
        """get_paper_details returns error JSON when neither title nor paper_id is given."""
        result = execute_mcp_tool("get_paper_details", {})
        data = json.loads(result)
        assert "error" in data
        assert "title" in data["error"] or "paper_id" in data["error"]

    def test_get_paper_details_year_normalized_from_string(self):
        """get_paper_details normalizes year from string to int."""
        mock_result = json.dumps({"papers_found": 0, "papers": []})
        with patch("abstracts_explorer.mcp_tools.get_paper_details", return_value=mock_result) as mock_fn:
            execute_mcp_tool("get_paper_details", {"title": "Paper", "year": "2023"})

        kwargs = mock_fn.call_args[1]
        assert kwargs["year"] == 2023
        assert isinstance(kwargs["year"], int)

    def test_get_paper_details_year_normalized_from_list(self):
        """get_paper_details normalizes year from list to int (first element)."""
        mock_result = json.dumps({"papers_found": 0, "papers": []})
        with patch("abstracts_explorer.mcp_tools.get_paper_details", return_value=mock_result) as mock_fn:
            execute_mcp_tool("get_paper_details", {"title": "Paper", "year": [2023, 2024]})

        kwargs = mock_fn.call_args[1]
        assert kwargs["year"] == 2023


# ---------------------------------------------------------------------------
# _normalize_get_paper_details_args unit tests
# ---------------------------------------------------------------------------


class TestNormalizeGetPaperDetailsArgs:
    """Tests for _normalize_get_paper_details_args()."""

    def test_year_as_int_unchanged(self):
        """year as int passes through unchanged."""
        args = {"title": "Paper", "year": 2023}
        result = _normalize_get_paper_details_args(args)
        assert result["year"] == 2023
        assert isinstance(result["year"], int)

    def test_year_as_string_converted_to_int(self):
        """year as string is converted to int."""
        result = _normalize_get_paper_details_args({"title": "Paper", "year": "2023"})
        assert result["year"] == 2023

    def test_year_as_list_uses_first_element(self):
        """year as list uses first element."""
        result = _normalize_get_paper_details_args({"title": "Paper", "year": [2023, 2024]})
        assert result["year"] == 2023

    def test_year_as_empty_list_becomes_none(self):
        """year as empty list becomes None."""
        result = _normalize_get_paper_details_args({"title": "Paper", "year": []})
        assert result["year"] is None

    def test_year_as_invalid_string_becomes_none(self):
        """year as non-numeric string becomes None."""
        result = _normalize_get_paper_details_args({"title": "Paper", "year": "not-a-year"})
        assert result["year"] is None

    def test_conference_as_list_uses_first_element(self):
        """conference as list uses first element."""
        result = _normalize_get_paper_details_args({"title": "Paper", "conference": ["NeurIPS", "ICLR"]})
        assert result["conference"] == "NeurIPS"

    def test_conference_as_empty_list_becomes_none(self):
        """conference as empty list becomes None."""
        result = _normalize_get_paper_details_args({"title": "Paper", "conference": []})
        assert result["conference"] is None

    def test_valid_args_unchanged(self):
        """Already-valid args pass through unchanged."""
        args = {"title": "My Paper", "year": 2024, "conference": "ICLR", "limit": 3}
        result = _normalize_get_paper_details_args(args)
        assert result == args


# ---------------------------------------------------------------------------
# _format_paper_details_result unit tests
# ---------------------------------------------------------------------------


class TestFormatPaperDetailsResult:
    """Tests for _format_paper_details_result()."""

    def test_no_papers(self):
        """Empty papers list shows appropriate message."""
        data = {"papers_found": 0, "papers": []}
        result = _format_paper_details_result(data)
        assert "Paper Details" in result
        assert "No papers found" in result

    def test_single_paper_basic_fields(self):
        """Single paper shows title, year, conference, and authors."""
        data = {
            "papers_found": 1,
            "papers": [
                {
                    "title": "Attention Is All You Need",
                    "year": 2017,
                    "conference": "NeurIPS",
                    "authors": ["Vaswani, Ashish", "Shazeer, Noam"],
                    "url": "",
                    "paper_pdf_url": "",
                    "session": "",
                    "room_name": "",
                    "keywords": "",
                    "award": "",
                    "abstract": "We propose a new model architecture.",
                }
            ],
        }
        result = _format_paper_details_result(data)
        assert "Attention Is All You Need" in result
        assert "2017" in result
        assert "NeurIPS" in result
        assert "Vaswani, Ashish" in result
        assert "Shazeer, Noam" in result
        assert "We propose" in result

    def test_paper_with_url_and_pdf(self):
        """URL and PDF fields are shown when present."""
        data = {
            "papers_found": 1,
            "papers": [
                {
                    "title": "My Paper",
                    "year": 2024,
                    "conference": "ICLR",
                    "authors": ["Author One"],
                    "url": "https://example.com/paper",
                    "paper_pdf_url": "https://example.com/paper.pdf",
                    "session": "",
                    "room_name": "",
                    "keywords": "",
                    "award": "",
                    "abstract": "",
                }
            ],
        }
        result = _format_paper_details_result(data)
        assert "https://example.com/paper" in result
        assert "https://example.com/paper.pdf" in result

    def test_paper_with_session_and_room(self):
        """Session and room information are formatted together."""
        data = {
            "papers_found": 1,
            "papers": [
                {
                    "title": "Paper",
                    "year": 2024,
                    "conference": "NeurIPS",
                    "authors": [],
                    "url": "",
                    "paper_pdf_url": "",
                    "session": "Poster Session 1",
                    "room_name": "Hall B",
                    "keywords": "",
                    "award": "",
                    "abstract": "",
                }
            ],
        }
        result = _format_paper_details_result(data)
        assert "Poster Session 1" in result
        assert "Hall B" in result

    def test_paper_with_award(self):
        """Award field is shown when present."""
        data = {
            "papers_found": 1,
            "papers": [
                {
                    "title": "Award-Winning Paper",
                    "year": 2024,
                    "conference": "NeurIPS",
                    "authors": ["Winner"],
                    "url": "",
                    "paper_pdf_url": "",
                    "session": "",
                    "room_name": "",
                    "keywords": "",
                    "award": "Outstanding Paper Award",
                    "abstract": "",
                }
            ],
        }
        result = _format_paper_details_result(data)
        assert "Outstanding Paper Award" in result

    def test_long_abstract_is_truncated(self):
        """Abstract longer than 200 characters is truncated."""
        long_abstract = "A" * 300
        data = {
            "papers_found": 1,
            "papers": [
                {
                    "title": "Paper",
                    "year": 2024,
                    "conference": "NeurIPS",
                    "authors": [],
                    "url": "",
                    "paper_pdf_url": "",
                    "session": "",
                    "room_name": "",
                    "keywords": "",
                    "award": "",
                    "abstract": long_abstract,
                }
            ],
        }
        result = _format_paper_details_result(data)
        assert "..." in result
        # Should not contain the full abstract
        assert long_abstract not in result

    def test_multiple_papers(self):
        """Multiple papers are all listed with numbering."""
        data = {
            "papers_found": 2,
            "papers": [
                {
                    "title": "First Paper",
                    "year": 2023,
                    "conference": "NeurIPS",
                    "authors": ["Author A"],
                    "url": "",
                    "paper_pdf_url": "",
                    "session": "",
                    "room_name": "",
                    "keywords": "",
                    "award": "",
                    "abstract": "",
                },
                {
                    "title": "Second Paper",
                    "year": 2024,
                    "conference": "ICLR",
                    "authors": ["Author B"],
                    "url": "",
                    "paper_pdf_url": "",
                    "session": "",
                    "room_name": "",
                    "keywords": "",
                    "award": "",
                    "abstract": "",
                },
            ],
        }
        result = _format_paper_details_result(data)
        assert "1. First Paper" in result
        assert "2. Second Paper" in result


# ---------------------------------------------------------------------------
# format_tool_result_for_llm – get_paper_details
# ---------------------------------------------------------------------------


def test_format_tool_result_for_llm_paper_details():
    """format_tool_result_for_llm dispatches to _format_paper_details_result."""
    data = {
        "papers_found": 1,
        "papers": [
            {
                "title": "My Paper",
                "year": 2024,
                "conference": "NeurIPS",
                "authors": ["Alice"],
                "url": "",
                "paper_pdf_url": "https://example.com/pdf",
                "session": "",
                "room_name": "",
                "keywords": "ml",
                "award": "",
                "abstract": "Some abstract text.",
            }
        ],
    }
    result = format_tool_result_for_llm("get_paper_details", json.dumps(data))
    assert "Paper Details" in result
    assert "My Paper" in result
    assert "Alice" in result


# ---------------------------------------------------------------------------
# get_mcp_tools_schema – get_paper_details entries
# ---------------------------------------------------------------------------


def test_get_mcp_tools_schema_paper_details_parameters():
    """get_paper_details schema has expected parameter definitions."""
    schema = get_mcp_tools_schema()
    pd_tool = next(t for t in schema if t["function"]["name"] == "get_paper_details")
    props = pd_tool["function"]["parameters"]["properties"]

    assert "title" in props
    assert props["title"]["type"] == "string"
    assert "paper_id" in props
    assert props["paper_id"]["type"] == "string"
    assert "conference" in props
    assert props["conference"]["type"] == "string"
    assert "year" in props
    assert props["year"]["type"] == "integer"
    assert "limit" in props


def test_get_mcp_tools_schema_paper_details_conference_enum():
    """Conference enum is injected into get_paper_details conference field."""
    conferences = ["NeurIPS", "ICLR"]
    schema = get_mcp_tools_schema(conferences=conferences)
    pd_tool = next(t for t in schema if t["function"]["name"] == "get_paper_details")
    assert pd_tool["function"]["parameters"]["properties"]["conference"]["enum"] == conferences


def test_get_mcp_tools_schema_paper_details_year_enum():
    """Year enum is injected into get_paper_details year field."""
    years = [2023, 2024, 2025]
    schema = get_mcp_tools_schema(years=years)
    pd_tool = next(t for t in schema if t["function"]["name"] == "get_paper_details")
    assert pd_tool["function"]["parameters"]["properties"]["year"]["enum"] == years


# ---------------------------------------------------------------------------
# _abbreviate_result tests
# ---------------------------------------------------------------------------


class TestAbbreviateResult:
    """Tests for the _abbreviate_result helper function."""

    def test_short_text_unchanged(self):
        """Short text is returned unchanged."""
        assert _abbreviate_result("short") == "short"

    def test_exact_max_length_unchanged(self):
        """Text exactly at max_length is returned unchanged."""
        text = "a" * 200
        assert _abbreviate_result(text) == text

    def test_long_text_truncated(self):
        """Text exceeding max_length is truncated with ellipsis."""
        text = "a" * 300
        result = _abbreviate_result(text)
        assert len(result) == 201  # 200 chars + '…'
        assert result.endswith("…")

    def test_custom_max_length(self):
        """Custom max_length is respected."""
        text = "a" * 50
        result = _abbreviate_result(text, max_length=10)
        assert len(result) == 11  # 10 chars + '…'

    def test_empty_string(self):
        """Empty string is returned unchanged."""
        assert _abbreviate_result("") == ""


# ---------------------------------------------------------------------------
# Logging tests for execute_mcp_tool
# ---------------------------------------------------------------------------


class TestExecuteMCPToolLogging:
    """Tests that execute_mcp_tool logs tool calls and return values."""

    def test_logs_tool_call_and_result(self, caplog):
        """execute_mcp_tool logs both the call and the return value."""
        mock_result = json.dumps({"topics": []})
        with patch("abstracts_explorer.mcp_tools.get_conference_topics", return_value=mock_result):
            with caplog.at_level("INFO", logger="abstracts_explorer.mcp_tools"):
                execute_mcp_tool("get_conference_topics", {"conferences": ["NeurIPS"]})

        assert "Executing MCP tool: get_conference_topics" in caplog.text
        assert "MCP tool get_conference_topics returned:" in caplog.text

    def test_logs_abbreviated_long_result(self, caplog):
        """Long return values are abbreviated in the log."""
        mock_result = json.dumps({"papers": [{"title": f"Paper {i}"} for i in range(100)]})
        with patch("abstracts_explorer.mcp_tools.search_papers", return_value=mock_result):
            with caplog.at_level("INFO", logger="abstracts_explorer.mcp_tools"):
                execute_mcp_tool("search_papers", {"topic_keywords": "test"})

        # The result log should contain the abbreviated marker
        log_lines = [r.message for r in caplog.records if "returned:" in r.message]
        assert len(log_lines) == 1
        # Long result should be truncated
        assert len(log_lines[0]) < len(mock_result) + 100

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
    _format_cluster_topics_result,
    _format_topic_evolution_result,
    _format_search_papers_result,
    _format_visualization_result,
    _normalize_search_papers_args,
    _normalize_get_topic_evolution_args,
    _normalize_analyze_topic_relevance_args,
    _filter_unknown_kwargs,
)


def test_get_mcp_tools_schema():
    """Test that MCP tools schema is in correct OpenAI format."""
    schema = get_mcp_tools_schema()

    # Should return a list of tool definitions
    assert isinstance(schema, list)
    assert len(schema) == 5  # 5 MCP tools (including analyze_topic_relevance)

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
    assert "get_cluster_topics" in tool_names
    assert "get_topic_evolution" in tool_names
    assert "search_papers" in tool_names
    assert "get_cluster_visualization" in tool_names


def test_execute_mcp_tool_cluster_topics():
    """Test executing get_cluster_topics tool."""
    with patch("abstracts_explorer.mcp_tools.get_cluster_topics") as mock_tool:
        mock_result = json.dumps({"statistics": {"n_clusters": 5, "total_papers": 100}, "clusters": []})
        mock_tool.return_value = mock_result

        result = execute_mcp_tool("get_cluster_topics", {"n_clusters": 5})

        assert result == mock_result
        mock_tool.assert_called_once_with(n_clusters=5)


def test_execute_mcp_tool_topic_evolution():
    """Test executing get_topic_evolution tool."""
    with patch("abstracts_explorer.mcp_tools.get_topic_evolution") as mock_tool:
        mock_result = json.dumps({"topic": "transformers", "year_counts": {"2023": 10, "2024": 15}})
        mock_tool.return_value = mock_result

        result = execute_mcp_tool("get_topic_evolution", {"topic_keywords": "transformers", "conference": "neurips"})

        assert result == mock_result
        mock_tool.assert_called_once_with(topic_keywords="transformers", conference="neurips")


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
    with patch("abstracts_explorer.mcp_tools.get_cluster_topics") as mock_tool:
        mock_tool.side_effect = Exception("Database connection failed")

        result = execute_mcp_tool("get_cluster_topics", {})

        result_data = json.loads(result)
        assert "error" in result_data
        assert "Tool execution failed" in result_data["error"]


def test_format_cluster_topics_result():
    """Test formatting cluster topics result for LLM."""
    data = {
        "statistics": {"n_clusters": 3, "total_papers": 150},
        "clusters": [
            {
                "cluster_id": 0,
                "paper_count": 50,
                "keywords": [{"keyword": "transformer", "count": 30}, {"keyword": "attention", "count": 25}],
            },
            {"cluster_id": 1, "paper_count": 60, "keywords": [{"keyword": "reinforcement learning", "count": 40}]},
        ],
    }

    result = _format_cluster_topics_result(data)

    assert "Cluster Analysis Results" in result
    assert "3 clusters" in result
    assert "150 papers" in result
    assert "Cluster 0" in result
    assert "50 papers" in result
    assert "transformer (30)" in result
    assert "attention (25)" in result


def test_format_topic_evolution_result():
    """Test formatting topic evolution result for LLM."""
    data = {"topic": "transformers", "year_counts": {"2022": 10, "2023": 15, "2024": 20}, "total_papers": 45}

    result = _format_topic_evolution_result(data)

    assert "Topic Evolution Analysis" in result
    assert "transformers" in result
    assert "2022: 10 papers" in result
    assert "2023: 15 papers" in result
    assert "2024: 20 papers" in result
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

    result = format_tool_result_for_llm("get_cluster_topics", error_result)

    assert "Tool execution failed" in result
    assert "Database not found" in result


def test_format_tool_result_invalid_json():
    """Test formatting tool result with invalid JSON."""
    invalid_result = "This is not JSON"

    result = format_tool_result_for_llm("get_cluster_topics", invalid_result)

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

    # Check get_cluster_topics parameters
    cluster_topics = next(t for t in schema if t["function"]["name"] == "get_cluster_topics")
    params = cluster_topics["function"]["parameters"]["properties"]
    assert "n_clusters" in params
    assert params["n_clusters"]["type"] == "integer"
    assert "clustering_method" in params
    assert params["clustering_method"]["type"] == "string"

    # Check get_topic_evolution parameters
    topic_evolution = next(t for t in schema if t["function"]["name"] == "get_topic_evolution")
    params = topic_evolution["function"]["parameters"]
    assert "topic_keywords" in params["properties"]
    assert "topic_keywords" in params["required"]

    # Check search_papers parameters
    recent_dev = next(t for t in schema if t["function"]["name"] == "search_papers")
    params = recent_dev["function"]["parameters"]
    assert "topic_keywords" in params["properties"]
    assert "years" in params["properties"]
    assert "topic_keywords" in params["required"]


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

    # get_topic_evolution has a single-string 'conference' field
    te = next(t for t in schema if t["function"]["name"] == "get_topic_evolution")
    assert te["function"]["parameters"]["properties"]["conference"]["enum"] == conferences


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

    def test_conference_list_first_element(self):
        """conference list uses first element."""
        result = _normalize_get_topic_evolution_args({"topic_keywords": "RL", "conference": ["NeurIPS", "ICLR"]})
        assert result["conference"] == "NeurIPS"

    def test_start_end_year_list(self):
        """start_year and end_year as list use first element."""
        result = _normalize_get_topic_evolution_args(
            {"topic_keywords": "RL", "start_year": [2022], "end_year": [2025]}
        )
        assert result["start_year"] == 2022
        assert result["end_year"] == 2025

    def test_valid_args_unchanged(self):
        """Already-valid args pass through unchanged."""
        args = {"topic_keywords": "transformers", "start_year": 2022, "end_year": 2025}
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

    def test_get_cluster_topics_unknown_kwarg_does_not_raise(self):
        """get_cluster_topics does not raise TypeError for unknown kwargs."""
        mock_result = json.dumps({"statistics": {"n_clusters": 2}, "clusters": []})
        with patch("abstracts_explorer.mcp_tools.get_cluster_topics", return_value=mock_result):
            result = execute_mcp_tool(
                "get_cluster_topics",
                {"n_clusters": 3, "unknown_extra": "ignored"},
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
            result = execute_mcp_tool("search_papers", {"topic_keywords": "deep learning", "n_results": 2})

        data = json.loads(result)
        assert "error" not in data
        assert "papers" in data
        assert len(data["papers"]) == 2
        assert data["papers"][0]["title"] == "Paper 1"

    # ------------------------------------------------------------------
    # get_cluster_topics
    # ------------------------------------------------------------------

    def test_get_cluster_topics_real_execution(self):
        """get_cluster_topics executes end-to-end with mocked clustering stack."""
        import numpy as np

        mock_cm = Mock()
        mock_cm.load_embeddings.return_value = 10
        mock_cm.cluster.return_value = np.array([0, 0, 1, 1, 0])
        mock_cm.reduce_dimensions.return_value = np.zeros((5, 2))
        mock_cm.get_cluster_statistics.return_value = {
            "n_clusters": 2,
            "n_noise": 0,
            "cluster_sizes": {0: 3, 1: 2},
            "total_papers": 5,
        }
        mock_cm.paper_ids = ["p1", "p2", "p3", "p4", "p5"]
        mock_cm.cluster_labels = np.array([0, 0, 1, 1, 0])
        mock_cm.metadatas = [
            {"title": "A", "keywords": "ml", "session": "S1", "year": 2025},
            {"title": "B", "keywords": "dl", "session": "S1", "year": 2025},
            {"title": "C", "keywords": "nlp", "session": "S2", "year": 2025},
            {"title": "D", "keywords": "nlp", "session": "S2", "year": 2025},
            {"title": "E", "keywords": "ml", "session": "S1", "year": 2025},
        ]
        mock_cm.embeddings_manager = Mock()
        mock_db = Mock()

        with patch("abstracts_explorer.mcp_server.load_clustering_data", return_value=(mock_cm, mock_db)):
            result = execute_mcp_tool("get_cluster_topics", {"n_clusters": 2})

        data = json.loads(result)
        assert "error" not in data
        assert "clusters" in data
        assert data["statistics"]["n_clusters"] == 2

    # ------------------------------------------------------------------
    # get_topic_evolution
    # ------------------------------------------------------------------

    def test_get_topic_evolution_real_execution(self):
        """get_topic_evolution executes end-to-end with mocked EmbeddingsManager."""
        mock_em = Mock()
        mock_em.search_similar.return_value = {
            "ids": [["p1", "p2", "p3"]],
            "metadatas": [
                [
                    {"title": "Paper 2022", "year": 2022, "session": "ML"},
                    {"title": "Paper 2023", "year": 2023, "session": "ML"},
                    {"title": "Paper 2024", "year": 2024, "session": "ML"},
                ]
            ],
            "distances": [[0.1, 0.2, 0.3]],
        }
        mock_db = Mock()

        with (
            patch("abstracts_explorer.mcp_server.EmbeddingsManager", return_value=mock_em),
            patch("abstracts_explorer.mcp_server.DatabaseManager", return_value=mock_db),
            patch("abstracts_explorer.mcp_server.get_config") as mock_cfg,
        ):
            mock_cfg.return_value = Mock(collection_name="papers")
            result = execute_mcp_tool("get_topic_evolution", {"topic_keywords": "transformers"})

        data = json.loads(result)
        assert "error" not in data
        assert data["topic"] == "transformers"
        assert data["total_papers"] == 3
        assert "year_counts" in data

    def test_get_topic_evolution_conference_list_normalized(self):
        """get_topic_evolution normalizes conference as list to string."""
        mock_em = Mock()
        mock_em.search_similar.return_value = {
            "ids": [["p1"]],
            "metadatas": [[{"title": "P", "year": 2025, "session": "S"}]],
            "distances": [[0.1]],
        }
        mock_db = Mock()

        with (
            patch("abstracts_explorer.mcp_server.EmbeddingsManager", return_value=mock_em),
            patch("abstracts_explorer.mcp_server.DatabaseManager", return_value=mock_db),
            patch("abstracts_explorer.mcp_server.get_config") as mock_cfg,
        ):
            mock_cfg.return_value = Mock(collection_name="papers")
            # conference passed as list — should not raise
            result = execute_mcp_tool(
                "get_topic_evolution",
                {"topic_keywords": "RL", "conference": ["NeurIPS"]},
            )

        data = json.loads(result)
        assert "error" not in data

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
            ]
        }

        with (
            patch("abstracts_explorer.mcp_server.EmbeddingsManager", return_value=mock_em),
            patch("abstracts_explorer.mcp_server.DatabaseManager", return_value=Mock()),
            patch("abstracts_explorer.mcp_server.get_config") as mock_cfg,
        ):
            mock_cfg.return_value = Mock(collection_name="papers")
            result = execute_mcp_tool(
                "analyze_topic_relevance",
                {"topic": "deep learning", "distance_threshold": 1.1},
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
            "papers": [{"title": "P", "year": 2025, "conference": "NeurIPS", "distance": 0.5}]
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

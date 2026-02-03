"""
Tests for MCP Tools Integration Module.

Tests the integration between MCP clustering tools and RAG chat,
including tool schema generation, execution, and result formatting.
"""

import json
from unittest.mock import patch

from abstracts_explorer.mcp_tools import (
    get_mcp_tools_schema,
    execute_mcp_tool,
    format_tool_result_for_llm,
    _format_cluster_topics_result,
    _format_topic_evolution_result,
    _format_recent_developments_result,
    _format_visualization_result,
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
    assert "get_recent_developments" in tool_names
    assert "get_cluster_visualization" in tool_names


def test_execute_mcp_tool_cluster_topics():
    """Test executing get_cluster_topics tool."""
    with patch("abstracts_explorer.mcp_tools.get_cluster_topics") as mock_tool:
        mock_result = json.dumps({
            "statistics": {"n_clusters": 5, "total_papers": 100},
            "clusters": []
        })
        mock_tool.return_value = mock_result
        
        result = execute_mcp_tool("get_cluster_topics", {"n_clusters": 5})
        
        assert result == mock_result
        mock_tool.assert_called_once_with(n_clusters=5)


def test_execute_mcp_tool_topic_evolution():
    """Test executing get_topic_evolution tool."""
    with patch("abstracts_explorer.mcp_tools.get_topic_evolution") as mock_tool:
        mock_result = json.dumps({
            "topic": "transformers",
            "year_counts": {"2023": 10, "2024": 15}
        })
        mock_tool.return_value = mock_result
        
        result = execute_mcp_tool(
            "get_topic_evolution",
            {"topic_keywords": "transformers", "conference": "neurips"}
        )
        
        assert result == mock_result
        mock_tool.assert_called_once_with(
            topic_keywords="transformers",
            conference="neurips"
        )


def test_execute_mcp_tool_recent_developments():
    """Test executing get_recent_developments tool."""
    with patch("abstracts_explorer.mcp_tools.get_recent_developments") as mock_tool:
        mock_result = json.dumps({
            "topic": "LLMs",
            "papers": [{"title": "Paper 1", "year": 2024}]
        })
        mock_tool.return_value = mock_result
        
        result = execute_mcp_tool(
            "get_recent_developments",
            {"topic_keywords": "LLMs", "n_years": 2}
        )
        
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
                "keywords": [
                    {"keyword": "transformer", "count": 30},
                    {"keyword": "attention", "count": 25}
                ]
            },
            {
                "cluster_id": 1,
                "paper_count": 60,
                "keywords": [
                    {"keyword": "reinforcement learning", "count": 40}
                ]
            }
        ]
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
    data = {
        "topic": "transformers",
        "year_counts": {
            "2022": 10,
            "2023": 15,
            "2024": 20
        },
        "total_papers": 45
    }
    
    result = _format_topic_evolution_result(data)
    
    assert "Topic Evolution Analysis" in result
    assert "transformers" in result
    assert "2022: 10 papers" in result
    assert "2023: 15 papers" in result
    assert "2024: 20 papers" in result
    assert "Total papers found: 45" in result


def test_format_recent_developments_result():
    """Test formatting recent developments result for LLM."""
    data = {
        "topic": "LLMs",
        "papers": [
            {
                "title": "GPT-4 Architecture",
                "year": 2024,
                "abstract": "This paper presents the architecture of GPT-4..."
            },
            {
                "title": "Scaling Laws for LLMs",
                "year": 2024,
                "abstract": "We study scaling laws..."
            }
        ]
    }
    
    result = _format_recent_developments_result(data)
    
    assert "Recent Developments" in result
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
        "output_path": "/tmp/clusters.json"
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
    
    # Check get_recent_developments parameters
    recent_dev = next(t for t in schema if t["function"]["name"] == "get_recent_developments")
    params = recent_dev["function"]["parameters"]
    assert "topic_keywords" in params["properties"]
    assert "n_years" in params["properties"]
    assert "topic_keywords" in params["required"]

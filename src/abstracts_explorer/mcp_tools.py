"""
MCP Tools Integration for RAG Chat
==================================

This module provides integration between MCP clustering tools and the RAG chat system.
It converts MCP tool definitions to OpenAI function calling format and handles tool execution.

The integration allows the LLM to automatically decide when to use clustering tools
to answer questions about conference topics, trends, and developments.
"""

import json
import logging
from typing import Dict, List, Any

from .mcp_server import (
    get_cluster_topics,
    get_topic_evolution,
    get_recent_developments,
    get_cluster_visualization,
)

logger = logging.getLogger(__name__)


class MCPToolsError(Exception):
    """Exception raised for MCP tools-related errors."""
    pass


# Define MCP tools in OpenAI function calling format
MCP_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_cluster_topics",
            "description": (
                "Analyze clustered paper embeddings to identify the most frequently mentioned topics. "
                "Use this tool when the user asks about: overall themes, main topics, research areas, "
                "or wants to understand what topics are covered in the conference."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "n_clusters": {
                        "type": "integer",
                        "description": "Number of clusters to create (default: 8)",
                        "default": 8
                    },
                    "reduction_method": {
                        "type": "string",
                        "enum": ["pca", "tsne"],
                        "description": "Dimensionality reduction method (default: 'pca')",
                        "default": "pca"
                    },
                    "clustering_method": {
                        "type": "string",
                        "enum": ["kmeans", "dbscan", "agglomerative"],
                        "description": "Clustering algorithm (default: 'kmeans')",
                        "default": "kmeans"
                    },
                    "collection_name": {
                        "type": "string",
                        "description": "Name of ChromaDB collection (optional, uses config default)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_topic_evolution",
            "description": (
                "Analyze how specific topics have evolved over the years. "
                "Use this tool when the user asks about: trends over time, historical development, "
                "how a topic has changed, or evolution of research areas."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic_keywords": {
                        "type": "string",
                        "description": "Keywords describing the topic (e.g., 'transformers attention', 'reinforcement learning')"
                    },
                    "conference": {
                        "type": "string",
                        "description": "Filter by conference name (e.g., 'neurips', 'iclr')"
                    },
                    "start_year": {
                        "type": "integer",
                        "description": "Start year for analysis (inclusive)"
                    },
                    "end_year": {
                        "type": "integer",
                        "description": "End year for analysis (inclusive)"
                    },
                    "where": {
                        "type": "object",
                        "description": "Custom ChromaDB WHERE clause for advanced filtering"
                    },
                    "collection_name": {
                        "type": "string",
                        "description": "Name of ChromaDB collection (optional)"
                    }
                },
                "required": ["topic_keywords"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_developments",
            "description": (
                "Find the most important recent developments in a specific topic. "
                "Use this tool when the user asks about: recent papers, latest research, "
                "current work, or new developments in a specific area."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic_keywords": {
                        "type": "string",
                        "description": "Keywords describing the topic to search for"
                    },
                    "n_years": {
                        "type": "integer",
                        "description": "Number of recent years to consider (default: 2)",
                        "default": 2
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of papers to return (default: 10)",
                        "default": 10
                    },
                    "conference": {
                        "type": "string",
                        "description": "Filter by conference name"
                    },
                    "where": {
                        "type": "object",
                        "description": "Custom ChromaDB WHERE clause for filtering"
                    },
                    "collection_name": {
                        "type": "string",
                        "description": "Name of ChromaDB collection (optional)"
                    }
                },
                "required": ["topic_keywords"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_cluster_visualization",
            "description": (
                "Generate visualization data for clustered embeddings. "
                "Use this tool when the user asks for: a visual representation, graphical view, "
                "or wants to see clusters displayed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "n_clusters": {
                        "type": "integer",
                        "description": "Number of clusters (default: 8)",
                        "default": 8
                    },
                    "reduction_method": {
                        "type": "string",
                        "enum": ["pca", "tsne"],
                        "description": "Reduction method (default: 'tsne')",
                        "default": "tsne"
                    },
                    "clustering_method": {
                        "type": "string",
                        "enum": ["kmeans", "dbscan", "agglomerative"],
                        "description": "Clustering method (default: 'kmeans')",
                        "default": "kmeans"
                    },
                    "n_components": {
                        "type": "integer",
                        "enum": [2, 3],
                        "description": "Number of dimensions: 2 or 3 (default: 2)",
                        "default": 2
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save visualization JSON (optional)"
                    },
                    "collection_name": {
                        "type": "string",
                        "description": "Name of ChromaDB collection (optional)"
                    }
                },
                "required": []
            }
        }
    }
]


def execute_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """
    Execute an MCP tool with the given arguments.

    Parameters
    ----------
    tool_name : str
        Name of the tool to execute
    arguments : dict
        Arguments to pass to the tool

    Returns
    -------
    str
        Tool execution result (JSON string)

    Raises
    ------
    MCPToolsError
        If tool execution fails or tool is unknown
    """
    logger.info(f"Executing MCP tool: {tool_name} with arguments: {arguments}")
    
    try:
        if tool_name == "get_cluster_topics":
            return get_cluster_topics(**arguments)
        elif tool_name == "get_topic_evolution":
            return get_topic_evolution(**arguments)
        elif tool_name == "get_recent_developments":
            return get_recent_developments(**arguments)
        elif tool_name == "get_cluster_visualization":
            return get_cluster_visualization(**arguments)
        else:
            # Return error JSON for unknown tools
            error_result = {"error": f"Unknown MCP tool: {tool_name}"}
            return json.dumps(error_result, indent=2)
    
    except Exception as e:
        logger.error(f"MCP tool execution failed: {str(e)}")
        error_result = {"error": f"Tool execution failed: {str(e)}"}
        return json.dumps(error_result, indent=2)


def get_mcp_tools_schema() -> List[Dict[str, Any]]:
    """
    Get the MCP tools schema for OpenAI function calling.

    Returns
    -------
    list
        List of tool definitions in OpenAI format
    """
    return MCP_TOOLS_SCHEMA


def format_tool_result_for_llm(tool_name: str, result: str) -> str:
    """
    Format tool execution result for LLM consumption.

    This function extracts the most relevant information from tool results
    and formats it in a way that's easy for the LLM to process.

    Parameters
    ----------
    tool_name : str
        Name of the tool that was executed
    result : str
        Raw tool result (JSON string)

    Returns
    -------
    str
        Formatted result suitable for LLM processing
    """
    try:
        result_data = json.loads(result)
        
        # Check for errors
        if "error" in result_data:
            return f"Tool execution failed: {result_data['error']}"
        
        # Format based on tool type
        if tool_name == "get_cluster_topics":
            return _format_cluster_topics_result(result_data)
        elif tool_name == "get_topic_evolution":
            return _format_topic_evolution_result(result_data)
        elif tool_name == "get_recent_developments":
            return _format_recent_developments_result(result_data)
        elif tool_name == "get_cluster_visualization":
            return _format_visualization_result(result_data)
        else:
            # Return raw result for unknown tools
            return result
    
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse tool result as JSON: {result[:100]}...")
        return result


def _format_cluster_topics_result(data: Dict[str, Any]) -> str:
    """Format cluster topics result for LLM."""
    lines = ["Cluster Analysis Results:\n"]
    
    stats = data.get("statistics", {})
    lines.append(f"Found {stats.get('n_clusters', 0)} clusters covering {stats.get('total_papers', 0)} papers.\n")
    
    clusters = data.get("clusters", [])
    for cluster in clusters[:10]:  # Limit to top 10 clusters
        cluster_id = cluster.get("cluster_id")
        paper_count = cluster.get("paper_count", 0)
        keywords = cluster.get("keywords", [])[:5]  # Top 5 keywords
        
        lines.append(f"\nCluster {cluster_id} ({paper_count} papers):")
        if keywords:
            keyword_strs = [f"{kw['keyword']} ({kw['count']})" for kw in keywords]
            lines.append(f"  Top keywords: {', '.join(keyword_strs)}")
    
    return "\n".join(lines)


def _format_topic_evolution_result(data: Dict[str, Any]) -> str:
    """Format topic evolution result for LLM."""
    lines = [f"Topic Evolution Analysis for '{data.get('topic', 'unknown')}':\n"]
    
    year_counts = data.get("year_counts", {})
    if year_counts:
        lines.append("Papers per year:")
        for year, count in sorted(year_counts.items()):
            lines.append(f"  {year}: {count} papers")
    
    total = data.get("total_papers", 0)
    lines.append(f"\nTotal papers found: {total}")
    
    return "\n".join(lines)


def _format_recent_developments_result(data: Dict[str, Any]) -> str:
    """Format recent developments result for LLM."""
    lines = [f"Recent Developments in '{data.get('topic', 'unknown')}':\n"]
    
    papers = data.get("papers", [])
    lines.append(f"Found {len(papers)} recent papers:\n")
    
    for i, paper in enumerate(papers[:5], 1):  # Top 5 papers
        title = paper.get("title", "Unknown")
        year = paper.get("year", "")
        lines.append(f"{i}. {title} ({year})")
        
        # Add abstract snippet if available
        abstract = paper.get("abstract", "")
        if abstract:
            snippet = abstract[:150] + "..." if len(abstract) > 150 else abstract
            lines.append(f"   {snippet}")
    
    return "\n".join(lines)


def _format_visualization_result(data: Dict[str, Any]) -> str:
    """Format visualization result for LLM."""
    lines = ["Cluster Visualization Data Generated:\n"]
    
    stats = data.get("statistics", {})
    n_points = data.get("n_points", 0)
    n_dims = data.get("n_dimensions", 0)
    
    lines.append(f"Generated {n_dims}D visualization with {n_points} points")
    lines.append(f"Clusters: {stats.get('n_clusters', 0)}")
    
    if data.get("visualization_saved"):
        lines.append(f"Saved to: {data.get('output_path')}")
    
    return "\n".join(lines)

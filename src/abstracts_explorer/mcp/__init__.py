"""
MCP (Model Context Protocol) server package.

Exposes clustering-analysis tools for LLM assistants. The public API is
re-exported here so ``from abstracts_explorer.mcp_server import ...`` keeps
working.
"""

from abstracts_explorer.mcp import tools as _tools  # noqa: F401  (registers MCP tools)
from abstracts_explorer.mcp.cache import _lookup_clustering_cache
from abstracts_explorer.mcp.errors import ClusterAnalysisError
from abstracts_explorer.mcp.resources import (
    _get_clustering_resources,
    _get_mcp_resources,
    close_mcp_resources,
    load_clustering_data,
)
from abstracts_explorer.mcp.server import mcp, run_mcp_server
from abstracts_explorer.mcp.tools import (
    analyze_cluster_topics,
    analyze_topic_relevance,
    get_cluster_visualization,
    get_conference_topics,
    get_paper_details,
    get_topic_evolution,
    recommend_related_papers,
    search_papers,
    suggest_authors_to_meet,
    summarize_interesting_papers,
)
from abstracts_explorer.mcp.tools.topics import _get_conference_topics_for_single_conference

__all__ = [
    "_get_clustering_resources",
    "_get_conference_topics_for_single_conference",
    "_get_mcp_resources",
    "_lookup_clustering_cache",
    "mcp",
    "run_mcp_server",
    "ClusterAnalysisError",
    "close_mcp_resources",
    "load_clustering_data",
    "analyze_cluster_topics",
    "get_conference_topics",
    "get_topic_evolution",
    "search_papers",
    "get_paper_details",
    "summarize_interesting_papers",
    "recommend_related_papers",
    "suggest_authors_to_meet",
    "analyze_topic_relevance",
    "get_cluster_visualization",
]

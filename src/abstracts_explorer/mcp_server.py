"""
MCP server facade.

The implementation lives in :mod:`abstracts_explorer.mcp`; this module is kept
as a stable import path.
"""

from abstracts_explorer.mcp import (
    ClusterAnalysisError,
    _apply_cached_cluster_labels,
    _get_clustering_resources,
    _get_conference_topics_for_single_conference,
    _get_mcp_resources,
    _lookup_clustering_cache,
    analyze_cluster_topics,
    analyze_topic_relevance,
    get_cluster_visualization,
    get_conference_topics,
    get_paper_details,
    get_topic_evolution,
    load_clustering_data,
    mcp,
    recommend_related_papers,
    run_mcp_server,
    search_papers,
    summarize_interesting_papers,
)

__all__ = [
    "_apply_cached_cluster_labels",
    "_get_clustering_resources",
    "_get_conference_topics_for_single_conference",
    "_get_mcp_resources",
    "_lookup_clustering_cache",
    "mcp",
    "run_mcp_server",
    "ClusterAnalysisError",
    "load_clustering_data",
    "analyze_cluster_topics",
    "get_conference_topics",
    "get_topic_evolution",
    "search_papers",
    "get_paper_details",
    "summarize_interesting_papers",
    "recommend_related_papers",
    "analyze_topic_relevance",
    "get_cluster_visualization",
]

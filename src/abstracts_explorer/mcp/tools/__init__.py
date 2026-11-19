"""MCP tool implementations (importing this package registers the tools)."""

from abstracts_explorer.mcp.tools.topics import (
    analyze_cluster_topics,
    _get_conference_topics_for_single_conference,
    get_conference_topics,
)
from abstracts_explorer.mcp.tools.trends import (
    get_topic_evolution,
)
from abstracts_explorer.mcp.tools.search import (
    search_papers,
    get_paper_details,
)
from abstracts_explorer.mcp.tools.interesting import (
    recommend_related_papers,
    suggest_authors_to_meet,
    summarize_interesting_papers,
)
from abstracts_explorer.mcp.tools.relevance import (
    analyze_topic_relevance,
)
from abstracts_explorer.mcp.tools.visualization import (
    get_cluster_visualization,
)

__all__ = [
    "analyze_cluster_topics",
    "_get_conference_topics_for_single_conference",
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

"""
RAG package.

Retrieval-Augmented Generation chat built on Pydantic AI with MCP tools.
The public API is re-exported here so ``from abstracts_explorer.rag import
...`` keeps working.
"""

from abstracts_explorer.rag.tool_context import RAGToolContext
from abstracts_explorer.rag.errors import RAGError
from abstracts_explorer.rag.manager import RAGChat
from abstracts_explorer.rag.agent import shutdown_tool_executor
from abstracts_explorer.rag.tools import (
    _run_rag_tool,
    _tool_analyze_topic_relevance,
    _tool_get_cluster_visualization,
    _tool_get_conference_topics,
    _tool_get_paper_details,
    _tool_get_topic_evolution,
    _tool_recommend_related_papers,
    _tool_search_papers,
    _tool_suggest_authors_to_meet,
    _tool_suggest_search_terms_from_orcid,
    _tool_summarize_interesting_papers,
)

__all__ = [
    "RAGChat",
    "RAGError",
    "RAGToolContext",
    "shutdown_tool_executor",
    "_run_rag_tool",
    "_tool_search_papers",
    "_tool_get_conference_topics",
    "_tool_get_topic_evolution",
    "_tool_analyze_topic_relevance",
    "_tool_get_cluster_visualization",
    "_tool_get_paper_details",
    "_tool_summarize_interesting_papers",
    "_tool_recommend_related_papers",
    "_tool_suggest_authors_to_meet",
    "_tool_suggest_search_terms_from_orcid",
]

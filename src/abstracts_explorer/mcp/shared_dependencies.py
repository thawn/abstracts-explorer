"""
Shared, patchable dependencies for MCP tools.

Tool modules access these through ``shared_dependencies.X`` so tests can patch a single
module instead of every tool module.
"""

from abstracts_explorer.clustering import ClusteringManager
from abstracts_explorer.config import get_config
from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.embeddings import EmbeddingsManager
from abstracts_explorer.interesting_papers import (
    recommend_related_papers as _recommend_related_papers,
    suggest_authors_to_meet as _suggest_authors_to_meet,
    summarize_interesting_papers as _summarize_interesting_papers,
)
from abstracts_explorer.utils.conference_utils import parse_conference_year

__all__ = [
    "ClusteringManager",
    "DatabaseManager",
    "EmbeddingsManager",
    "get_config",
    "parse_conference_year",
    "_recommend_related_papers",
    "_suggest_authors_to_meet",
    "_summarize_interesting_papers",
]

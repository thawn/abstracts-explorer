"""
Fixed clustering pipeline constants.

Shared by the CLI, web UI, MCP server, and cache keys so they stay consistent.
Do not mutate ``DEFAULT_CLUSTERING_PARAMS``.
"""

from typing import Any, Dict

DEFAULT_CLUSTERING_METHOD = "agglomerative"
DEFAULT_REDUCTION_METHOD = "tsne"
DEFAULT_N_COMPONENTS = 2
DEFAULT_CLUSTERING_PARAMS: Dict[str, Any] = {"linkage": "ward", "distance_threshold": 150.0}

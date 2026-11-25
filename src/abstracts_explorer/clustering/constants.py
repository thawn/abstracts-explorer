"""
Fixed clustering pipeline constants.

Shared by the CLI, web UI, MCP server, and cache keys so they stay consistent.
Do not mutate ``DEFAULT_CLUSTERING_PARAMS``.
"""

from typing import Any, Dict

DEFAULT_CLUSTERING_METHOD = "bertopic-agglomerative"
DEFAULT_REDUCTION_METHOD = "umap"
DEFAULT_N_COMPONENTS = 2

DEFAULT_NR_LEVELS = 10
DEFAULT_LEVEL_SCALE = "linear"
DEFAULT_CLUSTER_N_COMPONENTS = 5
DEFAULT_N_NEIGHBORS = 15
DEFAULT_MIN_DIST = 0.0
DEFAULT_RANDOM_STATE = 42
DEFAULT_DISTANCE_THRESHOLD = 0.5
DEFAULT_LINKAGE = "ward"
DEFAULT_METRIC = "euclidean"
PIPELINE_VERSION = 3

DEFAULT_CLUSTERING_PARAMS: Dict[str, Any] = {
    "distance_threshold": DEFAULT_DISTANCE_THRESHOLD,
    "linkage": DEFAULT_LINKAGE,
    "metric": DEFAULT_METRIC,
    "cluster_n_components": DEFAULT_CLUSTER_N_COMPONENTS,
    "n_neighbors": DEFAULT_N_NEIGHBORS,
    "min_dist": DEFAULT_MIN_DIST,
    "random_state": DEFAULT_RANDOM_STATE,
    "representation": "llm",
    "pipeline_version": PIPELINE_VERSION,
}


def clustering_params_for(use_llm_labels: bool = True, **overrides: Any) -> Dict[str, Any]:
    """
    Return effective clustering parameters for cache lookup and storage.

    Parameters
    ----------
    use_llm_labels : bool, optional
        Whether flat and hierarchy topics use LLM labels (default True).
    **overrides
        Parameter overrides for clustering parameters only. Hierarchy level
        presentation settings are intentionally not part of the cache key.

    Returns
    -------
    dict
        A new parameter dictionary with the representation mode applied.
    """
    params = dict(DEFAULT_CLUSTERING_PARAMS)
    params.update(overrides)
    params["representation"] = "llm" if use_llm_labels else "ctfidf"
    return params

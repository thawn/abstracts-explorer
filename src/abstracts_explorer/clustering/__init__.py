"""
Clustering package.

Clusters paper embeddings with BERTopic using Agglomerative clustering on UMAP
reduced embeddings. The server stores visualization data as JSON in the
clustering cache; Plotly rendering happens exclusively in the browser.

Public API is re-exported here so ``from abstracts_explorer.clustering import
...`` keeps working.
"""

from abstracts_explorer.clustering.constants import (
    DEFAULT_CLUSTERING_METHOD,
    DEFAULT_CLUSTERING_PARAMS,
    DEFAULT_N_COMPONENTS,
    DEFAULT_NR_LEVELS,
    DEFAULT_REDUCTION_METHOD,
    PIPELINE_VERSION,
    clustering_params_for,
)
from abstracts_explorer.clustering.errors import ClusteringError
from abstracts_explorer.clustering.manager import ClusteringManager
from abstracts_explorer.clustering.pipeline import (
    compute_clusters_with_cache,
    perform_clustering,
)

__all__ = [
    "ClusteringManager",
    "ClusteringError",
    "perform_clustering",
    "compute_clusters_with_cache",
    "DEFAULT_CLUSTERING_METHOD",
    "DEFAULT_REDUCTION_METHOD",
    "DEFAULT_N_COMPONENTS",
    "DEFAULT_NR_LEVELS",
    "DEFAULT_CLUSTERING_PARAMS",
    "PIPELINE_VERSION",
    "clustering_params_for",
]

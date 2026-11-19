"""
Clustering package.

Clusters paper embeddings with Agglomerative clustering (ward linkage) on the
full embeddings and uses t-SNE for 2-D visualization. ChromaDB stores the
vectors; paper metadata comes from the SQL database.

Public API is re-exported here so ``from abstracts_explorer.clustering import
...`` keeps working.
"""

from abstracts_explorer.clustering.algorithms import calculate_default_clusters
from abstracts_explorer.clustering.constants import (
    DEFAULT_CLUSTERING_METHOD,
    DEFAULT_CLUSTERING_PARAMS,
    DEFAULT_N_COMPONENTS,
    DEFAULT_REDUCTION_METHOD,
)
from abstracts_explorer.clustering.errors import ClusteringError
from abstracts_explorer.clustering.manager import ClusteringManager
from abstracts_explorer.clustering.pipeline import (
    compute_clusters_with_cache,
    perform_clustering,
    results_have_complete_flat_labels,
)

__all__ = [
    "ClusteringManager",
    "ClusteringError",
    "perform_clustering",
    "compute_clusters_with_cache",
    "results_have_complete_flat_labels",
    "calculate_default_clusters",
    "DEFAULT_CLUSTERING_METHOD",
    "DEFAULT_REDUCTION_METHOD",
    "DEFAULT_N_COMPONENTS",
    "DEFAULT_CLUSTERING_PARAMS",
]

"""
Clustering package.

Clusters paper embeddings with BERTopic using Agglomerative clustering on UMAP
reduced embeddings. The server stores visualization data as JSON in the
clustering cache; Plotly rendering happens exclusively in the browser.

Public API is re-exported here so ``from abstracts_explorer.clustering import
...`` keeps working.
"""

from typing import TYPE_CHECKING, Any

from abstracts_explorer.clustering.constants import (
    DEFAULT_CLUSTERING_METHOD,
    DEFAULT_CLUSTERING_PARAMS,
    DEFAULT_LEVEL_SCALE,
    DEFAULT_N_COMPONENTS,
    DEFAULT_NR_LEVELS,
    DEFAULT_REDUCTION_METHOD,
    PIPELINE_VERSION,
    clustering_params_for,
)
from abstracts_explorer.clustering.errors import ClusteringError

if TYPE_CHECKING:
    from abstracts_explorer.clustering.manager import ClusteringManager
    from abstracts_explorer.clustering.pipeline import (
        compute_clusters_with_cache,
        perform_clustering,
    )
    from abstracts_explorer.clustering.results import (
        build_cluster_centers_from_points,
        derive_hierarchy_levels,
    )

_LAZY_IMPORTS = {
    "ClusteringManager": "abstracts_explorer.clustering.manager",
    "perform_clustering": "abstracts_explorer.clustering.pipeline",
    "compute_clusters_with_cache": "abstracts_explorer.clustering.pipeline",
    "derive_hierarchy_levels": "abstracts_explorer.clustering.results",
    "build_cluster_centers_from_points": "abstracts_explorer.clustering.results",
}


def __getattr__(name: str) -> Any:
    """Lazily import heavy clustering symbols (BERTopic/sklearn stay unloaded)."""
    module_path = _LAZY_IMPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))


__all__ = [
    "ClusteringManager",
    "ClusteringError",
    "perform_clustering",
    "compute_clusters_with_cache",
    "derive_hierarchy_levels",
    "build_cluster_centers_from_points",
    "DEFAULT_CLUSTERING_METHOD",
    "DEFAULT_REDUCTION_METHOD",
    "DEFAULT_N_COMPONENTS",
    "DEFAULT_NR_LEVELS",
    "DEFAULT_LEVEL_SCALE",
    "DEFAULT_CLUSTERING_PARAMS",
    "PIPELINE_VERSION",
    "clustering_params_for",
]

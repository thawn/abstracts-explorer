"""
Clustering cache lookups and label restoration helpers.
"""

import logging
from typing import Any, Dict, List, Optional

from abstracts_explorer.mcp import shared_dependencies
from abstracts_explorer.clustering import (
    DEFAULT_CLUSTERING_METHOD,
    DEFAULT_CLUSTERING_PARAMS,
    DEFAULT_N_COMPONENTS,
    DEFAULT_REDUCTION_METHOD,
)

logger = logging.getLogger(__name__)


def _apply_cached_cluster_labels(
    cm: shared_dependencies.ClusteringManager,
    cached_results: Dict[str, Any],
) -> None:
    """
    Restore ``cluster_labels``, ``cluster_label_names``, and
    ``cluster_keywords`` on *cm* from cached clustering results.

    Parameters
    ----------
    cm : shared_dependencies.ClusteringManager
        Clustering manager with embeddings already loaded
        (``cm.paper_ids`` must be populated).
    cached_results : dict
        Cached results dict containing a ``"points"`` list where each
        element has ``"id"`` (or ``"paper_id"``) and ``"cluster"`` keys.
        May also contain ``"cluster_labels"`` (cluster name dict) and
        ``"cluster_keywords"`` (TF-IDF keyword dict).
    """
    import numpy as np

    point_id_to_cluster: Dict[str, int] = {}
    for point in cached_results.get("points", []):
        pid = point.get("id") or point.get("paper_id", "")
        point_id_to_cluster[pid] = point.get("cluster", -1)

    current_ids = cm.paper_ids or []
    cm.cluster_labels = np.array([point_id_to_cluster.get(pid, -1) for pid in current_ids])

    # Restore cluster label names (LLM-generated or TF-IDF-based names)
    if cached_results.get("cluster_labels"):
        cm.cluster_label_names = {int(k): v for k, v in cached_results["cluster_labels"].items()}

    # Restore TF-IDF cluster keywords
    if cached_results.get("cluster_keywords"):
        cm.cluster_keywords = {int(k): v for k, v in cached_results["cluster_keywords"].items()}


def _lookup_clustering_cache(
    db: "shared_dependencies.DatabaseManager",
    config: Any,
    conference: str,
    years: Optional[List[int]],
    collection_name: Optional[str] = None,
) -> Any:
    """
    Look up pre-computed clustering results from the database cache.

    A single requested year performs an exact per-year match and never falls
    back to an all-years entry.  Multi-year or empty ``years`` use the
    all-years entry (``year`` stored as ``NULL``).

    Parameters
    ----------
    db : shared_dependencies.DatabaseManager
        Open database connection.
    config : object
        Configuration object (needs ``embedding_model``).
    conference : str
        Conference name (already parsed, no trailing year).
    years : list of int or None
        Year filter.
    collection_name : str, optional
        ChromaDB collection the cache must belong to.

    Returns
    -------
    dict or None
        Cached clustering results, or ``None`` when nothing matches.
    """
    clustering_params: Dict[str, Any] = dict(DEFAULT_CLUSTERING_PARAMS)

    # Determine single year for column-based lookup
    cache_year = years[0] if years and len(years) == 1 else None

    return db.get_clustering_cache(
        embedding_model=config.embedding_model,
        reduction_method=DEFAULT_REDUCTION_METHOD,
        n_components=DEFAULT_N_COMPONENTS,
        clustering_method=DEFAULT_CLUSTERING_METHOD,
        n_clusters=None,
        clustering_params=clustering_params,
        conference=conference,
        year=cache_year,
        collection_name=collection_name or config.collection_name,
    )

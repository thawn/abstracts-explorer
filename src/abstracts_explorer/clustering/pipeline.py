"""
End-to-end clustering pipelines (uncached and cached).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from abstracts_explorer.clustering.constants import (
    DEFAULT_CLUSTERING_METHOD,
    DEFAULT_N_COMPONENTS,
    DEFAULT_REDUCTION_METHOD,
)
from abstracts_explorer.clustering.errors import ClusteringError
from abstracts_explorer.clustering.manager import ClusteringManager
from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.embeddings import EmbeddingsManager

logger = logging.getLogger(__name__)


def _run_pipeline(
    cm: ClusteringManager,
    *,
    n_components: int,
    n_clusters: Optional[int],
    random_state: int,
    limit: Optional[int] = None,
    conferences: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    clustering_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Run the shared clustering pipeline on *cm* in place.

    Loads embeddings, clusters on the full embeddings, reduces dimensions with
    t-SNE, and generates cluster labels (best effort).

    Parameters
    ----------
    cm : ClusteringManager
        Manager to populate.
    n_components : int
        Number of t-SNE components.
    n_clusters : int, optional
        Fixed cluster count; ``None`` uses the distance threshold or an
        automatically calculated count.
    random_state : int
        Random state for the t-SNE reduction.
    limit : int, optional
        Maximum number of embeddings to load.
    conferences : list of str, optional
        Filter by conferences.
    years : list of int, optional
        Filter by years.
    clustering_kwargs : dict, optional
        Extra Agglomerative parameters (e.g. ``distance_threshold``).
    """
    kwargs = dict(clustering_kwargs or {})

    logger.info("Loading embeddings...")
    cm.load_embeddings(limit=limit, conferences=conferences, years=years)

    logger.info("Clustering using Agglomerative on full embeddings...")
    cm.cluster(n_clusters=n_clusters, **kwargs)

    logger.info("Reducing dimensions using t-SNE for visualization...")
    cm.reduce_dimensions(n_components=n_components, random_state=random_state)

    logger.info("Generating cluster labels...")
    try:
        cm.extract_cluster_keywords(n_keywords=10)
        cm.generate_cluster_labels(use_llm=True, max_keywords=5)
    except Exception as e:
        logger.warning(f"Failed to generate cluster labels: {e}")


def perform_clustering(
    collection_name: str = "papers",
    n_components: int = DEFAULT_N_COMPONENTS,
    n_clusters: Optional[int] = None,
    output_path: Optional[Union[str, Path]] = None,
    random_state: int = 42,
    limit: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Perform complete clustering pipeline and optionally export results.

    Handles the full workflow: load embeddings, cluster on full embeddings
    (Agglomerative), reduce with t-SNE for visualization, and export if
    requested.

    Parameters
    ----------
    collection_name : str, optional
        Name of the ChromaDB collection, by default "papers"
    n_components : int, optional
        Number of components for dimensionality reduction, by default 2
    n_clusters : int, optional
        Number of clusters. Can be None if ``distance_threshold`` is provided
        in *kwargs*. If both are None, automatically calculated as
        n_papers / 100, clamped to [2, 500].
    output_path : str or Path, optional
        Path to export JSON results. If None, don't export.
    random_state : int, optional
        Random state for reproducibility, by default 42
    limit : int, optional
        Maximum number of embeddings to process. If None, process all.
    **kwargs
        Additional arguments passed to the clustering algorithm
        (e.g. distance_threshold, linkage)

    Returns
    -------
    dict
        Clustering results dictionary

    Raises
    ------
    ClusteringError
        If any step fails

    Examples
    --------
    >>> results = perform_clustering(
    ...     n_clusters=5,
    ...     output_path="clusters.json"
    ... )
    >>> print(f"Found {results['statistics']['n_clusters']} clusters")
    """
    try:
        with DatabaseManager() as db, EmbeddingsManager(collection_name=collection_name) as em:
            cm = ClusteringManager(em, db)
            _run_pipeline(
                cm,
                n_components=n_components,
                n_clusters=n_clusters,
                random_state=random_state,
                limit=limit,
                clustering_kwargs=kwargs,
            )
            results = cm.get_clustering_results()
            if output_path:
                logger.info(f"Exporting results to {output_path}...")
                cm.export_to_json(output_path)
            return results
    except Exception as e:
        raise ClusteringError(f"Clustering pipeline failed: {str(e)}") from e


def compute_clusters_with_cache(
    embeddings_manager: EmbeddingsManager,
    database: DatabaseManager,
    embedding_model: str,
    n_components: int = DEFAULT_N_COMPONENTS,
    n_clusters: Optional[int] = None,
    limit: Optional[int] = None,
    force: bool = False,
    conferences: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    **clustering_kwargs,
) -> Dict[str, Any]:
    """
    Compute clusters with caching support.

    Checks the cache first and returns cached results if available. On a cache
    miss or forced recompute, runs the Agglomerative + t-SNE pipeline, applies
    hierarchical labels, and saves the result to the cache.

    Parameters
    ----------
    embeddings_manager : EmbeddingsManager
        Embeddings manager instance
    database : DatabaseManager
        Database manager for cache operations
    embedding_model : str
        Current embedding model name
    n_components : int, optional
        Number of components for the t-SNE reduction, by default 2
    n_clusters : int, optional
        Number of clusters. Can be None if ``distance_threshold`` is provided
        in *clustering_kwargs*. If both are None, auto-calculated based on
        data size.
    limit : int, optional
        Maximum number of embeddings to process
    force : bool, optional
        Force recompute even if cache exists, by default False
    conferences : list of str, optional
        Filter to only cluster papers from these conferences.
    years : list of int, optional
        Filter to only cluster papers from these years.
    **clustering_kwargs
        Additional clustering parameters (e.g., distance_threshold, linkage)

    Returns
    -------
    dict
        Clustering results with points, statistics, and metadata

    Raises
    ------
    ClusteringError
        If clustering fails

    Examples
    --------
    >>> results = compute_clusters_with_cache(
    ...     em, db, "text-embedding-model",
    ...     distance_threshold=150.0,
    ...     linkage="ward"
    ... )
    """
    cache_conference = conferences[0] if conferences and len(conferences) == 1 else None
    cache_year = years[0] if years and len(years) == 1 else None

    if not force and not limit:
        cache_n_clusters: Optional[int] = n_clusters
        cache_params = dict(clustering_kwargs)
        if "distance_threshold" in cache_params:
            cache_n_clusters = None

        exact_cached = database.get_clustering_cache(
            embedding_model=embedding_model,
            reduction_method=DEFAULT_REDUCTION_METHOD,
            n_components=n_components,
            clustering_method=DEFAULT_CLUSTERING_METHOD,
            n_clusters=cache_n_clusters,
            clustering_params=cache_params or None,
            conference=cache_conference,
            year=cache_year,
        )
        if exact_cached:
            logger.info("Using exact cached clustering results (including reduction)")
            return exact_cached

    logger.info("Computing new clustering results...")
    cm = ClusteringManager(embeddings_manager, database)
    _run_pipeline(
        cm,
        n_components=n_components,
        n_clusters=n_clusters,
        random_state=42,
        limit=limit,
        conferences=conferences,
        years=years,
        clustering_kwargs=clustering_kwargs,
    )

    if cm.cluster_hierarchy is not None:
        linkage_method = clustering_kwargs.get("linkage", "ward")
        use_llm = clustering_kwargs.get("use_llm_labels", True)

        cached_hier_labels = database.get_hierarchical_label_cache(
            embedding_model=embedding_model,
            linkage=linkage_method,
        )

        if cached_hier_labels:
            logger.info(f"Using cached hierarchical labels ({len(cached_hier_labels)} nodes)")
            hierarchical_labels = cached_hier_labels
        else:
            logger.info("Generating hierarchical labels for agglomerative clustering...")
            try:
                hierarchical_labels = cm.generate_hierarchical_labels(use_llm=use_llm, max_keywords=5)
                if not limit:
                    try:
                        database.save_hierarchical_label_cache(
                            embedding_model=embedding_model,
                            labels=hierarchical_labels,
                            linkage=linkage_method,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to save hierarchical label cache: {e}")
            except Exception as e:
                logger.warning(f"Failed to generate hierarchical labels: {e}")
                hierarchical_labels = {}

        if hierarchical_labels and "tree" in cm.cluster_hierarchy and "nodes" in cm.cluster_hierarchy["tree"]:
            for node_id, label in hierarchical_labels.items():
                node_id_int = int(node_id)
                if node_id_int in cm.cluster_hierarchy["tree"]["nodes"]:
                    cm.cluster_hierarchy["tree"]["nodes"][node_id_int]["label"] = label
        logger.info(
            f"Applied {len(hierarchical_labels)} hierarchical labels "
            f"(LLM: {use_llm}, cached: {cached_hier_labels is not None})"
        )

    results = cm.get_clustering_results()

    if not limit:
        try:
            save_params = dict(clustering_kwargs)
            database.save_clustering_cache(
                embedding_model=embedding_model,
                reduction_method=DEFAULT_REDUCTION_METHOD,
                n_components=n_components,
                clustering_method=DEFAULT_CLUSTERING_METHOD,
                results=results,
                n_clusters=n_clusters,
                clustering_params=save_params or None,
                conference=cache_conference,
                year=cache_year,
            )
        except Exception as e:
            logger.warning(f"Failed to save clustering cache: {e}")

    return results

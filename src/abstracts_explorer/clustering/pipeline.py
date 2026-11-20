"""
End-to-end BERTopic clustering pipelines (uncached and cached).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from abstracts_explorer.clustering.constants import (
    DEFAULT_CLUSTERING_METHOD,
    DEFAULT_CLUSTERING_PARAMS,
    DEFAULT_CLUSTER_N_COMPONENTS,
    DEFAULT_LEVEL_SCALE,
    DEFAULT_N_COMPONENTS,
    DEFAULT_NR_LEVELS,
    DEFAULT_REDUCTION_METHOD,
)
from abstracts_explorer.clustering.errors import ClusteringError
from abstracts_explorer.clustering.manager import ClusteringManager
from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.embeddings import EmbeddingsManager

logger = logging.getLogger(__name__)


def _effective_params(
    use_llm_labels: bool,
    nr_levels: int,
    level_scale: str,
    clustering_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the effective parameter dictionary used as the cache key."""
    params = dict(DEFAULT_CLUSTERING_PARAMS)
    for key, value in (clustering_kwargs or {}).items():
        if key in params:
            params[key] = value
    params["nr_levels"] = int(nr_levels)
    params["level_scale"] = level_scale
    params["representation"] = "llm" if use_llm_labels else "ctfidf"
    return params


def _save_clustering_results(
    database: DatabaseManager,
    embedding_model: str,
    results: Dict[str, Any],
    clustering_params: Dict[str, Any],
    conference: Optional[str],
    year: Optional[int],
    collection: Optional[str],
) -> None:
    """Save clustering results to the database cache."""
    database.save_clustering_cache(
        embedding_model=embedding_model,
        reduction_method=DEFAULT_REDUCTION_METHOD,
        n_components=DEFAULT_N_COMPONENTS,
        clustering_method=DEFAULT_CLUSTERING_METHOD,
        results=results,
        n_clusters=None,
        clustering_params=clustering_params,
        conference=conference,
        year=year,
        collection_name=collection,
    )


def perform_clustering(
    collection_name: str = "papers",
    n_components: int = DEFAULT_N_COMPONENTS,
    output_path: Optional[Union[str, Path]] = None,
    limit: Optional[int] = None,
    show_progress: bool = False,
    conferences: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    use_llm_labels: bool = True,
    nr_levels: int = DEFAULT_NR_LEVELS,
    level_scale: str = DEFAULT_LEVEL_SCALE,
    cluster_n_components: int = DEFAULT_CLUSTER_N_COMPONENTS,
    requests_per_minute: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Perform the complete BERTopic clustering pipeline.

    Parameters
    ----------
    collection_name : str, optional
        Name of the ChromaDB collection (default ``"papers"``).
    n_components : int, optional
        Number of visualization dimensions (default 2).
    output_path : str or Path, optional
        Path to export JSON results.
    limit : int, optional
        Maximum number of embeddings to process.
    show_progress : bool, optional
        Enable BERTopic verbose output.
    conferences : list of str, optional
        Filter by conferences.
    years : list of int, optional
        Filter by years.
    use_llm_labels : bool, optional
        Whether to use LLM labels (default True).
    nr_levels : int, optional
        Number of hierarchy levels (default 10).
    level_scale : str, optional
        Level scaling mode (default ``"linear"``).
    cluster_n_components : int, optional
        UMAP components used before clustering (default 5).
    requests_per_minute : int, optional
        Maximum embedding/LLM API requests per minute. ``None`` uses the
        configured default; ``0`` disables rate limiting.
    **kwargs
        Ignored compatibility parameters.

    Returns
    -------
    dict
        Clustering results.

    Raises
    ------
    ClusteringError
        If any step fails.
    """
    try:
        with (
            DatabaseManager() as db,
            EmbeddingsManager(
                collection_name=collection_name,
                requests_per_minute=requests_per_minute,
            ) as em,
        ):
            cm = ClusteringManager(em, db)
            cm.load_embeddings(limit=limit, conferences=conferences, years=years)
            results = cm.fit(
                nr_levels=nr_levels,
                level_scale=level_scale,
                use_llm_labels=use_llm_labels,
                show_progress=show_progress,
                cluster_n_components=cluster_n_components,
            )
            if output_path:
                cm.export_to_json(output_path)
            return results
    except ClusteringError:
        raise
    except Exception as exc:
        raise ClusteringError(f"Clustering pipeline failed: {exc}") from exc


def compute_clusters_with_cache(
    embeddings_manager: EmbeddingsManager,
    database: DatabaseManager,
    embedding_model: str,
    limit: Optional[int] = None,
    force: bool = False,
    conferences: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    use_llm_labels: bool = True,
    show_progress: bool = False,
    nr_levels: int = DEFAULT_NR_LEVELS,
    level_scale: str = DEFAULT_LEVEL_SCALE,
    cluster_n_components: int = DEFAULT_CLUSTER_N_COMPONENTS,
    **clustering_kwargs: Any,
) -> Dict[str, Any]:
    """
    Compute BERTopic clusters with database caching.

    Parameters
    ----------
    embeddings_manager : EmbeddingsManager
        Embeddings manager instance.
    database : DatabaseManager
        Database manager for cache operations.
    embedding_model : str
        Current embedding model name.
    limit : int, optional
        Maximum number of embeddings to process. Results are not cached when
        a limit is set.
    force : bool, optional
        Force recomputation even when a cache entry exists.
    conferences : list of str, optional
        Filter to conferences.
    years : list of int, optional
        Filter to years.
    use_llm_labels : bool, optional
        Whether to use LLM labels (default True).
    show_progress : bool, optional
        Enable BERTopic verbose output.
    nr_levels : int, optional
        Number of hierarchy levels (default 10).
    level_scale : str, optional
        Level scaling mode (default ``"linear"``).
    cluster_n_components : int, optional
        UMAP components used before clustering (default 5).
    **clustering_kwargs
        Additional clustering parameters used in the cache key.

    Returns
    -------
    dict
        Clustering results.

    Raises
    ------
    ClusteringError
        If clustering fails.
    """
    cache_conference = conferences[0] if conferences and len(conferences) == 1 else None
    cache_year = years[0] if years and len(years) == 1 else None
    cache_collection = getattr(embeddings_manager, "collection_name", None)
    params = _effective_params(use_llm_labels, nr_levels, level_scale, clustering_kwargs)

    if not force and not limit:
        cached = database.get_clustering_cache(
            embedding_model=embedding_model,
            reduction_method=DEFAULT_REDUCTION_METHOD,
            n_components=DEFAULT_N_COMPONENTS,
            clustering_method=DEFAULT_CLUSTERING_METHOD,
            n_clusters=None,
            clustering_params=params,
            conference=cache_conference,
            year=cache_year,
            collection_name=cache_collection,
        )
        if cached:
            logger.info("Using cached BERTopic clustering results")
            return cached

    logger.info("Computing new BERTopic clustering results...")
    cm = ClusteringManager(embeddings_manager, database)
    cm.load_embeddings(limit=limit, conferences=conferences, years=years)
    results = cm.fit(
        nr_levels=nr_levels,
        level_scale=level_scale,
        use_llm_labels=use_llm_labels,
        show_progress=show_progress,
        cluster_n_components=cluster_n_components,
    )

    if not limit:
        _save_clustering_results(
            database,
            embedding_model,
            results,
            params,
            cache_conference,
            cache_year,
            cache_collection,
        )
    return results

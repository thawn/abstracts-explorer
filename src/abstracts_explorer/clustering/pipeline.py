"""
End-to-end clustering pipelines (uncached and cached).
"""

import copy
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

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


def _paper_fingerprint(paper_ids: List[str]) -> str:
    """Return a short fingerprint for a paper-ID sequence."""
    return hashlib.sha256("\n".join(str(uid) for uid in paper_ids).encode("utf-8")).hexdigest()[:16]


def _hierarchy_scope(
    collection: Optional[str],
    conference: Optional[str],
    year: Optional[int],
    linkage: str,
    paper_ids: List[str],
) -> str:
    """Build the hierarchy-label cache scope."""
    return "|".join(
        [
            str(collection or ""),
            str(conference or ""),
            str(year if year is not None else ""),
            linkage,
            _paper_fingerprint(paper_ids),
        ]
    )


def _flat_scope(
    collection: Optional[str],
    conference: Optional[str],
    year: Optional[int],
    linkage: str,
    paper_ids: List[str],
    clustering_kwargs: Dict[str, Any],
    n_clusters: Optional[int],
) -> str:
    """Build the flat-label cache scope."""
    cache_n_clusters = None if "distance_threshold" in clustering_kwargs else n_clusters
    payload = json.dumps(
        {"n_clusters": cache_n_clusters, "params": clustering_kwargs},
        sort_keys=True,
        default=str,
    )
    params_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{_hierarchy_scope(collection, conference, year, linkage, paper_ids)}|{params_hash}"


def _normalize_labels(labels: Optional[Dict[Any, Any]]) -> Dict[int, str]:
    """Normalize cached label keys to integers, dropping empty values."""
    return {int(key): value for key, value in (labels or {}).items() if isinstance(value, str) and value.strip()}


def _normalize_keywords(keywords: Optional[Dict[Any, Any]]) -> Dict[int, List[str]]:
    """Normalize cached keyword keys to integers."""
    return {int(key): list(value) for key, value in (keywords or {}).items() if value}


def _cluster_ids_from_results(results: Dict[str, Any]) -> List[int]:
    """Return sorted cluster IDs from cached points."""
    cluster_ids = {
        int(point["cluster"])
        for point in (results.get("points") or [])
        if isinstance(point, dict) and int(point.get("cluster", -1)) >= 0
    }
    return sorted(cluster_ids)


def _hierarchy_node_ids(results: Dict[str, Any]) -> List[int]:
    """Return hierarchy node IDs from cached results."""
    nodes = ((results.get("cluster_hierarchy") or {}).get("tree") or {}).get("nodes") or {}
    return sorted(int(node_id) for node_id in nodes)


def results_have_complete_flat_labels(results: Dict[str, Any]) -> bool:
    """
    Return whether every cluster in *results* has a non-empty label.

    Parameters
    ----------
    results : dict
        Clustering results.

    Returns
    -------
    bool
        ``True`` when all flat cluster labels are present.
    """
    labels = _normalize_labels(results.get("cluster_labels"))
    return all(cluster_id in labels for cluster_id in _cluster_ids_from_results(results))


def _results_have_complete_hierarchy_labels(results: Dict[str, Any]) -> bool:
    """Return whether every hierarchy node has a non-empty label."""
    labels = _normalize_labels(
        {
            int(node_id): node.get("label")
            for node_id, node in (
                ((results.get("cluster_hierarchy") or {}).get("tree") or {}).get("nodes") or {}
            ).items()
            if isinstance(node, dict)
        }
    )
    return all(node_id in labels for node_id in _hierarchy_node_ids(results))


def _merge_flat_labels(results: Dict[str, Any], cached: Optional[Dict[int, str]]) -> Dict[int, str]:
    """Merge flat labels stored in results and the flat label cache."""
    labels = _normalize_labels(results.get("cluster_labels"))
    labels.update(_normalize_labels(cached))
    return labels


def _merge_hierarchy_labels(results: Dict[str, Any], cached: Optional[Dict[int, str]]) -> Dict[int, str]:
    """Merge hierarchy labels stored in results and the hierarchy label cache."""
    nodes = ((results.get("cluster_hierarchy") or {}).get("tree") or {}).get("nodes") or {}
    labels = _normalize_labels(
        {
            int(node_id): node.get("label")
            for node_id, node in nodes.items()
            if isinstance(node, dict) and node.get("label")
        }
    )
    labels.update(_normalize_labels(cached))
    return labels


def _clear_labels_in_results(results: Dict[str, Any]) -> None:
    """Remove flat and hierarchy labels from a results dict before regeneration."""
    results.pop("cluster_labels", None)
    hierarchy = results.get("cluster_hierarchy")
    if hierarchy and "tree" in hierarchy and "nodes" in hierarchy["tree"]:
        for node in hierarchy["tree"]["nodes"].values():
            if isinstance(node, dict):
                node.pop("label", None)


def _sync_labels_to_results(
    results: Dict[str, Any],
    flat_labels: Dict[int, str],
    hierarchy_labels: Dict[int, str],
    keywords: Dict[int, List[str]],
) -> None:
    """Write current flat/hierarchy labels and keywords into a results dict."""
    if flat_labels:
        results["cluster_labels"] = flat_labels
    else:
        results.pop("cluster_labels", None)

    if keywords:
        results["cluster_keywords"] = keywords

    hierarchy = results.get("cluster_hierarchy")
    if hierarchy and "tree" in hierarchy and "nodes" in hierarchy["tree"]:
        nodes = hierarchy["tree"]["nodes"]
        for node_id, label in hierarchy_labels.items():
            if node_id in nodes:
                nodes[node_id]["label"] = label
            elif str(node_id) in nodes:
                nodes[str(node_id)]["label"] = label


def _save_clustering_results(
    database: DatabaseManager,
    embedding_model: str,
    results: Dict[str, Any],
    n_components: int,
    n_clusters: Optional[int],
    clustering_kwargs: Dict[str, Any],
    conference: Optional[str],
    year: Optional[int],
    collection: Optional[str],
) -> None:
    """Save clustering results to the database cache."""
    database.save_clustering_cache(
        embedding_model=embedding_model,
        reduction_method=DEFAULT_REDUCTION_METHOD,
        n_components=n_components,
        clustering_method=DEFAULT_CLUSTERING_METHOD,
        results=results,
        n_clusters=n_clusters,
        clustering_params=dict(clustering_kwargs) or None,
        conference=conference,
        year=year,
        collection_name=collection,
    )


def _restore_cached_clustering(
    embeddings_manager: EmbeddingsManager,
    database: DatabaseManager,
    results: Dict[str, Any],
    conferences: Optional[List[str]],
    years: Optional[List[int]],
    limit: Optional[int],
) -> Optional[ClusteringManager]:
    """Restore a ClusteringManager from cached clustering results."""
    points = results.get("points") or []
    hierarchy = results.get("cluster_hierarchy")
    if not points or not hierarchy:
        return None

    id_to_cluster = {
        str(point["id"]): int(point["cluster"])
        for point in points
        if isinstance(point, dict) and "id" in point and int(point.get("cluster", -1)) >= 0
    }
    if not id_to_cluster:
        return None

    cm = ClusteringManager(embeddings_manager, database)
    try:
        cm.load_embeddings(limit=limit, conferences=conferences, years=years)
    except Exception as e:
        logger.warning("Failed to load embeddings for label resume: %s", e)
        return None

    missing = [paper_id for paper_id in (cm.paper_ids or []) if paper_id not in id_to_cluster]
    if missing:
        logger.warning(
            "Cached clustering assignments are incomplete (%d papers missing); falling back to recompute",
            len(missing),
        )
        return None

    cm.cluster_labels = np.asarray([id_to_cluster[paper_id] for paper_id in (cm.paper_ids or [])])
    cm.cluster_hierarchy = copy.deepcopy(hierarchy)
    nodes = (cm.cluster_hierarchy.get("tree") or {}).get("nodes") or {}
    if nodes:
        cm.cluster_hierarchy["tree"]["nodes"] = {int(node_id): node for node_id, node in nodes.items()}
    cm.cluster_label_names = _normalize_labels(results.get("cluster_labels"))
    cm.cluster_keywords = _normalize_keywords(results.get("cluster_keywords"))
    return cm


def _ensure_cluster_keywords(cm: ClusteringManager) -> Dict[int, List[str]]:
    """Ensure every cluster has keywords, reusing cached values."""
    if cm.cluster_labels is None:
        raise ClusteringError("No clustering performed. Call cluster() first.")
    cluster_ids = sorted(int(label) for label in np.unique(cm.cluster_labels) if label >= 0)
    existing = _normalize_keywords(cm.cluster_keywords)
    if all(existing.get(cluster_id) for cluster_id in cluster_ids):
        return existing

    extracted = _normalize_keywords(cm.extract_cluster_keywords(n_keywords=10))
    for cluster_id in cluster_ids:
        if not existing.get(cluster_id):
            existing[cluster_id] = extracted.get(cluster_id, [])
    cm.cluster_keywords = existing
    return existing


def _generate_labels_for_manager(
    cm: ClusteringManager,
    results: Dict[str, Any],
    *,
    embedding_model: str,
    database: DatabaseManager,
    flat_scope: str,
    hierarchy_scope: str,
    hierarchy_linkage: str,
    use_llm_labels: bool,
    show_progress: bool,
    existing_flat: Optional[Dict[int, str]] = None,
    existing_hierarchy: Optional[Dict[int, str]] = None,
    save_callback: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """Generate resumable flat and hierarchy labels for *cm* and update results."""
    keywords = _ensure_cluster_keywords(cm)
    flat_labels = _normalize_labels(existing_flat)
    cm.cluster_label_names = flat_labels

    def on_flat_label(cluster_id: int, label: str) -> None:
        flat_labels[cluster_id] = label
        database.upsert_flat_label_cache(embedding_model, {cluster_id: label}, flat_scope)
        _sync_labels_to_results(results, flat_labels, _merge_hierarchy_labels(results, None), keywords)

    cm.generate_cluster_labels(
        use_llm=use_llm_labels,
        max_keywords=5,
        show_progress=show_progress,
        existing_labels=flat_labels,
        on_label=on_flat_label,
    )
    flat_labels = _normalize_labels(cm.cluster_label_names)
    _sync_labels_to_results(results, flat_labels, _merge_hierarchy_labels(results, None), keywords)
    if save_callback is not None:
        save_callback()

    missing_flat = [cluster_id for cluster_id in _cluster_ids_from_results(results) if cluster_id not in flat_labels]
    if missing_flat:
        logger.warning(
            "Flat labels remain incomplete (%d missing); skipping hierarchy labeling",
            len(missing_flat),
        )
        return results

    hierarchy_labels = _normalize_labels(existing_hierarchy)

    def on_hierarchy_label(node_id: int, label: str) -> None:
        hierarchy_labels[node_id] = label
        database.upsert_hierarchical_label_cache(
            embedding_model,
            {node_id: label},
            linkage=hierarchy_linkage,
            scope=hierarchy_scope,
        )
        _sync_labels_to_results(results, flat_labels, hierarchy_labels, keywords)

    try:
        hierarchy_labels = cm.generate_hierarchical_labels(
            use_llm=use_llm_labels,
            max_keywords=5,
            show_progress=show_progress,
            existing_labels=hierarchy_labels,
            on_label=on_hierarchy_label,
        )
    except Exception:
        _sync_labels_to_results(results, flat_labels, hierarchy_labels, keywords)
        if save_callback is not None:
            save_callback()
        raise

    _sync_labels_to_results(results, flat_labels, hierarchy_labels, keywords)
    if save_callback is not None:
        save_callback()
    return results


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
    use_llm_labels: bool = True,
    show_progress: bool = False,
    generate_labels: bool = True,
) -> None:
    """
    Run the shared clustering pipeline on *cm* in place.

    Loads embeddings, clusters on the full embeddings, reduces dimensions with
    t-SNE, and optionally generates flat cluster labels.

    Parameters
    ----------
    cm : ClusteringManager
        Manager to populate.
    n_components : int
        Number of t-SNE components.
    n_clusters : int, optional
        Fixed cluster count.
    random_state : int
        Random state for the t-SNE reduction.
    limit : int, optional
        Maximum number of embeddings to load.
    conferences : list of str, optional
        Filter by conferences.
    years : list of int, optional
        Filter by years.
    clustering_kwargs : dict, optional
        Extra Agglomerative parameters.
    use_llm_labels : bool, optional
        Whether flat cluster labels should use the LLM.
    show_progress : bool, optional
        Show progress bars while generating cluster labels.
    generate_labels : bool, optional
        Whether to generate flat labels immediately, by default True.
    """
    kwargs = dict(clustering_kwargs or {})

    logger.info("Loading embeddings...")
    cm.load_embeddings(limit=limit, conferences=conferences, years=years)

    logger.info("Clustering using Agglomerative on full embeddings...")
    cm.cluster(n_clusters=n_clusters, **kwargs)

    logger.info("Reducing dimensions using t-SNE for visualization...")
    cm.reduce_dimensions(n_components=n_components, random_state=random_state)

    if generate_labels:
        logger.info("Generating cluster labels...")
        cm.extract_cluster_keywords(n_keywords=10)
        cm.generate_cluster_labels(use_llm=use_llm_labels, max_keywords=5, show_progress=show_progress)


def perform_clustering(
    collection_name: str = "papers",
    n_components: int = DEFAULT_N_COMPONENTS,
    n_clusters: Optional[int] = None,
    output_path: Optional[Union[str, Path]] = None,
    random_state: int = 42,
    limit: Optional[int] = None,
    show_progress: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Perform complete clustering pipeline and optionally export results.

    Parameters
    ----------
    collection_name : str, optional
        Name of the ChromaDB collection, by default "papers".
    n_components : int, optional
        Number of components for dimensionality reduction, by default 2.
    n_clusters : int, optional
        Number of clusters.
    output_path : str or Path, optional
        Path to export JSON results.
    random_state : int, optional
        Random state for reproducibility, by default 42.
    limit : int, optional
        Maximum number of embeddings to process.
    show_progress : bool, optional
        Show progress bars while generating cluster labels.
    **kwargs
        Additional arguments passed to the clustering algorithm.

    Returns
    -------
    dict
        Clustering results dictionary.

    Raises
    ------
    ClusteringError
        If any step fails.
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
                show_progress=show_progress,
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
    force_recreate_labels: bool = False,
    conferences: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    use_llm_labels: bool = True,
    show_progress: bool = False,
    **clustering_kwargs: Any,
) -> Dict[str, Any]:
    """
    Compute clusters with resumable label caching.

    Parameters
    ----------
    embeddings_manager : EmbeddingsManager
        Embeddings manager instance.
    database : DatabaseManager
        Database manager for cache operations.
    embedding_model : str
        Current embedding model name.
    n_components : int, optional
        Number of components for the t-SNE reduction, by default 2.
    n_clusters : int, optional
        Number of clusters.
    limit : int, optional
        Maximum number of embeddings to process.
    force : bool, optional
        Force clustering recomputation.
    force_recreate_labels : bool, optional
        Regenerate all flat and hierarchy labels while reusing cached
        clustering.
    conferences : list of str, optional
        Filter to conferences.
    years : list of int, optional
        Filter to years.
    use_llm_labels : bool, optional
        Whether label generation should use the LLM.
    show_progress : bool, optional
        Show progress bars.
    **clustering_kwargs
        Additional clustering parameters.

    Returns
    -------
    dict
        Clustering results with points, statistics, and labels.

    Raises
    ------
    ClusteringError
        If clustering or hierarchy labeling fails.
    """
    cache_conference = conferences[0] if conferences and len(conferences) == 1 else None
    cache_year = years[0] if years and len(years) == 1 else None
    cache_collection = getattr(embeddings_manager, "collection_name", None)
    linkage = str(clustering_kwargs.get("linkage", "ward"))

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
            collection_name=cache_collection,
        )
        if exact_cached:
            paper_ids = [str(point["id"]) for point in (exact_cached.get("points") or []) if "id" in point]
            flat_scope = _flat_scope(
                cache_collection,
                cache_conference,
                cache_year,
                linkage,
                paper_ids,
                clustering_kwargs,
                n_clusters,
            )
            hierarchy_scope = _hierarchy_scope(
                cache_collection,
                cache_conference,
                cache_year,
                linkage,
                paper_ids,
            )

            if force_recreate_labels:
                database.clear_flat_label_cache(embedding_model, flat_scope)
                database.delete_hierarchical_label_cache(
                    embedding_model=embedding_model,
                    linkage=linkage,
                    scope=hierarchy_scope,
                )
                _clear_labels_in_results(exact_cached)
                existing_flat: Dict[int, str] = {}
                existing_hierarchy: Dict[int, str] = {}
            else:
                existing_flat = _merge_flat_labels(
                    exact_cached,
                    database.get_flat_label_cache(embedding_model, flat_scope),
                )
                existing_hierarchy = _merge_hierarchy_labels(
                    exact_cached,
                    database.get_hierarchical_label_cache(
                        embedding_model=embedding_model,
                        linkage=linkage,
                        scope=hierarchy_scope,
                    ),
                )

            flat_missing = [
                cluster_id
                for cluster_id in _cluster_ids_from_results(exact_cached)
                if cluster_id not in existing_flat
            ]
            hierarchy_missing = [
                node_id for node_id in _hierarchy_node_ids(exact_cached) if node_id not in existing_hierarchy
            ]

            if not force_recreate_labels and not flat_missing and not hierarchy_missing:
                results_flat = _normalize_labels(exact_cached.get("cluster_labels"))
                results_hierarchy = _merge_hierarchy_labels(exact_cached, None)
                if set(existing_flat) != set(results_flat) or set(existing_hierarchy) != set(results_hierarchy):
                    _sync_labels_to_results(
                        exact_cached,
                        existing_flat,
                        existing_hierarchy,
                        _normalize_keywords(exact_cached.get("cluster_keywords")),
                    )
                    _save_clustering_results(
                        database,
                        embedding_model,
                        exact_cached,
                        n_components,
                        n_clusters,
                        clustering_kwargs,
                        cache_conference,
                        cache_year,
                        cache_collection,
                    )
                logger.info("Using exact cached clustering results (including labels)")
                return exact_cached

            cm = _restore_cached_clustering(
                embeddings_manager,
                database,
                exact_cached,
                conferences,
                years,
                limit,
            )
            if cm is not None:

                def save_cached_callback() -> None:
                    _save_clustering_results(
                        database,
                        embedding_model,
                        exact_cached,
                        n_components,
                        n_clusters,
                        clustering_kwargs,
                        cache_conference,
                        cache_year,
                        cache_collection,
                    )

                return _generate_labels_for_manager(
                    cm,
                    exact_cached,
                    embedding_model=embedding_model,
                    database=database,
                    flat_scope=flat_scope,
                    hierarchy_scope=hierarchy_scope,
                    hierarchy_linkage=linkage,
                    use_llm_labels=use_llm_labels,
                    show_progress=show_progress,
                    existing_flat=existing_flat,
                    existing_hierarchy=existing_hierarchy,
                    save_callback=save_cached_callback,
                )
            logger.warning("Cached clustering state could not be restored; recomputing")

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
        use_llm_labels=use_llm_labels,
        show_progress=show_progress,
        generate_labels=False,
    )

    paper_ids = [str(uid) for uid in (cm.paper_ids or [])]
    flat_scope = _flat_scope(
        cache_collection,
        cache_conference,
        cache_year,
        linkage,
        paper_ids,
        clustering_kwargs,
        n_clusters,
    )
    hierarchy_scope = _hierarchy_scope(
        cache_collection,
        cache_conference,
        cache_year,
        linkage,
        paper_ids,
    )
    if force_recreate_labels:
        database.clear_flat_label_cache(embedding_model, flat_scope)
        database.delete_hierarchical_label_cache(
            embedding_model=embedding_model,
            linkage=linkage,
            scope=hierarchy_scope,
        )

    results = cm.get_clustering_results()
    if force_recreate_labels:
        _clear_labels_in_results(results)

    def save_callback() -> None:
        _save_clustering_results(
            database,
            embedding_model,
            results,
            n_components,
            n_clusters,
            clustering_kwargs,
            cache_conference,
            cache_year,
            cache_collection,
        )

    if not limit:
        save_callback()

    return _generate_labels_for_manager(
        cm,
        results,
        embedding_model=embedding_model,
        database=database,
        flat_scope=flat_scope,
        hierarchy_scope=hierarchy_scope,
        hierarchy_linkage=linkage,
        use_llm_labels=use_llm_labels,
        show_progress=show_progress,
        existing_flat={} if force_recreate_labels else _normalize_labels(results.get("cluster_labels")),
        existing_hierarchy={},
        save_callback=save_callback if not limit else None,
    )

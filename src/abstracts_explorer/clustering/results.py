"""
Cluster statistics, results assembly, and JSON export.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Union

import numpy as np

from abstracts_explorer.clustering.constants import DEFAULT_LEVEL_SCALE, DEFAULT_NR_LEVELS
from abstracts_explorer.clustering.errors import ClusteringError
from abstracts_explorer.clustering.hierarchy_levels import build_level_mappings_from_tree
from abstracts_explorer.clustering.labels import split_label_description
from abstracts_explorer.utils.json_utils import NumpyEncoder

logger = logging.getLogger(__name__)


def _truncate_title(title: str, max_length: int) -> str:
    """Return a title truncated to *max_length* characters."""
    if len(title) > max_length:
        return title[:max_length] + "..."
    return title


def build_statistics(topics: Sequence[int]) -> Dict[str, Any]:
    """
    Build cluster statistics from topic assignments.

    Parameters
    ----------
    topics : sequence of int
        Topic ID per paper.

    Returns
    -------
    dict
        Statistics with ``n_clusters``, ``n_noise``, ``cluster_sizes``, and
        ``total_papers``.
    """
    topic_array = np.asarray(topics)
    unique_labels = np.unique(topic_array)
    cluster_sizes = {int(label): int(np.sum(topic_array == label)) for label in unique_labels if int(label) >= 0}
    return {
        "n_clusters": len(cluster_sizes),
        "n_noise": int(np.sum(topic_array == -1)),
        "cluster_sizes": cluster_sizes,
        "total_papers": int(len(topic_array)),
    }


def build_points(
    paper_ids: Sequence[str],
    reduced_embeddings: np.ndarray,
    metadatas: Sequence[Dict[str, Any]],
    topics: Sequence[int],
    max_title_length: int = 100,
) -> List[Dict[str, Any]]:
    """
    Build visualization points.

    Parameters
    ----------
    paper_ids : sequence of str
        Paper IDs corresponding to the embeddings.
    reduced_embeddings : numpy.ndarray
        Two-dimensional embeddings.
    metadatas : sequence of dict
        Paper metadata.
    topics : sequence of int
        Topic ID per paper.
    max_title_length : int, optional
        Maximum title length stored in a point (default 100).

    Returns
    -------
    list of dict
        One point per paper.
    """
    points: List[Dict[str, Any]] = []
    for index, paper_id in enumerate(paper_ids):
        point: Dict[str, Any] = {
            "id": paper_id,
            "x": float(reduced_embeddings[index, 0]),
            "y": float(reduced_embeddings[index, 1]),
            "cluster": int(topics[index]),
        }
        if reduced_embeddings.shape[1] > 2:
            point["z"] = float(reduced_embeddings[index, 2])

        metadata = metadatas[index] if index < len(metadatas) else {}
        title = str(metadata.get("title") or "")
        point["title"] = _truncate_title(title, max_title_length)
        point["year"] = metadata.get("year", "")
        point["conference"] = metadata.get("conference", "")
        point["session"] = metadata.get("session", "")
        point["location"] = metadata.get("primary_location") or metadata.get("location") or ""
        point["locations"] = list(metadata.get("locations") or [])
        points.append(point)
    return points


def _median_center_from_coordinates(coordinates: np.ndarray) -> Dict[str, float]:
    """Return the per-axis median of a 2-D coordinate array."""
    median = np.median(coordinates, axis=0)
    center = {"x": float(median[0]), "y": float(median[1])}
    if len(median) > 2:
        center["z"] = float(median[2])
    return center


def build_cluster_centers(
    reduced_embeddings: np.ndarray,
    topics: Sequence[int],
) -> Dict[int, Dict[str, float]]:
    """
    Compute robust flat cluster centers in the reduced space.

    Centers use the per-axis median so that a few outlier papers do not drag
    the cluster marker away from the main mass of points.

    Parameters
    ----------
    reduced_embeddings : numpy.ndarray
        Two-dimensional embeddings.
    topics : sequence of int
        Topic ID per paper.

    Returns
    -------
    dict
        Mapping from cluster ID to center coordinates.
    """
    topic_array = np.asarray(topics)
    centers: Dict[int, Dict[str, float]] = {}
    for cluster_id in sorted(int(label) for label in np.unique(topic_array) if int(label) >= 0):
        cluster_points = reduced_embeddings[topic_array == cluster_id]
        centers[cluster_id] = _median_center_from_coordinates(cluster_points)
    return centers


def build_cluster_centers_from_points(points: Sequence[Mapping[str, Any]]) -> Dict[int, Dict[str, float]]:
    """
    Recompute flat cluster centers from cached visualization points.

    Parameters
    ----------
    points : sequence of mapping
        Cached points with ``cluster``, ``x``, and optional ``y``/``z`` keys.

    Returns
    -------
    dict
        Mapping from non-negative cluster ID to per-axis median coordinates.
    """
    groups: Dict[int, List[List[float]]] = defaultdict(list)
    has_z = any("z" in point for point in points)
    for point in points:
        cluster_id = int(point.get("cluster", -1))
        if cluster_id < 0:
            continue
        coordinates = [float(point["x"]), float(point.get("y", 0.0))]
        if has_z:
            coordinates.append(float(point.get("z", 0.0)))
        groups[cluster_id].append(coordinates)

    centers: Dict[int, Dict[str, float]] = {}
    for cluster_id, coordinate_rows in groups.items():
        centers[cluster_id] = _median_center_from_coordinates(np.asarray(coordinate_rows, dtype=float))
    return centers


def _aggregate_keywords(member_topics: Sequence[int], flat_keywords: Dict[int, List[str]]) -> List[str]:
    """Aggregate up to ten unique keywords from member flat topics."""
    keywords: List[str] = []
    seen = set()
    for topic_id in member_topics:
        for keyword in flat_keywords.get(int(topic_id), []):
            if keyword in seen:
                continue
            seen.add(keyword)
            keywords.append(keyword)
            if len(keywords) >= 10:
                return keywords
    return keywords


def _center_from_points(points: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Return the median x/y coordinate for a list of points."""
    if not points:
        return {"x": 0.0, "y": 0.0}
    return {
        "x": float(np.median([point["x"] for point in points])),
        "y": float(np.median([point["y"] for point in points])),
    }


def attach_tree_centers(
    tree: Sequence[Mapping[str, Any]],
    points: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Attach precomputed median centers to every hierarchy merge node.

    Each tree node stores the per-axis median of the points belonging to its
    member flat topics. Caching these lets ``build_hierarchy_levels`` derive
    every slider level without rescanning all points or recomputing medians.

    Parameters
    ----------
    tree : sequence of mapping
        Hierarchy merge-tree rows with ``id`` and ``topics`` keys.
    points : sequence of mapping
        Cached points with ``cluster``, ``x``, and optional ``y``/``z`` keys.

    Returns
    -------
    list of dict
        The tree nodes with an added ``center`` mapping.
    """
    points_by_topic: Dict[int, List[List[float]]] = defaultdict(list)
    has_z = any("z" in point for point in points)
    for point in points:
        topic_id = int(point.get("cluster", -1))
        if topic_id < 0:
            continue
        point_coordinates = [float(point["x"]), float(point.get("y", 0.0))]
        if has_z:
            point_coordinates.append(float(point.get("z", 0.0)))
        points_by_topic[topic_id].append(point_coordinates)

    enriched: List[Dict[str, Any]] = []
    for node in tree:
        member_topics = [int(topic) for topic in node.get("topics") or [] if int(topic) >= 0]
        coordinates: List[List[float]] = []
        for topic_id in member_topics:
            coordinates.extend(points_by_topic.get(topic_id, []))
        node_copy = dict(node)
        if coordinates:
            node_copy["center"] = _median_center_from_coordinates(np.asarray(coordinates, dtype=float))
        enriched.append(node_copy)
    return enriched


def build_hierarchy_tree(
    hierarchical_topics: Any,
    parent_labels: Mapping[int, str],
    parent_descriptions: Mapping[int, str] | None = None,
) -> List[Dict[str, Any]]:
    """
    Build the full BERTopic merge tree for the clustering cache.

    Parameters
    ----------
    hierarchical_topics : pandas.DataFrame
        Output of ``topic_model.hierarchical_topics()``.
    parent_labels : mapping
        Parent ID to label mapping.
    parent_descriptions : mapping, optional
        Parent ID to descriptive-sentence mapping.

    Returns
    -------
    list of dict
        One row per merge node, sorted in BERTopic order (parent ID
        descending). Each row has ``id``, ``topics``, ``label``,
        ``description``, ``distance``, ``child_left_id``, and
        ``child_right_id`` keys.
    """
    if hierarchical_topics is None or len(hierarchical_topics) == 0:
        return []

    parent_labels = parent_labels or {}
    parent_descriptions = parent_descriptions or {}
    tree: List[Dict[str, Any]] = []
    for row in hierarchical_topics.itertuples(index=False):
        parent_id = int(row.Parent_ID)
        raw_name = str(getattr(row, "Parent_Name", "") or "")
        if parent_id in parent_labels:
            label = parent_labels[parent_id]
            description = parent_descriptions.get(parent_id, "")
        else:
            label, description = split_label_description(raw_name)
        tree.append(
            {
                "id": parent_id,
                "topics": [int(topic) for topic in (getattr(row, "Topics", None) or [])],
                "label": str(label or "").replace("_", " "),
                "description": str(description or ""),
                "distance": float(getattr(row, "Distance", 0.0)),
                "child_left_id": int(getattr(row, "Child_Left_ID", -1)),
                "child_right_id": int(getattr(row, "Child_Right_ID", -1)),
            }
        )

    tree.sort(key=lambda node: int(node["id"]), reverse=True)
    return tree


def _int_keyed_map(values: Mapping[Any, Any] | None) -> Dict[int, Any]:
    """Normalize cached JSON mappings to integer keys."""
    if not values:
        return {}
    return {int(key): value for key, value in values.items()}


def build_hierarchy_levels(
    tree: Sequence[Mapping[str, Any]],
    flat_topic_ids: Sequence[int],
    flat_labels: Mapping[int, str],
    flat_keywords: Mapping[int, List[str]],
    statistics: Dict[str, Any],
    points: Sequence[Dict[str, Any]],
    flat_descriptions: Mapping[int, str] | None = None,
    flat_centers: Mapping[int, Mapping[str, float]] | None = None,
    nr_levels: int = DEFAULT_NR_LEVELS,
    level_scale: str = DEFAULT_LEVEL_SCALE,
    recompute_centers: bool = False,
) -> Dict[str, Any]:
    """
    Build the compact hierarchy payload used by the slider and MCP.

    By default cluster centers are read from the precomputed flat centers and
    tree-node centers stored in the clustering cache. Set ``recompute_centers``
    when the supplied points are a filtered subset (e.g. by location), in which
    case the median is computed from those points.

    Parameters
    ----------
    tree : sequence of mapping
        Full merge tree stored in the clustering cache.
    flat_topic_ids : sequence of int
        Flat topic IDs (excluding outliers).
    flat_labels : mapping
        Flat topic labels.
    flat_keywords : mapping
        Flat topic keywords.
    statistics : dict
        Flat cluster statistics.
    points : sequence of dict
        Visualization points.
    flat_descriptions : mapping, optional
        Flat topic ID to descriptive-sentence mapping.
    flat_centers : mapping, optional
        Precomputed flat cluster centers keyed by flat topic ID.
    nr_levels : int, optional
        Requested number of hierarchy levels (default 10).
    level_scale : str, optional
        Level scaling mode (default ``"linear"``).
    recompute_centers : bool, optional
        Compute median centers from ``points`` instead of reading the cache.

    Returns
    -------
    dict
        Hierarchy payload with ``levels`` and metadata.

    Raises
    ------
    ClusteringError
        If cached centers are required but missing.
    """
    level_mappings = build_level_mappings_from_tree(tree, flat_topic_ids, nr_levels, level_scale)
    cluster_sizes = _int_keyed_map(statistics.get("cluster_sizes"))
    flat_labels = _int_keyed_map(flat_labels)
    flat_keywords = _int_keyed_map(flat_keywords)
    flat_descriptions = _int_keyed_map(flat_descriptions)
    flat_centers = _int_keyed_map(flat_centers)
    label_by_id = {int(node["id"]): str(node.get("label") or "") for node in tree}
    description_by_id = {int(node["id"]): str(node.get("description") or "") for node in tree}
    node_centers = {int(node["id"]): node.get("center") for node in tree if node.get("center")}

    points_by_cluster: Dict[int, List[Dict[str, Any]]] = {}

    levels: List[Dict[str, Any]] = []
    for level_mapping in level_mappings:
        mapping = level_mapping["mapping"]
        if recompute_centers:
            points_by_cluster = defaultdict(list)
            for point in points:
                cluster_id = int(mapping.get(int(point["cluster"]), int(point["cluster"])))
                if cluster_id >= 0:
                    points_by_cluster[cluster_id].append(point)

        members_by_cluster: Dict[int, List[int]] = defaultdict(list)
        for topic_id in flat_topic_ids:
            cluster_id = int(mapping.get(topic_id, topic_id))
            if cluster_id >= 0:
                members_by_cluster[cluster_id].append(int(topic_id))

        clusters: List[Dict[str, Any]] = []
        for cluster_id, members in members_by_cluster.items():
            if cluster_id in flat_labels:
                label = flat_labels[cluster_id]
                description = flat_descriptions.get(cluster_id, "")
            else:
                label = label_by_id.get(cluster_id, f"Cluster {cluster_id}") or f"Cluster {cluster_id}"
                description = description_by_id.get(cluster_id, "")

            center: Mapping[str, float] | None
            if recompute_centers:
                center = _center_from_points(points_by_cluster.get(cluster_id, []))
            elif cluster_id in flat_centers:
                center = flat_centers[cluster_id]
            else:
                center = node_centers.get(cluster_id)
            if not center:
                raise ClusteringError(
                    "Cached clustering hierarchy is missing precomputed centers; "
                    "run 'python scripts/backfill_cluster_centers_median.py' "
                    "or regenerate with 'abstracts-explorer clustering cache populate --force'."
                )

            clusters.append(
                {
                    "id": int(cluster_id),
                    "label": label,
                    "description": description,
                    "size": int(sum(cluster_sizes.get(member, 0) for member in members)),
                    "center": dict(center),
                    "keywords": _aggregate_keywords(members, flat_keywords),
                    "source_topic_count": len(members),
                }
            )

        clusters.sort(key=lambda cluster: (-cluster["size"], cluster["id"]))
        levels.append(
            {
                "level": int(level_mapping["level"]),
                "max_distance": float(level_mapping["max_distance"]),
                "clusters": clusters,
                "mapping": {str(key): int(value) for key, value in mapping.items() if int(key) >= 0},
            }
        )

    return {
        "method": "bertopic",
        "nr_levels": len(levels),
        "level_scale": level_scale,
        "n_topics": int(statistics.get("n_clusters", 0)),
        "levels": levels,
    }


def derive_hierarchy_levels(
    results: Mapping[str, Any],
    nr_levels: int = DEFAULT_NR_LEVELS,
    level_scale: str = DEFAULT_LEVEL_SCALE,
    recompute_centers: bool = False,
) -> Dict[str, Any]:
    """
    Derive hierarchy levels from the cached tree.

    Cluster centers are read from the precomputed flat centers and tree-node
    centers stored in the cache. Set ``recompute_centers`` when the points have
    been filtered (e.g. by location) and the cached centers no longer apply.

    Parameters
    ----------
    results : mapping
        Cached clustering results.
    nr_levels : int, optional
        Requested number of hierarchy levels (default 10).
    level_scale : str, optional
        Level scaling mode (default ``"linear"``).
    recompute_centers : bool, optional
        Recompute median centers from ``results["points"]`` instead of using
        cached centers.

    Returns
    -------
    dict
        Hierarchy payload with derived ``levels``.

    Raises
    ------
    ClusteringError
        If the cached results do not contain a hierarchy tree, or cached
        centers are required but missing.
    """
    hierarchy = results.get("hierarchy") or {}
    if "tree" not in hierarchy:
        raise ClusteringError(
            "Cached clustering results are missing their hierarchy tree; "
            "regenerate them with 'abstracts-explorer clustering cache populate --force'."
        )

    tree = hierarchy.get("tree") or []
    statistics = results.get("statistics") or {}
    cluster_sizes = statistics.get("cluster_sizes") or {}
    flat_topic_ids = sorted(int(cluster_id) for cluster_id in cluster_sizes if int(cluster_id) >= 0)
    return build_hierarchy_levels(
        tree=tree,
        flat_topic_ids=flat_topic_ids,
        flat_labels=_int_keyed_map(results.get("cluster_labels")),
        flat_keywords=_int_keyed_map(results.get("cluster_keywords")),
        statistics=statistics,
        points=results.get("points") or [],
        flat_descriptions=_int_keyed_map(results.get("cluster_descriptions")),
        flat_centers=_int_keyed_map(results.get("cluster_centers")),
        nr_levels=nr_levels,
        level_scale=level_scale,
        recompute_centers=recompute_centers,
    )


def build_clustering_results(
    paper_ids: Sequence[str],
    reduced_embeddings: np.ndarray,
    metadatas: Sequence[Dict[str, Any]],
    topics: Sequence[int],
    flat_labels: Dict[int, str],
    flat_keywords: Dict[int, List[str]],
    hierarchical_topics: Any,
    parent_labels: Mapping[int, str],
    flat_descriptions: Dict[int, str] | None = None,
    parent_descriptions: Mapping[int, str] | None = None,
) -> Dict[str, Any]:
    """
    Build the complete clustering result payload.

    Parameters
    ----------
    paper_ids : sequence of str
        Paper IDs.
    reduced_embeddings : numpy.ndarray
        Two-dimensional embeddings.
    metadatas : sequence of dict
        Paper metadata.
    topics : sequence of int
        Topic ID per paper.
    flat_labels : dict
        Flat topic labels.
    flat_keywords : dict
        Flat topic keywords.
    hierarchical_topics : pandas.DataFrame
        Output of ``topic_model.hierarchical_topics()``.
    parent_labels : dict
        BERTopic parent ID to parent label mapping.
    flat_descriptions : dict, optional
        Flat topic ID to descriptive-sentence mapping.
    parent_descriptions : dict, optional
        Parent ID to descriptive-sentence mapping.

    Returns
    -------
    dict
        Clustering results with points, statistics, flat fields, and the full
        hierarchy tree.
    """
    statistics = build_statistics(topics)
    points = build_points(paper_ids, reduced_embeddings, metadatas, topics)
    centers = build_cluster_centers(reduced_embeddings, topics)
    flat_descriptions = flat_descriptions or {}
    tree = build_hierarchy_tree(hierarchical_topics, parent_labels, parent_descriptions)
    tree = attach_tree_centers(tree, points)

    return {
        "points": points,
        "statistics": statistics,
        "n_dimensions": int(reduced_embeddings.shape[1]),
        "cluster_centers": centers,
        "cluster_labels": flat_labels,
        "cluster_descriptions": flat_descriptions,
        "cluster_keywords": flat_keywords,
        "hierarchy": {
            "method": "bertopic",
            "n_topics": int(statistics.get("n_clusters", 0)),
            "tree": tree,
        },
    }


def export_results_to_json(results: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    Export clustering results to a JSON file.

    Parameters
    ----------
    results : dict
        Clustering results.
    output_path : str or Path
        Destination path.

    Raises
    ------
    ClusteringError
        If writing fails.
    """
    try:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        logger.info("Exported clustering results to %s", path)
    except Exception as exc:
        raise ClusteringError(f"Failed to export to JSON: {exc}") from exc


def results_have_flat_labels(results: Dict[str, Any]) -> bool:
    """
    Return whether every flat cluster in *results* has a label.

    Parameters
    ----------
    results : dict
        Clustering results.

    Returns
    -------
    bool
        ``True`` when all flat cluster labels are present.
    """
    labels = results.get("cluster_labels") or {}
    cluster_sizes = (results.get("statistics") or {}).get("cluster_sizes") or {}
    return all(str(cluster_id) in labels or cluster_id in labels for cluster_id in cluster_sizes)

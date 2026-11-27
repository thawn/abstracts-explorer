"""
Conference topic tools.
"""

import json
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

from abstracts_explorer.clustering import ClusteringError, derive_hierarchy_levels
from abstracts_explorer.mcp import cache, resources, shared_dependencies
from abstracts_explorer.mcp.errors import ClusterAnalysisError, ToolError
from abstracts_explorer.mcp.server import mcp

logger = logging.getLogger(__name__)

MCP_HIERARCHY_LEVEL = 6


def analyze_cluster_topics(
    cm: shared_dependencies.ClusteringManager,
    db: shared_dependencies.DatabaseManager,
    cluster_id: int,
    use_llm: bool = False,
) -> Dict[str, Any]:
    """
    Analyze a single flat topic (cluster) and return a concise summary.

    Parameters
    ----------
    cm : shared_dependencies.ClusteringManager
        Clustering manager with loaded data.
    db : shared_dependencies.DatabaseManager
        Database manager for paper metadata.
    cluster_id : int
        Internal cluster ID to analyze.
    use_llm : bool, optional
        Retained for compatibility; unused.

    Returns
    -------
    dict
        Dictionary containing ``topic`` (descriptive sentence when available),
        ``label`` (short legend label), ``paper_count``, ``keywords``, and
        ``sample_titles``.
    """
    if cm.cluster_labels is None or cm.paper_ids is None or cm.metadatas is None:
        raise ClusterAnalysisError("Clustering data not loaded. Call load_embeddings() and fit() first.")

    label_names = cm.cluster_label_names or {}
    label_descriptions = getattr(cm, "cluster_descriptions", None) or {}
    cluster_kws = cm.cluster_keywords or {}

    cluster_indices = [i for i, label in enumerate(cm.cluster_labels) if label == cluster_id]
    if not cluster_indices:
        fields = _topic_fields(label_names.get(cluster_id), label_descriptions.get(cluster_id))
        return {
            **fields,
            "paper_count": 0,
            "keywords": cluster_kws.get(cluster_id, []),
            "sample_titles": [],
        }

    sample_titles: list[str] = []
    for idx in cluster_indices:
        if len(sample_titles) >= 5:
            break
        title = cm.metadatas[idx].get("title", "")
        if title:
            sample_titles.append(title)

    fields = _topic_fields(label_names.get(cluster_id), label_descriptions.get(cluster_id))
    return {
        **fields,
        "paper_count": len(cluster_indices),
        "keywords": cluster_kws.get(cluster_id, []),
        "sample_titles": sample_titles,
    }


def _topic_fields(label: Any, description: Any) -> Dict[str, Any]:
    """Build the primary topic text plus the short label fallback."""
    short_label = str(label) if label else None
    long_label = str(description).strip() if description else ""
    return {
        "topic": long_label or short_label,
        "label": short_label,
    }


def _points_by_cluster(cached: Dict[str, Any], mapping: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    """Group cached points by their cluster ID at the selected level."""
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for point in cached.get("points", []):
        flat_topic = point.get("cluster", -1)
        cluster_id = mapping.get(str(flat_topic), flat_topic)
        try:
            cluster_id_int = int(cluster_id)
        except (TypeError, ValueError):
            continue
        if cluster_id_int >= 0:
            grouped[cluster_id_int].append(point)
    return grouped


def _sample_titles(points: List[Dict[str, Any]], limit: int = 5) -> List[str]:
    """Return up to *limit* non-empty titles from points."""
    titles: List[str] = []
    for point in points:
        title = point.get("title")
        if title:
            titles.append(str(title))
            if len(titles) >= limit:
                break
    return titles


def _select_hierarchy_level(cached: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the fixed MCP hierarchy level, clamped to available levels."""
    try:
        hierarchy = derive_hierarchy_levels(cached)
    except ClusteringError as exc:
        raise ToolError(str(exc)) from exc
    levels = hierarchy.get("levels") or []
    if not levels:
        return None
    return levels[min(MCP_HIERARCHY_LEVEL, len(levels) - 1)]


def _build_hierarchy_topics(cached: Dict[str, Any], level: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build topic entries from one cached hierarchy level."""
    mapping = level.get("mapping") or {}
    points_by_cluster = _points_by_cluster(cached, mapping)
    topics: List[Dict[str, Any]] = []
    for cluster in level.get("clusters", []):
        cluster_id = int(cluster.get("id", -1))
        cluster_points = points_by_cluster.get(cluster_id, [])
        fields = _topic_fields(
            cluster.get("label") or f"Topic {cluster_id}",
            cluster.get("description"),
        )
        topics.append(
            {
                **fields,
                "paper_count": int(cluster.get("size", len(cluster_points))),
                "keywords": list(cluster.get("keywords") or []),
                "sample_titles": _sample_titles(cluster_points),
                "source_topic_count": int(cluster.get("source_topic_count", 1)),
            }
        )
    topics.sort(key=lambda topic: topic["paper_count"], reverse=True)
    return topics


def _build_flat_topics(cached: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build topic entries from flat cached cluster data."""
    points_by_cluster = _points_by_cluster(cached, {})
    stats = cached.get("statistics") or {}
    cluster_sizes = stats.get("cluster_sizes") or {}
    labels = cached.get("cluster_labels") or {}
    descriptions = cached.get("cluster_descriptions") or {}
    keywords = cached.get("cluster_keywords") or {}

    topics: List[Dict[str, Any]] = []
    for cluster_id, size in cluster_sizes.items():
        try:
            cluster_id_int = int(cluster_id)
        except (TypeError, ValueError):
            continue
        cluster_points = points_by_cluster.get(cluster_id_int, [])
        label = labels.get(str(cluster_id), labels.get(cluster_id_int, f"Topic {cluster_id_int}"))
        description = descriptions.get(str(cluster_id), descriptions.get(cluster_id_int, ""))
        topics.append(
            {
                **_topic_fields(label, description),
                "paper_count": int(size),
                "keywords": list(keywords.get(str(cluster_id), keywords.get(cluster_id_int, [])) or []),
                "sample_titles": _sample_titles(cluster_points),
                "source_topic_count": 1,
            }
        )
    topics.sort(key=lambda topic: topic["paper_count"], reverse=True)
    return topics


def _topic_sizes(topics: List[Dict[str, Any]]) -> Dict[str, int]:
    """Build a topic-label to paper-count mapping."""
    return {str(topic["topic"]): int(topic["paper_count"]) for topic in topics}


def _get_conference_topics_for_single_conference(
    conference: str,
    years: Optional[List[int]] = None,
    collection_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrieve the main research topics for a single conference.

    Looks up pre-computed clustering results from the database cache and
    returns a hierarchy-aware topic summary.  Embeddings are not loaded.

    Parameters
    ----------
    conference : str
        Conference name (e.g. "NeurIPS", "ICLR", or "NeurIPS 2025").
    years : list of int, optional
        Filter by publication years.
    collection_name : str, optional
        Name of ChromaDB collection (uses config default if not provided).

    Returns
    -------
    dict
        Topics result dict with ``topic_sizes`` and ``topics`` keys.

    Raises
    ------
    ToolError
        If no cached clustering data exists for the requested scope.
    """
    config = shared_dependencies.get_config()
    collection_name = collection_name or config.collection_name

    conference, extracted_year = shared_dependencies.parse_conference_year(conference)
    if extracted_year is not None:
        if years is None:
            years = [extracted_year]
        elif extracted_year not in years:
            years = sorted(years + [extracted_year])

    with resources._get_mcp_resources(with_embeddings=False) as (_, db):
        cached = cache._lookup_clustering_cache(db, config, conference, years, collection_name)

        if not cached:
            raise ToolError(
                f"No pre-computed clustering data available for conference "
                f"'{conference}'"
                + (f" years={years}" if years else "")
                + ". Run 'abstracts-explorer clustering cache populate' first."
            )

        stats = cached.get("statistics") or {}
        level = _select_hierarchy_level(cached)
        if level is not None:
            topics = _build_hierarchy_topics(cached, level)
            return {
                "conference": conference,
                "total_papers": int(stats.get("total_papers", 0)),
                "n_topics": len(topics),
                "hierarchy_level": int(level.get("level", 0)),
                "hierarchy_max_distance": float(level.get("max_distance", 0.0)),
                "topic_sizes": _topic_sizes(topics),
                "topics": topics,
            }

        logger.warning("Cached clustering has no hierarchy; falling back to flat topics")
        topics = _build_flat_topics(cached)
        return {
            "conference": conference,
            "total_papers": int(stats.get("total_papers", 0)),
            "n_topics": len(topics),
            "topic_sizes": _topic_sizes(topics),
            "topics": topics,
        }


@mcp.tool()
def get_conference_topics(
    conferences: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    collection_name: Optional[str] = None,
) -> str:
    """
    Get the main research topics of a conference.

    Returns the key research topics covered at the conference, each with a
    descriptive name, representative keywords, paper count, and example
    paper titles.  A conference must be specified.

    When multiple conferences are provided, each conference is analyzed
    individually and results are combined.

    Parameters
    ----------
    conferences : list of str, optional
        Conference names (e.g. ["NeurIPS"]).
        Required - returns an error when not provided.
    years : list of int, optional
        Filter by publication years.
    collection_name : str, optional
        Name of ChromaDB collection (uses config default if not provided).

    Returns
    -------
    str
        JSON string containing the conference topics analysis.
    """
    try:
        if not conferences:
            raise ToolError(
                "A conference must be specified for topic analysis. Please provide conferences parameter."
            )

        all_results: List[Dict[str, Any]] = []
        for conf in conferences:
            result = _get_conference_topics_for_single_conference(
                conference=conf,
                years=years,
                collection_name=collection_name,
            )
            all_results.append(result)

        if len(all_results) == 1:
            return json.dumps(all_results[0], indent=2)

        return json.dumps({"conference_results": all_results}, indent=2)

    except ToolError:
        raise
    except Exception as e:
        logger.exception("Failed to get conference topics: %s", e)
        raise ToolError("Conference topic analysis failed; check server logs for details.") from e

"""
Conference topic tools.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from abstracts_explorer.mcp import cache, resources, shared_dependencies
from abstracts_explorer.mcp.errors import ClusterAnalysisError
from abstracts_explorer.mcp.server import mcp

logger = logging.getLogger(__name__)


def analyze_cluster_topics(
    cm: shared_dependencies.ClusteringManager,
    db: shared_dependencies.DatabaseManager,
    cluster_id: int,
    use_llm: bool = False,
) -> Dict[str, Any]:
    """
    Analyze a single topic (cluster) and return a concise summary.

    Each cluster represents a conference topic.  The returned dictionary
    is designed to be consumed directly by an LLM — field names use the
    word *topic* instead of *cluster* so the model does not need to know
    about the underlying clustering implementation.

    Parameters
    ----------
    cm : shared_dependencies.ClusteringManager
        Clustering manager with loaded data
    db : shared_dependencies.DatabaseManager
        Database manager for paper metadata
    cluster_id : int
        Internal cluster ID to analyze
    use_llm : bool, optional
        Whether to use LLM for topic extraction (default: False)

    Returns
    -------
    dict
        Dictionary containing:
        - topic: Human-readable topic name (or ``None``)
        - paper_count: Number of papers in this topic
        - keywords: Representative keywords for the topic
        - sample_titles: A few example paper titles
    """
    if cm.cluster_labels is None or cm.paper_ids is None or cm.metadatas is None:
        raise ClusterAnalysisError("Clustering data not loaded. Call load_embeddings() and cluster() first.")

    label_names = cm.cluster_label_names or {}
    cluster_kws = cm.cluster_keywords or {}

    # Find papers in this cluster
    cluster_indices = [i for i, label in enumerate(cm.cluster_labels) if label == cluster_id]

    if not cluster_indices:
        return {
            "topic": label_names.get(cluster_id),
            "paper_count": 0,
            "keywords": cluster_kws.get(cluster_id, []),
            "sample_titles": [],
        }

    # Extract sample titles
    sample_titles: list[str] = []
    for idx in cluster_indices:
        if len(sample_titles) >= 5:
            break
        title = cm.metadatas[idx].get("title", "")
        if title:
            sample_titles.append(title)

    return {
        "topic": label_names.get(cluster_id),
        "paper_count": len(cluster_indices),
        "keywords": cluster_kws.get(cluster_id, []),
        "sample_titles": sample_titles,
    }


def _get_conference_topics_for_single_conference(
    conference: str,
    years: Optional[List[int]] = None,
    collection_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrieve the main research topics for a single conference.

    Looks up pre-computed clustering results from the database cache and
    returns a topic-centric summary.  Returns an error dict when no cached
    results exist for the requested conference/year combination.

    If the *conference* string contains a trailing year
    (e.g. ``"NeurIPS 2025"``), the year is extracted and merged into *years*
    so that the cache lookup matches entries stored under the plain
    conference name.

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
        Topics result dict with ``"topic_sizes"`` and ``"topics"``
        keys, or an ``"error"`` key if no cache is available.
    """
    config = shared_dependencies.get_config()
    collection_name = collection_name or config.collection_name

    # Parse year from conference name if present (e.g. "NeurIPS 2025" → "NeurIPS", 2025)
    conference, extracted_year = shared_dependencies.parse_conference_year(conference)
    if extracted_year is not None:
        if years is None:
            years = [extracted_year]
        elif extracted_year not in years:
            years = sorted(years + [extracted_year])

    with resources._get_clustering_resources(collection_name) as (cm, db):
        cached = cache._lookup_clustering_cache(db, config, conference, years, collection_name)

        if not cached:
            return {
                "error": (
                    f"No pre-computed clustering data available for conference "
                    f"'{conference}'"
                    + (f" years={years}" if years else "")
                    + ". Run 'abstracts-explorer clustering pre-generate' first."
                ),
            }

        # Reconstruct shared_dependencies.ClusteringManager state from cached results
        cm.load_embeddings(conferences=[conference], years=years)
        cache._apply_cached_cluster_labels(cm, cached)

        stats = cm.get_cluster_statistics()

        # Build topic_sizes with human-readable names, sorted by size descending
        label_names = cm.cluster_label_names or {}
        named_sizes = {label_names.get(cid, f"Topic {cid}"): size for cid, size in stats["cluster_sizes"].items()}
        topic_sizes = dict(sorted(named_sizes.items(), key=lambda x: x[1], reverse=True))

        topics = []
        for cluster_id in range(stats["n_clusters"]):
            topic = analyze_cluster_topics(cm, db, cluster_id)
            topics.append(topic)

        # Sort topics by paper_count descending
        topics.sort(key=lambda t: t["paper_count"], reverse=True)

        return {
            "conference": conference,
            "total_papers": stats["total_papers"],
            "n_topics": stats["n_clusters"],
            "topic_sizes": topic_sizes,
            "topics": topics,
        }


@mcp.tool()
def get_conference_topics(
    conferences: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    collection_name: Optional[str] = None,
    **kwargs,
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
        Required – returns an error when not provided.
    years : list of int, optional
        Filter by publication years.
    collection_name : str, optional
        Name of ChromaDB collection (uses config default if not provided).
    **kwargs
        Ignored (for backwards compatibility with old tool schemas).

    Returns
    -------
    str
        JSON string containing the conference topics analysis.
    """
    try:
        if not conferences:
            return json.dumps(
                {
                    "error": (
                        "A conference must be specified for topic analysis. " "Please provide conferences parameter."
                    )
                },
                indent=2,
            )

        all_results: List[Dict[str, Any]] = []
        for conf in conferences:
            result = _get_conference_topics_for_single_conference(
                conference=conf,
                years=years,
                collection_name=collection_name,
            )
            all_results.append(result)

        # If only one conference, return its result directly
        if len(all_results) == 1:
            return json.dumps(all_results[0], indent=2)

        # Multiple conferences – combine
        return json.dumps({"conference_results": all_results}, indent=2)

    except Exception as e:
        logger.error(f"Failed to get conference topics: {str(e)}")
        return json.dumps({"error": str(e)}, indent=2)

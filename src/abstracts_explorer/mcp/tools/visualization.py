"""
Cluster visualization tools.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from abstracts_explorer.mcp import cache, resources, shared_dependencies
from abstracts_explorer.mcp.errors import ToolError
from abstracts_explorer.mcp.server import mcp

logger = logging.getLogger(__name__)


def _point_cluster_id(point: Dict[str, Any]) -> Optional[int]:
    """Return the integer cluster ID of a point, if available."""
    raw = point.get("cluster", point.get("cluster_id"))
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _merge_statistics(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge per-conference cluster statistics."""
    if not payloads:
        return {}
    combined = dict(payloads[0].get("statistics", {}))
    for payload in payloads[1:]:
        stats = payload.get("statistics", {})
        combined["n_clusters"] = combined.get("n_clusters", 0) + stats.get("n_clusters", 0)
        combined["total_papers"] = combined.get("total_papers", 0) + stats.get("total_papers", 0)
    return combined


def _merge_conference_payloads(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Combine cached clustering payloads from multiple conferences.

    With a single conference the flat cluster metadata is passed through
    unchanged.  With multiple conferences cluster IDs are renumbered
    sequentially so identical IDs from different conferences do not merge.

    Parameters
    ----------
    payloads : list of dict
        Cached clustering payloads (one per conference).

    Returns
    -------
    dict
        Combined points, statistics, centers, labels, and descriptions.
    """
    if len(payloads) <= 1:
        payload = payloads[0] if payloads else {}
        return {
            "points": payload.get("points", []),
            "statistics": payload.get("statistics", {}),
            "cluster_centers": payload.get("cluster_centers", {}),
            "cluster_labels": payload.get("cluster_labels", {}),
            "cluster_descriptions": payload.get("cluster_descriptions", {}),
        }

    points: List[Dict[str, Any]] = []
    cluster_centers: Dict[str, Any] = {}
    cluster_labels: Dict[str, Any] = {}
    cluster_descriptions: Dict[str, Any] = {}
    next_id = 0

    for payload in payloads:
        old_ids = set()
        for point in payload.get("points", []):
            cid = _point_cluster_id(point)
            if cid is not None and cid >= 0:
                old_ids.add(cid)
        for key in payload.get("cluster_centers", {}):
            try:
                cid = int(key)
            except (TypeError, ValueError):
                continue
            if cid >= 0:
                old_ids.add(cid)

        remap: Dict[int, int] = {}
        for cid in sorted(old_ids):
            remap[cid] = next_id
            next_id += 1

        for point in payload.get("points", []):
            new_point = dict(point)
            cid = _point_cluster_id(point)
            if cid is not None and cid in remap:
                if "cluster" in new_point:
                    new_point["cluster"] = remap[cid]
                if "cluster_id" in new_point:
                    new_point["cluster_id"] = remap[cid]
            points.append(new_point)

        for key, value in payload.get("cluster_centers", {}).items():
            try:
                cid = int(key)
            except (TypeError, ValueError):
                continue
            if cid in remap:
                cluster_centers[str(remap[cid])] = value

        for key, value in payload.get("cluster_labels", {}).items():
            try:
                cid = int(key)
            except (TypeError, ValueError):
                continue
            if cid in remap:
                cluster_labels[str(remap[cid])] = value

        for key, value in payload.get("cluster_descriptions", {}).items():
            try:
                cid = int(key)
            except (TypeError, ValueError):
                continue
            if cid in remap:
                cluster_descriptions[str(remap[cid])] = value

    return {
        "points": points,
        "statistics": _merge_statistics(payloads),
        "cluster_centers": cluster_centers,
        "cluster_labels": cluster_labels,
        "cluster_descriptions": cluster_descriptions,
    }


@mcp.tool()
def get_cluster_visualization(
    conferences: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    collection_name: Optional[str] = None,
) -> str:
    """
    Retrieve pre-computed visualization data for clustered embeddings.

    This tool looks up cached clustering results (pre-generated via CLI)
    and returns data suitable for visualization.  A conference must be
    specified.

    When multiple conferences are provided, each conference is looked up
    individually and results are combined.

    The chat frontend can use the returned data to generate a plot with
    plotly.js showing the clusters, including cluster-center markers.

    Parameters
    ----------
    conferences : list of str, optional
        Conference names to retrieve clusters for (e.g. ["NeurIPS"]).
        Required – returns an error when not provided.
    years : list of int, optional
        Filter by publication years.
    collection_name : str, optional
        Name of ChromaDB collection.

    Returns
    -------
    str
        JSON string containing visualization data with points, cluster centers,
        labels, descriptions, and statistics.
    """
    try:
        if not conferences:
            raise ToolError(
                "A conference must be specified for cluster visualization. Please provide conferences parameter."
            )

        config = shared_dependencies.get_config()
        collection_name = collection_name or config.collection_name

        payloads: List[Dict[str, Any]] = []

        for conf in conferences:
            # Parse year from conference name if present (e.g. "NeurIPS 2025")
            parsed_conf, extracted_year = shared_dependencies.parse_conference_year(conf)
            vis_years = list(years) if years else None
            if extracted_year is not None:
                if vis_years is None:
                    vis_years = [extracted_year]
                elif extracted_year not in vis_years:
                    vis_years = sorted(vis_years + [extracted_year])

            with resources._get_mcp_resources(with_embeddings=False) as (_, db):
                cached = cache._lookup_clustering_cache(db, config, parsed_conf, vis_years, collection_name)

            if not cached:
                raise ToolError(
                    f"No pre-computed clustering data available for conference "
                    f"'{parsed_conf}'"
                    + (f" years={vis_years}" if vis_years else "")
                    + ". Run 'abstracts-explorer clustering pre-generate' first."
                )

            payloads.append(cached)

        combined = _merge_conference_payloads(payloads)
        all_points = combined["points"]

        result = {
            "n_dimensions": 2,
            "n_points": len(all_points),
            "statistics": combined["statistics"],
            "points": all_points[:1000],
            "cluster_centers": combined["cluster_centers"],
            "cluster_labels": combined["cluster_labels"],
            "cluster_descriptions": combined["cluster_descriptions"],
        }

        return json.dumps(result, indent=2)

    except ToolError:
        raise
    except Exception as e:
        logger.exception("Failed to generate cluster visualization: %s", e)
        raise ToolError("Cluster visualization failed; check server logs for details.") from e

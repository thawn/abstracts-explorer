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
    plotly.js showing the clusters.

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
        JSON string containing visualization data with points, clusters, and statistics.
    """
    try:
        if not conferences:
            raise ToolError(
                "A conference must be specified for cluster visualization. Please provide conferences parameter."
            )

        config = shared_dependencies.get_config()
        collection_name = collection_name or config.collection_name

        all_points: List[Dict[str, Any]] = []
        combined_stats: Dict[str, Any] = {}

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

            all_points.extend(cached.get("points", []))
            if not combined_stats:
                combined_stats = cached.get("statistics", {})
            else:
                # Merge stats across conferences
                combined_stats["n_clusters"] = combined_stats.get("n_clusters", 0) + cached.get("statistics", {}).get(
                    "n_clusters", 0
                )
                combined_stats["total_papers"] = combined_stats.get("total_papers", 0) + cached.get(
                    "statistics", {}
                ).get("total_papers", 0)

        result = {
            "n_dimensions": 2,
            "n_points": len(all_points),
            "statistics": combined_stats,
            "points": all_points[:1000],
        }

        return json.dumps(result, indent=2)

    except ToolError:
        raise
    except Exception as e:
        logger.exception("Failed to generate cluster visualization: %s", e)
        raise ToolError("Cluster visualization failed; check server logs for details.") from e

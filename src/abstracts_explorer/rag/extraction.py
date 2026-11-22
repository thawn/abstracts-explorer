"""
Extract papers and visualizations from tool results.
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ExtractionMixin:
    """Post-process tool results into papers and visualizations."""

    @staticmethod
    def _extract_papers(tool_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract papers from tool results.

        Parameters
        ----------
        tool_results : list of dict
            Tool results with 'name' and 'raw_result' keys.

        Returns
        -------
        list of dict
            Extracted papers.
        """
        papers: List[Dict[str, Any]] = []
        seen_uids: set[str] = set()
        for tr in tool_results:
            if tr["name"] in (
                "search_papers",
                "analyze_topic_relevance",
                "summarize_interesting_papers",
                "recommend_related_papers",
            ):
                try:
                    result_json = json.loads(tr["raw_result"])
                    if "papers" in result_json:
                        for paper in result_json.get("papers", []):
                            uid = str(paper.get("uid", ""))
                            if uid and uid in seen_uids:
                                continue
                            papers.append(paper)
                            if uid:
                                seen_uids.add(uid)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse papers from {tr['name']} result")
        return papers

    @staticmethod
    def _extract_visualizations(tool_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract visualization data from tool results.

        Converts raw MCP tool output from ``get_topic_evolution`` and
        ``get_cluster_visualization`` into lightweight chart descriptors
        that the frontend can render with Plotly.

        Parameters
        ----------
        tool_results : list of dict
            Tool results with 'name' and 'raw_result' keys.

        Returns
        -------
        list of dict
            Visualization descriptors.  Each dict contains a ``type`` key
            (``"topic_evolution"`` or ``"cluster_visualization"``) and the
            data needed to draw the chart.
        """
        visualizations: List[Dict[str, Any]] = []
        topic_evolution_viz: Optional[Dict[str, Any]] = None
        for tr in tool_results:
            try:
                data = json.loads(tr["raw_result"])
            except (json.JSONDecodeError, TypeError, KeyError):
                continue

            if not isinstance(data, dict):
                continue

            if "error" in data:
                continue

            if tr["name"] == "get_topic_evolution":
                conference_data = data.get("conference_data", {})
                topic_name = data.get("topic", "")
                if conference_data and topic_name:
                    # Merge into existing topic_evolution visualization if one exists
                    if topic_evolution_viz is not None:
                        topic_evolution_viz["topics"].append(topic_name)
                        topic_evolution_viz["conference_data"][topic_name] = conference_data
                    else:
                        topic_evolution_viz = {
                            "type": "topic_evolution",
                            "topics": [topic_name],
                            "conference_data": {topic_name: conference_data},
                        }
                        visualizations.append(topic_evolution_viz)

            elif tr["name"] == "get_cluster_visualization":
                points = data.get("points", [])
                if points:
                    visualizations.append(
                        {
                            "type": "cluster_visualization",
                            "points": points,
                            "statistics": data.get("statistics", {}),
                            "cluster_centers": data.get("cluster_centers", {}),
                            "cluster_labels": data.get("cluster_labels", {}),
                            "cluster_descriptions": data.get("cluster_descriptions", {}),
                        }
                    )

        return visualizations

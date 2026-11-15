"""
Topic-relevance analysis tools.
"""

import json
import logging
from collections import Counter
from typing import Any, Dict, List, Optional

from abstracts_explorer.mcp import deps, resources
from abstracts_explorer.mcp.server import mcp

logger = logging.getLogger(__name__)


@mcp.tool()
def analyze_topic_relevance(
    topic: str,
    distance_threshold: float = 1.1,
    conferences: Optional[list[str]] = None,
    years: Optional[list[int]] = None,
    collection_name: Optional[str] = None,
) -> str:
    """
    Analyze the relevance of a topic by counting papers within a specified distance in embedding space.

    This tool measures topic relevance by finding papers semantically similar to the topic
    within a specified Euclidean distance threshold. It's useful for identifying how prevalent
    or relevant a research topic is at a conference.
    A conference must be specified.

    Parameters
    ----------
    topic : str
        The topic or research question to analyze (e.g., "Uncertainty quantification",
        "Graph neural networks", "Transformer architectures")
    distance_threshold : float, optional
        Maximum Euclidean distance in embedding space to consider papers relevant (default: 1.1).
        Lower values mean stricter matching. Typical range: 0.5-2.0 for normalized embeddings.
    conferences : list of str, optional
        Conference names to filter by (e.g., ["NeurIPS", "ICLR"]).
        Required – returns an error when not provided.
    years : list of int, optional
        Filter results to specific years (e.g., [2024, 2025])
    collection_name : str, optional
        Name of ChromaDB collection (uses config default if not provided)

    Returns
    -------
    str
        JSON string containing:
        - topic: The topic analyzed
        - distance_threshold: Distance threshold applied
        - total_papers: Number of papers found within distance
        - total_considered: Total number of filtered papers considered
        - conferences: Conferences represented (with counts)
        - years: Years represented (with counts)
        - sample_papers: Sample of closest papers with titles and distances
        - relevance_score: Percentage of filtered papers within distance (0-100)

    Examples
    --------
    Topic: "Uncertainty quantification"
    Result: 75 papers found within distance 1.1
    Interpretation: High relevance - this is a significant topic at the conference

    Query: "Quantum machine learning"
    Result: 3 papers found within distance 1.1
    Interpretation: Low relevance - emerging or niche topic
    """
    try:
        if not conferences:
            return json.dumps(
                {
                    "error": (
                        "A conference must be specified for topic relevance analysis. "
                        "Please provide conferences parameter."
                    )
                },
                indent=2,
            )

        config = deps.get_config()
        collection_name = collection_name or config.collection_name

        with resources._get_mcp_resources(collection_name) as (em, db):
            assert em is not None

            # Find papers within distance
            logger.info(f"Analyzing relevance for topic: {topic}")
            logger.info(f"Distance threshold: {distance_threshold}")
            if conferences:
                logger.info(f"Filtering by conferences: {conferences}")
            if years:
                logger.info(f"Filtering by years: {years}")

            candidate_ids = db.resolve_paper_uids(conferences=conferences, years=years)
            if candidate_ids:
                result_data = em.query_ids_within_distance(
                    query=topic,
                    distance_threshold=distance_threshold,
                    ids=candidate_ids,
                )
                papers_by_uid = {paper["uid"]: paper for paper in db.get_papers_by_uids(result_data["ids"])}
                papers: List[Dict[str, Any]] = []
                for uid, distance in zip(result_data["ids"], result_data["distances"]):
                    paper = papers_by_uid.get(uid)
                    if paper is None:
                        continue
                    paper["distance"] = distance
                    papers.append(paper)
                total_considered = result_data.get("total_considered", len(papers))
            else:
                papers = []
                total_considered = 0

            total_papers = len(papers)

            # Count by conference
            conference_counts: Counter[str] = Counter()
            year_counts: Counter[int] = Counter()
            for paper in papers:
                if paper.get("conference"):
                    conference_counts[paper["conference"]] += 1
                if paper.get("year"):
                    year_counts[paper["year"]] += 1

            # Calculate relevance score (0-100 scale)
            # Ratio of papers within distance threshold to total filtered papers
            if total_considered > 0:
                relevance_score = (total_papers / total_considered) * 100
            else:
                relevance_score = 0

            # Get sample papers (top 5 closest)
            sample_papers = []
            for paper in papers[:5]:
                sample_papers.append(
                    {
                        "title": paper.get("title", ""),
                        "year": paper.get("year"),
                        "conference": paper.get("conference", ""),
                        "distance": paper.get("distance"),
                    }
                )

            # Build result
            result = {
                "topic": topic,
                "distance_threshold": distance_threshold,
                "filters": {
                    "conferences": conferences,
                    "years": years,
                },
                "total_papers": total_papers,
                "total_considered": total_considered,
                "relevance_score": round(relevance_score, 1),
                "conferences": dict(sorted(conference_counts.items(), key=lambda x: (-x[1], x[0]))),
                "years": dict(sorted(year_counts.items(), key=lambda x: x[0])),
                "sample_papers": sample_papers,
                "closest_distance": papers[0].get("distance") if papers else None,
            }

            return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Failed to analyze topic relevance: {str(e)}")
        return json.dumps({"error": str(e)}, indent=2)

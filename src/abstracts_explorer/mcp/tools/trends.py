"""
Topic-evolution tools.
"""

import json
import logging
from typing import Any, Dict, Optional

from abstracts_explorer.embeddings import QueryEmbedding
from abstracts_explorer.mcp import resources, shared_dependencies
from abstracts_explorer.mcp.server import mcp

logger = logging.getLogger(__name__)


@mcp.tool()
def get_topic_evolution(
    topic_keywords: str,
    conferences: Optional[list[str]] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    distance_threshold: float = 1.1,
    collection_name: Optional[str] = None,
) -> str:
    """
    Analyze how topics have evolved over the years for one or more conferences.

    For each conference and year in the given range, this tool uses
    ``shared_dependencies.EmbeddingsManager.query_ids_within_distance()`` to count how many
    papers are semantically close to the topic keywords.  It also
    computes the relative percentage of matching papers with respect to
    the total number of papers for that conference and year.
    At least one conference must be specified.

    The chat frontend can use the returned data to generate a plot with
    plotly.js showing the topic evolution over time.

    Parameters
    ----------
    topic_keywords : str
        Keywords describing the topic to analyze (e.g., "transformers attention")
    conferences : list of str, optional
        Conference names to analyze (e.g., ["NeurIPS", "ICLR"]).
        Required – returns an error when not provided.
    start_year : int, optional
        Start year for analysis (inclusive)
    end_year : int, optional
        End year for analysis (inclusive)
    distance_threshold : float, optional
        Maximum Euclidean distance in embedding space to consider papers
        relevant (default: 1.1).  Lower values mean stricter matching.
    collection_name : str, optional
        Name of ChromaDB collection

    Returns
    -------
    str
        JSON string containing topic evolution analysis with per-conference
        year_counts, year_relative (percentage), and year_totals.
    """
    try:
        if not conferences:
            return json.dumps(
                {
                    "error": (
                        "A conference must be specified for topic evolution analysis. "
                        "Please provide conferences parameter."
                    )
                },
                indent=2,
            )
        config = shared_dependencies.get_config()
        collection_name = collection_name or config.collection_name

        with resources._get_mcp_resources(collection_name) as (em, db):
            assert em is not None
            logger.info(f"Analyzing topic evolution for: {topic_keywords}")
            logger.info(f"Conferences: {conferences}")
            logger.info(f"Distance threshold: {distance_threshold}")

            # Embed the query once here and reuse for every (conference, year) pair
            # to avoid redundant LLM API calls.
            query_embedding = QueryEmbedding(topic_keywords, em)

            conference_data: Dict[str, Dict[str, Any]] = {}
            total_papers = 0
            all_years: set[int] = set()

            for conference in conferences:
                # Determine year range from database for this conference
                available_years = db.get_years(conference=conference)
                if start_year is not None:
                    available_years = [y for y in available_years if y >= start_year]
                if end_year is not None:
                    available_years = [y for y in available_years if y <= end_year]

                logger.info(f"Conference: {conference}, years: {available_years}")

                year_counts: Dict[int, int] = {year: 0 for year in available_years}
                year_relative: Dict[int, float] = {}
                year_totals: Dict[int, int] = {}
                year_distribution: Dict[int, list] = {year: [] for year in available_years}

                if available_years:
                    # Query once per conference (all years at once) instead of
                    # once per year.  The latter scales with the number of years
                    # and can exceed reverse-proxy timeouts for conferences with
                    # many years (e.g. IGARSS).
                    candidate_ids = db.resolve_paper_uids(
                        conferences=[conference],
                        years=available_years,
                    )
                    if candidate_ids:
                        matches = em.query_ids_within_distance(
                            query=query_embedding,
                            distance_threshold=distance_threshold,
                            ids=candidate_ids,
                        )
                        summaries = {
                            paper["uid"]: paper
                            for paper in db.get_paper_summaries_by_uids([uid for uid, _ in matches])
                        }

                        for uid, distance in matches:
                            paper = summaries.get(uid)
                            if paper is None:
                                continue
                            paper_year = paper.get("year")
                            if paper_year is None:
                                continue
                            try:
                                paper_year = int(paper_year)
                            except (TypeError, ValueError):
                                continue
                            if paper_year not in year_counts:
                                continue

                            year_counts[paper_year] += 1

                            # Collect sample papers (top 3 closest).  Results are
                            # sorted by distance, so the first three per year are
                            # the closest matches.
                            if len(year_distribution[paper_year]) < 3:
                                year_distribution[paper_year].append(
                                    {
                                        "title": paper.get("title", ""),
                                        "session": paper.get("session", ""),
                                        "distance": distance,
                                    }
                                )

                for year in available_years:
                    count = year_counts[year]
                    total_papers += count

                    # Get total papers for this conference+year for relative percentage
                    stats = db.get_stats(year=year, conference=conference)
                    total_for_year = stats["total_papers"]
                    year_totals[year] = total_for_year
                    if total_for_year > 0:
                        year_relative[year] = round((count / total_for_year) * 100, 2)
                    else:
                        year_relative[year] = 0.0

                all_years.update(year_counts.keys())

                # Sort by year
                sorted_years = sorted(year_counts.keys())

                conference_data[conference] = {
                    "year_counts": dict(sorted(year_counts.items())),
                    "year_relative": dict(sorted(year_relative.items())),
                    "year_totals": dict(sorted(year_totals.items())),
                    "papers_by_year": {
                        year: {
                            "count": year_counts[year],
                            "relative_percent": year_relative[year],
                            "total_for_year": year_totals[year],
                            "sample_papers": year_distribution[year],
                        }
                        for year in sorted_years
                    },
                }

            sorted_all_years = sorted(all_years)

            # Build result
            result: Dict[str, Any] = {
                "topic": topic_keywords,
                "conferences": conferences,
                "distance_threshold": distance_threshold,
                "total_papers": total_papers,
                "year_range": {
                    "start": min(sorted_all_years) if sorted_all_years else None,
                    "end": max(sorted_all_years) if sorted_all_years else None,
                },
                "conference_data": conference_data,
            }

            return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Failed to get topic evolution: {str(e)}")
        return json.dumps({"error": str(e)}, indent=2)

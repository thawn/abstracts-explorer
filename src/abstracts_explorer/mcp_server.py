"""
MCP Server for Cluster Analysis
================================

This module provides a Model Context Protocol (MCP) server that exposes
tools for analyzing clustered embeddings. The server enables LLM-based
assistants to answer questions about conference paper topics, trends,
and developments.

Features:
- Get most frequently mentioned topics from clusters
- Analyze topic evolution over years
- Find recent developments in specific topics
- Generate cluster visualizations
"""

import logging
import json
import re
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Iterator
from collections import Counter

from mcp.server.mcpserver import MCPServer

from abstracts_explorer.embeddings import EmbeddingsManager, build_chromadb_where_clause
from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.clustering import ClusteringManager
from abstracts_explorer.config import get_config
from abstracts_explorer.interesting_papers import (
    recommend_related_papers as _recommend_related_papers,
    summarize_interesting_papers as _summarize_interesting_papers,
)

logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = MCPServer("Abstracts Explorer Cluster Analysis")


class ClusterAnalysisError(Exception):
    """Exception raised for cluster analysis errors."""

    pass


def load_clustering_data(
    collection_name: Optional[str] = None,
) -> tuple[ClusteringManager, DatabaseManager]:
    """
    Load clustering data and database.

    Parameters
    ----------
    collection_name : str, optional
        Name of the ChromaDB collection

    Returns
    -------
    tuple[ClusteringManager, DatabaseManager]
        Clustering manager and database manager instances

    Raises
    ------
    ClusterAnalysisError
        If loading fails
    """
    config = get_config()

    # Use config defaults if not provided
    collection_name = collection_name or config.collection_name

    em: Optional[EmbeddingsManager] = None
    db: Optional[DatabaseManager] = None

    try:
        # Initialize embeddings manager
        em = EmbeddingsManager(
            collection_name=collection_name,
        )

        # Initialize database manager
        db = DatabaseManager()

        # Initialize clustering manager
        cm = ClusteringManager(em, db)

        return cm, db

    except Exception as e:
        if em is not None:
            em.close()
        if db is not None:
            db.close()
        raise ClusterAnalysisError(f"Failed to load clustering data: {str(e)}") from e


@contextmanager
def _get_mcp_resources(
    collection_name: Optional[str] = None,
    *,
    with_embeddings: bool = True,
) -> Iterator[Tuple[Optional[EmbeddingsManager], DatabaseManager]]:
    """
    Provide MCP tool resources and guarantee they are closed.

    Parameters
    ----------
    collection_name : str, optional
        ChromaDB collection name.  Uses the configured default when embeddings
        are requested.
    with_embeddings : bool, optional
        Whether to create an ``EmbeddingsManager`` (default: ``True``).

    Yields
    ------
    tuple
        ``(embeddings_manager, database_manager)``; the embeddings manager is
        ``None`` when *with_embeddings* is ``False``.
    """
    em: Optional[EmbeddingsManager] = None
    db: Optional[DatabaseManager] = None
    try:
        if with_embeddings:
            resolved_collection = collection_name or get_config().collection_name
            em = EmbeddingsManager(collection_name=resolved_collection)
        db = DatabaseManager()
        yield em, db
    finally:
        if em is not None:
            em.close()
        if db is not None:
            db.close()


@contextmanager
def _get_clustering_resources(
    collection_name: Optional[str] = None,
) -> Iterator[Tuple[ClusteringManager, DatabaseManager]]:
    """
    Provide clustering resources and guarantee they are closed.

    Parameters
    ----------
    collection_name : str, optional
        ChromaDB collection name.

    Yields
    ------
    tuple
        ``(clustering_manager, database_manager)``.
    """
    cm, db = load_clustering_data(collection_name)
    try:
        yield cm, db
    finally:
        cm.embeddings_manager.close()
        db.close()


def _apply_cached_cluster_labels(
    cm: ClusteringManager,
    cached_results: Dict[str, Any],
) -> None:
    """
    Restore ``cluster_labels``, ``cluster_label_names``, and
    ``cluster_keywords`` on *cm* from cached clustering results.

    Parameters
    ----------
    cm : ClusteringManager
        Clustering manager with embeddings already loaded
        (``cm.paper_ids`` must be populated).
    cached_results : dict
        Cached results dict containing a ``"points"`` list where each
        element has ``"id"`` (or ``"paper_id"``) and ``"cluster"`` keys.
        May also contain ``"cluster_labels"`` (cluster name dict) and
        ``"cluster_keywords"`` (TF-IDF keyword dict).
    """
    import numpy as np

    point_id_to_cluster: Dict[str, int] = {}
    for point in cached_results.get("points", []):
        pid = point.get("id") or point.get("paper_id", "")
        point_id_to_cluster[pid] = point.get("cluster", -1)

    current_ids = cm.paper_ids or []
    cm.cluster_labels = np.array([point_id_to_cluster.get(pid, -1) for pid in current_ids])

    # Restore cluster label names (LLM-generated or TF-IDF-based names)
    if cached_results.get("cluster_labels"):
        cm.cluster_label_names = {int(k): v for k, v in cached_results["cluster_labels"].items()}

    # Restore TF-IDF cluster keywords
    if cached_results.get("cluster_keywords"):
        cm.cluster_keywords = {int(k): v for k, v in cached_results["cluster_keywords"].items()}


def analyze_cluster_topics(
    cm: ClusteringManager,
    db: DatabaseManager,
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
    cm : ClusteringManager
        Clustering manager with loaded data
    db : DatabaseManager
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


def _parse_conference_year(conference: str) -> Tuple[str, Optional[int]]:
    """
    Parse a trailing year from a conference name.

    LLMs often combine the conference name and year into a single string
    (e.g. ``"NeurIPS 2025"``).  This helper splits them so the conference
    name matches the database/cache entries which store the name and year
    separately.

    Parameters
    ----------
    conference : str
        Conference name, possibly with a trailing 4-digit year.

    Returns
    -------
    tuple of (str, int or None)
        ``(conference_name, year)`` — *year* is ``None`` when no trailing
        year was found.

    Examples
    --------
    >>> _parse_conference_year("NeurIPS 2025")
    ('NeurIPS', 2025)
    >>> _parse_conference_year("ICLR")
    ('ICLR', None)
    """
    match = re.match(r"^(.+)\s+(\d{4})$", conference.strip())
    if match:
        return match.group(1), int(match.group(2))
    return conference, None


def _lookup_clustering_cache(
    db: "DatabaseManager",
    config: Any,
    conference: str,
    years: Optional[List[int]],
) -> Any:
    """
    Look up pre-computed clustering results from the database cache.

    Tries an exact match first.  When *years* is non-empty and no exact
    match is found, retries without the year filter so that an all-years
    cache entry can serve as a fallback.

    Parameters
    ----------
    db : DatabaseManager
        Open database connection.
    config : object
        Configuration object (needs ``embedding_model``).
    conference : str
        Conference name (already parsed, no trailing year).
    years : list of int or None
        Year filter.

    Returns
    -------
    dict or None
        Cached clustering results, or ``None`` when nothing matches.
    """
    clustering_params: Dict[str, Any] = {
        "linkage": "ward",
        "distance_threshold": 150.0,
    }

    # Determine single year for column-based lookup
    cache_year = years[0] if years and len(years) == 1 else None

    cached = db.get_clustering_cache(
        embedding_model=config.embedding_model,
        reduction_method="tsne",
        n_components=2,
        clustering_method="agglomerative",
        n_clusters=None,
        clustering_params=clustering_params,
        conference=conference,
        year=cache_year,
    )

    # Fallback: if per-year cache not found, try the all-years cache
    if not cached and years:
        cached = db.get_clustering_cache(
            embedding_model=config.embedding_model,
            reduction_method="tsne",
            n_components=2,
            clustering_method="agglomerative",
            n_clusters=None,
            clustering_params=clustering_params,
            conference=conference,
            year=None,
        )

    return cached


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
    config = get_config()
    collection_name = collection_name or config.collection_name

    # Parse year from conference name if present (e.g. "NeurIPS 2025" → "NeurIPS", 2025)
    conference, extracted_year = _parse_conference_year(conference)
    if extracted_year is not None:
        if years is None:
            years = [extracted_year]
        elif extracted_year not in years:
            years = sorted(years + [extracted_year])

    with _get_clustering_resources(collection_name) as (cm, db):
        cached = _lookup_clustering_cache(db, config, conference, years)

        if not cached:
            return {
                "error": (
                    f"No pre-computed clustering data available for conference "
                    f"'{conference}'"
                    + (f" years={years}" if years else "")
                    + ". Run 'abstracts-explorer clustering pre-generate' first."
                ),
            }

        # Reconstruct ClusteringManager state from cached results
        cm.load_embeddings(conferences=[conference], years=years)
        _apply_cached_cluster_labels(cm, cached)

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
    ``EmbeddingsManager.find_papers_within_distance()`` to count how many
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
        config = get_config()
        collection_name = collection_name or config.collection_name

        with _get_mcp_resources(collection_name) as (em, db):
            assert em is not None
            logger.info(f"Analyzing topic evolution for: {topic_keywords}")
            logger.info(f"Conferences: {conferences}")
            logger.info(f"Distance threshold: {distance_threshold}")

            # Embed the query once here and reuse for every (conference, year) pair
            # to avoid redundant LLM API calls.
            query_embedding = em.generate_embedding(topic_keywords)

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

                year_counts: Dict[int, int] = {}
                year_relative: Dict[int, float] = {}
                year_totals: Dict[int, int] = {}
                year_distribution: Dict[int, list] = {}

                for year in available_years:
                    result_data = em.find_papers_within_distance(
                        database=db,
                        query=topic_keywords,
                        distance_threshold=distance_threshold,
                        conferences=[conference],
                        years=[year],
                        query_embedding=query_embedding,
                    )

                    count = result_data["count"]
                    year_counts[year] = count
                    total_papers += count

                    # Get total papers for this conference+year for relative percentage
                    stats = db.get_stats(year=year, conference=conference)
                    total_for_year = stats["total_papers"]
                    year_totals[year] = total_for_year
                    if total_for_year > 0:
                        year_relative[year] = round((count / total_for_year) * 100, 2)
                    else:
                        year_relative[year] = 0.0

                    # Collect sample papers (top 3 closest)
                    sample_papers = []
                    for paper in result_data["papers"][:3]:
                        sample_papers.append(
                            {
                                "title": paper.get("title", ""),
                                "session": paper.get("session", ""),
                                "distance": paper.get("distance"),
                            }
                        )
                    year_distribution[year] = sample_papers

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


@mcp.tool()
def search_papers(
    topic_keywords: str,
    years: Optional[List[int]] = None,
    n_results: int = 10,
    conference: Optional[str] = None,
    where: Optional[Dict[str, Any]] = None,
    collection_name: Optional[str] = None,
) -> str:
    """
    Search for papers on a specific topic.

    This tool searches for the most relevant papers about a topic, optionally
    filtered by specific years.  A conference must be specified.

    Parameters
    ----------
    topic_keywords : str
        Keywords describing the topic (e.g., "large language models")
    years : list of int, optional
        List of specific years to filter by (e.g., [2024, 2025]). If None, searches all years.
    n_results : int, optional
        Number of papers to return (default: 10)
    conference : str, optional
        Conference name to filter by (e.g., "NeurIPS", "ICLR").
        Required – returns an error when not provided.
    where : dict, optional
        Custom ChromaDB WHERE clause for filtering results by metadata.
        Supports ChromaDB query operators like $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin.
        Logical operators $and, $or are also supported.
        Examples: ``{"year": 2025}``, ``{"session": {"$in": ["Oral Session 1", "Oral Session 2"]}}``,
        ``{"$and": [{"year": {"$gte": 2024}}, {"conference": "NeurIPS"}]}``.
        Note: If 'conference' parameter is provided, it will be merged with this WHERE clause.
    collection_name : str, optional
        Name of ChromaDB collection

    Returns
    -------
    str
        JSON string containing search results
    """
    try:
        if not conference:
            return json.dumps(
                {
                    "error": (
                        "A conference must be specified for paper search. " "Please provide conference parameter."
                    )
                },
                indent=2,
            )
        config = get_config()
        collection_name = collection_name or config.collection_name

        with _get_mcp_resources(collection_name) as (em, db):
            assert em is not None

            # Validate custom WHERE clause and merge conference/year filters
            try:
                where = build_chromadb_where_clause(
                    conferences=[conference],
                    years=years,
                    base_where=where,
                )
            except ValueError as e:
                logger.error(f"Invalid WHERE clause: {str(e)}")
                return json.dumps({"error": f"Invalid WHERE clause: {str(e)}"}, indent=2)

            # Search for papers
            search_desc = f"papers from {years}" if years else "papers"
            logger.info(f"Searching for {search_desc} about: {topic_keywords}")
            if where:
                logger.info(f"Applying WHERE filter: {where}")
            if years:
                logger.info(f"Year filter: {years}")

            papers = db.search_papers_semantic(
                topic_keywords,
                limit=n_results,
                years=years,
                conferences=[conference],
                where=where,
                embeddings_manager=em,
            )

            result = {
                "topic": topic_keywords,
                "conference": conference,
                "years_filter": years,
                "papers_found": len(papers),
                "papers": papers,
            }

            return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Failed to search papers: {str(e)}")
        return json.dumps({"error": str(e)}, indent=2)


@mcp.tool()
def get_paper_details(
    title: Optional[str] = None,
    paper_id: Optional[str] = None,
    conference: Optional[str] = None,
    year: Optional[int] = None,
    limit: int = 5,
) -> str:
    """
    Get detailed information about papers from the database. Use for folow-up questions after searching for papers using semantic search.

    Returns full paper metadata including authors, URLs, PDF links, session info,
    keywords, awards, and other details stored in the database.

    At least one of *title* or *paper_id* must be provided.

    Parameters
    ----------
    title : str, optional
        Title or partial title to search for (case-insensitive).
    paper_id : str, optional
        Unique paper identifier (uid or original conference/OpenReview ID).
        When provided, performs an exact lookup and ignores *title*.
    conference : str, optional
        Filter results by conference name (e.g., "NeurIPS", "ICLR").
        Only applied when searching by *title*.
    year : int, optional
        Filter results by publication year.
        Only applied when searching by *title*.
    limit : int, optional
        Maximum number of papers to return when searching by title (default: 5).

    Returns
    -------
    str
        JSON string with fields:

        - ``papers_found`` – number of papers returned
        - ``papers`` – list of paper dicts, each containing:
          title, authors (list), abstract, url, paper_pdf_url,
          poster_image_url, session, room_name, starttime, endtime,
          poster_position, keywords, award, year, conference, original_id
    """
    if not title and not paper_id:
        return json.dumps(
            {"error": "Provide at least one of 'title' or 'paper_id' to look up a paper."},
            indent=2,
        )

    try:
        with _get_mcp_resources(with_embeddings=False) as (_, db):
            result_papers: List[Dict[str, Any]] = []

            if paper_id:
                # Exact lookup by uid or original_id (returns at most one paper)
                paper = db.get_paper_by_original_id_or_uid(paper_id)
                if paper is not None:
                    result_papers = [paper]

            if not result_papers and title:
                # Keyword search on title with optional conference/year filters
                result_papers = db.search_papers(
                    keyword=title,
                    conferences=[conference] if conference else None,
                    years=[year] if year is not None else None,
                    limit=limit,
                )

            result = {
                "papers_found": len(result_papers),
                "papers": result_papers,
            }

            return json.dumps(result, indent=2, default=str)

    except Exception as e:
        logger.error(f"Failed to get paper details: {str(e)}")
        return json.dumps({"error": str(e)}, indent=2)


@mcp.tool()
def summarize_interesting_papers(
    paper_ids: Optional[List[str]] = None,
    priorities: Optional[Dict[str, Any]] = None,
    conference: Optional[str] = None,
    years: Optional[List[int]] = None,
) -> str:
    """
    Summarise the papers a user marked as interesting.

    The web UI forwards the user's star ratings with each chat request, so
    this tool can summarise the papers they rated as interesting.  The
    response contains rating statistics, conferences, years, sessions, search
    terms, top keywords, highest-rated papers, and compact paper records.

    Use this tool when the user asks for a summary of the papers they marked,
    starred, liked, or rated as interesting.

    Parameters
    ----------
    paper_ids : list of str, optional
        UIDs of the interesting papers.  When omitted, the keys of
        *priorities* are used.
    priorities : dict, optional
        Mapping of paper UID to a dict with ``priority`` and optional
        ``searchTerm`` fields.  This is normally supplied automatically from
        the web UI.
    conference : str, optional
        Conference filter.  A trailing year (e.g. ``"ICLR 2026"``) is split
        automatically.
    years : list of int, optional
        Year filter (e.g. ``[2026]``).

    Returns
    -------
    str
        JSON string containing the interesting-paper summary.
    """
    if not paper_ids and not priorities:
        return json.dumps(
            {
                "error": (
                    "No interesting papers were provided. Mark papers with stars in the " "web UI and try again."
                )
            },
            indent=2,
        )

    try:
        with _get_mcp_resources(with_embeddings=False) as (_, db):
            result = _summarize_interesting_papers(
                database=db,
                priorities=priorities,
                paper_ids=paper_ids,
                conference=conference,
                years=years,
            )
            return json.dumps(result, indent=2, default=str)
    except Exception as e:
        logger.error(f"Failed to summarize interesting papers: {str(e)}")
        return json.dumps({"error": str(e)}, indent=2)


@mcp.tool()
def recommend_related_papers(
    paper_ids: Optional[List[str]] = None,
    priorities: Optional[Dict[str, Any]] = None,
    conferences: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    n_results: int = 10,
    exclude_ids: Optional[List[str]] = None,
    collection_name: Optional[str] = None,
) -> str:
    """
    Recommend further papers based on the papers a user marked as interesting.

    The tool computes a rating-weighted embedding centroid of the interesting
    papers, finds the nearest papers in embedding space, and excludes papers
    the user already rated.  Each recommendation includes a relevance score
    and short reasons (shared keywords, authors, session, or semantic
    closeness).

    Use this tool when the user asks for suggestions, recommendations, or
    related papers based on their interesting/starred papers.

    Parameters
    ----------
    paper_ids : list of str, optional
        UIDs of the interesting papers.  When omitted, the keys of
        *priorities* are used.
    priorities : dict, optional
        Mapping of paper UID to a dict with ``priority`` and optional
        ``searchTerm`` fields.  This is normally supplied automatically from
        the web UI.
    conferences : list of str, optional
        Restrict recommendations to these conferences.  When omitted, the
        dominant conference of the interesting papers is used.
    years : list of int, optional
        Restrict recommendations to these years.  When omitted, the years of
        the interesting papers are used.
    n_results : int, optional
        Number of recommendations to return (default: 10).
    exclude_ids : list of str, optional
        Additional paper UIDs to exclude from the results.
    collection_name : str, optional
        Name of the ChromaDB collection (uses config default if not provided).

    Returns
    -------
    str
        JSON string containing recommended papers and filter metadata.
    """
    if not paper_ids and not priorities:
        return json.dumps(
            {
                "error": (
                    "No interesting papers were provided. Mark papers with stars in the " "web UI and try again."
                )
            },
            indent=2,
        )

    config = get_config()
    collection_name = collection_name or config.collection_name

    try:
        with _get_mcp_resources(collection_name) as (em, db):
            assert em is not None
            result = _recommend_related_papers(
                database=db,
                embeddings_manager=em,
                priorities=priorities,
                paper_ids=paper_ids,
                conferences=conferences,
                years=years,
                n_results=n_results,
                exclude_ids=exclude_ids,
            )
            return json.dumps(result, indent=2, default=str)
    except Exception as e:
        logger.error(f"Failed to recommend related papers: {str(e)}")
        return json.dumps({"error": str(e)}, indent=2)


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

        config = get_config()
        collection_name = collection_name or config.collection_name

        with _get_mcp_resources(collection_name) as (em, db):
            assert em is not None

            # Find papers within distance
            logger.info(f"Analyzing relevance for topic: {topic}")
            logger.info(f"Distance threshold: {distance_threshold}")
            if conferences:
                logger.info(f"Filtering by conferences: {conferences}")
            if years:
                logger.info(f"Filtering by years: {years}")

            result_data = em.find_papers_within_distance(
                database=db,
                query=topic,
                distance_threshold=distance_threshold,
                conferences=conferences,
                years=years,
            )

            # Analyze results
            papers = result_data["papers"]
            total_papers = len(papers)
            total_considered = result_data.get("total_considered", total_papers)

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


@mcp.tool()
def get_cluster_visualization(
    conferences: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    output_path: Optional[str] = None,
    collection_name: Optional[str] = None,
    **kwargs,
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
    output_path : str, optional
        Path to save visualization JSON file (optional).
    collection_name : str, optional
        Name of ChromaDB collection.
    **kwargs
        Ignored (for backwards compatibility with old tool schemas).

    Returns
    -------
    str
        JSON string containing visualization data with points, clusters, and statistics.
    """
    try:
        if not conferences:
            return json.dumps(
                {
                    "error": (
                        "A conference must be specified for cluster visualization. "
                        "Please provide conferences parameter."
                    )
                },
                indent=2,
            )

        config = get_config()
        collection_name = collection_name or config.collection_name

        all_points: List[Dict[str, Any]] = []
        combined_stats: Dict[str, Any] = {}

        for conf in conferences:
            # Parse year from conference name if present (e.g. "NeurIPS 2025")
            parsed_conf, extracted_year = _parse_conference_year(conf)
            vis_years = list(years) if years else None
            if extracted_year is not None:
                if vis_years is None:
                    vis_years = [extracted_year]
                elif extracted_year not in vis_years:
                    vis_years = sorted(vis_years + [extracted_year])

            with _get_clustering_resources(collection_name) as (_, db):
                cached = _lookup_clustering_cache(db, config, parsed_conf, vis_years)

            if not cached:
                return json.dumps(
                    {
                        "error": (
                            f"No pre-computed clustering data available for conference "
                            f"'{parsed_conf}'"
                            + (f" years={vis_years}" if vis_years else "")
                            + ". Run 'abstracts-explorer clustering pre-generate' first."
                        ),
                    },
                    indent=2,
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

        # Export if requested
        if output_path:
            import pathlib

            try:
                export_data = {"points": all_points, "statistics": combined_stats}
                pathlib.Path(output_path).write_text(json.dumps(export_data, indent=2))
            except OSError as exc:
                logger.warning(f"Failed to write visualization to {output_path}: {exc}")
                output_path = None

        result = {
            "n_dimensions": 2,
            "n_points": len(all_points),
            "statistics": combined_stats,
            "points": all_points[:1000],  # Limit for MCP response size
            "visualization_saved": output_path is not None,
            "output_path": output_path if output_path else None,
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Failed to generate cluster visualization: {str(e)}")
        return json.dumps({"error": str(e)}, indent=2)


def run_mcp_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    transport: str = "streamable-http",
) -> None:
    """
    Run the MCP server.

    Parameters
    ----------
    host : str, optional
        Host address to bind to (default: "127.0.0.1")
    port : int, optional
        Port to listen on (default: 8000)
    transport : str, optional
        Transport method: 'streamable-http' or 'stdio' (default: 'streamable-http')

    Examples
    --------
    >>> run_mcp_server(host="0.0.0.0", port=8000)
    """
    logger.info(f"Starting MCP server on {host}:{port} with {transport} transport")

    if transport == "stdio":
        # Run with stdio transport (for local CLI integration)
        mcp.run(transport="stdio")
    else:
        # Run with streamable HTTP transport (for HTTP integration)
        mcp.run(transport="streamable-http", host=host, port=port)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run server
    run_mcp_server()

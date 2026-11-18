"""
Paper search and detail tools.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from abstracts_explorer.mcp import resources, shared_dependencies
from abstracts_explorer.mcp.errors import ToolError
from abstracts_explorer.mcp.server import mcp

logger = logging.getLogger(__name__)


@mcp.tool()
def search_papers(
    topic_keywords: str,
    years: Optional[List[int]] = None,
    limit: int = 10,
    conference: Optional[str] = None,
    where_filters: Optional[Dict[str, Any]] = None,
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
    limit : int, optional
        Number of papers to return (default: 10)
    conference : str, optional
        Conference name to filter by (e.g., "NeurIPS", "ICLR").
        Required – returns an error when not provided.
    where_filters : dict, optional
        SQL-style filter dictionary applied to the paper database.
        Supports operators like ``$eq``, ``$ne``, ``$gt``, ``$gte``,
        ``$lt``, ``$lte``, ``$in``, ``$nin``, ``$contains``, plus logical
        operators ``$and``, ``$or``, and ``$not``.
        Examples: ``{"year": 2025}``,
        ``{"session": {"$in": ["Oral Session 1", "Oral Session 2"]}}``,
        ``{"$and": [{"year": {"$gte": 2024}}, {"conference": "NeurIPS"}]}``.
        The ``conference`` parameter is merged with this filter.
    collection_name : str, optional
        Name of ChromaDB collection

    Returns
    -------
    str
        JSON string containing search results
    """
    try:
        limit = max(1, min(int(limit), 50))
        if not conference:
            raise ToolError("A conference must be specified for paper search. Please provide conference parameter.")
        config = shared_dependencies.get_config()
        collection_name = collection_name or config.collection_name

        with resources._get_mcp_resources(collection_name) as (em, db):
            assert em is not None

            search_desc = f"papers from {years}" if years else "papers"
            logger.info(f"Searching for {search_desc} about: {topic_keywords}")
            if where_filters:
                logger.info(f"Applying filters: {where_filters}")
            if years:
                logger.info(f"Year filter: {years}")

            papers, _ = db.search(
                topic_keywords,
                limit=limit,
                years=years,
                conferences=[conference],
                where_filters=where_filters,
                embeddings_manager=em,
                use_embeddings=True,
            )

            result = {
                "topic": topic_keywords,
                "conference": conference,
                "years_filter": years,
                "papers_found": len(papers),
                "papers": papers,
            }

            return json.dumps(result, indent=2)

    except ToolError:
        raise
    except Exception as e:
        logger.exception("Failed to search papers: %s", e)
        raise ToolError("Paper search failed; check server logs for details.") from e


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
        raise ToolError("Provide at least one of 'title' or 'paper_id' to look up a paper.")

    try:
        with resources._get_mcp_resources(with_embeddings=False) as (_, db):
            result_papers: List[Dict[str, Any]] = []

            if paper_id:
                # Exact lookup by uid or original_id (returns at most one paper)
                paper = db.get_paper_by_original_id_or_uid(paper_id)
                if paper is not None:
                    result_papers = [paper]

            if not result_papers and title:
                limit = max(1, min(int(limit), 50))
                # Keyword search on title with optional conference/year filters
                result_papers = db.keyword_search(
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

    except ToolError:
        raise
    except Exception as e:
        logger.exception("Failed to get paper details: %s", e)
        raise ToolError("Paper lookup failed; check server logs for details.") from e

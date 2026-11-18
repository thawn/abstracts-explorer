"""
Interesting-paper summary and recommendation tools.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from abstracts_explorer.mcp import resources, shared_dependencies
from abstracts_explorer.mcp.errors import ToolError
from abstracts_explorer.mcp.server import mcp

logger = logging.getLogger(__name__)


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
        raise ToolError("No interesting papers were provided. Mark papers with stars in the web UI and try again.")

    try:
        with resources._get_mcp_resources(with_embeddings=False) as (_, db):
            result = shared_dependencies._summarize_interesting_papers(
                database=db,
                priorities=priorities,
                paper_ids=paper_ids,
                conference=conference,
                years=years,
            )
            return json.dumps(result, indent=2, default=str)
    except ToolError:
        raise
    except (TypeError, ValueError) as e:
        raise ToolError(str(e)) from e
    except Exception as e:
        logger.exception("Failed to summarize interesting papers: %s", e)
        raise ToolError("Interesting-paper summary failed; check server logs for details.") from e


@mcp.tool()
def recommend_related_papers(
    paper_ids: Optional[List[str]] = None,
    priorities: Optional[Dict[str, Any]] = None,
    conferences: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    limit: int = 10,
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
    limit : int, optional
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
        raise ToolError("No interesting papers were provided. Mark papers with stars in the web UI and try again.")

    config = shared_dependencies.get_config()
    collection_name = collection_name or config.collection_name

    try:
        with resources._get_mcp_resources(collection_name) as (em, db):
            assert em is not None
            result = shared_dependencies._recommend_related_papers(
                database=db,
                embeddings_manager=em,
                priorities=priorities,
                paper_ids=paper_ids,
                conferences=conferences,
                years=years,
                limit=limit,
                exclude_ids=exclude_ids,
            )
            return json.dumps(result, indent=2, default=str)
    except ToolError:
        raise
    except (TypeError, ValueError) as e:
        raise ToolError(str(e)) from e
    except Exception as e:
        logger.exception("Failed to recommend related papers: %s", e)
        raise ToolError("Related-paper recommendation failed; check server logs for details.") from e

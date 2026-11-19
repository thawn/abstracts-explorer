"""
Summarise interesting (rated) papers.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.interesting_papers.constants import ABSTRACT_SNIPPET_LENGTH, MAX_SUMMARY_PAPERS
from abstracts_explorer.interesting_papers.loading import _as_int, load_interesting_papers
from abstracts_explorer.paper_utils import extract_top_keywords
from abstracts_explorer.utils.paper_presentation import truncate_abstract

logger = logging.getLogger(__name__)


def _paper_record(paper: Dict[str, Any], include_abstract: bool = False) -> Dict[str, Any]:
    """Build a compact, JSON-safe paper record."""
    record: Dict[str, Any] = {
        "uid": paper.get("uid"),
        "title": paper.get("title", ""),
        "authors": paper.get("authors") or [],
        "year": paper.get("year"),
        "conference": paper.get("conference", ""),
        "session": paper.get("session", ""),
        "keywords": paper.get("keywords") or [],
        "priority": paper.get("priority", 0),
        "searchTerm": paper.get("searchTerm", "Unknown"),
    }

    poster_position = paper.get("poster_position")
    if poster_position:
        record["poster_position"] = poster_position

    url = paper.get("url")
    if url:
        record["url"] = url

    for field in ("room_name", "starttime", "endtime"):
        value = paper.get(field)
        if value:
            record[field] = value

    if include_abstract:
        abstract = str(paper.get("abstract") or "")
        if abstract:
            record["abstract_snippet"] = truncate_abstract(abstract, ABSTRACT_SNIPPET_LENGTH)

    return record


def summarize_interesting_papers(
    database: DatabaseManager,
    priorities: Optional[Dict[str, Any]] = None,
    paper_ids: Optional[Sequence[str]] = None,
    conference: Optional[str] = None,
    years: Optional[Sequence[int]] = None,
    max_papers: int = MAX_SUMMARY_PAPERS,
) -> Dict[str, Any]:
    """
    Summarise the papers a user marked as interesting.

    Parameters
    ----------
    database : DatabaseManager
        Open database connection.
    priorities : dict, optional
        Raw priority mapping from the web UI.
    paper_ids : sequence of str, optional
        Explicit UID list.  When omitted, the keys of *priorities* are used.
    conference : str, optional
        Conference filter (e.g. ``"ICLR"`` or ``"ICLR 2026"``).
    years : sequence of int, optional
        Year filter.
    max_papers : int, optional
        Maximum number of paper records to include in the response.

    Returns
    -------
    dict
        Structured summary with rating statistics, topics, and paper records.
    """
    loaded = load_interesting_papers(
        database=database,
        priorities=priorities,
        paper_ids=paper_ids,
        conference=conference,
        years=years,
    )
    papers: List[Dict[str, Any]] = loaded["papers"]
    filters = loaded["filters"]

    if not papers:
        return {
            "filters": filters,
            "total_interesting": 0,
            "papers": [],
            "warning": (
                "No interesting papers found"
                + (
                    f" for conference={filters['conferences']} years={filters['years']}"
                    if filters["conferences"] or filters["years"]
                    else ""
                )
                + ". Mark papers with stars in the web UI first."
            ),
            "missing_uids": loaded["missing_uids"],
        }

    rated = [paper for paper in papers if _as_int(paper.get("priority")) > 0]
    rating_counts = Counter(_as_int(paper.get("priority")) for paper in rated)
    average_rating = round(sum(paper["priority"] for paper in rated) / len(rated), 2) if rated else 0.0

    conference_counts = Counter(str(paper.get("conference") or "Unknown") for paper in papers)
    year_counts = Counter(_as_int(paper.get("year")) for paper in papers if paper.get("year"))
    session_counts = Counter(str(paper.get("session") or "No session") for paper in papers)
    search_term_counts = Counter(str(paper.get("searchTerm") or "Unknown") for paper in papers)

    keyword_counts: Counter[str] = Counter()
    for paper in papers:
        for keyword in paper.get("keywords") or []:
            cleaned = str(keyword).strip()
            if cleaned:
                keyword_counts[cleaned] += 1

    top_keywords = [keyword for keyword, _ in keyword_counts.most_common(15)]
    if not top_keywords:
        top_keywords = extract_top_keywords(papers, n_keywords=10)

    top_rated = sorted(
        papers,
        key=lambda paper: (
            _as_int(paper.get("priority")),
            _as_int(paper.get("year")),
        ),
        reverse=True,
    )[:5]

    record_papers = [_paper_record(paper, include_abstract=True) for paper in papers[:max_papers]]
    truncated = len(papers) > len(record_papers)

    return {
        "filters": filters,
        "total_interesting": len(papers),
        "rated_count": len(rated),
        "rating_distribution": {str(rating): rating_counts[rating] for rating in range(5, 0, -1)},
        "average_rating": average_rating,
        "conferences": dict(conference_counts.most_common()),
        "years": {str(year): count for year, count in sorted(year_counts.items())},
        "sessions": [{"session": session, "count": count} for session, count in session_counts.most_common(10)],
        "search_terms": [{"search_term": term, "count": count} for term, count in search_term_counts.most_common(10)],
        "top_keywords": top_keywords,
        "top_rated_papers": [_paper_record(paper) for paper in top_rated],
        "papers": record_papers,
        "papers_truncated": truncated,
        "missing_uids": loaded["missing_uids"],
        "warning": None,
    }

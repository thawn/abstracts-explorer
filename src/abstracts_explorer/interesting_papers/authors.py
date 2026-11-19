"""
Suggest authors to meet based on a user's interesting papers.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.embeddings import EmbeddingsManager
from abstracts_explorer.interesting_papers.constants import (
    DEFAULT_AUTHOR_LIMIT,
    DEFAULT_AUTHOR_PAPER_LIMIT,
    MAX_AUTHOR_LIMIT,
    MAX_AUTHOR_PAPER_LIMIT,
)
from abstracts_explorer.interesting_papers.loading import _as_int, load_interesting_papers
from abstracts_explorer.interesting_papers.recommend import recommend_related_papers

logger = logging.getLogger(__name__)


def _normalise_author(value: Any) -> str:
    """Normalise an author name for aggregation."""
    return " ".join(str(value or "").strip().casefold().split())


def _display_author(value: Any) -> str:
    """Return an author name with normalised whitespace but original casing."""
    return " ".join(str(value or "").strip().split())


def _meeting_paper_record(
    paper: Dict[str, Any],
    *,
    recommendation_score: Optional[float] = None,
    priority: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a compact paper record with the fields needed to find an author."""
    record: Dict[str, Any] = {
        "uid": paper.get("uid"),
        "title": paper.get("title", ""),
        "conference": paper.get("conference", ""),
        "year": paper.get("year"),
        "session": paper.get("session", ""),
    }

    for field in ("room_name", "poster_position", "starttime", "endtime", "url"):
        value = paper.get(field)
        if value:
            record[field] = value

    if recommendation_score is not None:
        record["recommendation_score"] = round(float(recommendation_score), 4)
    if priority is not None:
        record["priority"] = priority

    return record


def _author_entry(
    aggregates: Dict[str, Dict[str, Any]],
    key: str,
    display_name: str,
) -> Dict[str, Any]:
    """Return (and create) the aggregate entry for an author."""
    entry = aggregates.get(key)
    if entry is None:
        entry = {
            "display_names": Counter(),
            "score": 0.0,
            "similar_papers": {},
            "starred_papers": {},
            "similarity_scores": [],
        }
        aggregates[key] = entry

    entry["display_names"][display_name] += 1
    return entry


def _pick_display_name(display_names: Counter[str]) -> str:
    """Choose the most common, most specific display variant of a name."""
    return max(display_names.items(), key=lambda item: (item[1], len(item[0]), item[0]))[0]


def _author_reasons(
    similar_paper_count: int,
    average_similarity: float,
    starred_author: bool,
) -> List[str]:
    """Explain why an author is worth meeting."""
    reasons: List[str] = []

    if similar_paper_count:
        plural = "paper" if similar_paper_count == 1 else "papers"
        reasons.append(f"Authored {similar_paper_count} similar {plural}")

    if average_similarity:
        reasons.append(f"Average relevance {average_similarity:.2f}")

    if starred_author:
        reasons.append("Authored a paper you starred ⭐")

    return reasons


def _aggregate_authors(
    seed_papers: List[Dict[str, Any]],
    recommended_papers: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Aggregate authors across recommended and starred papers."""
    aggregates: Dict[str, Dict[str, Any]] = {}

    for paper in recommended_papers:
        uid = str(paper.get("uid") or "")
        score = float(paper.get("recommendation_score") or 0.0)
        record = _meeting_paper_record(paper, recommendation_score=score)

        for raw_author in paper.get("authors") or []:
            key = _normalise_author(raw_author)
            if not key:
                continue
            entry = _author_entry(aggregates, key, _display_author(raw_author))
            entry["score"] += score
            entry["similarity_scores"].append(score)
            if uid:
                entry["similar_papers"].setdefault(uid, record)

    for paper in seed_papers:
        uid = str(paper.get("uid") or "")
        priority = max(1, _as_int(paper.get("priority"), default=1))
        record = _meeting_paper_record(paper, priority=priority)

        for raw_author in paper.get("authors") or []:
            key = _normalise_author(raw_author)
            if not key:
                continue
            entry = _author_entry(aggregates, key, _display_author(raw_author))
            entry["score"] += priority / 5.0
            if uid:
                entry["starred_papers"].setdefault(uid, record)

    authors: List[Dict[str, Any]] = []
    for entry in aggregates.values():
        similar_papers = list(entry["similar_papers"].values())
        starred_papers = list(entry["starred_papers"].values())
        similarity_scores: List[float] = entry["similarity_scores"]
        average_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0

        authors.append(
            {
                "name": _pick_display_name(entry["display_names"]),
                "score": round(entry["score"], 4),
                "similar_paper_count": len(similar_papers),
                "starred_paper_count": len(starred_papers),
                "average_similarity": round(average_similarity, 4),
                "starred_author": bool(starred_papers),
                "papers": similar_papers,
                "starred_papers": starred_papers,
                "why": _author_reasons(len(similar_papers), average_similarity, bool(starred_papers)),
            }
        )

    authors.sort(
        key=lambda item: (
            -item["score"],
            -item["similar_paper_count"],
            _normalise_author(item["name"]),
        )
    )
    return authors


def suggest_authors_to_meet(
    database: DatabaseManager,
    embeddings_manager: EmbeddingsManager,
    priorities: Optional[Dict[str, Any]] = None,
    paper_ids: Optional[Sequence[str]] = None,
    conferences: Optional[Sequence[str]] = None,
    years: Optional[Sequence[int]] = None,
    limit: int = DEFAULT_AUTHOR_LIMIT,
    paper_limit: int = DEFAULT_AUTHOR_PAPER_LIMIT,
    exclude_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Suggest authors to meet based on a user's interesting papers.

    Papers semantically similar to the interesting papers are retrieved with
    :func:`recommend_related_papers`.  Authors are ranked by their
    similarity-weighted frequency across those papers: each recommended paper
    contributes its relevance score, and each starred paper contributes
    ``max(priority, 1) / 5``.  Authors of starred papers are included and
    marked with ``starred_author``.

    Parameters
    ----------
    database : DatabaseManager
        Open database connection.
    embeddings_manager : EmbeddingsManager
        Connected embeddings manager.
    priorities : dict, optional
        Raw priority mapping from the web UI.
    paper_ids : sequence of str, optional
        Explicit UID list.  When omitted, the keys of *priorities* are used.
    conferences : sequence of str, optional
        Restrict similar papers to these conferences.
    years : sequence of int, optional
        Restrict similar papers to these years.
    limit : int, optional
        Maximum number of authors to return (default: 10).
    paper_limit : int, optional
        Maximum number of similar papers to consider (default: 50).
    exclude_ids : sequence of str, optional
        Additional UIDs to exclude from the similar-paper search.

    Returns
    -------
    dict
        Dictionary with ``authors`` (ranked author records), counts for papers
        and seeds, filter metadata, and an optional ``warning``.
    """
    limit = max(1, min(int(limit), MAX_AUTHOR_LIMIT))
    paper_limit = max(1, min(int(paper_limit), MAX_AUTHOR_PAPER_LIMIT))

    loaded = load_interesting_papers(
        database=database,
        priorities=priorities,
        paper_ids=paper_ids,
    )
    seed_papers: List[Dict[str, Any]] = loaded["papers"]

    if not seed_papers:
        return {
            "authors": [],
            "authors_count": 0,
            "papers_considered": 0,
            "seed_count": 0,
            "filters": loaded["filters"],
            "warning": (
                "No interesting papers were found. Mark papers with stars in the web UI "
                "before asking for authors to meet."
            ),
        }

    recommendations = recommend_related_papers(
        database=database,
        embeddings_manager=embeddings_manager,
        priorities=priorities,
        paper_ids=paper_ids,
        conferences=conferences,
        years=years,
        limit=paper_limit,
        exclude_ids=exclude_ids,
    )
    recommended_papers = recommendations.get("papers") or []
    authors = _aggregate_authors(seed_papers, recommended_papers)[:limit]

    warning = recommendations.get("warning")
    if not authors:
        warning = warning or (
            "No authors were found in the recommended papers. "
            "Try marking more papers as interesting or widening the filters."
        )

    return {
        "authors": authors,
        "authors_count": len(authors),
        "papers_considered": len(recommended_papers),
        "seed_count": len(seed_papers),
        "filters": recommendations.get("filters", loaded["filters"]),
        "warning": warning,
    }

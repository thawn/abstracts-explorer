"""
Load interesting (rated) papers and resolve scope filters.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.utils.conference_utils import parse_conference_year

logger = logging.getLogger(__name__)


def _as_int(value: Any, default: int = 0) -> int:
    """Best-effort integer conversion."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_priorities(priorities: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Normalise the web UI's ``paperPriorities`` object.

    Every value must be a mapping with a ``priority`` field and an optional
    ``searchTerm`` field::

        {"paper-uid": {"priority": 4, "searchTerm": "llms"}}

    Legacy scalar ratings (``{"paper-uid": 4}``) are rejected.

    Parameters
    ----------
    priorities : dict, optional
        Raw priority mapping from the browser.

    Returns
    -------
    dict
        Mapping ``uid -> {"priority": int, "searchTerm": str}``.

    Raises
    ------
    ValueError
        If any rating is not a dict.
    """
    normalized: Dict[str, Dict[str, Any]] = {}
    for raw_uid, value in (priorities or {}).items():
        uid = str(raw_uid)
        if not uid:
            continue

        if not isinstance(value, dict):
            raise ValueError(
                f"Invalid rating format for paper '{uid}': expected a dict with "
                "'priority' and optional 'searchTerm'."
            )

        priority = _as_int(value.get("priority", 0))
        search_term = value.get("searchTerm") or value.get("search_term") or "Unknown"

        priority = max(0, min(5, priority))
        normalized[uid] = {"priority": priority, "searchTerm": str(search_term) or "Unknown"}

    return normalized


def _ordered_uids(paper_ids: Optional[Sequence[str]], priorities: Dict[str, Dict[str, Any]]) -> List[str]:
    """Return a de-duplicated, order-preserving UID list."""
    if isinstance(paper_ids, str):
        raw_ids = [paper_ids]
    elif paper_ids:
        raw_ids = list(paper_ids)
    else:
        raw_ids = list(priorities.keys())
    seen: set[str] = set()
    ordered: List[str] = []
    for raw_uid in raw_ids:
        uid = str(raw_uid).strip()
        if uid and uid not in seen:
            seen.add(uid)
            ordered.append(uid)
    return ordered


def _attach_ratings(
    papers: List[Dict[str, Any]],
    priorities: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Attach ``priority`` and ``searchTerm`` to paper records."""
    for paper in papers:
        rating = priorities.get(str(paper.get("uid", "")))
        if rating is None:
            paper["priority"] = 0
            paper["searchTerm"] = "Unknown"
        else:
            paper["priority"] = rating["priority"]
            paper["searchTerm"] = rating["searchTerm"]
    return papers


def _matches_scope(
    paper: Dict[str, Any],
    conferences: Optional[Sequence[str]],
    years: Optional[Sequence[int]],
) -> bool:
    """Return whether *paper* matches the requested conference/year filters."""
    if conferences:
        paper_conf = str(paper.get("conference") or "").strip().lower()
        if paper_conf not in {str(conf).strip().lower() for conf in conferences}:
            return False

    if years:
        paper_year = _as_int(paper.get("year"), default=-1)
        if paper_year < 0 or paper_year not in {int(year) for year in years}:
            return False

    return True


def _resolve_scope(
    conference: Optional[str],
    years: Optional[Sequence[int]],
) -> Tuple[Optional[List[str]], Optional[List[int]]]:
    """Resolve a possibly year-suffixed conference string into filters."""
    if not conference:
        return None, list(years) if years else None

    parsed_conference, extracted_year = parse_conference_year(conference)
    resolved_years = list(years) if years else None
    if extracted_year is not None:
        if resolved_years is None:
            resolved_years = [extracted_year]
        elif extracted_year not in resolved_years:
            resolved_years = sorted(resolved_years + [extracted_year])

    return [parsed_conference], resolved_years


def _resolve_conferences_scope(
    conferences: Optional[Sequence[str]],
    years: Optional[Sequence[int]],
) -> Tuple[Optional[List[str]], Optional[List[int]]]:
    """Resolve year-suffixed entries in a conference list into filters."""
    if not conferences:
        return None, list(years) if years else None

    resolved_conferences: List[str] = []
    resolved_years = {int(year) for year in years} if years else set()
    for conference in conferences:
        parsed_conference, extracted_year = parse_conference_year(str(conference))
        if parsed_conference:
            resolved_conferences.append(parsed_conference)
        if extracted_year is not None:
            resolved_years.add(extracted_year)

    return resolved_conferences or None, sorted(resolved_years) or None


def load_interesting_papers(
    database: DatabaseManager,
    priorities: Optional[Dict[str, Any]] = None,
    paper_ids: Optional[Sequence[str]] = None,
    conference: Optional[str] = None,
    years: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """
    Load and filter the papers marked as interesting by the user.

    Parameters
    ----------
    database : DatabaseManager
        Open database connection.
    priorities : dict, optional
        Raw priority mapping from the web UI.
    paper_ids : sequence of str, optional
        Explicit UID list.  When omitted, the keys of *priorities* are used.
    conference : str, optional
        Conference filter.  A trailing year (e.g. ``"ICLR 2026"``) is split
        automatically.
    years : sequence of int, optional
        Year filter.

    Returns
    -------
    dict
        Dictionary containing ``papers`` (filtered, with ratings attached),
        ``priorities`` (normalised), ``filters``, and ``missing_uids``.
    """
    normalized = normalize_priorities(priorities)
    uids = _ordered_uids(paper_ids, normalized)
    conferences, resolved_years = _resolve_scope(conference, years)

    papers = database.get_papers_by_uids(uids) if uids else []
    _attach_ratings(papers, normalized)

    found_uids = {str(paper.get("uid", "")) for paper in papers}
    missing_uids = [uid for uid in uids if uid not in found_uids]
    filtered = [paper for paper in papers if _matches_scope(paper, conferences, resolved_years)]

    return {
        "papers": filtered,
        "priorities": normalized,
        "requested_uids": uids,
        "missing_uids": missing_uids,
        "filters": {
            "conferences": conferences,
            "years": resolved_years,
        },
    }

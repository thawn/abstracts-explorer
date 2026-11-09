"""
Interesting-paper summaries and recommendations.

This module implements the core logic for two MCP tools:

- :func:`summarize_interesting_papers` summarises the papers a user has
  marked as interesting (starred/rated in the web UI).
- :func:`recommend_related_papers` suggests further papers based on a
  rating-weighted embedding centroid of those interesting papers.

The user's ratings live in the browser's ``localStorage``.  The web UI
forwards them with every chat request; these functions therefore receive the
ratings as plain dictionaries instead of persisting them server-side.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.embeddings import EmbeddingsManager
from abstracts_explorer.paper_utils import extract_top_keywords

logger = logging.getLogger(__name__)

DEFAULT_RECOMMENDATION_LIMIT = 10
MAX_SUMMARY_PAPERS = 50
ABSTRACT_SNIPPET_LENGTH = 400


def _parse_conference_year(conference: str) -> Tuple[str, Optional[int]]:
    """
    Split a trailing year from a conference name.

    Supports ``"ICLR 2026"``, ``"ICLR'26"``, and similar forms.
    """
    match = re.match(r"^(.+?)[\s']+(\d{4}|\d{2})$", conference.strip())
    if not match:
        return conference.strip(), None

    name = match.group(1).strip()
    year = int(match.group(2))
    if year < 100:
        year += 2000
    return name, year


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
    raw_ids = list(paper_ids) if paper_ids else list(priorities.keys())
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

    parsed_conference, extracted_year = _parse_conference_year(conference)
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
        parsed_conference, extracted_year = _parse_conference_year(str(conference))
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

    if include_abstract:
        abstract = str(paper.get("abstract") or "")
        if abstract:
            snippet = abstract[:ABSTRACT_SNIPPET_LENGTH]
            if len(abstract) > ABSTRACT_SNIPPET_LENGTH:
                snippet += "..."
            record["abstract_snippet"] = snippet

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


def _normalise_token(value: Any) -> str:
    """Normalise a metadata token for comparison."""
    return " ".join(str(value or "").strip().lower().split())


def _seed_context(papers: List[Dict[str, Any]]) -> Dict[str, set[str]]:
    """Collect authors, keywords, sessions, and search terms from seed papers."""
    authors: set[str] = set()
    keywords: set[str] = set()
    sessions: set[str] = set()
    search_terms: set[str] = set()

    for paper in papers:
        authors.update(_normalise_token(author) for author in paper.get("authors") or [] if _normalise_token(author))
        keywords.update(
            _normalise_token(keyword) for keyword in paper.get("keywords") or [] if _normalise_token(keyword)
        )
        session = _normalise_token(paper.get("session"))
        if session:
            sessions.add(session)
        search_term = _normalise_token(paper.get("searchTerm"))
        if search_term and search_term != "unknown":
            search_terms.add(search_term)

    return {
        "authors": authors,
        "keywords": keywords,
        "sessions": sessions,
        "search_terms": search_terms,
    }


def _recommendation_reasons(
    paper: Dict[str, Any],
    seed_context: Dict[str, set[str]],
    semantic_score: float,
) -> List[str]:
    """Explain why a candidate paper is related to the interesting papers."""
    reasons: List[str] = []

    paper_keywords = {
        _normalise_token(keyword) for keyword in paper.get("keywords") or [] if _normalise_token(keyword)
    }
    shared_keywords = sorted(paper_keywords & seed_context["keywords"])
    if shared_keywords:
        reasons.append(f"Shares keywords: {', '.join(shared_keywords[:3])}")

    paper_authors = {_normalise_token(author) for author in paper.get("authors") or [] if _normalise_token(author)}
    shared_authors = sorted(paper_authors & seed_context["authors"])
    if shared_authors:
        reasons.append(f"Shares author(s): {', '.join(shared_authors[:3])}")

    session = _normalise_token(paper.get("session"))
    if session and session in seed_context["sessions"]:
        reasons.append(f"Same session: {paper.get('session')}")

    if semantic_score >= 0.55:
        reasons.append("Semantically close to your interesting papers")

    if not reasons:
        reasons.append("Related through the topic landscape around your interesting papers")

    return reasons[:3]


def _build_where_filter(
    conferences: Optional[Sequence[str]],
    years: Optional[Sequence[int]],
) -> Optional[Dict[str, Any]]:
    """Build a ChromaDB metadata filter from conference/year constraints."""
    conditions: List[Dict[str, Any]] = []
    if conferences:
        conditions.append({"conference": {"$in": [str(conf) for conf in conferences]}})
    if years:
        conditions.append({"year": {"$in": [str(year) for year in years]}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _infer_scope(papers: List[Dict[str, Any]]) -> Tuple[Optional[List[str]], Optional[List[int]]]:
    """Infer a sensible candidate scope from the interesting papers."""
    conference_counts = Counter(
        str(paper.get("conference")).strip() for paper in papers if str(paper.get("conference") or "").strip()
    )
    if not conference_counts:
        return None, None

    conference = conference_counts.most_common(1)[0][0]
    years = sorted(
        {
            _as_int(paper.get("year"))
            for paper in papers
            if str(paper.get("conference") or "").strip() == conference and _as_int(paper.get("year")) > 0
        }
    )
    return [conference], years or None


def _seed_embeddings(
    embeddings_manager: EmbeddingsManager,
    papers: List[Dict[str, Any]],
) -> Tuple[List[List[float]], List[float]]:
    """Load stored embeddings for seed papers with rating-based weights."""
    uids = [str(paper.get("uid")) for paper in papers if paper.get("uid")]
    if not uids:
        return [], []

    result = embeddings_manager.collection.get(ids=uids, include=["embeddings"])
    result_ids = [str(uid) for uid in (result.get("ids") or [])]
    raw_embeddings = result.get("embeddings")
    if raw_embeddings is None:
        return [], []

    priority_by_uid = {str(paper.get("uid")): max(1, _as_int(paper.get("priority"), default=1)) for paper in papers}

    vectors: List[List[float]] = []
    weights: List[float] = []
    for index, uid in enumerate(result_ids):
        if index >= len(raw_embeddings) or raw_embeddings[index] is None:
            continue
        vectors.append([float(value) for value in raw_embeddings[index]])
        weights.append(float(priority_by_uid.get(uid, 1)))

    return vectors, weights


def _weighted_centroid(vectors: List[List[float]], weights: List[float]) -> Optional[List[float]]:
    """Compute a weight-normalised centroid of embedding vectors."""
    if not vectors:
        return None

    matrix = np.asarray(vectors, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    total_weight = float(weight_array.sum())
    if total_weight <= 0:
        return matrix.mean(axis=0).tolist()

    centroid = (matrix * weight_array[:, None]).sum(axis=0) / total_weight
    norm = float(np.linalg.norm(centroid))
    if norm > 0:
        centroid = centroid / norm
    return centroid.tolist()


def _fallback_recommendations(
    database: DatabaseManager,
    seed_papers: List[Dict[str, Any]],
    excluded_uids: set[str],
    conferences: Optional[Sequence[str]],
    years: Optional[Sequence[int]],
    n_results: int,
    reason: str,
) -> Dict[str, Any]:
    """Metadata/keyword-only fallback when embeddings are unavailable."""
    keyword_counts: Counter[str] = Counter()
    for paper in seed_papers:
        for keyword in paper.get("keywords") or []:
            cleaned = str(keyword).strip()
            if cleaned:
                keyword_counts[cleaned] += 1

    if not keyword_counts:
        for keyword in extract_top_keywords(seed_papers, n_keywords=8):
            keyword_counts[keyword] += 1

    candidates: Dict[str, Dict[str, Any]] = {}
    match_counts: Counter[str] = Counter()
    for keyword, _ in keyword_counts.most_common(8):
        matches = database.search_papers(
            keyword=keyword,
            conferences=list(conferences) if conferences else None,
            years=list(years) if years else None,
            limit=50,
        )
        for paper in matches:
            uid = str(paper.get("uid") or "")
            if not uid or uid in excluded_uids:
                continue
            candidates[uid] = paper
            match_counts[uid] += 1

    seed_context = _seed_context(seed_papers)
    max_matches = max(match_counts.values()) if match_counts else 1
    ranked = sorted(candidates.values(), key=lambda paper: match_counts[str(paper.get("uid"))], reverse=True)

    recommendations: List[Dict[str, Any]] = []
    for paper in ranked[:n_results]:
        uid = str(paper.get("uid"))
        score = round(match_counts[uid] / max_matches, 4)
        record = _paper_record(paper, include_abstract=True)
        record["recommendation_score"] = score
        record["recommendation_reasons"] = _recommendation_reasons(paper, seed_context, score)
        recommendations.append(record)

    return {
        "papers": recommendations,
        "recommendations_count": len(recommendations),
        "candidate_count": len(candidates),
        "warning": reason,
    }


def recommend_related_papers(
    database: DatabaseManager,
    embeddings_manager: EmbeddingsManager,
    priorities: Optional[Dict[str, Any]] = None,
    paper_ids: Optional[Sequence[str]] = None,
    conferences: Optional[Sequence[str]] = None,
    years: Optional[Sequence[int]] = None,
    n_results: int = DEFAULT_RECOMMENDATION_LIMIT,
    exclude_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Recommend papers based on a user's interesting papers.

    A rating-weighted centroid is computed from the interesting papers'
    embedding vectors, then the nearest papers in embedding space are
    returned.  Already-rated papers are excluded.

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
        Restrict recommendations to these conferences.
    years : sequence of int, optional
        Restrict recommendations to these years.
    n_results : int, optional
        Number of recommendations to return.
    exclude_ids : sequence of str, optional
        Additional UIDs to exclude from the results.

    Returns
    -------
    dict
        Dictionary with ``papers`` (recommendations), filter metadata, and
        an optional ``warning``.
    """
    n_results = max(1, min(int(n_results), 50))
    loaded = load_interesting_papers(
        database=database,
        priorities=priorities,
        paper_ids=paper_ids,
    )
    seed_papers: List[Dict[str, Any]] = loaded["papers"]

    if not seed_papers:
        return {
            "papers": [],
            "recommendations_count": 0,
            "seed_count": 0,
            "candidate_count": 0,
            "filters": {
                "conferences": list(conferences) if conferences else None,
                "years": list(years) if years else None,
            },
            "warning": (
                "No interesting papers were found. Mark papers with stars in the web UI "
                "before asking for recommendations."
            ),
        }

    if not conferences and not years:
        inferred_conferences, inferred_years = _infer_scope(seed_papers)
        conferences = inferred_conferences
        years = inferred_years

    resolved_conferences, resolved_years = _resolve_conferences_scope(conferences, years)
    where = _build_where_filter(resolved_conferences, resolved_years)

    excluded_uids = {str(paper.get("uid")) for paper in seed_papers if paper.get("uid")}
    for uid in exclude_ids or []:
        if str(uid).strip():
            excluded_uids.add(str(uid).strip())

    filters = {
        "conferences": resolved_conferences,
        "years": resolved_years,
    }

    try:
        seed_vectors, seed_weights = _seed_embeddings(embeddings_manager, seed_papers)
    except Exception as exc:
        logger.warning("Could not load seed embeddings: %s", exc)
        seed_vectors, seed_weights = [], []

    centroid = _weighted_centroid(seed_vectors, seed_weights)
    if centroid is None:
        fallback = _fallback_recommendations(
            database=database,
            seed_papers=seed_papers,
            excluded_uids=excluded_uids,
            conferences=resolved_conferences,
            years=resolved_years,
            n_results=n_results,
            reason=(
                "No stored embeddings were available for the interesting papers; "
                "recommendations use metadata keyword overlap instead."
            ),
        )
        fallback["seed_count"] = len(seed_papers)
        fallback["filters"] = filters
        return fallback

    try:
        collection_size = int(embeddings_manager.collection.count())
    except Exception:
        collection_size = 0

    if collection_size <= 0:
        fallback = _fallback_recommendations(
            database=database,
            seed_papers=seed_papers,
            excluded_uids=excluded_uids,
            conferences=resolved_conferences,
            years=resolved_years,
            n_results=n_results,
            reason="The embedding collection is empty; recommendations use metadata keyword overlap instead.",
        )
        fallback["seed_count"] = len(seed_papers)
        fallback["filters"] = filters
        return fallback

    n_query = min(collection_size, max(n_results + len(excluded_uids) + 20, n_results * 4))

    try:
        results = embeddings_manager.collection.query(
            query_embeddings=[centroid],
            n_results=n_query,
            where=where,
        )
    except Exception as exc:
        logger.warning("Embedding-based recommendation query failed: %s", exc)
        fallback = _fallback_recommendations(
            database=database,
            seed_papers=seed_papers,
            excluded_uids=excluded_uids,
            conferences=resolved_conferences,
            years=resolved_years,
            n_results=n_results,
            reason=f"Embedding search failed ({exc}); recommendations use metadata keyword overlap instead.",
        )
        fallback["seed_count"] = len(seed_papers)
        fallback["filters"] = filters
        return fallback

    seed_context = _seed_context(seed_papers)
    recommendations: List[Dict[str, Any]] = []
    seen: set[str] = set()

    ids = results.get("ids", [[]])
    distances = results.get("distances", [[]])
    metadatas = results.get("metadatas", [[]])
    documents = results.get("documents", [[]])

    if ids and ids[0]:
        for index, raw_uid in enumerate(ids[0]):
            uid = str(raw_uid)
            if not uid or uid in excluded_uids or uid in seen:
                continue

            metadata: Dict[str, Any] = {}
            if metadatas and metadatas[0] and index < len(metadatas[0]):
                try:
                    metadata = embeddings_manager.parse_chromadb_metadata(metadatas[0][index] or {})
                except Exception:
                    metadata = dict(metadatas[0][index] or {})

            paper = dict(metadata)
            paper["uid"] = uid
            if not paper.get("abstract") and documents and documents[0] and index < len(documents[0]):
                paper["abstract"] = documents[0][index] or ""

            distance = 0.0
            if distances and distances[0] and index < len(distances[0]):
                distance = float(distances[0][index])

            score = round(max(0.0, 1.0 - distance), 4)
            record = _paper_record(paper, include_abstract=True)
            record["recommendation_score"] = score
            record["recommendation_reasons"] = _recommendation_reasons(paper, seed_context, score)

            recommendations.append(record)
            seen.add(uid)

            if len(recommendations) >= n_results:
                break

    warning = None
    if not recommendations:
        warning = (
            "No recommendations matched the requested conference/year scope. "
            "Try widening the filters or marking more papers as interesting."
        )

    return {
        "papers": recommendations,
        "recommendations_count": len(recommendations),
        "seed_count": len(seed_papers),
        "candidate_count": len(ids[0]) if ids and ids[0] else 0,
        "filters": filters,
        "warning": warning,
    }

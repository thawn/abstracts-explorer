"""
Recommend papers related to a user's interesting papers.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.embeddings import EmbeddingsManager
from abstracts_explorer.interesting_papers.constants import DEFAULT_RECOMMENDATION_LIMIT
from abstracts_explorer.interesting_papers.loading import (
    _as_int,
    _resolve_conferences_scope,
    load_interesting_papers,
)
from abstracts_explorer.interesting_papers.summary import _paper_record
from abstracts_explorer.paper_utils import extract_top_keywords

logger = logging.getLogger(__name__)


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
    limit: int,
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
        matches = database.keyword_search(
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
    for paper in ranked[:limit]:
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
    limit: int = DEFAULT_RECOMMENDATION_LIMIT,
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
    limit : int, optional
        Number of recommendations to return.
    exclude_ids : sequence of str, optional
        Additional UIDs to exclude from the results.

    Returns
    -------
    dict
        Dictionary with ``papers`` (recommendations), filter metadata, and
        an optional ``warning``.
    """
    limit = max(1, min(int(limit), 50))
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

    excluded_uids = {str(paper.get("uid")) for paper in seed_papers if paper.get("uid")}
    for uid in exclude_ids or []:
        if str(uid).strip():
            excluded_uids.add(str(uid).strip())

    filters = {
        "conferences": resolved_conferences,
        "years": resolved_years,
    }

    has_scope = bool(resolved_conferences or resolved_years)
    candidate_uids = (
        database.resolve_paper_uids(
            conferences=resolved_conferences,
            years=resolved_years,
        )
        if has_scope
        else None
    )
    if candidate_uids is not None and not candidate_uids:
        return {
            "papers": [],
            "recommendations_count": 0,
            "seed_count": len(seed_papers),
            "candidate_count": 0,
            "filters": filters,
            "warning": (
                "No recommendations matched the requested conference/year scope. "
                "Try widening the filters or marking more papers as interesting."
            ),
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
            limit=limit,
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
            limit=limit,
            reason="The embedding collection is empty; recommendations use metadata keyword overlap instead.",
        )
        fallback["seed_count"] = len(seed_papers)
        fallback["filters"] = filters
        return fallback

    candidate_limit = min(
        len(candidate_uids) if candidate_uids is not None else collection_size,
        max(limit + len(excluded_uids) + 20, limit * 4),
    )

    try:
        results = embeddings_manager.collection.query(
            query_embeddings=[centroid],
            n_results=candidate_limit,
            ids=candidate_uids,
            include=["distances"],
        )
    except Exception as exc:
        logger.warning("Embedding-based recommendation query failed: %s", exc)
        fallback = _fallback_recommendations(
            database=database,
            seed_papers=seed_papers,
            excluded_uids=excluded_uids,
            conferences=resolved_conferences,
            years=resolved_years,
            limit=limit,
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
    result_uids = [str(uid) for uid in ids[0]] if ids and ids[0] else []
    papers_by_uid = {paper["uid"]: paper for paper in database.get_papers_by_uids(result_uids)}

    if result_uids:
        for index, uid in enumerate(result_uids):
            if not uid or uid in excluded_uids or uid in seen:
                continue

            paper = papers_by_uid.get(uid)
            if paper is None:
                continue

            distance = 0.0
            if distances and distances[0] and index < len(distances[0]):
                distance = float(distances[0][index])

            score = round(max(0.0, 1.0 - distance), 4)
            record = _paper_record(paper, include_abstract=True)
            record["recommendation_score"] = score
            record["recommendation_reasons"] = _recommendation_reasons(paper, seed_context, score)

            recommendations.append(record)
            seen.add(uid)

            if len(recommendations) >= limit:
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

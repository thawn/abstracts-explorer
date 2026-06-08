"""
Graph-based paper recommendations for Abstracts Explorer.

The recommender builds a small heterogeneous graph from papers, authors,
keywords, sessions, and embedding-nearest paper links, then runs Personalized
PageRank from the user's query seeds.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from abstracts_explorer.config import get_config
from abstracts_explorer.database import DatabaseManager

logger = logging.getLogger(__name__)

DEFAULT_RECOMMENDATION_LIMIT = 50
DEFAULT_SEMANTIC_NEIGHBORS = 8
DEFAULT_DAMPING = 0.85
DEFAULT_MAX_ITER = 60
DEFAULT_TOLERANCE = 1e-8


def _normalise(value: Any) -> str:
    """Normalise a graph token for stable node IDs."""
    return " ".join(str(value or "").strip().lower().split())


def _node(kind: str, value: Any) -> str:
    return f"{kind}:{_normalise(value)}"


def _paper_node(uid: str) -> str:
    return f"paper:{uid}"


def _add_edge(graph: Dict[str, Dict[str, float]], left: str, right: str, weight: float) -> None:
    if not left or not right or left == right or weight <= 0:
        return
    graph[left][right] += weight
    graph[right][left] += weight


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _safe_uid(paper: Dict[str, Any]) -> Optional[str]:
    uid = paper.get("uid")
    return str(uid) if uid else None


def _cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalised = vectors / norms
    return normalised @ normalised.T


def _cosine_similarity_to_query(vectors: np.ndarray, query_vector: Sequence[float]) -> np.ndarray:
    query = np.asarray(query_vector, dtype=float)
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return np.zeros(vectors.shape[0], dtype=float)
    vector_norms = np.linalg.norm(vectors, axis=1)
    vector_norms[vector_norms == 0] = 1.0
    return (vectors @ query) / (vector_norms * query_norm)


def personalized_pagerank(
    graph: Dict[str, Dict[str, float]],
    personalization: Dict[str, float],
    damping: float = DEFAULT_DAMPING,
    max_iter: int = DEFAULT_MAX_ITER,
    tolerance: float = DEFAULT_TOLERANCE,
) -> Dict[str, float]:
    """
    Run weighted Personalized PageRank on an undirected graph.

    Parameters
    ----------
    graph
        Mapping of node -> neighbor -> edge weight.
    personalization
        Restart distribution. Nodes missing from the graph are ignored.
    damping
        Probability of following graph edges instead of restarting.
    max_iter
        Maximum power iterations.
    tolerance
        L1 convergence threshold.
    """
    nodes = set(graph.keys())
    for neighbors in graph.values():
        nodes.update(neighbors.keys())
    if not nodes:
        return {}

    restart = {node: max(0.0, personalization.get(node, 0.0)) for node in nodes}
    restart_total = sum(restart.values())
    if restart_total <= 0:
        uniform = 1.0 / len(nodes)
        restart = {node: uniform for node in nodes}
    else:
        restart = {node: value / restart_total for node, value in restart.items()}

    ranks = dict(restart)
    outgoing_totals = {
        node: sum(max(0.0, weight) for weight in graph.get(node, {}).values()) for node in nodes
    }

    for _ in range(max_iter):
        next_ranks = {node: (1.0 - damping) * restart[node] for node in nodes}
        dangling_mass = 0.0

        for node, rank in ranks.items():
            total = outgoing_totals.get(node, 0.0)
            if total <= 0:
                dangling_mass += rank
                continue
            share = damping * rank / total
            for neighbor, weight in graph.get(node, {}).items():
                if weight > 0:
                    next_ranks[neighbor] += share * weight

        if dangling_mass:
            for node in nodes:
                next_ranks[node] += damping * dangling_mass * restart[node]

        delta = sum(abs(next_ranks[node] - ranks.get(node, 0.0)) for node in nodes)
        ranks = next_ranks
        if delta < tolerance:
            break

    return ranks


def _get_embeddings_for_papers(embeddings_manager: Any, paper_uids: List[str]) -> Dict[str, List[float]]:
    if not paper_uids:
        return {}

    collection_result = embeddings_manager.collection.get(ids=paper_uids, include=["embeddings"])
    result_ids = collection_result.get("ids", []) or []
    result_embeddings = collection_result.get("embeddings")
    if result_embeddings is None:
        return {}

    return {
        str(uid): list(embedding)
        for uid, embedding in zip(result_ids, result_embeddings)
        if embedding is not None
    }


def _build_seed_papers(
    database: DatabaseManager,
    query: str,
    field_filters: Dict[str, str],
    remaining_query: str,
    sessions: Optional[List[str]],
    years: Optional[List[int]],
    conferences: Optional[List[str]],
) -> Tuple[Dict[str, Dict[str, Any]], set[str]]:
    seed_papers: Dict[str, Dict[str, Any]] = {}
    exact_seed_uids: set[str] = set()

    if field_filters:
        matches = database.search_papers(
            field_filters=field_filters,
            sessions=sessions,
            years=years,
            conferences=conferences,
            limit=0,
        )
        for paper in matches:
            uid = _safe_uid(paper)
            if uid:
                seed_papers[uid] = paper
                exact_seed_uids.add(uid)

    author_query = remaining_query or query
    if author_query and "authors" not in field_filters:
        author_matches = database.search_papers(
            field_filters={"authors": author_query},
            sessions=sessions,
            years=years,
            conferences=conferences,
            limit=0,
        )
        for paper in author_matches:
            uid = _safe_uid(paper)
            if uid:
                seed_papers[uid] = paper
                exact_seed_uids.add(uid)

    if remaining_query and not seed_papers:
        keyword_matches = database.search_papers(
            keyword=remaining_query,
            sessions=sessions,
            years=years,
            conferences=conferences,
            limit=25,
        )
        for paper in keyword_matches:
            uid = _safe_uid(paper)
            if uid:
                seed_papers[uid] = paper
                exact_seed_uids.add(uid)

    return seed_papers, exact_seed_uids


def _build_graph(
    papers: Dict[str, Dict[str, Any]],
    embeddings: Dict[str, List[float]],
    semantic_neighbors: int = DEFAULT_SEMANTIC_NEIGHBORS,
) -> Dict[str, Dict[str, float]]:
    graph: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    paper_ids = list(papers)
    for uid in paper_ids:
        paper = papers[uid]
        paper_node = _paper_node(uid)
        graph[paper_node]

        authors = _as_list(paper.get("authors"))
        for author in authors:
            _add_edge(graph, paper_node, _node("author", author), 1.25 / max(1, len(authors)))

        keywords = _as_list(paper.get("keywords"))[:8]
        for keyword in keywords:
            _add_edge(graph, paper_node, _node("keyword", keyword), 0.55)

        if paper.get("session"):
            _add_edge(graph, paper_node, _node("session", paper["session"]), 0.45)
        if paper.get("conference"):
            _add_edge(graph, paper_node, _node("conference", paper["conference"]), 0.12)
        if paper.get("year"):
            _add_edge(graph, paper_node, _node("year", paper["year"]), 0.08)

    embedded_paper_ids = [uid for uid in paper_ids if uid in embeddings]
    if len(embedded_paper_ids) > 1 and semantic_neighbors > 0:
        matrix = np.asarray([embeddings[uid] for uid in embedded_paper_ids], dtype=float)
        similarities = _cosine_similarity_matrix(matrix)
        np.fill_diagonal(similarities, -math.inf)
        neighbor_count = min(semantic_neighbors, len(embedded_paper_ids) - 1)

        for i, uid in enumerate(embedded_paper_ids):
            row = similarities[i]
            if neighbor_count <= 0:
                continue
            if neighbor_count < len(row):
                neighbor_indices = np.argpartition(row, -neighbor_count)[-neighbor_count:]
            else:
                neighbor_indices = np.arange(len(row))
            for j in neighbor_indices:
                similarity = float(row[j])
                if not math.isfinite(similarity) or similarity <= 0:
                    continue
                _add_edge(graph, _paper_node(uid), _paper_node(embedded_paper_ids[int(j)]), 0.9 * similarity)

    return graph


def _build_personalization(
    query: str,
    field_filters: Dict[str, str],
    seed_papers: Dict[str, Dict[str, Any]],
    semantic_seed_scores: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, set[str]]]:
    personalization: Dict[str, float] = defaultdict(float)
    seed_context = {"authors": set(), "keywords": set(), "sessions": set()}

    for uid, paper in seed_papers.items():
        personalization[_paper_node(uid)] += 2.0
        for author in _as_list(paper.get("authors")):
            norm = _normalise(author)
            seed_context["authors"].add(norm)
            personalization[_node("author", author)] += 0.45
        for keyword in _as_list(paper.get("keywords")):
            seed_context["keywords"].add(_normalise(keyword))
        if paper.get("session"):
            seed_context["sessions"].add(_normalise(paper["session"]))

    if field_filters.get("authors"):
        personalization[_node("author", field_filters["authors"])] += 3.0
    if field_filters.get("keywords"):
        personalization[_node("keyword", field_filters["keywords"])] += 1.2
    if field_filters.get("session"):
        personalization[_node("session", field_filters["session"])] += 1.2

    for uid, score in semantic_seed_scores.items():
        personalization[_paper_node(uid)] += max(0.01, score)

    if query and not personalization:
        # Defensive fallback; callers normally add semantic seeds first.
        personalization[_node("query", query)] += 1.0

    return dict(personalization), seed_context


def _normalise_scores(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    max_score = max(scores.values())
    if max_score <= 0:
        return {key: 0.0 for key in scores}
    return {key: value / max_score for key, value in scores.items()}


def _recommendation_reasons(
    paper: Dict[str, Any],
    seed_context: Dict[str, set[str]],
    semantic_score: float,
    pagerank_score: float,
) -> List[str]:
    reasons: List[str] = []

    paper_authors = {_normalise(author) for author in _as_list(paper.get("authors"))}
    shared_authors = sorted(paper_authors & seed_context.get("authors", set()))
    if shared_authors:
        reasons.append("Connected through the searched author/coauthor neighborhood")

    paper_keywords = {_normalise(keyword) for keyword in _as_list(paper.get("keywords"))}
    shared_keywords = sorted(paper_keywords & seed_context.get("keywords", set()))
    if shared_keywords:
        reasons.append(f"Shares keyword: {shared_keywords[0]}")

    session = _normalise(paper.get("session"))
    if session and session in seed_context.get("sessions", set()):
        reasons.append("Appears in the same session neighborhood")

    if semantic_score >= 0.55:
        reasons.append("Abstract is semantically close to the search")
    if pagerank_score >= 0.5 and not reasons:
        reasons.append("Central in the Personalized PageRank graph around your search")
    if not reasons:
        reasons.append("Graph connection from authors, topics, sessions, or nearby embeddings")

    return reasons[:3]


def recommend_papers(
    database: DatabaseManager,
    embeddings_manager: Any,
    query: str,
    sessions: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    conferences: Optional[List[str]] = None,
    limit: int = DEFAULT_RECOMMENDATION_LIMIT,
    unsupported_filters: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Recommend papers with Personalized PageRank over a heterogeneous graph.
    """
    field_filters, remaining_query = DatabaseManager.parse_field_filters(query)
    semantic_query = remaining_query or ("" if field_filters else query)

    candidate_papers = database.search_papers(
        sessions=sessions,
        years=years,
        conferences=conferences,
        limit=0,
    )
    candidate_by_uid = {uid: paper for paper in candidate_papers if (uid := _safe_uid(paper))}
    if not candidate_by_uid:
        return {
            "papers": [],
            "count": 0,
            "query": query,
            "seed_count": 0,
            "candidate_count": 0,
            "algorithm": "personalized_pagerank",
            "unsupported_filters": unsupported_filters or [],
        }

    try:
        embeddings = _get_embeddings_for_papers(embeddings_manager, list(candidate_by_uid.keys()))
    except Exception as exc:
        logger.warning("Could not load stored embeddings for recommendations: %s", exc)
        embeddings = {}
    embedded_candidate_count = len(embeddings)

    seed_papers, exact_seed_uids = _build_seed_papers(
        database=database,
        query=query,
        field_filters=field_filters,
        remaining_query=remaining_query,
        sessions=sessions,
        years=years,
        conferences=conferences,
    )
    seed_papers = {uid: paper for uid, paper in seed_papers.items() if uid in candidate_by_uid}
    exact_seed_uids = {uid for uid in exact_seed_uids if uid in candidate_by_uid}

    semantic_scores: Dict[str, float] = {}
    semantic_seed_scores: Dict[str, float] = {}
    query_embedding: Optional[List[float]] = None
    if semantic_query and embeddings:
        try:
            query_embedding = embeddings_manager.generate_embedding(semantic_query)
        except Exception as exc:
            logger.warning("Could not generate query embedding for recommendations: %s", exc)

    paper_ids = [uid for uid in candidate_by_uid if uid in embeddings]
    if query_embedding is not None:
        matrix = np.asarray([embeddings[uid] for uid in paper_ids], dtype=float)
        similarities = _cosine_similarity_to_query(matrix, query_embedding)
        semantic_scores = {uid: max(0.0, float(score)) for uid, score in zip(paper_ids, similarities)}
        semantic_seed_count = min(24, len(paper_ids))
        if semantic_seed_count:
            top_indices = np.argsort(similarities)[-semantic_seed_count:]
            semantic_seed_scores = {
                paper_ids[int(idx)]: max(0.01, float(similarities[int(idx)]))
                for idx in top_indices
                if float(similarities[int(idx)]) > 0
            }

    if not seed_papers and not semantic_seed_scores:
        return {
            "papers": [],
            "count": 0,
            "query": query,
            "seed_count": 0,
            "candidate_count": len(candidate_by_uid),
            "embedded_candidate_count": embedded_candidate_count,
            "algorithm": "personalized_pagerank",
            "unsupported_filters": unsupported_filters or [],
            "warning": "No seed papers or semantic seed matches were found for the recommendation graph.",
        }

    graph = _build_graph(candidate_by_uid, embeddings)
    personalization, seed_context = _build_personalization(
        query=query,
        field_filters=field_filters,
        seed_papers=seed_papers,
        semantic_seed_scores=semantic_seed_scores,
    )
    ranks = personalized_pagerank(graph, personalization)
    paper_rank_scores = {uid: ranks.get(_paper_node(uid), 0.0) for uid in candidate_by_uid}
    rank_norm = _normalise_scores(paper_rank_scores)
    semantic_norm = _normalise_scores(semantic_scores)

    scored: List[Tuple[float, str]] = []
    for uid in candidate_by_uid:
        if uid in exact_seed_uids:
            continue
        pr = rank_norm.get(uid, 0.0)
        semantic = semantic_norm.get(uid, 0.0)
        final = 0.75 * pr + 0.25 * semantic if semantic_norm else pr
        if final > 0:
            scored.append((final, uid))

    scored.sort(key=lambda item: item[0], reverse=True)

    recommendations: List[Dict[str, Any]] = []
    for final_score, uid in scored[:limit]:
        paper = dict(candidate_by_uid[uid])
        pagerank_score = rank_norm.get(uid, 0.0)
        semantic_score = semantic_norm.get(uid, 0.0)
        paper["recommendation_score"] = final_score
        paper["pagerank_score"] = pagerank_score
        paper["semantic_score"] = semantic_score
        paper["recommendation_reasons"] = _recommendation_reasons(
            paper=paper,
            seed_context=seed_context,
            semantic_score=semantic_score,
            pagerank_score=pagerank_score,
        )
        recommendations.append(paper)

    return {
        "papers": recommendations,
        "count": len(recommendations),
        "query": query,
        "seed_count": len(seed_papers) + len(semantic_seed_scores),
        "direct_seed_count": len(exact_seed_uids),
        "candidate_count": len(candidate_by_uid),
        "embedded_candidate_count": embedded_candidate_count,
        "algorithm": "personalized_pagerank",
        "recommendation_mode": "hybrid_graph" if embedded_candidate_count else "metadata_graph",
        "limit": limit,
        "unsupported_filters": unsupported_filters or [],
        "warning": None
        if embedded_candidate_count
        else "No stored embeddings were found, so recommendations used metadata-only PageRank.",
    }


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def explain_recommendations(
    embeddings_manager: Any,
    query: str,
    papers: List[Dict[str, Any]],
    top_n: int = 10,
) -> Dict[str, Any]:
    """
    Ask the configured chat model to explain the top recommendations.

    The function mutates the provided paper dictionaries by adding
    ``llm_explanation`` when a parseable explanation is returned.
    """
    if top_n <= 0 or not papers:
        return {"explained_count": 0}

    config = get_config()
    top_papers = papers[:top_n]
    payload = [
        {
            "uid": paper.get("uid"),
            "title": paper.get("title"),
            "authors": paper.get("authors", []),
            "year": paper.get("year"),
            "conference": paper.get("conference"),
            "score": round(float(paper.get("recommendation_score", 0.0)), 4),
            "graph_reasons": paper.get("recommendation_reasons", []),
            "abstract": (paper.get("abstract") or "")[:900],
        }
        for paper in top_papers
    ]

    system_message = (
        "You explain research-paper recommendations. Return only valid JSON with the shape "
        '{"explanations":[{"uid":"...","explanation":"one concise sentence"}]}. '
        "Do not invent facts that are not present in the provided paper data."
    )
    user_message = (
        f"Search/query seed: {query}\n\n"
        "Explain why each recommended paper may be interesting to the user. "
        "Use the graph reasons and abstract/title evidence.\n\n"
        f"Papers:\n{json.dumps(payload, ensure_ascii=False)}"
    )

    try:
        response = embeddings_manager.openai_client.chat.completions.create(
            model=config.chat_model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
            max_tokens=1400,
        )
        content = response.choices[0].message.content or ""
        parsed = _extract_json_object(content)
        if not parsed:
            return {"explained_count": 0, "raw_explanation": content}

        by_uid = {str(paper.get("uid")): paper for paper in top_papers}
        explained = 0
        for item in parsed.get("explanations", []):
            uid = str(item.get("uid", ""))
            explanation = str(item.get("explanation", "")).strip()
            if uid in by_uid and explanation:
                by_uid[uid]["llm_explanation"] = explanation
                explained += 1
        return {"explained_count": explained}
    except Exception as exc:
        logger.warning("LLM recommendation explanation failed: %s", exc)
        return {"explained_count": 0, "error": str(exc)}

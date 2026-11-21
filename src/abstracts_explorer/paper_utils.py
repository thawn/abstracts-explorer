"""
Paper formatting utilities for NeurIPS abstracts.

This module provides shared utilities for formatting papers from various sources
(database, search results, ChromaDB) with consistent structure and error handling.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# TF-IDF settings for keyword extraction from search results.
_TFIDF_MAX_FEATURES = 500
_TFIDF_SMALL_CORPUS_THRESHOLD = 3  # corpora smaller than this use min_df=1
_TFIDF_MIN_DF_SMALL = 1
_TFIDF_MIN_DF_REGULAR = 2


class PaperFormattingError(Exception):
    """Exception raised when paper formatting fails."""

    pass


def format_search_results(
    search_results: Dict[str, Any],
    database,
) -> List[Dict[str, Any]]:
    """
    Format ChromaDB search results into complete paper records.

    Converts search results from ChromaDB into fully-populated paper dictionaries
    by fetching complete data from the database. Fails early if required data
    is missing rather than returning incomplete records.

    Parameters
    ----------
    search_results : dict
        Search results from ChromaDB with ``ids`` and ``distances``.
    database : DatabaseManager
        Database instance for fetching complete paper details.

    Returns
    -------
    list
        List of complete paper dictionaries with authors, similarity scores, and all fields.

    Raises
    ------
    PaperFormattingError
        If search_results format is invalid or required data is missing.

    Examples
    --------
    >>> results = embeddings_manager.search_similar(QueryEmbedding("transformers", embeddings_manager), limit=5)
    >>> papers = format_search_results(results, database)
    >>> for paper in papers:
    ...     print(paper['title'], paper['similarity'])
    """
    # Validate search results structure
    if not isinstance(search_results, dict):
        raise PaperFormattingError("search_results must be a dictionary.")

    if "ids" not in search_results or not search_results["ids"]:
        raise PaperFormattingError("search_results must contain 'ids' field with results.")

    if not search_results["ids"][0]:
        # Empty results - valid but return empty list
        return []

    if database is None:
        raise PaperFormattingError("Database connection is required but not provided.")

    # Validate consistent result lengths
    result_count = len(search_results["ids"][0])
    if "distances" in search_results and search_results["distances"][0]:
        if len(search_results["distances"][0]) != result_count:
            raise PaperFormattingError("Inconsistent result lengths in search_results.")

    ids = search_results["ids"][0]
    valid_uids: List[str] = []
    for paper_uid in ids:
        if isinstance(paper_uid, str):
            valid_uids.append(paper_uid)
        else:
            logger.warning(f"Invalid paper_uid format: {paper_uid} ({type(paper_uid)})")

    papers_by_uid: Dict[str, Dict[str, Any]] = {}
    if valid_uids:
        unique_uids = list(dict.fromkeys(valid_uids))
        papers_by_uid = {paper["uid"]: paper for paper in database.get_papers_by_uids(unique_uids)}

    papers = []
    for i, paper_uid in enumerate(ids):
        if not isinstance(paper_uid, str):
            continue

        fetched_paper = papers_by_uid.get(paper_uid)
        if fetched_paper is None:
            logger.warning(f"Skipping paper {paper_uid}: Paper with uid={paper_uid} not found in database.")
            continue

        paper = dict(fetched_paper)

        # Add similarity/distance scores if available
        if "distances" in search_results and search_results["distances"][0]:
            distance = search_results["distances"][0][i]
            paper["distance"] = distance
            paper["similarity"] = max(0.0, 1.0 - distance)

        papers.append(paper)

    if not papers:
        raise PaperFormattingError(
            f"No valid papers could be formatted from {result_count} search results. "
            "Check database connectivity and paper IDs."
        )

    return papers


def extract_top_keywords(
    papers: list,
    n_keywords: int = 5,
    *,
    ngram_range: tuple[int, int] = (2, 3),
) -> list:
    """
    Extract top keywords from a list of papers using TF-IDF.

    Parameters
    ----------
    papers : list
        List of paper dicts, each with optional 'title' and 'abstract' keys.
    n_keywords : int, optional
        Number of top keywords to return (default: 5).
    ngram_range : tuple of int, optional
        Minimum and maximum n-gram sizes to extract (default: ``(2, 3)``).

    Returns
    -------
    list
        List of keyword strings, ordered by relevance (highest TF-IDF first).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = []
    for paper in papers:
        title = paper.get("title", "") or ""
        abstract = paper.get("abstract", "") or ""
        text = f"{title}\n\n{abstract}".strip()
        if text:
            docs.append(text)

    if not docs:
        return []

    try:
        min_df = _TFIDF_MIN_DF_SMALL if len(docs) < _TFIDF_SMALL_CORPUS_THRESHOLD else _TFIDF_MIN_DF_REGULAR
        tfidf = TfidfVectorizer(
            max_features=_TFIDF_MAX_FEATURES,
            min_df=min_df,
            stop_words="english",
            ngram_range=ngram_range,
        )
        tfidf_matrix = tfidf.fit_transform(docs)
        feature_names = tfidf.get_feature_names_out()
        mean_tfidf = tfidf_matrix.mean(axis=0).A1
        top_indices = mean_tfidf.argsort()[-n_keywords:][::-1]
        keywords = [feature_names[i] for i in top_indices if mean_tfidf[i] > 0]
        return keywords[:n_keywords]
    except Exception:
        return []

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
    include_documents: bool = True,
) -> List[Dict[str, Any]]:
    """
    Format ChromaDB search results into complete paper records.

    Converts search results from ChromaDB into fully-populated paper dictionaries
    by fetching complete data from the database. Fails early if required data
    is missing rather than returning incomplete records.

    Parameters
    ----------
    search_results : dict
        Search results from ChromaDB with 'ids', 'distances', 'metadatas', 'documents'.
    database : DatabaseManager
        Database instance for fetching complete paper details.
    include_documents : bool, optional
        Whether to include document text (abstract) from search results, by default True.

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
    >>> results = embeddings_manager.search_similar("transformers", n_results=5)
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
    if "documents" in search_results and search_results["documents"][0]:
        if len(search_results["documents"][0]) != result_count:
            raise PaperFormattingError("Inconsistent result lengths in search_results.")

    papers = []
    for i in range(result_count):
        paper_uid = search_results["ids"][0][i]

        try:
            # ChromaDB stores UIDs as strings - use them directly
            if not isinstance(paper_uid, str):
                logger.warning(f"Invalid paper_uid format: {paper_uid} ({type(paper_uid)})")
                continue

            # Get complete paper from database (this validates paper exists)
            paper = database.get_paper_by_uid(paper_uid)
            if paper is None:
                raise PaperFormattingError(f"Paper with uid={paper_uid} not found in database.")

            # Add similarity/distance scores if available
            if "distances" in search_results and search_results["distances"][0]:
                distance = search_results["distances"][0][i]
                paper["distance"] = distance
                # Convert distance to similarity (0-1 range where 1 is most similar)
                paper["similarity"] = max(0.0, 1.0 - distance)

            # Add abstract from search results if needed and available
            if include_documents and "documents" in search_results and search_results["documents"][0]:
                # Only override if database abstract is missing
                if not paper.get("abstract"):
                    paper["abstract"] = search_results["documents"][0][i]

            papers.append(paper)

        except PaperFormattingError as e:
            # Log the error but continue with other papers
            logger.warning(f"Skipping paper {paper_uid}: {str(e)}")
            continue

    if not papers:
        raise PaperFormattingError(
            f"No valid papers could be formatted from {result_count} search results. "
            "Check database connectivity and paper IDs."
        )

    return papers


def build_context_from_papers(papers: List[Dict[str, Any]]) -> str:
    """
    Build a formatted context string from papers for RAG.

    Parameters
    ----------
    papers : list
        List of paper dictionaries with at minimum: title/name, authors, abstract.

    Returns
    -------
    str
        Formatted context string for LLM consumption.

    Raises
    ------
    PaperFormattingError
        If papers list is invalid or papers missing required fields.

    Examples
    --------
    >>> papers = format_search_results(results, database)
    >>> context = build_context_from_papers(papers)
    >>> print(context)
    """
    if not isinstance(papers, list):
        raise PaperFormattingError("papers must be a list.")

    if not papers:
        raise PaperFormattingError("papers list is empty. Cannot build context.")

    context_parts = []
    for i, paper in enumerate(papers, 1):
        if not isinstance(paper, dict):
            raise PaperFormattingError(f"Paper {i} is not a dictionary.")

        # Validate required fields
        title = paper.get("title")
        if not title:
            raise PaperFormattingError(f"Paper {i} missing required 'title' field.")

        authors = paper.get("authors")
        if not authors:
            logger.warning(f"Paper {i} ({title}) has no authors.")
            authors = "N/A"
        elif isinstance(authors, list):
            authors = ", ".join(authors) if authors else "N/A"

        abstract = paper.get("abstract", "")
        if not abstract:
            logger.warning(f"Paper {i} ({title}) has no abstract.")
            abstract = "N/A"

        # Build context for this paper
        context_parts.append(f"Paper {i}:")
        context_parts.append(f"Title: {title}")
        context_parts.append(f"Authors: {authors}")

        # Optional fields
        if paper.get("topic"):
            context_parts.append(f"Topic: {paper['topic']}")
        if paper.get("decision"):
            context_parts.append(f"Decision: {paper['decision']}")

        context_parts.append(f"Abstract: {abstract}")
        context_parts.append("")  # Empty line between papers

    return "\n".join(context_parts)


def extract_top_keywords(papers: list, n_keywords: int = 5) -> list:
    """
    Extract top keywords from a list of papers using TF-IDF.

    Parameters
    ----------
    papers : list
        List of paper dicts, each with optional 'title' and 'abstract' keys.
    n_keywords : int, optional
        Number of top keywords to return (default: 5).

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
            ngram_range=(2, 3),
        )
        tfidf_matrix = tfidf.fit_transform(docs)
        feature_names = tfidf.get_feature_names_out()
        mean_tfidf = tfidf_matrix.mean(axis=0).A1
        top_indices = mean_tfidf.argsort()[-n_keywords:][::-1]
        keywords = [feature_names[i] for i in top_indices if mean_tfidf[i] > 0]
        return keywords[:n_keywords]
    except Exception:
        return []

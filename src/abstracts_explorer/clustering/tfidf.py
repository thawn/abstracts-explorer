"""
TF-IDF keyword extraction for clusters and hierarchy nodes.

The vectorizer can be fitted once and reused across many target groups (e.g.
all clusters in one pass) via :func:`fit_tfidf` + :func:`keywords_from_matrix`.
"""

import logging
from typing import Any, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


def fit_tfidf(
    documents: List[str],
    min_df: int = 2,
) -> Tuple[Any, np.ndarray]:
    """
    Fit a TF-IDF vectorizer on a corpus.

    Parameters
    ----------
    documents : List[str]
        Full corpus of documents.
    min_df : int, optional
        Minimum document frequency for a term, by default 2.

    Returns
    -------
    tuple of (matrix, feature_names)
        Sparse TF-IDF matrix and the corresponding feature names.
    """
    tfidf = TfidfVectorizer(
        max_features=1000,
        min_df=min_df,
        stop_words="english",
        ngram_range=(1, 2),
    )
    matrix = tfidf.fit_transform(documents)
    return matrix, tfidf.get_feature_names_out()


def keywords_from_matrix(
    matrix: Any,
    feature_names: np.ndarray,
    target_indices: List[int],
    n_keywords: int,
) -> List[str]:
    """
    Return the top keywords for selected documents from a fitted matrix.

    Parameters
    ----------
    matrix : sparse matrix
        TF-IDF matrix returned by :func:`fit_tfidf`.
    feature_names : np.ndarray
        Feature names returned by :func:`fit_tfidf`.
    target_indices : List[int]
        Indices into the corpus identifying the documents to summarize.
    n_keywords : int
        Maximum number of keywords to return.

    Returns
    -------
    List[str]
        Keywords ordered by descending mean TF-IDF score.
    """
    if not target_indices or n_keywords <= 0:
        return []

    target_matrix = matrix[np.asarray(target_indices)]
    target_tfidf = np.asarray(target_matrix.mean(axis=0)).ravel()
    top_indices = target_tfidf.argsort()[-n_keywords:][::-1]
    keywords = [str(feature_names[i]) for i in top_indices if target_tfidf[i] > 0]
    return keywords[:n_keywords]


def compute_tfidf_keywords(
    documents: List[str],
    target_indices: List[int],
    n_keywords: int,
    min_df: int = 2,
) -> List[str]:
    """
    Compute top TF-IDF keywords for selected documents within a corpus.

    Fits a TF-IDF vectorizer on the full corpus, then ranks terms by their
    mean TF-IDF weight across ``target_indices``.

    Parameters
    ----------
    documents : List[str]
        Full corpus of documents used to fit the vectorizer.
    target_indices : List[int]
        Indices into ``documents`` identifying the documents to summarize.
    n_keywords : int
        Maximum number of keywords to return.
    min_df : int, optional
        Minimum document frequency for a term, by default 2

    Returns
    -------
    List[str]
        Keywords ordered by descending mean TF-IDF score. Empty if no valid
        documents or no positive-scoring terms are found.
    """
    if not documents or not target_indices or n_keywords <= 0:
        return []

    try:
        matrix, feature_names = fit_tfidf(documents, min_df=min_df)
        return keywords_from_matrix(matrix, feature_names, target_indices, n_keywords)
    except Exception as e:
        logger.warning(f"Failed to extract keywords: {e}")
        return []

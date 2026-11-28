"""
TF-IDF generation of per-paper keywords.

Keywords are generated once per ``(conference, year)`` corpus from each paper's
title and abstract.  The title is repeated once to boost title terms.  Terms are
stored as a comma-separated string in ``papers.generated_keywords``.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import text

from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.plugin import serialize_keywords_to_string

logger = logging.getLogger(__name__)

GENERATOR_VERSION = "tfidf-v1"

_MIN_KEYWORDS = 5
_MAX_KEYWORDS = 8
_MAX_FEATURES = 2000
_NGRAM_RANGE: Tuple[int, int] = (1, 2)
_BATCH_SIZE = 1000

_GENERIC_BLACKLIST = {
    "abstract",
    "analysis",
    "approach",
    "approaches",
    "article",
    "articles",
    "based",
    "competition",
    "competitions",
    "course",
    "courses",
    "data",
    "dataset",
    "datasets",
    "demonstrate",
    "demonstrates",
    "evaluation",
    "experiment",
    "experiments",
    "experimental",
    "framework",
    "frameworks",
    "introduce",
    "introduces",
    "method",
    "methods",
    "model",
    "models",
    "novel",
    "paper",
    "papers",
    "performance",
    "poster",
    "posters",
    "present",
    "presents",
    "problem",
    "problems",
    "propose",
    "proposed",
    "proposes",
    "result",
    "results",
    "show",
    "shows",
    "study",
    "studies",
    "task",
    "tasks",
    "technique",
    "techniques",
    "using",
    "video",
    "videos",
    "work",
    "works",
    "workshop",
    "workshops",
}


def _normalize_term(term: str) -> str:
    """Normalize a candidate term for blacklist and duplicate checks."""
    return " ".join(term.strip().lower().split())


def _is_valid_term(term: str, kept: Sequence[str]) -> bool:
    """
    Return whether *term* should be kept as a generated keyword.

    Parameters
    ----------
    term : str
        Candidate term.
    kept : sequence of str
        Terms already selected for the same paper.

    Returns
    -------
    bool
        ``True`` when the term is not blacklisted, not purely numeric, and not
        a substring/near duplicate of an already kept term.
    """
    normalized = _normalize_term(term)
    if not normalized or normalized in _GENERIC_BLACKLIST:
        return False

    tokens = re.findall(r"[a-z0-9]+", normalized)
    if not tokens:
        return False
    if any(token in _GENERIC_BLACKLIST for token in tokens):
        return False
    if any(len(token) <= 1 for token in tokens):
        return False
    if all(token.isdigit() for token in tokens):
        return False

    for existing in kept:
        existing_normalized = _normalize_term(existing)
        if (
            normalized == existing_normalized
            or normalized in existing_normalized
            or existing_normalized in normalized
        ):
            return False

    return True


def _clean_terms(terms: Sequence[str], max_keywords: int) -> List[str]:
    """Filter and deduplicate TF-IDF terms for one document."""
    kept: List[str] = []
    for term in terms:
        if _is_valid_term(term, kept):
            kept.append(term)
        if len(kept) >= max_keywords:
            break
    return kept


def _vectorizer(document_count: int):
    """Build a TF-IDF vectorizer sized for the corpus."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    min_df = 1 if document_count < 5 else 2
    max_df = 1.0 if document_count < 5 else 0.95
    return TfidfVectorizer(
        max_features=_MAX_FEATURES,
        min_df=min_df,
        max_df=max_df,
        stop_words="english",
        ngram_range=_NGRAM_RANGE,
    )


def top_terms_for_documents(
    documents: Sequence[str],
    *,
    max_keywords: int = _MAX_KEYWORDS,
) -> List[List[str]]:
    """
    Extract top TF-IDF terms for every document in a corpus.

    Parameters
    ----------
    documents : sequence of str
        Corpus documents.  Empty documents are allowed but yield no terms.
    max_keywords : int, optional
        Maximum number of terms to return per document (default 8).

    Returns
    -------
    list of list of str
        Cleaned terms for each input document, in input order.
    """
    if not documents:
        return []

    matrix: Any
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = _vectorizer(len(documents))
        matrix = vectorizer.fit_transform(documents)
    except ValueError:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            vectorizer = TfidfVectorizer(
                max_features=_MAX_FEATURES,
                min_df=1,
                max_df=1.0,
                stop_words="english",
                ngram_range=_NGRAM_RANGE,
            )
            matrix = vectorizer.fit_transform(documents)
        except ValueError:
            return [[] for _ in documents]

    feature_names = vectorizer.get_feature_names_out()
    results: List[List[str]] = []
    for index in range(matrix.shape[0]):
        row = matrix.getrow(index)
        if row.nnz == 0:
            results.append([])
            continue
        indices = row.indices
        scores = row.data
        ranked = indices[scores.argsort()[::-1]]
        terms = [str(feature_names[i]) for i in ranked]
        results.append(_clean_terms(terms, max_keywords))
    return results


def _build_document(title: Optional[str], abstract: Optional[str]) -> str:
    """Build the indexed text for a paper, boosting the title once."""
    clean_title = (title or "").strip()
    clean_abstract = (abstract or "").strip()
    if not clean_title and not clean_abstract:
        return ""
    return f"{clean_title}\n{clean_title}\n\n{clean_abstract}".strip()


def _fetch_scope_rows(
    db: DatabaseManager,
    conference: str,
    year: int,
    force: bool,
) -> List[Dict[str, Any]]:
    """Read the rows for a scope, filtering existing keywords unless forced."""
    stmt = text(
        "SELECT uid, title, abstract, generated_keywords FROM papers "
        "WHERE conference = :conference AND year = :year ORDER BY uid"
    )
    rows = db.session.execute(stmt, {"conference": conference, "year": year}).all()
    result: List[Dict[str, Any]] = []
    for row in rows:
        existing = row.generated_keywords
        if not force and existing and str(existing).strip():
            continue
        result.append(
            {
                "uid": str(row.uid),
                "title": row.title,
                "abstract": row.abstract,
                "generated_keywords": existing,
            }
        )
    return result


def generate_keywords_for_scope(
    db: DatabaseManager,
    conference: str,
    year: int,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> int:
    """
    Generate and store keywords for one ``(conference, year)`` scope.

    Parameters
    ----------
    db : DatabaseManager
        Open database connection.
    conference : str
        Conference name to process.
    year : int
        Conference year to process.
    force : bool, optional
        Regenerate keywords even when they already exist (default ``False``).
    dry_run : bool, optional
        Compute keywords without writing them to the database
        (default ``False``).

    Returns
    -------
    int
        Number of papers whose keywords were (or would be) updated.
    """
    rows = _fetch_scope_rows(db, conference, year, force)
    if not rows:
        return 0

    documents = [_build_document(row["title"], row["abstract"]) for row in rows]
    terms_per_document = top_terms_for_documents(documents)

    pending: List[Dict[str, str]] = []
    updated = 0
    for row, terms in zip(rows, terms_per_document):
        if not terms:
            continue
        serialized = serialize_keywords_to_string(terms)
        if not serialized:
            continue
        pending.append({"keywords": serialized, "uid": row["uid"]})
        updated += 1
        if not dry_run and len(pending) >= _BATCH_SIZE:
            db.session.execute(
                text("UPDATE papers SET generated_keywords = :keywords WHERE uid = :uid"),
                pending,
            )
            db.session.commit()
            pending.clear()

    if dry_run:
        return updated

    if pending:
        db.session.execute(
            text("UPDATE papers SET generated_keywords = :keywords WHERE uid = :uid"),
            pending,
        )
        db.session.commit()

    if updated:
        logger.info("Generated keywords for %d paper(s) in %s %s", updated, conference, year)
    return updated

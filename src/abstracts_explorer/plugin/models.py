"""
Plugin data models and validation helpers.

Defines :class:`LightweightPaper`, the Pydantic model returned by downloader
plugins, plus author/keyword serialization helpers used by the database layer.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ValidationError, field_validator

logger = logging.getLogger(__name__)


class LightweightPaper(BaseModel):
    """
    Lightweight paper model for plugin data validation.

    This model validates the schema returned by downloader plugins.
    It requires only essential fields and optionally supports additional metadata.

    Required Fields
    ---------------
    title : str
        Paper title
    authors : list
        List of author names (strings)
    abstract : str
        Paper abstract
    session : str
        Session/workshop/track name
    poster_position : str
        Poster position identifier
    year : int
        Conference year (e.g., 2025)
    conference : str
        Conference name (e.g., "NeurIPS", "ML4PS")

    Optional Fields
    ---------------
    original_id : int or str
        Paper ID from the original source (numeric ID or source slug/hash).
    paper_pdf_url : str
        URL to paper PDF
    poster_image_url : str
        URL to poster image
    url : str
        General URL (OpenReview, ArXiv, etc.)
    room_name : str
        Room name for presentation
    keywords : list
        List of keywords/tags
    starttime : str
        Start time
    endtime : str
        End time
    award : str
        Award name (e.g., "Best Paper Award")
    """

    # Required fields
    title: str
    authors: List[str]
    abstract: str
    session: str
    poster_position: str
    year: int
    conference: str

    # Optional fields
    original_id: Optional[Union[int, str]] = None
    paper_pdf_url: Optional[str] = None
    poster_image_url: Optional[str] = None
    url: Optional[str] = None
    room_name: Optional[str] = None
    keywords: Optional[List[str]] = None
    starttime: Optional[str] = None
    endtime: Optional[str] = None
    award: Optional[str] = None

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Ensure title is not empty."""
        if not v or not v.strip():
            raise ValueError("Paper title cannot be empty")
        return v.strip()

    @field_validator("authors")
    @classmethod
    def validate_authors(cls, v: List[str]) -> List[str]:
        """Ensure authors list is not empty and properly formatted.

        Empty or whitespace-only author entries are silently filtered out so
        that a single malformed entry does not cause the entire paper to be
        rejected.  A ``ValueError`` is still raised when the list is empty
        after filtering, or when any remaining entry contains a semicolon.
        """
        filtered = [author for author in v if author.strip()]
        if not filtered:
            raise ValueError("Authors list cannot be empty")
        for author in filtered:
            if ";" in author:
                raise ValueError("Author names cannot contain semicolons")
        return filtered

    @field_validator("abstract")
    @classmethod
    def validate_abstract(cls, v: str) -> str:
        """Ensure abstract is not empty."""
        if not v or not v.strip():
            raise ValueError("Paper abstract cannot be empty")
        return v.strip()

    @field_validator("session")
    @classmethod
    def validate_session(cls, v: str) -> str:
        """Ensure session is not empty."""
        if not v or not v.strip():
            raise ValueError("Session cannot be empty")
        return v.strip()

    @field_validator("conference")
    @classmethod
    def validate_conference(cls, v: str) -> str:
        """Ensure conference is not empty."""
        if not v or not v.strip():
            raise ValueError("Conference cannot be empty")
        return v.strip()

    @field_validator("year")
    @classmethod
    def validate_year(cls, v: int) -> int:
        """Ensure year is reasonable."""
        if v < 1900 or v > 2100:
            raise ValueError(f"Year {v} is not reasonable (must be between 1900 and 2100)")
        return v


def sanitize_author_names(authors: List[str]) -> List[str]:
    """
    Filter out semicolons from author names.

    Semicolons are not allowed in author names because they would interfere
    with the semicolon-separated format used to store authors in the database.
    This function replaces semicolons with spaces and normalizes whitespace.

    Parameters
    ----------
    authors : list of str
        List of author names to sanitize

    Returns
    -------
    list of str
        List of author names with semicolons replaced by spaces

    Examples
    --------
    >>> sanitize_author_names(["John Doe", "Jane; Smith", "Bob;Johnson"])
    ['John Doe', 'Jane Smith', 'Bob Johnson']

    >>> sanitize_author_names(["Alice"])
    ['Alice']

    >>> sanitize_author_names([])
    []

    >>> sanitize_author_names(["Multi;;Semicolons"])
    ['Multi Semicolons']

    Notes
    -----
    This function is useful when importing data from sources that may contain
    semicolons in author names. The LightweightPaper model will reject author
    names containing semicolons during validation.

    Multiple consecutive spaces are normalized to a single space.
    """
    return [re.sub(r"\s+", " ", author.replace(";", " ")).strip() for author in authors]


def serialize_authors_to_string(authors: List[str]) -> str:
    """
    Serialize a list of author names to a semicolon-separated string.

    Used by :func:`~abstracts_explorer.database.DatabaseManager.add_paper`
    to store a consistent semicolon-separated format in the SQL database.

    Parameters
    ----------
    authors : list of str
        Author names to serialize.  An empty list or ``None`` returns an empty
        string.

    Returns
    -------
    str
        Semicolon-separated author names, e.g. ``"Alice; Bob; Charlie"``.
        Returns an empty string if *authors* is empty or ``None``.
    """
    if not authors:
        return ""
    return "; ".join(str(a).strip() for a in authors)


def deserialize_authors_from_string(authors_str: str) -> List[str]:
    """
    Deserialize a semicolon-separated authors string to a list of author names.

    Used when loading paper metadata from ChromaDB to convert the stored
    semicolon-separated string back into a list format for validation and use in
    the application.

    Parameters
    ----------
    authors_str : str
        Semicolon-separated authors string, e.g. ``"Alice; Bob; Charlie"``.
        An empty string returns an empty list.

    Returns
    -------
    list of str
        List of author names, e.g. ``["Alice", "Bob", "Charlie"]``.
        Returns an empty list if *authors_str* is empty or only contains whitespace.
    """
    if not authors_str or not authors_str.strip():
        return []
    return [a.strip() for a in authors_str.split(";") if a.strip()]


def serialize_keywords_to_string(keywords: List[str]) -> str:
    """
    Serialize a list of keywords to a comma-separated string.

    Used by :func:`~abstracts_explorer.database.DatabaseManager.add_paper`
    to store a consistent comma-separated format in the SQL database.

    Parameters
    ----------
    keywords : list of str
        Keywords to serialize.  An empty list or ``None`` returns an empty
        string.

    Returns
    -------
    str
        Comma-separated keywords, e.g. ``"machine learning, deep learning"``.
        Returns an empty string if *keywords* is empty or ``None``.
    """
    if not keywords:
        return ""
    return ", ".join(str(k).strip() for k in keywords)


def deserialize_keywords_from_string(keywords_str: str) -> List[str]:
    """
    Deserialize a comma-separated keywords string to a list of keywords.

    Used when loading paper metadata from ChromaDB to convert the stored
    comma-separated string back into a list format for validation and use in
    the application.

    Parameters
    ----------
    keywords_str : str
        Comma-separated keywords string, e.g. ``"machine learning, deep learning"``.
        An empty string returns an empty list.

    Returns
    -------
    list of str
        List of keywords, e.g. ``["machine learning", "deep learning"]``.
        Returns an empty list if *keywords_str* is empty or only contains whitespace.
    """
    if not keywords_str or not keywords_str.strip():
        return []
    return [k.strip() for k in keywords_str.split(",") if k.strip()]


def validate_lightweight_paper(paper: Dict[str, Any]) -> LightweightPaper:
    """
    Validate a paper dict against the lightweight schema.

    Parameters
    ----------
    paper : dict
        Paper data to validate

    Returns
    -------
    LightweightPaper
        Validated paper model

    Raises
    ------
    ValidationError
        If the paper data is invalid
    """
    return LightweightPaper(**paper)


def validate_lightweight_papers(papers: List[Dict[str, Any]]) -> List[LightweightPaper]:
    """
    Validate a list of papers against the lightweight schema.

    Papers that fail validation are logged as warnings and skipped
    rather than aborting the entire import.

    Parameters
    ----------
    papers : list
        List of paper dicts to validate

    Returns
    -------
    list of LightweightPaper
        List of validated paper models (papers that failed validation are excluded)
    """
    validated: List[LightweightPaper] = []
    for paper in papers:
        try:
            validated.append(validate_lightweight_paper(paper))
        except ValidationError as exc:
            title = paper.get("title", "<unknown>")
            logger.warning("Skipping paper '%s': validation failed: %s", title, exc)
    if len(validated) < len(papers):
        logger.warning("Skipped %d of %d papers due to validation errors", len(papers) - len(validated), len(papers))
    return validated

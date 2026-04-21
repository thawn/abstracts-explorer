"""
Plugin Framework
================

This module provides the plugin framework for extending neurips-abstracts
with custom data downloaders.

The framework consists of:
- Base classes for plugin implementation (DownloaderPlugin, LightweightDownloaderPlugin)
- Schema conversion utilities (convert_to_lightweight_schema)
- Plugin registry for managing plugins (PluginRegistry)
- Pydantic models for data validation (LightweightPaper)
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from pydantic import BaseModel, ValidationError, field_validator

logger = logging.getLogger(__name__)


class DownloaderPlugin(ABC):
    """
    Base class for all downloader plugins.

    Each plugin must implement the download method and provide metadata
    about its capabilities.

    Subclasses should set ``_start_year`` and override ``get_url(year)``
    to get automatic ``supported_years`` computation with a current-year
    availability check.
    """

    # Plugin metadata (should be overridden in subclasses)
    plugin_name: str = "base"
    plugin_description: str = "Base downloader plugin"
    _start_year: int = 0

    @property
    def supported_years(self) -> List[int]:
        """
        Dynamically computed supported years.

        Builds the range ``[_start_year, current_year)`` and appends the
        current year when its data URL is already accessible (checked via
        a HEAD request to ``get_url(current_year)``).

        Returns
        -------
        list of int
            Supported conference years.
        """
        if not hasattr(self, "_supported_years_cache"):
            if self._start_year > 0:
                current_year = datetime.now().year
                years = list(range(self._start_year, current_year))
                if self._check_current_year_available(current_year):
                    years.append(current_year)
                self._supported_years_cache: List[int] = years
            else:
                self._supported_years_cache = []
        return self._supported_years_cache

    @supported_years.setter
    def supported_years(self, value: List[int]) -> None:
        self._supported_years_cache = value

    def get_url(self, year: int) -> str:
        """
        Get the data URL for a specific year.

        Override in subclasses to enable automatic current-year availability
        checking in :attr:`supported_years`.

        Parameters
        ----------
        year : int
            Conference year

        Returns
        -------
        str
            URL used for downloading or probing availability.

        Raises
        ------
        NotImplementedError
            When the subclass does not provide an implementation.
        """
        raise NotImplementedError("Subclasses must implement get_url()")

    def _check_current_year_available(self, current_year: int) -> bool:
        """
        Check whether the data URL for *current_year* is already reachable.

        Sends a HEAD request to ``get_url(current_year)`` with a 3-second
        timeout.  Returns ``False`` when ``get_url`` is not implemented or
        the request fails.

        Parameters
        ----------
        current_year : int
            Year to probe.

        Returns
        -------
        bool
            ``True`` when the URL responds with HTTP 200.
        """
        try:
            url = self.get_url(current_year)
            response = requests.head(url, timeout=3, allow_redirects=True)
            return response.status_code == 200
        except (requests.RequestException, NotImplementedError):
            logger.debug(
                "%s: current-year availability check failed for year %d",
                self.plugin_name,
                current_year,
            )
            return False

    @abstractmethod
    def download(
        self,
        year: Optional[int] = None,
        output_path: Optional[str] = None,
        force_download: bool = False,
        **kwargs: Any,
    ) -> List["LightweightPaper"]:
        """
        Download papers from the data source.

        Parameters
        ----------
        year : int, optional
            Year to download papers for (if applicable)
        output_path : str, optional
            Path to save the downloaded data
        force_download : bool
            Force re-download even if data exists
        **kwargs : Any
            Additional plugin-specific parameters

        Returns
        -------
        list of LightweightPaper
            List of validated paper objects ready for database insertion
        """
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get plugin metadata.

        Returns
        -------
        dict
            Plugin metadata including name, description, supported years, etc.
        """
        pass

    def validate_year(self, year: Optional[int]) -> None:
        """
        Validate that the requested year is supported.

        Parameters
        ----------
        year : int or None
            Year to validate

        Raises
        ------
        ValueError
            If year is not supported by this plugin
        """
        if year is not None and self.supported_years and year not in self.supported_years:
            raise ValueError(
                f"Year {year} not supported by {self.plugin_name}. " f"Supported years: {self.supported_years}"
            )


class LightweightDownloaderPlugin(DownloaderPlugin):
    """
    Lightweight base class for downloader plugins using simplified schema.

    This plugin type uses a simpler data format that only requires essential fields,
    making it easier to implement new plugins. The data is automatically converted
    to the full NeurIPS schema when loaded into the database.

    Required fields per paper:
        - title (str): Paper title
        - authors (list): List of author names (strings) or author dicts with 'fullname'
        - abstract (str): Paper abstract
        - session (str): Session/workshop/track name
        - poster_position (str): Poster position identifier
        - year (int): Conference year (e.g., 2025)
        - conference (str): Conference name (e.g., "NeurIPS", "ICLR")

    Optional fields per paper:
        - paper_pdf_url (str): URL to paper PDF
        - poster_image_url (str): URL to poster image
        - url (str): General URL (e.g., OpenReview, ArXiv)
        - room_name (str): Room name for presentation
        - keywords (list): List of keywords/tags
        - starttime (str): Start time (ISO format or readable string)
        - endtime (str): End time (ISO format or readable string)
        - id (int): Paper ID (auto-generated if not provided)
        - award (str): Award name (e.g., "Best Paper Award")
    """

    # Plugin metadata (should be overridden in subclasses)
    plugin_name: str = "lightweight_base"
    plugin_description: str = "Lightweight base downloader plugin"


"""
Schema Converter
================

Utilities for converting between lightweight and full NeurIPS schema formats.
"""


def convert_to_lightweight_schema(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert full NeurIPS schema to lightweight paper format.

    This function extracts only the fields needed for the lightweight schema
    from papers in the full NeurIPS format, making it easier to work with
    simplified data structures.

    Parameters
    ----------
    papers : list
        List of papers in full NeurIPS format with fields like:
        - id (int)
        - title or name (str) - will use 'title', fallback to 'name'
        - authors (list of dict with 'fullname' or list of str)
        - abstract (str)
        - session (str)
        - poster_position (str)
        - paper_pdf_url (str, optional)
        - poster_image_url (str, optional)
        - url (str, optional)
        - room_name (str, optional)
        - keywords (list or str, optional)
        - starttime (str, optional)
        - endtime (str, optional)
        - award or decision (str, optional)
        - year (int, optional)
        - conference (str, optional)

    Returns
    -------
    list of dict
        Papers in lightweight format with only essential fields.
        Authors are returned as lists of strings.

    Examples
    --------
    >>> papers = [
    ...     {
    ...         'id': 123,
    ...         'title': 'Deep Learning',
    ...         'authors': [
    ...             {'id': 1, 'fullname': 'John Doe', 'institution': 'MIT'},
    ...             {'id': 2, 'fullname': 'Jane Smith', 'institution': 'Stanford'}
    ...         ],
    ...         'abstract': 'A paper about deep learning',
    ...         'session': 'Session A',
    ...         'poster_position': 'A-42',
    ...         'paper_pdf_url': 'https://example.com/paper.pdf',
    ...         'year': 2025,
    ...         'conference': 'NeurIPS'
    ...     }
    ... ]
    >>> lightweight = convert_to_lightweight_schema(papers)
    >>> lightweight[0]['authors']
    ['John Doe', 'Jane Smith']

    Notes
    -----
    - Author objects are converted to lists of name strings
    - Extra NeurIPS-specific fields are dropped
    - 'name' field is converted to 'title' if needed
    - Keywords are converted from string to list if needed
    """
    lightweight_papers = []

    for paper in papers:
        # Extract title (handle both 'title' and legacy 'name')
        title = paper.get("title") or paper.get("name", "")
        if not title:
            continue  # Skip papers without title

        # Extract and convert authors to list
        authors_data = paper.get("authors", [])
        authors = []

        if isinstance(authors_data, list):
            for author in authors_data:
                if isinstance(author, dict):
                    # Extract fullname from dict
                    name = author.get("fullname") or author.get("name", "")
                    if name:
                        authors.append(name)
                elif isinstance(author, str):
                    # Already a string
                    authors.append(author)
        elif isinstance(authors_data, str):
            # Authors is a semicolon-separated string, split it
            authors = [a.strip() for a in authors_data.split(";") if a.strip()]

        # Sanitize author names to remove semicolons (required by LightweightPaper validation)
        authors = sanitize_author_names(authors)

        # Build lightweight paper
        # Note: Use 'or ""' pattern to handle None values from source data
        lightweight_paper = {
            "title": title,
            "authors": authors,
            "abstract": paper.get("abstract") or "",
            "session": paper.get("session") or "No session",
            "poster_position": paper.get("poster_position") or "",
            "year": paper.get("year") or 0,
            "conference": paper.get("conference") or "",
        }

        # Add optional fields if present
        if "id" in paper:
            lightweight_paper["original_id"] = paper["id"]

        if paper.get("paper_pdf_url"):
            lightweight_paper["paper_pdf_url"] = paper["paper_pdf_url"]
        elif paper.get("paper_url"):
            lightweight_paper["paper_pdf_url"] = paper["paper_url"]

        if paper.get("poster_image_url"):
            lightweight_paper["poster_image_url"] = paper["poster_image_url"]

        if paper.get("url"):
            lightweight_paper["url"] = paper["url"]

        if paper.get("room_name"):
            lightweight_paper["room_name"] = paper["room_name"]

        # Handle keywords (can be list or string in NeurIPS schema)
        keywords = paper.get("keywords")
        if keywords:
            if isinstance(keywords, str):
                # Convert string to list
                lightweight_paper["keywords"] = [k.strip() for k in keywords.split(",") if k.strip()]
            elif isinstance(keywords, list):
                lightweight_paper["keywords"] = keywords

        if paper.get("starttime"):
            lightweight_paper["starttime"] = paper["starttime"]

        if paper.get("endtime"):
            lightweight_paper["endtime"] = paper["endtime"]

        # Use award if present, otherwise fall back to decision
        decision = paper.get("decision") or ""
        award = paper.get("award") or (paper.get("decision") if "award" in decision.lower() else None)
        if award:
            lightweight_paper["award"] = award

        lightweight_papers.append(lightweight_paper)

    return lightweight_papers


"""
Plugin Registry
===============

Registry for managing and accessing downloader plugins.
"""

# DownloaderPlugin is defined earlier in this file

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Registry for managing downloader plugins."""

    def __init__(self):
        self._plugins: Dict[str, DownloaderPlugin] = {}

    def register(self, plugin: DownloaderPlugin) -> None:
        """
        Register a new plugin.

        Parameters
        ----------
        plugin : DownloaderPlugin
            Plugin instance to register
        """
        if not isinstance(plugin, DownloaderPlugin):
            raise TypeError(f"Plugin must be an instance of DownloaderPlugin, got {type(plugin)}")

        self._plugins[plugin.plugin_name] = plugin
        logger.info(f"Registered plugin: {plugin.plugin_name}")

    def unregister(self, plugin_name: str) -> None:
        """
        Unregister a plugin.

        Parameters
        ----------
        plugin_name : str
            Name of plugin to unregister
        """
        if plugin_name in self._plugins:
            del self._plugins[plugin_name]
            logger.info(f"Unregistered plugin: {plugin_name}")
        else:
            logger.warning(f"Plugin not found: {plugin_name}")

    def get(self, plugin_name: str) -> Optional[DownloaderPlugin]:
        """
        Get a plugin by name.

        Parameters
        ----------
        plugin_name : str
            Name of plugin to retrieve

        Returns
        -------
        DownloaderPlugin or None
            Plugin instance or None if not found
        """
        return self._plugins.get(plugin_name)

    def list_plugins(self) -> List[Dict[str, Any]]:
        """
        List all registered plugins with their metadata.

        Returns
        -------
        list
            List of plugin metadata dictionaries
        """
        return [plugin.get_metadata() for plugin in self._plugins.values()]

    def list_plugin_names(self) -> List[str]:
        """
        List names of all registered plugins.

        Returns
        -------
        list
            List of plugin names
        """
        return list(self._plugins.keys())


# Global plugin registry
_registry = PluginRegistry()


def register_plugin(plugin: DownloaderPlugin) -> None:
    """
    Register a plugin with the global registry.

    Parameters
    ----------
    plugin : DownloaderPlugin
        Plugin instance to register
    """
    _registry.register(plugin)


def get_plugin(plugin_name: str) -> Optional[DownloaderPlugin]:
    """
    Get a plugin from the global registry.

    Parameters
    ----------
    plugin_name : str
        Name of plugin to retrieve

    Returns
    -------
    DownloaderPlugin or None
        Plugin instance or None if not found
    """
    return _registry.get(plugin_name)


def list_plugins() -> List[Dict[str, Any]]:
    """
    List all registered plugins.

    Returns
    -------
    list
        List of plugin metadata dictionaries
    """
    return _registry.list_plugins()


def list_plugin_names() -> List[str]:
    """
    List names of all registered plugins.

    Returns
    -------
    list
        List of plugin names
    """
    return _registry.list_plugin_names()


def get_all_plugins() -> List[DownloaderPlugin]:
    """
    Get all registered plugin instances.

    Returns
    -------
    list of DownloaderPlugin
        List of all registered plugin instances
    """
    plugins = []
    for name in _registry.list_plugin_names():
        plugin = _registry.get(name)
        if plugin is not None:
            plugins.append(plugin)
    return plugins


def get_available_filters() -> Dict[str, Any]:
    """
    Get available conferences and years from registered plugins.

    Returns a mapping of conferences to their supported years based on
    the registered downloader plugins.

    Returns
    -------
    dict
        Dictionary with:
        - conferences: list of conference names
        - years: list of all unique years across all plugins
        - conference_years: dict mapping conference names to their supported years

    Examples
    --------
    >>> filters = get_available_filters()
    >>> print(filters['conferences'])  # ['NeurIPS', 'ICLR', ...]
    >>> print(filters['years'])  # [2025, 2024, 2023, ...]
    >>> print(filters['conference_years'])  # {'NeurIPS': [2025, 2024], ...}
    """
    # Get all registered plugins
    plugins = list_plugins()

    # Build mapping of conferences to years
    conference_years: Dict[str, List[int]] = {}
    all_years: set = set()

    for plugin_info in plugins:
        conference_name = plugin_info.get("conference_name")
        supported_years = plugin_info.get("supported_years", [])

        if conference_name and supported_years:
            if conference_name not in conference_years:
                conference_years[conference_name] = []
            conference_years[conference_name].extend(supported_years)
            all_years.update(supported_years)

    # Sort years and deduplicate
    all_years_sorted = sorted(list(all_years), reverse=True)
    conferences = sorted(conference_years.keys())

    # Sort years for each conference
    for conf in conference_years:
        conference_years[conf] = sorted(conference_years[conf], reverse=True)

    return {"conferences": conferences, "years": all_years_sorted, "conference_years": conference_years}


def resolve_conference_from_url(url_path: str) -> Optional[str]:
    """
    Resolve a URL path segment to a canonical conference name using plugins.

    Performs case-insensitive matching against:
    1. Plugin conference names (e.g. ``NeurIPS``, ``ICLR``)
    2. Plugin names (e.g. ``neurips``, ``iclr``)

    Parameters
    ----------
    url_path : str
        URL path segment to resolve (e.g. ``"neurips"``, ``"ICLR"``).

    Returns
    -------
    str or None
        The canonical conference name if a match is found, ``None`` otherwise.

    Examples
    --------
    >>> resolve_conference_from_url("neurips")  # matches plugin name
    'NeurIPS'
    >>> resolve_conference_from_url("ICLR")  # matches conference name
    'ICLR'
    >>> resolve_conference_from_url("unknown") is None
    True
    """
    lookup: Dict[str, str] = {}
    available = get_available_filters()
    for conf in available.get("conferences", []):
        lookup[conf.lower()] = conf

    for meta in list_plugins():
        plugin_name = meta.get("name", "")
        conf_name = meta.get("conference_name", "")
        if plugin_name and conf_name:
            lookup[plugin_name.lower()] = conf_name

    return lookup.get(url_path.lower())


"""
Plugin Data Models
==================

Pydantic models for validating plugin data in lightweight schema format.
"""


# ============================================================================
# Lightweight Schema Models (for LightweightDownloaderPlugin)
# ============================================================================


class LightweightPaper(BaseModel):
    """
    Lightweight paper model for plugin data validation.

    This model validates the simplified schema used by LightweightDownloaderPlugin.
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
    original_id : int
        Paper ID from the original source
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
    original_id: Optional[int] = None
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
        # Filter out empty/whitespace-only names rather than rejecting the paper
        filtered = [author for author in v if author.strip()]
        if not filtered:
            raise ValueError("Authors list cannot be empty")
        for author in filtered:
            # no semicolons allowed in author names
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


# ============================================================================
# Validation Helper Functions
# ============================================================================


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
    import re

    return [re.sub(r"\s+", " ", author.replace(";", " ")).strip() for author in authors]


def serialize_authors_to_string(authors: List[str]) -> str:
    """
    Serialize a list of author names to a semicolon-separated string.

    Used by both :func:`~abstracts_explorer.database.DatabaseManager.add_paper`
    and :meth:`~abstracts_explorer.embeddings.EmbeddingsManager._serialize_metadata_for_chromadb`
    to ensure a consistent storage format in both the SQL database and ChromaDB.

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

    Used by both :func:`~abstracts_explorer.database.DatabaseManager.add_paper`
    and :meth:`~abstracts_explorer.embeddings.EmbeddingsManager._serialize_metadata_for_chromadb`
    to ensure a consistent storage format in both the SQL database and ChromaDB.

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


def prepare_chroma_db_paper_data(paper: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare paper data from chroma_db for validation by normalizing and converting fields.

    Parameters
    ----------
    paper : dict
        Paper data to prepare

    Returns
    -------
    dict
        Prepared paper data
    """
    # Split authors and keywords into lists, stripping whitespace from each entry
    paper["authors"] = [a.strip() for a in paper["authors"].split(";") if a.strip()]
    paper["year"] = int(paper["year"])
    if "keywords" in paper:
        kws = [k.strip() for k in paper["keywords"].split(",") if k.strip()]
        if kws:
            paper["keywords"] = kws
        else:
            # Empty string → remove so LightweightPaper defaults keywords to None
            del paper["keywords"]
    if "original_id" in paper:
        if paper["original_id"]:
            paper["original_id"] = int(paper["original_id"])
        else:
            del paper["original_id"]
    # Remove empty strings for optional string fields so they default to None in
    # LightweightPaper, preserving round-trip fidelity with the SQL database (where
    # None values are serialised to "" by _serialize_metadata_for_chromadb).
    for field in ("paper_pdf_url", "poster_image_url", "url", "room_name", "starttime", "endtime", "award"):
        if field in paper and paper[field] == "":
            del paper[field]
    return paper


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


# Export public API
__all__ = [
    # Plugin base classes
    "DownloaderPlugin",
    "LightweightDownloaderPlugin",
    # Registry
    "PluginRegistry",
    "register_plugin",
    "get_plugin",
    "list_plugins",
    "list_plugin_names",
    # Conversion utilities
    "convert_to_lightweight_schema",
    # Pydantic models
    "LightweightPaper",
    # Validation functions
    "sanitize_author_names",
    "validate_lightweight_paper",
    "validate_lightweight_papers",
]

"""
Plugin Framework
================

This module provides the plugin framework for extending neurips-abstracts
with custom data downloaders.

The framework consists of:
- Base classes for plugin implementation (DownloaderPlugin, LightweightDownloaderPlugin)
- Schema conversion utilities (convert_lightweight_to_neurips_schema)
- Plugin registry for managing plugins (PluginRegistry)
- Pydantic models for data validation (LightweightPaper, PaperModel, etc.)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class DownloaderPlugin(ABC):
    """
    Base class for all downloader plugins.

    Each plugin must implement the download method and provide metadata
    about its capabilities.
    """

    # Plugin metadata (should be overridden in subclasses)
    plugin_name: str = "base"
    plugin_description: str = "Base downloader plugin"
    supported_years: List[int] = []

    @abstractmethod
    def download(
        self,
        year: Optional[int] = None,
        output_path: Optional[str] = None,
        force_download: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
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
        dict
            Downloaded data in the standardized format:
            {
                'count': int,
                'next': None,
                'previous': None,
                'results': [list of papers]
            }
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
    supported_years: List[int] = []

    # No need to redefine download() and get_metadata() - inherited from DownloaderPlugin
    # No need to redefine validate_year() - inherited from DownloaderPlugin


"""
Schema Converter
================

Utilities for converting between lightweight and full NeurIPS schema formats.
"""

from typing import Any, Dict, List
from datetime import datetime


def convert_lightweight_to_neurips_schema(
    papers: List[Dict[str, Any]],
    session_default: str = "Workshop",
    event_type: str = "Poster",
    source_url: str = "",
) -> Dict[str, Any]:
    """
    Convert lightweight paper format to full NeurIPS schema.

    Parameters
    ----------
    papers : list
        List of papers in lightweight format with required fields:
        - title (str)
        - authors (list of str or list of dict with 'fullname')
        - abstract (str)
        - session (str)
        - poster_position (str)

        And optional fields:
        - paper_pdf_url (str)
        - poster_image_url (str)
        - url (str)
        - room_name (str)
        - keywords (list)
        - starttime (str)
        - endtime (str)
        - id (int)
        - award (str)
    session_default : str
        Default session name if not specified in paper
    event_type : str
        Event type (e.g., "Poster", "Oral", "Workshop Poster")
    source_url : str
        Source URL for the data

    Returns
    -------
    dict
        Data in full NeurIPS schema format

    Raises
    ------
    ValueError
        If required fields are missing
    """
    results = []
    author_id_counter = 1

    for idx, paper in enumerate(papers):
        # Validate required fields
        required_fields = ["title", "authors", "abstract", "session", "poster_position"]
        missing_fields = [f for f in required_fields if f not in paper]
        if missing_fields:
            raise ValueError(
                f"Paper at index {idx} missing required fields: {missing_fields}. "
                f"Required fields: {required_fields}"
            )

        # Get or generate paper ID
        paper_id = paper.get("id", idx + 1)

        # Generate UID using conference, year, and paper_id
        conference = paper.get("conference", "unknown")
        year = paper.get("year", "unknown")
        uid = f"{conference}_{year}_{paper_id}"

        # Process authors
        authors_list = []
        author_data = paper["authors"]

        if isinstance(author_data, list):
            for author in author_data:
                if isinstance(author, str):
                    # Simple string author
                    authors_list.append({"id": author_id_counter, "fullname": author, "url": "", "institution": ""})
                    author_id_counter += 1
                elif isinstance(author, dict):
                    # Author dict - use provided data
                    authors_list.append(
                        {
                            "id": author.get("id", author_id_counter),
                            "fullname": author.get("fullname", author.get("name", "Unknown")),
                            "url": author.get("url", ""),
                            "institution": author.get("institution", ""),
                        }
                    )
                    author_id_counter += 1

        # Build eventmedia list
        eventmedia = []
        media_id = paper_id * 1000
        timestamp = datetime.now().isoformat()

        # Add URL if provided
        if paper.get("url"):
            eventmedia.append(
                {
                    "id": media_id + 1,
                    "modified": timestamp,
                    "display_section": 1,
                    "type": "URL",
                    "name": "Paper Link",
                    "visible": True,
                    "sortkey": 0,
                    "is_live_content": False,
                    "uri": paper["url"],
                    "resourcetype": "UriEventmedia",
                }
            )

        # Add poster image if provided
        if paper.get("poster_image_url"):
            eventmedia.append(
                {
                    "id": media_id + 2,
                    "file": paper["poster_image_url"],
                    "modified": timestamp,
                    "display_section": 1,
                    "type": "Poster",
                    "name": "Poster",
                    "visible": True,
                    "sortkey": 0,
                    "is_live_content": False,
                    "detailed_kind": "",
                    "generated_from": None,
                    "resourcetype": "EventmediaImageFile",
                }
            )

        # Add PDF if provided
        if paper.get("paper_pdf_url"):
            eventmedia.append(
                {
                    "id": media_id + 3,
                    "modified": timestamp,
                    "display_section": 1,
                    "type": "PDF",
                    "name": "Paper PDF",
                    "visible": True,
                    "sortkey": 0,
                    "is_live_content": False,
                    "uri": paper["paper_pdf_url"],
                    "resourcetype": "UriEventmedia",
                }
            )

        # Create full paper entry
        neurips_paper = {
            "id": paper_id,
            "uid": uid,
            "name": paper["title"],
            "authors": authors_list,
            "abstract": paper["abstract"],
            "topic": paper.get("topic", session_default),
            "keywords": paper.get("keywords", []),
            "decision": paper.get("award") or paper.get("decision", "Accept (poster)"),
            "session": paper["session"],
            "eventtype": event_type,
            "event_type": event_type,
            "room_name": paper.get("room_name", ""),
            "virtualsite_url": paper.get("virtualsite_url", ""),
            "url": paper.get("url", ""),
            "sourceid": None,
            "sourceurl": source_url,
            "starttime": paper.get("starttime", ""),
            "endtime": paper.get("endtime", ""),
            "starttime2": None,
            "endtime2": None,
            "diversity_event": None,
            "paper_url": paper.get("url", ""),
            "paper_pdf_url": paper.get("paper_pdf_url", ""),
            "children_url": None,
            "children": [],
            "children_ids": [],
            "parent1": "",
            "parent2": None,
            "parent2_id": None,
            "eventmedia": eventmedia,
            "show_in_schedule_overview": False,
            "visible": True,
            "poster_position": paper["poster_position"],
            "schedule_html": "",
            "latitude": None,
            "longitude": None,
            "related_events": [],
            "related_events_ids": [],
            "year": paper.get("year"),
            "conference": paper.get("conference", ""),
        }

        results.append(neurips_paper)

    return {"count": len(results), "next": None, "previous": None, "results": results}


"""
Plugin Registry
===============

Registry for managing and accessing downloader plugins.
"""

from typing import Any, Dict, List, Optional
import logging

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


"""
Plugin Data Models
==================

Pydantic models for validating plugin data in both lightweight and full schema formats.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator


# ============================================================================
# Lightweight Schema Models (for LightweightDownloaderPlugin)
# ============================================================================


class LightweightAuthor(BaseModel):
    """
    Lightweight author model for plugins.

    Can be a simple string (author name) or a dict with additional fields.
    """

    name: str
    affiliation: Optional[str] = None
    email: Optional[str] = None

    model_config = ConfigDict(extra="allow")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure name is not empty."""
        if not v or not v.strip():
            raise ValueError("Author name cannot be empty")
        return v.strip()


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
        List of author names (strings) or LightweightAuthor objects
    abstract : str
        Paper abstract
    session : str
        Session/workshop/track name
    poster_position : str
        Poster position identifier

    Optional Fields
    ---------------
    id : int
        Paper ID (auto-generated if not provided)
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
    year : int
        Conference year (e.g., 2025)
    conference : str
        Conference name (e.g., "NeurIPS", "ML4PS")
    """

    # Required fields
    title: str
    authors: List[Union[str, Dict[str, Any]]]
    abstract: str
    session: str
    poster_position: str

    # Optional fields
    id: Optional[int] = None
    paper_pdf_url: Optional[str] = None
    poster_image_url: Optional[str] = None
    url: Optional[str] = None
    room_name: Optional[str] = None
    keywords: Optional[List[str]] = None
    starttime: Optional[str] = None
    endtime: Optional[str] = None
    award: Optional[str] = None
    year: Optional[int] = None
    conference: Optional[str] = None

    model_config = ConfigDict(extra="allow")  # Allow extra fields

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Ensure title is not empty."""
        if not v or not v.strip():
            raise ValueError("Paper title cannot be empty")
        return v.strip()

    @field_validator("authors")
    @classmethod
    def validate_authors(cls, v: List[Union[str, Dict[str, Any]]]) -> List[Union[str, Dict[str, Any]]]:
        """Ensure authors list is not empty."""
        if not v or len(v) == 0:
            raise ValueError("Authors list cannot be empty")
        return v

    @field_validator("session")
    @classmethod
    def validate_session(cls, v: str) -> str:
        """Ensure session is not empty."""
        if not v or not v.strip():
            raise ValueError("Session cannot be empty")
        return v.strip()


# ============================================================================
# Full Schema Models (for DownloaderPlugin and database)
# ============================================================================


class EventMediaModel(BaseModel):
    """
    Pydantic model for event media item validation.

    Attributes
    ----------
    id : int, optional
        Media item identifier.
    type : str, optional
        Type of media (e.g., "Poster", "URL", "Image").
    name : str, optional
        Name/description of the media item.
    file : str, optional
        File path for the media (e.g., "/media/PosterPDFs/...").
    url : str, optional
        Direct URL to the media.
    uri : str, optional
        URI/link (e.g., OpenReview link).
    modified : str, optional
        Modification timestamp.
    display_section : int, optional
        Display section number.
    visible : bool, optional
        Visibility flag.
    sortkey : int, optional
        Sort key for ordering.
    is_live_content : bool, optional
        Flag indicating if content is live.
    detailed_kind : str, optional
        Detailed kind/subtype (e.g., "thumb" for thumbnails).
    generated_from : int, optional
        ID of the source media this was generated from.
    resourcetype : str, optional
        Resource type identifier (e.g., "UriEventmedia", "EventmediaImageFile").
    """

    id: Optional[int] = None
    type: Optional[str] = ""
    name: Optional[str] = ""
    file: Optional[str] = None
    uri: Optional[str] = None
    modified: Optional[str] = None
    display_section: Optional[int] = None
    visible: Optional[bool] = None
    sortkey: Optional[int] = None
    is_live_content: Optional[bool] = None
    detailed_kind: Optional[str] = None
    generated_from: Optional[int] = None
    resourcetype: Optional[str] = None

    model_config = ConfigDict(extra="allow")  # Allow extra fields not in the model


class AuthorModel(BaseModel):
    """
    Pydantic model for author data validation.

    Attributes
    ----------
    id : int
        Unique author identifier.
    fullname : str
        Full name of the author.
    url : str, optional
        URL to author profile.
    institution : str, optional
        Author's institution.
    """

    id: int
    fullname: str
    url: Optional[str] = ""
    institution: Optional[str] = ""

    @field_validator("fullname")
    @classmethod
    def validate_fullname(cls, v: str) -> str:
        """Ensure fullname is not empty."""
        if not v or not v.strip():
            raise ValueError("Author fullname cannot be empty")
        return v.strip()


class PaperModel(BaseModel):
    """
    Pydantic model for paper data validation (full NeurIPS schema).

    Attributes
    ----------
    id : int
        Unique paper identifier.
    name : str
        Paper title.
    authors : list or str, optional
        List of author objects or comma-separated string of author IDs.
    abstract : str, optional
        Paper abstract.
    uid : str, optional
        Unique identifier string.
    topic : str, optional
        Paper topic/category.
    keywords : list or str, optional
        List of keywords or comma-separated string.
    decision : str, optional
        Acceptance decision (e.g., "Accept (poster)").
    session : str, optional
        Conference session.
    eventtype : str, optional
        Type of event (e.g., "Poster", "Oral").
    event_type : str, optional
        Alternative event type field.
    room_name : str, optional
        Room location.
    virtualsite_url : str, optional
        Virtual site URL.
    url : str, optional
        General URL.
    sourceid : int, optional
        Source identifier.
    sourceurl : str, optional
        Source URL.
    starttime : str, optional
        Event start time.
    endtime : str, optional
        Event end time.
    starttime2 : str, optional
        Alternative start time.
    endtime2 : str, optional
        Alternative end time.
    diversity_event : str, optional
        Diversity event indicator.
    paper_url : str, optional
        Paper URL.
    paper_pdf_url : str, optional
        PDF URL.
    children_url : str, optional
        Children URL.
    children : list, optional
        Child events.
    children_ids : list, optional
        Child event IDs.
    parent1 : str, optional
        First parent.
    parent2 : str, optional
        Second parent.
    parent2_id : str, optional
        Second parent ID.
    eventmedia : list, optional
        Event media items.
    show_in_schedule_overview : bool, optional
        Schedule visibility flag.
    visible : bool, optional
        General visibility flag.
    poster_position : str, optional
        Poster position/number.
    schedule_html : str, optional
        Schedule HTML content.
    latitude : float, optional
        Location latitude.
    longitude : float, optional
        Location longitude.
    related_events : list, optional
        Related events.
    related_events_ids : list, optional
        Related event IDs.
    year : int, optional
        Conference year (e.g., 2025).
    conference : str, optional
        Conference name (e.g., "NeurIPS", "ML4PS").
    """

    id: int
    name: str
    authors: Optional[Union[List[Dict[str, Any]], str]] = ""
    abstract: Optional[str] = ""
    uid: Optional[str] = ""
    topic: Optional[str] = ""
    keywords: Optional[Union[List[str], str]] = ""
    decision: Optional[str] = ""
    session: Optional[str] = ""
    eventtype: Optional[str] = ""
    event_type: Optional[str] = ""
    room_name: Optional[str] = ""
    virtualsite_url: Optional[str] = ""
    url: Optional[str] = None
    sourceid: Optional[int] = None
    sourceurl: Optional[str] = ""
    starttime: Optional[str] = ""
    endtime: Optional[str] = ""
    starttime2: Optional[str] = None
    endtime2: Optional[str] = None
    diversity_event: Optional[Union[str, bool]] = None
    paper_url: Optional[str] = ""
    paper_pdf_url: Optional[str] = None
    children_url: Optional[str] = None
    children: Optional[List[Any]] = Field(default_factory=list)
    children_ids: Optional[List[Any]] = Field(default_factory=list)
    parent1: Optional[str] = ""
    parent2: Optional[str] = None
    parent2_id: Optional[str] = None
    eventmedia: Optional[List[Any]] = Field(default_factory=list)
    show_in_schedule_overview: Optional[bool] = False
    visible: Optional[bool] = True
    poster_position: Optional[str] = ""
    schedule_html: Optional[str] = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    related_events: Optional[List[Any]] = Field(default_factory=list)
    related_events_ids: Optional[List[Any]] = Field(default_factory=list)
    year: Optional[int] = None
    conference: Optional[str] = ""

    model_config = ConfigDict(extra="allow")  # Allow extra fields not in the model

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure paper name is not empty."""
        if not v or not v.strip():
            raise ValueError("Paper name cannot be empty")
        return v.strip()

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: Any) -> int:
        """Ensure ID is a valid integer."""
        if v is None:
            raise ValueError("Paper ID cannot be None")
        try:
            return int(v)
        except (ValueError, TypeError):
            raise ValueError(f"Paper ID must be an integer, got {type(v).__name__}")


# ============================================================================
# Validation Helper Functions
# ============================================================================


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


def validate_paper(paper: Dict[str, Any]) -> PaperModel:
    """
    Validate a paper dict against the full NeurIPS schema.

    Parameters
    ----------
    paper : dict
        Paper data to validate

    Returns
    -------
    PaperModel
        Validated paper model

    Raises
    ------
    ValidationError
        If the paper data is invalid
    """
    return PaperModel(**paper)


def validate_lightweight_papers(papers: List[Dict[str, Any]]) -> List[LightweightPaper]:
    """
    Validate a list of papers against the lightweight schema.

    Parameters
    ----------
    papers : list
        List of paper dicts to validate

    Returns
    -------
    list of LightweightPaper
        List of validated paper models

    Raises
    ------
    ValidationError
        If any paper data is invalid
    """
    return [validate_lightweight_paper(paper) for paper in papers]


def validate_papers(papers: List[Dict[str, Any]]) -> List[PaperModel]:
    """
    Validate a list of papers against the full NeurIPS schema.

    Parameters
    ----------
    papers : list
        List of paper dicts to validate

    Returns
    -------
    list of PaperModel
        List of validated paper models

    Raises
    ------
    ValidationError
        If any paper data is invalid
    """
    return [validate_paper(paper) for paper in papers]


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
    "convert_lightweight_to_neurips_schema",
    # Pydantic models
    "LightweightAuthor",
    "LightweightPaper",
    "EventMediaModel",
    "AuthorModel",
    "PaperModel",
    # Validation functions
    "validate_lightweight_paper",
    "validate_lightweight_papers",
    "validate_paper",
    "validate_papers",
]

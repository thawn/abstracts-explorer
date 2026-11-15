"""
Plugin base classes.

Defines :class:`DownloaderPlugin` and :class:`LightweightDownloaderPlugin`,
including year handling and the per-year JSON cache helpers shared by all
downloaders.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from abstracts_explorer.plugin.models import LightweightPaper, validate_lightweight_papers

logger = logging.getLogger(__name__)


@dataclass
class DownloadOptions:
    """
    Common per-download options shared by downloader plugins.

    Plugins resolve this from the ``**kwargs`` passed to
    :meth:`DownloaderPlugin.download`, layering their own defaults on top of
    the fields below.
    """

    timeout: float = 30
    verify_ssl: bool = True
    request_delay: float = 0.0
    max_workers: int = 1
    page_delay: float = 0.0
    resume: bool = True
    progress: Optional[bool] = None

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any], defaults: Optional[Dict[str, Any]] = None) -> "DownloadOptions":
        """
        Build options from a download ``**kwargs`` mapping.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments passed to ``download``.
        defaults : dict, optional
            Plugin-specific defaults that take precedence over the dataclass
            defaults when a key is absent from *kwargs*.

        Returns
        -------
        DownloadOptions
            Resolved options.
        """
        defaults = defaults or {}
        values = {}
        for field_name in cls.__dataclass_fields__:
            values[field_name] = kwargs.get(field_name, defaults.get(field_name, getattr(cls, field_name)))
        return cls(**values)


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

    def download(
        self,
        year: Optional[int] = None,
        output_path: Optional[str] = None,
        force_download: bool = False,
        **kwargs: Any,
    ) -> List[LightweightPaper]:
        """
        Download papers from the data source.

        This template resolves the year and options, serves a cached JSON
        file when available, delegates the source-specific work to
        :meth:`_download_uncached`, and persists the result. Plugins with
        fundamentally different input sources (e.g. a manual JSON file) may
        override this method and call the shared helpers directly.

        Parameters
        ----------
        year : int, optional
            Year to download papers for. Defaults to the newest supported year.
        output_path : str, optional
            Path to save the downloaded data. When the file exists and
            *force_download* is false, it is loaded instead of downloading.
        force_download : bool
            Force a fresh download even when *output_path* exists.
        **kwargs : Any
            Plugin-specific options (see :class:`DownloadOptions`).

        Returns
        -------
        list of LightweightPaper
            List of validated paper objects ready for database insertion.
        """
        options = self.resolve_options(kwargs)
        year = self._resolve_year(year)
        self.validate_year(year)

        if output_path and not force_download:
            cached = self._load_papers_json(output_path, self._cache_source_label)
            if cached is not None:
                return self._handle_cached(cached, year, output_path, options)

        papers = self._download_uncached(year, output_path, options)
        self._save_papers_json(papers, output_path, self._save_label)
        return papers

    @property
    def _cache_source_label(self) -> str:
        """Label used when reporting cache load failures."""
        return getattr(self, "_SOURCE_LABEL", "data")

    @property
    def _save_label(self) -> str:
        """Label used when logging saved papers."""
        return getattr(self, "conference_name", self.plugin_name)

    def resolve_options(self, kwargs: Dict[str, Any], **defaults: Any) -> DownloadOptions:
        """
        Resolve :class:`DownloadOptions` from download kwargs.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments passed to :meth:`download`.
        **defaults : Any
            Plugin-specific defaults overriding the dataclass defaults.

        Returns
        -------
        DownloadOptions
            Resolved options.
        """
        base: Dict[str, Any] = {
            "timeout": getattr(self, "timeout", 30),
            "verify_ssl": getattr(self, "verify_ssl", True),
            "request_delay": getattr(self, "_DEFAULT_REQUEST_DELAY", 0.0),
            "max_workers": getattr(self, "_DEFAULT_MAX_WORKERS", 1),
            "page_delay": getattr(self, "_DEFAULT_PAGE_DELAY", 0.0),
            "resume": getattr(self, "_DEFAULT_RESUME", True),
            "progress": getattr(self, "_DEFAULT_PROGRESS", None),
        }
        base.update(defaults)
        return DownloadOptions.from_kwargs(kwargs, base)

    def _resolve_year(self, year: Optional[int]) -> int:
        """
        Resolve the download year, defaulting to the newest supported year.

        Parameters
        ----------
        year : int, optional
            Requested year.

        Returns
        -------
        int
            The requested year, or ``max(supported_years)`` when omitted.
        """
        if year is not None:
            return year
        return max(self.supported_years, default=self._start_year)

    def _handle_cached(
        self,
        cached: List[LightweightPaper],
        year: int,
        output_path: str,
        options: DownloadOptions,
    ) -> List[LightweightPaper]:
        """
        Apply plugin-specific handling to a successful cache hit.

        Parameters
        ----------
        cached : list of LightweightPaper
            Papers loaded from *output_path*.
        year : int
            Requested year.
        output_path : str
            Cache file that was loaded.
        options : DownloadOptions
            Resolved download options.

        Returns
        -------
        list of LightweightPaper
            Papers to return to the caller.
        """
        return cached

    def _download_uncached(
        self,
        year: int,
        output_path: Optional[str],
        options: DownloadOptions,
    ) -> List[LightweightPaper]:
        """
        Perform the source-specific download.

        Parameters
        ----------
        year : int
            Year to download.
        output_path : str, optional
            Cache path (for plugins that track sidecars/checkpoints).
        options : DownloadOptions
            Resolved download options.

        Returns
        -------
        list of LightweightPaper
            Downloaded and validated papers.
        """
        raise NotImplementedError("Subclasses must implement _download_uncached()")

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

    def _save_papers_json(self, papers: List["LightweightPaper"], output_path: Optional[str], label: str) -> None:
        """
        Save validated papers to a per-year lightweight JSON file.

        Skips the write entirely when *papers* is empty so a year with no
        abstracts does not leave a stray ``[]`` file in the data directory.

        Parameters
        ----------
        papers : list of LightweightPaper
            Papers to persist.
        output_path : str or None
            Destination path. When ``None``, nothing is written.
        label : str
            Conference/source name used only in log messages.
        """
        if output_path and papers:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            papers_json = [paper.model_dump() for paper in papers]
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(papers_json, f, indent=2, ensure_ascii=False)
            logger.info("Saved %d %s papers to %s", len(papers), label, output_path)
        elif output_path:
            logger.info("No papers found; skipping save of %s", output_path)

    def _load_papers_json(self, output_path: Optional[str], source_label: str) -> Optional[List["LightweightPaper"]]:
        """
        Load cached lightweight papers from a previously written JSON file.

        Returns ``None`` when there is nothing to load (no path given, file
        absent, or the file cannot be parsed), signalling the caller to fall
        through to a fresh download.

        Parameters
        ----------
        output_path : str or None
            Path to an existing lightweight JSON file.
        source_label : str
            Where the fresh download comes from, used only in the fallback
            warning message.

        Returns
        -------
        list of LightweightPaper or None
        """
        if output_path and Path(output_path).exists():
            logger.info("Loading existing %s from: %s", source_label, output_path)
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                papers = validate_lightweight_papers(cached_data)
                logger.info("Successfully loaded %d papers from local file", len(papers))
                return papers
            except (json.JSONDecodeError, Exception) as e:
                logger.warning("Failed to load local file: %s. Downloading from %s...", e, source_label)
        return None


class LightweightDownloaderPlugin(DownloaderPlugin):
    """
    Lightweight base class for downloader plugins using the simplified schema.

    This plugin type returns :class:`~abstracts_explorer.plugin.LightweightPaper`
    objects, which the database layer stores directly.

    Required fields per paper:
        - title (str): Paper title
        - authors (list of str): Author names
        - abstract (str): Paper abstract
        - session (str): Session/workshop/track name
        - poster_position (str): Poster position identifier
        - year (int): Conference year (e.g., 2025)
        - conference (str): Conference name (e.g., "NeurIPS", "ICLR")

    Optional fields per paper:
        - original_id (int): Paper ID from the original source
        - paper_pdf_url (str): URL to paper PDF
        - poster_image_url (str): URL to poster image
        - url (str): General URL (e.g., OpenReview, ArXiv)
        - room_name (str): Room name for presentation
        - keywords (list): List of keywords/tags
        - starttime (str): Start time (ISO format or readable string)
        - endtime (str): End time (ISO format or readable string)
        - award (str): Award name (e.g., "Best Paper Award")
    """

    # Plugin metadata (should be overridden in subclasses)
    plugin_name: str = "lightweight_base"
    plugin_description: str = "Lightweight base downloader plugin"

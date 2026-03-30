"""
CHI Conference Downloader Plugin
=================================

Plugin for loading papers from the CHI (ACM Conference on Human Factors in
Computing Systems) conference.

Data source: ``programs.sigchi.org/chi/<year>``

The plugin first looks for a pre-downloaded JSON file at
``data/CHI_{year}_program.json``.  If the file is not found, it instructs
the user to download it manually from the SIGCHI conference program PWA
by clicking the **"Get conference data JSON"** button.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from abstracts_explorer.config import get_config
from abstracts_explorer.plugin import (
    LightweightDownloaderPlugin,
    LightweightPaper,
    validate_lightweight_papers,
)
from pydantic import ValidationError

logger = logging.getLogger(__name__)


class CHIDownloaderPlugin(LightweightDownloaderPlugin):
    """
    Plugin for loading CHI conference papers from the SIGCHI program JSON.

    CHI data is distributed via a Progressive Web App (PWA) at
    ``programs.sigchi.org/chi/<year>``.  A "Get conference data JSON"
    button on that page downloads a JSON file whose structure is parsed
    by this plugin.

    The downloaded JSON contains the following top-level keys:

    - ``conference``   – conference metadata (year, fullName, …)
    - ``contentTypes`` – list of content-type records (Paper, LBW, …)
    - ``contents``     – list of paper/presentation records
    - ``sessions``     – list of session records
    - ``people``       – list of author/person records

    Usage
    -----
    Download the JSON for CHI 2024 from the PWA, then run::

        abstracts-explorer download --plugin chi --year 2024 \\
            --input-file chi_2024_program.json

    Supported content types
    -----------------------
    All content items that carry a non-empty ``abstract`` field are included
    (Papers, Late-Breaking Work, Case Studies, Journals, Workshops, …).
    """

    plugin_name = "chi"
    plugin_description = "CHI (ACM CHI) conference data loaded from the SIGCHI program JSON"
    _start_year = 2018
    conference_name = "CHI"

    def get_url(self, year: int) -> str:
        """
        Get the SIGCHI program page URL for a specific year.

        Parameters
        ----------
        year : int
            Conference year

        Returns
        -------
        str
            URL to the SIGCHI program page for the given year.
        """
        return f"https://programs.sigchi.org/chi/{year}"

    #: Mapping from the raw ``award`` field values used in the SIGCHI JSON
    #: to human-readable strings stored in :class:`LightweightPaper`.
    AWARD_NAMES: Dict[str, str] = {
        "BEST_PAPER": "Best Paper Award",
        "HONORABLE_MENTION": "Honorable Mention",
        "BEST_DEMO": "Best Demo Award",
        "BEST_POSTER": "Best Poster Award",
    }

    def download(
        self,
        year: Optional[int] = None,
        output_path: Optional[str] = None,
        force_download: bool = False,
        input_path: Optional[str] = None,
        **kwargs: Any,
    ) -> List[LightweightPaper]:
        """
        Load and convert CHI conference data from a SIGCHI program JSON file.

        When ``input_path`` is not provided, the plugin looks for a file at
        ``data/CHI_{year}_program.json`` before requesting manual interaction.

        Parameters
        ----------
        year : int, optional
            Conference year (2023, 2024, or 2025).  Used for validation;
            if omitted the year embedded in the JSON file is used.
        output_path : str, optional
            Path where the converted lightweight JSON should be saved.
            If the file already exists and ``force_download`` is ``False``,
            it is loaded directly (skipping re-parsing).
        force_download : bool
            Re-parse the CHI JSON even when a cached ``output_path`` exists.
        input_path : str, optional
            Path to the CHI program JSON downloaded from the SIGCHI PWA.
            If omitted, ``data/CHI_{year}_program.json`` is tried automatically.
        **kwargs : Any
            Ignored; accepted for interface compatibility.

        Returns
        -------
        list of LightweightPaper
            Validated paper objects ready for database insertion.

        Raises
        ------
        ValueError
            If the year is not supported, or the JSON year does not match
            the requested year, or no input file is found.
        FileNotFoundError
            If ``input_path`` does not point to an existing file.
        RuntimeError
            If no papers with abstracts are found in the JSON file.
        """
        if year is not None:
            self.validate_year(year)

        # Return cached lightweight data when it already exists
        if output_path and not force_download and Path(output_path).exists():
            logger.info("Loading CHI papers from cached file: %s", output_path)
            try:
                return self._load_lightweight_papers(output_path)
            except Exception as exc:  # pragma: no cover – edge case
                logger.warning("Failed to load %s (%s); re-parsing CHI JSON.", output_path, exc)

        # Require the CHI program JSON for a fresh parse
        if not input_path:
            # Auto-detect well-known file location
            example_year = year if year is not None else max(self.supported_years)
            default_path = self._get_default_input_path(example_year)
            if default_path.exists():
                logger.info("Auto-detected CHI program JSON: %s", default_path)
                input_path = str(default_path)
            else:
                raise ValueError(
                    "CHI conference data must be downloaded manually from the SIGCHI program PWA.\n"
                    f"  1. Go to https://programs.sigchi.org/chi/{example_year}\n"
                    "  2. Scroll to the very bottom of the page and click 'Get conference data JSON'\n"
                    f"  3. Save the file as {default_path}\n"
                    "     or pass it with --input-file <path>"
                )

        if not Path(input_path).exists():
            raise FileNotFoundError(f"CHI program JSON not found: {input_path}")

        papers = self._parse_chi_json(input_path, year)

        if output_path:
            self._save_lightweight_papers(papers, output_path)

        return papers

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_default_input_path(self, year: int) -> Path:
        """
        Get the default expected path for the CHI program JSON file.

        Parameters
        ----------
        year : int
            Conference year

        Returns
        -------
        Path
            Expected path to the CHI program JSON file for the given year.
        """
        config = get_config()
        return Path(config.data_dir) / f"CHI_{year}_program.json"

    def _load_lightweight_papers(self, path: str) -> List[LightweightPaper]:
        """
        Load previously-converted lightweight papers from a JSON file.

        Parameters
        ----------
        path : str
            Path to the lightweight JSON file.

        Returns
        -------
        list of LightweightPaper
        """
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError(f"Expected a list of papers in {path}, got {type(data).__name__}")
        return validate_lightweight_papers(data)

    def _save_lightweight_papers(self, papers: List[LightweightPaper], path: str) -> None:
        """
        Save lightweight papers to a JSON file.

        Parameters
        ----------
        papers : list of LightweightPaper
        path : str
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump([p.model_dump() for p in papers], fh, indent=2, ensure_ascii=False)
        logger.info("Saved %d CHI papers to %s", len(papers), path)

    def _parse_chi_json(self, input_path: str, year: Optional[int]) -> List[LightweightPaper]:
        """
        Parse the SIGCHI program JSON file and return :class:`LightweightPaper` objects.

        Parameters
        ----------
        input_path : str
            Path to the downloaded CHI program JSON file.
        year : int or None
            Expected conference year (used for validation).

        Returns
        -------
        list of LightweightPaper

        Raises
        ------
        ValueError
            If the JSON year does not match ``year``, or no papers are found.
        RuntimeError
            If parsing produces zero papers.
        """
        with open(input_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        conference = data.get("conference", {})
        actual_year: Optional[int] = conference.get("year")

        if year is not None and actual_year is not None and actual_year != year:
            raise ValueError(
                f"CHI JSON file contains year {actual_year}, but year {year} was requested. "
                "Please provide the correct JSON file."
            )

        paper_year = actual_year if actual_year is not None else year
        if paper_year is None:
            raise ValueError("Could not determine conference year from CHI JSON file.")

        # Build look-up tables
        people_by_id: Dict[int, Dict[str, Any]] = {p["id"]: p for p in data.get("people", [])}
        sessions_by_id: Dict[int, Dict[str, Any]] = {s["id"]: s for s in data.get("sessions", [])}
        content_types_by_id: Dict[int, str] = {ct["id"]: ct["name"] for ct in data.get("contentTypes", [])}

        papers: List[LightweightPaper] = []
        for item in data.get("contents", []):
            if not item.get("abstract", "").strip():
                continue
            paper = self._convert_to_lightweight(item, paper_year, people_by_id, sessions_by_id, content_types_by_id)
            if paper is not None:
                papers.append(paper)

        logger.info("Parsed %d papers from CHI %d JSON", len(papers), paper_year)

        if not papers:
            raise RuntimeError(f"No papers with abstracts found in CHI JSON file: {input_path}")

        return validate_lightweight_papers([p.model_dump() for p in papers])

    def _convert_to_lightweight(
        self,
        item: Dict[str, Any],
        year: int,
        people_by_id: Dict[int, Dict[str, Any]],
        sessions_by_id: Dict[int, Dict[str, Any]],
        content_types_by_id: Dict[int, str],
    ) -> Optional[LightweightPaper]:
        """
        Convert a single CHI content record to a :class:`LightweightPaper`.

        Parameters
        ----------
        item : dict
            Raw content record from the CHI JSON ``contents`` array.
        year : int
            Conference year.
        people_by_id : dict
            Mapping of person ID → person record (for resolving author names).
        sessions_by_id : dict
            Mapping of session ID → session record (for resolving session names).
        content_types_by_id : dict
            Mapping of type ID → type name.

        Returns
        -------
        LightweightPaper or None
            ``None`` if the record is missing a title, abstract, or authors.
        """
        title = (item.get("title") or "").strip()
        abstract = (item.get("abstract") or "").strip()
        if not title or not abstract:
            return None

        # Resolve author names via the ``people`` look-up table
        authors: List[str] = []
        for author_info in item.get("authors", []):
            person_id = author_info.get("personId")
            if person_id is None:
                continue
            person = people_by_id.get(person_id, {})
            first = (person.get("firstName") or "").strip()
            last = (person.get("lastName") or "").strip()
            full_name = f"{first} {last}".strip()
            if full_name:
                authors.append(full_name)

        if not authors:
            return None

        # Resolve session name (fall back to content-type name)
        session_ids = item.get("sessionIds") or []
        session_name = ""
        if session_ids:
            session = sessions_by_id.get(session_ids[0], {})
            session_name = (session.get("name") or "").strip()
        if not session_name:
            type_id: Optional[int] = item.get("typeId")
            session_name = content_types_by_id.get(type_id, "CHI") if type_id is not None else "CHI"

        # Build URL from DOI addon
        url: Optional[str] = None
        doi_addon = (item.get("addons") or {}).get("doi", {})
        doi_value = (doi_addon.get("url") or "").strip()
        if doi_value:
            if doi_value.startswith("https://"):
                url = doi_value
            elif doi_value.startswith("http://"):
                url = doi_value.replace("http://", "https://", 1)
            else:
                # Strip any leading doi.org/ prefix before normalising
                doi_path = doi_value
                for prefix in ("doi.org/",):
                    if doi_path.startswith(prefix):
                        doi_path = doi_path[len(prefix) :]
                        break
                url = f"https://doi.org/{doi_path}"

        # Human-readable award string
        award: Optional[str] = None
        raw_award = item.get("award")
        if raw_award:
            award = self.AWARD_NAMES.get(raw_award, raw_award)

        # Content-type name as a keyword tag
        content_type_id: Optional[int] = item.get("typeId")
        content_type_name = content_types_by_id.get(content_type_id, "") if content_type_id is not None else ""
        keywords: Optional[List[str]] = [content_type_name] if content_type_name else None

        try:
            return LightweightPaper(
                title=title,
                abstract=abstract,
                authors=authors,
                session=session_name,
                poster_position=str(item.get("id", "")),
                year=year,
                conference="CHI",
                original_id=item.get("id"),
                url=url,
                paper_pdf_url=url,
                award=award,
                keywords=keywords,
            )
        except ValidationError as exc:
            logger.warning("Skipping paper '%s': validation failed: %s", title, exc)
            return None

    def get_metadata(self) -> Dict[str, Any]:
        """
        Return metadata describing this plugin.

        Returns
        -------
        dict
            Plugin name, description, supported years, and accepted parameters.
        """
        return {
            "name": self.plugin_name,
            "description": self.plugin_description,
            "conference_name": self.conference_name,
            "supported_years": self.supported_years,
            "parameters": {
                "year": "int – Conference year (2023, 2024, or 2025)",
                "input_path": "str – Path to CHI program JSON downloaded from programs.sigchi.org",
                "output_path": "str – Path to save converted lightweight JSON",
                "force_download": "bool – Force re-parsing even when output_path exists",
            },
        }


# ---------------------------------------------------------------------------
# Auto-registration
# ---------------------------------------------------------------------------


def _register() -> None:
    """Auto-register the CHI plugin when this module is imported."""
    from abstracts_explorer.plugins import register_plugin

    plugin = CHIDownloaderPlugin()
    register_plugin(plugin)
    logger.debug("CHIDownloaderPlugin registered")


_register()

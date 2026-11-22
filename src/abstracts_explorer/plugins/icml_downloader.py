"""
ICML Official Downloader Plugin
===============================

Plugin for downloading papers from the official ICML conference data.

Years 2020 and later use the EventHosts virtual-site JSON API. Earlier years
(2013 onward) are loaded from the PMLR proceedings ``citeproc.yaml`` files.
"""

import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

import yaml
from pydantic import ValidationError

from abstracts_explorer.plugin import DownloadOptions, LightweightPaper, sanitize_author_names
from abstracts_explorer.plugins.json_conference_downloader import JSONConferenceDownloaderPlugin

logger = logging.getLogger(__name__)

_PMLR_VOLUMES = {
    2013: 28,
    2014: 32,
    2015: 37,
    2016: 48,
    2017: 70,
    2018: 80,
    2019: 97,
}


class ICMLDownloaderPlugin(JSONConferenceDownloaderPlugin):
    """
    Plugin for downloading papers from official ICML conference.

    This plugin downloads data from the ICML virtual conference site using
    their JSON API endpoint for years 2020 onward, and from PMLR proceedings
    for earlier years.
    """

    plugin_name = "icml"
    plugin_description = "Official ICML conference data downloader"
    _start_year = 2013
    _eventhosts_start_year = 2020
    conference_name = "ICML"
    _DEFAULT_REQUEST_DELAY = 0.5
    _USER_AGENT = "abstracts-explorer ICML downloader"
    _SOURCE_LABEL = "ICML proceedings"

    def get_url(self, year: int) -> str:
        """
        Get the download URL for ICML data.

        Parameters
        ----------
        year : int
            Conference year

        Returns
        -------
        str
            EventHosts JSON URL for 2020+, otherwise the PMLR volume URL.
        """
        if year >= self._eventhosts_start_year:
            return f"https://icml.cc/static/virtual/data/icml-{year}-orals-posters.json"
        volume = _PMLR_VOLUMES.get(year)
        if volume is None:
            raise RuntimeError(f"No PMLR proceedings volume configured for ICML {year}")
        return f"https://proceedings.mlr.press/v{volume}/"

    def _download_uncached(
        self,
        year: int,
        output_path: Optional[str],
        options: DownloadOptions,
    ) -> List[LightweightPaper]:
        """
        Download ICML papers for *year* from the appropriate source.

        Parameters
        ----------
        year : int
            Conference year.
        output_path : str, optional
            Cache path (unused by the legacy PMLR loader).
        options : DownloadOptions
            Resolved download options.

        Returns
        -------
        list of LightweightPaper
            Downloaded and validated papers.
        """
        if year >= self._eventhosts_start_year:
            return super()._download_uncached(year, output_path, options)
        return self._download_pmlr(year, options)

    def _download_pmlr(self, year: int, options: DownloadOptions) -> List[LightweightPaper]:
        """Load a year's papers from the PMLR ``citeproc.yaml`` bibliography."""
        volume = _PMLR_VOLUMES.get(year)
        if volume is None:
            raise RuntimeError(f"No PMLR proceedings volume configured for ICML {year}")
        url = f"https://proceedings.mlr.press/v{volume}/assets/bib/citeproc.yaml"
        logger.info("Downloading ICML %d proceedings from %s", year, url)

        response = self._request(
            url,
            options.timeout,
            options.verify_ssl,
            request_delay=options.request_delay,
        )
        response.encoding = "utf-8"
        try:
            records = yaml.safe_load(response.text)
        except yaml.YAMLError as exc:
            raise RuntimeError(f"Invalid PMLR citation YAML from {url}: {exc}") from exc

        if not isinstance(records, list) or not records:
            raise RuntimeError(f"No PMLR citation records found for ICML {year}: {url}")

        papers: List[LightweightPaper] = []
        skipped = 0
        for record in records:
            if not isinstance(record, dict):
                skipped += 1
                continue
            paper = self._parse_pmlr_record(record, year)
            if paper is None:
                skipped += 1
                continue
            papers.append(paper)

        if not papers:
            raise RuntimeError(f"No valid ICML papers found in PMLR volume {volume} ({year})")
        if skipped:
            logger.warning("Skipped %d incomplete PMLR record(s) for ICML %d", skipped, year)
        logger.info("Successfully downloaded %d ICML %d papers from PMLR", len(papers), year)
        return papers

    def _parse_pmlr_record(self, record: Dict[str, Any], year: int) -> Optional[LightweightPaper]:
        """Map one PMLR citation record to a validated lightweight paper."""
        title = str(record.get("title") or "").strip()
        abstract = str(record.get("abstract") or "").strip()
        authors = self._parse_pmlr_authors(record.get("authors", record.get("author")))
        url = str(record.get("url") or record.get("URL") or "").strip()
        pdf_url = str(record.get("pdf") or record.get("PDF") or "").strip() or None
        original_id = str(record.get("id") or "").strip()
        if not original_id and url:
            original_id = urlsplit(url).path.rstrip("/").rsplit("/", 1)[-1].removesuffix(".html")

        try:
            return LightweightPaper(
                title=title,
                authors=sanitize_author_names(authors),
                abstract=abstract,
                session=f"ICML {year}",
                poster_position="",
                year=year,
                conference=self.conference_name,
                original_id=original_id or None,
                url=url or None,
                paper_pdf_url=pdf_url,
            )
        except ValidationError as exc:
            logger.debug("Skipping PMLR record '%s': %s", title or "<untitled>", exc)
            return None

    @staticmethod
    def _parse_pmlr_authors(value: Any) -> List[str]:
        """Convert PMLR author structures into a list of author names."""
        if isinstance(value, list):
            names: List[str] = []
            for item in value:
                if isinstance(item, dict):
                    given = str(item.get("given") or "").strip()
                    family = str(item.get("family") or "").strip()
                    name = f"{given} {family}".strip()
                    if name:
                        names.append(name)
                else:
                    text = str(item).strip()
                    if text:
                        names.append(text)
            return names
        if not isinstance(value, str):
            return []

        authors: List[str] = []
        for name in (part.strip() for part in value.split(" and ")):
            if not name:
                continue
            if "," in name:
                last, first = name.split(",", 1)
                name = f"{first.strip()} {last.strip()}".strip()
            if name:
                authors.append(name)
        return authors


# Auto-register the plugin when imported
def _register():
    """Auto-register the ICML plugin."""
    from abstracts_explorer.plugins import register_plugin

    plugin = ICMLDownloaderPlugin()
    register_plugin(plugin)
    logger.debug("ICML downloader plugin registered")


# Register on import
_register()

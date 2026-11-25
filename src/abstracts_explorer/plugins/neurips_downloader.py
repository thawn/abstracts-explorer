"""
NeurIPS Official Downloader Plugin
===================================

Plugin for downloading papers from the official NeurIPS conference data.

Years 2020 and later use the EventHosts virtual-site JSON API. Older years are
scraped from the official NeurIPS proceedings at ``papers.nips.cc``.
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from pydantic import ValidationError

from abstracts_explorer.plugin import DownloadOptions, LightweightPaper, sanitize_author_names
from abstracts_explorer.plugin.locations import assign_locations, build_neurips_location_map
from abstracts_explorer.plugin.persistence import FailureSidecar
from abstracts_explorer.plugins.json_conference_downloader import JSONConferenceDownloaderPlugin

logger = logging.getLogger(__name__)

_HASH_RE = re.compile(r"/hash/([0-9a-f]+)-Abstract")


class NeurIPSDownloaderPlugin(JSONConferenceDownloaderPlugin):
    """
    Plugin for downloading papers from official NeurIPS conference.

    This plugin downloads data from the NeurIPS virtual conference site
    using their JSON API endpoint for years 2020 onward, and from the NeurIPS
    proceedings site for earlier years.
    """

    plugin_name = "neurips"
    plugin_description = "Official NeurIPS conference data downloader"
    _start_year = 1987
    _eventhosts_start_year = 2020
    conference_name = "NeurIPS"
    _legacy_index_url = "https://papers.nips.cc/paper_files/paper/{year}"
    _DEFAULT_MAX_WORKERS = 8
    _USER_AGENT = "abstracts-explorer NeurIPS downloader"
    _SOURCE_LABEL = "NeurIPS proceedings"

    def get_url(self, year: int) -> str:
        """
        Get the download URL for NeurIPS data.

        Parameters
        ----------
        year : int
            Conference year

        Returns
        -------
        str
            EventHosts JSON URL for 2020+, otherwise the proceedings index URL.
        """
        if year >= self._eventhosts_start_year:
            return f"https://neurips.cc/static/virtual/data/neurips-{year}-orals-posters.json"
        return self._legacy_index_url.format(year=year)

    def _download_uncached(
        self,
        year: int,
        output_path: Optional[str],
        options: DownloadOptions,
    ) -> List[LightweightPaper]:
        """
        Download NeurIPS papers for *year* from the appropriate source.

        Parameters
        ----------
        year : int
            Conference year.
        output_path : str, optional
            Cache path used by legacy downloads to record failures.
        options : DownloadOptions
            Resolved download options.

        Returns
        -------
        list of LightweightPaper
            Downloaded and validated papers.
        """
        if year >= self._eventhosts_start_year:
            papers = super()._download_uncached(year, output_path, options)
            return self._assign_locations(papers, year, options)
        return self._download_legacy(year, output_path, options)

    def _assign_locations(
        self,
        papers: List[LightweightPaper],
        year: int,
        options: DownloadOptions,
    ) -> List[LightweightPaper]:
        """Derive and assign physical locations from the EventHosts site."""
        try:
            location_map = build_neurips_location_map(
                year,
                lambda url: self._request(
                    url,
                    options.timeout,
                    options.verify_ssl,
                    request_delay=options.request_delay,
                ).text,
            )
            if location_map.is_multi_location:
                assign_locations(papers, location_map)
                logger.info("Assigned locations for NeurIPS %d: %s", year, ", ".join(location_map.locations))
        except Exception as exc:
            logger.warning("Could not derive NeurIPS %d locations: %s", year, exc)
        return papers

    def _handle_cached(
        self,
        cached: List[LightweightPaper],
        year: int,
        output_path: str,
        options: DownloadOptions,
    ) -> List[LightweightPaper]:
        """Retry failed legacy pages when a cache with a failures sidecar is reused."""
        if year >= self._eventhosts_start_year:
            cached = super()._handle_cached(cached, year, output_path, options)
            if any(not paper.location for paper in cached):
                cached = self._assign_locations(cached, year, options)
            return cached
        failures = FailureSidecar(output_path, self.conference_name, year).load()
        if not failures:
            return cached
        logger.info("Retrying %d previously failed NeurIPS %d page(s).", len(failures), year)
        return self._retry_legacy_failures(cached, failures, year, output_path, options)

    def _download_legacy(
        self,
        year: int,
        output_path: Optional[str],
        options: DownloadOptions,
    ) -> List[LightweightPaper]:
        """Scrape the NeurIPS proceedings index and abstract pages for *year*."""
        index_url = self._legacy_index_url.format(year=year)
        logger.info("Downloading NeurIPS %d proceedings from %s", year, index_url)
        soup = self._fetch_soup(index_url, options.timeout, options.verify_ssl, options.request_delay)
        entries = self._parse_index(soup, index_url)
        if not entries:
            raise RuntimeError(f"No NeurIPS papers found in proceedings index for {year}: {index_url}")

        papers: List[LightweightPaper] = []
        failures: List[Dict[str, Any]] = []
        seen_urls: set[str] = set()
        skipped_incomplete = 0
        max_workers = max(1, options.max_workers)
        logger.info(
            "Fetching abstracts for %d NeurIPS %d papers using %d workers...", len(entries), year, max_workers
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_entry = {
                executor.submit(self._fetch_legacy_paper, entry, year, options): entry for entry in entries
            }
            for future in as_completed(future_to_entry):
                entry = future_to_entry[future]
                try:
                    paper = future.result()
                except RuntimeError as exc:
                    logger.warning("Skipping NeurIPS paper %s: %s", entry["url"], exc)
                    failures.append(
                        {
                            "url": entry["url"],
                            "type": "paper",
                            "title": entry.get("title", ""),
                            "authors": entry.get("authors", []),
                        }
                    )
                    continue
                if paper is None:
                    logger.debug(
                        "Skipping NeurIPS paper with missing required metadata: %s",
                        entry["url"],
                    )
                    skipped_incomplete += 1
                    continue
                if paper.url in seen_urls:
                    continue
                seen_urls.add(paper.url or "")
                papers.append(paper)

        FailureSidecar(output_path, self.conference_name, year).save(failures)
        if skipped_incomplete:
            logger.info(
                "Skipped %d NeurIPS %d paper(s) with missing required metadata",
                skipped_incomplete,
                year,
            )
        if not papers and entries:
            raise RuntimeError(f"No valid NeurIPS papers with abstracts found for {year}")
        logger.info("Successfully downloaded %d NeurIPS %d papers from proceedings", len(papers), year)
        return papers

    def _parse_index(self, soup: BeautifulSoup, index_url: str) -> List[Dict[str, Any]]:
        """Parse title, authors and abstract URL from a proceedings index page."""
        entries: List[Dict[str, Any]] = []
        for item in soup.select("ul.paper-list li"):
            link = item.select_one(".paper-content a[href]")
            if link is None:
                continue
            title = link.get_text(" ", strip=True)
            href = link.get("href")
            if not title or not isinstance(href, str):
                continue
            authors_tag = item.select_one(".paper-authors")
            authors_text = authors_tag.get_text(" ", strip=True) if authors_tag else ""
            authors = [name.strip() for name in authors_text.split(",") if name.strip()]
            entries.append(
                {
                    "title": title,
                    "authors": authors,
                    "url": urljoin(index_url, href),
                }
            )
        return entries

    def _fetch_legacy_paper(
        self,
        entry: Dict[str, Any],
        year: int,
        options: DownloadOptions,
    ) -> Optional[LightweightPaper]:
        """Fetch and parse a single legacy NeurIPS abstract page."""
        soup = self._fetch_soup(entry["url"], options.timeout, options.verify_ssl, options.request_delay)
        return self._parse_legacy_paper(soup, entry, year)

    def _parse_legacy_paper(
        self,
        soup: BeautifulSoup,
        entry: Dict[str, Any],
        year: int,
    ) -> Optional[LightweightPaper]:
        """Map a NeurIPS abstract page to a validated lightweight paper."""
        abstract_tag = soup.select_one(".paper-abstract")
        abstract = abstract_tag.get_text(" ", strip=True) if abstract_tag else ""

        authors = entry.get("authors") or self._authors_from_metadata(soup)
        title = entry.get("title") or self._meta_content(soup, "citation_title")
        pdf_url = self._meta_content(soup, "citation_pdf_url")
        match = _HASH_RE.search(entry["url"])

        try:
            return LightweightPaper(
                title=title,
                authors=sanitize_author_names(authors),
                abstract=abstract,
                session=f"NeurIPS {year}",
                poster_position="",
                year=year,
                conference=self.conference_name,
                original_id=match.group(1) if match else None,
                url=entry["url"],
                paper_pdf_url=pdf_url or None,
            )
        except ValidationError as exc:
            logger.debug("Skipping NeurIPS paper %s: %s", entry["url"], exc)
            return None

    @staticmethod
    def _meta_content(soup: BeautifulSoup, name: str) -> str:
        """Return the content of the first matching citation meta tag."""
        tag = soup.select_one(f'meta[name="{name}"][content]')
        return str(tag.get("content", "")).strip() if tag is not None else ""

    def _authors_from_metadata(self, soup: BeautifulSoup) -> List[str]:
        """Build author names from ``citation_author`` metadata (``Last, First``)."""
        authors: List[str] = []
        for tag in soup.select('meta[name="citation_author"][content]'):
            value = str(tag.get("content", "")).strip()
            if not value:
                continue
            if "," in value:
                last, first = value.split(",", 1)
                value = f"{first.strip()} {last.strip()}".strip()
            authors.append(value)
        return authors

    def _retry_legacy_failures(
        self,
        cached: List[LightweightPaper],
        failures: List[Dict[str, Any]],
        year: int,
        output_path: str,
        options: DownloadOptions,
    ) -> List[LightweightPaper]:
        """Retry failed legacy pages and merge recovered papers into the cache."""
        papers = list(cached)
        seen_urls = {paper.url for paper in papers if paper.url}
        remaining: List[Dict[str, Any]] = []
        for entry in failures:
            try:
                paper = self._fetch_legacy_paper(entry, year, options)
            except RuntimeError as exc:
                logger.warning("Retry failed for NeurIPS paper %s: %s", entry.get("url"), exc)
                remaining.append(entry)
                continue
            if paper is None:
                logger.debug("Skipping incomplete NeurIPS paper %s", entry.get("url"))
                continue
            if paper.url and paper.url in seen_urls:
                continue
            papers.append(paper)
            if paper.url:
                seen_urls.add(paper.url)

        self._save_papers_json(papers, output_path, self._save_label)
        FailureSidecar(output_path, self.conference_name, year).save(remaining)
        return papers


# Auto-register the plugin when imported
def _register():
    """Auto-register the NeurIPS plugin."""
    from abstracts_explorer.plugins import register_plugin

    plugin = NeurIPSDownloaderPlugin()
    register_plugin(plugin)
    logger.debug("NeurIPS downloader plugin registered")


# Register on import
_register()

"""Download AAAI abstracts and metadata from the official OJS proceedings."""

import logging
import re
import time
from datetime import datetime
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, Tag
from pydantic import ValidationError

from abstracts_explorer.plugin import (
    LightweightDownloaderPlugin,
    LightweightPaper,
    register_plugin,
    sanitize_author_names,
)

logger = logging.getLogger(__name__)


class AAAIDownloaderPlugin(LightweightDownloaderPlugin):
    """Download AAAI papers, excluding co-located IAAI and EAAI sections.

    Parameters
    ----------
    timeout : int, optional
        HTTP timeout in seconds (default: 30).
    verify_ssl : bool, optional
        Verify TLS certificates (default: True).

    Notes
    -----
    The official OJS archive covers 2010 onward. A year may span many issues.
    Conference years come from the issue series, not article publication dates.
    Missing abstracts and invalid records are logged and skipped. HTTP failures
    abort the download so an incomplete result is not saved as a complete cache.
    """

    plugin_name = "aaai"
    plugin_description = "Official AAAI proceedings downloader"
    conference_name = "AAAI"
    _start_year = 2010
    _ARCHIVE_URL = "https://ojs.aaai.org/index.php/AAAI/issue/archive"
    _EXCLUDED_SECTIONS = re.compile(
        r"front\s*matter|\b[IE]AAI\d*\b|"
        r"innovative applications of artificial intelligence|"
        r"education(?:al)? advances in artificial intelligence",
        re.IGNORECASE,
    )

    def __init__(self, timeout: int = 30, verify_ssl: bool = True) -> None:
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self._session = requests.Session()
        self._session.headers.update(
            {"User-Agent": "abstracts-explorer AAAI proceedings downloader"}
        )

    def get_url(self, year: int) -> str:
        """Return the archive URL shared by all conference years.

        Parameters
        ----------
        year : int
            Conference year; issues are selected after fetching the archive.

        Returns
        -------
        str
            Official proceedings archive URL.
        """
        return self._ARCHIVE_URL

    @staticmethod
    def _issue_year(summary: Tag) -> int | None:
        """Read the series year, or convert the annual volume to a year."""
        series = summary.select_one(".series")
        text = series.get_text(" ", strip=True) if series else ""
        match = re.search(r"\((\d{4})\)", text)
        if match:
            return int(match.group(1))
        # Modern OJS entries omit the year: volume 24 = 2010, 40 = 2026.
        match = re.search(r"Vol\.\s*(\d+)", text)
        return int(match.group(1)) + 1986 if match else None

    def _check_current_year_available(self, current_year: int) -> bool:
        """Check for this year's issues, not just a reachable archive URL."""
        try:
            soup = self._fetch_page(self._ARCHIVE_URL, 3, self.verify_ssl, 0)
        except RuntimeError:
            return False
        for summary in soup.select(".obj_issue_summary"):
            link = summary.select_one("a.title")
            if (
                link
                and re.search(r"\bAAAI\b", link.get_text(" "), re.IGNORECASE)
                and self._issue_year(summary) == current_year
            ):
                return True
        return False

    def _fetch_page(
        self, url: str, timeout: int, verify_ssl: bool, request_delay: float
    ) -> BeautifulSoup:
        """Fetch HTML with a timeout, request delay, and contextual errors."""
        if request_delay > 0:
            time.sleep(request_delay)
        try:
            response = self._session.get(url, timeout=timeout, verify=verify_ssl)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to fetch AAAI proceedings from {url}") from exc
        return BeautifulSoup(response.text, "html.parser")

    def _discover_issues(
        self, year: int, timeout: int, verify_ssl: bool, request_delay: float
    ) -> list[str]:
        """Follow archive pagination and collect all AAAI issues for a year."""
        issues: dict[str, None] = {}
        visited: set[str] = set()
        archive_count = 0
        archive_total: int | None = None
        url = self.get_url(year)
        while url:
            if url in visited:
                raise RuntimeError(f"Repeated AAAI archive page: {url}")
            visited.add(url)
            soup = self._fetch_page(url, timeout, verify_ssl, request_delay)
            summaries = soup.select(".obj_issue_summary")
            if not summaries:
                raise RuntimeError(f"No issue summaries on AAAI archive page: {url}")
            for summary in summaries:
                link = summary.select_one("a.title[href]")
                if not link:
                    raise RuntimeError(f"Unrecognized AAAI archive issue link: {url}")
                if not re.search(r"\bAAAI\b", link.get_text(" "), re.IGNORECASE):
                    continue
                issue_year = self._issue_year(summary)
                if issue_year is None:
                    raise RuntimeError("Could not determine year of an AAAI issue")
                if issue_year == year:
                    issues[urljoin(url, str(link["href"]))] = None
            counter = soup.select_one(".cmp_pagination .current")
            counts = re.fullmatch(
                r"(\d+)\s*[-–]\s*(\d+)\s+of\s+(\d+)",
                counter.get_text(" ", strip=True) if counter else "",
            )
            if not counts:
                raise RuntimeError(f"Unrecognized AAAI archive pagination: {url}")
            start, end, total = map(int, counts.groups())
            if (
                start != archive_count + 1
                or end != archive_count + len(summaries)
                or end > total
                or (archive_total is not None and total != archive_total)
            ):
                raise RuntimeError(f"Inconsistent AAAI archive pagination: {url}")
            archive_count, archive_total = end, total
            next_page = soup.select_one(".cmp_pagination a.next[href]")
            if end < total and not next_page:
                raise RuntimeError(f"Missing next AAAI archive page: {url}")
            url = urljoin(url, str(next_page["href"])) if next_page else ""
        if not issues:
            raise RuntimeError(f"No AAAI proceedings issues found for {year}")
        return list(issues)

    def _parse_article(
        self, soup: BeautifulSoup, url: str, session: str, year: int
    ) -> LightweightPaper | None:
        """Map OJS citation metadata and abstract text to a validated paper."""
        metadata: dict[str, list[str]] = {}
        for meta in soup.select("meta[name][content]"):
            value = str(meta["content"]).strip()
            if value:
                metadata.setdefault(str(meta["name"]), []).append(value)

        def first(name: str) -> str:
            return next(iter(metadata.get(name, [])), "")

        abstract = first("DC.Description")
        if not abstract:
            abstract_tag = soup.select_one(".item.abstract")
            if abstract_tag:
                for heading in abstract_tag.select(".label"):
                    heading.decompose()
                abstract = abstract_tag.get_text(" ", strip=True)
        # Some OJS descriptions include HTML even inside the meta attribute.
        abstract = BeautifulSoup(abstract, "html.parser").get_text(" ", strip=True)
        article_id = re.search(r"/article/view/(\d+)/?$", url)
        pdf_url = first("citation_pdf_url")
        keywords = metadata.get("DC.Subject", [])
        try:
            return LightweightPaper(
                title=first("citation_title") or first("DC.Title"),
                authors=sanitize_author_names(
                    metadata.get("citation_author")
                    or metadata.get("DC.Creator.PersonalName", [])
                ),
                abstract=abstract,
                session=session,
                poster_position="",
                year=year,
                conference=self.conference_name,
                original_id=int(article_id.group(1)) if article_id else None,
                url=url,
                paper_pdf_url=urljoin(url, pdf_url) if pdf_url else None,
                keywords=keywords or None,
            )
        except ValidationError as exc:
            logger.warning("Skipping AAAI article %s: %s", url, exc)
            return None

    def download(
        self,
        year: int | None = None,
        output_path: str | None = None,
        force_download: bool = False,
        **kwargs: Any,
    ) -> list[LightweightPaper]:
        """Download all AAAI proceedings issues for one conference year.

        Parameters
        ----------
        year : int, optional
            Conference year; defaults to the latest supported year.
        output_path : str, optional
            Lightweight JSON cache path. Existing files are loaded offline.
        force_download : bool, optional
            Ignore an existing cache and fetch fresh proceedings.
        **kwargs : Any
            Overrides for ``timeout``, ``verify_ssl``, and ``request_delay``
            (default: 0.5 seconds before each HTTP request).

        Returns
        -------
        list of LightweightPaper
            Unique, validated AAAI papers in proceedings order.

        Raises
        ------
        ValueError
            If the requested year is unsupported.
        RuntimeError
            If fetching or issue discovery fails, or no valid papers remain.
        """
        if year is None:
            year = max(self.supported_years)
        if year < self._start_year or year > datetime.now().year:
            raise ValueError(f"Year {year} not supported by {self.plugin_name}")
        if output_path and not force_download:
            cached = self._load_papers_json(output_path, "AAAI")
            if cached and all(
                paper.year == year and paper.conference == self.conference_name
                for paper in cached
            ):
                return cached
        # Explicit requests use actual issue discovery, not a cached metadata
        # probe, which can fail transiently or use different HTTP options.
        timeout = kwargs.get("timeout", self.timeout)
        verify_ssl = kwargs.get("verify_ssl", self.verify_ssl)
        request_delay = kwargs.get("request_delay", 0.5)
        issue_urls = self._discover_issues(year, timeout, verify_ssl, request_delay)
        papers: list[LightweightPaper] = []
        seen: set[int] = set()
        for issue_url in issue_urls:
            soup = self._fetch_page(issue_url, timeout, verify_ssl, request_delay)
            sections = soup.select(".section")
            if not sections:
                raise RuntimeError(
                    f"No proceedings sections in AAAI issue: {issue_url}"
                )
            for section in sections:
                heading = section.select_one("h2")
                session = (
                    heading.get_text(" ", strip=True) if heading else "AAAI proceedings"
                )
                if self._EXCLUDED_SECTIONS.search(session):
                    continue
                summaries = section.select(".obj_article_summary")
                entries = section.select(".articles > li")
                if entries and len(entries) != len(summaries):
                    raise RuntimeError(
                        f"Inconsistent AAAI article listing: {issue_url} ({session})"
                    )
                if not summaries:
                    raise RuntimeError(
                        f"No article summaries in AAAI section: {issue_url} ({session})"
                    )
                for summary in summaries:
                    link = summary.select_one(".title a[href]")
                    if not link:
                        raise RuntimeError(
                            f"Unrecognized AAAI article link: {issue_url}"
                        )
                    url = urljoin(issue_url, str(link["href"]))
                    match = re.search(r"/AAAI/article/view/(\d+)/?$", url)
                    if not match:
                        raise RuntimeError(f"Unrecognized AAAI article URL: {url}")
                    article_id = int(match.group(1))
                    if article_id in seen:
                        continue
                    seen.add(article_id)
                    article = self._fetch_page(url, timeout, verify_ssl, request_delay)
                    paper = self._parse_article(article, url, session, year)
                    if paper is not None:
                        papers.append(paper)
        if not papers:
            raise RuntimeError(f"No valid AAAI papers with abstracts found for {year}")
        logger.info("Downloaded %d unique AAAI %d papers", len(papers), year)
        self._save_papers_json(papers, output_path, "AAAI")
        return papers

    def get_metadata(self) -> dict[str, Any]:
        """Return conference metadata and accepted download parameters.

        Returns
        -------
        dict
            Plugin name, canonical conference name, years, and parameters.
        """
        return {
            "name": self.plugin_name,
            "description": self.plugin_description,
            "conference_name": self.conference_name,
            "supported_years": self.supported_years,
            "parameters": {
                "year": "int - Conference year (default: latest supported)",
                "output_path": "str - Lightweight JSON cache path",
                "force_download": "bool - Ignore existing cache",
                "timeout": "int - HTTP timeout in seconds",
                "verify_ssl": "bool - Verify TLS certificates",
                "request_delay": "float - Delay before requests (default: 0.5)",
            },
        }


register_plugin(AAAIDownloaderPlugin())

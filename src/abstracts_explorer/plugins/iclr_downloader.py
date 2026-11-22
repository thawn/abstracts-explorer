"""
ICLR Official Downloader Plugin
===============================

Plugin for downloading papers from the official ICLR conference data.

Years 2020 and later use the EventHosts virtual-site JSON API. Earlier years
(2017 onward) are loaded from OpenReview, with a Semantic Scholar fallback
when the OpenReview API is unavailable (for example, behind a bot challenge).
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from abstracts_explorer.plugin import DownloadOptions, LightweightPaper, sanitize_author_names
from abstracts_explorer.plugins.json_conference_downloader import JSONConferenceDownloaderPlugin

logger = logging.getLogger(__name__)

_OPENREVIEW_VENUE_IDS = {
    2017: "ICLR.cc/2017/conference",
    2018: "ICLR.cc/2018/Conference",
    2019: "ICLR.cc/2019/Conference",
}

_S2_FIELDS = "title,abstract,authors,externalIds,url"


class ICLRDownloaderPlugin(JSONConferenceDownloaderPlugin):
    """
    Plugin for downloading papers from official ICLR conference.

    This plugin downloads data from the ICLR virtual conference site using
    their JSON API endpoint for years 2020 onward, and from OpenReview (with a
    Semantic Scholar fallback) for earlier years.
    """

    plugin_name = "iclr"
    plugin_description = "Official ICLR conference data downloader"
    _start_year = 2017
    _eventhosts_start_year = 2020
    conference_name = "ICLR"
    _USER_AGENT = "abstracts-explorer ICLR downloader"
    _SOURCE_LABEL = "ICLR proceedings"
    _OPENREVIEW_API_URL = "https://api.openreview.net"

    def get_url(self, year: int) -> str:
        """
        Get the download URL for ICLR data.

        Parameters
        ----------
        year : int
            Conference year

        Returns
        -------
        str
            EventHosts JSON URL for 2020+, otherwise the OpenReview API URL.
        """
        if year >= self._eventhosts_start_year:
            return f"https://iclr.cc/static/virtual/data/iclr-{year}-orals-posters.json"
        venue_id = _OPENREVIEW_VENUE_IDS.get(year)
        if venue_id is None:
            raise RuntimeError(f"No OpenReview venue configured for ICLR {year}")
        return f"https://api.openreview.net/notes?content.venueid={venue_id}"

    def _download_uncached(
        self,
        year: int,
        output_path: Optional[str],
        options: DownloadOptions,
    ) -> List[LightweightPaper]:
        """
        Download ICLR papers for *year* from the appropriate source.

        Parameters
        ----------
        year : int
            Conference year.
        output_path : str, optional
            Cache path (unused by the legacy loader).
        options : DownloadOptions
            Resolved download options.

        Returns
        -------
        list of LightweightPaper
            Downloaded and validated papers.
        """
        if year >= self._eventhosts_start_year:
            return super()._download_uncached(year, output_path, options)
        return self._download_legacy(year, options)

    def _download_legacy(self, year: int, options: DownloadOptions) -> List[LightweightPaper]:
        """Load ICLR papers from OpenReview, falling back to Semantic Scholar."""
        try:
            papers = self._fetch_openreview(year, options)
        except RuntimeError as exc:
            logger.warning("OpenReview download failed for ICLR %d: %s", year, exc)
            papers = []

        if papers:
            logger.info("Successfully downloaded %d ICLR %d papers from OpenReview", len(papers), year)
            return papers

        logger.info("Falling back to Semantic Scholar for ICLR %d", year)
        papers = self._fetch_semantic_scholar(year, options)
        if not papers:
            raise RuntimeError(f"No valid ICLR papers with abstracts found for {year}")
        logger.info("Successfully downloaded %d ICLR %d papers from Semantic Scholar", len(papers), year)
        return papers

    def _fetch_openreview(self, year: int, options: DownloadOptions) -> List[LightweightPaper]:
        """Fetch accepted papers from the OpenReview API."""
        token = self._resolve_openreview_token(options)
        headers = {"Authorization": f"Bearer {token}"} if token else None
        venue_id = _OPENREVIEW_VENUE_IDS[year]
        url = f"{self._OPENREVIEW_API_URL}/notes"
        papers: List[LightweightPaper] = []
        offset = 0
        limit = 1000
        while True:
            params = {"content.venueid": venue_id, "limit": limit, "offset": offset}
            payload = self._fetch_json(
                url,
                options.timeout,
                options.verify_ssl,
                params=params,
                request_delay=options.request_delay,
                headers=headers,
            )
            notes = payload.get("notes") if isinstance(payload, dict) else None
            if not isinstance(notes, list) or not notes:
                break
            for note in notes:
                paper = self._parse_openreview_note(note, year)
                if paper is not None:
                    papers.append(paper)
            if len(notes) < limit:
                break
            offset += limit

        if not papers:
            raise RuntimeError(f"OpenReview returned no ICLR {year} records for venue {venue_id}")
        return papers

    def _resolve_openreview_token(self, options: DownloadOptions) -> Optional[str]:
        """
        Return an OpenReview bearer token from options, config, or a login.

        Parameters
        ----------
        options : DownloadOptions
            Resolved download options carrying optional OpenReview credentials.

        Returns
        -------
        str or None
            Bearer token, or ``None`` when no credentials are configured.

        Raises
        ------
        RuntimeError
            If a login was attempted and failed.
        """
        if options.openreview_token:
            return str(options.openreview_token).strip()

        from abstracts_explorer.config import get_config

        config = get_config()
        if config.openreview_token:
            return str(config.openreview_token).strip()
        email = str(options.openreview_email or config.openreview_email or "").strip()
        password = str(options.openreview_password or config.openreview_password or "")
        if not email or not password:
            return None

        login_url = f"{self._OPENREVIEW_API_URL}/login"
        logger.debug("Logging in to OpenReview as %s", email)
        try:
            response = self._request(
                login_url,
                options.timeout,
                options.verify_ssl,
                request_delay=options.request_delay,
                method="POST",
                json_body={"id": email, "password": password},
            )
        except RuntimeError as exc:
            raise RuntimeError(f"OpenReview login failed: {exc}") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(f"OpenReview login returned invalid JSON: {exc}") from exc
        token = payload.get("token") if isinstance(payload, dict) else None
        if not token:
            raise RuntimeError("OpenReview login failed: no token returned")
        return str(token)

    def _parse_openreview_note(self, note: Any, year: int) -> Optional[LightweightPaper]:
        """Map one OpenReview note to a validated lightweight paper."""
        if not isinstance(note, dict):
            return None
        content = note.get("content")
        if not isinstance(content, dict):
            return None

        title = self._content_value(content, "title")
        abstract = self._content_value(content, "abstract")
        authors_value = content.get("authors")
        authors = authors_value if isinstance(authors_value, list) else []
        authors = [str(author).strip() for author in authors if str(author).strip()]

        forum = str(note.get("forum") or note.get("id") or "").strip()
        url = f"https://openreview.net/forum?id={forum}" if forum else None
        pdf_url = f"https://openreview.net/pdf?id={forum}" if forum else None

        try:
            return LightweightPaper(
                title=title,
                authors=sanitize_author_names(authors),
                abstract=abstract,
                session=f"ICLR {year}",
                poster_position="",
                year=year,
                conference=self.conference_name,
                original_id=forum or None,
                url=url,
                paper_pdf_url=pdf_url,
            )
        except ValidationError as exc:
            logger.debug("Skipping OpenReview note %s: %s", forum or "<unknown>", exc)
            return None

    @staticmethod
    def _content_value(content: Dict[str, Any], key: str) -> str:
        """Return a scalar value from OpenReview content (v1 or v2 shape)."""
        value = content.get(key)
        if isinstance(value, dict):
            value = value.get("value")
        return str(value).strip() if value is not None else ""

    def _fetch_semantic_scholar(self, year: int, options: DownloadOptions) -> List[LightweightPaper]:
        """Fetch ICLR papers from the Semantic Scholar bulk search API."""
        url = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
        papers: List[LightweightPaper] = []
        token: Optional[str] = None
        seen_tokens: set[str] = set()
        while True:
            params: Dict[str, Any] = {
                "query": "*",
                "venue": "ICLR",
                "year": year,
                "fields": _S2_FIELDS,
                "limit": 1000,
            }
            if token:
                params["token"] = token
            try:
                payload = self._fetch_json(
                    url,
                    options.timeout,
                    options.verify_ssl,
                    params=params,
                    request_delay=options.request_delay,
                )
            except RuntimeError as exc:
                if not papers:
                    raise
                logger.warning("Semantic Scholar pagination failed for ICLR %d: %s", year, exc)
                break

            if not isinstance(payload, dict):
                break
            records = payload.get("data")
            if not isinstance(records, list) or not records:
                break
            for record in records:
                paper = self._parse_s2_record(record, year)
                if paper is not None:
                    papers.append(paper)

            token = payload.get("token")
            if not token or token in seen_tokens:
                break
            seen_tokens.add(token)

        return papers

    def _parse_s2_record(self, record: Any, year: int) -> Optional[LightweightPaper]:
        """Map one Semantic Scholar record to a validated lightweight paper."""
        if not isinstance(record, dict):
            return None
        authors_data = record.get("authors")
        authors = []
        if isinstance(authors_data, list):
            authors = [
                str(author.get("name", "")).strip()
                for author in authors_data
                if isinstance(author, dict) and str(author.get("name", "")).strip()
            ]

        paper_id = str(record.get("paperId") or "").strip()
        url = str(record.get("url") or "").strip()
        if not url and paper_id:
            url = f"https://www.semanticscholar.org/paper/{paper_id}"

        try:
            return LightweightPaper(
                title=str(record.get("title") or "").strip(),
                authors=sanitize_author_names(authors),
                abstract=str(record.get("abstract") or "").strip(),
                session=f"ICLR {year}",
                poster_position="",
                year=year,
                conference=self.conference_name,
                original_id=paper_id or None,
                url=url or None,
            )
        except ValidationError as exc:
            logger.debug("Skipping Semantic Scholar record %s: %s", paper_id or "<unknown>", exc)
            return None


# Auto-register the plugin when imported
def _register():
    """Auto-register the ICLR plugin."""
    from abstracts_explorer.plugins import register_plugin

    plugin = ICLRDownloaderPlugin()
    register_plugin(plugin)
    logger.debug("ICLR downloader plugin registered")


# Register on import
_register()

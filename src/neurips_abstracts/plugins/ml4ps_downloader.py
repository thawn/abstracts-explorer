"""
ML4PS Workshop Downloader Plugin
=================================

Plugin for downloading papers from the Machine Learning for Physical Sciences workshop.
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
import logging
import requests
from bs4 import BeautifulSoup
import json
import re
import time
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from datetime import datetime

from neurips_abstracts.plugins import DownloaderPlugin

logger = logging.getLogger(__name__)


class ML4PSDownloaderPlugin(DownloaderPlugin):
    """
    Plugin for downloading papers from ML4PS (Machine Learning for Physical Sciences) workshop.

    This plugin scrapes papers from the ML4PS workshop website and fetches abstracts
    from the NeurIPS virtual conference pages.
    """

    plugin_name = "ml4ps"
    plugin_description = "ML4PS (Machine Learning for Physical Sciences) workshop downloader"
    supported_years = [2025]  # Currently only 2025 is implemented

    BASE_URL = "https://ml4physicalsciences.github.io/2025/"
    NEURIPS_VIRTUAL_BASE = "https://neurips.cc/virtual/2025/loc/san-diego/poster/"

    def __init__(self):
        """Initialize the ML4PS downloader plugin."""
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
        self.stats_lock = Lock()

    def download(
        self,
        year: Optional[int] = None,
        output_path: Optional[str] = None,
        force_download: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Download papers from ML4PS workshop.

        Parameters
        ----------
        year : int, optional
            Workshop year (default: 2025)
        output_path : str, optional
            Path to save the downloaded JSON file
        force_download : bool
            Force re-download even if file exists
        fetch_abstracts : bool, optional
            Whether to fetch abstracts from NeurIPS virtual site (default: True)
        max_workers : int, optional
            Maximum parallel workers for fetching abstracts (default: 20)
        **kwargs : Any
            Additional parameters

        Returns
        -------
        dict
            Downloaded data in format:
            {
                'count': int,
                'next': None,
                'previous': None,
                'results': [list of papers]
            }
        """
        if year is None:
            year = 2025

        # Validate year
        self.validate_year(year)

        # Check if file already exists and should be loaded
        if output_path and not force_download:
            output_file = Path(output_path)
            if output_file.exists():
                logger.info(f"Loading existing data from: {output_file}")
                try:
                    with open(output_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    logger.info(f"Successfully loaded {data.get('count', 0)} papers from local file")
                    return data
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Failed to load local file: {str(e)}. Downloading from web...")

        # Get options from kwargs
        fetch_abstracts = kwargs.get("fetch_abstracts", True)
        max_workers = kwargs.get("max_workers", 10)

        logger.info(f"Scraping ML4PS {year} workshop papers...")

        # Scrape papers
        papers = self._scrape_papers()

        if not papers:
            logger.error("No papers were scraped")
            return {"count": 0, "next": None, "previous": None, "results": []}

        # Fetch abstracts if requested
        if fetch_abstracts:
            logger.info(f"Fetching abstracts for {len(papers)} papers...")
            self._fetch_abstracts_for_papers(papers, max_workers=max_workers)

        # Convert to database format
        data = self._convert_to_database_format(papers)

        # Save to file if path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved to {output_file}")

        logger.info(f"Successfully downloaded {data['count']} papers from ML4PS {year}")

        return data

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get plugin metadata.

        Returns
        -------
        dict
            Plugin metadata including name, description, supported years
        """
        return {
            "name": self.plugin_name,
            "description": self.plugin_description,
            "supported_years": self.supported_years,
            "parameters": {
                "year": {
                    "type": "int",
                    "required": True,
                    "description": "Workshop year to download",
                    "default": 2025,
                },
                "output_path": {"type": "str", "required": False, "description": "Path to save the downloaded data"},
                "force_download": {
                    "type": "bool",
                    "required": False,
                    "description": "Force re-download even if file exists",
                    "default": False,
                },
                "fetch_abstracts": {
                    "type": "bool",
                    "required": False,
                    "description": "Whether to fetch abstracts from NeurIPS virtual site",
                    "default": True,
                },
                "max_workers": {
                    "type": "int",
                    "required": False,
                    "description": "Maximum parallel workers for fetching abstracts",
                    "default": 20,
                },
            },
        }

    # Internal helper methods

    def _fetch_page(self, url: str, max_retries: int = 3) -> Optional[BeautifulSoup]:
        """Fetch and parse a web page."""
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                return BeautifulSoup(response.content, "html.parser")
            except requests.RequestException as e:
                logger.warning(f"Failed to fetch {url}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    return None

    def _clean_text(self, text: str) -> str:
        """Clean text by removing extra whitespace and unwanted characters."""
        text = re.sub(r"\[\s*paper\s*\]", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\[\s*poster\s*\]", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\[\s*video\s*\]", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\[\s*\]", "", text)
        text = re.sub(r"[🏅]", "", text)
        text = " ".join(text.split())
        return text.strip()

    def _parse_authors(self, authors_str: str) -> List[Dict[str, Any]]:
        """Parse authors string into list of author objects."""
        if not authors_str:
            return []

        authors = []
        author_names = [name.strip() for name in authors_str.split(",") if name.strip()]

        for idx, fullname in enumerate(author_names):
            author_id = -(idx + 1) * 1000000  # Temporary ID scheme
            authors.append({"id": author_id, "fullname": fullname, "url": "", "institution": ""})

        return authors

    def _extract_paper_id_from_poster_url(self, poster_url: str) -> Optional[str]:
        """Extract paper_id from poster URL."""
        match = re.search(r"/(\d+)\.png$", poster_url)
        if match:
            return match.group(1)
        return None

    def _create_eventmedia(
        self,
        paper_id: int,
        paper_url: Optional[str],
        poster_url: Optional[str],
        video_url: Optional[str],
        neurips_paper_id: Optional[str],
        openreview_url: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Create eventmedia list in the format expected by the database schema."""
        eventmedia = []
        media_id = paper_id * 1000
        timestamp = datetime.now().isoformat()

        # Add OpenReview link if available
        if openreview_url:
            eventmedia.append(
                {
                    "id": media_id + 1,
                    "modified": timestamp,
                    "display_section": 1,
                    "type": "URL",
                    "name": "OpenReview",
                    "visible": True,
                    "sortkey": 0,
                    "is_live_content": False,
                    "uri": openreview_url,
                    "resourcetype": "UriEventmedia",
                }
            )

        # Add NeurIPS Virtual link if we have NeurIPS paper ID
        if neurips_paper_id:
            eventmedia.append(
                {
                    "id": media_id + 2,
                    "modified": timestamp,
                    "display_section": 1,
                    "type": "URL",
                    "name": "NeurIPS Virtual",
                    "visible": True,
                    "sortkey": 0,
                    "is_live_content": False,
                    "uri": f"https://neurips.cc/virtual/2025/loc/san-diego/poster/{neurips_paper_id}",
                    "resourcetype": "UriEventmedia",
                }
            )

        # Add poster
        if poster_url:
            eventmedia.append(
                {
                    "id": media_id + 3,
                    "file": poster_url,
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

        # Add PDF
        if paper_url:
            eventmedia.append(
                {
                    "id": media_id + 4,
                    "modified": timestamp,
                    "display_section": 1,
                    "type": "PDF",
                    "name": "Paper",
                    "visible": True,
                    "sortkey": 0,
                    "is_live_content": False,
                    "uri": paper_url,
                    "resourcetype": "UriEventmedia",
                }
            )

        # Add video
        if video_url:
            eventmedia.append(
                {
                    "id": media_id + 5,
                    "modified": timestamp,
                    "display_section": 1,
                    "type": "URL",
                    "name": "Video",
                    "visible": True,
                    "sortkey": 0,
                    "is_live_content": False,
                    "uri": video_url,
                    "resourcetype": "UriEventmedia",
                }
            )

        return eventmedia

    def _fetch_abstract_and_openreview(self, paper_id: str) -> tuple[Optional[str], Optional[str]]:
        """Fetch abstract and OpenReview URL from NeurIPS virtual conference page."""
        url = f"{self.NEURIPS_VIRTUAL_BASE}{paper_id}"
        soup = self._fetch_page(url)

        if not soup:
            return None, None

        abstract_text = None
        openreview_url = None

        # Find the Abstract section
        abstract_heading = soup.find(["h3", "h4", "h5"], string=re.compile(r"Abstract", re.IGNORECASE))

        if abstract_heading:
            abstract_content = abstract_heading.find_next(["p", "div"])
            if abstract_content:
                abstract_text = abstract_content.get_text(strip=True)
                abstract_text = " ".join(abstract_text.split())
                abstract_text = re.sub(r"^Abstract\s*", "", abstract_text, flags=re.IGNORECASE)

        # Alternative: Look for div with class containing 'abstract'
        if not abstract_text:
            abstract_div = soup.find("div", class_=re.compile(r"abstract", re.IGNORECASE))
            if abstract_div:
                abstract_text = abstract_div.get_text(strip=True)
                abstract_text = " ".join(abstract_text.split())
                abstract_text = re.sub(r"^Abstract\s*", "", abstract_text, flags=re.IGNORECASE)

        # Find OpenReview link
        openreview_link = soup.find("a", class_=re.compile(r"action-btn"), href=re.compile(r"openreview\.net"))

        if not openreview_link:
            openreview_link = soup.find("a", href=re.compile(r"openreview\.net"))

        if openreview_link:
            openreview_url = openreview_link.get("href")
            if openreview_url and not openreview_url.startswith("http"):
                openreview_url = urljoin("https://openreview.net", openreview_url)

        return abstract_text, openreview_url

    def _extract_paper_info_from_row(self, row) -> Optional[Dict]:
        """Extract paper information from a table row."""
        cells = row.find_all("td")
        if len(cells) < 2:
            return None

        try:
            # Extract paper ID
            paper_id = cells[0].get_text(strip=True)
            if not paper_id.isdigit():
                return None

            # Extract content from second cell
            content = cells[1]

            # Extract links
            links = content.find_all("a")
            paper_url = None
            poster_url = None
            video_url = None

            for link in links:
                href = link.get("href", "")
                link_text = link.get_text(strip=True).lower()

                if link_text == "paper":
                    paper_url = urljoin(self.BASE_URL, href)
                elif link_text == "poster":
                    poster_url = urljoin(self.BASE_URL, href)
                elif link_text == "video":
                    video_url = urljoin(self.BASE_URL, href)

            # Extract title from <strong> tag
            strong_tag = content.find("strong")
            if strong_tag:
                title = strong_tag.get_text(strip=True)
            else:
                title = ""

            # Extract authors
            authors = ""
            br_tag = content.find("br")
            if br_tag:
                if br_tag.string:
                    authors = br_tag.string
                elif br_tag.contents:
                    authors_parts = []
                    for item in br_tag.contents:
                        if isinstance(item, str):
                            authors_parts.append(item.strip())
                        else:
                            authors_parts.append(item.get_text(strip=True))
                    authors = " ".join(authors_parts)

            # Clean title and authors
            title = self._clean_text(title)
            authors_str = self._clean_text(authors)

            # Extract awards and determine event type
            full_text = content.get_text(separator=" ", strip=True)
            awards = []
            eventtype = "Poster"
            decision = "Accept (poster)"

            if "Spotlight Talk" in full_text:
                awards.append("Spotlight Talk")
                eventtype = "Spotlight"
                decision = "Accept (spotlight)"
            if "Best Poster (by Popular Vote -- tie)" in full_text:
                awards.append("Best Poster (by Popular Vote -- tie)")
            elif "Best Poster" in full_text:
                awards.append("Best Poster")
            if "Reproducibility Prize" in full_text:
                awards.append("Reproducibility Prize")

            # Remove awards from authors if they got mixed in
            for award in awards:
                authors_str = authors_str.replace(award, "").replace("🏅", "")
            authors_str = " ".join(authors_str.split())

            paper_info = {
                "id": int(paper_id),
                "title": title,
                "authors_str": authors_str,
                "paper_url": paper_url,
                "poster_url": poster_url,
                "video_url": video_url,
                "awards": awards if awards else [],
                "abstract": None,
                "neurips_paper_id": None,
                "eventtype": eventtype,
                "decision": decision,
            }

            return paper_info

        except Exception as e:
            logger.warning(f"Error parsing paper row: {e}")
            return None

    def _scrape_papers(self) -> List[Dict]:
        """Scrape all papers from the workshop page."""
        logger.info("Scraping ML4PS 2025 workshop papers...")

        soup = self._fetch_page(self.BASE_URL)
        if not soup:
            return []

        # Find papers section
        papers_section = soup.find("h2", string=re.compile(r"Papers"))
        if not papers_section:
            logger.error("Could not find Papers section")
            return []

        # Find table
        table = papers_section.find_next("table")
        if not table:
            logger.error("Could not find papers table")
            return []

        # Extract papers
        papers = []
        rows = table.find_all("tr")
        logger.info(f"Processing {len(rows)} rows...")

        for row in rows:
            paper = self._extract_paper_info_from_row(row)
            if paper:
                papers.append(paper)
                if len(papers) % 50 == 0:
                    logger.info(f"Extracted {len(papers)} papers...")

        logger.info(f"Successfully scraped {len(papers)} papers")
        return papers

    def _fetch_single_abstract(self, paper: Dict) -> tuple[Dict, bool]:
        """Fetch abstract for a single paper."""
        if not paper.get("poster_url"):
            return paper, False

        neurips_paper_id = self._extract_paper_id_from_poster_url(paper["poster_url"])

        if not neurips_paper_id:
            return paper, False

        paper["neurips_paper_id"] = neurips_paper_id

        # Fetch abstract and OpenReview URL
        abstract, openreview_url = self._fetch_abstract_and_openreview(neurips_paper_id)

        if abstract:
            paper["abstract"] = abstract

        if openreview_url:
            paper["openreview_url"] = openreview_url

        success = abstract is not None or openreview_url is not None
        return paper, success

    def _fetch_abstracts_for_papers(self, papers: List[Dict], max_workers: int = 20):
        """Fetch abstracts for all papers in parallel."""
        logger.info(f"Fetching abstracts for {len(papers)} papers using {max_workers} parallel workers...")

        success_count = 0
        fail_count = 0
        processed_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_paper = {executor.submit(self._fetch_single_abstract, paper): paper for paper in papers}

            for future in as_completed(future_to_paper):
                paper, success = future.result()

                with self.stats_lock:
                    processed_count += 1
                    if success:
                        success_count += 1
                    else:
                        fail_count += 1

                    if processed_count % 10 == 0:
                        logger.info(
                            f"Progress: {processed_count}/{len(papers)} papers processed "
                            f"({success_count} abstracts fetched, {fail_count} failed)"
                        )

        logger.info(f"Abstract fetching complete: {success_count} successful, {fail_count} failed")

    def _convert_to_database_format(self, papers: List[Dict]) -> Dict[str, Any]:
        """Convert scraped papers to database schema format."""
        results = []
        author_counter = 1

        for paper in papers:
            # Parse authors
            authors_list = self._parse_authors(paper.get("authors_str", ""))

            # Update author IDs to be unique
            for author in authors_list:
                author["id"] = author_counter
                author_counter += 1

            # Create eventmedia
            eventmedia = self._create_eventmedia(
                paper["id"],
                paper.get("paper_url"),
                paper.get("poster_url"),
                paper.get("video_url"),
                paper.get("neurips_paper_id"),
                paper.get("openreview_url"),
            )

            # Determine virtualsite_url
            virtualsite_url = ""
            if paper.get("neurips_paper_id"):
                virtualsite_url = f"/virtual/2025/loc/san-diego/poster/{paper['neurips_paper_id']}"

            openreview_url = paper.get("openreview_url", "")
            pdf_url = paper.get("paper_url", "")

            # Create paper in database format
            db_paper = {
                "id": paper["id"],
                "uid": f"ml4ps2025_{paper['id']}",
                "name": paper["title"],
                "authors": authors_list,
                "abstract": paper.get("abstract", ""),
                "topic": "ML4PhysicalSciences Workshop",
                "keywords": [],
                "decision": paper.get("decision", "Accept (poster)"),
                "session": "ML4PhysicalSciences 2025 Workshop",
                "eventtype": paper.get("eventtype", "Poster"),
                "event_type": "Workshop Poster",
                "room_name": "",
                "virtualsite_url": virtualsite_url,
                "url": openreview_url or pdf_url,
                "sourceid": None,
                "sourceurl": "https://ml4physicalsciences.github.io/2025/",
                "starttime": "",
                "endtime": "",
                "starttime2": None,
                "endtime2": None,
                "diversity_event": None,
                "paper_url": openreview_url,
                "paper_pdf_url": pdf_url,
                "children_url": None,
                "children": [],
                "children_ids": [],
                "parent1": "",
                "parent2": None,
                "parent2_id": None,
                "eventmedia": eventmedia,
                "show_in_schedule_overview": False,
                "visible": True,
                "poster_position": "",
                "schedule_html": "",
                "latitude": None,
                "longitude": None,
                "related_events": [],
                "related_events_ids": [],
            }

            # Add awards as keywords
            if paper.get("awards"):
                db_paper["keywords"] = paper["awards"]

            results.append(db_paper)

        return {"count": len(results), "next": None, "previous": None, "results": results}


# Auto-register the plugin when imported
def _register():
    """Auto-register the ML4PS plugin."""
    from neurips_abstracts.plugins import register_plugin

    plugin = ML4PSDownloaderPlugin()
    register_plugin(plugin)
    logger.debug("ML4PS downloader plugin registered")


# Register on import
_register()

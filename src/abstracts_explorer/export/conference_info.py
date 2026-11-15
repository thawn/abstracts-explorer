"""
Conference website metadata fetching.
"""

import logging
import re
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


def fetch_conference_info() -> Optional[Dict[str, Any]]:
    """
    Fetch conference information from NeurIPS website.

    Returns
    -------
    dict or None
        Conference information dictionary with keys: name, dates, location, description
        Returns None if fetching fails

    Examples
    --------
    >>> conf_info = fetch_conference_info()
    >>> if conf_info:
    ...     print(conf_info['name'], conf_info['dates'])
    """
    try:
        # Try to import BeautifulSoup
        try:
            from bs4 import BeautifulSoup  # type: ignore[import-not-found]
        except ImportError:
            logger.warning("BeautifulSoup not installed, using fallback conference info")
            return None

        conference_url = "https://neurips.cc/Conferences/2025"
        response = requests.get(conference_url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        conf_info = {
            "name": "38th Conference on Neural Information Processing Systems (NeurIPS 2025)",
            "dates": None,
            "location": None,
            "description": None,
        }

        # Try to find conference title
        title_elem = soup.find("h1") or soup.find("title")
        if title_elem:
            title_text = title_elem.get_text(strip=True)
            if "NeurIPS" in title_text or "Neural Information Processing Systems" in title_text:
                conf_info["name"] = title_text

        # Try to find dates - look for patterns like "December 9-15, 2025"
        page_text = soup.get_text()
        date_patterns = [
            r"(December\s+\d+[-–]\d+,\s+\d{4})",
            r"(Dec\s+\d+[-–]\d+,\s+\d{4})",
            r"(\d{1,2}[-–]\d{1,2}\s+December\s+\d{4})",
        ]
        for pattern in date_patterns:
            match = re.search(pattern, page_text)
            if match:
                conf_info["dates"] = match.group(1)
                break

        # Try to find location - look for "Vancouver" mentions
        location_patterns = [
            r"(Vancouver\s+Convention\s+Centre(?:,\s+(?:Vancouver|BC|British\s+Columbia))?(?:,\s+Canada)?)",
            r"(Vancouver(?:,\s+(?:BC|British\s+Columbia))?(?:,\s+Canada)?)",
        ]
        for pattern in location_patterns:
            match = re.search(pattern, page_text)
            if match:
                conf_info["location"] = match.group(1)
                break

        # Try to find conference description/about text
        about_section = soup.find(["div", "p"], class_=re.compile(r"about|description", re.I))
        if about_section:
            desc_text = about_section.get_text(strip=True)
            # Limit description length
            if len(desc_text) > 500:
                desc_text = desc_text[:500] + "..."
            conf_info["description"] = desc_text

        logger.info(f"Fetched conference info: {conf_info}")
        return conf_info

    except Exception as e:
        logger.warning(f"Failed to fetch conference info from website: {e}")
        return None

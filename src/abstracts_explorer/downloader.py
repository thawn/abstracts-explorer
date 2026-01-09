"""
Downloader Module
=================

This module provides functionality to download JSON data from URLs.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import requests

logger = logging.getLogger(__name__)


class DownloadError(Exception):
    """Exception raised when download fails."""

    pass


def download_json(
    url: str,
    output_path: Optional[Union[str, Path]] = None,
    timeout: int = 30,
    verify_ssl: bool = True,
    force_download: bool = False,
) -> Dict[str, Any]:
    """
    Download JSON data from a URL.

    Parameters
    ----------
    url : str
        The URL to download JSON data from.
    output_path : str or Path, optional
        Path to save the downloaded JSON file. If None, the data is not saved to disk.
    timeout : int, default=30
        Request timeout in seconds.
    verify_ssl : bool, default=True
        Whether to verify SSL certificates.
    force_download : bool, default=False
        If True, download even if the file already exists locally.
        If False, load from local file if it exists.

    Returns
    -------
    dict
        The downloaded JSON data as a dictionary.

    Raises
    ------
    DownloadError
        If the download fails or the response is not valid JSON.
    ValueError
        If the URL is empty or invalid.

    Examples
    --------
    >>> data = download_json("https://example.com/data.json")
    >>> print(data.keys())

    >>> data = download_json(
    ...     "https://example.com/data.json",
    ...     output_path="data/neurips.json"
    ... )

    >>> # Force re-download even if file exists
    >>> data = download_json(
    ...     "https://example.com/data.json",
    ...     output_path="data/neurips.json",
    ...     force_download=True
    ... )
    """
    if not url or not isinstance(url, str):
        raise ValueError("URL must be a non-empty string")

    # Check if file already exists locally and load it if not forcing download
    if output_path and not force_download:
        output_file = Path(output_path)
        if output_file.exists():
            logger.info(f"Loading existing data from: {output_file}")
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info(f"Successfully loaded JSON data from local file ({len(str(data))} bytes)")
                return data
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load local file: {str(e)}. Downloading from URL...")
                # Continue to download if local file is corrupted

    logger.info(f"Downloading JSON from: {url}")

    try:
        response = requests.get(url, timeout=timeout, verify=verify_ssl)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise DownloadError(f"Failed to download from {url}: {str(e)}") from e

    try:
        data = response.json()
    except json.JSONDecodeError as e:
        raise DownloadError(f"Invalid JSON response from {url}: {str(e)}") from e

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved JSON data to: {output_file}")

    logger.info(f"Successfully downloaded JSON data ({len(str(data))} bytes)")
    return data


def download_conference_data(
    year: int = 2025,
    output_path: Optional[Union[str, Path]] = None,
    timeout: int = 30,
    force_download: bool = False,
) -> Dict[str, Any]:
    """
    Download NeurIPS conference data for a specific year.

    Parameters
    ----------
    year : int, default=2025
        The year of the NeurIPS conference.
    output_path : str or Path, optional
        Path to save the downloaded JSON file. If None, the data is not saved to disk.
    timeout : int, default=30
        Request timeout in seconds.
    force_download : bool, default=False
        If True, download even if the file already exists locally.
        If False, load from local file if it exists.

    Returns
    -------
    dict
        The downloaded NeurIPS data as a dictionary.

    Raises
    ------
    DownloadError
        If the download fails.

    Examples
    --------
    >>> data = download_conference_data(2025)
    >>> papers = data.get('results', [])
    >>> print(f"Found {len(papers)} papers")

    >>> # Save to file and reuse on subsequent calls
    >>> data = download_conference_data(2025, output_path="data/neurips_2025.json")
    >>> # Next call will load from local file without downloading
    >>> data = download_conference_data(2025, output_path="data/neurips_2025.json")

    >>> # Force re-download
    >>> data = download_conference_data(2025, output_path="data/neurips_2025.json", force_download=True)
    """
    url = f"https://neurips.cc/static/virtual/data/neurips-{year}-orals-posters.json"
    return download_json(url, output_path=output_path, timeout=timeout, force_download=force_download)

"""
Registry module for uploading and downloading data to/from OCI-compatible container registries.

This module provides functionality to push and pull abstracts-explorer data artifacts
(paper databases, embedding databases, clustering caches) to OCI-compatible registries
such as GitHub Container Registry (ghcr.io).

Artifacts are pushed and pulled using the `oras <https://oras-project.github.io/oras-py/>`_
Python SDK. Each artifact is tagged by conference (e.g. ``neurips``) or by conference
and year (e.g. ``neurips-2024``).  A conference-only tag contains all available years
with each year stored as its own pair of OCI layers (paper DB + embeddings).

Examples
--------
Upload data for a specific year::

    from abstracts_explorer.registry import RegistryClient

    client = RegistryClient(
        repository="ghcr.io/owner/abstracts-data",
        token="ghp_xxxx",
    )
    client.upload(conference="neurips", year=2024)

Upload all available years for a conference::

    client.upload(conference="neurips")

Download data from the registry::

    client.download(conference="neurips", year=2024)

List available tags::

    tags = client.list_tags()
"""

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import oras.client
import oras.defaults
import oras.oci
import oras.provider

logger = logging.getLogger(__name__)

# Custom media types for abstracts-explorer artifacts
PAPER_DB_MEDIA_TYPE = "application/vnd.abstracts-explorer.paper-db.v1.tar+gzip"
EMBEDDING_DB_MEDIA_TYPE = "application/vnd.abstracts-explorer.embedding-db.v1.tar+gzip"
CONFIG_MEDIA_TYPE = "application/vnd.abstracts-explorer.config.v1+json"


class RegistryError(Exception):
    """Exception raised for registry operation errors."""

    pass


def _build_tag(conference: str, year: Optional[int] = None) -> str:
    """
    Build an OCI tag from conference name and optional year.

    Parameters
    ----------
    conference : str
        Conference name.
    year : int, optional
        Conference year.  When ``None``, the tag contains only the
        conference name (e.g. ``neurips``).

    Returns
    -------
    str
        Tag string (e.g. ``neurips-2024`` or ``neurips``).
    """
    safe_name = conference.lower().replace(" ", "-").replace("/", "-").replace("@", "-")
    if year is not None:
        return f"{safe_name}-{year}"
    return safe_name


class RegistryClient:
    """
    Client for pushing and pulling data artifacts to/from OCI-compatible registries.

    Uses the `oras <https://oras-project.github.io/oras-py/>`_ Python SDK to
    interact with OCI registries.

    The smallest unit of upload/download is a **conference + year** combination.
    Each artifact always contains the paper database, embeddings, and clustering
    cache together to prevent inconsistent data.

    When ``year`` is omitted, all available years for the conference are uploaded
    or downloaded, with each year stored as its own pair of OCI layers.

    Parameters
    ----------
    repository : str
        Full OCI repository path (e.g., ``ghcr.io/owner/abstracts-data``).
    token : str, optional
        Authentication token (e.g., GitHub Personal Access Token).
        If not provided, will try the ``GITHUB_TOKEN`` environment variable.

    Raises
    ------
    RegistryError
        If the repository format is invalid.

    Examples
    --------
    >>> client = RegistryClient("ghcr.io/owner/abstracts-data", token="ghp_xxxx")
    >>> client.list_tags()
    ['neurips-2024', 'iclr-2025']
    """

    def __init__(self, repository: str, token: Optional[str] = None):
        parts = repository.split("/", 1)
        if len(parts) < 2 or not parts[0] or not parts[1]:
            raise RegistryError(
                f"Invalid repository format: '{repository}'. "
                "Expected format: 'registry/owner/name' (e.g., 'ghcr.io/owner/abstracts-data')"
            )

        self.registry = parts[0]
        self.name = parts[1]
        self.repository = repository
        self.token = token or os.environ.get("GITHUB_TOKEN", "")

        # Create oras client
        self._client = oras.client.OrasClient(hostname=self.registry, insecure=False)
        if self.token:
            self._client.login(
                username="_token",
                password=self.token,
                hostname=self.registry,
            )

    def list_tags(self) -> List[str]:
        """
        List available tags in the repository.

        Returns
        -------
        list of str
            Available tags.

        Raises
        ------
        RegistryError
            If listing fails.
        """
        try:
            return self._client.get_tags(self.repository)
        except Exception as e:
            raise RegistryError(f"Failed to list tags: {e}") from e

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_years_for_conference(conference: str) -> List[int]:
        """
        Return the distinct years available in the local database for *conference*.

        Parameters
        ----------
        conference : str
            Conference name.

        Returns
        -------
        list of int
            Sorted list of years.
        """
        from .database import DatabaseManager

        with DatabaseManager() as db:
            db.create_tables()
            return db.get_years_for_conference(conference)

    def _export_year(
        self,
        conference: str,
        year: int,
        temp_dir: Path,
        progress: Callable[[str], None],
    ) -> Dict[str, Any]:
        """
        Export paper DB and embeddings for a single conference+year.

        Returns a dict with ``paper_db_path``, ``embeddings_path``,
        ``paper_count``, and ``embedding_count``.
        """
        from .database import DatabaseManager
        from .embeddings import EmbeddingsManager

        # --- paper DB ---
        progress(f"Exporting paper database for {conference}/{year}...")
        paper_db_path = temp_dir / f"papers-{year}.db"
        with DatabaseManager() as db:
            db.create_tables()
            paper_count = db.export_papers_to_sqlite(paper_db_path, conference, year)

        if paper_count == 0:
            raise RegistryError(f"No papers found for {conference}/{year}. Download the conference data first.")

        progress(f"  Exported {paper_count} papers")

        # --- embeddings ---
        progress(f"Exporting embeddings for {conference}/{year}...")
        em = EmbeddingsManager()
        embeddings_data = em.export_embeddings(conference, year)
        embedding_count = len(embeddings_data.get("ids", []))

        if embedding_count == 0:
            raise RegistryError(
                f"No embeddings found for {conference}/{year}."
                " Create embeddings first with 'abstracts-explorer create-embeddings'."
            )

        embeddings_path = temp_dir / f"embeddings-{year}.json"
        embeddings_path.write_text(json.dumps(embeddings_data))
        progress(f"  Exported {embedding_count} embeddings")

        return {
            "paper_db_path": paper_db_path,
            "embeddings_path": embeddings_path,
            "paper_count": paper_count,
            "embedding_count": embedding_count,
        }

    def _import_year(
        self,
        conference: str,
        year: int,
        paper_db_file: Path,
        embeddings_file: Path,
        progress: Callable[[str], None],
    ) -> Dict[str, int]:
        """
        Import paper DB and embeddings for a single conference+year.

        Returns a dict with ``paper_count`` and ``embedding_count``.
        """
        from .database import DatabaseManager
        from .embeddings import EmbeddingsManager

        result: Dict[str, int] = {}

        # --- paper DB ---
        if paper_db_file.exists():
            progress(f"Importing paper database for {conference}/{year}...")
            with DatabaseManager() as db:
                db.create_tables()
                paper_count = db.import_papers_from_sqlite(paper_db_file, conference, year)
            progress(f"  Imported {paper_count} papers")
            result["paper_count"] = paper_count
        else:
            result["paper_count"] = 0

        # --- embeddings ---
        if embeddings_file.exists():
            progress(f"Importing embeddings for {conference}/{year}...")
            embeddings_data = json.loads(embeddings_file.read_text())
            em = EmbeddingsManager()
            embedding_count = em.import_embeddings(embeddings_data, conference, year)
            progress(f"  Imported {embedding_count} embeddings")
            result["embedding_count"] = embedding_count
        else:
            result["embedding_count"] = 0

        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upload(
        self,
        conference: str,
        year: Optional[int] = None,
        tag: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Upload data for a conference (and optionally a specific year) to the registry.

        Packages the paper database, embeddings, and clustering cache as OCI
        layers and pushes them together.  All three must be present for every
        year; an error is raised if any data is missing.

        When *year* is ``None``, all years available in the local database for
        the given conference are uploaded.  Each year is stored as its own pair
        of layers (paper DB + embeddings) so that individual years can be
        identified inside the manifest.

        Parameters
        ----------
        conference : str
            Conference name (e.g. ``neurips``).
        year : int, optional
            Conference year (e.g. ``2024``).  When ``None``, all available
            years are uploaded.
        tag : str, optional
            Custom tag.  If ``None``, derived from conference and year.
        progress_callback : callable, optional
            Function called with status messages during upload.

        Returns
        -------
        dict
            Upload summary with paper count, embedding count, years, and tag.

        Raises
        ------
        RegistryError
            If upload fails or required data is missing.
        """
        from ._version import __version__

        if tag is None:
            tag = _build_tag(conference, year)

        def _progress(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)
            logger.info(msg)

        # Determine which years to upload
        if year is not None:
            years = [year]
        else:
            years = self._get_years_for_conference(conference)
            if not years:
                raise RegistryError(
                    f"No data found for conference '{conference}'. Download the conference data first."
                )
            _progress(f"Found years for {conference}: {years}")

        temp_dir = Path(tempfile.mkdtemp())
        try:
            files: List[str] = []
            total_papers = 0
            total_embeddings = 0

            for yr in years:
                yr_data = self._export_year(conference, yr, temp_dir, _progress)
                files.append(str(yr_data["paper_db_path"]))
                files.append(str(yr_data["embeddings_path"]))
                total_papers += yr_data["paper_count"]
                total_embeddings += yr_data["embedding_count"]

            # --- Build config metadata ---
            config_data = {
                "version": __version__,
                "conference": conference,
                "years": years,
                "paper_count": total_papers,
                "embedding_count": total_embeddings,
            }
            config_path = temp_dir / "config.json"
            config_path.write_text(json.dumps(config_data, indent=2))

            # --- Push via oras ---
            _progress("Uploading to registry...")
            target = f"{self.repository}:{tag}"

            manifest_annotations = {
                "com.abstracts-explorer.version": __version__,
                "com.abstracts-explorer.conference": conference,
                "com.abstracts-explorer.years": ",".join(str(y) for y in years),
                "com.abstracts-explorer.paper-count": str(total_papers),
                "com.abstracts-explorer.embedding-count": str(total_embeddings),
            }

            self._client.push(
                target=target,
                files=files,
                manifest_config=str(config_path),
                manifest_annotations=manifest_annotations,
            )

            _progress(f"Successfully pushed {target}")

            return {
                "tag": tag,
                "conference": conference,
                "years": years,
                "paper_count": total_papers,
                "embedding_count": total_embeddings,
            }

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def download(
        self,
        conference: str,
        year: Optional[int] = None,
        tag: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Download data for a conference (and optionally a specific year) from the registry.

        Pulls the paper database, embeddings, and clustering cache and
        replaces existing local data for the specified conference and year(s).

        When *year* is ``None``, all years contained in the artifact are
        downloaded.

        Parameters
        ----------
        conference : str
            Conference name (e.g. ``neurips``).
        year : int, optional
            Conference year (e.g. ``2024``).  When ``None``, all years
            in the artifact are imported.
        tag : str, optional
            Custom tag.  If ``None``, derived from conference and year.
        progress_callback : callable, optional
            Function called with status messages during download.

        Returns
        -------
        dict
            Download summary with paper count and embedding count.

        Raises
        ------
        RegistryError
            If download fails.
        """
        if tag is None:
            tag = _build_tag(conference, year)

        def _progress(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)
            logger.info(msg)

        temp_dir = Path(tempfile.mkdtemp())
        try:
            # --- 1. Pull from oras ---
            target = f"{self.repository}:{tag}"
            _progress(f"Pulling {target}...")

            pulled_files = self._client.pull(target=target, outdir=str(temp_dir))
            _progress(f"Downloaded {len(pulled_files)} files")

            # Read config metadata if available
            metadata: Dict[str, Any] = {}
            for fpath in pulled_files:
                p = Path(fpath)
                if p.name == "config.json":
                    metadata = json.loads(p.read_text())
                    _progress(f"Artifact version: {metadata.get('version', 'unknown')}")
                    break

            # --- 2. Group files by year ---
            # Files are named papers-YYYY.db and embeddings-YYYY.json
            # (or papers.db / embeddings.json for legacy single-year artifacts)
            year_files: Dict[int, Dict[str, Path]] = {}
            for fpath in pulled_files:
                p = Path(fpath)
                name = p.name
                if name.startswith("papers-") and name.endswith(".db"):
                    try:
                        yr = int(name[len("papers-") : -len(".db")])
                    except ValueError:
                        logger.warning(f"Skipping file with invalid year format: {name}")
                        continue
                    year_files.setdefault(yr, {})["paper_db"] = p
                elif name.startswith("embeddings-") and name.endswith(".json"):
                    try:
                        yr = int(name[len("embeddings-") : -len(".json")])
                    except ValueError:
                        logger.warning(f"Skipping file with invalid year format: {name}")
                        continue
                    year_files.setdefault(yr, {})["embeddings"] = p
                elif name == "papers.db":
                    # Legacy single-year format
                    legacy_year = year or metadata.get("year")
                    if legacy_year:
                        year_files.setdefault(int(legacy_year), {})["paper_db"] = p
                elif name == "embeddings.json":
                    legacy_year = year or metadata.get("year")
                    if legacy_year:
                        year_files.setdefault(int(legacy_year), {})["embeddings"] = p

            # If user requested a specific year, filter
            if year is not None:
                year_files = {yr: files for yr, files in year_files.items() if yr == year}

            # --- 3. Import each year ---
            total_papers = 0
            total_embeddings = 0
            imported_years: List[int] = []

            for yr in sorted(year_files.keys()):
                files = year_files[yr]
                paper_db = files.get("paper_db")
                embeddings = files.get("embeddings")

                if not paper_db or not embeddings:
                    _progress(f"Warning: Incomplete data for {conference}/{yr}, skipping")
                    continue

                result = self._import_year(conference, yr, paper_db, embeddings, _progress)
                total_papers += result["paper_count"]
                total_embeddings += result["embedding_count"]
                imported_years.append(yr)

            if not imported_years:
                _progress("Warning: No data found in artifact to import")

            _progress("Download complete!")
            return {
                "tag": tag,
                "conference": conference,
                "years": imported_years,
                "paper_count": total_papers,
                "embedding_count": total_embeddings,
                "metadata": metadata,
            }

        except RegistryError:
            raise
        except Exception as e:
            raise RegistryError(f"Failed to download: {e}") from e
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def get_artifact_info(self, tag: str) -> Dict[str, Any]:
        """
        Get metadata about a specific artifact tag.

        Parameters
        ----------
        tag : str
            Tag to inspect.

        Returns
        -------
        dict
            Artifact metadata including version, conference, year, and counts.

        Raises
        ------
        RegistryError
            If the tag is not found or cannot be read.
        """
        try:
            target = f"{self.repository}:{tag}"
            manifest = self._client.get_manifest(target)

            info: Dict[str, Any] = {
                "tag": tag,
                "annotations": manifest.get("annotations", {}),
                "layers": [],
            }

            for layer in manifest.get("layers", []):
                layer_info = {
                    "media_type": layer.get("mediaType", ""),
                    "size": layer.get("size", 0),
                    "annotations": layer.get("annotations", {}),
                }
                info["layers"].append(layer_info)

            return info

        except Exception as e:
            raise RegistryError(f"Failed to get artifact info for '{tag}': {e}") from e

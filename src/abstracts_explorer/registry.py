"""
Registry module for uploading and downloading data to/from OCI-compatible container registries.

This module provides functionality to push and pull abstracts-explorer data artifacts
(paper databases, embedding databases, clustering caches) to OCI-compatible registries
such as GitHub Container Registry (ghcr.io).

Artifacts are pushed and pulled using the `oras <https://oras-project.github.io/oras-py/>`_
Python SDK. Each artifact is tagged by conference and year (e.g. ``neurips-2024``) and
always contains the paper database, embedding database, and clustering cache together.

Examples
--------
Upload data to GitHub Container Registry::

    from abstracts_explorer.registry import RegistryClient

    client = RegistryClient(
        repository="ghcr.io/owner/abstracts-data",
        token="ghp_xxxx",
    )
    client.upload(conference="neurips", year=2024)

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


def _build_tag(conference: str, year: int) -> str:
    """
    Build an OCI tag from conference name and year.

    Parameters
    ----------
    conference : str
        Conference name.
    year : int
        Conference year.

    Returns
    -------
    str
        Tag string (e.g. ``neurips-2024``).
    """
    safe_name = conference.lower().replace(" ", "-").replace("/", "-").replace("@", "-")
    return f"{safe_name}-{year}"


class RegistryClient:
    """
    Client for pushing and pulling data artifacts to/from OCI-compatible registries.

    Uses the `oras <https://oras-project.github.io/oras-py/>`_ Python SDK to
    interact with OCI registries.

    The smallest unit of upload/download is a **conference + year** combination.
    Each artifact always contains the paper database, embeddings, and clustering
    cache together to prevent inconsistent data.

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

    def upload(
        self,
        conference: str,
        year: int,
        tag: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Upload data for a conference and year to the registry.

        Packages the paper database, embeddings, and clustering cache as OCI
        layers and pushes them together. All three must be present; an error
        is raised if any data is missing.

        Parameters
        ----------
        conference : str
            Conference name (e.g. ``neurips``).
        year : int
            Conference year (e.g. ``2024``).
        tag : str, optional
            Custom tag. If ``None``, derived from conference and year.
        progress_callback : callable, optional
            Function called with status messages during upload.

        Returns
        -------
        dict
            Upload summary with paper count, embedding count, and tag.

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

        temp_dir = Path(tempfile.mkdtemp())
        try:
            # --- 1. Export paper database ---
            _progress("Exporting paper database...")
            paper_db_path = temp_dir / "papers.db"

            from .database import DatabaseManager

            with DatabaseManager() as db:
                db.create_tables()
                paper_count = db.export_papers_to_sqlite(paper_db_path, conference, year)

            if paper_count == 0:
                raise RegistryError(f"No papers found for {conference}/{year}. Download the conference data first.")
            _progress(f"Exported {paper_count} papers")

            # --- 2. Export embeddings ---
            _progress("Exporting embeddings...")
            from .embeddings import EmbeddingsManager

            em = EmbeddingsManager()
            embeddings_data = em.export_embeddings(conference, year)
            embedding_count = len(embeddings_data.get("ids", []))

            if embedding_count == 0:
                raise RegistryError(
                    f"No embeddings found for {conference}/{year}."
                    " Create embeddings first with 'abstracts-explorer create-embeddings'."
                )

            embeddings_path = temp_dir / "embeddings.json"
            embeddings_path.write_text(json.dumps(embeddings_data))
            _progress(f"Exported {embedding_count} embeddings")

            # --- 3. Build config metadata ---
            config_data = {
                "version": __version__,
                "conference": conference,
                "year": year,
                "paper_count": paper_count,
                "embedding_count": embedding_count,
            }
            config_path = temp_dir / "config.json"
            config_path.write_text(json.dumps(config_data, indent=2))

            # --- 4. Push via oras ---
            _progress("Uploading to registry...")
            target = f"{self.repository}:{tag}"

            files = [str(paper_db_path), str(embeddings_path)]
            manifest_annotations = {
                "com.abstracts-explorer.version": __version__,
                "com.abstracts-explorer.conference": conference,
                "com.abstracts-explorer.year": str(year),
                "com.abstracts-explorer.paper-count": str(paper_count),
                "com.abstracts-explorer.embedding-count": str(embedding_count),
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
                "year": year,
                "paper_count": paper_count,
                "embedding_count": embedding_count,
            }

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def download(
        self,
        conference: str,
        year: int,
        tag: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Download data for a conference and year from the registry.

        Pulls the paper database, embeddings, and clustering cache and
        replaces existing local data for the specified conference and year.

        Parameters
        ----------
        conference : str
            Conference name (e.g. ``neurips``).
        year : int
            Conference year (e.g. ``2024``).
        tag : str, optional
            Custom tag. If ``None``, derived from conference and year.
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

            # Locate the expected files
            paper_db_file = None
            embeddings_file = None
            config_file = None

            for fpath in pulled_files:
                p = Path(fpath)
                if p.name.endswith(".db"):
                    paper_db_file = p
                elif p.name == "embeddings.json":
                    embeddings_file = p
                elif p.name == "config.json":
                    config_file = p

            # Read config metadata if available
            metadata: Dict[str, Any] = {}
            if config_file and config_file.exists():
                metadata = json.loads(config_file.read_text())
                _progress(f"Artifact version: {metadata.get('version', 'unknown')}")

            summary: Dict[str, Any] = {
                "tag": tag,
                "conference": conference,
                "year": year,
                "metadata": metadata,
            }

            # --- 2. Import paper database ---
            if paper_db_file and paper_db_file.exists():
                _progress("Importing paper database...")
                from .database import DatabaseManager

                with DatabaseManager() as db:
                    db.create_tables()
                    paper_count = db.import_papers_from_sqlite(paper_db_file, conference, year)
                _progress(f"Imported {paper_count} papers")
                summary["paper_count"] = paper_count
            else:
                _progress("Warning: No paper database found in artifact")
                summary["paper_count"] = 0

            # --- 3. Import embeddings ---
            if embeddings_file and embeddings_file.exists():
                _progress("Importing embeddings...")
                embeddings_data = json.loads(embeddings_file.read_text())

                from .embeddings import EmbeddingsManager

                em = EmbeddingsManager()
                embedding_count = em.import_embeddings(embeddings_data, conference, year)
                _progress(f"Imported {embedding_count} embeddings")
                summary["embedding_count"] = embedding_count
            else:
                _progress("Warning: No embeddings file found in artifact")
                summary["embedding_count"] = 0

            _progress("Download complete!")
            return summary

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

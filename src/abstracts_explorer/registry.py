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
import re
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


def _sanitize_model_name(model: str) -> str:
    """
    Sanitize an embedding model name for use as an OCI tag component.

    The name is lowercased and characters not in ``[a-z0-9._-]`` are
    replaced with ``-``.  OCI tags allow ``[a-zA-Z0-9_.-]``.

    Parameters
    ----------
    model : str
        Embedding model name.

    Returns
    -------
    str
        Tag-safe model name.
    """
    safe = model.lower()
    safe = re.sub(r"[^a-z0-9._-]", "-", safe)
    # Collapse consecutive hyphens
    safe = re.sub(r"-{2,}", "-", safe)
    return safe.strip("-")


def _build_tag(
    conference: str,
    year: Optional[int] = None,
    *,
    embedding_model: str,
) -> str:
    """
    Build an OCI tag from conference name, embedding model and optional year.

    Parameters
    ----------
    conference : str
        Conference name.
    year : int, optional
        Conference year.  When ``None``, the tag contains only the
        conference name (e.g. ``neurips``).
    embedding_model : str
        Embedding model name.  Appended to the tag after a ``_``
        separator (e.g. ``neurips-2024_text-embedding-ada-002``).

    Returns
    -------
    str
        Tag string (e.g. ``neurips-2024_text-embedding-ada-002``).
    """
    safe_name = conference.lower().replace(" ", "-").replace("/", "-").replace("@", "-")
    if year is not None:
        tag = f"{safe_name}-{year}"
    else:
        tag = safe_name
    tag = f"{tag}_{_sanitize_model_name(embedding_model)}"
    return tag


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

    @staticmethod
    def _get_all_conferences() -> List[str]:
        """
        Return distinct conferences available in the local database.

        Returns
        -------
        list of str
            Sorted list of conference names.
        """
        from .database import DatabaseManager

        with DatabaseManager() as db:
            db.create_tables()
            filters = db.get_filter_options()
            return sorted(filters.get("conferences", []))

    @staticmethod
    def _get_embedding_model() -> Optional[str]:
        """
        Return the embedding model stored in the local database.

        Returns
        -------
        str or None
            Embedding model name, or ``None`` if not set.
        """
        from .database import DatabaseManager

        with DatabaseManager() as db:
            db.create_tables()
            return db.get_embedding_model()

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

        Both *paper_db_file* and *embeddings_file* must exist.  If either
        import fails, any already-imported data for this conference+year is
        rolled back to prevent inconsistency between the paper DB and the
        embedding DB.

        Returns a dict with ``paper_count`` and ``embedding_count``.

        Raises
        ------
        RegistryError
            If either file is missing or an import step fails.
        """
        from .database import DatabaseManager
        from .embeddings import EmbeddingsManager

        # --- pre-flight: both files must exist ---
        if not paper_db_file.exists() or not embeddings_file.exists():
            missing = []
            if not paper_db_file.exists():
                missing.append(f"paper DB ({paper_db_file.name})")
            if not embeddings_file.exists():
                missing.append(f"embeddings ({embeddings_file.name})")
            raise RegistryError(
                f"Incomplete data for {conference}/{year}: missing {', '.join(missing)}. "
                "Cannot import — both paper DB and embeddings must be present."
            )

        # --- import paper DB first ---
        progress(f"Importing paper database for {conference}/{year}...")
        with DatabaseManager() as db:
            db.create_tables()
            paper_count = db.import_papers_from_sqlite(paper_db_file, conference, year)
        progress(f"  Imported {paper_count} papers")

        # --- import embeddings; rollback paper DB on failure ---
        try:
            progress(f"Importing embeddings for {conference}/{year}...")
            embeddings_data = json.loads(embeddings_file.read_text())
            em = EmbeddingsManager()
            embedding_count = em.import_embeddings(embeddings_data, conference, year)
            progress(f"  Imported {embedding_count} embeddings")
        except Exception as embed_err:
            # Roll back paper DB import so both stay consistent
            progress(f"  Embedding import failed — rolling back paper DB for {conference}/{year}...")
            try:
                from sqlalchemy import and_ as sa_and
                from sqlalchemy import delete as sa_delete

                from .db_models import Paper

                with DatabaseManager() as db:
                    db.create_tables()
                    db._session.execute(  # type: ignore[union-attr]
                        sa_delete(Paper).where(sa_and(Paper.conference == conference, Paper.year == year))
                    )
                    db._session.commit()  # type: ignore[union-attr]
            except Exception:
                logger.warning("Failed to roll back paper DB import after embedding failure", exc_info=True)
            raise RegistryError(
                f"Embedding import failed for {conference}/{year}: {embed_err}. "
                "Paper DB changes have been rolled back."
            ) from embed_err

        return {
            "paper_count": paper_count,
            "embedding_count": embedding_count,
        }

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
            Custom tag.  If ``None``, derived from embedding model, conference and year.
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

        def _progress(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)
            logger.info(msg)

        # --- Determine embedding model (needed for auto-tag) ---
        embedding_model = self._get_embedding_model()
        if not embedding_model:
            raise RegistryError(
                "No embedding model found in local database. "
                "Create embeddings first with 'abstracts-explorer create-embeddings'."
            )

        if tag is None:
            tag = _build_tag(conference, year, embedding_model=embedding_model)

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
                "embedding_model": embedding_model,
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
                "com.abstracts-explorer.embedding-model": embedding_model,
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
        embedding_model: Optional[str] = None,
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
            Custom tag.  If ``None``, derived from embedding model, conference and year.
        embedding_model : str, optional
            Embedding model name used for tag derivation.  When ``None``
            and *tag* is also ``None``, the model is read from the local
            database metadata or the ``EMBEDDING_MODEL`` configuration.
            A ``RegistryError`` is raised if the model cannot be determined.
        progress_callback : callable, optional
            Function called with status messages during download.

        Returns
        -------
        dict
            Download summary with paper count and embedding count.

        Raises
        ------
        RegistryError
            If download fails or the embedding model cannot be determined.
        """
        if tag is None:
            if embedding_model is None:
                embedding_model = self._get_embedding_model()
            if not embedding_model:
                raise RegistryError(
                    "No embedding model specified and none found in local database. "
                    "Use --embedding-model to specify the model name."
                )
            tag = _build_tag(conference, year, embedding_model=embedding_model)

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

            # --- 3. Validate completeness ---
            for yr in sorted(year_files.keys()):
                files = year_files[yr]
                if not files.get("paper_db") or not files.get("embeddings"):
                    missing = []
                    if not files.get("paper_db"):
                        missing.append("paper DB")
                    if not files.get("embeddings"):
                        missing.append("embeddings")
                    raise RegistryError(
                        f"Incomplete data for {conference}/{yr}: missing {', '.join(missing)}. "
                        "Cannot import — both paper DB and embeddings must be present."
                    )

            # --- 4. Import each year ---
            total_papers = 0
            total_embeddings = 0
            imported_years: List[int] = []

            for yr in sorted(year_files.keys()):
                files = year_files[yr]
                paper_db = files["paper_db"]
                embeddings = files["embeddings"]

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

    def upload_all(
        self,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Upload data for **all** conferences available locally.

        Each conference is uploaded as a separate OCI artifact with a
        conference-only tag (containing all years for that conference).

        Parameters
        ----------
        progress_callback : callable, optional
            Function called with status messages during upload.

        Returns
        -------
        list of dict
            Upload summaries, one per conference.

        Raises
        ------
        RegistryError
            If no conferences are found or any upload fails.
        """
        conferences = self._get_all_conferences()
        if not conferences:
            raise RegistryError("No conference data found in local database.")

        def _progress(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)
            logger.info(msg)

        _progress(f"Found {len(conferences)} conference(s): {conferences}")

        summaries: List[Dict[str, Any]] = []
        for conf in conferences:
            _progress(f"\n--- Uploading {conf} ---")
            summary = self.upload(conference=conf, progress_callback=progress_callback)
            summaries.append(summary)

        return summaries

    def download_all(
        self,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Download data for **all** conference tags in the registry.

        Lists available tags and downloads every conference-level tag
        (i.e. tags without a year suffix).

        Parameters
        ----------
        progress_callback : callable, optional
            Function called with status messages during download.

        Returns
        -------
        list of dict
            Download summaries, one per conference tag.

        Raises
        ------
        RegistryError
            If no tags are found or any download fails.
        """
        tags = self.list_tags()
        if not tags:
            raise RegistryError("No tags found in registry.")

        def _progress(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)
            logger.info(msg)

        _progress(f"Found {len(tags)} tag(s) in registry")

        summaries: List[Dict[str, Any]] = []
        for tag in sorted(tags):
            _progress(f"\n--- Downloading {tag} ---")
            # Read manifest annotations to derive conference/year
            try:
                info = self.get_artifact_info(tag)
                annotations = info.get("annotations", {})
                conf = annotations.get("com.abstracts-explorer.conference", "")
                years_str = annotations.get("com.abstracts-explorer.years", "")
            except RegistryError:
                conf = ""
                years_str = ""

            if not conf:
                # Fallback: derive conference from tag by splitting at underscore
                # (tag format: "conference[-year]_model" or legacy "conference[-year]")
                base = tag.split("_", 1)[0]
                parts = base.rsplit("-", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    conf = parts[0]
                else:
                    conf = base

            # Determine year from annotations
            yr: Optional[int] = None
            if years_str:
                year_vals = [int(y) for y in years_str.split(",") if y.strip().isdigit()]
                if len(year_vals) == 1:
                    yr = year_vals[0]
                # If multiple years, leave yr=None to download all
            else:
                # Fallback: derive year from tag
                base = tag.split("_", 1)[0]
                parts = base.rsplit("-", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    yr = int(parts[1])

            summary = self.download(
                conference=conf,
                year=yr,
                tag=tag,
                progress_callback=progress_callback,
            )
            summaries.append(summary)

        return summaries

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

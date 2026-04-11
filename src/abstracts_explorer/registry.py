"""
Registry module for uploading and downloading data to/from OCI-compatible container registries.

This module provides functionality to push and pull abstracts-explorer data artifacts
(paper databases, embedding databases, clustering caches) to OCI-compatible registries
such as GitHub Container Registry (ghcr.io).

Artifacts are pushed and pulled using the `oras <https://oras-project.github.io/oras-py/>`_
Python SDK. Each artifact is tagged by conference (e.g. ``neurips``) or by conference
and year (e.g. ``neurips-2024``).  A conference-only tag contains all available years
with each year stored as its own set of OCI layers (paper DB + embeddings +
clustering cache).

Examples
--------
Upload data for a specific year::

    from abstracts_explorer.registry import RegistryClient

    client = RegistryClient(
        repository="ghcr.io/thawn/abstracts-data",
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
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import oras.client
import oras.defaults
import oras.oci
import oras.provider
import requests
from packaging.version import InvalidVersion, Version

from abstracts_explorer._version import __version__
from abstracts_explorer.config import get_config

logger = logging.getLogger(__name__)

# Custom media types for abstracts-explorer artifacts
PAPER_DB_MEDIA_TYPE = "application/vnd.abstracts-explorer.paper-db.v1.tar+gzip"
EMBEDDING_DB_MEDIA_TYPE = "application/vnd.abstracts-explorer.embedding-db.v1.tar+gzip"
CLUSTERING_CACHE_MEDIA_TYPE = "application/vnd.abstracts-explorer.clustering-cache.v1.tar+gzip"
CONFIG_MEDIA_TYPE = "application/vnd.abstracts-explorer.config.v1+json"


class RegistryError(Exception):
    """Exception raised for registry operation errors."""

    pass


class EmbeddingModelMismatchError(RegistryError):
    """
    Raised when the embedding model in the local database does not match the model in the downloaded artifact.

    Attributes
    ----------
    local_model : str
        Embedding model currently stored in the local database.
    remote_model : str
        Embedding model used by the downloaded artifact.
    """

    def __init__(self, local_model: str, remote_model: str) -> None:
        self.local_model = local_model
        self.remote_model = remote_model
        super().__init__(
            f"Embedding model mismatch: local database uses '{local_model}' "
            f"but downloaded artifact uses '{remote_model}'. "
            f"Cannot import data created with a different embedding model."
        )


def _sanitize_str_for_oci_tag(value: str) -> str:
    """
    Sanitize a string for use as an OCI tag component.

    The value is lowercased and characters not in ``[a-z0-9._-]`` are
    replaced with ``-``.  OCI tags allow ``[a-zA-Z0-9_.-]``.  In
    particular the ``+`` local-version separator used by PEP 440
    (e.g. ``1.2.3+g1a2b3c4``) is replaced with ``-``.  Consecutive
    hyphens are collapsed and leading/trailing hyphens are stripped.

    Parameters
    ----------
    value : str
        String to sanitize (e.g. a model name or a PEP 440 version).

    Returns
    -------
    str
        Tag-safe string (e.g. ``text-embedding-ada-002`` or
        ``0.1.dev2-g2abcfb2a2``).
    """
    safe = value.lower()
    safe = re.sub(r"[^a-z0-9._-]", "-", safe)
    # Collapse consecutive hyphens
    safe = re.sub(r"-{2,}", "-", safe)
    return safe.strip("-")


def _build_tag(
    conference: str,
    year: Optional[int] = None,
    *,
    embedding_model: str,
    version: Optional[str] = None,
) -> str:
    """
    Build an OCI tag from conference name, embedding model, version and optional year.

    Parameters
    ----------
    conference : str
        Conference name.
    year : int, optional
        Conference year.  When ``None``, the tag contains only the
        conference name (e.g. ``neurips``).
    embedding_model : str
        Embedding model name.  Appended to the tag after a ``_``
        separator (e.g. ``neurips-2024_text-embedding-ada-002_1.0.0``).
    version : str, optional
        abstracts-explorer version string.  When ``None``, the current
        package version (``__version__``) is used.  The version is
        sanitized for OCI tag use and appended after a ``_`` separator
        (e.g. ``neurips-2024_text-embedding-ada-002_0.1.0``).

    Returns
    -------
    str
        Tag string (e.g. ``neurips-2024_text-embedding-ada-002_0.1.0``).
    """
    if version is None:
        version = __version__
    safe_name = conference.lower().replace(" ", "-").replace("/", "-").replace("@", "-")
    if year is not None:
        tag = f"{safe_name}-{year}"
    else:
        tag = safe_name
    tag = f"{tag}_{_sanitize_str_for_oci_tag(embedding_model)}_{_sanitize_str_for_oci_tag(version)}"
    return tag


def _parse_version_from_tag(tag: str) -> Optional[Version]:
    """
    Extract and parse the version component from an OCI tag.

    OCI tags in this project have the format
    ``{conference}[-{year}]_{model}_{version}``.  The version is always the
    last ``_``-separated component.  Dots are preserved by the sanitization
    step so a normal semantic-version string (e.g. ``0.4.1``) round-trips
    unchanged.

    OCI sanitization (``_sanitize_str_for_oci_tag``) replaces the ``+``
    PEP 440 local-version separator with ``-``, so dev versions such as
    ``0.4.6.dev16+g7005b7837`` appear in tags as ``0.4.6.dev16-g7005b7837``.
    This function recovers the original version by trying all ``-`` → ``+``
    substitution positions until one produces a valid PEP 440 version.

    Parameters
    ----------
    tag : str
        OCI tag string (without repository prefix), e.g.
        ``neurips-2024_text-embedding-ada-002_0.4.1`` or
        ``ml4ps-neurips-2022_model_0.4.6.dev16-g7005b7837``.

    Returns
    -------
    packaging.version.Version or None
        Parsed version, or ``None`` if the tag contains no ``_`` separator
        or the version component cannot be parsed.
    """
    if "_" not in tag:
        return None
    raw = tag.rsplit("_", 1)[-1]
    # Fast path: standard release or pre-release without a local segment.
    try:
        return Version(raw)
    except InvalidVersion:
        pass
    # OCI sanitization replaces '+' (PEP 440 local-version separator) with '-'.
    # Try restoring '+' at each '-' position until we find a valid version.
    parts = raw.split("-")
    for i in range(1, len(parts)):
        candidate = "-".join(parts[:i]) + "+" + "-".join(parts[i:])
        try:
            return Version(candidate)
        except InvalidVersion:
            continue
    return None


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
        Full OCI repository path (e.g., ``ghcr.io/thawn/abstracts-data``).
    token : str, optional
        Authentication token (e.g., GitHub Personal Access Token).
        If not provided, will try the ``GITHUB_TOKEN`` environment variable.

    Raises
    ------
    RegistryError
        If the repository format is invalid.

    Examples
    --------
    >>> client = RegistryClient("ghcr.io/thawn/abstracts-data", token="ghp_xxxx")
    >>> client.list_tags()
    ['neurips-2024', 'iclr-2025']
    """

    def __init__(self, repository: str, token: Optional[str] = None):
        parts = repository.split("/", 1)
        if len(parts) < 2 or not parts[0] or not parts[1]:
            raise RegistryError(
                f"Invalid repository format: '{repository}'. "
                "Expected format: 'registry/owner/name' (e.g., 'ghcr.io/thawn/abstracts-data')"
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
        from abstracts_explorer.database import DatabaseManager

        with DatabaseManager() as db:
            db.create_tables()
            return db.get_years_for_conference(conference)

    @staticmethod
    def _get_embedding_model_database() -> Optional[str]:
        """
        Return the embedding model stored in the local database.

        Returns
        -------
        str or None
            Embedding model name, or ``None`` if not set.
        """
        from abstracts_explorer.database import DatabaseManager

        with DatabaseManager() as db:
            db.create_tables()
            return db.get_embedding_model()

    @staticmethod
    def _is_conference_level_tag(tag: str) -> bool:
        """
        Return ``True`` when *tag* is a conference-level tag (no year suffix).

        OCI tags in this project have the format
        ``{conference}[-{year}]_{model}_{version}``.  Year-specific tags
        contain a ``-YYYY`` suffix in the base component (before the first
        ``_``), e.g. ``chi-2026_model_0.4.1``.  Conference-level tags
        omit the year, e.g. ``chi_model_0.4.1``.

        Download operations should only use conference-level tags so that
        each year is not imported twice (once from its dedicated year tag
        and again from the combined conference tag).

        Parameters
        ----------
        tag : str
            OCI tag string (without repository prefix).

        Returns
        -------
        bool
            ``True`` if the tag does not carry a year suffix.
        """
        base = tag.split("_", 1)[0]
        parts = base.rsplit("-", 1)
        if len(parts) != 2:
            return True
        suffix = parts[1]
        # A year suffix is exactly 4 digits.
        return not (suffix.isdigit() and len(suffix) == 4)

    def _find_best_matching_tag(self, tag: str) -> str:
        """
        Resolve *tag* to the best matching tag available in the registry.

        OCI tags in this project have the format
        ``{conference}[-year]_{model}_{version}``.  When the exact tag does not
        exist in the registry (e.g. because the local package version differs
        from the version used when the artifact was pushed), this method strips
        the version suffix and looks for other tags that share the same
        ``{conference}[-year]_{model}`` prefix.  The candidate with the
        lexicographically highest version suffix is returned.

        If the exact tag exists, it is returned unchanged.  If listing tags
        fails or no prefix-matching candidate is found, *tag* is returned
        unchanged so that the caller can still attempt the operation and
        produce an informative error message.

        Parameters
        ----------
        tag : str
            OCI tag to resolve (without repository prefix, e.g.
            ``neurips-2024_my-model_0.4.2``).

        Returns
        -------
        str
            The resolved tag.
        """
        try:
            available_tags = self._client.get_tags(self.repository)
        except Exception as exc:
            logger.debug("Could not list tags for tag resolution: %s", exc)
            return tag

        if tag in available_tags:
            return tag

        # Strip the version suffix (the last '_'-separated component) and search
        # for tags that share the same prefix.  The version is always the last
        # component, so rsplit("_", 1) correctly isolates it even when the model
        # name itself contains underscores.
        if "_" not in tag:
            return tag

        # first split off the version suffix, then iteratively strip components from the end of the prefix until we find candidates that match the start of the tag.  This allows us to resolve to a tag with a different version and/or model, as long as the conference and year match.
        candidates: List[str] = []
        prefix: str = tag
        maxsplit: int = 1
        while candidates == [] and "_" in prefix:
            prefix = prefix.rsplit("_", 1)[0]
            candidates = [t for t in available_tags if t.rsplit("_", maxsplit=maxsplit)[0] == prefix]
            maxsplit += 1

        if not candidates:
            return tag

        # Return the candidate with the highest lexicographic version suffix.
        # This works correctly for standard semver strings (e.g. "0.4.1" < "0.4.2").
        resolved = max(candidates)
        logger.debug("Tag '%s' not found; resolved to closest match '%s'", tag, resolved)
        return resolved

    def _get_manifest_embedding_model(self, target: str) -> Optional[str]:
        """
        Retrieve the embedding model name from the OCI manifest for *target*.

        Fetches the manifest from the registry and reads the
        ``com.abstracts-explorer.embedding-model`` label without downloading any
        artifact data.  Checks both the ``labels`` field (used by
        abstracts-explorer when pushing) and the ``annotations`` field
        (used by some alternative OCI implementations).

        Parameters
        ----------
        target : str
            Full OCI reference including tag (e.g.
            ``ghcr.io/thawn/abstracts-data:neurips-2024_model_1.0.0``).

        Returns
        -------
        str or None
            The embedding model stored in the manifest, or ``None`` if the
            manifest has no such label (e.g. legacy artifacts) or if the
            manifest could not be fetched.
        """
        try:
            manifest = self._client.get_manifest(target)
            labels = manifest.get("labels") or manifest.get("annotations") or {}
            result = labels.get("com.abstracts-explorer.embedding-model")
            return result if isinstance(result, str) else None
        except Exception as exc:
            logger.debug("Could not fetch manifest for %s: %s", target, exc)
            return None

    @staticmethod
    def clear_local_embedding_data() -> None:
        """
        Clear all local embedding data — metadata, ChromaDB collection, and clustering cache.

        This is a destructive operation that removes *all* embedding-related data from
        the local databases so that data with a different embedding model can be imported.
        Use with care.

        After calling this method, the next download will import fresh data and establish
        a new embedding model association in the local database.
        """
        from abstracts_explorer.database import DatabaseManager
        from abstracts_explorer.db_models import EmbeddingsMetadata, HierarchicalLabelCache
        from abstracts_explorer.embeddings import EmbeddingsManager
        from sqlalchemy import delete as sa_delete

        # 1. Clear EmbeddingsMetadata and clustering/hierarchical caches from SQLite
        with DatabaseManager() as db:
            db.create_tables()
            db._session.execute(sa_delete(EmbeddingsMetadata))  # type: ignore[union-attr]
            db.clear_clustering_cache()
            db._session.execute(sa_delete(HierarchicalLabelCache))  # type: ignore[union-attr]
            db._session.commit()  # type: ignore[union-attr]

        # 2. Reset the ChromaDB collection
        em = EmbeddingsManager()
        em.create_collection(reset=True)

    def _export_year(
        self,
        conference: str,
        year: int,
        temp_dir: Path,
        progress: Callable[[str], None],
    ) -> Dict[str, Any]:
        """
        Export paper DB, embeddings, and clustering cache for a single conference+year.

        Returns a dict with ``paper_db_path``, ``embeddings_path``,
        ``clustering_cache_path``, ``paper_count``, ``embedding_count``,
        and ``clustering_cache_count``.
        """
        from abstracts_explorer.database import DatabaseManager
        from abstracts_explorer.embeddings import EmbeddingsManager

        # --- paper DB (without clustering cache — it goes into a separate layer) ---
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

        # --- clustering cache (separate layer) ---
        progress(f"Exporting clustering cache for {conference}/{year}...")
        with DatabaseManager() as db:
            db.create_tables()
            cache_data = db.export_clustering_cache_to_json(conference, year)
        clustering_cache_count = len(cache_data.get("entries", []))

        if clustering_cache_count == 0:
            raise RegistryError(
                f"No clustering cache found for {conference}/{year}."
                " Generate the clustering cache first with 'abstracts-explorer clustering pre-generate'."
            )

        clustering_cache_path = temp_dir / f"clustering-{year}.json"
        clustering_cache_path.write_text(json.dumps(cache_data))
        progress(f"  Exported {clustering_cache_count} clustering cache entries")

        return {
            "paper_db_path": paper_db_path,
            "embeddings_path": embeddings_path,
            "clustering_cache_path": clustering_cache_path,
            "paper_count": paper_count,
            "embedding_count": embedding_count,
            "clustering_cache_count": clustering_cache_count,
        }

    @staticmethod
    def _read_artifact_embedding_model(paper_db_file: Path) -> Optional[str]:
        """
        Read the embedding model name from the ``embeddings_metadata`` table
        in a downloaded artifact's paper DB.

        Parameters
        ----------
        paper_db_file : Path
            Path to the artifact's SQLite paper database.

        Returns
        -------
        str or None
            The embedding model stored in the artifact, or ``None`` if
            the table does not exist or is empty (legacy artifacts).
        """
        try:
            with sqlite3.connect(str(paper_db_file)) as conn:
                row = conn.execute(
                    "SELECT embedding_model FROM embeddings_metadata ORDER BY updated_at DESC LIMIT 1"
                ).fetchone()
                return row[0] if row else None
        except (sqlite3.OperationalError, sqlite3.DatabaseError) as exc:
            logger.debug(
                "Could not read embedding model from artifact DB %s: %s: %s",
                paper_db_file.name,
                type(exc).__name__,
                exc,
            )
            return None

    @staticmethod
    def _replace_artifact_embedding_model(paper_db_file: Path, new_model: str) -> None:
        """
        Overwrite the embedding model in the artifact's paper DB so that
        subsequent imports do not trigger model-consistency checks.

        Parameters
        ----------
        paper_db_file : Path
            Path to the artifact's SQLite paper database.
        new_model : str
            The model name to write.
        """
        try:
            with sqlite3.connect(str(paper_db_file)) as conn:
                conn.execute(
                    "UPDATE embeddings_metadata SET embedding_model = ? "
                    "WHERE rowid = (SELECT rowid FROM embeddings_metadata ORDER BY updated_at DESC LIMIT 1)",
                    (new_model,),
                )
        except (sqlite3.OperationalError, sqlite3.DatabaseError) as exc:
            logger.debug(
                "Could not update embedding model in artifact DB %s: %s: %s",
                paper_db_file.name,
                type(exc).__name__,
                exc,
            )

    @staticmethod
    def _check_embedding_model(
        paper_db_file: Path,
        embedding_model: str,
        ignore_embedding_model_mismatch: bool,
        progress: Callable[[str], None],
    ) -> None:
        """
        Single authoritative embedding-model check for one conference/year import.

        Reads the model stored in the artifact paper DB and compares it to
        *embedding_model*.  When the models differ:

        * If *ignore_embedding_model_mismatch* is ``False``,
          ``EmbeddingModelMismatchError`` is raised.
        * If *ignore_embedding_model_mismatch* is ``True``, the artifact
          DB is patched in-place so that the downstream
          ``import_papers_from_sqlite()`` consistency check will not
          trigger again.

        Parameters
        ----------
        paper_db_file : Path
            Path to the artifact's SQLite paper database.
        embedding_model : str
            The configured/expected embedding model name.
        ignore_embedding_model_mismatch : bool
            When ``True``, overwrite the artifact model and continue.
        progress : callable
            Status-message callback.

        Raises
        ------
        EmbeddingModelMismatchError
            When models differ and *ignore_embedding_model_mismatch* is ``False``.
        """
        artifact_model = RegistryClient._read_artifact_embedding_model(paper_db_file)
        if not artifact_model:
            return  # Legacy artifact without metadata — nothing to check

        if _sanitize_str_for_oci_tag(artifact_model) == _sanitize_str_for_oci_tag(embedding_model):
            return  # Models match — nothing to do

        if not ignore_embedding_model_mismatch:
            raise EmbeddingModelMismatchError(local_model=embedding_model, remote_model=artifact_model)

        # Models differ but the user explicitly asked to proceed.
        progress(
            f"⚠️  Embedding model mismatch ignored for {paper_db_file.name}:\n"
            f"  Configured model: '{embedding_model}'\n"
            f"  Artifact model:   '{artifact_model}'\n"
            f"  Replacing artifact model with configured model."
        )
        RegistryClient._replace_artifact_embedding_model(paper_db_file, embedding_model)

    def _import_year(
        self,
        conference: str,
        year: int,
        paper_db_file: Path,
        embeddings_file: Path,
        progress: Callable[[str], None],
        embedding_model: Optional[str] = None,
        ignore_embedding_model_mismatch: bool = False,
        clustering_cache_file: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Import paper DB, embeddings, and clustering cache for a single conference+year.

        All three files (*paper_db_file*, *embeddings_file*, and
        *clustering_cache_file*) must exist.  If any import fails, any
        already-imported data for this conference+year is rolled back to
        prevent inconsistency between the paper DB and the embedding DB.

        The embedding-model consistency check happens **here** — this is
        the single authoritative location.  When
        *ignore_embedding_model_mismatch* is ``True`` and the models
        differ, the artifact DB is patched before the import so that the
        downstream ``import_papers_from_sqlite()`` check will not trigger
        again.

        Returns a dict with ``paper_count``, ``embedding_count``, and
        ``clustering_cache_count``.

        Raises
        ------
        EmbeddingModelMismatchError
            If the artifact's embedding model differs from *embedding_model*
            and *ignore_embedding_model_mismatch* is ``False``.
        RegistryError
            If any file is missing or an import step fails.
        """
        from abstracts_explorer.database import DatabaseManager
        from abstracts_explorer.embeddings import EmbeddingsManager

        # --- pre-flight: all three files must exist ---
        missing = []
        if not paper_db_file.exists():
            missing.append(f"paper DB ({paper_db_file.name})")
        if not embeddings_file.exists():
            missing.append(f"embeddings ({embeddings_file.name})")
        if clustering_cache_file is None or not clustering_cache_file.exists():
            missing.append(
                f"clustering cache ({clustering_cache_file.name if clustering_cache_file else 'not provided'})"
            )
        if missing:
            raise RegistryError(
                f"Incomplete data for {conference}/{year}: missing {', '.join(missing)}. "
                "Cannot import — paper DB, embeddings, and clustering cache must all be present."
            )

        # --- single authoritative embedding-model check ---
        if embedding_model:
            self._check_embedding_model(paper_db_file, embedding_model, ignore_embedding_model_mismatch, progress)

        # --- import paper DB first ---
        progress(f"Importing paper database for {conference}/{year}...")
        try:
            with DatabaseManager() as db:
                db.create_tables()
                paper_count = db.import_papers_from_sqlite(paper_db_file, conference, year)
        except Exception as db_err:
            from abstracts_explorer.database import EmbeddingModelConflictError

            if isinstance(db_err, EmbeddingModelConflictError):
                raise EmbeddingModelMismatchError(db_err.local_model, db_err.remote_model) from db_err
            raise RegistryError(f"Paper DB import failed for {conference}/{year}: {db_err}") from db_err
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

                from abstracts_explorer.db_models import Paper

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

        # --- import clustering cache from separate layer ---
        clustering_cache_count = 0
        progress(f"Importing clustering cache for {conference}/{year}...")
        try:
            cache_data = json.loads(clustering_cache_file.read_text())  # type: ignore[union-attr]
            with DatabaseManager() as db:
                db.create_tables()
                clustering_cache_count = db.import_clustering_cache_from_json(
                    cache_data,
                    conference,
                    year,
                    overwrite_embedding_model=embedding_model if embedding_model else None,
                )
            progress(f"  Imported {clustering_cache_count} clustering cache entries")
        except Exception as cache_err:
            raise RegistryError(f"Clustering cache import failed for {conference}/{year}: {cache_err}") from cache_err

        return {
            "paper_count": paper_count,
            "embedding_count": embedding_count,
            "clustering_cache_count": clustering_cache_count,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _push_tag(
        self,
        conference: str,
        years_in_tag: List[int],
        files: List[str],
        config_path: Path,
        tag: str,
        embedding_model: str,
        paper_count: int,
        embedding_count: int,
        progress: Callable[[str], None],
    ) -> None:
        """Push a single OCI tag with the given files.

        Parameters
        ----------
        conference : str
            Conference name stored in manifest annotations.
        years_in_tag : list of int
            Years whose data files are included in this tag.
        files : list of str
            Absolute paths to the blob files to push.
        config_path : Path
            Path to the JSON config blob.
        tag : str
            OCI tag string (without repository prefix).
        embedding_model : str
            Embedding model name stored in manifest annotations.
        paper_count : int
            Total paper count stored in manifest annotations.
        embedding_count : int
            Total embedding count stored in manifest annotations.
        progress : callable
            Progress message callback.
        """
        target = f"{self.repository}:{tag}"
        manifest_annotations = {
            "com.abstracts-explorer.version": __version__,
            "com.abstracts-explorer.conference": conference,
            "com.abstracts-explorer.years": ",".join(str(y) for y in years_in_tag),
            "com.abstracts-explorer.paper-count": str(paper_count),
            "com.abstracts-explorer.embedding-count": str(embedding_count),
            "com.abstracts-explorer.embedding-model": embedding_model,
        }
        self._client.push(
            target=target,
            files=files,
            manifest_config=str(config_path),
            manifest_annotations=manifest_annotations,
            disable_path_validation=True,
        )
        progress(f"Successfully pushed {target}")

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

        When *year* is not ``None``, a single per-year tag is pushed
        (e.g. ``neurips-2024_model``).

        When *year* is ``None``, every year available locally is first pushed as
        its own individual tag (e.g. ``neurips-2024_model``, ``neurips-2025_model``)
        and then an all-years summary tag (e.g. ``neurips_model``) is pushed
        containing all years' files as layers.  Because OCI blobs are
        content-addressed, the registry deduplicates the files — no data is
        actually stored twice.

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
            Upload summary with paper count, embedding count, years, tag, and
            (when multiple years) ``year_tags`` listing the per-year tags pushed.

        Raises
        ------
        RegistryError
            If upload fails or required data is missing.
        """

        def _progress(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)
            logger.info(msg)

        # --- Determine embedding model (needed for auto-tag) ---
        embedding_model = self._get_embedding_model_database()
        if not embedding_model:
            raise RegistryError(
                "No embedding model found in local database. "
                "Create embeddings first with 'abstracts-explorer create-embeddings'."
            )

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

        # Build the target tag (for single-year or all-years summary)
        if tag is None:
            tag = _build_tag(conference, year, embedding_model=embedding_model)

        temp_dir = Path(tempfile.mkdtemp())
        try:
            all_files: List[str] = []
            total_papers = 0
            total_embeddings = 0
            total_clustering_cache = 0
            year_tags: List[str] = []

            for yr in years:
                yr_data = self._export_year(conference, yr, temp_dir, _progress)
                yr_files = [str(yr_data["paper_db_path"]), str(yr_data["embeddings_path"])]
                yr_files.append(str(yr_data["clustering_cache_path"]))
                all_files.extend(yr_files)
                total_papers += yr_data["paper_count"]
                total_embeddings += yr_data["embedding_count"]
                total_clustering_cache += yr_data["clustering_cache_count"]

                # When uploading multiple years, push each year as its own tag first
                if year is None:
                    yr_tag = _build_tag(conference, yr, embedding_model=embedding_model)
                    year_tags.append(yr_tag)

                    # Write per-year config
                    yr_config_data = {
                        "version": __version__,
                        "conference": conference,
                        "years": [yr],
                        "paper_count": yr_data["paper_count"],
                        "embedding_count": yr_data["embedding_count"],
                        "clustering_cache_count": yr_data["clustering_cache_count"],
                        "embedding_model": embedding_model,
                    }
                    yr_config_path = temp_dir / f"config-{yr}.json"
                    yr_config_path.write_text(json.dumps(yr_config_data, indent=2))

                    _progress(f"Uploading {yr_tag}...")
                    self._push_tag(
                        conference=conference,
                        years_in_tag=[yr],
                        files=yr_files,
                        config_path=yr_config_path,
                        tag=yr_tag,
                        embedding_model=embedding_model,
                        paper_count=yr_data["paper_count"],
                        embedding_count=yr_data["embedding_count"],
                        progress=_progress,
                    )

            # --- Build all-years (or single-year) config metadata ---
            config_data = {
                "version": __version__,
                "conference": conference,
                "years": years,
                "paper_count": total_papers,
                "embedding_count": total_embeddings,
                "clustering_cache_count": total_clustering_cache,
                "embedding_model": embedding_model,
            }
            config_path = temp_dir / "config.json"
            config_path.write_text(json.dumps(config_data, indent=2))

            # --- Push the final (single-year or all-years summary) tag ---
            _progress(f"Uploading {tag}...")
            self._push_tag(
                conference=conference,
                years_in_tag=years,
                files=all_files,
                config_path=config_path,
                tag=tag,
                embedding_model=embedding_model,
                paper_count=total_papers,
                embedding_count=total_embeddings,
                progress=_progress,
            )

            summary: Dict[str, Any] = {
                "tag": tag,
                "conference": conference,
                "years": years,
                "paper_count": total_papers,
                "embedding_count": total_embeddings,
                "clustering_cache_count": total_clustering_cache,
            }
            if year_tags:
                summary["year_tags"] = year_tags
            return summary

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def download(
        self,
        conference: str,
        year: Optional[int] = None,
        tag: Optional[str] = None,
        embedding_model: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
        ignore_embedding_model_mismatch: bool = False,
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
            and *tag* is also ``None``, the model is read from the
            ``EMBEDDING_MODEL`` configuration.
            A ``RegistryError`` is raised if the model cannot be determined.
        progress_callback : callable, optional
            Function called with status messages during download.
        ignore_embedding_model_mismatch : bool, optional
            When ``True``, proceed with the download even if the artifact's embedding model
            differs from the configured model.  After a successful import the local embedding
            model metadata is updated to match *embedding_model*.  Only use this option when
            the mismatch is caused by the same model having different names on different
            backends (e.g. LM Studio vs. Ollama).  Default is ``False``.

        Returns
        -------
        dict
            Download summary with paper count and embedding count.

        Raises
        ------
        EmbeddingModelMismatchError
            If the artifact's embedding model differs from *embedding_model* and
            *ignore_embedding_model_mismatch* is ``False``.
        RegistryError
            If download fails or the embedding model cannot be determined.
        """
        if embedding_model is None:
            embedding_model = get_config().embedding_model
        if not embedding_model:
            raise RegistryError(
                "No embedding model specified and none found in the configuration. "
                "Use --embedding-model to specify the model name."
            )
        if tag is None:
            tag = _build_tag(conference, year, embedding_model=embedding_model)

        def _progress(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)
            logger.info(msg)

        target = f"{self.repository}:{tag}"

        # --- 0a. Resolve tag: find the best matching tag in the registry ---
        # The locally-built tag includes the current package version, which may
        # differ from the version used when the artifact was pushed.  Resolve to
        # the closest matching tag so that both the manifest check and the pull
        # use a tag that actually exists.
        resolved_tag = self._find_best_matching_tag(tag)
        if resolved_tag != tag:
            _progress(f"Tag '{tag}' not found in registry; using closest match '{resolved_tag}'")
            tag = resolved_tag
            target = f"{self.repository}:{tag}"

        # --- 0b. Pre-download: check embedding model from manifest labels ---
        # Fail fast before pulling data when the artifact's model does not
        # match and the user has not opted to ignore the mismatch.
        manifest_embedding_model = self._get_manifest_embedding_model(target)
        if manifest_embedding_model and embedding_model:
            if _sanitize_str_for_oci_tag(manifest_embedding_model) != _sanitize_str_for_oci_tag(embedding_model):
                if not ignore_embedding_model_mismatch:
                    raise EmbeddingModelMismatchError(
                        local_model=embedding_model,
                        remote_model=manifest_embedding_model,
                    )

        temp_dir = Path(tempfile.mkdtemp())
        try:
            # --- 1. Pull from oras ---
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
            # Files are named papers-YYYY.db, embeddings-YYYY.json, and clustering-YYYY.json
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
                elif name.startswith("clustering-") and name.endswith(".json"):
                    try:
                        yr = int(name[len("clustering-") : -len(".json")])
                    except ValueError:
                        logger.warning(f"Skipping file with invalid year format: {name}")
                        continue
                    year_files.setdefault(yr, {})["clustering_cache"] = p

            # If user requested a specific year, filter
            if year is not None:
                year_files = {yr: files for yr, files in year_files.items() if yr == year}

            # --- 3. Validate completeness ---
            for yr in sorted(year_files.keys()):
                files = year_files[yr]
                missing = []
                if not files.get("paper_db"):
                    missing.append("paper DB")
                if not files.get("embeddings"):
                    missing.append("embeddings")
                if not files.get("clustering_cache"):
                    missing.append("clustering cache")
                if missing:
                    raise RegistryError(
                        f"Incomplete data for {conference}/{yr}: missing {', '.join(missing)}. "
                        "Cannot import — paper DB, embeddings, and clustering cache must all be present."
                    )

            # --- 4. Import each year ---
            total_papers = 0
            total_embeddings = 0
            total_clustering_cache = 0
            imported_years: List[int] = []

            for yr in sorted(year_files.keys()):
                files = year_files[yr]
                paper_db = files["paper_db"]
                embeddings = files["embeddings"]
                clustering_cache = files["clustering_cache"]

                result = self._import_year(
                    conference,
                    yr,
                    paper_db,
                    embeddings,
                    _progress,
                    embedding_model=embedding_model,
                    ignore_embedding_model_mismatch=ignore_embedding_model_mismatch,
                    clustering_cache_file=clustering_cache,
                )
                total_papers += result["paper_count"]
                total_embeddings += result["embedding_count"]
                total_clustering_cache += result["clustering_cache_count"]
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
                "clustering_cache_count": total_clustering_cache,
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
        from abstracts_explorer.database import DatabaseManager

        with DatabaseManager() as db:
            db.create_tables()
            filters = db.get_conferences()
            conferences = sorted(filters)
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
        ignore_embedding_model_mismatch: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Download data for **all** conference tags in the registry.

        Lists available tags and downloads every conference-level tag
        (i.e. tags without a year suffix).

        Parameters
        ----------
        progress_callback : callable, optional
            Function called with status messages during download.

        ignore_embedding_model_mismatch : bool, optional
            If True, ignore embedding model mismatches during download.

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

        # Only download conference-level tags (e.g. "chi_model_0.4.1").
        # Year-specific tags (e.g. "chi-2026_model_0.4.1") are subsets of the
        # conference tag and would cause each year to be imported twice.
        conference_tags = [t for t in tags if self._is_conference_level_tag(t)]
        if not conference_tags:
            raise RegistryError("No conference-level tags found in registry.")

        def _progress(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)
            logger.info(msg)

        _progress(
            f"Found {len(conference_tags)} conference tag(s) in registry (skipping {len(tags) - len(conference_tags)} year-specific tag(s))"
        )

        summaries: List[Dict[str, Any]] = []
        for tag in sorted(conference_tags):
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
                ignore_embedding_model_mismatch=ignore_embedding_model_mismatch,
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

    # ------------------------------------------------------------------
    # GitHub Packages API helpers (deletion)
    # ------------------------------------------------------------------

    def _github_api_headers(self) -> Dict[str, str]:
        """Return HTTP headers for authenticated GitHub API requests."""
        headers: Dict[str, str] = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _list_github_package_versions(self) -> List[Dict[str, Any]]:
        """
        List all versions of the GHCR package via the GitHub Packages API.

        Returns
        -------
        list of dict
            Each entry contains at minimum ``id``, ``name`` (digest), and
            ``metadata.container.tags`` (list of OCI tags for that version).

        Raises
        ------
        RegistryError
            If the API call fails or the registry is not hosted on ``ghcr.io``.
        """
        if not self.registry.lower() == "ghcr.io":
            raise RegistryError(
                f"Deletion via the GitHub Packages API is only supported for 'ghcr.io' registries, "
                f"not '{self.registry}'."
            )

        # self.name is "{owner}/{package_name}" (everything after "ghcr.io/")
        name_parts = self.name.split("/", 1)
        if len(name_parts) != 2 or not name_parts[0] or not name_parts[1]:
            raise RegistryError(
                f"Cannot determine owner and package name from repository '{self.repository}'. "
                "Expected format: 'ghcr.io/{{owner}}/{{package-name}}'."
            )
        owner, package_name = name_parts
        # URL-encode the package name (slashes → %2F)
        package_name_encoded = package_name.replace("/", "%2F")

        headers = self._github_api_headers()
        all_versions: List[Dict[str, Any]] = []
        page = 1
        while True:
            url = (
                f"https://api.github.com/users/{owner}/packages/container"
                f"/{package_name_encoded}/versions?per_page=100&page={page}"
            )
            try:
                response = requests.get(url, headers=headers, timeout=30)
            except requests.RequestException as e:
                raise RegistryError(f"GitHub API request failed: {e}") from e

            if response.status_code == 401:
                raise RegistryError("GitHub API authentication failed. Check that your token is valid.")
            if response.status_code == 403:
                raise RegistryError("GitHub API access forbidden. Ensure your token has the 'delete:packages' scope.")
            if response.status_code == 404:
                # Try org-level endpoint as fallback
                url_org = (
                    f"https://api.github.com/orgs/{owner}/packages/container"
                    f"/{package_name_encoded}/versions?per_page=100&page={page}"
                )
                try:
                    response = requests.get(url_org, headers=headers, timeout=30)
                except requests.RequestException as e:
                    raise RegistryError(f"GitHub API request failed: {e}") from e
                if response.status_code != 200:
                    raise RegistryError(
                        f"GitHub API returned HTTP {response.status_code} for package '{package_name}' "
                        f"under owner/org '{owner}'. Verify the repository path and token permissions."
                    )

            if response.status_code != 200:
                raise RegistryError(f"GitHub API returned HTTP {response.status_code}: {response.text[:200]}")

            page_data = response.json()
            if not page_data:
                break
            all_versions.extend(page_data)
            if len(page_data) < 100:
                break
            page += 1

        return all_versions

    def _delete_github_package_version(self, owner: str, package_name: str, version_id: int) -> None:
        """
        Delete a single package version via the GitHub Packages API.

        Parameters
        ----------
        owner : str
            GitHub username or organisation name.
        package_name : str
            Package name (e.g. ``abstracts-data``).
        version_id : int
            Numeric version ID returned by the list-versions endpoint.

        Raises
        ------
        RegistryError
            If the deletion API call fails.
        """
        package_name_encoded = package_name.replace("/", "%2F")
        headers = self._github_api_headers()

        # Try user endpoint first, fall back to org endpoint on 404.
        for endpoint in ("users", "orgs"):
            url = (
                f"https://api.github.com/{endpoint}/{owner}/packages/container"
                f"/{package_name_encoded}/versions/{version_id}"
            )
            try:
                response = requests.delete(url, headers=headers, timeout=30)
            except requests.RequestException as e:
                raise RegistryError(f"GitHub API request failed: {e}") from e

            if response.status_code == 404 and endpoint == "users":
                # Package might be org-owned — try org endpoint
                continue
            if response.status_code in (204, 200):
                return
            raise RegistryError(
                f"Failed to delete package version {version_id}: "
                f"HTTP {response.status_code} — {response.text[:200]}"
            )

    def delete_old_versions(
        self,
        below_version: str,
        conference: Optional[str] = None,
        dry_run: bool = False,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Delete registry package versions whose tag version is older than *below_version*.

        Uses the `GitHub Packages API
        <https://docs.github.com/en/rest/packages/packages>`_ to list and
        delete container image versions.  Only versions that carry at least
        one OCI tag matching the abstracts-explorer tag format
        (``{conference}[-{year}]_{model}_{version}``) are considered;
        untagged (dangling) versions are left untouched.

        Parameters
        ----------
        below_version : str
            Threshold version string (PEP 440). Versions **strictly older**
            than this value are deleted.  Example: ``"0.4.0"`` deletes all
            versions tagged with a version < 0.4.0.
        conference : str, optional
            When provided, only tags whose base component starts with
            *conference* (case-insensitive) are examined.  Tags for other
            conferences are ignored.
        dry_run : bool, optional
            When ``True``, log which versions *would* be deleted but perform
            no actual deletions (default: ``False``).
        progress_callback : callable, optional
            Function called with status messages during the operation.

        Returns
        -------
        list of dict
            One entry per deleted (or, in dry-run mode, would-be-deleted)
            version.  Each dict contains ``version_id``, ``tags``, and
            ``version``.

        Raises
        ------
        RegistryError
            If the registry is not hosted on ``ghcr.io``, if the GitHub API
            call fails, or if *below_version* cannot be parsed.
        ValueError
            If *below_version* is not a valid PEP 440 version string.
        """
        try:
            threshold = Version(below_version)
        except InvalidVersion as e:
            raise ValueError(f"Invalid version string '{below_version}': {e}") from e

        def _progress(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)
            logger.info(msg)

        _progress(f"Fetching package versions from GitHub Packages API ({self.repository}) …")
        all_versions = self._list_github_package_versions()
        _progress(f"Found {len(all_versions)} package version(s) in registry.")

        # Determine owner / package name for deletion calls
        name_parts = self.name.split("/", 1)
        owner, package_name = name_parts[0], name_parts[1]

        deleted: List[Dict[str, Any]] = []

        for pkg_version in all_versions:
            version_id: int = pkg_version["id"]
            tags: List[str] = pkg_version.get("metadata", {}).get("container", {}).get("tags", [])

            if not tags:
                # Skip untagged (dangling) versions
                continue

            # A package version may carry multiple tags; check whether *any*
            # of them belongs to the requested conference and is old enough.
            should_delete = False
            matched_tags: List[Tuple[str, Version]] = []
            for tag in tags:
                # Optional conference filter
                if conference is not None:
                    base = tag.split("_", 1)[0]
                    conf_prefix = _sanitize_str_for_oci_tag(conference)
                    # Accept both exact match (e.g. "neurips") and year-specific
                    # (e.g. "neurips-2024").
                    if not (base == conf_prefix or base.startswith(conf_prefix + "-")):
                        continue

                tag_version = _parse_version_from_tag(tag)
                if tag_version is not None and tag_version < threshold:
                    matched_tags.append((tag, tag_version))
                    should_delete = True

            if not should_delete:
                continue

            tag_summary = ", ".join(f"{t} (v{v})" for t, v in matched_tags)
            if dry_run:
                _progress(f"  [dry-run] Would delete version {version_id}: {tag_summary}")
            else:
                _progress(f"  Deleting version {version_id}: {tag_summary} …")
                self._delete_github_package_version(owner, package_name, version_id)
                _progress(f"  ✓ Deleted version {version_id}.")

            deleted.append(
                {
                    "version_id": version_id,
                    "tags": tags,
                    "version": str(matched_tags[0][1]) if matched_tags else None,
                }
            )

        action = "would be deleted" if dry_run else "deleted"
        _progress(f"\nDone. {len(deleted)} version(s) {action}.")
        return deleted

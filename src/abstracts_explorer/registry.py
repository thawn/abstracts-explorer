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
from typing import Any, Callable, Dict, List, Optional

import oras.client
import oras.defaults
import oras.oci
import oras.provider

from abstracts_explorer._version import __version__
from abstracts_explorer.config import get_config

logger = logging.getLogger(__name__)

# Custom media types for abstracts-explorer artifacts
PAPER_DB_MEDIA_TYPE = "application/vnd.abstracts-explorer.paper-db.v1.tar+gzip"
EMBEDDING_DB_MEDIA_TYPE = "application/vnd.abstracts-explorer.embedding-db.v1.tar+gzip"
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
    def _get_all_conferences() -> List[str]:
        """
        Return distinct conferences available in the local database.

        Returns
        -------
        list of str
            Sorted list of conference names.
        """
        from abstracts_explorer.database import DatabaseManager

        with DatabaseManager() as db:
            db.create_tables()
            filters = db.get_filter_options()
            return sorted(filters.get("conferences", []))

    @staticmethod
    def _resolve_conference_name(conference: str) -> str:
        """
        Resolve *conference* to the actual name as stored in the local database.

        Performs a case-insensitive comparison against all conferences currently
        in the local database.  Returns the stored name if a match is found, or
        the input string unchanged if no match exists (e.g. when uploading for
        the first time or using an explicit tag).

        Parameters
        ----------
        conference : str
            Conference name supplied by the caller (may differ in case from the
            stored name, e.g. ``neurips`` vs. ``NeurIPS``).

        Returns
        -------
        str
            The conference name as it appears in the local database, or the
            input string if no case-insensitive match is found.
        """
        all_conferences = RegistryClient._get_all_conferences()
        for conf in all_conferences:
            if conf.lower() == conference.lower():
                return conf
        return conference

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
        Export paper DB and embeddings for a single conference+year.

        Returns a dict with ``paper_db_path``, ``embeddings_path``,
        ``paper_count``, and ``embedding_count``.
        """
        from abstracts_explorer.database import DatabaseManager
        from abstracts_explorer.embeddings import EmbeddingsManager

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
        embedding_model: Optional[str] = None,
        ignore_embedding_model_mismatch: bool = False,
    ) -> Dict[str, Any]:
        """
        Import paper DB and embeddings for a single conference+year.

        Both *paper_db_file* and *embeddings_file* must exist.  If either
        import fails, any already-imported data for this conference+year is
        rolled back to prevent inconsistency between the paper DB and the
        embedding DB.

        Returns a dict with ``paper_count``, ``embedding_count``, and
        ``mismatch_was_ignored`` (True if an embedding model mismatch was
        detected but overridden via *ignore_embedding_model_mismatch*).

        Raises
        ------
        EmbeddingModelMismatchError
            If the artifact's embedding model differs from *embedding_model*
            and *ignore_embedding_model_mismatch* is ``False``.
        RegistryError
            If either file is missing or an import step fails.
        """
        from abstracts_explorer.database import DatabaseManager
        from abstracts_explorer.embeddings import EmbeddingsManager

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

        # --- pre-flight: validate embedding model from artifact paper DB ---
        # This check catches mismatches even when the local DB is empty (in which case
        # the normal import_papers_from_sqlite() check is skipped).
        mismatch_was_ignored = False
        if embedding_model:
            artifact_model: Optional[str] = None
            try:
                with sqlite3.connect(str(paper_db_file)) as conn:
                    row = conn.execute(
                        "SELECT embedding_model FROM embeddings_metadata ORDER BY updated_at DESC LIMIT 1"
                    ).fetchone()
                    if row:
                        artifact_model = row[0]
            except (sqlite3.OperationalError, sqlite3.DatabaseError) as exc:
                # Legacy DB without embeddings_metadata table or not a valid SQLite file
                logger.debug("Could not read embedding model from artifact DB %s: %s", paper_db_file.name, exc)

            if artifact_model and _sanitize_str_for_oci_tag(artifact_model) != _sanitize_str_for_oci_tag(
                embedding_model
            ):
                if not ignore_embedding_model_mismatch:
                    raise EmbeddingModelMismatchError(local_model=embedding_model, remote_model=artifact_model)
                mismatch_was_ignored = True
                logger.debug(
                    f"⚠️  WARNING: Embedding model mismatch detected for {conference}/{year}!\n"
                    f"  Configured model: '{embedding_model}'\n"
                    f"  Artifact model:   '{artifact_model}'\n"
                    f"  Proceeding because --ignore-embedding-model-mismatch was set.\n"
                    f"  ⚠️  Only do this if both names refer to the same model on different backends!"
                )

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

        return {
            "paper_count": paper_count,
            "embedding_count": embedding_count,
            "mismatch_was_ignored": mismatch_was_ignored,
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

        # Resolve the conference name to the actual stored name (case-insensitive)
        conference = self._resolve_conference_name(conference)

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
            year_tags: List[str] = []

            for yr in years:
                yr_data = self._export_year(conference, yr, temp_dir, _progress)
                yr_files = [str(yr_data["paper_db_path"]), str(yr_data["embeddings_path"])]
                all_files.extend(yr_files)
                total_papers += yr_data["paper_count"]
                total_embeddings += yr_data["embedding_count"]

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
        # Fetch the manifest before pulling any data so we can fail fast if
        # the artifact was built with a different embedding model.
        mismatch_was_ignored = False
        manifest_embedding_model = self._get_manifest_embedding_model(target)
        if manifest_embedding_model and embedding_model:
            if _sanitize_str_for_oci_tag(manifest_embedding_model) != _sanitize_str_for_oci_tag(embedding_model):
                if not ignore_embedding_model_mismatch:
                    raise EmbeddingModelMismatchError(
                        local_model=embedding_model,
                        remote_model=manifest_embedding_model,
                    )
                mismatch_was_ignored = True
                _progress(
                    f"⚠️  WARNING: Embedding model mismatch detected!\n"
                    f"  Configured model: '{embedding_model}'\n"
                    f"  Artifact model:   '{manifest_embedding_model}'\n"
                    f"  Proceeding because --ignore-embedding-model-mismatch was set.\n"
                    f"  ⚠️  Only do this if both names refer to the same model on different backends!"
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

                result = self._import_year(
                    conference,
                    yr,
                    paper_db,
                    embeddings,
                    _progress,
                    embedding_model=embedding_model,
                    ignore_embedding_model_mismatch=ignore_embedding_model_mismatch,
                )
                total_papers += result["paper_count"]
                total_embeddings += result["embedding_count"]
                if result.get("mismatch_was_ignored"):
                    mismatch_was_ignored = True
                imported_years.append(yr)

            if not imported_years:
                _progress("Warning: No data found in artifact to import")

            # --- 5. Update local embedding model if mismatch was ignored ---
            if mismatch_was_ignored and embedding_model:
                from abstracts_explorer.database import DatabaseManager

                _progress(f"Updating local embedding model metadata to configured model: '{embedding_model}'")
                with DatabaseManager() as db:
                    db.create_tables()
                    db.set_embedding_model(embedding_model)

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

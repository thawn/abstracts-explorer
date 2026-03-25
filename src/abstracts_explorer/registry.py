"""
Registry module for uploading and downloading data to/from OCI-compatible container registries.

This module provides functionality to push and pull abstracts-explorer data artifacts
(paper databases, embedding databases, clustering caches) to OCI-compatible registries
such as GitHub Container Registry (ghcr.io).

The artifacts are stored as OCI images with custom media types, following the
OCI Distribution Specification.

Examples
--------
Upload data to GitHub Container Registry::

    from abstracts_explorer.registry import RegistryClient

    client = RegistryClient(
        repository="ghcr.io/owner/abstracts-data",
        token="ghp_xxxx",
    )
    client.upload(tag="neurips-2024")

Download data from the registry::

    client.download(tag="neurips-2024", merge=True)

List available tags::

    tags = client.list_tags()
"""

import hashlib
import io
import json
import logging
import os
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# OCI media types
MANIFEST_MEDIA_TYPE = "application/vnd.oci.image.manifest.v1+json"
CONFIG_MEDIA_TYPE = "application/vnd.abstracts-explorer.config.v1+json"
PAPER_DB_MEDIA_TYPE = "application/vnd.abstracts-explorer.paper-db.v1.tar+gzip"
EMBEDDING_DB_MEDIA_TYPE = "application/vnd.abstracts-explorer.embedding-db.v1.tar+gzip"

# Upload chunk size (5 MB)
CHUNK_SIZE = 5 * 1024 * 1024


class RegistryError(Exception):
    """Exception raised for registry operation errors."""

    pass


def _compute_sha256(data: bytes) -> str:
    """
    Compute SHA-256 hex digest of data.

    Parameters
    ----------
    data : bytes
        Data to hash.

    Returns
    -------
    str
        Hex digest string.
    """
    return hashlib.sha256(data).hexdigest()


def _make_descriptor(data: bytes, media_type: str, annotations: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Create an OCI content descriptor for a blob.

    Parameters
    ----------
    data : bytes
        Blob data.
    media_type : str
        IANA media type for the blob.
    annotations : dict, optional
        Additional annotations for the descriptor.

    Returns
    -------
    dict
        OCI descriptor dictionary.
    """
    digest = f"sha256:{_compute_sha256(data)}"
    descriptor: Dict[str, Any] = {
        "mediaType": media_type,
        "digest": digest,
        "size": len(data),
    }
    if annotations:
        descriptor["annotations"] = annotations
    return descriptor


def package_file_as_tar_gz(file_path: Path) -> bytes:
    """
    Package a single file as a tar.gz archive.

    Parameters
    ----------
    file_path : Path
        Path to the file to package.

    Returns
    -------
    bytes
        Compressed tar archive data.

    Raises
    ------
    RegistryError
        If the file does not exist or cannot be read.
    """
    if not file_path.exists():
        raise RegistryError(f"File not found: {file_path}")

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(str(file_path), arcname=file_path.name)
    return buf.getvalue()


def package_directory_as_tar_gz(dir_path: Path) -> bytes:
    """
    Package a directory as a tar.gz archive.

    Parameters
    ----------
    dir_path : Path
        Path to the directory to package.

    Returns
    -------
    bytes
        Compressed tar archive data.

    Raises
    ------
    RegistryError
        If the directory does not exist.
    """
    if not dir_path.exists():
        raise RegistryError(f"Directory not found: {dir_path}")

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(str(dir_path), arcname=dir_path.name)
    return buf.getvalue()


def extract_tar_gz(data: bytes, output_dir: Path) -> Path:
    """
    Extract a tar.gz archive to a directory.

    Parameters
    ----------
    data : bytes
        Compressed tar archive data.
    output_dir : Path
        Directory to extract to.

    Returns
    -------
    Path
        Path to the extracted content (first entry in archive).

    Raises
    ------
    RegistryError
        If extraction fails or archive contains unsafe paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO(data)
    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        # Security: check for path traversal
        for member in tar.getmembers():
            member_path = (output_dir / member.name).resolve()
            if not str(member_path).startswith(str(output_dir.resolve())):
                raise RegistryError(f"Unsafe path in archive: {member.name}")

        tar.extractall(output_dir, filter="data")

        # Return path to first entry
        members = tar.getmembers()
        if members:
            return output_dir / members[0].name
    return output_dir


def export_papers_to_sqlite(
    source_db_url: str,
    output_path: Path,
    conferences: Optional[List[str]] = None,
) -> int:
    """
    Export papers from the database to a standalone SQLite file.

    Parameters
    ----------
    source_db_url : str
        SQLAlchemy database URL for the source database.
    output_path : Path
        Path for the output SQLite file.
    conferences : list of str, optional
        If provided, only export papers from these conferences.

    Returns
    -------
    int
        Number of papers exported.
    """
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import Session

    from .db_models import Base, ClusteringCache, EmbeddingsMetadata, HierarchicalLabelCache, Paper

    # Create export database
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_engine = create_engine(f"sqlite:///{output_path}")
    Base.metadata.create_all(export_engine)

    # Connect to source
    connect_args = {}
    if source_db_url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
    source_engine = create_engine(source_db_url, connect_args=connect_args)

    paper_count = 0
    with Session(source_engine) as source_session, Session(export_engine) as export_session:
        # Export papers
        query = select(Paper)
        if conferences:
            query = query.where(Paper.conference.in_(conferences))

        for paper in source_session.execute(query).scalars():
            paper_dict = {c.name: getattr(paper, c.name) for c in Paper.__table__.columns}
            export_session.add(Paper(**paper_dict))
            paper_count += 1

        # Export clustering cache (always full)
        for entry in source_session.execute(select(ClusteringCache)).scalars():
            entry_dict = {c.name: getattr(entry, c.name) for c in ClusteringCache.__table__.columns}
            export_session.add(ClusteringCache(**entry_dict))

        # Export hierarchical labels
        for entry in source_session.execute(select(HierarchicalLabelCache)).scalars():
            entry_dict = {c.name: getattr(entry, c.name) for c in HierarchicalLabelCache.__table__.columns}
            export_session.add(HierarchicalLabelCache(**entry_dict))

        # Export embeddings metadata
        for entry in source_session.execute(select(EmbeddingsMetadata)).scalars():
            entry_dict = {c.name: getattr(entry, c.name) for c in EmbeddingsMetadata.__table__.columns}
            export_session.add(EmbeddingsMetadata(**entry_dict))

        export_session.commit()

    source_engine.dispose()
    export_engine.dispose()

    return paper_count


def import_papers_from_sqlite(
    sqlite_path: Path,
    target_db_url: str,
    merge: bool = False,
) -> int:
    """
    Import papers from a SQLite file into the target database.

    Parameters
    ----------
    sqlite_path : Path
        Path to the SQLite file to import from.
    target_db_url : str
        SQLAlchemy database URL for the target database.
    merge : bool
        If True, merge with existing data (skip duplicates).
        If False, clear target tables before importing.

    Returns
    -------
    int
        Number of papers imported.
    """
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import Session

    from .db_models import Base, ClusteringCache, EmbeddingsMetadata, HierarchicalLabelCache, Paper

    source_engine = create_engine(f"sqlite:///{sqlite_path}", connect_args={"check_same_thread": False})

    connect_args = {}
    if target_db_url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
    target_engine = create_engine(target_db_url, connect_args=connect_args)
    Base.metadata.create_all(target_engine)

    paper_count = 0
    with Session(source_engine) as source_session, Session(target_engine) as target_session:
        if not merge:
            # Clear target tables
            target_session.query(Paper).delete()
            target_session.query(ClusteringCache).delete()
            target_session.query(HierarchicalLabelCache).delete()
            target_session.query(EmbeddingsMetadata).delete()
            target_session.commit()

        # Import papers
        for paper in source_session.execute(select(Paper)).scalars():
            paper_dict = {c.name: getattr(paper, c.name) for c in Paper.__table__.columns}
            uid = paper_dict.get("uid")

            if merge and uid:
                existing = target_session.execute(select(Paper).where(Paper.uid == uid)).scalar_one_or_none()
                if existing:
                    continue

            target_session.add(Paper(**paper_dict))
            paper_count += 1

        # Import clustering cache
        for entry in source_session.execute(select(ClusteringCache)).scalars():
            entry_dict = {c.name: getattr(entry, c.name) for c in ClusteringCache.__table__.columns}
            if not merge:
                target_session.add(ClusteringCache(**entry_dict))
            else:
                # For merge, just add (may create duplicates, but cache lookup picks most recent)
                entry_dict.pop("id", None)
                target_session.add(ClusteringCache(**entry_dict))

        # Import hierarchical labels
        for entry in source_session.execute(select(HierarchicalLabelCache)).scalars():
            entry_dict = {c.name: getattr(entry, c.name) for c in HierarchicalLabelCache.__table__.columns}
            if not merge:
                target_session.add(HierarchicalLabelCache(**entry_dict))
            else:
                entry_dict.pop("id", None)
                target_session.add(HierarchicalLabelCache(**entry_dict))

        # Import embeddings metadata
        for entry in source_session.execute(select(EmbeddingsMetadata)).scalars():
            entry_dict = {c.name: getattr(entry, c.name) for c in EmbeddingsMetadata.__table__.columns}
            if not merge:
                target_session.add(EmbeddingsMetadata(**entry_dict))
            else:
                entry_dict.pop("id", None)
                target_session.add(EmbeddingsMetadata(**entry_dict))

        target_session.commit()

    source_engine.dispose()
    target_engine.dispose()

    return paper_count


def export_embeddings_to_json(
    embeddings_manager: Any,
    conferences: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Export embeddings from ChromaDB to a JSON-serializable dictionary.

    Parameters
    ----------
    embeddings_manager : EmbeddingsManager
        Initialized embeddings manager with a connected collection.
    conferences : list of str, optional
        If provided, only export embeddings for papers from these conferences.

    Returns
    -------
    dict
        Dictionary containing ids, documents, metadatas, and embeddings.
    """
    collection = embeddings_manager.collection

    kwargs: Dict[str, Any] = {"include": ["documents", "embeddings", "metadatas"]}
    if conferences:
        kwargs["where"] = {"conference": {"$in": conferences}}

    results = collection.get(**kwargs)

    return {
        "ids": results.get("ids", []),
        "documents": results.get("documents", []),
        "metadatas": results.get("metadatas", []),
        "embeddings": results.get("embeddings", []),
    }


def import_embeddings_from_json(
    embeddings_manager: Any,
    data: Dict[str, Any],
    merge: bool = False,
    batch_size: int = 100,
) -> int:
    """
    Import embeddings from a JSON dictionary into ChromaDB.

    Parameters
    ----------
    embeddings_manager : EmbeddingsManager
        Initialized embeddings manager.
    data : dict
        Dictionary containing ids, documents, metadatas, and embeddings.
    merge : bool
        If True, skip existing embeddings. If False, reset collection first.
    batch_size : int
        Number of embeddings to add per batch.

    Returns
    -------
    int
        Number of embeddings imported.
    """
    if not merge:
        embeddings_manager.create_collection(reset=True)

    collection = embeddings_manager.collection
    ids = data.get("ids", [])
    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])
    embeddings = data.get("embeddings", [])

    if not ids:
        return 0

    imported = 0
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i : i + batch_size]
        batch_docs = documents[i : i + batch_size] if documents else None
        batch_metas = metadatas[i : i + batch_size] if metadatas else None
        batch_embeds = embeddings[i : i + batch_size] if embeddings else None

        if merge:
            # Filter out existing IDs
            new_indices = []
            for j, doc_id in enumerate(batch_ids):
                try:
                    existing = collection.get(ids=[doc_id])
                    if not existing["ids"]:
                        new_indices.append(j)
                except Exception:
                    new_indices.append(j)

            if not new_indices:
                continue

            batch_ids = [batch_ids[j] for j in new_indices]
            if batch_docs:
                batch_docs = [batch_docs[j] for j in new_indices]
            if batch_metas:
                batch_metas = [batch_metas[j] for j in new_indices]
            if batch_embeds:
                batch_embeds = [batch_embeds[j] for j in new_indices]

        add_kwargs: Dict[str, Any] = {"ids": batch_ids}
        if batch_docs:
            add_kwargs["documents"] = batch_docs
        if batch_metas:
            add_kwargs["metadatas"] = batch_metas
        if batch_embeds:
            add_kwargs["embeddings"] = batch_embeds

        collection.add(**add_kwargs)
        imported += len(batch_ids)

    return imported


class RegistryClient:
    """
    Client for pushing and pulling data artifacts to/from OCI-compatible container registries.

    This client implements the OCI Distribution Specification for pushing and pulling
    artifacts to container registries like GitHub Container Registry (ghcr.io).

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
    ['latest', 'neurips-2024']
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
        self.token = token or os.environ.get("GITHUB_TOKEN", "")
        self._bearer_tokens: Dict[str, str] = {}
        self.base_url = f"https://{self.registry}"
        self.api_url = f"{self.base_url}/v2/{self.name}"

    def _get_bearer_token(self, scope: str) -> str:
        """
        Exchange credentials for a short-lived bearer token.

        Parameters
        ----------
        scope : str
            Authorization scope (e.g., ``repository:owner/name:pull,push``).

        Returns
        -------
        str
            Bearer token for authenticated requests.

        Raises
        ------
        RegistryError
            If authentication fails.
        """
        if scope in self._bearer_tokens:
            return self._bearer_tokens[scope]

        token_url = f"{self.base_url}/token"
        params = {
            "service": self.registry,
            "scope": scope,
        }

        auth = None
        if self.token:
            auth = ("_token", self.token)

        try:
            resp = requests.get(token_url, params=params, auth=auth, timeout=30)
            resp.raise_for_status()
            bearer_token = resp.json().get("token", "")
            self._bearer_tokens[scope] = bearer_token
            return bearer_token
        except requests.RequestException as e:
            raise RegistryError(f"Authentication failed: {e}") from e

    def _auth_headers(self, push: bool = False) -> Dict[str, str]:
        """
        Get authorization headers for registry requests.

        Parameters
        ----------
        push : bool
            Whether write access is needed.

        Returns
        -------
        dict
            Headers dictionary with Authorization header.
        """
        actions = "pull,push" if push else "pull"
        scope = f"repository:{self.name}:{actions}"
        token = self._get_bearer_token(scope)
        return {"Authorization": f"Bearer {token}"}

    def check_blob_exists(self, digest: str) -> bool:
        """
        Check if a blob exists in the registry.

        Parameters
        ----------
        digest : str
            Blob digest (e.g., ``sha256:abc123...``).

        Returns
        -------
        bool
            True if blob exists.
        """
        url = f"{self.api_url}/blobs/{digest}"
        try:
            resp = requests.head(url, headers=self._auth_headers(), timeout=30, allow_redirects=True)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def push_blob(self, data: bytes) -> str:
        """
        Upload a blob to the registry.

        Parameters
        ----------
        data : bytes
            Blob data to upload.

        Returns
        -------
        str
            Digest of the uploaded blob (``sha256:...``).

        Raises
        ------
        RegistryError
            If upload fails.
        """
        digest = f"sha256:{_compute_sha256(data)}"

        # Check if blob already exists
        if self.check_blob_exists(digest):
            logger.info(f"Blob {digest[:27]}... already exists, skipping upload")
            return digest

        headers = self._auth_headers(push=True)

        # Start upload session
        upload_url = f"{self.api_url}/blobs/uploads/"
        try:
            resp = requests.post(upload_url, headers=headers, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            raise RegistryError(f"Failed to start blob upload: {e}") from e

        # Get upload URL from Location header
        location = resp.headers.get("Location", "")
        if not location:
            raise RegistryError("Registry did not return upload location")

        # Make location absolute if relative
        if location.startswith("/"):
            location = f"{self.base_url}{location}"

        # Complete upload with PUT (monolithic)
        separator = "&" if "?" in location else "?"
        put_url = f"{location}{separator}digest={digest}"

        try:
            resp = requests.put(
                put_url,
                headers={**headers, "Content-Type": "application/octet-stream"},
                data=data,
                timeout=600,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            raise RegistryError(f"Failed to upload blob: {e}") from e

        logger.info(f"Uploaded blob {digest[:27]}... ({len(data)} bytes)")
        return digest

    def pull_blob(self, digest: str) -> bytes:
        """
        Download a blob from the registry.

        Parameters
        ----------
        digest : str
            Blob digest to download.

        Returns
        -------
        bytes
            Blob data.

        Raises
        ------
        RegistryError
            If download fails.
        """
        url = f"{self.api_url}/blobs/{digest}"
        try:
            resp = requests.get(url, headers=self._auth_headers(), timeout=600, allow_redirects=True)
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as e:
            raise RegistryError(f"Failed to download blob {digest}: {e}") from e

    def push_manifest(self, manifest: Dict[str, Any], tag: str) -> str:
        """
        Upload a manifest to the registry.

        Parameters
        ----------
        manifest : dict
            OCI manifest dictionary.
        tag : str
            Tag for the manifest (e.g., ``latest``).

        Returns
        -------
        str
            Digest of the uploaded manifest.

        Raises
        ------
        RegistryError
            If upload fails.
        """
        manifest_json = json.dumps(manifest, indent=2).encode("utf-8")
        url = f"{self.api_url}/manifests/{tag}"
        headers = {
            **self._auth_headers(push=True),
            "Content-Type": MANIFEST_MEDIA_TYPE,
        }

        try:
            resp = requests.put(url, headers=headers, data=manifest_json, timeout=60)
            resp.raise_for_status()
        except requests.RequestException as e:
            raise RegistryError(f"Failed to push manifest: {e}") from e

        digest = f"sha256:{_compute_sha256(manifest_json)}"
        logger.info(f"Pushed manifest with tag '{tag}'")
        return digest

    def pull_manifest(self, tag: str) -> Dict[str, Any]:
        """
        Download a manifest from the registry.

        Parameters
        ----------
        tag : str
            Tag or digest to fetch.

        Returns
        -------
        dict
            OCI manifest dictionary.

        Raises
        ------
        RegistryError
            If download fails or manifest not found.
        """
        url = f"{self.api_url}/manifests/{tag}"
        headers = {
            **self._auth_headers(),
            "Accept": MANIFEST_MEDIA_TYPE,
        }

        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            raise RegistryError(f"Failed to pull manifest for tag '{tag}': {e}") from e

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
        url = f"{self.api_url}/tags/list"
        try:
            resp = requests.get(url, headers=self._auth_headers(), timeout=30)
            resp.raise_for_status()
            return resp.json().get("tags", [])
        except requests.RequestException as e:
            raise RegistryError(f"Failed to list tags: {e}") from e

    def upload(
        self,
        tag: str = "latest",
        paper_db: bool = True,
        embedding_db: bool = True,
        conferences: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Upload data artifacts to the registry.

        Packages the paper database and/or embedding database as OCI artifacts
        and pushes them to the registry.

        Parameters
        ----------
        tag : str
            Tag for the upload (default: ``latest``).
        paper_db : bool
            Whether to upload the paper database.
        embedding_db : bool
            Whether to upload the embedding database.
        conferences : list of str, optional
            If provided, only upload data for these conferences.
        progress_callback : callable, optional
            Function called with status messages during upload.

        Returns
        -------
        dict
            Upload summary with artifact details.

        Raises
        ------
        RegistryError
            If upload fails.
        """
        from ._version import __version__

        config = _get_config()
        layers = []
        summary: Dict[str, Any] = {
            "tag": tag,
            "version": __version__,
            "conferences": conferences,
            "layers": [],
        }

        def _progress(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)
            logger.info(msg)

        temp_dir = Path(tempfile.mkdtemp())
        try:
            # Package paper database
            if paper_db:
                _progress("Exporting paper database...")
                export_path = temp_dir / "papers.db"
                paper_count = export_papers_to_sqlite(
                    config.database_url,
                    export_path,
                    conferences=conferences,
                )
                _progress(f"Exported {paper_count} papers")

                _progress("Packaging paper database...")
                paper_data = package_file_as_tar_gz(export_path)
                _progress(f"Paper database size: {len(paper_data) / 1024 / 1024:.1f} MB")

                _progress("Uploading paper database...")
                self.push_blob(paper_data)

                layer = _make_descriptor(
                    paper_data,
                    PAPER_DB_MEDIA_TYPE,
                    annotations={
                        "org.opencontainers.image.title": "papers.db.tar.gz",
                        "com.abstracts-explorer.paper-count": str(paper_count),
                    },
                )
                layers.append(layer)
                summary["layers"].append({"type": "paper-db", "papers": paper_count, "size": len(paper_data)})

            # Package embedding database
            if embedding_db:
                _progress("Exporting embeddings...")
                em = _create_embeddings_manager()

                embeddings_data = export_embeddings_to_json(em, conferences=conferences)
                embedding_count = len(embeddings_data.get("ids", []))
                _progress(f"Exported {embedding_count} embeddings")

                embeddings_json = json.dumps(embeddings_data).encode("utf-8")
                embeddings_archive = _create_json_tar_gz(embeddings_json, "embeddings.json")
                _progress(f"Embeddings archive size: {len(embeddings_archive) / 1024 / 1024:.1f} MB")

                _progress("Uploading embeddings...")
                self.push_blob(embeddings_archive)

                layer = _make_descriptor(
                    embeddings_archive,
                    EMBEDDING_DB_MEDIA_TYPE,
                    annotations={
                        "org.opencontainers.image.title": "embeddings.json.tar.gz",
                        "com.abstracts-explorer.embedding-count": str(embedding_count),
                    },
                )
                layers.append(layer)
                summary["layers"].append(
                    {"type": "embedding-db", "embeddings": embedding_count, "size": len(embeddings_archive)}
                )

            # Create config blob with metadata
            config_data = _build_config_metadata(
                conferences=conferences,
                paper_db=paper_db,
                embedding_db=embedding_db,
            )
            config_json = json.dumps(config_data, indent=2).encode("utf-8")
            self.push_blob(config_json)
            config_descriptor = _make_descriptor(config_json, CONFIG_MEDIA_TYPE)

            # Create and push manifest
            manifest = {
                "schemaVersion": 2,
                "mediaType": MANIFEST_MEDIA_TYPE,
                "config": config_descriptor,
                "layers": layers,
                "annotations": {
                    "org.opencontainers.image.created": datetime.now(timezone.utc).isoformat(),
                    "org.opencontainers.image.source": "https://github.com/thawn/abstracts-explorer",
                    "com.abstracts-explorer.version": config_data["version"],
                },
            }

            _progress(f"Pushing manifest with tag '{tag}'...")
            self.push_manifest(manifest, tag)
            _progress("Upload complete!")

            summary["manifest"] = manifest
            return summary

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def download(
        self,
        tag: str = "latest",
        paper_db: bool = True,
        embedding_db: bool = True,
        merge: bool = False,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Download data artifacts from the registry.

        Pulls the paper database and/or embedding database from the registry
        and imports them into the local instance.

        Parameters
        ----------
        tag : str
            Tag to download (default: ``latest``).
        paper_db : bool
            Whether to download the paper database.
        embedding_db : bool
            Whether to download the embedding database.
        merge : bool
            If True, merge with existing data (skip duplicates).
            If False, replace existing data.
        progress_callback : callable, optional
            Function called with status messages during download.

        Returns
        -------
        dict
            Download summary with import details.

        Raises
        ------
        RegistryError
            If download fails.
        """
        config = _get_config()
        summary: Dict[str, Any] = {
            "tag": tag,
            "merge": merge,
            "layers": [],
        }

        def _progress(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)
            logger.info(msg)

        _progress(f"Pulling manifest for tag '{tag}'...")
        manifest = self.pull_manifest(tag)

        # Read config metadata
        config_descriptor = manifest.get("config", {})
        if config_descriptor.get("digest"):
            config_data = json.loads(self.pull_blob(config_descriptor["digest"]))
            summary["metadata"] = config_data
            _progress(f"Artifact version: {config_data.get('version', 'unknown')}")
        else:
            config_data = {}

        layers = manifest.get("layers", [])

        temp_dir = Path(tempfile.mkdtemp())
        try:
            for layer in layers:
                media_type = layer.get("mediaType", "")
                digest = layer.get("digest", "")

                if media_type == PAPER_DB_MEDIA_TYPE and paper_db:
                    _progress("Downloading paper database...")
                    blob_data = self.pull_blob(digest)
                    _progress(f"Downloaded {len(blob_data) / 1024 / 1024:.1f} MB")

                    # Extract and import
                    _progress("Importing paper database...")
                    extracted = extract_tar_gz(blob_data, temp_dir / "paper_db")

                    # Find the .db file
                    db_file = _find_file_in_extracted(extracted, ".db")
                    if db_file:
                        paper_count = import_papers_from_sqlite(db_file, config.database_url, merge=merge)
                        _progress(f"Imported {paper_count} papers")
                        summary["layers"].append({"type": "paper-db", "papers": paper_count})
                    else:
                        _progress("Warning: No database file found in paper-db archive")

                elif media_type == EMBEDDING_DB_MEDIA_TYPE and embedding_db:
                    _progress("Downloading embeddings...")
                    blob_data = self.pull_blob(digest)
                    _progress(f"Downloaded {len(blob_data) / 1024 / 1024:.1f} MB")

                    # Extract and import
                    _progress("Importing embeddings...")
                    extracted = extract_tar_gz(blob_data, temp_dir / "embedding_db")

                    # Find the JSON file
                    json_file = _find_file_in_extracted(extracted, ".json")
                    if json_file:
                        with open(json_file) as f:
                            embeddings_data = json.load(f)
                        em = _create_embeddings_manager()
                        embedding_count = import_embeddings_from_json(em, embeddings_data, merge=merge)
                        _progress(f"Imported {embedding_count} embeddings")
                        summary["layers"].append({"type": "embedding-db", "embeddings": embedding_count})
                    else:
                        _progress("Warning: No JSON file found in embedding-db archive")

            _progress("Download complete!")
            return summary

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
            Artifact metadata including version, conferences, and layer info.

        Raises
        ------
        RegistryError
            If the tag is not found or cannot be read.
        """
        manifest = self.pull_manifest(tag)
        info: Dict[str, Any] = {
            "tag": tag,
            "annotations": manifest.get("annotations", {}),
            "layers": [],
        }

        # Read config blob
        config_descriptor = manifest.get("config", {})
        if config_descriptor.get("digest"):
            try:
                config_data = json.loads(self.pull_blob(config_descriptor["digest"]))
                info["metadata"] = config_data
            except RegistryError:
                pass

        for layer in manifest.get("layers", []):
            layer_info = {
                "media_type": layer.get("mediaType", ""),
                "size": layer.get("size", 0),
                "annotations": layer.get("annotations", {}),
            }
            info["layers"].append(layer_info)

        return info


def _get_config() -> Any:
    """Get the current configuration."""
    from .config import get_config

    return get_config()


def _create_embeddings_manager() -> Any:
    """Create an EmbeddingsManager from current config."""
    from .embeddings import EmbeddingsManager

    return EmbeddingsManager()


def _create_json_tar_gz(json_data: bytes, filename: str) -> bytes:
    """
    Create a tar.gz archive containing a single JSON file.

    Parameters
    ----------
    json_data : bytes
        JSON data to archive.
    filename : str
        Name for the file inside the archive.

    Returns
    -------
    bytes
        Compressed tar archive data.
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name=filename)
        info.size = len(json_data)
        tar.addfile(info, io.BytesIO(json_data))
    return buf.getvalue()


def _find_file_in_extracted(path: Path, extension: str) -> Optional[Path]:
    """
    Find a file with the given extension in an extracted archive path.

    Parameters
    ----------
    path : Path
        Root path to search in.
    extension : str
        File extension to look for (e.g., ``.db``).

    Returns
    -------
    Path or None
        Path to the found file, or None.
    """
    if path.is_file() and path.name.endswith(extension):
        return path

    if path.is_dir():
        for child in path.rglob(f"*{extension}"):
            return child

    return None


def _build_config_metadata(
    conferences: Optional[List[str]] = None,
    paper_db: bool = True,
    embedding_db: bool = True,
) -> Dict[str, Any]:
    """
    Build the config blob metadata for an OCI artifact.

    Parameters
    ----------
    conferences : list of str, optional
        Conferences included in the artifact.
    paper_db : bool
        Whether the paper database is included.
    embedding_db : bool
        Whether the embedding database is included.

    Returns
    -------
    dict
        Configuration metadata dictionary.
    """
    from ._version import __version__

    config = _get_config()

    metadata: Dict[str, Any] = {
        "version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "includes": {
            "paper_db": paper_db,
            "embedding_db": embedding_db,
        },
    }

    if conferences:
        metadata["conferences"] = conferences

    if embedding_db:
        metadata["embedding_model"] = config.embedding_model

    return metadata

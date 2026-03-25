"""
Tests for the registry module.

Tests OCI registry client, data packaging/unpackaging, database export/import,
and CLI command integration.
"""

import argparse
import io
import json
import tarfile
from unittest.mock import MagicMock, Mock, patch

import pytest

from abstracts_explorer.config import get_config
from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.plugin import LightweightPaper
from abstracts_explorer.registry import (
    CONFIG_MEDIA_TYPE,
    MANIFEST_MEDIA_TYPE,
    PAPER_DB_MEDIA_TYPE,
    RegistryClient,
    RegistryError,
    _build_config_metadata,
    _compute_sha256,
    _create_json_tar_gz,
    _find_file_in_extracted,
    _make_descriptor,
    export_embeddings_to_json,
    export_papers_to_sqlite,
    extract_tar_gz,
    import_embeddings_from_json,
    import_papers_from_sqlite,
    package_directory_as_tar_gz,
    package_file_as_tar_gz,
)
from tests.conftest import get_env_test_path, set_test_db

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _make_sample_papers():
    """Create sample LightweightPaper objects for testing."""
    return [
        LightweightPaper(
            title="Paper One",
            authors=["Author A"],
            abstract="Abstract for paper one",
            session="Session 1",
            poster_position="A1",
            year=2024,
            conference="neurips",
        ),
        LightweightPaper(
            title="Paper Two",
            authors=["Author B", "Author C"],
            abstract="Abstract for paper two",
            session="Session 2",
            poster_position="B2",
            year=2024,
            conference="iclr",
        ),
        LightweightPaper(
            title="Paper Three",
            authors=["Author D"],
            abstract="Abstract for paper three",
            session="Session 3",
            poster_position="C3",
            year=2025,
            conference="neurips",
        ),
    ]


def _populate_test_db(db_path):
    """Create and populate a test database, return the database URL."""
    set_test_db(db_path)
    db = DatabaseManager()
    db.connect()
    db.create_tables()
    db.add_papers(_make_sample_papers())
    db.close()
    return f"sqlite:///{db_path}"


# ---------------------------------------------------------------------------
# Tests: Utility functions
# ---------------------------------------------------------------------------


class TestUtilityFunctions:
    """Tests for low-level utility functions."""

    def test_compute_sha256(self):
        """SHA-256 digest is correctly computed."""
        data = b"hello world"
        digest = _compute_sha256(data)
        assert len(digest) == 64
        assert digest == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"

    def test_make_descriptor(self):
        """OCI descriptor has correct structure."""
        data = b"test data"
        desc = _make_descriptor(data, "application/octet-stream")
        assert desc["mediaType"] == "application/octet-stream"
        assert desc["digest"].startswith("sha256:")
        assert desc["size"] == len(data)

    def test_make_descriptor_with_annotations(self):
        """OCI descriptor includes annotations when provided."""
        data = b"test data"
        annotations = {"key": "value"}
        desc = _make_descriptor(data, "application/octet-stream", annotations=annotations)
        assert desc["annotations"] == {"key": "value"}


# ---------------------------------------------------------------------------
# Tests: Packaging functions
# ---------------------------------------------------------------------------


class TestPackaging:
    """Tests for tar.gz packaging and extraction."""

    def test_package_file_as_tar_gz(self, tmp_path):
        """A file is correctly packaged as tar.gz."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        data = package_file_as_tar_gz(test_file)
        assert len(data) > 0

        # Verify it's a valid tar.gz
        buf = io.BytesIO(data)
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            names = tar.getnames()
            assert "test.txt" in names

    def test_package_file_not_found(self, tmp_path):
        """RegistryError raised for non-existent file."""
        with pytest.raises(RegistryError, match="File not found"):
            package_file_as_tar_gz(tmp_path / "nonexistent.txt")

    def test_package_directory_as_tar_gz(self, tmp_path):
        """A directory is correctly packaged as tar.gz."""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        (test_dir / "file1.txt").write_text("content1")
        (test_dir / "file2.txt").write_text("content2")

        data = package_directory_as_tar_gz(test_dir)
        assert len(data) > 0

        # Verify contents
        buf = io.BytesIO(data)
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            names = tar.getnames()
            assert any("file1.txt" in n for n in names)
            assert any("file2.txt" in n for n in names)

    def test_package_directory_not_found(self, tmp_path):
        """RegistryError raised for non-existent directory."""
        with pytest.raises(RegistryError, match="Directory not found"):
            package_directory_as_tar_gz(tmp_path / "nonexistent_dir")

    def test_extract_tar_gz(self, tmp_path):
        """tar.gz archive is correctly extracted."""
        # Create a tar.gz
        test_file = tmp_path / "input" / "test.txt"
        test_file.parent.mkdir()
        test_file.write_text("hello world")
        data = package_file_as_tar_gz(test_file)

        # Extract it
        output_dir = tmp_path / "output"
        extract_tar_gz(data, output_dir)

        assert output_dir.exists()
        assert (output_dir / "test.txt").exists()
        assert (output_dir / "test.txt").read_text() == "hello world"

    def test_extract_tar_gz_path_traversal(self, tmp_path):
        """RegistryError raised for archives with path traversal."""
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            info = tarfile.TarInfo(name="../../../etc/passwd")
            info.size = 5
            tar.addfile(info, io.BytesIO(b"evil!"))
        data = buf.getvalue()

        with pytest.raises(RegistryError, match="Unsafe path"):
            extract_tar_gz(data, tmp_path / "output")

    def test_create_json_tar_gz(self):
        """JSON data is correctly packaged as tar.gz."""
        json_data = json.dumps({"key": "value"}).encode("utf-8")
        archive = _create_json_tar_gz(json_data, "test.json")

        buf = io.BytesIO(archive)
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            names = tar.getnames()
            assert "test.json" in names

            member = tar.getmember("test.json")
            f = tar.extractfile(member)
            assert f is not None
            content = json.loads(f.read())
            assert content == {"key": "value"}

    def test_find_file_in_extracted_file(self, tmp_path):
        """Find a file by extension when path is a file."""
        test_file = tmp_path / "test.db"
        test_file.write_text("data")
        assert _find_file_in_extracted(test_file, ".db") == test_file

    def test_find_file_in_extracted_directory(self, tmp_path):
        """Find a file by extension when path is a directory."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        test_file = subdir / "data.json"
        test_file.write_text("{}")
        assert _find_file_in_extracted(tmp_path, ".json") == test_file

    def test_find_file_in_extracted_not_found(self, tmp_path):
        """Return None when no file with extension is found."""
        assert _find_file_in_extracted(tmp_path, ".xyz") is None


# ---------------------------------------------------------------------------
# Tests: Database export/import
# ---------------------------------------------------------------------------


class TestDatabaseExportImport:
    """Tests for paper database export and import."""

    def test_export_papers_to_sqlite(self, tmp_path):
        """Papers are exported to a standalone SQLite file."""
        db_url = _populate_test_db(tmp_path / "source.db")
        export_path = tmp_path / "export.db"

        count = export_papers_to_sqlite(db_url, export_path)
        assert count == 3
        assert export_path.exists()

    def test_export_papers_with_conference_filter(self, tmp_path):
        """Only papers from specified conferences are exported."""
        db_url = _populate_test_db(tmp_path / "source.db")
        export_path = tmp_path / "export.db"

        count = export_papers_to_sqlite(db_url, export_path, conferences=["neurips"])
        assert count == 2  # Only neurips papers

    def test_import_papers_replace(self, tmp_path):
        """Papers are imported, replacing existing data."""
        # Create source database
        source_url = _populate_test_db(tmp_path / "source.db")
        export_path = tmp_path / "export.db"
        export_papers_to_sqlite(source_url, export_path)

        # Create target database with some data
        target_path = tmp_path / "target.db"
        target_url = _populate_test_db(target_path)

        # Import (replace mode)
        count = import_papers_from_sqlite(export_path, target_url, merge=False)
        assert count == 3

    def test_import_papers_merge(self, tmp_path):
        """Papers are merged, skipping duplicates."""
        # Create source database
        source_url = _populate_test_db(tmp_path / "source.db")
        export_path = tmp_path / "export.db"
        export_papers_to_sqlite(source_url, export_path)

        # Create target database with same data
        target_path = tmp_path / "target.db"
        target_url = _populate_test_db(target_path)

        # Import (merge mode) - should skip all duplicates
        count = import_papers_from_sqlite(export_path, target_url, merge=True)
        assert count == 0  # All papers already exist

    def test_import_papers_merge_new(self, tmp_path):
        """New papers are added during merge."""
        # Create source with neurips papers only
        source_url = _populate_test_db(tmp_path / "source.db")
        export_path = tmp_path / "export_neurips.db"
        export_papers_to_sqlite(source_url, export_path, conferences=["neurips"])

        # Create target with iclr papers only
        iclr_export = tmp_path / "export_iclr.db"
        export_papers_to_sqlite(source_url, iclr_export, conferences=["iclr"])

        target_path = tmp_path / "target.db"
        target_url = f"sqlite:///{target_path}"
        import_papers_from_sqlite(iclr_export, target_url, merge=False)

        # Merge neurips papers into target (which only has iclr)
        count = import_papers_from_sqlite(export_path, target_url, merge=True)
        assert count == 2  # 2 neurips papers added


# ---------------------------------------------------------------------------
# Tests: Embeddings export/import
# ---------------------------------------------------------------------------


class TestEmbeddingsExportImport:
    """Tests for embeddings export and import."""

    def test_export_embeddings_to_json(self):
        """Embeddings are exported to a JSON-serializable dict."""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": ["id1", "id2"],
            "documents": ["doc1", "doc2"],
            "metadatas": [{"key": "val1"}, {"key": "val2"}],
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
        }

        mock_em = MagicMock()
        mock_em.collection = mock_collection

        result = export_embeddings_to_json(mock_em)
        assert result["ids"] == ["id1", "id2"]
        assert result["documents"] == ["doc1", "doc2"]
        assert result["embeddings"] == [[0.1, 0.2], [0.3, 0.4]]

        # Verify no conference filter was used
        mock_collection.get.assert_called_once_with(include=["documents", "embeddings", "metadatas"])

    def test_export_embeddings_with_conference_filter(self):
        """Conference filter is passed to ChromaDB query."""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": ["id1"],
            "documents": ["doc1"],
            "metadatas": [{"conference": "neurips"}],
            "embeddings": [[0.1, 0.2]],
        }

        mock_em = MagicMock()
        mock_em.collection = mock_collection

        export_embeddings_to_json(mock_em, conferences=["neurips"])

        mock_collection.get.assert_called_once_with(
            include=["documents", "embeddings", "metadatas"],
            where={"conference": {"$in": ["neurips"]}},
        )

    def test_import_embeddings_replace(self):
        """Embeddings are imported, replacing existing collection."""
        mock_collection = MagicMock()
        mock_em = MagicMock()
        mock_em.collection = mock_collection

        data = {
            "ids": ["id1", "id2"],
            "documents": ["doc1", "doc2"],
            "metadatas": [{"key": "val1"}, {"key": "val2"}],
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
        }

        count = import_embeddings_from_json(mock_em, data, merge=False)
        assert count == 2
        mock_em.create_collection.assert_called_once_with(reset=True)
        mock_collection.add.assert_called_once()

    def test_import_embeddings_merge(self):
        """New embeddings are added during merge, existing ones skipped."""
        mock_collection = MagicMock()
        # id1 exists, id2 does not
        mock_collection.get.side_effect = [
            {"ids": ["id1"]},  # id1 exists
            {"ids": []},  # id2 doesn't exist
        ]
        mock_em = MagicMock()
        mock_em.collection = mock_collection

        data = {
            "ids": ["id1", "id2"],
            "documents": ["doc1", "doc2"],
            "metadatas": [{"key": "val1"}, {"key": "val2"}],
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
        }

        count = import_embeddings_from_json(mock_em, data, merge=True, batch_size=100)
        assert count == 1
        mock_em.create_collection.assert_not_called()

    def test_import_embeddings_empty(self):
        """Empty import returns 0."""
        mock_em = MagicMock()
        data = {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
        count = import_embeddings_from_json(mock_em, data, merge=False)
        assert count == 0


# ---------------------------------------------------------------------------
# Tests: Config metadata
# ---------------------------------------------------------------------------


class TestConfigMetadata:
    """Tests for config metadata generation."""

    def test_build_config_metadata(self):
        """Config metadata includes version and timestamps."""
        with patch("abstracts_explorer._version.__version__", "1.2.3"):
            metadata = _build_config_metadata()

        assert metadata["version"] == "1.2.3"
        assert "created_at" in metadata
        assert metadata["includes"]["paper_db"] is True
        assert metadata["includes"]["embedding_db"] is True

    def test_build_config_metadata_with_conferences(self):
        """Config metadata includes conference list when specified."""
        with patch("abstracts_explorer._version.__version__", "1.0.0"):
            metadata = _build_config_metadata(conferences=["neurips", "iclr"])

        assert metadata["conferences"] == ["neurips", "iclr"]

    def test_build_config_metadata_partial(self):
        """Config metadata reflects which databases are included."""
        with patch("abstracts_explorer._version.__version__", "1.0.0"):
            metadata = _build_config_metadata(paper_db=True, embedding_db=False)

        assert metadata["includes"]["paper_db"] is True
        assert metadata["includes"]["embedding_db"] is False
        assert "embedding_model" not in metadata


# ---------------------------------------------------------------------------
# Tests: RegistryClient
# ---------------------------------------------------------------------------


class TestRegistryClient:
    """Tests for RegistryClient OCI operations."""

    def test_init_valid_repository(self):
        """Client initializes correctly with valid repository."""
        client = RegistryClient("ghcr.io/owner/abstracts-data", token="test-token")
        assert client.registry == "ghcr.io"
        assert client.name == "owner/abstracts-data"
        assert client.token == "test-token"
        assert client.base_url == "https://ghcr.io"
        assert client.api_url == "https://ghcr.io/v2/owner/abstracts-data"

    def test_init_invalid_repository(self):
        """RegistryError raised for invalid repository format."""
        with pytest.raises(RegistryError, match="Invalid repository format"):
            RegistryClient("invalid")

    def test_init_empty_parts(self):
        """RegistryError raised for repository with empty parts."""
        with pytest.raises(RegistryError, match="Invalid repository format"):
            RegistryClient("/name")

    def test_init_token_from_env(self, monkeypatch):
        """Token is read from GITHUB_TOKEN environment variable."""
        monkeypatch.setenv("GITHUB_TOKEN", "env-token")
        client = RegistryClient("ghcr.io/owner/repo")
        assert client.token == "env-token"

    def test_get_bearer_token(self):
        """Bearer token is obtained from registry auth endpoint."""
        client = RegistryClient("ghcr.io/owner/repo", token="pat-token")

        mock_response = Mock()
        mock_response.json.return_value = {"token": "bearer-token-123"}
        mock_response.raise_for_status = Mock()

        with patch("abstracts_explorer.registry.requests.get", return_value=mock_response) as mock_get:
            token = client._get_bearer_token("repository:owner/repo:pull")

        assert token == "bearer-token-123"
        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args
        assert call_kwargs[1]["auth"] == ("_token", "pat-token")

    def test_get_bearer_token_cached(self):
        """Bearer tokens are cached by scope."""
        client = RegistryClient("ghcr.io/owner/repo", token="pat-token")
        client._bearer_tokens["repository:owner/repo:pull"] = "cached-token"

        token = client._get_bearer_token("repository:owner/repo:pull")
        assert token == "cached-token"

    def test_get_bearer_token_failure(self):
        """RegistryError raised when auth fails."""
        client = RegistryClient("ghcr.io/owner/repo", token="bad-token")

        import requests as req

        with patch(
            "abstracts_explorer.registry.requests.get",
            side_effect=req.exceptions.ConnectionError("connection error"),
        ):
            with pytest.raises(RegistryError, match="Authentication failed"):
                client._get_bearer_token("repository:owner/repo:pull")

    def test_check_blob_exists_true(self):
        """check_blob_exists returns True for existing blob."""
        client = RegistryClient("ghcr.io/owner/repo", token="token")

        mock_response = Mock()
        mock_response.status_code = 200

        with patch.object(client, "_auth_headers", return_value={"Authorization": "Bearer t"}):
            with patch("abstracts_explorer.registry.requests.head", return_value=mock_response):
                assert client.check_blob_exists("sha256:abc123") is True

    def test_check_blob_exists_false(self):
        """check_blob_exists returns False for non-existing blob."""
        client = RegistryClient("ghcr.io/owner/repo", token="token")

        mock_response = Mock()
        mock_response.status_code = 404

        with patch.object(client, "_auth_headers", return_value={"Authorization": "Bearer t"}):
            with patch("abstracts_explorer.registry.requests.head", return_value=mock_response):
                assert client.check_blob_exists("sha256:abc123") is False

    def test_push_blob(self):
        """Blob is uploaded via OCI Distribution API."""
        client = RegistryClient("ghcr.io/owner/repo", token="token")

        post_response = Mock()
        post_response.raise_for_status = Mock()
        post_response.headers = {"Location": "/v2/owner/repo/blobs/uploads/uuid123"}

        put_response = Mock()
        put_response.raise_for_status = Mock()

        with patch.object(client, "_auth_headers", return_value={"Authorization": "Bearer t"}):
            with patch.object(client, "check_blob_exists", return_value=False):
                with patch("abstracts_explorer.registry.requests.post", return_value=post_response):
                    with patch("abstracts_explorer.registry.requests.put", return_value=put_response):
                        digest = client.push_blob(b"test data")

        assert digest.startswith("sha256:")

    def test_push_blob_already_exists(self):
        """Existing blob is skipped during push."""
        client = RegistryClient("ghcr.io/owner/repo", token="token")

        with patch.object(client, "_auth_headers", return_value={"Authorization": "Bearer t"}):
            with patch.object(client, "check_blob_exists", return_value=True):
                digest = client.push_blob(b"test data")

        assert digest.startswith("sha256:")

    def test_push_blob_no_location(self):
        """RegistryError raised when POST doesn't return Location header."""
        client = RegistryClient("ghcr.io/owner/repo", token="token")

        post_response = Mock()
        post_response.raise_for_status = Mock()
        post_response.headers = {}

        with patch.object(client, "_auth_headers", return_value={"Authorization": "Bearer t"}):
            with patch.object(client, "check_blob_exists", return_value=False):
                with patch("abstracts_explorer.registry.requests.post", return_value=post_response):
                    with pytest.raises(RegistryError, match="upload location"):
                        client.push_blob(b"test data")

    def test_pull_blob(self):
        """Blob is downloaded from registry."""
        client = RegistryClient("ghcr.io/owner/repo", token="token")

        mock_response = Mock()
        mock_response.content = b"blob data"
        mock_response.raise_for_status = Mock()

        with patch.object(client, "_auth_headers", return_value={"Authorization": "Bearer t"}):
            with patch("abstracts_explorer.registry.requests.get", return_value=mock_response):
                data = client.pull_blob("sha256:abc123")

        assert data == b"blob data"

    def test_push_manifest(self):
        """Manifest is uploaded to registry."""
        client = RegistryClient("ghcr.io/owner/repo", token="token")

        mock_response = Mock()
        mock_response.raise_for_status = Mock()

        manifest = {"schemaVersion": 2, "mediaType": MANIFEST_MEDIA_TYPE}

        with patch.object(client, "_auth_headers", return_value={"Authorization": "Bearer t"}):
            with patch("abstracts_explorer.registry.requests.put", return_value=mock_response):
                digest = client.push_manifest(manifest, "latest")

        assert digest.startswith("sha256:")

    def test_pull_manifest(self):
        """Manifest is downloaded from registry."""
        client = RegistryClient("ghcr.io/owner/repo", token="token")

        expected_manifest = {
            "schemaVersion": 2,
            "mediaType": MANIFEST_MEDIA_TYPE,
            "layers": [],
        }
        mock_response = Mock()
        mock_response.json.return_value = expected_manifest
        mock_response.raise_for_status = Mock()

        with patch.object(client, "_auth_headers", return_value={"Authorization": "Bearer t"}):
            with patch("abstracts_explorer.registry.requests.get", return_value=mock_response):
                manifest = client.pull_manifest("latest")

        assert manifest == expected_manifest

    def test_list_tags(self):
        """Tags are listed from registry."""
        client = RegistryClient("ghcr.io/owner/repo", token="token")

        mock_response = Mock()
        mock_response.json.return_value = {"tags": ["latest", "v1.0"]}
        mock_response.raise_for_status = Mock()

        with patch.object(client, "_auth_headers", return_value={"Authorization": "Bearer t"}):
            with patch("abstracts_explorer.registry.requests.get", return_value=mock_response):
                tags = client.list_tags()

        assert tags == ["latest", "v1.0"]


# ---------------------------------------------------------------------------
# Tests: Upload/Download orchestration
# ---------------------------------------------------------------------------


class TestUploadDownload:
    """Tests for high-level upload/download orchestration."""

    def test_upload_paper_db_only(self, tmp_path):
        """Upload with paper_db=True, embedding_db=False."""
        _populate_test_db(tmp_path / "test.db")

        client = RegistryClient("ghcr.io/owner/repo", token="token")

        with patch.object(client, "push_blob", return_value="sha256:abc") as mock_push:
            with patch.object(client, "push_manifest", return_value="sha256:manifest") as mock_manifest:
                summary = client.upload(
                    tag="test",
                    paper_db=True,
                    embedding_db=False,
                )

        assert len(summary["layers"]) == 1
        assert summary["layers"][0]["type"] == "paper-db"
        assert summary["layers"][0]["papers"] == 3
        assert mock_push.call_count == 2  # paper db blob + config blob
        mock_manifest.assert_called_once()

    def test_upload_embedding_db_only(self):
        """Upload with paper_db=False, embedding_db=True."""
        client = RegistryClient("ghcr.io/owner/repo", token="token")

        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": ["id1"],
            "documents": ["doc1"],
            "metadatas": [{"key": "val"}],
            "embeddings": [[0.1]],
        }
        mock_em = MagicMock()
        mock_em.collection = mock_collection

        with patch("abstracts_explorer.registry._create_embeddings_manager", return_value=mock_em):
            with patch.object(client, "push_blob", return_value="sha256:abc"):
                with patch.object(client, "push_manifest", return_value="sha256:manifest"):
                    summary = client.upload(
                        tag="test",
                        paper_db=False,
                        embedding_db=True,
                    )

        assert len(summary["layers"]) == 1
        assert summary["layers"][0]["type"] == "embedding-db"
        assert summary["layers"][0]["embeddings"] == 1

    def test_upload_with_conference_filter(self, tmp_path):
        """Upload filters papers by conference."""
        _populate_test_db(tmp_path / "test.db")

        client = RegistryClient("ghcr.io/owner/repo", token="token")

        with patch.object(client, "push_blob", return_value="sha256:abc"):
            with patch.object(client, "push_manifest", return_value="sha256:manifest"):
                summary = client.upload(
                    tag="neurips-2024",
                    paper_db=True,
                    embedding_db=False,
                    conferences=["neurips"],
                )

        assert summary["layers"][0]["papers"] == 2  # Only neurips papers

    def test_download_paper_db(self, tmp_path):
        """Download imports paper database."""
        # Create a source database and package it
        source_url = _populate_test_db(tmp_path / "source.db")
        export_path = tmp_path / "export.db"
        export_papers_to_sqlite(source_url, export_path)
        paper_archive = package_file_as_tar_gz(export_path)

        # Set up target database
        target_path = tmp_path / "target.db"
        set_test_db(target_path)

        # Build config data
        config_json = json.dumps({"version": "1.0.0"}).encode("utf-8")

        # Build manifest
        manifest = {
            "schemaVersion": 2,
            "mediaType": MANIFEST_MEDIA_TYPE,
            "config": _make_descriptor(config_json, CONFIG_MEDIA_TYPE),
            "layers": [
                _make_descriptor(
                    paper_archive,
                    PAPER_DB_MEDIA_TYPE,
                    annotations={"org.opencontainers.image.title": "papers.db.tar.gz"},
                ),
            ],
        }

        client = RegistryClient("ghcr.io/owner/repo", token="token")

        # Mock the network calls
        def mock_pull_blob(digest):
            if digest == manifest["config"]["digest"]:
                return config_json
            return paper_archive

        with patch.object(client, "pull_manifest", return_value=manifest):
            with patch.object(client, "pull_blob", side_effect=mock_pull_blob):
                summary = client.download(
                    tag="latest",
                    paper_db=True,
                    embedding_db=False,
                    merge=False,
                )

        assert len(summary["layers"]) == 1
        assert summary["layers"][0]["type"] == "paper-db"
        assert summary["layers"][0]["papers"] == 3

    def test_download_merge_mode(self, tmp_path):
        """Download with merge=True preserves existing data."""
        # Create source database
        source_url = _populate_test_db(tmp_path / "source.db")
        export_path = tmp_path / "export_neurips.db"
        export_papers_to_sqlite(source_url, export_path, conferences=["neurips"])
        paper_archive = package_file_as_tar_gz(export_path)

        # Set up target database (with iclr papers)
        target_path = tmp_path / "target.db"
        iclr_export = tmp_path / "iclr_export.db"
        export_papers_to_sqlite(source_url, iclr_export, conferences=["iclr"])
        target_url = f"sqlite:///{target_path}"
        import_papers_from_sqlite(iclr_export, target_url, merge=False)
        set_test_db(target_path)

        config_json = json.dumps({"version": "1.0.0"}).encode("utf-8")
        manifest = {
            "schemaVersion": 2,
            "mediaType": MANIFEST_MEDIA_TYPE,
            "config": _make_descriptor(config_json, CONFIG_MEDIA_TYPE),
            "layers": [
                _make_descriptor(paper_archive, PAPER_DB_MEDIA_TYPE),
            ],
        }

        client = RegistryClient("ghcr.io/owner/repo", token="token")

        def mock_pull_blob(digest):
            if digest == manifest["config"]["digest"]:
                return config_json
            return paper_archive

        with patch.object(client, "pull_manifest", return_value=manifest):
            with patch.object(client, "pull_blob", side_effect=mock_pull_blob):
                summary = client.download(
                    tag="latest",
                    paper_db=True,
                    embedding_db=False,
                    merge=True,
                )

        # Should have imported 2 new neurips papers
        assert summary["layers"][0]["papers"] == 2

    def test_upload_progress_callback(self, tmp_path):
        """Progress callback is called during upload."""
        _populate_test_db(tmp_path / "test.db")

        client = RegistryClient("ghcr.io/owner/repo", token="token")
        messages = []

        with patch.object(client, "push_blob", return_value="sha256:abc"):
            with patch.object(client, "push_manifest", return_value="sha256:manifest"):
                client.upload(
                    tag="test",
                    paper_db=True,
                    embedding_db=False,
                    progress_callback=messages.append,
                )

        assert len(messages) > 0
        assert any("Exporting" in m for m in messages)
        assert any("complete" in m.lower() for m in messages)


# ---------------------------------------------------------------------------
# Tests: get_artifact_info
# ---------------------------------------------------------------------------


class TestGetArtifactInfo:
    """Tests for artifact metadata retrieval."""

    def test_get_artifact_info(self):
        """Artifact info includes metadata and layer details."""
        client = RegistryClient("ghcr.io/owner/repo", token="token")

        config_data = {"version": "1.0.0", "conferences": ["neurips"]}
        config_json = json.dumps(config_data).encode("utf-8")

        manifest = {
            "schemaVersion": 2,
            "mediaType": MANIFEST_MEDIA_TYPE,
            "config": _make_descriptor(config_json, CONFIG_MEDIA_TYPE),
            "layers": [
                {
                    "mediaType": PAPER_DB_MEDIA_TYPE,
                    "size": 1024,
                    "annotations": {"com.abstracts-explorer.paper-count": "100"},
                },
            ],
            "annotations": {"com.abstracts-explorer.version": "1.0.0"},
        }

        with patch.object(client, "pull_manifest", return_value=manifest):
            with patch.object(client, "pull_blob", return_value=config_json):
                info = client.get_artifact_info("latest")

        assert info["tag"] == "latest"
        assert info["metadata"]["version"] == "1.0.0"
        assert info["metadata"]["conferences"] == ["neurips"]
        assert len(info["layers"]) == 1
        assert info["layers"][0]["size"] == 1024


# ---------------------------------------------------------------------------
# Tests: CLI commands
# ---------------------------------------------------------------------------


class TestCLICommands:
    """Tests for CLI registry commands."""

    def test_registry_upload_no_repository(self, capsys, monkeypatch):
        """Upload fails when no repository is specified."""
        monkeypatch.delenv("REGISTRY_REPOSITORY", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        get_config(reload=True, env_path=get_env_test_path())

        from abstracts_explorer.cli import registry_upload_command

        args = argparse.Namespace(
            repository=None,
            token=None,
            tag="latest",
            conference=None,
            paper_db=False,
            embedding_db=False,
            all=True,
        )

        result = registry_upload_command(args)
        assert result == 1
        captured = capsys.readouterr()
        assert "Repository not specified" in captured.err

    def test_registry_upload_no_token(self, capsys, monkeypatch):
        """Upload fails when no token is specified."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.setenv("REGISTRY_REPOSITORY", "ghcr.io/owner/repo")
        get_config(reload=True, env_path=get_env_test_path())

        from abstracts_explorer.cli import registry_upload_command

        args = argparse.Namespace(
            repository="ghcr.io/owner/repo",
            token=None,
            tag="latest",
            conference=None,
            paper_db=False,
            embedding_db=False,
            all=True,
        )

        result = registry_upload_command(args)
        assert result == 1
        captured = capsys.readouterr()
        assert "Authentication token not specified" in captured.err

    def test_registry_download_no_repository(self, capsys, monkeypatch):
        """Download fails when no repository is specified."""
        monkeypatch.delenv("REGISTRY_REPOSITORY", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        get_config(reload=True, env_path=get_env_test_path())

        from abstracts_explorer.cli import registry_download_command

        args = argparse.Namespace(
            repository=None,
            token=None,
            tag="latest",
            paper_db=False,
            embedding_db=False,
            all=True,
            merge=False,
            yes=False,
        )

        result = registry_download_command(args)
        assert result == 1
        captured = capsys.readouterr()
        assert "Repository not specified" in captured.err

    def test_registry_list_no_repository(self, capsys, monkeypatch):
        """List fails when no repository is specified."""
        monkeypatch.delenv("REGISTRY_REPOSITORY", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        get_config(reload=True, env_path=get_env_test_path())

        from abstracts_explorer.cli import registry_list_command

        args = argparse.Namespace(
            repository=None,
            token=None,
            tag=None,
        )

        result = registry_list_command(args)
        assert result == 1
        captured = capsys.readouterr()
        assert "Repository not specified" in captured.err

    def test_registry_upload_success(self, tmp_path, capsys, monkeypatch):
        """Upload succeeds with valid arguments."""
        _populate_test_db(tmp_path / "test.db")

        from abstracts_explorer.cli import registry_upload_command

        args = argparse.Namespace(
            repository="ghcr.io/owner/repo",
            token="test-token",
            tag="test",
            conference=None,
            paper_db=True,
            embedding_db=False,
            all=False,
        )

        with patch("abstracts_explorer.registry.RegistryClient") as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.upload.return_value = {
                "tag": "test",
                "layers": [{"type": "paper-db", "papers": 3}],
            }

            result = registry_upload_command(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Upload complete" in captured.out

    def test_registry_list_tags(self, capsys, monkeypatch):
        """List command shows available tags."""
        from abstracts_explorer.cli import registry_list_command

        args = argparse.Namespace(
            repository="ghcr.io/owner/repo",
            token="test-token",
            tag=None,
        )

        with patch("abstracts_explorer.registry.RegistryClient") as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.list_tags.return_value = ["latest", "neurips-2024"]

            result = registry_list_command(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "latest" in captured.out
        assert "neurips-2024" in captured.out

    def test_registry_list_specific_tag(self, capsys, monkeypatch):
        """List command with --tag shows artifact details."""
        from abstracts_explorer.cli import registry_list_command

        args = argparse.Namespace(
            repository="ghcr.io/owner/repo",
            token="test-token",
            tag="latest",
        )

        with patch("abstracts_explorer.registry.RegistryClient") as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.get_artifact_info.return_value = {
                "tag": "latest",
                "metadata": {
                    "version": "1.0.0",
                    "created_at": "2024-01-01T00:00:00Z",
                    "conferences": ["neurips"],
                    "embedding_model": "test-model",
                },
                "layers": [
                    {
                        "media_type": PAPER_DB_MEDIA_TYPE,
                        "size": 1024 * 1024,
                        "annotations": {"com.abstracts-explorer.paper-count": "100"},
                    },
                ],
                "annotations": {},
            }

            result = registry_list_command(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "1.0.0" in captured.out

    def test_registry_list_error(self, capsys, monkeypatch):
        """List command handles registry errors gracefully."""
        from abstracts_explorer.cli import registry_list_command

        args = argparse.Namespace(
            repository="ghcr.io/owner/repo",
            token="test-token",
            tag=None,
        )

        with patch("abstracts_explorer.registry.RegistryClient") as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.list_tags.side_effect = RegistryError("Connection failed")

            result = registry_list_command(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Registry error" in captured.err

    def test_main_dispatch_registry_upload(self):
        """Main dispatches to registry upload command."""
        from abstracts_explorer.cli import main

        with patch("sys.argv", ["abstracts-explorer", "registry", "upload", "-r", "ghcr.io/o/r", "-t", "tok"]):
            with patch("abstracts_explorer.cli.registry_upload_command", return_value=0) as mock_cmd:
                result = main()

        assert result == 0
        mock_cmd.assert_called_once()

    def test_main_dispatch_registry_download(self):
        """Main dispatches to registry download command."""
        from abstracts_explorer.cli import main

        with patch("sys.argv", ["abstracts-explorer", "registry", "download", "-r", "ghcr.io/o/r", "-t", "tok"]):
            with patch("abstracts_explorer.cli.registry_download_command", return_value=0) as mock_cmd:
                result = main()

        assert result == 0
        mock_cmd.assert_called_once()

    def test_main_dispatch_registry_list(self):
        """Main dispatches to registry list command."""
        from abstracts_explorer.cli import main

        with patch("sys.argv", ["abstracts-explorer", "registry", "list", "-r", "ghcr.io/o/r"]):
            with patch("abstracts_explorer.cli.registry_list_command", return_value=0) as mock_cmd:
                result = main()

        assert result == 0
        mock_cmd.assert_called_once()

    def test_main_dispatch_registry_no_subcommand(self, capsys):
        """Registry without subcommand shows help."""
        from abstracts_explorer.cli import main

        with patch("sys.argv", ["abstracts-explorer", "registry"]):
            result = main()

        assert result == 1

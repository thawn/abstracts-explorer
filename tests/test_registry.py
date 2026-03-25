"""
Tests for the registry module.

Tests the oras-based registry client, DatabaseManager export/import methods,
EmbeddingsManager export/import methods, and CLI command integration.
"""

import argparse
import json
from unittest.mock import MagicMock, Mock, patch

import pytest

from abstracts_explorer.config import get_config
from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.plugin import LightweightPaper
from abstracts_explorer.registry import (
    RegistryClient,
    RegistryError,
    _build_tag,
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
    """Create and populate a test database, return the DatabaseManager."""
    set_test_db(db_path)
    db = DatabaseManager()
    db.connect()
    db.create_tables()
    db.add_papers(_make_sample_papers())
    return db


# ---------------------------------------------------------------------------
# Tests: _build_tag
# ---------------------------------------------------------------------------


class TestBuildTag:
    """Tests for the _build_tag helper."""

    def test_simple_tag(self):
        """Tag is built from conference and year."""
        assert _build_tag("neurips", 2024) == "neurips-2024"

    def test_case_normalization(self):
        """Conference name is lowercased."""
        assert _build_tag("NeurIPS", 2024) == "neurips-2024"

    def test_special_characters(self):
        """Special characters are replaced with hyphens."""
        assert _build_tag("ML4PS/workshop", 2025) == "ml4ps-workshop-2025"


# ---------------------------------------------------------------------------
# Tests: RegistryClient initialisation
# ---------------------------------------------------------------------------


class TestRegistryClient:
    """Tests for RegistryClient construction."""

    def test_init_valid_repository(self):
        """Client initializes correctly with valid repository."""
        with patch("oras.client.OrasClient"):
            client = RegistryClient("ghcr.io/owner/abstracts-data", token="test-token")

        assert client.registry == "ghcr.io"
        assert client.name == "owner/abstracts-data"
        assert client.repository == "ghcr.io/owner/abstracts-data"
        assert client.token == "test-token"

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
        with patch("oras.client.OrasClient"):
            client = RegistryClient("ghcr.io/owner/repo")
        assert client.token == "env-token"

    def test_list_tags(self):
        """Tags are listed from registry."""
        with patch("oras.client.OrasClient") as MockOras:
            mock_oras = MockOras.return_value
            mock_oras.get_tags.return_value = ["neurips-2024", "iclr-2025"]
            client = RegistryClient("ghcr.io/owner/repo", token="token")
            tags = client.list_tags()

        assert tags == ["neurips-2024", "iclr-2025"]

    def test_list_tags_error(self):
        """RegistryError raised when listing fails."""
        with patch("oras.client.OrasClient") as MockOras:
            mock_oras = MockOras.return_value
            mock_oras.get_tags.side_effect = Exception("network error")
            client = RegistryClient("ghcr.io/owner/repo", token="token")

            with pytest.raises(RegistryError, match="Failed to list tags"):
                client.list_tags()

    def test_get_artifact_info(self):
        """Artifact info includes annotations and layers."""
        manifest = {
            "annotations": {
                "com.abstracts-explorer.version": "1.0.0",
                "com.abstracts-explorer.conference": "neurips",
                "com.abstracts-explorer.year": "2024",
            },
            "layers": [
                {
                    "mediaType": "application/octet-stream",
                    "size": 1024,
                    "annotations": {"org.opencontainers.image.title": "papers.db"},
                },
            ],
        }

        with patch("oras.client.OrasClient") as MockOras:
            mock_oras = MockOras.return_value
            mock_oras.get_manifest.return_value = manifest
            client = RegistryClient("ghcr.io/owner/repo", token="token")
            info = client.get_artifact_info("neurips-2024")

        assert info["tag"] == "neurips-2024"
        assert info["annotations"]["com.abstracts-explorer.conference"] == "neurips"
        assert len(info["layers"]) == 1


# ---------------------------------------------------------------------------
# Tests: DatabaseManager export/import
# ---------------------------------------------------------------------------


class TestDatabaseExportImport:
    """Tests for DatabaseManager.export_papers_to_sqlite and import_papers_from_sqlite."""

    def test_export_papers_to_sqlite(self, tmp_path):
        """Papers for a specific conference+year are exported."""
        db = _populate_test_db(tmp_path / "source.db")
        try:
            export_path = tmp_path / "export.db"
            count = db.export_papers_to_sqlite(export_path, "neurips", 2024)
            assert count == 1  # Only 1 neurips paper for 2024
            assert export_path.exists()
        finally:
            db.close()

    def test_export_papers_different_year(self, tmp_path):
        """Papers for a different year are exported correctly."""
        db = _populate_test_db(tmp_path / "source.db")
        try:
            export_path = tmp_path / "export.db"
            count = db.export_papers_to_sqlite(export_path, "neurips", 2025)
            assert count == 1  # 1 neurips paper for 2025
        finally:
            db.close()

    def test_export_no_papers_returns_zero(self, tmp_path):
        """Export returns 0 when no papers match."""
        db = _populate_test_db(tmp_path / "source.db")
        try:
            export_path = tmp_path / "export.db"
            count = db.export_papers_to_sqlite(export_path, "icml", 2024)
            assert count == 0
        finally:
            db.close()

    def test_import_replaces_existing(self, tmp_path):
        """Import replaces existing papers for conference+year."""
        db = _populate_test_db(tmp_path / "source.db")
        try:
            export_path = tmp_path / "export.db"
            db.export_papers_to_sqlite(export_path, "neurips", 2024)

            # Import into same database (replaces)
            count = db.import_papers_from_sqlite(export_path, "neurips", 2024)
            assert count == 1

            # Verify the iclr papers still exist (different conference)
            # Note: clustering cache/hierarchical labels are cleared on import
            from abstracts_explorer.db_models import Paper

            total = db._session.query(Paper).count()
            # neurips 2024 (1 replaced) + iclr 2024 (1) + neurips 2025 (1) = 3
            assert total == 3
        finally:
            db.close()

    def test_import_from_export_roundtrip(self, tmp_path):
        """Export + import roundtrip preserves data."""
        # Create source
        db1 = _populate_test_db(tmp_path / "db1.db")
        try:
            export_path = tmp_path / "export.db"
            db1.export_papers_to_sqlite(export_path, "neurips", 2024)
        finally:
            db1.close()

        # Import into fresh database
        target_path = tmp_path / "target.db"
        set_test_db(target_path)
        db2 = DatabaseManager()
        db2.connect()
        db2.create_tables()
        try:
            count = db2.import_papers_from_sqlite(export_path, "neurips", 2024)
            assert count == 1

            from abstracts_explorer.db_models import Paper

            papers = db2._session.query(Paper).all()
            assert len(papers) == 1
            assert papers[0].conference == "neurips"
            assert papers[0].year == 2024
        finally:
            db2.close()

    def test_export_not_connected(self, tmp_path):
        """Export raises error when not connected."""
        set_test_db(tmp_path / "not_connected.db")
        db = DatabaseManager()
        with pytest.raises(Exception, match="Not connected"):
            db.export_papers_to_sqlite(tmp_path / "export.db", "neurips", 2024)

    def test_import_not_connected(self, tmp_path):
        """Import raises error when not connected."""
        set_test_db(tmp_path / "not_connected.db")
        db = DatabaseManager()
        with pytest.raises(Exception, match="Not connected"):
            db.import_papers_from_sqlite(tmp_path / "source.db", "neurips", 2024)


# ---------------------------------------------------------------------------
# Tests: EmbeddingsManager export/import
# ---------------------------------------------------------------------------


class TestEmbeddingsExportImport:
    """Tests for EmbeddingsManager.export_embeddings and import_embeddings."""

    def test_export_embeddings(self):
        """Embeddings for a conference+year are exported correctly."""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": ["id1", "id2"],
            "documents": ["doc1", "doc2"],
            "metadatas": [{"conference": "neurips", "year": "2024"}, {"conference": "neurips", "year": "2024"}],
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
        }

        mock_em = MagicMock()
        mock_em.collection = mock_collection
        mock_em.export_embeddings = MagicMock()

        # Test the actual method by calling it on a real-ish object
        from abstracts_explorer.embeddings import EmbeddingsManager

        em = EmbeddingsManager.__new__(EmbeddingsManager)
        em._collection = mock_collection

        result = em.export_embeddings("neurips", 2024)
        assert result["ids"] == ["id1", "id2"]
        assert result["embeddings"] == [[0.1, 0.2], [0.3, 0.4]]

        # Verify the filter was used
        call_kwargs = mock_collection.get.call_args[1]
        assert call_kwargs["where"] == {"$and": [{"conference": "neurips"}, {"year": "2024"}]}

    def test_import_embeddings(self):
        """Embeddings are imported with replace semantics."""
        mock_collection = MagicMock()
        # Existing embeddings to delete
        mock_collection.get.return_value = {"ids": ["old1"]}

        from abstracts_explorer.embeddings import EmbeddingsManager

        em = EmbeddingsManager.__new__(EmbeddingsManager)
        em._collection = mock_collection

        data = {
            "ids": ["id1", "id2"],
            "documents": ["doc1", "doc2"],
            "metadatas": [{"conference": "neurips", "year": "2024"}, {"conference": "neurips", "year": "2024"}],
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
        }

        count = em.import_embeddings(data, "neurips", 2024)
        assert count == 2

        # Verify old embeddings were deleted
        mock_collection.delete.assert_called_once_with(ids=["old1"])
        # Verify new embeddings were added
        mock_collection.add.assert_called_once()

    def test_import_embeddings_empty(self):
        """Empty import returns 0."""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": []}

        from abstracts_explorer.embeddings import EmbeddingsManager

        em = EmbeddingsManager.__new__(EmbeddingsManager)
        em._collection = mock_collection

        data = {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
        count = em.import_embeddings(data, "neurips", 2024)
        assert count == 0


# ---------------------------------------------------------------------------
# Tests: Upload/Download orchestration
# ---------------------------------------------------------------------------


class TestUploadDownload:
    """Tests for high-level upload/download orchestration."""

    def test_upload_validates_papers(self, tmp_path):
        """Upload fails when no papers exist for conference+year."""
        _populate_test_db(tmp_path / "test.db")

        with patch("oras.client.OrasClient"):
            client = RegistryClient("ghcr.io/owner/repo", token="token")

        with pytest.raises(RegistryError, match="No papers found"):
            client.upload(conference="icml", year=2024)

    def test_upload_validates_embeddings(self, tmp_path):
        """Upload fails when no embeddings exist for conference+year."""
        _populate_test_db(tmp_path / "test.db")

        with patch("oras.client.OrasClient"):
            client = RegistryClient("ghcr.io/owner/repo", token="token")

        mock_em = MagicMock()
        mock_em.export_embeddings.return_value = {"ids": [], "documents": [], "metadatas": [], "embeddings": []}

        with patch("abstracts_explorer.embeddings.EmbeddingsManager", return_value=mock_em):
            with pytest.raises(RegistryError, match="No embeddings found"):
                client.upload(conference="neurips", year=2024)

    def test_upload_success(self, tmp_path):
        """Upload succeeds with valid data."""
        _populate_test_db(tmp_path / "test.db")

        with patch("oras.client.OrasClient") as MockOras:
            mock_oras = MockOras.return_value
            mock_oras.push.return_value = Mock(status_code=201)
            client = RegistryClient("ghcr.io/owner/repo", token="token")

        mock_em = MagicMock()
        mock_em.export_embeddings.return_value = {
            "ids": ["id1"],
            "documents": ["doc1"],
            "metadatas": [{"conference": "neurips", "year": "2024"}],
            "embeddings": [[0.1, 0.2]],
        }

        with patch("abstracts_explorer.embeddings.EmbeddingsManager", return_value=mock_em):
            summary = client.upload(conference="neurips", year=2024)

        assert summary["paper_count"] == 1
        assert summary["embedding_count"] == 1
        assert summary["tag"] == "neurips-2024"
        mock_oras.push.assert_called_once()

    def test_upload_custom_tag(self, tmp_path):
        """Upload uses custom tag when provided."""
        _populate_test_db(tmp_path / "test.db")

        with patch("oras.client.OrasClient") as MockOras:
            mock_oras = MockOras.return_value
            mock_oras.push.return_value = Mock(status_code=201)
            client = RegistryClient("ghcr.io/owner/repo", token="token")

        mock_em = MagicMock()
        mock_em.export_embeddings.return_value = {
            "ids": ["id1"],
            "documents": ["doc1"],
            "metadatas": [{}],
            "embeddings": [[0.1]],
        }

        with patch("abstracts_explorer.embeddings.EmbeddingsManager", return_value=mock_em):
            summary = client.upload(conference="neurips", year=2024, tag="custom-tag")

        assert summary["tag"] == "custom-tag"
        call_kwargs = mock_oras.push.call_args[1]
        assert "custom-tag" in call_kwargs["target"]

    def test_upload_progress_callback(self, tmp_path):
        """Progress callback is invoked during upload."""
        _populate_test_db(tmp_path / "test.db")

        with patch("oras.client.OrasClient") as MockOras:
            mock_oras = MockOras.return_value
            mock_oras.push.return_value = Mock(status_code=201)
            client = RegistryClient("ghcr.io/owner/repo", token="token")

        mock_em = MagicMock()
        mock_em.export_embeddings.return_value = {
            "ids": ["id1"],
            "documents": ["d"],
            "metadatas": [{}],
            "embeddings": [[0.1]],
        }

        messages = []
        with patch("abstracts_explorer.embeddings.EmbeddingsManager", return_value=mock_em):
            client.upload(
                conference="neurips",
                year=2024,
                progress_callback=messages.append,
            )

        assert len(messages) > 0
        assert any("Exporting" in m for m in messages)

    def test_download_success(self, tmp_path):
        """Download imports paper database and embeddings."""
        set_test_db(tmp_path / "target.db")

        # Create a fake paper db to be pulled
        source_db_path = tmp_path / "source.db"
        source_db = _populate_test_db(source_db_path)
        export_path = tmp_path / "papers.db"
        source_db.export_papers_to_sqlite(export_path, "neurips", 2024)
        source_db.close()

        # Create fake embeddings file
        embeddings_data = {
            "ids": ["id1"],
            "documents": ["doc1"],
            "metadatas": [{"conference": "neurips", "year": "2024"}],
            "embeddings": [[0.1, 0.2]],
        }
        embeddings_path = tmp_path / "embeddings.json"
        embeddings_path.write_text(json.dumps(embeddings_data))

        with patch("oras.client.OrasClient") as MockOras:
            mock_oras = MockOras.return_value
            mock_oras.pull.return_value = [str(export_path), str(embeddings_path)]
            client = RegistryClient("ghcr.io/owner/repo", token="token")

        mock_em = MagicMock()
        mock_em.import_embeddings.return_value = 1

        # Reset target DB
        set_test_db(tmp_path / "target.db")

        with patch("abstracts_explorer.embeddings.EmbeddingsManager", return_value=mock_em):
            summary = client.download(conference="neurips", year=2024)

        assert summary["paper_count"] == 1
        assert summary["embedding_count"] == 1
        mock_em.import_embeddings.assert_called_once()


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
            conference="neurips",
            year=2024,
            tag=None,
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
            conference="neurips",
            year=2024,
            tag=None,
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
            conference="neurips",
            year=2024,
            tag=None,
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
        from abstracts_explorer.cli import registry_upload_command

        args = argparse.Namespace(
            repository="ghcr.io/owner/repo",
            token="test-token",
            conference="neurips",
            year=2024,
            tag=None,
        )

        with patch("abstracts_explorer.registry.RegistryClient") as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.upload.return_value = {
                "tag": "neurips-2024",
                "paper_count": 3,
                "embedding_count": 3,
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
            mock_instance.list_tags.return_value = ["neurips-2024", "iclr-2025"]

            result = registry_list_command(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "neurips-2024" in captured.out
        assert "iclr-2025" in captured.out

    def test_registry_list_specific_tag(self, capsys, monkeypatch):
        """List command with --tag shows artifact details."""
        from abstracts_explorer.cli import registry_list_command

        args = argparse.Namespace(
            repository="ghcr.io/owner/repo",
            token="test-token",
            tag="neurips-2024",
        )

        with patch("abstracts_explorer.registry.RegistryClient") as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.get_artifact_info.return_value = {
                "tag": "neurips-2024",
                "annotations": {
                    "com.abstracts-explorer.version": "1.0.0",
                    "com.abstracts-explorer.conference": "neurips",
                    "com.abstracts-explorer.year": "2024",
                    "com.abstracts-explorer.paper-count": "100",
                    "com.abstracts-explorer.embedding-count": "100",
                },
                "layers": [],
            }

            result = registry_list_command(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "1.0.0" in captured.out
        assert "neurips" in captured.out

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

        with patch(
            "sys.argv",
            [
                "abstracts-explorer",
                "registry",
                "upload",
                "-r",
                "ghcr.io/o/r",
                "--token",
                "tok",
                "-c",
                "n",
                "-y",
                "24",
            ],
        ):
            with patch("abstracts_explorer.cli.registry_upload_command", return_value=0) as mock_cmd:
                result = main()

        assert result == 0
        mock_cmd.assert_called_once()

    def test_main_dispatch_registry_download(self):
        """Main dispatches to registry download command."""
        from abstracts_explorer.cli import main

        with patch(
            "sys.argv",
            [
                "abstracts-explorer",
                "registry",
                "download",
                "-r",
                "ghcr.io/o/r",
                "--token",
                "tok",
                "-c",
                "n",
                "-y",
                "24",
            ],
        ):
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

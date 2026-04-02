"""
Tests for the registry module.

Tests the oras-based registry client, DatabaseManager export/import methods,
EmbeddingsManager export/import methods, and CLI command integration.
"""

import argparse
import json
from unittest.mock import MagicMock, Mock, patch

import pytest

from abstracts_explorer._version import __version__
from abstracts_explorer.config import get_config
from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.plugin import LightweightPaper
from abstracts_explorer.registry import (
    EmbeddingModelMismatchError,
    RegistryClient,
    RegistryError,
    _build_tag,
    _sanitize_str_for_oci_tag,
)
from tests.conftest import get_env_test_path, set_test_db

# Sanitized version string used in expected tag assertions
_VER = _sanitize_str_for_oci_tag(__version__)

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
# Tests: _sanitize_str_for_oci_tag and _build_tag
# ---------------------------------------------------------------------------


class TestSanitizeStrForOciTag:
    """Tests for the _sanitize_str_for_oci_tag helper (model names and version strings)."""

    def test_simple_model(self):
        """Simple model name passes through."""
        assert _sanitize_str_for_oci_tag("text-embedding-ada-002") == "text-embedding-ada-002"

    def test_uppercase(self):
        """Value is lowercased."""
        assert _sanitize_str_for_oci_tag("Text-Embedding-ADA-002") == "text-embedding-ada-002"

    def test_special_chars(self):
        """Non-allowed characters are replaced with hyphens."""
        assert _sanitize_str_for_oci_tag("model/name:v1") == "model-name-v1"

    def test_collapsed_hyphens(self):
        """Consecutive hyphens are collapsed."""
        assert _sanitize_str_for_oci_tag("my--model") == "my-model"

    def test_dots_underscores_preserved(self):
        """Dots and underscores are kept."""
        assert _sanitize_str_for_oci_tag("model_v1.2") == "model_v1.2"

    def test_simple_version(self):
        """Simple release version passes through unchanged."""
        assert _sanitize_str_for_oci_tag("1.0.0") == "1.0.0"

    def test_dev_version(self):
        """Dev pre-release version passes through unchanged."""
        assert _sanitize_str_for_oci_tag("0.1.dev2") == "0.1.dev2"

    def test_local_segment_plus_replaced(self):
        """PEP 440 '+' local-version separator is replaced with '-'."""
        assert _sanitize_str_for_oci_tag("0.1.dev2+g2abcfb2a2") == "0.1.dev2-g2abcfb2a2"

    def test_version_uppercase_lowercased(self):
        """Version string is lowercased."""
        assert _sanitize_str_for_oci_tag("1.0.0.Post1") == "1.0.0.post1"


class TestBuildTag:
    """Tests for the _build_tag helper."""

    def test_simple_tag(self):
        """Tag is built from conference, year, embedding model and version."""
        assert _build_tag("neurips", 2024, embedding_model="model-a", version="1.0.0") == "neurips-2024_model-a_1.0.0"

    def test_case_normalization(self):
        """Conference name is lowercased."""
        assert _build_tag("NeurIPS", 2024, embedding_model="model-a", version="1.0.0") == "neurips-2024_model-a_1.0.0"

    def test_special_characters(self):
        """Special characters are replaced with hyphens."""
        assert (
            _build_tag("ML4PS/workshop", 2025, embedding_model="model-a", version="1.0.0")
            == "ml4ps-workshop-2025_model-a_1.0.0"
        )

    def test_conference_only_tag(self):
        """Tag without year contains conference, model and version."""
        assert _build_tag("neurips", embedding_model="model-a", version="1.0.0") == "neurips_model-a_1.0.0"

    def test_conference_only_tag_normalized(self):
        """Conference-only tag is lowercased and sanitized."""
        assert (
            _build_tag("ML4PS/workshop", embedding_model="model-a", version="1.0.0") == "ml4ps-workshop_model-a_1.0.0"
        )

    def test_tag_with_embedding_model(self):
        """Embedding model is appended after underscore separator."""
        assert _build_tag("neurips", 2024, embedding_model="text-embedding-ada-002", version="1.0.0") == (
            "neurips-2024_text-embedding-ada-002_1.0.0"
        )

    def test_tag_conference_only_with_model(self):
        """Conference-only tag includes the embedding model and version."""
        assert (
            _build_tag("neurips", embedding_model="text-embedding-ada-002", version="1.0.0")
            == "neurips_text-embedding-ada-002_1.0.0"
        )

    def test_version_with_local_segment(self):
        """Dev version with local segment (PEP 440 '+') is sanitized."""
        assert _build_tag("neurips", 2024, embedding_model="model-a", version="0.1.dev2+g2abcfb2a2") == (
            "neurips-2024_model-a_0.1.dev2-g2abcfb2a2"
        )

    def test_default_version_uses_package_version(self):
        """When version is not given, the package __version__ is used."""
        tag = _build_tag("neurips", 2024, embedding_model="model-a")
        expected_version = _sanitize_str_for_oci_tag(__version__)
        assert tag == f"neurips-2024_model-a_{expected_version}"


# ---------------------------------------------------------------------------
# Tests: RegistryClient initialisation
# ---------------------------------------------------------------------------


class TestRegistryClient:
    """Tests for RegistryClient construction."""

    def test_init_valid_repository(self):
        """Client initializes correctly with valid repository."""
        with patch("oras.client.OrasClient"):
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="test-token")

        assert client.registry == "ghcr.io"
        assert client.name == "thawn/abstracts-data"
        assert client.repository == "ghcr.io/thawn/abstracts-data"
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
            client = RegistryClient("ghcr.io/thawn/abstracts-data")
        assert client.token == "env-token"

    def test_list_tags(self):
        """Tags are listed from registry."""
        with patch("oras.client.OrasClient") as MockOras:
            mock_oras = MockOras.return_value
            mock_oras.get_tags.return_value = ["neurips-2024", "iclr-2025"]
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")
            tags = client.list_tags()

        assert tags == ["neurips-2024", "iclr-2025"]

    def test_list_tags_error(self):
        """RegistryError raised when listing fails."""
        with patch("oras.client.OrasClient") as MockOras:
            mock_oras = MockOras.return_value
            mock_oras.get_tags.side_effect = Exception("network error")
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

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
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")
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

    def test_import_embedding_model_mismatch(self, tmp_path):
        """Import raises EmbeddingModelConflictError when embedding models don't match."""
        from abstracts_explorer.database import EmbeddingModelConflictError
        from abstracts_explorer.db_models import EmbeddingsMetadata

        # Create source database with one embedding model
        db1 = _populate_test_db(tmp_path / "db1.db")
        try:
            db1._session.add(EmbeddingsMetadata(embedding_model="model-A"))
            db1._session.commit()
            export_path = tmp_path / "export.db"
            db1.export_papers_to_sqlite(export_path, "neurips", 2024)
        finally:
            db1.close()

        # Create target database with a different embedding model
        target_path = tmp_path / "target.db"
        db2 = _populate_test_db(target_path)
        try:
            db2._session.add(EmbeddingsMetadata(embedding_model="model-B"))
            db2._session.commit()

            with pytest.raises(EmbeddingModelConflictError) as exc_info:
                db2.import_papers_from_sqlite(export_path, "neurips", 2024)
            assert exc_info.value.local_model == "model-B"
            assert exc_info.value.remote_model == "model-A"
        finally:
            db2.close()

    def test_import_embedding_model_consistent(self, tmp_path):
        """Import succeeds when embedding models match."""
        from abstracts_explorer.db_models import EmbeddingsMetadata

        # Create source database
        db1 = _populate_test_db(tmp_path / "db1.db")
        try:
            db1._session.add(EmbeddingsMetadata(embedding_model="model-A"))
            db1._session.commit()
            export_path = tmp_path / "export.db"
            db1.export_papers_to_sqlite(export_path, "neurips", 2024)
        finally:
            db1.close()

        # Create target database with same model
        target_path = tmp_path / "target.db"
        db2 = _populate_test_db(target_path)
        try:
            db2._session.add(EmbeddingsMetadata(embedding_model="model-A"))
            db2._session.commit()

            count = db2.import_papers_from_sqlite(export_path, "neurips", 2024)
            assert count == 1
        finally:
            db2.close()

    def test_import_embedding_model_alias_prefix_ignored(self, tmp_path):
        """Import succeeds when models differ only by alias- prefix."""
        from abstracts_explorer.db_models import EmbeddingsMetadata

        # Create source database with alias-prefixed model
        db1 = _populate_test_db(tmp_path / "db1.db")
        try:
            db1._session.add(EmbeddingsMetadata(embedding_model="alias-qwen3-embeddings-8b"))
            db1._session.commit()
            export_path = tmp_path / "export.db"
            db1.export_papers_to_sqlite(export_path, "neurips", 2024)
        finally:
            db1.close()

        # Create target database with model without alias- prefix
        target_path = tmp_path / "target.db"
        db2 = _populate_test_db(target_path)
        try:
            db2._session.add(EmbeddingsMetadata(embedding_model="qwen3-embeddings-8b"))
            db2._session.commit()

            count = db2.import_papers_from_sqlite(export_path, "neurips", 2024)
            assert count == 1
        finally:
            db2.close()

    def test_import_embedding_metadata_not_overwritten(self, tmp_path):
        """Existing embedding metadata is preserved when consistent."""
        from abstracts_explorer.db_models import EmbeddingsMetadata

        # Create source database
        db1 = _populate_test_db(tmp_path / "db1.db")
        try:
            db1._session.add(EmbeddingsMetadata(embedding_model="model-A"))
            db1._session.commit()
            export_path = tmp_path / "export.db"
            db1.export_papers_to_sqlite(export_path, "neurips", 2024)
        finally:
            db1.close()

        # Create target database with same model
        target_path = tmp_path / "target.db"
        db2 = _populate_test_db(target_path)
        try:
            db2._session.add(EmbeddingsMetadata(embedding_model="model-A"))
            db2._session.commit()

            db2.import_papers_from_sqlite(export_path, "neurips", 2024)

            # Verify only one metadata row remains (the existing one)
            meta_count = db2._session.query(EmbeddingsMetadata).count()
            assert meta_count == 1
        finally:
            db2.close()

    def test_import_scoped_clustering_cache_deletion(self, tmp_path):
        """Only clustering cache entries matching conference+year are deleted on import."""
        from abstracts_explorer.db_models import ClusteringCache

        db = _populate_test_db(tmp_path / "db.db")
        try:
            # Add clustering cache: one for neurips/2024, one for iclr/2024
            import json as _json

            db._session.add(
                ClusteringCache(
                    embedding_model="test-model",
                    reduction_method="pca",
                    n_components=2,
                    clustering_method="kmeans",
                    n_clusters=5,
                    clustering_params=_json.dumps({"conferences": ["neurips"], "years": [2024]}),
                    results_json="{}",
                )
            )
            db._session.add(
                ClusteringCache(
                    embedding_model="test-model",
                    reduction_method="pca",
                    n_components=2,
                    clustering_method="kmeans",
                    n_clusters=5,
                    clustering_params=_json.dumps({"conferences": ["iclr"], "years": [2024]}),
                    results_json="{}",
                )
            )
            db._session.commit()

            # Export neurips 2024
            export_path = tmp_path / "export.db"
            db.export_papers_to_sqlite(export_path, "neurips", 2024)

            # Import neurips 2024 - only the neurips entry should be deleted
            db.import_papers_from_sqlite(export_path, "neurips", 2024)

            remaining = db._session.query(ClusteringCache).all()
            # iclr entry should survive, neurips entry replaced by imported one
            params_list = [_json.loads(e.clustering_params) for e in remaining if e.clustering_params]
            iclr_found = any(p.get("conferences") == ["iclr"] for p in params_list)
            assert iclr_found, "The iclr clustering cache entry should be preserved"
        finally:
            db.close()

    def test_import_unscoped_clustering_cache_preserved(self, tmp_path):
        """Clustering cache entries without conference/year are preserved on import."""
        from abstracts_explorer.db_models import ClusteringCache

        db = _populate_test_db(tmp_path / "db.db")
        try:
            # Add a global (unscoped) clustering cache entry
            db._session.add(
                ClusteringCache(
                    embedding_model="test-model",
                    reduction_method="pca",
                    n_components=2,
                    clustering_method="kmeans",
                    n_clusters=5,
                    clustering_params=None,
                    results_json="{}",
                )
            )
            db._session.commit()

            export_path = tmp_path / "export.db"
            db.export_papers_to_sqlite(export_path, "neurips", 2024)
            db.import_papers_from_sqlite(export_path, "neurips", 2024)

            remaining = db._session.query(ClusteringCache).all()
            # Global entry should survive (no conference/year match)
            unscoped = [e for e in remaining if e.clustering_params is None]
            assert len(unscoped) == 1, "Global clustering cache entry should be preserved"
        finally:
            db.close()


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
            "metadatas": [
                {
                    "title": "Paper 1",
                    "authors": "Author A;Author B",
                    "abstract": "Abstract 1",
                    "session": "Session 1",
                    "poster_position": "1",
                    "conference": "neurips",
                    "year": "2024",
                },
                {
                    "title": "Paper 2",
                    "authors": "Author C",
                    "abstract": "Abstract 2",
                    "session": "Session 2",
                    "poster_position": "2",
                    "conference": "neurips",
                    "year": "2024",
                },
            ],
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

    def test_export_embeddings_converts_ndarray(self):
        """Numpy ndarrays in embeddings are converted to plain lists (JSON-serializable)."""
        import json

        import numpy as np

        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": ["id1"],
            "documents": ["doc1"],
            "metadatas": [
                {
                    "title": "Paper 1",
                    "authors": "Author A",
                    "abstract": "Abstract 1",
                    "session": "Session 1",
                    "poster_position": "1",
                    "conference": "neurips",
                    "year": "2024",
                }
            ],
            "embeddings": [np.array([0.1, 0.2, 0.3])],
        }

        from abstracts_explorer.embeddings import EmbeddingsManager

        em = EmbeddingsManager.__new__(EmbeddingsManager)
        em._collection = mock_collection

        result = em.export_embeddings("neurips", 2024)

        # Must be JSON-serializable (no ndarray objects)
        serialized = json.dumps(result)
        assert serialized is not None
        # Values should match
        assert result["embeddings"] == [[0.1, 0.2, 0.3]]

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
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        with patch.object(RegistryClient, "_get_embedding_model_database", return_value="test-model"):
            with pytest.raises(RegistryError, match="No papers found"):
                client.upload(conference="icml", year=2024)

    def test_upload_validates_embeddings(self, tmp_path):
        """Upload fails when no embeddings exist for conference+year."""
        _populate_test_db(tmp_path / "test.db")

        with patch("oras.client.OrasClient"):
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        mock_em = MagicMock()
        mock_em.export_embeddings.return_value = {"ids": [], "documents": [], "metadatas": [], "embeddings": []}

        with (
            patch("abstracts_explorer.embeddings.EmbeddingsManager", return_value=mock_em),
            patch.object(RegistryClient, "_get_embedding_model_database", return_value="test-model"),
        ):
            with pytest.raises(RegistryError, match="No embeddings found"):
                client.upload(conference="neurips", year=2024)

    def test_upload_success(self, tmp_path):
        """Upload succeeds with valid data."""
        _populate_test_db(tmp_path / "test.db")

        with patch("oras.client.OrasClient") as MockOras:
            mock_oras = MockOras.return_value
            mock_oras.push.return_value = Mock(status_code=201)
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        mock_em = MagicMock()
        mock_em.export_embeddings.return_value = {
            "ids": ["id1"],
            "documents": ["doc1"],
            "metadatas": [{"conference": "neurips", "year": "2024"}],
            "embeddings": [[0.1, 0.2]],
        }

        with (
            patch("abstracts_explorer.embeddings.EmbeddingsManager", return_value=mock_em),
            patch.object(RegistryClient, "_get_embedding_model_database", return_value="test-model"),
        ):
            summary = client.upload(conference="neurips", year=2024)

        assert summary["paper_count"] == 1
        assert summary["embedding_count"] == 1
        assert summary["tag"] == f"neurips-2024_test-model_{_VER}"
        assert summary["years"] == [2024]
        mock_oras.push.assert_called_once()

    def test_upload_custom_tag(self, tmp_path):
        """Upload uses custom tag when provided."""
        _populate_test_db(tmp_path / "test.db")

        with patch("oras.client.OrasClient") as MockOras:
            mock_oras = MockOras.return_value
            mock_oras.push.return_value = Mock(status_code=201)
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        mock_em = MagicMock()
        mock_em.export_embeddings.return_value = {
            "ids": ["id1"],
            "documents": ["doc1"],
            "metadatas": [{}],
            "embeddings": [[0.1]],
        }

        with (
            patch("abstracts_explorer.embeddings.EmbeddingsManager", return_value=mock_em),
            patch.object(RegistryClient, "_get_embedding_model_database", return_value="test-model"),
        ):
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
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        mock_em = MagicMock()
        mock_em.export_embeddings.return_value = {
            "ids": ["id1"],
            "documents": ["d"],
            "metadatas": [{}],
            "embeddings": [[0.1]],
        }

        messages = []
        with (
            patch("abstracts_explorer.embeddings.EmbeddingsManager", return_value=mock_em),
            patch.object(RegistryClient, "_get_embedding_model_database", return_value="test-model"),
        ):
            client.upload(
                conference="neurips",
                year=2024,
                progress_callback=messages.append,
            )

        assert len(messages) > 0
        assert any("Exporting" in m for m in messages)

    def test_upload_conference_only(self, tmp_path):
        """Upload without year uploads each year individually plus an all-years tag."""
        _populate_test_db(tmp_path / "test.db")

        with patch("oras.client.OrasClient") as MockOras:
            mock_oras = MockOras.return_value
            mock_oras.push.return_value = Mock(status_code=201)
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        mock_em = MagicMock()
        mock_em.export_embeddings.return_value = {
            "ids": ["id1"],
            "documents": ["doc1"],
            "metadatas": [{}],
            "embeddings": [[0.1]],
        }

        with (
            patch("abstracts_explorer.embeddings.EmbeddingsManager", return_value=mock_em),
            patch.object(RegistryClient, "_get_embedding_model_database", return_value="test-model"),
        ):
            summary = client.upload(conference="neurips")

        assert summary["tag"] == f"neurips_test-model_{_VER}"
        assert sorted(summary["years"]) == [2024, 2025]
        # push is called once per year (individual tags) + once for the all-years tag
        assert mock_oras.push.call_count == 3
        pushed_targets = [call[1]["target"] for call in mock_oras.push.call_args_list]
        assert f"ghcr.io/thawn/abstracts-data:neurips-2024_test-model_{_VER}" in pushed_targets
        assert f"ghcr.io/thawn/abstracts-data:neurips-2025_test-model_{_VER}" in pushed_targets
        assert f"ghcr.io/thawn/abstracts-data:neurips_test-model_{_VER}" in pushed_targets
        assert "year_tags" in summary
        assert sorted(summary["year_tags"]) == [f"neurips-2024_test-model_{_VER}", f"neurips-2025_test-model_{_VER}"]

    def test_upload_conference_only_no_data(self, tmp_path):
        """Upload without year fails when no data exists."""
        _populate_test_db(tmp_path / "test.db")

        with patch("oras.client.OrasClient"):
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        with patch.object(RegistryClient, "_get_embedding_model_database", return_value="test-model"):
            with pytest.raises(RegistryError, match="No data found"):
                client.upload(conference="icml")

    def test_upload_conference_name_case_insensitive_resolution(self, tmp_path):
        """Upload resolves conference name case-insensitively against DB data."""
        # Populate DB with mixed-case conference name "NeurIPS"
        set_test_db(tmp_path / "test.db")
        db = DatabaseManager()
        db.connect()
        db.create_tables()
        db.add_papers(
            [
                LightweightPaper(
                    title="Paper One",
                    authors=["Author A"],
                    abstract="Abstract",
                    session="Session 1",
                    poster_position="A1",
                    year=2024,
                    conference="NeurIPS",
                )
            ]
        )

        with patch("oras.client.OrasClient") as MockOras:
            mock_oras = MockOras.return_value
            mock_oras.push.return_value = Mock(status_code=201)
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        mock_em = MagicMock()
        mock_em.export_embeddings.return_value = {
            "ids": ["id1"],
            "documents": ["doc1"],
            "metadatas": [{}],
            "embeddings": [[0.1]],
        }

        with (
            patch("abstracts_explorer.embeddings.EmbeddingsManager", return_value=mock_em),
            patch.object(RegistryClient, "_get_embedding_model_database", return_value="test-model"),
        ):
            # User passes lowercase "neurips", DB has "NeurIPS" — should succeed
            summary = client.upload(conference="neurips")

        assert summary["paper_count"] == 1
        # Tag is built from the resolved "NeurIPS" name (sanitized)
        assert "neurips" in summary["tag"]

    def test_upload_no_embedding_model(self, tmp_path):
        """Upload fails when no embedding model is available."""
        _populate_test_db(tmp_path / "test.db")

        with patch("oras.client.OrasClient"):
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        with patch.object(RegistryClient, "_get_embedding_model_database", return_value=None):
            with pytest.raises(RegistryError, match="No embedding model found"):
                client.upload(conference="neurips", year=2024)

    def test_download_conference_only(self, tmp_path):
        """Download without year imports all years in artifact."""
        set_test_db(tmp_path / "target.db")

        # Create fake multi-year pulled files
        source_db_path = tmp_path / "source.db"
        source_db = _populate_test_db(source_db_path)
        papers_2024 = tmp_path / "papers-2024.db"
        source_db.export_papers_to_sqlite(papers_2024, "neurips", 2024)
        papers_2025 = tmp_path / "papers-2025.db"
        source_db.export_papers_to_sqlite(papers_2025, "neurips", 2025)
        source_db.close()

        embeddings_2024 = tmp_path / "embeddings-2024.json"
        embeddings_2024.write_text(
            json.dumps({"ids": ["a"], "documents": ["d"], "metadatas": [{}], "embeddings": [[0.1]]})
        )
        embeddings_2025 = tmp_path / "embeddings-2025.json"
        embeddings_2025.write_text(
            json.dumps({"ids": ["b"], "documents": ["d"], "metadatas": [{}], "embeddings": [[0.2]]})
        )

        with patch("oras.client.OrasClient") as MockOras:
            mock_oras = MockOras.return_value
            mock_oras.pull.return_value = [
                str(papers_2024),
                str(embeddings_2024),
                str(papers_2025),
                str(embeddings_2025),
            ]
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        mock_em = MagicMock()
        mock_em.import_embeddings.return_value = 1

        set_test_db(tmp_path / "target.db")

        with patch("abstracts_explorer.embeddings.EmbeddingsManager", return_value=mock_em):
            summary = client.download(conference="neurips", embedding_model="test-model")

        assert sorted(summary["years"]) == [2024, 2025]
        assert summary["paper_count"] == 2  # 1 per year
        assert mock_em.import_embeddings.call_count == 2

    def test_download_success(self, tmp_path):
        """Download imports paper database and embeddings."""
        set_test_db(tmp_path / "target.db")

        # Create a fake paper db to be pulled
        source_db_path = tmp_path / "source.db"
        source_db = _populate_test_db(source_db_path)
        export_path = tmp_path / "papers-2024.db"
        source_db.export_papers_to_sqlite(export_path, "neurips", 2024)
        source_db.close()

        # Create fake embeddings file
        embeddings_data = {
            "ids": ["id1"],
            "documents": ["doc1"],
            "metadatas": [{"conference": "neurips", "year": "2024"}],
            "embeddings": [[0.1, 0.2]],
        }
        embeddings_path = tmp_path / "embeddings-2024.json"
        embeddings_path.write_text(json.dumps(embeddings_data))

        with patch("oras.client.OrasClient") as MockOras:
            mock_oras = MockOras.return_value
            mock_oras.pull.return_value = [str(export_path), str(embeddings_path)]
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        mock_em = MagicMock()
        mock_em.import_embeddings.return_value = 1

        # Reset target DB
        set_test_db(tmp_path / "target.db")

        with patch("abstracts_explorer.embeddings.EmbeddingsManager", return_value=mock_em):
            summary = client.download(conference="neurips", year=2024, embedding_model="test-model")

        assert summary["paper_count"] == 1
        assert summary["embedding_count"] == 1
        assert 2024 in summary["years"]
        mock_em.import_embeddings.assert_called_once()

    def test_download_incomplete_data_raises_error(self, tmp_path):
        """Download raises RegistryError when artifact has paper DB but no embeddings."""
        set_test_db(tmp_path / "target.db")

        # Create only paper DB, no embeddings file
        source_db_path = tmp_path / "source.db"
        source_db = _populate_test_db(source_db_path)
        papers_2024 = tmp_path / "papers-2024.db"
        source_db.export_papers_to_sqlite(papers_2024, "neurips", 2024)
        source_db.close()

        with patch("oras.client.OrasClient") as MockOras:
            mock_oras = MockOras.return_value
            mock_oras.pull.return_value = [str(papers_2024)]
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        set_test_db(tmp_path / "target.db")

        with pytest.raises(RegistryError, match="Incomplete data.*missing.*embeddings"):
            client.download(conference="neurips", year=2024, embedding_model="test-model")

    def test_import_year_missing_paper_db(self, tmp_path):
        """_import_year raises RegistryError when paper DB file is missing."""
        set_test_db(tmp_path / "target.db")

        with patch("oras.client.OrasClient"):
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        embeddings_path = tmp_path / "embeddings-2024.json"
        embeddings_path.write_text(json.dumps({"ids": [], "documents": [], "metadatas": [], "embeddings": []}))

        with pytest.raises(RegistryError, match="Incomplete data.*missing.*paper DB"):
            client._import_year("neurips", 2024, tmp_path / "papers-2024.db", embeddings_path, lambda m: None)

    def test_import_year_missing_embeddings(self, tmp_path):
        """_import_year raises RegistryError when embeddings file is missing."""
        set_test_db(tmp_path / "target.db")

        source_db = _populate_test_db(tmp_path / "source.db")
        papers_2024 = tmp_path / "papers-2024.db"
        source_db.export_papers_to_sqlite(papers_2024, "neurips", 2024)
        source_db.close()

        with patch("oras.client.OrasClient"):
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        with pytest.raises(RegistryError, match="Incomplete data.*missing.*embeddings"):
            client._import_year("neurips", 2024, papers_2024, tmp_path / "embeddings-2024.json", lambda m: None)

    def test_import_year_rollback_on_embedding_failure(self, tmp_path):
        """Paper DB import is rolled back when embedding import fails."""
        set_test_db(tmp_path / "target.db")

        # Set up a fresh target database
        db = DatabaseManager()
        db.connect()
        db.create_tables()
        db.close()

        # Create a valid paper DB export
        source_db = _populate_test_db(tmp_path / "source.db")
        papers_2024 = tmp_path / "papers-2024.db"
        source_db.export_papers_to_sqlite(papers_2024, "neurips", 2024)
        source_db.close()

        # Create embeddings file with valid JSON but make import fail
        embeddings_path = tmp_path / "embeddings-2024.json"
        embeddings_path.write_text(
            json.dumps(
                {
                    "ids": ["id1"],
                    "documents": ["doc1"],
                    "metadatas": [{"conference": "neurips", "year": "2024"}],
                    "embeddings": [[0.1]],
                }
            )
        )

        set_test_db(tmp_path / "target.db")

        with patch("oras.client.OrasClient"):
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        # Make EmbeddingsManager.import_embeddings raise an error
        with patch("abstracts_explorer.embeddings.EmbeddingsManager") as MockEM:
            MockEM.return_value.import_embeddings.side_effect = Exception("ChromaDB failure")

            with pytest.raises(RegistryError, match="Embedding import failed.*rolled back"):
                client._import_year("neurips", 2024, papers_2024, embeddings_path, lambda m: None)

        # Verify that paper DB was rolled back (no neurips/2024 papers in target)
        from abstracts_explorer.db_models import Paper

        set_test_db(tmp_path / "target.db")
        db = DatabaseManager()
        db.connect()
        db.create_tables()
        try:
            from sqlalchemy import and_, select

            stmt = select(Paper).where(and_(Paper.conference == "neurips", Paper.year == 2024))
            papers = db._session.execute(stmt).scalars().all()
            assert len(papers) == 0, "Papers should be rolled back after embedding import failure"
        finally:
            db.close()

    def test_upload_includes_embedding_model(self, tmp_path):
        """Upload manifest includes the embedding model annotation."""
        _populate_test_db(tmp_path / "test.db")

        with patch("oras.client.OrasClient") as MockOras:
            mock_oras = MockOras.return_value
            mock_oras.push.return_value = Mock(status_code=201)
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        mock_em = MagicMock()
        mock_em.export_embeddings.return_value = {
            "ids": ["id1"],
            "documents": ["doc1"],
            "metadatas": [{"conference": "neurips", "year": "2024"}],
            "embeddings": [[0.1, 0.2]],
        }

        # Mock the embedding model lookup
        with (
            patch("abstracts_explorer.embeddings.EmbeddingsManager", return_value=mock_em),
            patch.object(RegistryClient, "_get_embedding_model_database", return_value="text-embedding-ada-002"),
        ):
            summary = client.upload(conference="neurips", year=2024)

        assert summary["tag"] == f"neurips-2024_text-embedding-ada-002_{_VER}"
        # Verify embedding model is in the manifest annotations
        push_kwargs = mock_oras.push.call_args[1]
        annotations = push_kwargs.get("manifest_annotations", {})
        assert annotations.get("com.abstracts-explorer.embedding-model") == "text-embedding-ada-002"
        # disable_path_validation must be True so temp-dir blobs are accepted
        assert push_kwargs.get("disable_path_validation") is True

    def test_upload_all_conferences(self, tmp_path):
        """upload_all uploads data for all conferences."""
        _populate_test_db(tmp_path / "test.db")

        with patch("oras.client.OrasClient") as MockOras:
            mock_oras = MockOras.return_value
            mock_oras.push.return_value = Mock(status_code=201)
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        mock_em = MagicMock()
        mock_em.export_embeddings.return_value = {
            "ids": ["id1"],
            "documents": ["doc1"],
            "metadatas": [{}],
            "embeddings": [[0.1]],
        }

        with (
            patch("abstracts_explorer.embeddings.EmbeddingsManager", return_value=mock_em),
            patch.object(RegistryClient, "_get_embedding_model_database", return_value="test-model"),
        ):
            summaries = client.upload_all(progress_callback=lambda m: None)

        # We have neurips and iclr in our test data
        conferences = sorted(s["conference"] for s in summaries)
        assert "iclr" in conferences
        assert "neurips" in conferences

    def test_upload_all_no_conferences(self, tmp_path):
        """upload_all raises RegistryError when no conferences exist."""
        set_test_db(tmp_path / "empty.db")
        db = DatabaseManager()
        db.connect()
        db.create_tables()
        db.close()

        with patch("oras.client.OrasClient"):
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        with pytest.raises(RegistryError, match="No conference data found"):
            client.upload_all()

    def test_download_all_conferences(self, tmp_path):
        """download_all downloads all tags from registry."""
        set_test_db(tmp_path / "target.db")

        # Create fake pulled files for each download
        source_db = _populate_test_db(tmp_path / "source.db")

        # For iclr-2024 tag (alphabetically first)
        papers_i = tmp_path / "pull_i" / "papers-2024.db"
        papers_i.parent.mkdir()
        source_db.export_papers_to_sqlite(papers_i, "iclr", 2024)
        emb_i = tmp_path / "pull_i" / "embeddings-2024.json"
        emb_i.write_text(json.dumps({"ids": ["b"], "documents": ["d"], "metadatas": [{}], "embeddings": [[0.2]]}))

        # For neurips-2024 tag
        papers_n = tmp_path / "pull_n" / "papers-2024.db"
        papers_n.parent.mkdir()
        source_db.export_papers_to_sqlite(papers_n, "neurips", 2024)
        emb_n = tmp_path / "pull_n" / "embeddings-2024.json"
        emb_n.write_text(json.dumps({"ids": ["a"], "documents": ["d"], "metadatas": [{}], "embeddings": [[0.1]]}))
        source_db.close()

        mock_em = MagicMock()
        mock_em.import_embeddings.return_value = 1

        set_test_db(tmp_path / "target.db")

        with patch("oras.client.OrasClient") as MockOras:
            mock_oras = MockOras.return_value
            mock_oras.get_tags.return_value = [
                "neurips-2024_text-embedding-ada-002",
                "iclr-2024_text-embedding-ada-002",
            ]
            # get_manifest returns annotations for each tag
            mock_oras.get_manifest.side_effect = [
                {
                    "annotations": {
                        "com.abstracts-explorer.conference": "iclr",
                        "com.abstracts-explorer.years": "2024",
                    }
                },
                {
                    "annotations": {
                        "com.abstracts-explorer.conference": "neurips",
                        "com.abstracts-explorer.years": "2024",
                    }
                },
            ]
            # Side effect returns files in order of download (sorted tags: iclr..., neurips...)
            mock_oras.pull.side_effect = [
                [str(papers_i), str(emb_i)],  # first call: iclr-2024
                [str(papers_n), str(emb_n)],  # second call: neurips-2024
            ]
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

            with patch("abstracts_explorer.embeddings.EmbeddingsManager", return_value=mock_em):
                summaries = client.download_all(progress_callback=lambda m: None)

        assert len(summaries) == 2
        assert mock_em.import_embeddings.call_count == 2

    def test_download_all_no_tags(self, tmp_path):
        """download_all raises RegistryError when no tags exist."""
        with patch("oras.client.OrasClient") as MockOras:
            mock_oras = MockOras.return_value
            mock_oras.get_tags.return_value = []
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        with pytest.raises(RegistryError, match="No tags found"):
            client.download_all()

    def test_cli_upload_all_conferences(self, tmp_path, capsys, monkeypatch):
        """CLI upload with --conference all invokes upload_all."""
        from abstracts_explorer.cli import registry_upload_command

        args = argparse.Namespace(
            repository="ghcr.io/thawn/abstracts-data",
            token="test-token",
            conference="all",
            year=None,
            tag=None,
        )

        with patch("abstracts_explorer.registry.RegistryClient") as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.upload_all.return_value = [
                {"conference": "neurips", "tag": "neurips", "paper_count": 3, "embedding_count": 3, "years": [2024]},
                {"conference": "iclr", "tag": "iclr", "paper_count": 2, "embedding_count": 2, "years": [2025]},
            ]
            result = registry_upload_command(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Upload complete" in captured.out
        assert "2 conference(s)" in captured.out

    def test_cli_download_all_conferences(self, tmp_path, capsys, monkeypatch):
        """CLI download with --conference all invokes download_all."""
        from abstracts_explorer.cli import registry_download_command

        args = argparse.Namespace(
            repository="ghcr.io/thawn/abstracts-data",
            token="test-token",
            conference="all",
            year=None,
            tag=None,
            yes=True,
        )

        with patch("abstracts_explorer.registry.RegistryClient") as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.download_all.return_value = [
                {"conference": "neurips", "paper_count": 3, "embedding_count": 3},
                {"conference": "iclr", "paper_count": 2, "embedding_count": 2},
            ]
            result = registry_download_command(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Download complete" in captured.out
        assert "2 artifact(s)" in captured.out


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
        monkeypatch.setenv("REGISTRY_REPOSITORY", "ghcr.io/thawn/abstracts-data")
        get_config(reload=True, env_path=get_env_test_path())

        from abstracts_explorer.cli import registry_upload_command

        args = argparse.Namespace(
            repository="ghcr.io/thawn/abstracts-data",
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
            repository="ghcr.io/thawn/abstracts-data",
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
                "years": [2024],
            }

            result = registry_upload_command(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Upload complete" in captured.out

    def test_registry_list_tags(self, capsys, monkeypatch):
        """List command shows available tags."""
        from abstracts_explorer.cli import registry_list_command

        args = argparse.Namespace(
            repository="ghcr.io/thawn/abstracts-data",
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
            repository="ghcr.io/thawn/abstracts-data",
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
                    "com.abstracts-explorer.years": "2024",
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
            repository="ghcr.io/thawn/abstracts-data",
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

    def test_registry_upload_conference_case_insensitive(self, capsys):
        """Upload treats conference name as case-insensitive (e.g. 'ALL' == 'all')."""
        from abstracts_explorer.cli import registry_upload_command

        args = argparse.Namespace(
            repository="ghcr.io/thawn/abstracts-data",
            token="test-token",
            conference="ALL",
            year=None,
            tag=None,
        )

        with patch("abstracts_explorer.registry.RegistryClient") as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.upload_all.return_value = []

            result = registry_upload_command(args)

        assert result == 0
        mock_instance.upload_all.assert_called_once()

    def test_registry_download_conference_case_insensitive(self, capsys):
        """Download treats conference name as case-insensitive (e.g. 'ALL' == 'all')."""
        from abstracts_explorer.cli import registry_download_command

        args = argparse.Namespace(
            repository="ghcr.io/thawn/abstracts-data",
            token="test-token",
            conference="ALL",
            year=None,
            tag=None,
            yes=True,
            embedding_model=None,
        )

        with patch("abstracts_explorer.registry.RegistryClient") as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.download_all.return_value = []

            result = registry_download_command(args)

        assert result == 0
        mock_instance.download_all.assert_called_once()

    def test_embedding_model_mismatch_error_attributes(self):
        """EmbeddingModelMismatchError carries local_model and remote_model."""

        err = EmbeddingModelMismatchError("model-a", "model-b")
        assert err.local_model == "model-a"
        assert err.remote_model == "model-b"
        assert "model-a" in str(err)
        assert "model-b" in str(err)

    def test_import_year_raises_mismatch_error(self, tmp_path):
        """_import_year raises EmbeddingModelMismatchError when embedding models differ."""
        from abstracts_explorer.database import EmbeddingModelConflictError

        set_test_db(tmp_path / "target.db")

        paper_db_path = tmp_path / "papers-2024.db"
        paper_db_path.touch()  # Just needs to exist for pre-flight check
        embeddings_path = tmp_path / "embeddings-2024.json"
        embeddings_path.write_text(json.dumps({"ids": [], "documents": [], "metadatas": [], "embeddings": []}))

        with patch("oras.client.OrasClient"):
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        with patch(
            "abstracts_explorer.database.DatabaseManager.import_papers_from_sqlite",
            side_effect=EmbeddingModelConflictError("model-a", "model-b"),
        ):
            with pytest.raises(EmbeddingModelMismatchError) as exc_info:
                client._import_year("neurips", 2024, paper_db_path, embeddings_path, lambda m: None)

        assert exc_info.value.local_model == "model-a"
        assert exc_info.value.remote_model == "model-b"

    def _make_artifact_paper_db(self, path, embedding_model: str) -> None:
        """Create a SQLite DB file with the real schema and an embeddings_metadata row for testing."""
        from tests.conftest import set_test_db

        set_test_db(path)
        with DatabaseManager() as db:
            db.create_tables()
            db.set_embedding_model(embedding_model)

    def test_import_year_raises_mismatch_when_db_empty_and_model_differs(self, tmp_path):
        """_import_year raises EmbeddingModelMismatchError when local DB is empty but artifact model differs."""
        set_test_db(tmp_path / "target.db")

        paper_db_path = tmp_path / "papers-2024.db"
        self._make_artifact_paper_db(paper_db_path, "artifact-model-b")
        embeddings_path = tmp_path / "embeddings-2024.json"
        embeddings_path.write_text(json.dumps({"ids": [], "documents": [], "metadatas": [], "embeddings": []}))

        with patch("oras.client.OrasClient"):
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        with pytest.raises(EmbeddingModelMismatchError) as exc_info:
            client._import_year(
                "neurips",
                2024,
                paper_db_path,
                embeddings_path,
                lambda m: None,
                embedding_model="configured-model-a",
            )

        assert exc_info.value.local_model == "configured-model-a"
        assert exc_info.value.remote_model == "artifact-model-b"

    def test_import_year_ignores_mismatch_when_flag_set_and_db_empty(self, tmp_path):
        """_import_year proceeds and returns mismatch_was_ignored=True when flag is set and local DB is empty."""
        set_test_db(tmp_path / "target.db")

        paper_db_path = tmp_path / "papers-2024.db"
        self._make_artifact_paper_db(paper_db_path, "artifact-model-b")
        embeddings_path = tmp_path / "embeddings-2024.json"
        embeddings_path.write_text(json.dumps({"ids": [], "documents": [], "metadatas": [], "embeddings": []}))

        with patch("oras.client.OrasClient"):
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        with patch("abstracts_explorer.database.DatabaseManager.import_papers_from_sqlite", return_value=0):
            with patch("abstracts_explorer.embeddings.EmbeddingsManager.import_embeddings", return_value=0):
                result = client._import_year(
                    "neurips",
                    2024,
                    paper_db_path,
                    embeddings_path,
                    lambda m: None,
                    embedding_model="configured-model-a",
                    ignore_embedding_model_mismatch=True,
                )

        assert result["mismatch_was_ignored"] is True

    def test_import_year_no_mismatch_when_models_match(self, tmp_path):
        """_import_year returns mismatch_was_ignored=False when models match."""
        set_test_db(tmp_path / "target.db")

        paper_db_path = tmp_path / "papers-2024.db"
        self._make_artifact_paper_db(paper_db_path, "same-model")
        embeddings_path = tmp_path / "embeddings-2024.json"
        embeddings_path.write_text(json.dumps({"ids": [], "documents": [], "metadatas": [], "embeddings": []}))

        with patch("oras.client.OrasClient"):
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        with patch("abstracts_explorer.database.DatabaseManager.import_papers_from_sqlite", return_value=0):
            with patch("abstracts_explorer.embeddings.EmbeddingsManager.import_embeddings", return_value=0):
                result = client._import_year(
                    "neurips",
                    2024,
                    paper_db_path,
                    embeddings_path,
                    lambda m: None,
                    embedding_model="same-model",
                )

        assert result["mismatch_was_ignored"] is False

    def test_download_raises_mismatch_when_db_empty_and_no_manifest_model(self, tmp_path):
        """download() raises EmbeddingModelMismatchError from artifact paper DB when manifest has no model label."""
        set_test_db(tmp_path / "target.db")

        paper_db_path = tmp_path / "papers-2024.db"
        self._make_artifact_paper_db(paper_db_path, "artifact-model-b")
        embeddings_path = tmp_path / "embeddings-2024.json"
        embeddings_path.write_text(json.dumps({"ids": [], "documents": [], "metadatas": [], "embeddings": []}))

        pulled_files = [str(paper_db_path), str(embeddings_path)]

        with patch("oras.client.OrasClient"):
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        with patch.object(client, "_get_manifest_embedding_model", return_value=None):
            with patch.object(client._client, "pull", return_value=pulled_files):
                # Manifest has no model (legacy) → falls through to _import_year() check on paper DB
                with pytest.raises(EmbeddingModelMismatchError) as exc_info:
                    client.download(
                        conference="neurips",
                        year=2024,
                        tag="neurips-2024_model-a",
                        embedding_model="configured-model-a",
                    )

        assert exc_info.value.local_model == "configured-model-a"
        assert exc_info.value.remote_model == "artifact-model-b"

    def test_download_mismatch_with_matching_config_prompts_clear(self, tmp_path, capsys, monkeypatch):
        """Download offers to clear local data when configured model matches remote model."""
        from abstracts_explorer.cli import registry_download_command

        args = argparse.Namespace(
            repository="ghcr.io/thawn/abstracts-data",
            token="test-token",
            conference="neurips",
            year=2024,
            tag=None,
            yes=False,
            embedding_model="model-b",
        )

        mismatch = EmbeddingModelMismatchError("model-a", "model-b")
        success_summary = {
            "tag": "neurips-2024_model-b",
            "conference": "neurips",
            "years": [2024],
            "paper_count": 10,
            "embedding_count": 10,
            "metadata": {},
        }

        # Simulate user confirming the clear
        monkeypatch.setattr("builtins.input", lambda _: "y")

        with patch("abstracts_explorer.registry.RegistryClient") as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.download.side_effect = [mismatch, success_summary]

            with patch("abstracts_explorer.registry.RegistryClient.clear_local_embedding_data") as mock_clear:
                result = registry_download_command(args)

        mock_clear.assert_called_once()
        assert result == 0

    def test_download_mismatch_with_matching_config_yes_flag(self, tmp_path, capsys):
        """Download clears local data without prompting when --yes is set."""
        from abstracts_explorer.cli import registry_download_command

        args = argparse.Namespace(
            repository="ghcr.io/thawn/abstracts-data",
            token="test-token",
            conference="neurips",
            year=2024,
            tag=None,
            yes=True,
            embedding_model="model-b",
        )

        mismatch = EmbeddingModelMismatchError("model-a", "model-b")
        success_summary = {
            "tag": "neurips-2024_model-b",
            "conference": "neurips",
            "years": [2024],
            "paper_count": 5,
            "embedding_count": 5,
            "metadata": {},
        }

        with patch("abstracts_explorer.registry.RegistryClient") as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.download.side_effect = [mismatch, success_summary]

            with patch("abstracts_explorer.registry.RegistryClient.clear_local_embedding_data") as mock_clear:
                result = registry_download_command(args)

        mock_clear.assert_called_once()
        assert result == 0

    def test_download_mismatch_user_declines_clear(self, tmp_path, capsys, monkeypatch):
        """Download aborts when user declines to clear local data."""
        from abstracts_explorer.cli import registry_download_command

        args = argparse.Namespace(
            repository="ghcr.io/thawn/abstracts-data",
            token="test-token",
            conference="neurips",
            year=2024,
            tag=None,
            yes=False,
            embedding_model="model-b",
        )

        mismatch = EmbeddingModelMismatchError("model-a", "model-b")

        monkeypatch.setattr("builtins.input", lambda _: "n")

        with patch("abstracts_explorer.registry.RegistryClient") as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.download.side_effect = mismatch

            with patch("abstracts_explorer.registry.RegistryClient.clear_local_embedding_data") as mock_clear:
                result = registry_download_command(args)

        mock_clear.assert_not_called()
        assert result == 1

    def test_download_mismatch_config_does_not_match_remote(self, tmp_path, capsys):
        """Download shows error when configured model doesn't match remote model."""
        from abstracts_explorer.cli import registry_download_command

        args = argparse.Namespace(
            repository="ghcr.io/thawn/abstracts-data",
            token="test-token",
            conference="neurips",
            year=2024,
            tag=None,
            yes=True,  # skip initial confirmation
            embedding_model="model-c",  # different from remote "model-b"
        )

        mismatch = EmbeddingModelMismatchError("model-a", "model-b")

        with patch("abstracts_explorer.registry.RegistryClient") as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.download.side_effect = mismatch

            with patch("abstracts_explorer.registry.RegistryClient.clear_local_embedding_data") as mock_clear:
                result = registry_download_command(args)

        mock_clear.assert_not_called()
        assert result == 1

    def test_download_mismatch_config_does_not_match_remote_suggests_ignore_flag(self, tmp_path, capsys):
        """Download error message mentions --ignore-embedding-model-mismatch when configured model differs."""
        from abstracts_explorer.cli import registry_download_command

        args = argparse.Namespace(
            repository="ghcr.io/thawn/abstracts-data",
            token="test-token",
            conference="neurips",
            year=2024,
            tag=None,
            yes=True,
            embedding_model="model-c",
        )

        mismatch = EmbeddingModelMismatchError("model-c", "model-b")

        with patch("abstracts_explorer.registry.RegistryClient") as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.download.side_effect = mismatch

            result = registry_download_command(args)

        captured = capsys.readouterr()
        assert result == 1
        assert "--ignore-embedding-model-mismatch" in captured.err

    def test_download_ignore_embedding_model_mismatch_proceeds(self, tmp_path, capsys):
        """Download proceeds when --ignore-embedding-model-mismatch is set and models differ."""
        from abstracts_explorer.cli import registry_download_command

        args = argparse.Namespace(
            repository="ghcr.io/thawn/abstracts-data",
            token="test-token",
            conference="neurips",
            year=2024,
            tag=None,
            yes=True,
            embedding_model="model-a",
            ignore_embedding_model_mismatch=True,
        )

        success_summary = {
            "tag": "neurips-2024_model-a",
            "conference": "neurips",
            "years": [2024],
            "paper_count": 5,
            "embedding_count": 5,
            "metadata": {},
        }

        with patch("abstracts_explorer.registry.RegistryClient") as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.download.return_value = success_summary

            result = registry_download_command(args)

        assert result == 0
        mock_instance.download.assert_called_once_with(
            conference="neurips",
            year=2024,
            tag=None,
            embedding_model="model-a",
            progress_callback=mock_instance.download.call_args.kwargs["progress_callback"],
            ignore_embedding_model_mismatch=True,
        )

    def test_download_ignore_mismatch_flag_passed_to_client(self, tmp_path, capsys):
        """When --ignore-embedding-model-mismatch is set, the flag is passed to client.download()."""
        from abstracts_explorer.cli import registry_download_command

        args = argparse.Namespace(
            repository="ghcr.io/thawn/abstracts-data",
            token="test-token",
            conference="neurips",
            year=2024,
            tag=None,
            yes=True,
            embedding_model="model-a",
            ignore_embedding_model_mismatch=True,
        )

        with patch("abstracts_explorer.registry.RegistryClient") as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.download.return_value = {
                "tag": "neurips-2024_model-a",
                "conference": "neurips",
                "years": [2024],
                "paper_count": 0,
                "embedding_count": 0,
                "metadata": {},
            }

            registry_download_command(args)

        call_kwargs = mock_instance.download.call_args.kwargs
        assert call_kwargs["ignore_embedding_model_mismatch"] is True

    def test_download_ignore_mismatch_flag_default_false(self, tmp_path, capsys):
        """When --ignore-embedding-model-mismatch is not set, false is passed to client.download()."""
        from abstracts_explorer.cli import registry_download_command

        args = argparse.Namespace(
            repository="ghcr.io/thawn/abstracts-data",
            token="test-token",
            conference="neurips",
            year=2024,
            tag=None,
            yes=True,
            embedding_model="model-a",
            # no ignore_embedding_model_mismatch attribute — should default to False
        )

        with patch("abstracts_explorer.registry.RegistryClient") as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.download.return_value = {
                "tag": "neurips-2024_model-a",
                "conference": "neurips",
                "years": [2024],
                "paper_count": 0,
                "embedding_count": 0,
                "metadata": {},
            }

            registry_download_command(args)

        call_kwargs = mock_instance.download.call_args.kwargs
        assert call_kwargs["ignore_embedding_model_mismatch"] is False

    # ------------------------------------------------------------------
    # _find_best_matching_tag tests
    # ------------------------------------------------------------------

    def test_find_best_matching_tag_exact_match_returned(self):
        """Returns the exact tag when it exists in the registry."""
        with patch("oras.client.OrasClient"):
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        available = ["neurips-2024_my-model_0.4.0", "neurips-2024_my-model_0.4.1"]
        with patch.object(client._client, "get_tags", return_value=available):
            result = client._find_best_matching_tag("neurips-2024_my-model_0.4.1")

        assert result == "neurips-2024_my-model_0.4.1"

    def test_find_best_matching_tag_resolves_version_mismatch(self):
        """Resolves a tag with the wrong version to the highest available version."""
        with patch("oras.client.OrasClient"):
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        available = ["neurips-2024_my-model_0.4.0", "neurips-2024_my-model_0.4.1"]
        with patch.object(client._client, "get_tags", return_value=available):
            result = client._find_best_matching_tag("neurips-2024_my-model_0.5.0")

        assert result == "neurips-2024_my-model_0.4.1"

    def test_find_best_matching_tag_no_prefix_match_returns_original(self):
        """Returns the original tag when no prefix-matching tag exists."""
        with patch("oras.client.OrasClient"):
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        available = ["iclr-2024_other-model_0.4.0"]
        with patch.object(client._client, "get_tags", return_value=available):
            result = client._find_best_matching_tag("neurips-2024_my-model_0.5.0")

        assert result == "neurips-2024_my-model_0.5.0"

    def test_find_best_matching_tag_list_failure_returns_original(self):
        """Returns the original tag when listing tags fails."""
        with patch("oras.client.OrasClient"):
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        with patch.object(client._client, "get_tags", side_effect=Exception("network error")):
            result = client._find_best_matching_tag("neurips-2024_my-model_0.5.0")

        assert result == "neurips-2024_my-model_0.5.0"

    def test_find_best_matching_tag_model_with_underscore(self):
        """Correctly resolves a tag where the model name contains underscores."""
        with patch("oras.client.OrasClient"):
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        available = ["neurips-2024_my_model_name_0.4.0", "neurips-2024_my_model_name_0.4.1"]
        with patch.object(client._client, "get_tags", return_value=available):
            result = client._find_best_matching_tag("neurips-2024_my_model_name_0.5.0")

        assert result == "neurips-2024_my_model_name_0.4.1"

    def test_download_resolves_version_mismatch_before_pull(self, tmp_path):
        """download() resolves the tag version before pulling when exact tag is not in registry."""
        set_test_db(tmp_path / "target.db")

        paper_db_path = tmp_path / "papers-2024.db"
        paper_db_path.touch()
        embeddings_path = tmp_path / "embeddings-2024.json"
        embeddings_path.write_text(json.dumps({"ids": [], "documents": [], "metadatas": [], "embeddings": []}))

        available_tags = ["neurips-2024_my-model_0.4.1"]
        pulled_files = [str(paper_db_path), str(embeddings_path)]

        with patch("oras.client.OrasClient"):
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        mock_pull = MagicMock(return_value=pulled_files)
        mock_manifest = MagicMock(return_value=None)
        with patch.object(client._client, "get_tags", return_value=available_tags):
            with patch.object(client, "_get_manifest_embedding_model", mock_manifest):
                with patch.object(client._client, "pull", mock_pull):
                    with patch.object(client, "_import_year", return_value={"paper_count": 0, "embedding_count": 0}):
                        client.download(
                            conference="neurips",
                            year=2024,
                            tag="neurips-2024_my-model_0.5.0",
                            embedding_model="my-model",
                        )

        # pull() must have been called with the resolved tag (0.4.1), not the original (0.5.0)
        pull_target = mock_pull.call_args.kwargs.get("target") or mock_pull.call_args.args[0]
        assert "0.4.1" in pull_target
        assert "0.5.0" not in pull_target

        # _get_manifest_embedding_model must also have been called with the resolved tag
        manifest_target = mock_manifest.call_args.args[0]
        assert "0.4.1" in manifest_target
        assert "0.5.0" not in manifest_target

    def test_download_raises_mismatch_from_manifest_before_pull(self, tmp_path):
        """download() raises EmbeddingModelMismatchError from manifest labels BEFORE pulling any data."""
        set_test_db(tmp_path / "target.db")

        with patch("oras.client.OrasClient"):
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        mock_pull = MagicMock(return_value=[])
        with patch.object(client, "_get_manifest_embedding_model", return_value="artifact-model-b"):
            with patch.object(client._client, "pull", mock_pull):
                with pytest.raises(EmbeddingModelMismatchError) as exc_info:
                    client.download(
                        conference="neurips",
                        year=2024,
                        tag="neurips-2024_model-a",
                        embedding_model="configured-model-a",
                    )

        # pull() must NOT have been called — mismatch detected from manifest before download
        mock_pull.assert_not_called()
        assert exc_info.value.local_model == "configured-model-a"
        assert exc_info.value.remote_model == "artifact-model-b"

    def test_download_raises_mismatch_when_artifact_model_differs(self, tmp_path):
        """download() raises EmbeddingModelMismatchError when artifact embedding model differs from configured."""
        set_test_db(tmp_path / "target.db")

        with patch("oras.client.OrasClient"):
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        with patch.object(client, "_get_manifest_embedding_model", return_value="artifact-model-b"):
            with pytest.raises(EmbeddingModelMismatchError) as exc_info:
                client.download(
                    conference="neurips",
                    year=2024,
                    tag="neurips-2024_model-a",
                    embedding_model="configured-model-a",
                )

        assert exc_info.value.local_model == "configured-model-a"
        assert exc_info.value.remote_model == "artifact-model-b"

    def test_download_ignore_mismatch_updates_embedding_model(self, tmp_path):
        """download() updates local embedding model metadata when ignore_embedding_model_mismatch=True."""
        set_test_db(tmp_path / "target.db")

        paper_db_path = tmp_path / "papers-2024.db"
        paper_db_path.touch()
        embeddings_path = tmp_path / "embeddings-2024.json"
        embeddings_path.write_text(json.dumps({"ids": [], "documents": [], "metadatas": [], "embeddings": []}))

        pulled_files = [str(paper_db_path), str(embeddings_path)]

        with patch("oras.client.OrasClient"):
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        with patch.object(client, "_get_manifest_embedding_model", return_value="artifact-model-b"):
            with patch.object(client._client, "pull", return_value=pulled_files):
                with patch.object(client, "_import_year", return_value={"paper_count": 0, "embedding_count": 0}):
                    with patch("abstracts_explorer.database.DatabaseManager.set_embedding_model") as mock_set_model:
                        client.download(
                            conference="neurips",
                            year=2024,
                            tag="neurips-2024_model-a",
                            embedding_model="configured-model-a",
                            ignore_embedding_model_mismatch=True,
                        )

        mock_set_model.assert_called_once_with("configured-model-a")

    def test_download_no_mismatch_check_when_no_manifest_model(self, tmp_path):
        """download() skips model check when manifest has no embedding model label (legacy artifact)."""
        set_test_db(tmp_path / "target.db")

        paper_db_path = tmp_path / "papers-2024.db"
        paper_db_path.touch()
        embeddings_path = tmp_path / "embeddings-2024.json"
        embeddings_path.write_text(json.dumps({"ids": [], "documents": [], "metadatas": [], "embeddings": []}))

        pulled_files = [str(paper_db_path), str(embeddings_path)]

        with patch("oras.client.OrasClient"):
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        with patch.object(client, "_get_manifest_embedding_model", return_value=None):
            with patch.object(client._client, "pull", return_value=pulled_files):
                with patch.object(client, "_import_year", return_value={"paper_count": 0, "embedding_count": 0}):
                    # Should not raise even though configured model differs from artifact (no manifest model label)
                    result = client.download(
                        conference="neurips",
                        year=2024,
                        tag="neurips-2024_model-a",
                        embedding_model="configured-model-a",
                    )

        assert result["paper_count"] == 0

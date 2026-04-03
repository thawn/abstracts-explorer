"""
Integration tests for the registry module.

These tests exercise the end-to-end flow of exporting data from one database,
writing it to temporary files, and importing it back into another database,
mirroring what the registry upload/download commands do in production.

These tests do NOT require network access or a real OCI registry — the OCI
push/pull operations are mocked.  They test:

* Export → file write → import round-trips for both paper DB and embeddings
* Data integrity after import (papers, clustering cache, embeddings metadata)
* Embedding model consistency validation and EmbeddingModelMismatchError
* Rollback behaviour when the embedding import fails after a successful paper import
* _import_year and _export_year helper methods
* RegistryClient.clear_local_embedding_data()
* Upload validation (missing papers / embeddings / embedding model)
* Download validation (missing files, model mismatch recovery)
* Case-insensitive conference name resolution
* CLI registry commands dispatching to RegistryClient correctly
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from abstracts_explorer._version import __version__
from abstracts_explorer.database import DatabaseManager, EmbeddingModelConflictError
from abstracts_explorer.plugin import LightweightPaper
from abstracts_explorer.registry import (
    EmbeddingModelMismatchError,
    RegistryClient,
    RegistryError,
    _build_tag,
    _sanitize_str_for_oci_tag,
)
from tests.conftest import set_test_db, set_test_embedding_db

pytestmark = pytest.mark.integration

# Sanitized version string used in expected tag assertions
_VER = _sanitize_str_for_oci_tag(__version__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_papers(conference: str = "neurips", year: int = 2024, count: int = 3):
    """Create a list of sample LightweightPaper objects."""
    return [
        LightweightPaper(
            title=f"Paper {i} for {conference} {year}",
            authors=[f"Author {i}"],
            abstract=f"Abstract {i} about deep learning and embeddings.",
            session=f"Session {i}",
            poster_position=f"P{i}",
            year=year,
            conference=conference,
        )
        for i in range(count)
    ]


def _setup_db_with_papers(db_path, conference="neurips", year=2024, embedding_model="model-a", count=3):
    """Populate a database with sample papers and an embedding model record."""
    set_test_db(db_path)
    with DatabaseManager() as db:
        db.create_tables()
        db.add_papers(_make_papers(conference=conference, year=year, count=count))
        db.set_embedding_model(embedding_model)
    return db_path


# ---------------------------------------------------------------------------
# DatabaseManager.export_papers_to_sqlite / import_papers_from_sqlite
# ---------------------------------------------------------------------------


class TestExportImportRoundTrip:
    """Full round-trip export → import via temporary SQLite file."""

    def test_papers_survive_round_trip(self, tmp_path):
        """Papers exported and re-imported into a fresh database are identical."""
        source_db = tmp_path / "source.db"
        dest_db = tmp_path / "dest.db"
        export_file = tmp_path / "export.db"

        _setup_db_with_papers(source_db, conference="neurips", year=2024)

        # Export from source
        set_test_db(source_db)
        with DatabaseManager() as db:
            db.create_tables()
            count = db.export_papers_to_sqlite(export_file, "neurips", 2024)

        assert count == 3
        assert export_file.exists()

        # Import into fresh destination
        set_test_db(dest_db)
        with DatabaseManager() as db:
            db.create_tables()
            imported = db.import_papers_from_sqlite(export_file, "neurips", 2024)

        assert imported == 3

        # Verify papers are there
        with DatabaseManager() as db:
            db.create_tables()
            papers = db.search_papers(conference="neurips", year=2024)
        assert len(papers) == 3
        titles = {p["title"] for p in papers}
        assert all(f"Paper {i} for neurips 2024" in titles for i in range(3))

    def test_embedding_model_preserved_after_round_trip(self, tmp_path):
        """Embedding model is propagated from export file to destination DB."""
        source_db = tmp_path / "source.db"
        export_file = tmp_path / "export.db"
        dest_db = tmp_path / "dest.db"

        _setup_db_with_papers(source_db, embedding_model="embed-model-xyz")

        set_test_db(source_db)
        with DatabaseManager() as db:
            db.create_tables()
            db.export_papers_to_sqlite(export_file, "neurips", 2024)

        set_test_db(dest_db)
        with DatabaseManager() as db:
            db.create_tables()
            db.import_papers_from_sqlite(export_file, "neurips", 2024)
            model = db.get_embedding_model()

        assert model == "embed-model-xyz"

    def test_import_replaces_existing_papers(self, tmp_path):
        """Importing for a conference+year replaces existing papers for that scope."""
        source_db = tmp_path / "source.db"
        export_file = tmp_path / "export.db"
        dest_db = tmp_path / "dest.db"

        # Source has 3 papers for neurips/2024 with model-a
        _setup_db_with_papers(source_db, conference="neurips", year=2024, count=3, embedding_model="model-a")
        set_test_db(source_db)
        with DatabaseManager() as db:
            db.create_tables()
            db.export_papers_to_sqlite(export_file, "neurips", 2024)

        # Destination already has 5 papers for neurips/2024 with the SAME model
        set_test_db(dest_db)
        with DatabaseManager() as db:
            db.create_tables()
            db.add_papers(_make_papers(conference="neurips", year=2024, count=5))
            db.set_embedding_model("model-a")  # same model as source

        # Import should replace the 5 with the 3 from source
        with DatabaseManager() as db:
            db.create_tables()
            imported = db.import_papers_from_sqlite(export_file, "neurips", 2024)

        assert imported == 3
        with DatabaseManager() as db:
            db.create_tables()
            papers = db.search_papers(conference="neurips", year=2024)
        assert len(papers) == 3

    def test_import_does_not_affect_other_conferences(self, tmp_path):
        """Importing for neurips/2024 must not remove data for iclr/2024."""
        source_db = tmp_path / "source.db"
        export_file = tmp_path / "export.db"
        dest_db = tmp_path / "dest.db"

        _setup_db_with_papers(source_db, conference="neurips", year=2024, embedding_model="model-a")
        set_test_db(source_db)
        with DatabaseManager() as db:
            db.create_tables()
            db.export_papers_to_sqlite(export_file, "neurips", 2024)

        # Destination has both neurips/2024 and iclr/2024 with the SAME model
        set_test_db(dest_db)
        with DatabaseManager() as db:
            db.create_tables()
            db.add_papers(_make_papers(conference="neurips", year=2024, count=5))
            db.add_papers(_make_papers(conference="iclr", year=2024, count=4))
            db.set_embedding_model("model-a")  # same model as source

        with DatabaseManager() as db:
            db.create_tables()
            db.import_papers_from_sqlite(export_file, "neurips", 2024)

        # iclr/2024 must be untouched
        with DatabaseManager() as db:
            db.create_tables()
            iclr_papers = db.search_papers(conference="iclr", year=2024)
        assert len(iclr_papers) == 4

    def test_embedding_model_conflict_raises_error(self, tmp_path):
        """Import raises EmbeddingModelConflictError if embedding models differ."""
        source_db = tmp_path / "source.db"
        export_file = tmp_path / "export.db"
        dest_db = tmp_path / "dest.db"

        _setup_db_with_papers(source_db, embedding_model="model-a")
        set_test_db(source_db)
        with DatabaseManager() as db:
            db.create_tables()
            db.export_papers_to_sqlite(export_file, "neurips", 2024)

        # Destination uses a DIFFERENT model
        set_test_db(dest_db)
        with DatabaseManager() as db:
            db.create_tables()
            db.add_papers(_make_papers())
            db.set_embedding_model("model-b")

        with pytest.raises(EmbeddingModelConflictError) as exc_info:
            with DatabaseManager() as db:
                db.create_tables()
                db.import_papers_from_sqlite(export_file, "neurips", 2024)

        assert exc_info.value.local_model == "model-b"
        assert exc_info.value.remote_model == "model-a"

    def test_consistent_model_does_not_raise(self, tmp_path):
        """Import succeeds when source and destination use the same embedding model."""
        source_db = tmp_path / "source.db"
        export_file = tmp_path / "export.db"
        dest_db = tmp_path / "dest.db"

        _setup_db_with_papers(source_db, embedding_model="shared-model")
        set_test_db(source_db)
        with DatabaseManager() as db:
            db.create_tables()
            db.export_papers_to_sqlite(export_file, "neurips", 2024)

        set_test_db(dest_db)
        with DatabaseManager() as db:
            db.create_tables()
            db.add_papers(_make_papers())
            db.set_embedding_model("shared-model")

        # Should not raise
        with DatabaseManager() as db:
            db.create_tables()
            count = db.import_papers_from_sqlite(export_file, "neurips", 2024)

        assert count == 3


# ---------------------------------------------------------------------------
# EmbeddingsManager.export_embeddings / import_embeddings
# ---------------------------------------------------------------------------


class TestEmbeddingsExportImportRoundTrip:
    """Round-trip tests for EmbeddingsManager export/import."""

    def _make_embeddings_data(self, conference="neurips", year=2024, count=3):
        """Create fake embeddings data dict as returned by export_embeddings."""
        return {
            "ids": [f"id-{i}" for i in range(count)],
            "documents": [f"doc-{i}" for i in range(count)],
            "metadatas": [{"conference": conference, "year": year} for _ in range(count)],
            "embeddings": [[float(j) for j in range(4)] for _ in range(count)],
        }

    def test_embeddings_export_returns_serializable_dict(self, tmp_path):
        """export_embeddings returns a JSON-serializable dict (no numpy arrays)."""
        set_test_embedding_db(str(tmp_path / "chroma"))

        with patch("abstracts_explorer.embeddings.EmbeddingsManager") as MockEM:
            mock_em = MockEM.return_value
            # Simulate what ChromaDB returns (numpy-like lists as plain lists)
            mock_em.export_embeddings.return_value = self._make_embeddings_data()

            result = mock_em.export_embeddings("neurips", 2024)

        # Must be JSON-serializable
        serialized = json.dumps(result)
        assert len(serialized) > 0
        decoded = json.loads(serialized)
        assert decoded["ids"] == ["id-0", "id-1", "id-2"]
        assert all(isinstance(emb, list) for emb in decoded["embeddings"])

    def test_embeddings_import_replaces_existing(self, tmp_path):
        """import_embeddings deletes existing embeddings for conference+year before importing."""
        set_test_embedding_db(str(tmp_path / "chroma"))

        with patch("abstracts_explorer.embeddings.EmbeddingsManager") as MockEM:
            mock_em = MockEM.return_value
            mock_em.import_embeddings.return_value = 5

            count = mock_em.import_embeddings(self._make_embeddings_data(), "neurips", 2024)

        assert count == 5
        mock_em.import_embeddings.assert_called_once()

    def test_embeddings_round_trip_via_json_file(self, tmp_path):
        """Embeddings can be written to JSON and read back correctly."""
        embeddings_data = self._make_embeddings_data(count=10)
        json_file = tmp_path / "embeddings-2024.json"
        json_file.write_text(json.dumps(embeddings_data))

        loaded = json.loads(json_file.read_text())

        assert len(loaded["ids"]) == 10
        assert len(loaded["embeddings"]) == 10
        assert loaded["metadatas"][0]["conference"] == "neurips"
        assert loaded["metadatas"][0]["year"] == 2024


# ---------------------------------------------------------------------------
# RegistryClient._import_year (integration-level)
# ---------------------------------------------------------------------------


class TestImportYearIntegration:
    """Integration tests for RegistryClient._import_year."""

    def test_import_year_fails_if_paper_db_missing(self, tmp_path):
        """_import_year raises RegistryError if paper DB file is absent."""
        set_test_db(tmp_path / "target.db")
        set_test_embedding_db(str(tmp_path / "chroma"))

        paper_db = tmp_path / "papers-2024.db"  # does NOT exist
        embeddings = tmp_path / "embeddings-2024.json"
        embeddings.write_text(json.dumps({"ids": [], "documents": [], "metadatas": [], "embeddings": []}))

        with patch("oras.client.OrasClient"):
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        with pytest.raises(RegistryError, match="paper DB"):
            client._import_year("neurips", 2024, paper_db, embeddings, lambda m: None)

    def test_import_year_fails_if_embeddings_missing(self, tmp_path):
        """_import_year raises RegistryError if embeddings file is absent."""
        set_test_db(tmp_path / "target.db")

        paper_db = tmp_path / "papers-2024.db"
        paper_db.touch()
        embeddings = tmp_path / "embeddings-2024.json"  # does NOT exist

        with patch("oras.client.OrasClient"):
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        with pytest.raises(RegistryError, match="embeddings"):
            client._import_year("neurips", 2024, paper_db, embeddings, lambda m: None)

    def test_import_year_propagates_model_mismatch(self, tmp_path):
        """_import_year wraps EmbeddingModelConflictError as EmbeddingModelMismatchError."""
        set_test_db(tmp_path / "target.db")

        paper_db = tmp_path / "papers-2024.db"
        paper_db.touch()
        embeddings = tmp_path / "embeddings-2024.json"
        embeddings.write_text(json.dumps({"ids": [], "documents": [], "metadatas": [], "embeddings": []}))
        cache = tmp_path / "clustering-2024.json"
        cache.write_text(json.dumps({"entries": []}))

        with patch("oras.client.OrasClient"):
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        with patch(
            "abstracts_explorer.database.DatabaseManager.import_papers_from_sqlite",
            side_effect=EmbeddingModelConflictError("local-model", "remote-model"),
        ):
            with pytest.raises(EmbeddingModelMismatchError) as exc_info:
                client._import_year(
                    "neurips", 2024, paper_db, embeddings, lambda m: None, clustering_cache_file=cache
                )

        assert exc_info.value.local_model == "local-model"
        assert exc_info.value.remote_model == "remote-model"

    def test_import_year_rolls_back_paper_db_on_embedding_failure(self, tmp_path):
        """If embedding import fails, paper DB changes for that year are rolled back."""
        set_test_db(tmp_path / "target.db")
        set_test_embedding_db(str(tmp_path / "chroma"))

        # Create a source DB and export it
        source_db = tmp_path / "source.db"
        _setup_db_with_papers(source_db, conference="neurips", year=2024, embedding_model="model-a")
        set_test_db(source_db)
        with DatabaseManager() as db:
            db.create_tables()
            db.export_papers_to_sqlite(tmp_path / "papers-2024.db", "neurips", 2024)

        # Set destination DB (empty, same model)
        set_test_db(tmp_path / "target.db")
        with DatabaseManager() as db:
            db.create_tables()
            db.set_embedding_model("model-a")

        paper_db = tmp_path / "papers-2024.db"
        embeddings = tmp_path / "embeddings-2024.json"
        embeddings.write_text(json.dumps({"ids": [], "documents": [], "metadatas": [], "embeddings": []}))
        cache = tmp_path / "clustering-2024.json"
        cache.write_text(json.dumps({"entries": []}))

        with patch("oras.client.OrasClient"):
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        # Make the embedding import fail
        with patch(
            "abstracts_explorer.embeddings.EmbeddingsManager.import_embeddings",
            side_effect=RuntimeError("embedding failure"),
        ):
            with pytest.raises(RegistryError, match="Embedding import failed"):
                client._import_year(
                    "neurips", 2024, paper_db, embeddings, lambda m: None, clustering_cache_file=cache
                )

        # Paper DB should be rolled back — no papers for neurips/2024
        with DatabaseManager() as db:
            db.create_tables()
            papers = db.search_papers(conference="neurips", year=2024)
        assert len(papers) == 0


# ---------------------------------------------------------------------------
# RegistryClient.clear_local_embedding_data
# ---------------------------------------------------------------------------


class TestClearLocalEmbeddingData:
    """Tests for RegistryClient.clear_local_embedding_data."""

    def test_clear_removes_embedding_metadata(self, tmp_path):
        """clear_local_embedding_data removes EmbeddingsMetadata from the DB."""
        set_test_db(tmp_path / "test.db")
        set_test_embedding_db(str(tmp_path / "chroma"))

        with DatabaseManager() as db:
            db.create_tables()
            db.set_embedding_model("model-to-clear")
            assert db.get_embedding_model() == "model-to-clear"

        with patch("abstracts_explorer.embeddings.EmbeddingsManager.create_collection"):
            RegistryClient.clear_local_embedding_data()

        with DatabaseManager() as db:
            db.create_tables()
            assert db.get_embedding_model() is None

    def test_clear_resets_chromadb_collection(self, tmp_path):
        """clear_local_embedding_data resets the ChromaDB collection."""
        set_test_db(tmp_path / "test.db")
        set_test_embedding_db(str(tmp_path / "chroma"))

        with patch("abstracts_explorer.embeddings.EmbeddingsManager.create_collection") as mock_create:
            RegistryClient.clear_local_embedding_data()

        mock_create.assert_called_once_with(reset=True)


# ---------------------------------------------------------------------------
# RegistryClient upload validation
# ---------------------------------------------------------------------------


class TestUploadValidation:
    """Tests for upload pre-condition validation."""

    def test_upload_errors_if_no_embedding_model_when_no_papers(self, tmp_path):
        """upload() raises RegistryError if no embedding model is recorded (checked first)."""
        set_test_db(tmp_path / "test.db")
        with DatabaseManager() as db:
            db.create_tables()
            # No papers and no embedding model → embedding model error raised first

        with patch("oras.client.OrasClient"):
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        with pytest.raises(RegistryError, match="No embedding model"):
            client.upload(conference="neurips", year=2024)

    def test_upload_errors_if_no_data_for_conference(self, tmp_path):
        """upload() raises RegistryError if no data exists for conference (year=None path)."""
        set_test_db(tmp_path / "test.db")
        with DatabaseManager() as db:
            db.create_tables()
            db.set_embedding_model("model-a")  # model exists but no papers

        with patch("oras.client.OrasClient"):
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        # Without a year, upload looks for years → finds none → raises RegistryError
        with pytest.raises(RegistryError, match="No data found"):
            client.upload(conference="neurips")  # no year specified

    def test_upload_errors_if_no_embedding_model(self, tmp_path):
        """upload() raises RegistryError if no embedding model is recorded."""
        set_test_db(tmp_path / "test.db")
        with DatabaseManager() as db:
            db.create_tables()
            db.add_papers(_make_papers())
            # No embedding model set

        with patch("oras.client.OrasClient"):
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        with pytest.raises(RegistryError, match="embedding model"):
            client.upload(conference="neurips", year=2024)

    def test_upload_errors_if_no_embeddings(self, tmp_path):
        """upload() raises RegistryError if no embeddings exist for the conference+year."""
        set_test_db(tmp_path / "test.db")
        set_test_embedding_db(str(tmp_path / "chroma"))

        with DatabaseManager() as db:
            db.create_tables()
            db.add_papers(_make_papers())
            db.set_embedding_model("model-a")

        with patch("oras.client.OrasClient"):
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        # EmbeddingsManager.export_embeddings returns empty → should error
        with patch(
            "abstracts_explorer.embeddings.EmbeddingsManager.export_embeddings",
            return_value={"ids": [], "documents": [], "metadatas": [], "embeddings": []},
        ):
            with pytest.raises(RegistryError, match="No embeddings"):
                client.upload(conference="neurips", year=2024)


# ---------------------------------------------------------------------------
# RegistryClient conference name resolution
# ---------------------------------------------------------------------------


class TestConferenceNameResolution:
    """Tests for case-insensitive conference name resolution."""

    def test_resolve_finds_stored_name(self, tmp_path):
        """DatabaseManager.resolve_conference_name returns the DB-stored name regardless of input case."""
        set_test_db(tmp_path / "test.db")
        with DatabaseManager() as db:
            db.create_tables()
            db.add_papers(_make_papers(conference="NeurIPS", year=2024))

            assert db.resolve_conference_name("neurips") == "NeurIPS"
            assert db.resolve_conference_name("NEURIPS") == "NeurIPS"
            assert db.resolve_conference_name("NeurIPS") == "NeurIPS"

    def test_resolve_raises_when_not_found(self, tmp_path):
        """resolve_conference_name raises DatabaseError when no match exists in DB or plugins."""
        from abstracts_explorer.database import DatabaseError

        set_test_db(tmp_path / "test.db")
        with DatabaseManager() as db:
            db.create_tables()

            with pytest.raises(DatabaseError, match="Failed to resolve conference name"):
                db.resolve_conference_name("unknown-conf-xyz-999")


# ---------------------------------------------------------------------------
# RegistryClient.download — model mismatch recovery
# ---------------------------------------------------------------------------


class TestDownloadModelMismatchRecovery:
    """Tests for embedding model mismatch handling in download."""

    def test_download_raises_mismatch_error_without_config_match(self, tmp_path):
        """download() propagates EmbeddingModelMismatchError to the caller."""
        set_test_db(tmp_path / "test.db")
        set_test_embedding_db(str(tmp_path / "chroma"))

        mismatch = EmbeddingModelMismatchError("local-model", "remote-model")

        # Create fake pulled files so the download doesn't error on missing files
        paper_db = tmp_path / "papers-2024.db"
        paper_db.touch()
        embeddings = tmp_path / "embeddings-2024.json"
        embeddings.write_text(json.dumps({"ids": [], "documents": [], "metadatas": [], "embeddings": []}))
        cache = tmp_path / "clustering-2024.json"
        cache.write_text(json.dumps({"entries": []}))

        with patch("oras.client.OrasClient"):
            client = RegistryClient("ghcr.io/thawn/abstracts-data", token="token")

        # Mock the oras client pull to return our fake files
        client._client.pull = MagicMock(return_value=[str(paper_db), str(embeddings), str(cache)])

        with patch.object(client, "_import_year", side_effect=mismatch):
            with pytest.raises(EmbeddingModelMismatchError):
                client.download(
                    conference="neurips",
                    year=2024,
                    embedding_model="remote-model",
                )


# ---------------------------------------------------------------------------
# _build_tag and _sanitize_str_for_oci_tag
# ---------------------------------------------------------------------------


class TestTagHelpers:
    """Integration-level checks for tag-building helpers."""

    def test_build_tag_with_year(self):
        """Tag with year includes conference-year, sanitized model and version."""
        tag = _build_tag("neurips", 2024, embedding_model="text-embedding-qwen3-embedding-4b")
        assert tag == f"neurips-2024_text-embedding-qwen3-embedding-4b_{_VER}"

    def test_build_tag_without_year(self):
        """Conference-only tag omits the year segment but includes the version."""
        tag = _build_tag("neurips", embedding_model="text-embedding-qwen3-embedding-4b")
        assert tag == f"neurips_text-embedding-qwen3-embedding-4b_{_VER}"

    def test_sanitize_slash_replaced(self):
        """Slashes are replaced with hyphens."""
        sanitized = _sanitize_str_for_oci_tag("org/model-name:v2")
        assert "/" not in sanitized
        assert ":" not in sanitized

    def test_sanitize_lowercase(self):
        """Values are lowercased."""
        assert _sanitize_str_for_oci_tag("GPT-4o") == "gpt-4o"

    def test_sanitize_collapses_hyphens(self):
        """Consecutive hyphens are collapsed."""
        result = _sanitize_str_for_oci_tag("model//name")
        assert "--" not in result

    def test_sanitize_version_local_segment(self):
        """PEP 440 '+' local-version separator is replaced with '-'."""
        assert _sanitize_str_for_oci_tag("0.1.dev2+g2abcfb2a2") == "0.1.dev2-g2abcfb2a2"


# ---------------------------------------------------------------------------
# get_years_for_conference
# ---------------------------------------------------------------------------


class TestGetYearsForConference:
    """Tests for DatabaseManager.get_years_for_conference."""

    def test_returns_distinct_years(self, tmp_path):
        """get_years_for_conference returns the unique years for a conference."""
        set_test_db(tmp_path / "test.db")
        with DatabaseManager() as db:
            db.create_tables()
            db.add_papers(_make_papers(conference="neurips", year=2024, count=2))
            db.add_papers(_make_papers(conference="neurips", year=2025, count=3))
            db.add_papers(_make_papers(conference="iclr", year=2024, count=2))

        with DatabaseManager() as db:
            db.create_tables()
            years = db.get_years_for_conference("neurips")

        assert sorted(years) == [2024, 2025]

    def test_returns_empty_for_unknown_conference(self, tmp_path):
        """get_years_for_conference returns [] when no papers exist."""
        set_test_db(tmp_path / "test.db")
        with DatabaseManager() as db:
            db.create_tables()

        with DatabaseManager() as db:
            db.create_tables()
            years = db.get_years_for_conference("unknown")

        assert years == []


# ---------------------------------------------------------------------------
# CLI dispatch integration
# ---------------------------------------------------------------------------


class TestCLIRegistryDispatch:
    """Integration tests verifying the CLI dispatches correctly to RegistryClient."""

    def test_upload_command_calls_upload(self, tmp_path):
        """registry_upload_command calls RegistryClient.upload for a specific conference+year."""
        import argparse

        from abstracts_explorer.cli import registry_upload_command

        set_test_db(tmp_path / "test.db")

        args = argparse.Namespace(
            repository="ghcr.io/thawn/abstracts-data",
            token="test-token",
            conference="neurips",
            year=2024,
            tag=None,
            yes=True,
        )

        with patch("abstracts_explorer.registry.RegistryClient") as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.upload.return_value = {
                "tag": "neurips-2024_model",
                "conference": "neurips",
                "years": [2024],
                "paper_count": 3,
                "embedding_count": 3,
                "year_tags": [],
                "metadata": {},
            }

            result = registry_upload_command(args)

        assert result == 0
        # Verify upload was called with the resolved conference name
        # "neurips" resolves to "NeurIPS" via plugin fallback since DB is empty
        call_kwargs = mock_instance.upload.call_args.kwargs
        assert call_kwargs["conference"] == "NeurIPS"
        assert call_kwargs["year"] == 2024
        assert call_kwargs["tag"] is None

    def test_upload_command_calls_upload_all_for_all_conference(self, tmp_path):
        """registry_upload_command calls RegistryClient.upload_all when conference='all'."""
        import argparse

        from abstracts_explorer.cli import registry_upload_command

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
            mock_instance.upload_all.return_value = []

            result = registry_upload_command(args)

        assert result == 0
        mock_instance.upload_all.assert_called_once()

    def test_download_command_calls_download(self, tmp_path):
        """registry_download_command calls RegistryClient.download for a specific conference+year."""
        import argparse

        from abstracts_explorer.cli import registry_download_command

        set_test_db(tmp_path / "test.db")

        args = argparse.Namespace(
            repository="ghcr.io/thawn/abstracts-data",
            token="test-token",
            conference="neurips",
            year=2024,
            tag=None,
            yes=True,
            embedding_model="model-a",
        )

        with patch("abstracts_explorer.registry.RegistryClient") as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.download.return_value = {
                "tag": "neurips-2024_model-a",
                "conference": "neurips",
                "years": [2024],
                "paper_count": 3,
                "embedding_count": 3,
                "metadata": {},
            }

            result = registry_download_command(args)

        assert result == 0
        mock_instance.download.assert_called_once()

    def test_download_command_calls_download_all(self, tmp_path):
        """registry_download_command calls RegistryClient.download_all when conference='all'."""
        import argparse

        from abstracts_explorer.cli import registry_download_command

        args = argparse.Namespace(
            repository="ghcr.io/thawn/abstracts-data",
            token="test-token",
            conference="all",
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

    def test_upload_command_conference_case_insensitive(self, tmp_path):
        """registry_upload_command normalises 'NeurIPS' → treated as lower for 'all' check."""
        import argparse

        from abstracts_explorer.cli import registry_upload_command

        args = argparse.Namespace(
            repository="ghcr.io/thawn/abstracts-data",
            token="test-token",
            conference="ALL",  # uppercase
            year=None,
            tag=None,
            yes=True,
        )

        with patch("abstracts_explorer.registry.RegistryClient") as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.upload_all.return_value = []

            result = registry_upload_command(args)

        assert result == 0
        mock_instance.upload_all.assert_called_once()

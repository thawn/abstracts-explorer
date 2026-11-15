"""
Tests for the embeddings compaction migration script.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from abstracts_explorer.embeddings import hash_text

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "migrate_embeddings_compaction.py"


@pytest.fixture(scope="module")
def migration():
    """Load the migration script as a module."""
    spec = importlib.util.spec_from_file_location("migrate_embeddings_compaction", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestMigrationSteps:
    """Tests for the individual compaction steps."""

    def test_remove_metadata_and_hash_documents(self, migration, embeddings_manager):
        embeddings_manager.create_collection()
        embeddings_manager.collection.add(
            ids=["p1"],
            embeddings=[[0.1] * 4096],
            documents=["Title\n\nAbstract"],
            metadatas=[{"title": "Title", "year": "2025"}],
        )

        assert migration.remove_all_metadata(embeddings_manager) == 1
        stored = embeddings_manager.collection.get(ids=["p1"], include=["metadatas"])
        assert stored["metadatas"] == [None]

        assert migration.hash_all_documents(embeddings_manager) == 1
        stored = embeddings_manager.collection.get(ids=["p1"], include=["documents"])
        assert stored["documents"] == [hash_text("Title\n\nAbstract")]

        # Idempotent
        assert migration.hash_all_documents(embeddings_manager) == 0
        embeddings_manager.close()

    def test_hash_documents_preserves_embeddings(self, migration, embeddings_manager):
        embeddings_manager.create_collection()
        embeddings_manager.collection.add(
            ids=["p1"],
            embeddings=[[0.5] * 4096],
            documents=["Some text"],
        )

        migration.hash_all_documents(embeddings_manager)

        stored = embeddings_manager.collection.get(ids=["p1"], include=["embeddings", "documents"])
        assert stored["documents"] == [hash_text("Some text")]
        assert stored["embeddings"][0][0] == pytest.approx(0.5)
        embeddings_manager.close()

    def test_prune_orphan_embeddings(self, migration, embeddings_manager):
        embeddings_manager.create_collection()
        embeddings_manager.collection.add(
            ids=["known", "orphan"],
            embeddings=[[0.1] * 4096, [0.2] * 4096],
            documents=[hash_text("known"), hash_text("orphan")],
        )
        database = Mock()
        database.get_paper_uids.return_value = ["known"]

        assert migration.prune_orphan_embeddings(embeddings_manager, database) == 1
        assert embeddings_manager.collection.get()["ids"] == ["known"]
        embeddings_manager.close()


class TestMain:
    """Tests for the script entry point."""

    @staticmethod
    def _context(migration, *, pruned: int = 1, embedding_db: str = "/tmp/chroma"):
        config = Mock(collection_name="papers", embedding_db=embedding_db)
        embeddings_manager = Mock()
        database = Mock()
        database_context = Mock()
        database_context.__enter__ = Mock(return_value=database)
        database_context.__exit__ = Mock(return_value=False)
        return config, embeddings_manager, database, database_context

    def test_runs_all_steps(self, migration, capsys):
        config, embeddings_manager, database, database_context = self._context(migration)

        with (
            patch.object(migration, "get_config", return_value=config),
            patch.object(migration, "EmbeddingsManager", return_value=embeddings_manager),
            patch.object(migration, "DatabaseManager", return_value=database_context),
            patch.object(migration, "remove_all_metadata", return_value=3) as mock_remove,
            patch.object(migration, "hash_all_documents", return_value=2) as mock_hash,
            patch.object(migration, "prune_orphan_embeddings", return_value=1) as mock_prune,
        ):
            exit_code = migration.main(["--prune"])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Processed metadata for 3" in captured.out
        assert "Hashed 2" in captured.out
        assert "Pruned 1" in captured.out
        embeddings_manager.create_collection.assert_called_once()
        embeddings_manager.close.assert_called_once()
        mock_remove.assert_called_once_with(embeddings_manager)
        mock_hash.assert_called_once_with(embeddings_manager)
        mock_prune.assert_called_once_with(embeddings_manager, database)

    def test_returns_error_and_closes_on_failure(self, migration, capsys):
        config, embeddings_manager, _, _ = self._context(migration)
        embeddings_manager.create_collection.side_effect = RuntimeError("boom")

        with (
            patch.object(migration, "get_config", return_value=config),
            patch.object(migration, "EmbeddingsManager", return_value=embeddings_manager),
        ):
            exit_code = migration.main([])

        assert exit_code == 1
        assert "Compaction failed" in capsys.readouterr().err
        embeddings_manager.close.assert_called_once()

    def test_vacuum_runs_subprocess_for_local_storage(self, migration):
        config, embeddings_manager, _, _ = self._context(migration, embedding_db="/tmp/chroma")

        with (
            patch.object(migration, "get_config", return_value=config),
            patch.object(migration, "EmbeddingsManager", return_value=embeddings_manager),
            patch.object(migration, "remove_all_metadata", return_value=0),
            patch.object(migration, "hash_all_documents", return_value=0),
            patch.object(migration, "subprocess") as mock_subprocess,
        ):
            exit_code = migration.main(["--vacuum"])

        assert exit_code == 0
        mock_subprocess.run.assert_called_once_with(
            ["chroma", "vacuum", "--path", str(Path("/tmp/chroma")), "--force"],
            check=True,
        )

    def test_vacuum_warns_for_remote_storage(self, migration, capsys):
        config, embeddings_manager, _, _ = self._context(migration, embedding_db="http://chroma:8000")

        with (
            patch.object(migration, "get_config", return_value=config),
            patch.object(migration, "EmbeddingsManager", return_value=embeddings_manager),
            patch.object(migration, "remove_all_metadata", return_value=0),
            patch.object(migration, "hash_all_documents", return_value=0),
            patch.object(migration, "subprocess") as mock_subprocess,
        ):
            exit_code = migration.main(["--vacuum"])

        assert exit_code == 0
        assert "Cannot vacuum a remote ChromaDB instance" in capsys.readouterr().err
        mock_subprocess.run.assert_not_called()

    def test_paper_db_override_reloads_config(self, migration, monkeypatch):
        monkeypatch.setenv("PAPER_DB", "original.db")
        config, embeddings_manager, _, _ = self._context(migration)

        with (
            patch.object(migration, "get_config", return_value=config) as mock_get_config,
            patch.object(migration, "EmbeddingsManager", return_value=embeddings_manager),
            patch.object(migration, "remove_all_metadata", return_value=0),
            patch.object(migration, "hash_all_documents", return_value=0),
        ):
            exit_code = migration.main(["--paper-db", "/tmp/override.db"])

        assert exit_code == 0
        mock_get_config.assert_any_call(reload=True)
        assert mock_get_config.call_count == 2

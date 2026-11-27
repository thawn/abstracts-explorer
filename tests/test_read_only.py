"""Tests for read-only mode across configuration, database, and embeddings."""

import argparse

import pytest

from abstracts_explorer.config import get_config
from abstracts_explorer.database import DatabaseError, DatabaseManager, ReadOnlyError
from abstracts_explorer.embeddings import EmbeddingsError, EmbeddingsManager
from abstracts_explorer.embeddings import locking
from abstracts_explorer.plugin import LightweightPaper
from abstracts_explorer.utils.read_only import require_writable
from tests.conftest import get_env_test_path, set_test_db


def _enable_read_only(monkeypatch) -> None:
    """Set READ_ONLY and reload the test configuration."""
    monkeypatch.setenv("READ_ONLY", "true")
    get_config(reload=True, env_path=get_env_test_path())


@pytest.fixture(autouse=True)
def _restore_config(monkeypatch):
    """Restore the shared configuration after each read-only test."""
    yield
    monkeypatch.delenv("READ_ONLY", raising=False)
    get_config(reload=True, env_path=get_env_test_path())


class _WritableProbe:
    """Small helper class for testing the decorator."""

    read_only = True

    @require_writable("write the probe")
    def write(self) -> str:
        return "written"


class TestRequireWritable:
    """Tests for the shared decorator."""

    def test_raises_in_read_only_mode(self):
        with pytest.raises(ReadOnlyError, match="write the probe"):
            _WritableProbe().write()

    def test_allows_writes_when_enabled(self):
        probe = _WritableProbe()
        probe.read_only = False
        assert probe.write() == "written"


class TestDatabaseReadOnly:
    """Tests for DatabaseManager in read-only mode."""

    def test_reads_work_and_writes_raise(self, test_database, monkeypatch):
        _enable_read_only(monkeypatch)
        db = DatabaseManager()
        try:
            assert db.read_only is True
            assert db.get_paper_count() == 3
            paper = LightweightPaper(
                title="Blocked Paper",
                authors=["A"],
                abstract="Blocked abstract",
                session="Blocked session",
                poster_position="P0",
                year=2025,
                conference="NeurIPS",
            )
            with pytest.raises(ReadOnlyError):
                db.add_paper(paper)
            with pytest.raises(ReadOnlyError):
                db.add_papers([paper])
            with pytest.raises(ReadOnlyError):
                db.set_embedding_model("some-model")
            with pytest.raises(ReadOnlyError):
                db.delete_papers_by_conference_year("NeurIPS", 2025)
            # Existing schemas may be verified but not created.
            db.create_tables()
        finally:
            db.close()

    def test_missing_database_is_rejected(self, tmp_path, monkeypatch):
        db_path = tmp_path / "missing.db"
        set_test_db(db_path)
        _enable_read_only(monkeypatch)
        db = DatabaseManager()
        try:
            with pytest.raises(DatabaseError, match="read-only"):
                _ = db.session
        finally:
            db.close()


class TestEmbeddingsReadOnly:
    """Tests for EmbeddingsManager in read-only mode."""

    def test_existing_collection_is_readable_and_writes_raise(self, embeddings_manager, monkeypatch):
        embeddings_manager.create_collection()
        embeddings_manager.collection.upsert(ids=["1"], embeddings=[[0.0] * 4], documents=["hash-1"])
        embeddings_manager.close()

        # Chroma caches clients per path; the read-only client uses different
        # settings, so drop the cached writer system before reconnecting.
        import chromadb.api.shared_system_client

        chromadb.api.shared_system_client.SharedSystemClient._identifier_to_system.clear()

        _enable_read_only(monkeypatch)
        em = EmbeddingsManager(collection_name="test_collection")
        try:
            assert em.read_only is True
            assert em.get_collection_stats()["count"] == 1
            with pytest.raises(ReadOnlyError):
                em.add_paper({"uid": "2", "title": "T", "abstract": "A"})
            with pytest.raises(ReadOnlyError):
                em.delete_embeddings_by_ids(["1"])
            with pytest.raises(ReadOnlyError):
                em.create_collection()
        finally:
            em.close()

    def test_missing_collection_is_rejected(self, tmp_path, monkeypatch):
        chroma_path = tmp_path / "empty_chroma"
        chroma_path.mkdir()
        monkeypatch.setenv("EMBEDDING_DB", str(chroma_path))
        _enable_read_only(monkeypatch)

        em = EmbeddingsManager(collection_name="missing")
        try:
            with pytest.raises(EmbeddingsError):
                _ = em.collection
        finally:
            em.close()

    def test_missing_local_store_is_rejected(self, tmp_path, monkeypatch):
        chroma_path = tmp_path / "does-not-exist"
        monkeypatch.setenv("EMBEDDING_DB", str(chroma_path))
        _enable_read_only(monkeypatch)

        em = EmbeddingsManager()
        try:
            with pytest.raises(EmbeddingsError, match="not found"):
                _ = em.collection
        finally:
            em.close()


class TestLocalChromaReadLock:
    """Tests for the shared read lock used by local ChromaDB."""

    def test_multiple_readers_exclude_a_writer(self, tmp_path):
        chroma_path = tmp_path / "chroma"
        chroma_path.mkdir()
        path = str(chroma_path)

        locking.acquire_local_chroma_read_lock(path)
        locking.acquire_local_chroma_read_lock(path)
        try:
            with pytest.raises(locking.LocalChromaInUseError):
                locking.acquire_local_chroma_lock(path)
        finally:
            locking.release_local_chroma_read_lock(path)
            locking.release_local_chroma_read_lock(path)

    def test_writer_excludes_readers(self, tmp_path):
        chroma_path = tmp_path / "chroma"
        chroma_path.mkdir()
        path = str(chroma_path)

        locking.acquire_local_chroma_lock(path)
        try:
            with pytest.raises(locking.LocalChromaInUseError):
                locking.acquire_local_chroma_read_lock(path)
        finally:
            locking.release_local_chroma_lock(path)


class TestCliReadOnlyRouting:
    """Tests for deciding which CLI commands write to storage."""

    @pytest.mark.parametrize(
        "args, expected",
        [
            (argparse.Namespace(_command_path=("data", "download")), True),
            (argparse.Namespace(_command_path=("embeddings", "create")), True),
            (argparse.Namespace(_command_path=("data", "delete")), True),
            (argparse.Namespace(_command_path=("papers", "search")), False),
            (argparse.Namespace(_command_path=("clustering", "run")), True),
            (argparse.Namespace(_command_path=("clustering", "cache", "clear")), True),
            (argparse.Namespace(_command_path=("eval", "verify")), True),
            (argparse.Namespace(_command_path=("eval", "list")), False),
            (argparse.Namespace(_command_path=("feedback", "clear")), True),
            (argparse.Namespace(_command_path=("feedback", "stats")), False),
            (argparse.Namespace(_command_path=("data", "check"), fix=True), True),
            (argparse.Namespace(_command_path=("data", "check"), fix=False), False),
        ],
    )
    def test_write_command_detection(self, args, expected):
        from abstracts_explorer.cli._common import _command_writes_storage

        assert _command_writes_storage(args) is expected

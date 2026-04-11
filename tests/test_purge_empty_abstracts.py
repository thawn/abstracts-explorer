"""
Tests for the purge_empty_abstracts script.
"""

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from abstracts_explorer.database import DatabaseError, DatabaseManager
from abstracts_explorer.db_models import Paper
from abstracts_explorer.plugin import LightweightPaper
from tests.conftest import set_test_db

# Import functions from the scripts directory without mutating sys.path.
_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "purge_empty_abstracts.py"
_spec = importlib.util.spec_from_file_location("purge_empty_abstracts", _SCRIPT_PATH)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
find_papers_without_abstract = _module.find_papers_without_abstract
delete_papers_by_uids = _module.delete_papers_by_uids
delete_embeddings_by_uids = _module.delete_embeddings_by_uids

# ---------------------------------------------------------------------------
# find_papers_without_abstract
# ---------------------------------------------------------------------------


class TestFindPapersWithoutAbstract:
    """Tests for find_papers_without_abstract()."""

    def test_returns_empty_when_all_have_abstracts(self, tmp_path):
        """No UIDs returned when every paper has a non-empty abstract."""
        set_test_db(tmp_path / "test.db")
        with DatabaseManager() as db:
            db.create_tables()
            paper = LightweightPaper(
                title="Paper With Abstract",
                authors=["Alice"],
                abstract="A real abstract.",
                session="S",
                poster_position="1",
                year=2025,
                conference="NeurIPS",
            )
            db.add_paper(paper)
            uids = find_papers_without_abstract(db)
        assert uids == []

    def test_finds_paper_with_null_abstract(self, tmp_path):
        """Papers with NULL abstract are returned."""
        set_test_db(tmp_path / "test.db")
        with DatabaseManager() as db:
            db.create_tables()
            # Add a paper with a real abstract first
            paper = LightweightPaper(
                title="Good Paper",
                authors=["Bob"],
                abstract="Some abstract.",
                session="S",
                poster_position="1",
                year=2025,
                conference="NeurIPS",
            )
            db.add_paper(paper)

            # Manually insert a paper row with NULL abstract
            from sqlalchemy import insert as sa_insert

            db._session.execute(
                sa_insert(Paper).values(
                    uid="deadbeef0001",
                    title="No Abstract Paper",
                    abstract=None,
                    authors="Charlie",
                    session="S",
                    poster_position="2",
                    year=2024,
                    conference="ICLR",
                )
            )
            db._session.commit()

            uids = find_papers_without_abstract(db)

        assert "deadbeef0001" in uids
        assert len(uids) == 1

    def test_finds_paper_with_empty_string_abstract(self, tmp_path):
        """Papers with empty-string abstract are returned."""
        set_test_db(tmp_path / "test.db")
        with DatabaseManager() as db:
            db.create_tables()
            from sqlalchemy import insert as sa_insert

            db._session.execute(
                sa_insert(Paper).values(
                    uid="deadbeef0002",
                    title="Empty Abstract Paper",
                    abstract="",
                    authors="Dave",
                    session="S",
                    poster_position="3",
                    year=2024,
                    conference="ICLR",
                )
            )
            db._session.commit()

            uids = find_papers_without_abstract(db)

        assert "deadbeef0002" in uids

    def test_raises_when_not_connected(self):
        """DatabaseError is raised if the session is not open."""
        db = DatabaseManager.__new__(DatabaseManager)
        db._session = None
        with pytest.raises(DatabaseError):
            find_papers_without_abstract(db)


# ---------------------------------------------------------------------------
# delete_papers_by_uids
# ---------------------------------------------------------------------------


class TestDeletePapersByUids:
    """Tests for delete_papers_by_uids()."""

    def test_deletes_papers_by_uid(self, tmp_path):
        """Papers with matching UIDs are removed from the database."""
        set_test_db(tmp_path / "test.db")
        with DatabaseManager() as db:
            db.create_tables()
            from sqlalchemy import insert as sa_insert

            db._session.execute(
                sa_insert(Paper).values(
                    uid="aabbccdd0001",
                    title="To Delete",
                    abstract=None,
                    authors="Eve",
                    session="S",
                    poster_position="1",
                    year=2024,
                    conference="ICML",
                )
            )
            db._session.commit()

            count = delete_papers_by_uids(db, ["aabbccdd0001"])
            assert count == 1

            remaining = find_papers_without_abstract(db)
            assert "aabbccdd0001" not in remaining

    def test_returns_zero_for_empty_list(self, tmp_path):
        """Calling with an empty list deletes nothing and returns 0."""
        set_test_db(tmp_path / "test.db")
        with DatabaseManager() as db:
            db.create_tables()
            count = delete_papers_by_uids(db, [])
        assert count == 0

    def test_raises_when_not_connected(self):
        """DatabaseError is raised if the session is not open."""
        db = DatabaseManager.__new__(DatabaseManager)
        db._session = None
        with pytest.raises(DatabaseError):
            delete_papers_by_uids(db, ["some-uid"])


# ---------------------------------------------------------------------------
# delete_embeddings_by_uids
# ---------------------------------------------------------------------------


class TestDeleteEmbeddingsByUids:
    """Tests for delete_embeddings_by_uids()."""

    def test_deletes_present_embeddings(self):
        """Only IDs that exist in ChromaDB are deleted; count matches."""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": ["uid1", "uid2"]}

        em = MagicMock()
        em.collection = mock_collection

        count = delete_embeddings_by_uids(em, ["uid1", "uid2", "uid3"])

        mock_collection.get.assert_called_once_with(ids=["uid1", "uid2", "uid3"])
        mock_collection.delete.assert_called_once_with(ids=["uid1", "uid2"])
        assert count == 2

    def test_returns_zero_when_none_present(self):
        """Returns 0 and does not call delete() when no IDs exist."""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": []}

        em = MagicMock()
        em.collection = mock_collection

        count = delete_embeddings_by_uids(em, ["ghost-uid"])

        mock_collection.delete.assert_not_called()
        assert count == 0

    def test_returns_zero_for_empty_input(self):
        """Empty UID list skips the collection entirely and returns 0."""
        em = MagicMock()
        count = delete_embeddings_by_uids(em, [])
        em.collection.get.assert_not_called()
        assert count == 0

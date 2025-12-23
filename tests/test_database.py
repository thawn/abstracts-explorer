"""
Tests for the database module.

Note: This test file contains tests for core database functionality (connection,
table creation, context managers, etc.). Tests using the old schema (string IDs,
old field names) have been removed. See test_authors.py for comprehensive tests
using the new schema with integer IDs and proper author relationships.
"""

import pytest
import sqlite3

from neurips_abstracts.database import DatabaseManager, DatabaseError
from neurips_abstracts.plugin import LightweightPaper
from pydantic import ValidationError

# Fixtures are now imported from conftest.py:
# - db_manager: DatabaseManager instance with temporary database
# - connected_db: Connected database with tables created


@pytest.fixture
def sample_paper():
    """Create a sample LightweightPaper for testing."""
    return LightweightPaper(
        title="Test Paper",
        authors=["John Doe", "Jane Smith"],
        abstract="This is a test abstract for a sample paper.",
        session="Session 1",
        poster_position="P1",
        year=2025,
        conference="NeurIPS",
        paper_pdf_url="https://example.com/paper.pdf",
        url="https://example.com/paper",
        keywords=["machine learning", "deep learning"],
        award="Best Paper",
    )


@pytest.fixture
def sample_paper_minimal():
    """Create a minimal LightweightPaper with only required fields."""
    return LightweightPaper(
        title="Minimal Paper",
        authors=["Author One"],
        abstract="Minimal abstract.",
        session="Session A",
        poster_position="A1",
        year=2024,
        conference="ICLR",
    )


class TestDatabaseManager:
    """Tests for DatabaseManager class."""

    def test_init(self, tmp_path):
        """Test DatabaseManager initialization."""
        db_path = tmp_path / "test.db"
        db = DatabaseManager(db_path)

        assert db.db_path == db_path
        assert db.connection is None

    def test_connect(self, db_manager):
        """Test database connection."""
        db_manager.connect()

        assert db_manager.connection is not None
        assert isinstance(db_manager.connection, sqlite3.Connection)

        db_manager.close()

    def test_connect_creates_directories(self, tmp_path):
        """Test that connect creates parent directories."""
        db_path = tmp_path / "subdir" / "another" / "test.db"
        db = DatabaseManager(db_path)
        db.connect()

        assert db_path.parent.exists()
        assert db.connection is not None

        db.close()

    def test_close(self, db_manager):
        """Test database close."""
        db_manager.connect()
        db_manager.close()

        assert db_manager.connection is None

    def test_close_without_connection(self, db_manager):
        """Test closing without connection doesn't raise error."""
        db_manager.close()  # Should not raise
        assert db_manager.connection is None

    def test_context_manager(self, tmp_path):
        """Test DatabaseManager as context manager."""
        db_path = tmp_path / "test.db"

        with DatabaseManager(db_path) as db:
            assert db.connection is not None

        # Connection should be closed after exiting context
        assert db.connection is None

    def test_create_tables(self, db_manager):
        """Test table creation."""
        db_manager.connect()
        db_manager.create_tables()

        # Check if tables exist
        cursor = db_manager.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        assert "papers" in tables

        db_manager.close()

    def test_create_tables_without_connection(self, db_manager):
        """Test create_tables raises error when not connected."""
        with pytest.raises(DatabaseError, match="Not connected to database"):
            db_manager.create_tables()

    def test_query_without_connection(self, db_manager):
        """Test query raises error when not connected."""
        with pytest.raises(DatabaseError, match="Not connected to database"):
            db_manager.query("SELECT * FROM papers")

    def test_query_with_invalid_sql(self, connected_db):
        """Test query with invalid SQL."""
        with pytest.raises(DatabaseError, match="Query failed"):
            connected_db.query("SELECT * FROM nonexistent_table")

    def test_get_paper_count_empty(self, connected_db):
        """Test getting paper count on empty database."""
        assert connected_db.get_paper_count() == 0

    def test_get_filter_options_empty(self, connected_db):
        """Test getting filter options on empty database."""
        filters = connected_db.get_filter_options()
        assert isinstance(filters, dict)
        assert "sessions" in filters
        assert "years" in filters
        assert "conferences" in filters
        assert filters["sessions"] == []
        assert filters["years"] == []
        assert filters["conferences"] == []

    def test_get_filter_options_with_data(self, connected_db):
        """Test getting filter options with sample data."""
        # Use add_paper method to insert test data
        paper1 = LightweightPaper(
            title="Paper 1",
            authors=["Author 1"],
            abstract="Abstract 1",
            session="Session A",
            poster_position="P1",
            year=2025,
            conference="NeurIPS",
        )
        paper2 = LightweightPaper(
            title="Paper 2",
            authors=["Author 2"],
            abstract="Abstract 2",
            session="Session B",
            poster_position="P2",
            year=2024,
            conference="ICLR",
        )
        paper3 = LightweightPaper(
            title="Paper 3",
            authors=["Author 3"],
            abstract="Abstract 3",
            session="Session A",
            poster_position="P3",
            year=2025,
            conference="NeurIPS",
        )

        connected_db.add_paper(paper1)
        connected_db.add_paper(paper2)
        connected_db.add_paper(paper3)

        filters = connected_db.get_filter_options()
        assert isinstance(filters, dict)
        assert sorted(filters["sessions"]) == ["Session A", "Session B"]
        assert sorted(filters["years"]) == [2024, 2025]
        assert sorted(filters["conferences"]) == ["ICLR", "NeurIPS"]


class TestAddPaper:
    """Tests for the add_paper method."""

    def test_add_paper_basic(self, connected_db, sample_paper):
        """Test adding a single paper with all fields."""
        paper_uid = connected_db.add_paper(sample_paper)

        assert paper_uid is not None
        assert isinstance(paper_uid, str)

        # Verify paper was added
        result = connected_db.query("SELECT * FROM papers WHERE uid = ?", (paper_uid,))
        assert len(result) == 1

        row = result[0]
        assert row["title"] == "Test Paper"
        assert row["authors"] == "John Doe; Jane Smith"
        assert row["abstract"] == "This is a test abstract for a sample paper."
        assert row["session"] == "Session 1"
        assert row["poster_position"] == "P1"
        assert row["year"] == 2025
        assert row["conference"] == "NeurIPS"
        assert row["paper_pdf_url"] == "https://example.com/paper.pdf"
        assert row["url"] == "https://example.com/paper"
        assert row["keywords"] == "machine learning, deep learning"
        assert row["award"] == "Best Paper"

    def test_add_paper_minimal(self, connected_db, sample_paper_minimal):
        """Test adding a paper with only required fields."""
        paper_uid = connected_db.add_paper(sample_paper_minimal)

        assert paper_uid is not None

        # Verify paper was added
        result = connected_db.query("SELECT * FROM papers WHERE uid = ?", (paper_uid,))
        assert len(result) == 1

        row = result[0]
        assert row["title"] == "Minimal Paper"
        assert row["authors"] == "Author One"
        assert row["abstract"] == "Minimal abstract."
        assert row["session"] == "Session A"
        assert row["poster_position"] == "A1"
        assert row["year"] == 2024
        assert row["conference"] == "ICLR"
        # Optional fields should be None or empty
        assert row["paper_pdf_url"] is None
        assert row["url"] is None
        assert row["keywords"] == ""
        assert row["award"] is None

    def test_add_paper_duplicate(self, connected_db, sample_paper):
        """Test adding a duplicate paper returns None."""
        # Add paper first time
        paper_uid1 = connected_db.add_paper(sample_paper)
        assert paper_uid1 is not None

        # Add same paper again (same title, conference, year -> same UID)
        paper_uid2 = connected_db.add_paper(sample_paper)
        assert paper_uid2 is None

        # Verify only one paper in database
        count = connected_db.get_paper_count()
        assert count == 1

    def test_add_paper_without_connection(self, db_manager, sample_paper):
        """Test add_paper raises error when not connected."""
        with pytest.raises(DatabaseError, match="Not connected to database"):
            db_manager.add_paper(sample_paper)

    def test_add_paper_with_original_id(self, connected_db):
        """Test adding a paper with an original_id."""
        paper = LightweightPaper(
            original_id=12345,
            title="Paper with ID",
            authors=["Author"],
            abstract="Abstract",
            session="Session",
            poster_position="P1",
            year=2025,
            conference="NeurIPS",
        )

        paper_uid = connected_db.add_paper(paper)
        assert paper_uid is not None

        # Verify the original_id was stored
        result = connected_db.query("SELECT * FROM papers WHERE uid = ?", (paper_uid,))
        assert len(result) == 1
        assert result[0]["original_id"] == "12345"

    def test_add_paper_generates_uid(self, connected_db, sample_paper):
        """Test that add_paper generates a UID correctly."""
        paper_uid = connected_db.add_paper(sample_paper)

        result = connected_db.query("SELECT uid FROM papers WHERE uid = ?", (paper_uid,))
        assert len(result) == 1

        uid = result[0]["uid"]
        assert uid is not None
        assert len(uid) == 16  # SHA256 hash truncated to 16 chars
        assert isinstance(uid, str)


class TestAddPapers:
    """Tests for the add_papers method."""

    def test_add_papers_multiple(self, connected_db):
        """Test adding multiple papers in batch."""
        papers = [
            LightweightPaper(
                title=f"Paper {i}",
                authors=[f"Author {i}"],
                abstract=f"Abstract {i}",
                session=f"Session {i}",
                poster_position=f"P{i}",
                year=2025,
                conference="NeurIPS",
            )
            for i in range(5)
        ]

        count = connected_db.add_papers(papers)
        assert count == 5

        # Verify all papers were added
        total_count = connected_db.get_paper_count()
        assert total_count == 5

    def test_add_papers_empty_list(self, connected_db):
        """Test adding empty list returns 0."""
        count = connected_db.add_papers([])
        assert count == 0

    def test_add_papers_with_duplicates(self, connected_db):
        """Test adding papers with some duplicates."""
        paper1 = LightweightPaper(
            title="Unique Paper 1",
            authors=["Author 1"],
            abstract="Abstract 1",
            session="Session 1",
            poster_position="P1",
            year=2025,
            conference="NeurIPS",
        )
        paper2 = LightweightPaper(
            title="Unique Paper 2",
            authors=["Author 2"],
            abstract="Abstract 2",
            session="Session 2",
            poster_position="P2",
            year=2025,
            conference="NeurIPS",
        )
        # Duplicate of paper1
        paper1_dup = LightweightPaper(
            title="Unique Paper 1",  # Same title, conference, year
            authors=["Author 1 Updated"],  # Different author
            abstract="Updated abstract",  # Different abstract
            session="Session 1",
            poster_position="P1",
            year=2025,
            conference="NeurIPS",
        )

        # Add first batch
        count1 = connected_db.add_papers([paper1, paper2])
        assert count1 == 2

        # Add second batch with duplicate
        count2 = connected_db.add_papers([paper1_dup, paper2])
        assert count2 == 0  # Both are duplicates

        # Total papers should still be 2
        total_count = connected_db.get_paper_count()
        assert total_count == 2

    def test_add_papers_without_connection(self, db_manager):
        """Test add_papers raises error when not connected."""
        papers = [
            LightweightPaper(
                title="Paper",
                authors=["Author"],
                abstract="Abstract",
                session="Session",
                poster_position="P1",
                year=2025,
                conference="NeurIPS",
            )
        ]

        with pytest.raises(DatabaseError, match="Not connected to database"):
            db_manager.add_papers(papers)

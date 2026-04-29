"""
Tests for the database module.

Note: This test file contains tests for core database functionality (connection,
table creation, context managers, etc.). Tests using the old schema (string IDs,
old field names) have been removed. See test_authors.py for comprehensive tests
using the new schema with integer IDs and proper author relationships.
"""

import pytest
import sqlite3
from unittest.mock import patch

from abstracts_explorer.database import DatabaseManager, DatabaseError, normalize_model_name
from abstracts_explorer.plugin import LightweightPaper
from tests.conftest import set_test_db

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


class TestNormalizeModelName:
    """Tests for normalize_model_name function."""

    def test_no_alias_prefix(self):
        """Model names without alias- prefix are unchanged."""
        assert normalize_model_name("qwen3-embeddings-8b") == "qwen3-embeddings-8b"

    def test_strips_alias_prefix(self):
        """Model names with alias- prefix have it stripped."""
        assert normalize_model_name("alias-qwen3-embeddings-8b") == "qwen3-embeddings-8b"

    def test_alias_prefix_case_insensitive(self):
        """The alias- prefix is stripped regardless of case."""
        assert normalize_model_name("Alias-qwen3-embeddings-8b") == "qwen3-embeddings-8b"
        assert normalize_model_name("ALIAS-qwen3-embeddings-8b") == "qwen3-embeddings-8b"

    def test_only_leading_alias_stripped(self):
        """Only a leading alias- prefix is stripped, not occurrences elsewhere."""
        assert normalize_model_name("model-alias-name") == "model-alias-name"

    def test_empty_after_strip(self):
        """Edge case: model name that is exactly 'alias-' results in empty string."""
        assert normalize_model_name("alias-") == ""

    def test_plain_alias_word(self):
        """The word 'alias' without the hyphen is not stripped."""
        assert normalize_model_name("aliasmodel") == "aliasmodel"


class TestDatabaseManager:
    """Tests for DatabaseManager class."""

    def test_init(self, tmp_path):
        """Test DatabaseManager initialization."""
        db_path = tmp_path / "test.db"
        set_test_db(db_path)

        db = DatabaseManager()

        assert db.database_url == f"sqlite:///{db_path}"
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
        set_test_db(db_path)

        db = DatabaseManager()
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
        set_test_db(db_path)

        with DatabaseManager() as db:
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

    def test_create_tables_idempotent(self, db_manager):
        """Test that create_tables can be called multiple times without error."""
        db_manager.connect()

        # Call create_tables multiple times - should not raise errors
        db_manager.create_tables()
        db_manager.create_tables()
        db_manager.create_tables()

        # Check tables still exist and work
        cursor = db_manager.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        assert "papers" in tables

        db_manager.close()

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

    def test_get_sessions_empty(self, connected_db):
        """Test getting sessions on empty database."""
        sessions = connected_db.get_sessions()
        assert sessions == []

    def test_get_conferences_empty(self, connected_db):
        """Test getting conferences on empty database."""
        conferences = connected_db.get_conferences()
        assert conferences == []

    def test_get_years_empty(self, connected_db):
        """Test getting years on empty database."""
        years = connected_db.get_years()
        assert years == []

    def test_get_sessions_with_data(self, connected_db):
        """Test getting sessions with sample data."""
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

        sessions = connected_db.get_sessions()
        assert sorted(sessions) == ["Session A", "Session B"]

        conferences = connected_db.get_conferences()
        assert sorted(conferences) == ["ICLR", "NeurIPS"]

        years = connected_db.get_years()
        assert years == [2025, 2024]

    def test_get_sessions_filtered(self, connected_db):
        """Test getting sessions filtered by conference and year."""
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
        connected_db.add_paper(paper1)
        connected_db.add_paper(paper2)

        # Filter by conference
        sessions = connected_db.get_sessions(conference="NeurIPS")
        assert sessions == ["Session A"]

        # Filter by year
        sessions = connected_db.get_sessions(year=2024)
        assert sessions == ["Session B"]

    def test_get_conferences_filtered_by_year(self, connected_db):
        """Test getting conferences filtered by year."""
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
        connected_db.add_paper(paper1)
        connected_db.add_paper(paper2)

        conferences = connected_db.get_conferences(year=2025)
        assert conferences == ["NeurIPS"]

        conferences = connected_db.get_conferences(year=2024)
        assert conferences == ["ICLR"]

    def test_get_years_filtered_by_conference(self, connected_db):
        """Test getting years filtered by conference."""
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
        connected_db.add_paper(paper1)
        connected_db.add_paper(paper2)

        years = connected_db.get_years(conference="NeurIPS")
        assert years == [2025]

        years = connected_db.get_years(conference="ICLR")
        assert years == [2024]


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

    def test_add_papers_skips_erroneous_abstract_and_continues(self, connected_db, mocker):
        """Test that an import error on one abstract doesn't abort the rest."""
        good_paper = LightweightPaper(
            title="Good Paper",
            authors=["Author"],
            abstract="Good abstract",
            session="Session",
            poster_position="P1",
            year=2025,
            conference="NeurIPS",
        )
        bad_paper = LightweightPaper(
            title="Bad Paper",
            authors=["Author"],
            abstract="Bad abstract",
            session="Session",
            poster_position="P2",
            year=2025,
            conference="NeurIPS",
        )
        another_good_paper = LightweightPaper(
            title="Another Good Paper",
            authors=["Author"],
            abstract="Another good abstract",
            session="Session",
            poster_position="P3",
            year=2025,
            conference="NeurIPS",
        )

        original_add_paper = connected_db.add_paper

        def add_paper_side_effect(paper):
            if paper.title == "Bad Paper":
                raise DatabaseError("Simulated import error")
            return original_add_paper(paper)

        mocker.patch.object(connected_db, "add_paper", side_effect=add_paper_side_effect)

        count = connected_db.add_papers([good_paper, bad_paper, another_good_paper])

        # Two good papers inserted, bad one skipped
        assert count == 2
        total_count = connected_db.get_paper_count()
        assert total_count == 2


class TestParseFieldFilters:
    """Tests for the parse_field_filters static method."""

    def test_no_filter(self):
        """Test query without any field filter returns empty dict and original query."""
        filters, remaining = DatabaseManager.parse_field_filters("transformers")
        assert filters == {}
        assert remaining == "transformers"

    def test_authors_filter_only(self):
        """Test query with only authors filter."""
        filters, remaining = DatabaseManager.parse_field_filters('authors:"John Smith"')
        assert filters == {"authors": "John Smith"}
        assert remaining == ""

    def test_authors_filter_with_keyword(self):
        """Test authors filter combined with keyword."""
        filters, remaining = DatabaseManager.parse_field_filters('authors:"John Smith" transformers')
        assert filters == {"authors": "John Smith"}
        assert remaining == "transformers"

    def test_keyword_before_filter(self):
        """Test keyword before field filter."""
        filters, remaining = DatabaseManager.parse_field_filters('deep learning authors:"Jane Doe"')
        assert filters == {"authors": "Jane Doe"}
        assert remaining == "deep learning"

    def test_filter_in_middle(self):
        """Test field filter in the middle of query."""
        filters, remaining = DatabaseManager.parse_field_filters('deep authors:"Jane Doe" learning')
        assert filters == {"authors": "Jane Doe"}
        assert remaining == "deep learning"

    def test_empty_query(self):
        """Test empty query."""
        filters, remaining = DatabaseManager.parse_field_filters("")
        assert filters == {}
        assert remaining == ""

    def test_multiple_filters(self):
        """Test multiple field filters in one query."""
        filters, remaining = DatabaseManager.parse_field_filters('authors:"Doe" award:"Best Paper" transformers')
        assert filters == {"authors": "Doe", "award": "Best Paper"}
        assert remaining == "transformers"

    def test_conference_filter(self):
        """Test conference field filter."""
        filters, remaining = DatabaseManager.parse_field_filters('conference:"NeurIPS"')
        assert filters == {"conference": "NeurIPS"}
        assert remaining == ""

    def test_session_filter(self):
        """Test session field filter."""
        filters, remaining = DatabaseManager.parse_field_filters('session:"Oral Session" attention')
        assert filters == {"session": "Oral Session"}
        assert remaining == "attention"

    def test_title_filter(self):
        """Test title field filter."""
        filters, remaining = DatabaseManager.parse_field_filters('title:"Transformer"')
        assert filters == {"title": "Transformer"}
        assert remaining == ""

    def test_year_filter(self):
        """Test year field filter."""
        filters, remaining = DatabaseManager.parse_field_filters('year:"2025"')
        assert filters == {"year": "2025"}
        assert remaining == ""

    def test_unknown_field_left_in_query(self):
        """Test that unrecognised field names are left in the query."""
        filters, remaining = DatabaseManager.parse_field_filters('foo:"bar" transformers')
        assert filters == {}
        assert remaining == 'foo:"bar" transformers'

    def test_keywords_filter(self):
        """Test keywords field filter."""
        filters, remaining = DatabaseManager.parse_field_filters('keywords:"deep learning"')
        assert filters == {"keywords": "deep learning"}
        assert remaining == ""

    def test_author_alias_only(self):
        """Test that 'author' is accepted as an alias for 'authors'."""
        filters, remaining = DatabaseManager.parse_field_filters('author:"John Smith"')
        assert filters == {"authors": "John Smith"}
        assert remaining == ""

    def test_author_alias_with_keyword(self):
        """Test 'author' alias combined with keyword."""
        filters, remaining = DatabaseManager.parse_field_filters('author:"Vaswani" transformer')
        assert filters == {"authors": "Vaswani"}
        assert remaining == "transformer"


class TestSearchPapersFieldFilters:
    """Tests for field filtering in search_papers and search_papers_keyword."""

    @pytest.fixture
    def db_with_papers(self, connected_db):
        """Create a database with papers by different authors."""
        papers = [
            LightweightPaper(
                title="Attention is All You Need",
                authors=["Ashish Vaswani", "Noam Shazeer"],
                abstract="We propose the Transformer architecture.",
                session="Session 1",
                poster_position="P1",
                year=2017,
                conference="NeurIPS",
                award="Best Paper",
                keywords=["attention", "transformer"],
            ),
            LightweightPaper(
                title="BERT Paper",
                authors=["Jacob Devlin", "Ming-Wei Chang"],
                abstract="We introduce BERT for NLP.",
                session="Session 2",
                poster_position="P2",
                year=2019,
                conference="NeurIPS",
                keywords=["bert", "pretraining"],
            ),
            LightweightPaper(
                title="ResNet Paper",
                authors=["Kaiming He", "Xiangyu Zhang"],
                abstract="Deep residual learning for image recognition.",
                session="Session 3",
                poster_position="P3",
                year=2016,
                conference="ICLR",
                keywords=["resnet", "computer vision"],
            ),
        ]
        for paper in papers:
            connected_db.add_paper(paper)
        return connected_db

    def test_search_papers_by_author(self, db_with_papers):
        """Test search_papers with authors field filter."""
        results = db_with_papers.search_papers(field_filters={"authors": "Vaswani"})
        assert len(results) == 1
        assert results[0]["title"] == "Attention is All You Need"

    def test_search_papers_by_author_case_insensitive(self, db_with_papers):
        """Test that field filter search is case-insensitive."""
        results = db_with_papers.search_papers(field_filters={"authors": "vaswani"})
        assert len(results) == 1
        assert results[0]["title"] == "Attention is All You Need"

    def test_search_papers_author_partial_match(self, db_with_papers):
        """Test that field filter matches partial values."""
        results = db_with_papers.search_papers(field_filters={"authors": "Kaiming"})
        assert len(results) == 1
        assert results[0]["title"] == "ResNet Paper"

    def test_search_papers_author_no_match(self, db_with_papers):
        """Test field filter with no matches."""
        results = db_with_papers.search_papers(field_filters={"authors": "Nonexistent Author"})
        assert len(results) == 0

    def test_search_papers_field_filter_and_keyword(self, db_with_papers):
        """Test combining field filter and keyword search."""
        # Author matches but keyword doesn't
        results = db_with_papers.search_papers(field_filters={"authors": "Vaswani"}, keyword="BERT")
        assert len(results) == 0

        # Both author and keyword match
        results = db_with_papers.search_papers(field_filters={"authors": "Vaswani"}, keyword="Transformer")
        assert len(results) == 1

    def test_search_papers_keyword_author_syntax(self, db_with_papers):
        """Test search_papers_keyword with authors:"Name" syntax."""
        results = db_with_papers.search_papers_keyword(query='authors:"Vaswani"', limit=10)
        assert len(results) == 1
        assert results[0]["title"] == "Attention is All You Need"
        # Authors should be parsed into a list
        assert isinstance(results[0]["authors"], list)

    def test_search_papers_keyword_author_with_keyword(self, db_with_papers):
        """Test search_papers_keyword with authors and keyword combined."""
        results = db_with_papers.search_papers_keyword(query='authors:"Devlin" BERT', limit=10)
        assert len(results) == 1
        assert results[0]["title"] == "BERT Paper"

    def test_search_papers_keyword_author_no_match_keyword(self, db_with_papers):
        """Test authors matches but keyword doesn't."""
        results = db_with_papers.search_papers_keyword(query='authors:"Vaswani" BERT', limit=10)
        assert len(results) == 0

    def test_search_by_award(self, db_with_papers):
        """Test search_papers_keyword with award field filter."""
        results = db_with_papers.search_papers_keyword(query='award:"Best Paper"', limit=10)
        assert len(results) == 1
        assert results[0]["title"] == "Attention is All You Need"

    def test_search_by_conference(self, db_with_papers):
        """Test search_papers_keyword with conference field filter."""
        results = db_with_papers.search_papers_keyword(query='conference:"ICLR"', limit=10)
        assert len(results) == 1
        assert results[0]["title"] == "ResNet Paper"

    def test_search_by_multiple_fields(self, db_with_papers):
        """Test search_papers_keyword with multiple field filters."""
        results = db_with_papers.search_papers_keyword(query='authors:"Vaswani" conference:"NeurIPS"', limit=10)
        assert len(results) == 1
        assert results[0]["title"] == "Attention is All You Need"

    def test_search_by_keywords_field(self, db_with_papers):
        """Test search_papers_keyword with keywords field filter."""
        results = db_with_papers.search_papers_keyword(query='keywords:"computer vision"', limit=10)
        assert len(results) == 1
        assert results[0]["title"] == "ResNet Paper"

    def test_author_alias_in_keyword_search(self, db_with_papers):
        """Test that 'author' alias works the same as 'authors'."""
        results = db_with_papers.search_papers_keyword(query='author:"Vaswani"', limit=10)
        assert len(results) == 1
        assert results[0]["title"] == "Attention is All You Need"
        assert isinstance(results[0]["authors"], list)

    def test_case_insensitive_field_name_parsing(self, db_with_papers):
        """Test that field names in field:\"value\" syntax are case-insensitive."""
        # Uppercase
        results = db_with_papers.search_papers_keyword(query='AUTHOR:"Vaswani"', limit=10)
        assert len(results) == 1
        assert results[0]["title"] == "Attention is All You Need"

        # Title case
        results = db_with_papers.search_papers_keyword(query='Author:"Vaswani"', limit=10)
        assert len(results) == 1
        assert results[0]["title"] == "Attention is All You Need"

        # Mixed case field with keyword
        results = db_with_papers.search_papers_keyword(query='Authors:"Devlin" BERT', limit=10)
        assert len(results) == 1
        assert results[0]["title"] == "BERT Paper"

    def test_case_insensitive_parse_field_filters(self):
        """Test parse_field_filters with various case combinations."""
        # Uppercase alias
        filters, remaining = DatabaseManager.parse_field_filters('AUTHOR:"Smith" topic')
        assert filters == {"authors": "Smith"}
        assert remaining == "topic"

        # Title-case canonical field
        filters, remaining = DatabaseManager.parse_field_filters('Title:"Transformer"')
        assert filters == {"title": "Transformer"}
        assert remaining == ""

        # Mixed-case award field
        filters, remaining = DatabaseManager.parse_field_filters('Award:"Best Paper" deep learning')
        assert filters == {"award": "Best Paper"}
        assert remaining == "deep learning"


class TestEmbeddingModelMetadata:
    """Tests for embedding model metadata functionality."""

    def test_get_embedding_model_none_when_not_set(self, connected_db):
        """Test that get_embedding_model returns None when not set."""
        model = connected_db.get_embedding_model()
        assert model is None

    def test_set_and_get_embedding_model(self, connected_db):
        """Test setting and retrieving embedding model."""
        model_name = "text-embedding-qwen3-embedding-4b"
        connected_db.set_embedding_model(model_name)

        retrieved_model = connected_db.get_embedding_model()
        assert retrieved_model == model_name

    def test_update_embedding_model(self, connected_db):
        """Test updating the embedding model."""
        # Set initial model
        model1 = "text-embedding-model-v1"
        connected_db.set_embedding_model(model1)
        assert connected_db.get_embedding_model() == model1

        # Update to new model
        model2 = "text-embedding-model-v2"
        connected_db.set_embedding_model(model2)
        assert connected_db.get_embedding_model() == model2

    def test_embedding_model_persists_across_connections(self, tmp_path):
        """Test that embedding model persists across database connections."""
        db_path = tmp_path / "test.db"
        model_name = "persistent-model"

        # Set PAPER_DB and reload config
        set_test_db(db_path)

        # First connection: set the model
        with DatabaseManager() as db1:
            db1.create_tables()
            db1.set_embedding_model(model_name)

        # Second connection: retrieve the model
        with DatabaseManager() as db2:
            retrieved_model = db2.get_embedding_model()
            assert retrieved_model == model_name


class TestClusteringCache:
    """Tests for clustering cache functionality."""

    def test_get_clustering_cache_none_when_not_set(self, connected_db):
        """Test that get_clustering_cache returns None when no cache exists."""
        cache = connected_db.get_clustering_cache(
            embedding_model="test-model",
            reduction_method="pca",
            n_components=2,
            clustering_method="kmeans",
            n_clusters=5,
        )
        assert cache is None

    def test_save_and_get_clustering_cache(self, connected_db):
        """Test saving and retrieving clustering cache."""
        # Create sample clustering results
        results = {
            "points": [
                {"id": "paper1", "x": 1.0, "y": 2.0, "cluster": 0},
                {"id": "paper2", "x": 3.0, "y": 4.0, "cluster": 1},
            ],
            "statistics": {"n_clusters": 2, "total_papers": 2},
            "cluster_centers": {0: {"x": 1.0, "y": 2.0}, 1: {"x": 3.0, "y": 4.0}},
        }

        # Save cache
        connected_db.save_clustering_cache(
            embedding_model="test-model",
            reduction_method="pca",
            n_components=2,
            clustering_method="kmeans",
            results=results,
            n_clusters=2,
        )

        # Retrieve cache with exact match
        cached = connected_db.get_clustering_cache(
            embedding_model="test-model",
            reduction_method="pca",
            n_components=2,
            clustering_method="kmeans",
            n_clusters=2,
        )

        assert cached is not None
        assert cached["points"] == results["points"]
        assert cached["statistics"] == results["statistics"]
        # JSON serialization converts int keys to strings
        assert cached["cluster_centers"]["0"] == results["cluster_centers"][0]
        assert cached["cluster_centers"]["1"] == results["cluster_centers"][1]

    def test_cache_invalidation_by_embedding_model(self, connected_db):
        """Test that cache is invalidated when embedding model changes."""
        results = {
            "points": [{"id": "p1", "x": 1.0, "y": 2.0, "cluster": 0}],
            "statistics": {"n_clusters": 1},
        }

        # Save with model1
        connected_db.save_clustering_cache(
            embedding_model="model1",
            reduction_method="pca",
            n_components=2,
            clustering_method="kmeans",
            results=results,
            n_clusters=5,
        )

        # Try to get with model2 - should return None
        cached = connected_db.get_clustering_cache(
            embedding_model="model2",
            reduction_method="pca",
            n_components=2,
            clustering_method="kmeans",
            n_clusters=5,
        )
        assert cached is None

    def test_cache_reuse_on_different_reduction_method(self, connected_db):
        """Test that clustering results are reusable when only reduction_method differs."""
        results = {
            "points": [{"id": "p1", "x": 1.0, "y": 2.0, "cluster": 0}],
            "statistics": {"n_clusters": 2},
        }

        # Save with pca
        connected_db.save_clustering_cache(
            embedding_model="model1",
            reduction_method="pca",
            n_components=2,
            clustering_method="kmeans",
            results=results,
            n_clusters=2,
        )

        # Exact match with pca should work
        cached = connected_db.get_clustering_cache(
            embedding_model="model1",
            reduction_method="pca",
            n_components=2,
            clustering_method="kmeans",
            n_clusters=2,
        )
        assert cached is not None

        # Exact match with tsne should miss
        cached_tsne = connected_db.get_clustering_cache(
            embedding_model="model1",
            reduction_method="tsne",
            n_components=2,
            clustering_method="kmeans",
            n_clusters=2,
        )
        assert cached_tsne is None

        # Clustering-only match (no reduction_method) should find the pca entry
        cached_any = connected_db.get_clustering_cache(
            embedding_model="model1",
            clustering_method="kmeans",
            n_clusters=2,
        )
        assert cached_any is not None
        assert cached_any["statistics"]["n_clusters"] == 2

    def test_clear_clustering_cache(self, connected_db):
        """Test clearing clustering cache."""
        results = {"points": [], "statistics": {}}

        # Save two cache entries
        connected_db.save_clustering_cache(
            embedding_model="model1",
            reduction_method="pca",
            n_components=2,
            clustering_method="kmeans",
            results=results,
            n_clusters=5,
        )
        connected_db.save_clustering_cache(
            embedding_model="model1",
            reduction_method="tsne",
            n_components=2,
            clustering_method="kmeans",
            results=results,
            n_clusters=5,
        )

        # Clear all cache
        count = connected_db.clear_clustering_cache()
        assert count == 2

        # Verify cache is empty
        cached = connected_db.get_clustering_cache(
            embedding_model="model1",
            reduction_method="pca",
            n_components=2,
            clustering_method="kmeans",
            n_clusters=5,
        )
        assert cached is None

    def test_clear_clustering_cache_by_model(self, connected_db):
        """Test clearing clustering cache for specific model."""
        results = {"points": [], "statistics": {}}

        # Save cache for two models
        connected_db.save_clustering_cache(
            embedding_model="model1",
            reduction_method="pca",
            n_components=2,
            clustering_method="kmeans",
            results=results,
            n_clusters=5,
        )
        connected_db.save_clustering_cache(
            embedding_model="model2",
            reduction_method="pca",
            n_components=2,
            clustering_method="kmeans",
            results=results,
            n_clusters=5,
        )

        # Clear only model1 cache
        count = connected_db.clear_clustering_cache(embedding_model="model1")
        assert count == 1

        # Verify model1 cache is cleared but model2 remains
        cached1 = connected_db.get_clustering_cache(
            embedding_model="model1",
            reduction_method="pca",
            n_components=2,
            clustering_method="kmeans",
            n_clusters=5,
        )
        assert cached1 is None

        cached2 = connected_db.get_clustering_cache(
            embedding_model="model2",
            reduction_method="pca",
            n_components=2,
            clustering_method="kmeans",
            n_clusters=5,
        )
        assert cached2 is not None

    def test_save_and_get_cache_with_conference_year(self, connected_db):
        """Test that conference/year columns scope the cache correctly."""
        results = {
            "points": [{"id": "p1", "x": 1.0, "y": 2.0, "cluster": 0}],
            "statistics": {"n_clusters": 2},
        }

        # Save with conference/year
        connected_db.save_clustering_cache(
            embedding_model="model1",
            reduction_method="pca",
            n_components=2,
            clustering_method="kmeans",
            results=results,
            n_clusters=2,
            conference="NeurIPS",
            year=2024,
        )

        # Exact match should work
        cached = connected_db.get_clustering_cache(
            embedding_model="model1",
            reduction_method="pca",
            n_components=2,
            clustering_method="kmeans",
            n_clusters=2,
            conference="NeurIPS",
            year=2024,
        )
        assert cached is not None

        # Different conference should miss
        cached_miss = connected_db.get_clustering_cache(
            embedding_model="model1",
            reduction_method="pca",
            n_components=2,
            clustering_method="kmeans",
            n_clusters=2,
            conference="ICLR",
            year=2024,
        )
        assert cached_miss is None

        # No conference (global) should miss
        cached_global = connected_db.get_clustering_cache(
            embedding_model="model1",
            reduction_method="pca",
            n_components=2,
            clustering_method="kmeans",
            n_clusters=2,
            conference=None,
            year=None,
        )
        assert cached_global is None

    def test_save_fills_n_clusters_from_statistics(self, connected_db):
        """Test that n_clusters is auto-filled from results statistics when None."""
        results = {
            "points": [{"id": "p1", "x": 1.0, "y": 2.0, "cluster": 0}],
            "statistics": {"n_clusters": 7, "total_papers": 1},
        }

        connected_db.save_clustering_cache(
            embedding_model="model1",
            reduction_method="pca",
            n_components=2,
            clustering_method="agglomerative",
            results=results,
            n_clusters=None,
            conference="NeurIPS",
            year=2024,
        )

        from abstracts_explorer.db_models import ClusteringCache

        entry = connected_db._session.query(ClusteringCache).first()
        assert entry is not None
        assert entry.n_clusters == 7

    def test_update_clustering_cache_embedding_model(self, connected_db):
        """Test updating the embedding model in clustering cache entries."""
        results = {"points": [], "statistics": {"n_clusters": 3}}

        connected_db.save_clustering_cache(
            embedding_model="old-model",
            reduction_method="pca",
            n_components=2,
            clustering_method="kmeans",
            results=results,
            n_clusters=3,
        )
        connected_db.save_clustering_cache(
            embedding_model="old-model",
            reduction_method="tsne",
            n_components=2,
            clustering_method="kmeans",
            results=results,
            n_clusters=3,
        )

        count = connected_db.update_clustering_cache_embedding_model("old-model", "new-model")
        assert count == 2

        # Old model entries should no longer be accessible
        cached_old = connected_db.get_clustering_cache(
            embedding_model="old-model",
            reduction_method="pca",
            n_components=2,
            clustering_method="kmeans",
            n_clusters=3,
        )
        assert cached_old is None

        # New model entries should be accessible
        cached_new = connected_db.get_clustering_cache(
            embedding_model="new-model",
            reduction_method="pca",
            n_components=2,
            clustering_method="kmeans",
            n_clusters=3,
        )
        assert cached_new is not None

    def test_update_clustering_cache_embedding_model_no_match(self, connected_db):
        """Test that updating a non-existent model returns 0."""
        count = connected_db.update_clustering_cache_embedding_model("nonexistent-model", "new-model")
        assert count == 0

    def test_update_clustering_cache_embedding_model_preserves_other_entries(self, connected_db):
        """Test that updating only affects entries with the matching model."""
        results = {"points": [], "statistics": {}}

        connected_db.save_clustering_cache(
            embedding_model="model-a",
            reduction_method="pca",
            n_components=2,
            clustering_method="kmeans",
            results=results,
            n_clusters=5,
        )
        connected_db.save_clustering_cache(
            embedding_model="model-b",
            reduction_method="pca",
            n_components=2,
            clustering_method="kmeans",
            results=results,
            n_clusters=5,
        )

        count = connected_db.update_clustering_cache_embedding_model("model-a", "model-c")
        assert count == 1

        # model-b entry should be untouched
        cached_b = connected_db.get_clustering_cache(
            embedding_model="model-b",
            reduction_method="pca",
            n_components=2,
            clustering_method="kmeans",
            n_clusters=5,
        )
        assert cached_b is not None

    def test_save_and_get_hierarchical_label_cache(self, connected_db):
        """Test saving and retrieving hierarchical label cache."""
        labels = {0: "Root", 5: "Sub-cluster A", 6: "Sub-cluster B", 10: "Leaf"}

        connected_db.save_hierarchical_label_cache(
            embedding_model="test-model",
            labels=labels,
            linkage="ward",
        )

        cached = connected_db.get_hierarchical_label_cache(
            embedding_model="test-model",
            linkage="ward",
        )

        assert cached is not None
        assert cached == labels

    def test_hierarchical_label_cache_miss_on_different_linkage(self, connected_db):
        """Test that hierarchical label cache misses when linkage differs."""
        labels = {0: "Root"}
        connected_db.save_hierarchical_label_cache(
            embedding_model="test-model",
            labels=labels,
            linkage="ward",
        )

        cached = connected_db.get_hierarchical_label_cache(
            embedding_model="test-model",
            linkage="complete",
        )
        assert cached is None

    def test_hierarchical_label_cache_miss_on_different_model(self, connected_db):
        """Test that hierarchical label cache misses when embedding model differs."""
        labels = {0: "Root"}
        connected_db.save_hierarchical_label_cache(
            embedding_model="model-A",
            labels=labels,
            linkage="ward",
        )

        cached = connected_db.get_hierarchical_label_cache(
            embedding_model="model-B",
            linkage="ward",
        )
        assert cached is None


class TestValidationData:
    """Tests for validation data donation functionality."""

    def test_donate_validation_data_success(self, connected_db):
        """Test successful donation of validation data."""
        paper_priorities = {
            "test_uid_1": {"priority": 5, "searchTerm": "machine learning"},
            "test_uid_2": {"priority": 4, "searchTerm": "deep learning"},
        }

        count = connected_db.donate_validation_data(paper_priorities)
        assert count == 2

        # Verify data was stored
        from abstracts_explorer.db_models import ValidationData
        from sqlalchemy.orm import Session

        session = Session(connected_db.engine)
        try:
            entries = session.query(ValidationData).all()
            assert len(entries) == 2

            # Check first entry
            entry1 = next(e for e in entries if e.paper_uid == "test_uid_1")
            assert entry1.priority == 5
            assert entry1.search_term == "machine learning"

            # Check second entry
            entry2 = next(e for e in entries if e.paper_uid == "test_uid_2")
            assert entry2.priority == 4
            assert entry2.search_term == "deep learning"
        finally:
            session.close()

    def test_donate_validation_data_empty(self, connected_db):
        """Test donation with empty data raises ValueError."""
        import pytest

        with pytest.raises(ValueError, match="No data provided"):
            connected_db.donate_validation_data({})

    def test_donate_validation_data_invalid_format(self, connected_db):
        """Test donation with invalid data format raises ValueError."""
        import pytest

        paper_priorities = {"test_uid": 5}  # Invalid: should be dict, not int

        with pytest.raises(ValueError, match="Invalid data format"):
            connected_db.donate_validation_data(paper_priorities)

    def test_donate_validation_data_not_connected(self):
        """Test donation without connection raises DatabaseError."""
        import pytest
        from abstracts_explorer.database import DatabaseError

        db = DatabaseManager()
        # Don't connect

        with pytest.raises(DatabaseError, match="Not connected"):
            db.donate_validation_data({"test": {"priority": 5}})


class TestChatDonation:
    """Tests for chat transcript donation functionality."""

    def test_donate_chat_transcript_success(self, connected_db):
        """Test successful donation of a chat transcript."""
        transcript = [
            {"role": "user", "text": "What papers discuss transformers?"},
            {"role": "assistant", "text": "Here are some relevant papers..."},
        ]

        donation_id = connected_db.donate_chat_transcript("up", transcript)
        assert donation_id is not None

        # Verify data was stored
        import json
        from abstracts_explorer.db_models import ChatDonation
        from sqlalchemy.orm import Session

        session = Session(connected_db.engine)
        try:
            entry = session.query(ChatDonation).filter_by(id=donation_id).first()
            assert entry is not None
            assert entry.rating == "up"
            stored_transcript = json.loads(entry.transcript)
            assert len(stored_transcript) == 2
            assert stored_transcript[0]["role"] == "user"
            assert stored_transcript[1]["role"] == "assistant"
            assert entry.donated_at is not None
        finally:
            session.close()

    def test_donate_chat_transcript_thumbs_down(self, connected_db):
        """Test donation with thumbs down rating."""
        transcript = [
            {"role": "user", "text": "Tell me about NLP"},
            {"role": "assistant", "text": "NLP is..."},
        ]

        donation_id = connected_db.donate_chat_transcript("down", transcript)
        assert donation_id is not None

        from abstracts_explorer.db_models import ChatDonation
        from sqlalchemy.orm import Session

        session = Session(connected_db.engine)
        try:
            entry = session.query(ChatDonation).filter_by(id=donation_id).first()
            assert entry.rating == "down"
        finally:
            session.close()

    def test_donate_chat_transcript_invalid_rating(self, connected_db):
        """Test donation with invalid rating raises ValueError."""
        transcript = [{"role": "user", "text": "test"}]

        with pytest.raises(ValueError, match="Invalid rating"):
            connected_db.donate_chat_transcript("neutral", transcript)

    def test_donate_chat_transcript_empty_transcript(self, connected_db):
        """Test donation with empty transcript raises ValueError."""
        with pytest.raises(ValueError, match="non-empty list"):
            connected_db.donate_chat_transcript("up", [])

    def test_donate_chat_transcript_invalid_transcript_type(self, connected_db):
        """Test donation with non-list transcript raises ValueError."""
        with pytest.raises(ValueError, match="non-empty list"):
            connected_db.donate_chat_transcript("up", "not a list")

    def test_donate_chat_transcript_invalid_message_format(self, connected_db):
        """Test donation with invalid message format raises ValueError."""
        transcript = [{"role": "user"}]  # Missing 'text' key

        with pytest.raises(ValueError, match="'role' and 'text'"):
            connected_db.donate_chat_transcript("up", transcript)


class TestGetChatDonations:
    """Tests for get_chat_donations() and get_chat_donation_stats()."""

    def _make_transcript(self, n=2):
        return [
            {"role": "user", "text": f"msg {i}"} if i % 2 == 0 else {"role": "assistant", "text": f"reply {i}"}
            for i in range(n)
        ]

    def test_get_chat_donations_empty(self, connected_db):
        """Returns empty list when no donations exist."""
        result = connected_db.get_chat_donations()
        assert result == []

    def test_get_chat_donations_returns_all(self, connected_db):
        """Returns all donations when no filters applied."""
        connected_db.donate_chat_transcript("up", self._make_transcript(2))
        connected_db.donate_chat_transcript("down", self._make_transcript(4))
        result = connected_db.get_chat_donations()
        assert len(result) == 2

    def test_get_chat_donations_filter_by_rating(self, connected_db):
        """Filters donations by rating."""
        connected_db.donate_chat_transcript("up", self._make_transcript(2))
        connected_db.donate_chat_transcript("down", self._make_transcript(2))
        up_results = connected_db.get_chat_donations(rating="up")
        assert all(r["rating"] == "up" for r in up_results)
        assert len(up_results) == 1

    def test_get_chat_donations_limit(self, connected_db):
        """Respects limit parameter."""
        for _ in range(5):
            connected_db.donate_chat_transcript("up", self._make_transcript(2))
        result = connected_db.get_chat_donations(limit=3)
        assert len(result) == 3

    def test_get_chat_donations_offset(self, connected_db):
        """Respects offset parameter for pagination."""
        for _ in range(4):
            connected_db.donate_chat_transcript("up", self._make_transcript(2))
        result_all = connected_db.get_chat_donations()
        result_offset = connected_db.get_chat_donations(offset=2)
        assert len(result_offset) == 2
        assert result_offset[0]["id"] == result_all[2]["id"]

    def test_get_chat_donations_includes_parsed_transcript(self, connected_db):
        """Returned dicts include parsed transcript (list)."""
        transcript = [{"role": "user", "text": "hello"}, {"role": "assistant", "text": "hi"}]
        connected_db.donate_chat_transcript("up", transcript)
        result = connected_db.get_chat_donations()
        assert isinstance(result[0]["transcript"], list)
        assert result[0]["transcript"][0]["role"] == "user"

    def test_get_chat_donation_stats_empty(self, connected_db):
        """Stats returns zeros when no donations exist."""
        stats = connected_db.get_chat_donation_stats()
        assert stats["total"] == 0
        assert stats["up"] == 0
        assert stats["down"] == 0
        assert stats["avg_turns"] == 0.0

    def test_get_chat_donation_stats_counts(self, connected_db):
        """Stats correctly counts up/down and avg_turns."""
        connected_db.donate_chat_transcript("up", self._make_transcript(2))
        connected_db.donate_chat_transcript("up", self._make_transcript(4))
        connected_db.donate_chat_transcript("down", self._make_transcript(6))
        stats = connected_db.get_chat_donation_stats()
        assert stats["total"] == 3
        assert stats["up"] == 2
        assert stats["down"] == 1
        assert stats["avg_turns"] == pytest.approx((2 + 4 + 6) / 3)

    def test_delete_chat_donations_all(self, connected_db):
        """Deletes all donations when no ids given."""
        connected_db.donate_chat_transcript("up", self._make_transcript(2))
        connected_db.donate_chat_transcript("down", self._make_transcript(2))
        deleted = connected_db.delete_chat_donations()
        assert deleted == 2
        assert connected_db.get_chat_donations() == []

    def test_delete_chat_donations_by_ids(self, connected_db):
        """Deletes only specified donation IDs."""
        id1 = connected_db.donate_chat_transcript("up", self._make_transcript(2))
        id2 = connected_db.donate_chat_transcript("down", self._make_transcript(2))
        deleted = connected_db.delete_chat_donations(ids=[id1])
        assert deleted == 1
        remaining = connected_db.get_chat_donations()
        assert len(remaining) == 1
        assert remaining[0]["id"] == id2

    def test_delete_chat_donations_empty_ids(self, connected_db):
        """Deletes nothing when ids is an empty list."""
        connected_db.donate_chat_transcript("up", self._make_transcript(2))
        deleted = connected_db.delete_chat_donations(ids=[])
        assert deleted == 0
        assert len(connected_db.get_chat_donations()) == 1


class TestGetValidationData:
    """Tests for get_validation_data() and get_validation_data_stats()."""

    def _donate(self, connected_db, uid="uid1", priority=3, search_term=None):
        connected_db.donate_validation_data({uid: {"priority": priority, "searchTerm": search_term}})

    def test_get_validation_data_empty(self, connected_db):
        """Returns empty list when no entries exist."""
        result = connected_db.get_validation_data()
        assert result == []

    def test_get_validation_data_returns_all(self, connected_db):
        """Returns all entries."""
        self._donate(connected_db, "uid1", 5)
        self._donate(connected_db, "uid2", 3)
        result = connected_db.get_validation_data()
        assert len(result) == 2

    def test_get_validation_data_limit(self, connected_db):
        """Respects limit parameter."""
        for i in range(5):
            self._donate(connected_db, f"uid{i}", i + 1)
        result = connected_db.get_validation_data(limit=3)
        assert len(result) == 3

    def test_get_validation_data_offset(self, connected_db):
        """Respects offset parameter for pagination."""
        for i in range(4):
            self._donate(connected_db, f"uid{i}", i + 1)
        result_all = connected_db.get_validation_data()
        result_offset = connected_db.get_validation_data(offset=2)
        assert len(result_offset) == 2
        assert result_offset[0]["id"] == result_all[2]["id"]

    def test_get_validation_data_includes_fields(self, connected_db):
        """Returned dicts include all expected fields."""
        self._donate(connected_db, "myuid", 4, "deep learning")
        result = connected_db.get_validation_data()
        assert result[0]["paper_uid"] == "myuid"
        assert result[0]["priority"] == 4
        assert result[0]["search_term"] == "deep learning"
        assert result[0]["donated_at"] is not None

    def test_get_validation_data_stats_empty(self, connected_db):
        """Stats returns zeros when no entries exist."""
        stats = connected_db.get_validation_data_stats()
        assert stats["total"] == 0
        assert stats["unique_papers"] == 0
        assert stats["avg_priority"] == 0.0
        assert stats["priority_distribution"] == {}

    def test_get_validation_data_stats_counts(self, connected_db):
        """Stats correctly counts totals and distribution."""
        self._donate(connected_db, "uid1", 5)
        self._donate(connected_db, "uid2", 3)
        self._donate(connected_db, "uid1", 5)  # Duplicate uid, different entry
        stats = connected_db.get_validation_data_stats()
        assert stats["total"] == 3
        assert stats["unique_papers"] == 2
        assert stats["avg_priority"] == pytest.approx((5 + 3 + 5) / 3)
        assert stats["priority_distribution"] == {5: 2, 3: 1}

    def test_delete_validation_data_all(self, connected_db):
        """Deletes all entries when no ids given."""
        self._donate(connected_db, "uid1", 3)
        self._donate(connected_db, "uid2", 4)
        deleted = connected_db.delete_validation_data()
        assert deleted == 2
        assert connected_db.get_validation_data() == []

    def test_delete_validation_data_by_ids(self, connected_db):
        """Deletes only specified entry IDs."""
        # Donate two entries and grab their IDs
        connected_db.donate_validation_data({"uid1": {"priority": 5, "searchTerm": None}})
        connected_db.donate_validation_data({"uid2": {"priority": 3, "searchTerm": None}})
        all_entries = connected_db.get_validation_data()
        id_to_delete = all_entries[0]["id"]
        deleted = connected_db.delete_validation_data(ids=[id_to_delete])
        assert deleted == 1
        remaining = connected_db.get_validation_data()
        assert len(remaining) == 1
        assert remaining[0]["id"] != id_to_delete

    def test_delete_validation_data_empty_ids(self, connected_db):
        """Deletes nothing when ids is an empty list."""
        self._donate(connected_db, "uid1", 3)
        deleted = connected_db.delete_validation_data(ids=[])
        assert deleted == 0
        assert len(connected_db.get_validation_data()) == 1


class TestGetConferenceYearsFromDb:
    """Tests for DatabaseManager.get_conference_years_from_db()."""

    def test_returns_empty_dict_when_db_empty(self, connected_db):
        """Returns an empty dict when no papers exist."""
        result = connected_db.get_conference_years_from_db()
        assert result == {}

    def test_returns_correct_mapping(self, connected_db):
        """Returns conference → sorted-descending years for existing papers."""
        from abstracts_explorer.plugin import LightweightPaper

        papers = [
            LightweightPaper(
                title="P1",
                authors=["A"],
                abstract="a",
                session="s",
                poster_position="p",
                year=2024,
                conference="NeurIPS",
            ),
            LightweightPaper(
                title="P2",
                authors=["A"],
                abstract="a",
                session="s",
                poster_position="p",
                year=2025,
                conference="NeurIPS",
            ),
            LightweightPaper(
                title="P3",
                authors=["A"],
                abstract="a",
                session="s",
                poster_position="p",
                year=2024,
                conference="ICLR",
            ),
        ]
        for p in papers:
            connected_db.add_paper(p)

        result = connected_db.get_conference_years_from_db()
        assert result["NeurIPS"] == [2025, 2024]
        assert result["ICLR"] == [2024]

    def test_returns_empty_dict_when_not_connected(self):
        """Returns empty dict (not an error) when session is None."""
        db = DatabaseManager()
        assert db.get_conference_years_from_db() == {}


class TestResolveDefaultConferenceYear:
    """Tests for DatabaseManager.resolve_default_conference_year()."""

    def _make_mock_db(self, conferences_with_years):
        """Return a partial mock that delegates resolve_default_conference_year to the real method."""
        from unittest.mock import MagicMock

        mock_db = MagicMock()
        mock_db.get_conferences.return_value = sorted(conferences_with_years.keys())
        mock_db.get_years.side_effect = lambda conference=None: (
            conferences_with_years.get(conference, [])
            if conference
            else sorted({y for years in conferences_with_years.values() for y in years}, reverse=True)
        )
        mock_db.get_conference_years_from_db.return_value = conferences_with_years
        mock_db.resolve_default_conference_year.side_effect = (
            lambda conf, year: DatabaseManager.resolve_default_conference_year(mock_db, conf, year)
        )
        return mock_db

    def test_returns_configured_values_when_match_exists(self):
        """Configured conference and year are returned when they match DB data."""
        mock_db = self._make_mock_db({"NeurIPS": [2025, 2024]})
        conf, year = mock_db.resolve_default_conference_year("NeurIPS", 2024)
        assert conf == "NeurIPS"
        assert year == 2024

    def test_case_insensitive_conference_match(self):
        """Conference name is matched case-insensitively."""
        mock_db = self._make_mock_db({"NeurIPS": [2025, 2024]})
        conf, year = mock_db.resolve_default_conference_year("neurips", 2025)
        assert conf == "NeurIPS"
        assert year == 2025

    def test_falls_back_to_most_recent_year_when_configured_year_missing(self):
        """Most recent DB year is used when configured year has no data."""
        mock_db = self._make_mock_db({"NeurIPS": [2025, 2024]})
        conf, year = mock_db.resolve_default_conference_year("NeurIPS", 2020)
        assert conf == "NeurIPS"
        assert year == 2025

    def test_falls_back_when_configured_conference_has_no_data(self):
        """Falls back to most-recent conference/year when configured conference is absent."""
        mock_db = self._make_mock_db({"ICLR": [2024]})
        conf, year = mock_db.resolve_default_conference_year("ICML", 2024)
        assert conf == "ICLR"
        assert year == 2024

    def test_most_recent_conference_chosen_when_no_default_configured(self):
        """Selects the conference with the most recent year when no default is set."""
        mock_db = self._make_mock_db({"NeurIPS": [2024], "ICLR": [2025]})
        conf, year = mock_db.resolve_default_conference_year("", None)
        assert conf == "ICLR"
        assert year == 2025

    def test_returns_configured_values_when_db_empty(self):
        """Configured values are returned unchanged when the DB has no data."""
        mock_db = self._make_mock_db({})
        conf, year = mock_db.resolve_default_conference_year("NeurIPS", 2024)
        assert conf == "NeurIPS"
        assert year == 2024


class TestResolveConferenceForUrl:
    """Tests for DatabaseManager.resolve_conference_for_url()."""

    def _make_mock_db(self, db_conference_years):
        """Return a partial mock that delegates resolve_conference_for_url to the real method."""
        from unittest.mock import MagicMock

        mock_db = MagicMock()
        mock_db.get_conference_years_from_db.return_value = db_conference_years
        mock_db.resolve_conference_for_url.side_effect = lambda url_path: DatabaseManager.resolve_conference_for_url(
            mock_db, url_path
        )
        return mock_db

    @patch("abstracts_explorer.plugin.resolve_conference_from_url", return_value="NeurIPS")
    @patch(
        "abstracts_explorer.plugin.get_available_filters",
        return_value={"conferences": ["NeurIPS"], "years": [2025], "conference_years": {"NeurIPS": [2025]}},
    )
    def test_valid_conference_with_data(self, _mock_filters, _mock_resolve):
        """Conference found and has data → returns conference name."""
        mock_db = self._make_mock_db({"NeurIPS": [2025]})
        result = mock_db.resolve_conference_for_url("neurips")
        assert result["conference"] == "NeurIPS"
        assert result["error"] is None

    @patch("abstracts_explorer.plugin.resolve_conference_from_url", return_value="ICML")
    @patch(
        "abstracts_explorer.plugin.get_available_filters",
        return_value={"conferences": ["ICML"], "years": [2025], "conference_years": {"ICML": [2025]}},
    )
    def test_known_conference_without_data(self, _mock_filters, _mock_resolve):
        """Conference found in plugins but no data in DB → returns error."""
        mock_db = self._make_mock_db({"NeurIPS": [2025]})
        result = mock_db.resolve_conference_for_url("icml")
        assert result["conference"] is None
        assert "No data available" in result["error"]["message"]
        assert "NeurIPS" in result["error"]["available_conferences"]

    @patch("abstracts_explorer.plugin.resolve_conference_from_url", return_value=None)
    @patch(
        "abstracts_explorer.plugin.get_available_filters",
        return_value={"conferences": ["NeurIPS"], "years": [2025], "conference_years": {"NeurIPS": [2025]}},
    )
    def test_unknown_conference(self, _mock_filters, _mock_resolve):
        """Conference not found at all → returns error with available conferences."""
        mock_db = self._make_mock_db({"NeurIPS": [2025]})
        result = mock_db.resolve_conference_for_url("unknownconf")
        assert result["conference"] is None
        assert "not found" in result["error"]["message"]
        assert "NeurIPS" in result["error"]["available_conferences"]

    @patch("abstracts_explorer.plugin.resolve_conference_from_url", return_value=None)
    @patch(
        "abstracts_explorer.plugin.get_available_filters",
        return_value={"conferences": [], "years": [], "conference_years": {}},
    )
    def test_db_conference_fallback(self, _mock_filters, _mock_resolve):
        """Conference in DB but not in plugins → resolved via DB lookup."""
        mock_db = self._make_mock_db({"CustomConf": [2025]})
        result = mock_db.resolve_conference_for_url("customconf")
        assert result["conference"] == "CustomConf"
        assert result["error"] is None

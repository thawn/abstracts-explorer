"""
Tests for authors table functionality.
"""

import pytest
import sqlite3
from pathlib import Path

from neurips_abstracts.database import DatabaseManager, DatabaseError

# Fixtures are now imported from conftest.py:
# - sample_neurips_data: Sample NeurIPS data with authors table structure


@pytest.fixture
def db_manager(tmp_path):
    """Create a test database manager."""
    db_path = tmp_path / "test_authors.db"
    manager = DatabaseManager(db_path)
    manager.connect()
    manager.create_tables()
    yield manager
    manager.close()


def test_authors_table_creation(db_manager):
    """Test that authors table is created correctly."""
    cursor = db_manager.connection.cursor()

    # Check that authors table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='authors'")
    assert cursor.fetchone() is not None

    # Check table schema
    cursor.execute("PRAGMA table_info(authors)")
    columns = {row[1]: row[2] for row in cursor.fetchall()}

    assert "id" in columns
    assert "fullname" in columns
    assert "url" in columns
    assert "institution" in columns
    assert "created_at" in columns


def test_paper_authors_table_creation(db_manager):
    """Test that paper_authors junction table is created correctly."""
    cursor = db_manager.connection.cursor()

    # Check that paper_authors table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='paper_authors'")
    assert cursor.fetchone() is not None

    # Check table schema
    cursor.execute("PRAGMA table_info(paper_authors)")
    columns = {row[1]: row[2] for row in cursor.fetchall()}

    assert "paper_id" in columns
    assert "author_id" in columns
    assert "author_order" in columns


def test_load_data_with_authors(db_manager, sample_neurips_data):
    """Test loading data populates authors and paper_authors tables."""
    # Load data
    count = db_manager.load_json_data(sample_neurips_data)
    assert count == 2

    # Check authors were inserted
    cursor = db_manager.connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM authors")
    author_count = cursor.fetchone()[0]
    assert author_count == 3  # Three unique authors

    # Check paper_authors relationships
    cursor.execute("SELECT COUNT(*) FROM paper_authors")
    relationship_count = cursor.fetchone()[0]
    assert relationship_count == 4  # 2 authors for paper 1, 2 for paper 2

    # Check author data
    cursor.execute("SELECT * FROM authors WHERE id = 457880")
    author = cursor.fetchone()
    assert author is not None
    # Note: sqlite3.Row indices are by position
    cursor.execute("PRAGMA table_info(authors)")
    columns = [col[1] for col in cursor.fetchall()]
    author_dict = dict(zip(columns, author))
    assert author_dict["fullname"] == "Miaomiao Huang"
    assert author_dict["institution"] == "Northeastern University"


def test_get_paper_authors(db_manager, sample_neurips_data):
    """Test getting authors for a specific paper."""
    db_manager.load_json_data(sample_neurips_data)

    # Get authors for first paper
    authors = db_manager.get_paper_authors(123456)
    assert len(authors) == 2

    # Check ordering
    assert authors[0]["fullname"] == "Miaomiao Huang"
    assert authors[0]["author_order"] == 1
    assert authors[1]["fullname"] == "John Smith"
    assert authors[1]["author_order"] == 2


def test_get_author_papers(db_manager, sample_neurips_data):
    """Test getting papers by a specific author."""
    db_manager.load_json_data(sample_neurips_data)

    # Get papers by John Smith (appears in both papers)
    papers = db_manager.get_author_papers(457881)
    assert len(papers) == 2

    paper_names = {paper["name"] for paper in papers}
    assert "Deep Learning with Neural Networks" in paper_names
    assert "Advances in Computer Vision" in paper_names


def test_search_authors_by_name(db_manager, sample_neurips_data):
    """Test searching authors by name."""
    db_manager.load_json_data(sample_neurips_data)

    # Search by partial name
    authors = db_manager.search_authors(name="Smith")
    assert len(authors) == 1
    assert authors[0]["fullname"] == "John Smith"

    # Search by partial name (case insensitive)
    authors = db_manager.search_authors(name="huang")
    assert len(authors) == 1
    assert authors[0]["fullname"] == "Miaomiao Huang"


def test_search_authors_by_institution(db_manager, sample_neurips_data):
    """Test searching authors by institution."""
    db_manager.load_json_data(sample_neurips_data)

    # Search by institution
    authors = db_manager.search_authors(institution="MIT")
    assert len(authors) == 1
    assert authors[0]["fullname"] == "John Smith"

    # Search by partial institution name
    authors = db_manager.search_authors(institution="University")
    assert len(authors) == 2  # Northeastern University and Stanford University


def test_search_authors_combined(db_manager, sample_neurips_data):
    """Test searching authors by name and institution."""
    db_manager.load_json_data(sample_neurips_data)

    # Search by name and institution
    authors = db_manager.search_authors(name="Jane", institution="Stanford")
    assert len(authors) == 1
    assert authors[0]["fullname"] == "Jane Doe"


def test_get_author_count(db_manager, sample_neurips_data):
    """Test getting total author count."""
    db_manager.load_json_data(sample_neurips_data)

    count = db_manager.get_author_count()
    assert count == 3


def test_duplicate_authors_not_inserted(db_manager, sample_neurips_data):
    """Test that duplicate authors are not inserted."""
    # Load data twice
    db_manager.load_json_data(sample_neurips_data)
    db_manager.load_json_data(sample_neurips_data)

    # Check that authors are still unique
    count = db_manager.get_author_count()
    assert count == 3  # Should still be 3, not 6


def test_author_order_preserved(db_manager, sample_neurips_data):
    """Test that author order is preserved in paper_authors table."""
    db_manager.load_json_data(sample_neurips_data)

    # Get authors for first paper
    authors = db_manager.get_paper_authors(123456)

    # Check that they are returned in the correct order
    assert authors[0]["author_order"] == 1
    assert authors[0]["fullname"] == "Miaomiao Huang"
    assert authors[1]["author_order"] == 2
    assert authors[1]["fullname"] == "John Smith"


def test_authors_with_no_institution(db_manager):
    """Test handling authors without institution."""
    data = [
        {
            "id": 123456,
            "uid": "test123",
            "name": "Test Paper",
            "abstract": "Test abstract",
            "authors": [
                {
                    "id": 999999,
                    "fullname": "No Institution Author",
                    "url": "http://neurips.cc/api/miniconf/users/999999?format=json",
                    "institution": None,
                }
            ],
            "keywords": "test",
            "topic": "Test",
            "decision": "Accept (poster)",
            "session": "Test Session",
            "eventtype": "Poster",
            "event_type": "poster_template",
            "room_name": "Test Room",
            "virtualsite_url": "https://test.com",
            "url": "https://test.com",
            "sourceid": 123456,
            "sourceurl": "https://test.com",
            "starttime": "2025-12-10T10:00:00",
            "endtime": "2025-12-10T12:00:00",
            "starttime2": None,
            "endtime2": None,
            "diversity_event": False,
            "paper_url": "https://test.com",
            "paper_pdf_url": "https://test.com/pdf",
            "children_url": None,
            "children": [],
            "children_ids": [],
            "parent1": None,
            "parent2": None,
            "parent2_id": None,
            "eventmedia": None,
            "show_in_schedule_overview": True,
            "visible": True,
            "poster_position": "A-1",
            "schedule_html": "<p>Test</p>",
            "latitude": 0.0,
            "longitude": 0.0,
            "related_events": [],
            "related_events_ids": [],
        }
    ]

    count = db_manager.load_json_data(data)
    assert count == 1

    # Check author was inserted with None institution
    authors = db_manager.search_authors(name="No Institution")
    assert len(authors) == 1
    assert authors[0]["institution"] is None

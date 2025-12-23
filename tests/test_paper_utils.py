"""
Tests for paper_utils module.

Tests the shared utilities for formatting papers from various sources.
"""

import pytest
from unittest.mock import Mock

from neurips_abstracts.paper_utils import (
    get_paper_with_authors,
    format_search_results,
    build_context_from_papers,
    PaperFormattingError,
)


class TestGetPaperWithAuthors:
    """Tests for get_paper_with_authors function."""

    def test_get_paper_with_authors_success(self):
        """Test successful paper retrieval with authors."""
        # Mock database
        mock_db = Mock()
        mock_db.query.return_value = [
            {
                "uid": "abc123",
                "title": "Test Paper",
                "abstract": "Test abstract",
                "authors": "John Doe, Jane Smith",
                "session": "ML",
            }
        ]

        paper = get_paper_with_authors(mock_db, "abc123")

        assert paper["uid"] == "abc123"
        assert paper["title"] == "Test Paper"
        assert paper["authors"] == ["John Doe", "Jane Smith"]
        mock_db.query.assert_called_once_with("SELECT * FROM papers WHERE uid = ?", ("abc123",))

    def test_get_paper_with_authors_invalid_id(self):
        """Test that invalid paper UIDs raise error."""
        mock_db = Mock()

        # Test empty string
        with pytest.raises(PaperFormattingError, match="Invalid paper_uid"):
            get_paper_with_authors(mock_db, "")

        # Test whitespace only
        with pytest.raises(PaperFormattingError, match="Invalid paper_uid"):
            get_paper_with_authors(mock_db, "   ")

        # Test non-string
        with pytest.raises(PaperFormattingError, match="Invalid paper_uid"):
            get_paper_with_authors(mock_db, 123)

    def test_get_paper_with_authors_no_database(self):
        """Test that missing database raises error."""
        with pytest.raises(PaperFormattingError, match="Database connection is required"):
            get_paper_with_authors(None, "abc123")

    def test_get_paper_with_authors_not_found(self):
        """Test that missing paper raises error."""
        mock_db = Mock()
        mock_db.query.return_value = []

        with pytest.raises(PaperFormattingError, match="not found in database"):
            get_paper_with_authors(mock_db, "nonexistent")

    def test_get_paper_with_authors_no_authors(self):
        """Test paper with no authors."""
        mock_db = Mock()
        mock_db.query.return_value = [{"uid": "abc123", "title": "Test Paper", "authors": None}]

        paper = get_paper_with_authors(mock_db, "abc123")

        assert paper["authors"] == []

    def test_get_paper_with_authors_database_error(self):
        """Test database error handling."""
        mock_db = Mock()
        mock_db.query.side_effect = Exception("Database connection lost")

        with pytest.raises(PaperFormattingError, match="Failed to retrieve paper"):
            get_paper_with_authors(mock_db, "abc123")


class TestFormatSearchResults:
    """Tests for format_search_results function."""

    def test_format_search_results_success(self):
        """Test successful formatting of search results."""
        search_results = {
            "ids": [["abc123", "def456"]],
            "distances": [[0.2, 0.5]],
            "documents": [["Abstract 1", "Abstract 2"]],
            "metadatas": [[{}, {}]],
        }

        mock_db = Mock()
        mock_db.query.side_effect = [
            [{"uid": "abc123", "title": "Paper 1", "abstract": "Abstract 1", "authors": "Author 1"}],
            [{"uid": "def456", "title": "Paper 2", "abstract": "Abstract 2", "authors": "Author 2"}],
        ]

        papers = format_search_results(search_results, mock_db)

        assert len(papers) == 2
        assert papers[0]["uid"] == "abc123"
        assert papers[0]["title"] == "Paper 1"
        assert papers[0]["authors"] == ["Author 1"]
        assert papers[0]["distance"] == 0.2
        assert papers[0]["similarity"] == 0.8
        assert papers[1]["uid"] == "def456"
        assert papers[1]["similarity"] == 0.5

    def test_format_search_results_empty_results(self):
        """Test empty search results."""
        search_results = {"ids": [[]], "distances": [[]], "documents": [[]]}
        mock_db = Mock()

        papers = format_search_results(search_results, mock_db)

        assert papers == []
        mock_db.query.assert_not_called()

    def test_format_search_results_invalid_input(self):
        """Test invalid search results structure."""
        # Not a dict
        with pytest.raises(PaperFormattingError, match="must be a dictionary"):
            format_search_results("invalid", Mock())

        # Missing ids field
        with pytest.raises(PaperFormattingError, match="must contain 'ids'"):
            format_search_results({}, Mock())

        # ids is None
        with pytest.raises(PaperFormattingError, match="must contain 'ids'"):
            format_search_results({"ids": None}, Mock())

    def test_format_search_results_no_database(self):
        """Test that missing database raises error."""
        search_results = {"ids": [["abc123", "def456"]]}

        with pytest.raises(PaperFormattingError, match="Database connection is required"):
            format_search_results(search_results, None)

    def test_format_search_results_inconsistent_lengths(self):
        """Test inconsistent result lengths."""
        search_results = {
            "ids": [["abc123", "def456"]],
            "distances": [[0.2]],  # Only 1 distance for 2 IDs
        }
        mock_db = Mock()

        with pytest.raises(PaperFormattingError, match="Inconsistent result lengths"):
            format_search_results(search_results, mock_db)

    def test_format_search_results_skip_invalid_papers(self):
        """Test that invalid papers are skipped with warning."""
        search_results = {
            "ids": [["abc123", "nonexist", "def456"]],  # nonexist doesn't exist
            "distances": [[0.2, 0.3, 0.5]],
        }

        mock_db = Mock()
        mock_db.query.side_effect = [
            [{"uid": "abc123", "title": "Paper 1", "authors": "Author 1"}],
            [],  # Paper nonexist not found
            [{"uid": "def456", "title": "Paper 2", "authors": "Author 2"}],
        ]

        papers = format_search_results(search_results, mock_db)

        # Should have 2 papers, skipping the invalid one
        assert len(papers) == 2
        assert papers[0]["uid"] == "abc123"
        assert papers[1]["uid"] == "def456"

    def test_format_search_results_all_invalid(self):
        """Test that all invalid papers raises error."""
        search_results = {"ids": [["nonexist1", "nonexist2"]]}

        mock_db = Mock()
        mock_db.query.return_value = []  # No papers found

        with pytest.raises(PaperFormattingError, match="No valid papers could be formatted"):
            format_search_results(search_results, mock_db)

    def test_format_search_results_without_documents(self):
        """Test formatting without including documents."""
        search_results = {
            "ids": [["abc123"]],
            "distances": [[0.2]],
        }

        mock_db = Mock()
        mock_db.query.return_value = [
            {"uid": "abc123", "title": "Paper 1", "abstract": "From DB", "authors": "Author 1"}
        ]

        papers = format_search_results(search_results, mock_db, include_documents=False)

        assert len(papers) == 1
        assert papers[0]["abstract"] == "From DB"

    def test_format_search_results_with_documents_override(self):
        """Test that documents from search results override missing abstracts."""
        search_results = {
            "ids": [["abc123"]],
            "documents": [["From search"]],
        }

        mock_db = Mock()
        mock_db.query.return_value = [{"uid": "abc123", "title": "Paper 1", "authors": ""}]  # No abstract

        papers = format_search_results(search_results, mock_db, include_documents=True)

        assert papers[0]["abstract"] == "From search"


class TestBuildContextFromPapers:
    """Tests for build_context_from_papers function."""

    def test_build_context_from_papers_success(self):
        """Test successful context building."""
        papers = [
            {
                "title": "Paper 1",
                "authors": ["Author 1", "Author 2"],
                "abstract": "Abstract 1",
                "topic": "ML",
                "decision": "Accept",
            },
            {
                "title": "Paper 2",
                "authors": ["Author 3"],
                "abstract": "Abstract 2",
            },
        ]

        context = build_context_from_papers(papers)

        assert "Paper 1:" in context
        assert "Title: Paper 1" in context
        assert "Authors: Author 1, Author 2" in context
        assert "Abstract: Abstract 1" in context
        assert "Topic: ML" in context
        assert "Decision: Accept" in context
        assert "Paper 2:" in context
        assert "Title: Paper 2" in context
        assert "Authors: Author 3" in context

    def test_build_context_from_papers_empty_list(self):
        """Test that empty papers list raises error."""
        with pytest.raises(PaperFormattingError, match="papers list is empty"):
            build_context_from_papers([])

    def test_build_context_from_papers_not_list(self):
        """Test that non-list input raises error."""
        with pytest.raises(PaperFormattingError, match="must be a list"):
            build_context_from_papers("not a list")

    def test_build_context_from_papers_invalid_paper(self):
        """Test that non-dict papers raise error."""
        with pytest.raises(PaperFormattingError, match="is not a dictionary"):
            build_context_from_papers(["not a dict"])

    def test_build_context_from_papers_missing_title(self):
        """Test that missing title raises error."""
        papers = [{"authors": ["Author"], "abstract": "Abstract"}]

        with pytest.raises(PaperFormattingError, match="missing required 'title'"):
            build_context_from_papers(papers)

    def test_build_context_from_papers_missing_authors(self):
        """Test papers with missing authors (should use N/A)."""
        papers = [{"title": "Paper", "abstract": "Abstract"}]

        context = build_context_from_papers(papers)

        assert "Authors: N/A" in context

    def test_build_context_from_papers_empty_authors(self):
        """Test papers with empty authors list."""
        papers = [{"title": "Paper", "authors": [], "abstract": "Abstract"}]

        context = build_context_from_papers(papers)

        assert "Authors: N/A" in context

    def test_build_context_from_papers_missing_abstract(self):
        """Test papers with missing abstract (should use N/A)."""
        papers = [{"title": "Paper", "authors": ["Author"]}]

        context = build_context_from_papers(papers)

        assert "Abstract: N/A" in context

    def test_build_context_from_papers_authors_as_string(self):
        """Test handling authors as string."""
        papers = [
            {
                "title": "Paper",
                "authors": "Single Author",
                "abstract": "Abstract",
            }
        ]

        context = build_context_from_papers(papers)

        assert "Authors: Single Author" in context

    def test_build_context_from_papers_optional_fields(self):
        """Test that optional fields are included when present."""
        papers = [
            {
                "title": "Paper",
                "authors": ["Author"],
                "abstract": "Abstract",
                "topic": "Computer Vision",
                "decision": "Reject",
            }
        ]

        context = build_context_from_papers(papers)

        assert "Topic: Computer Vision" in context
        assert "Decision: Reject" in context

    def test_build_context_from_papers_multiple_papers(self):
        """Test context with multiple papers is properly numbered."""
        papers = [
            {"title": "First", "authors": ["A1"], "abstract": "Abs1"},
            {"title": "Second", "authors": ["A2"], "abstract": "Abs2"},
            {"title": "Third", "authors": ["A3"], "abstract": "Abs3"},
        ]

        context = build_context_from_papers(papers)

        assert "Paper 1:" in context
        assert "Paper 2:" in context
        assert "Paper 3:" in context
        assert context.index("Paper 1:") < context.index("Paper 2:")
        assert context.index("Paper 2:") < context.index("Paper 3:")

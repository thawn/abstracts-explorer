"""
Tests for paper_utils module.

Tests the shared utilities for formatting papers from various sources.
"""

import pytest
from unittest.mock import Mock

from abstracts_explorer.paper_utils import (
    format_search_results,
    PaperFormattingError,
)


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
        mock_db.get_papers_by_uids.return_value = [
            {"uid": "abc123", "title": "Paper 1", "abstract": "Abstract 1", "authors": ["Author 1"]},
            {"uid": "def456", "title": "Paper 2", "abstract": "Abstract 2", "authors": ["Author 2"]},
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
        mock_db.get_papers_by_uids.assert_called_once_with(["abc123", "def456"])

    def test_format_search_results_empty_results(self):
        """Test empty search results."""
        search_results = {"ids": [[]], "distances": [[]], "documents": [[]]}
        mock_db = Mock()

        papers = format_search_results(search_results, mock_db)

        assert papers == []
        mock_db.get_papers_by_uids.assert_not_called()

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
        mock_db.get_papers_by_uids.return_value = [
            {"uid": "abc123", "title": "Paper 1", "authors": ["Author 1"]},
            {"uid": "def456", "title": "Paper 2", "authors": ["Author 2"]},
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
        mock_db.get_papers_by_uids.return_value = []

        with pytest.raises(PaperFormattingError, match="No valid papers could be formatted"):
            format_search_results(search_results, mock_db)

    def test_format_search_results_uses_database_abstract(self):
        """Abstracts always come from the database."""
        search_results = {
            "ids": [["abc123"]],
            "distances": [[0.2]],
        }

        mock_db = Mock()
        mock_db.get_papers_by_uids.return_value = [
            {
                "uid": "abc123",
                "title": "Paper 1",
                "abstract": "From DB",
                "authors": ["Author 1"],
            }
        ]

        papers = format_search_results(search_results, mock_db)

        assert len(papers) == 1
        assert papers[0]["abstract"] == "From DB"

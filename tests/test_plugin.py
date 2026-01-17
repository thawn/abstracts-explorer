"""
Tests for plugin.py module.

This module tests all plugin-related functionality including:
- Plugin helper functions (sanitize_author_names, convert_to_lightweight_schema)
- Plugin data models (LightweightPaper, validation)
- Plugin base classes and interfaces  
- Year/conference field handling
"""

import pytest
import tempfile
from pathlib import Path
from pydantic import ValidationError
from unittest.mock import patch, Mock

from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.config import get_config
from abstracts_explorer.plugin import (
    sanitize_author_names,
    convert_to_lightweight_schema,
    validate_lightweight_paper,
    LightweightPaper,
)
from abstracts_explorer.plugins import (
    validate_lightweight_papers,
)
from abstracts_explorer.plugins.neurips_downloader import NeurIPSDownloaderPlugin
from abstracts_explorer.plugins.icml_downloader import ICMLDownloaderPlugin
from abstracts_explorer.plugins.ml4ps_downloader import ML4PSDownloaderPlugin


# ============================================================
# Tests from test_plugin_helpers.py
# ============================================================

class TestSanitizeAuthorNames:
    """Tests for sanitize_author_names helper function."""

    def test_basic_semicolon_removal(self):
        """Test basic semicolon removal from author names."""
        authors = ["John Doe", "Jane; Smith", "Bob;Johnson"]
        result = sanitize_author_names(authors)
        assert result == ["John Doe", "Jane Smith", "Bob Johnson"]

    def test_single_name_no_semicolons(self):
        """Test that names without semicolons are unchanged."""
        authors = ["Alice Johnson"]
        result = sanitize_author_names(authors)
        assert result == ["Alice Johnson"]

    def test_empty_list(self):
        """Test that empty list returns empty list."""
        authors = []
        result = sanitize_author_names(authors)
        assert result == []

    def test_multiple_semicolons(self):
        """Test handling of multiple consecutive semicolons."""
        authors = ["Test;Semicolon", "Multi;;Semicolons", "Triple;;;Test"]
        result = sanitize_author_names(authors)
        assert result == ["Test Semicolon", "Multi Semicolons", "Triple Test"]

    def test_whitespace_normalization(self):
        """Test that whitespace is normalized correctly."""
        authors = ["  Spaces  ;  Around  ", "Multiple   Spaces"]
        result = sanitize_author_names(authors)
        assert result == ["Spaces Around", "Multiple Spaces"]

    def test_semicolon_at_start(self):
        """Test semicolon at the start of name."""
        authors = [";Leading", "Normal Name"]
        result = sanitize_author_names(authors)
        assert result == ["Leading", "Normal Name"]

    def test_semicolon_at_end(self):
        """Test semicolon at the end of name."""
        authors = ["Trailing;", "Normal Name"]
        result = sanitize_author_names(authors)
        assert result == ["Trailing", "Normal Name"]

    def test_only_semicolons(self):
        """Test name that is only semicolons."""
        authors = [";;;", "Normal Name"]
        result = sanitize_author_names(authors)
        assert result == ["", "Normal Name"]

    def test_unicode_names(self):
        """Test that unicode characters are preserved."""
        authors = ["José García", "Müller;Schmidt", "李明"]
        result = sanitize_author_names(authors)
        assert result == ["José García", "Müller Schmidt", "李明"]

    def test_preserves_hyphens_and_apostrophes(self):
        """Test that hyphens and apostrophes in names are preserved."""
        authors = ["O'Brien", "Anne-Marie", "Van Der Berg"]
        result = sanitize_author_names(authors)
        assert result == ["O'Brien", "Anne-Marie", "Van Der Berg"]


class TestConvertNeuripsToLightweightSchema:
    """Tests for convert_to_lightweight_schema helper function."""

    def test_basic_conversion(self):
        """Test basic conversion with all required fields."""
        papers = [
            {
                "id": 123,
                "title": "Test Paper",
                "authors": [
                    {"id": 1, "fullname": "John Doe", "institution": "MIT"},
                    {"id": 2, "fullname": "Jane Smith", "institution": "Stanford"},
                ],
                "abstract": "Test abstract",
                "session": "Session A",
                "poster_position": "A-42",
            }
        ]

        result = convert_to_lightweight_schema(papers)

        assert len(result) == 1
        assert result[0]["title"] == "Test Paper"
        assert result[0]["authors"] == ["John Doe", "Jane Smith"]
        assert result[0]["abstract"] == "Test abstract"
        assert result[0]["session"] == "Session A"
        assert result[0]["poster_position"] == "A-42"
        assert result[0]["original_id"] == 123

    def test_legacy_name_field(self):
        """Test conversion of legacy 'name' field to 'title'."""
        papers = [
            {
                "id": 1,
                "name": "Legacy Paper",  # Using 'name' instead of 'title'
                "authors": [{"fullname": "John Doe"}],
                "abstract": "Abstract",
                "session": "Session A",
                "poster_position": "A1",
            }
        ]

        result = convert_to_lightweight_schema(papers)

        assert result[0]["title"] == "Legacy Paper"
        assert "name" not in result[0]

    def test_authors_as_strings(self):
        """Test conversion when authors are already strings."""
        papers = [
            {
                "id": 1,
                "title": "Test Paper",
                "authors": ["John Doe", "Jane Smith"],  # Already strings
                "abstract": "Abstract",
                "session": "Session A",
                "poster_position": "A1",
            }
        ]

        result = convert_to_lightweight_schema(papers)

        assert result[0]["authors"] == ["John Doe", "Jane Smith"]

    def test_authors_as_semicolon_separated_string(self):
        """Test conversion when authors is a semicolon-separated string."""
        papers = [
            {
                "id": 1,
                "title": "Test Paper",
                "authors": "John Doe; Jane Smith; Bob Johnson",
                "abstract": "Abstract",
                "session": "Session A",
                "poster_position": "A1",
            }
        ]

        result = convert_to_lightweight_schema(papers)

        assert result[0]["authors"] == ["John Doe", "Jane Smith", "Bob Johnson"]

    def test_optional_fields(self):
        """Test that optional fields are included when present."""
        papers = [
            {
                "id": 1,
                "title": "Test Paper",
                "authors": ["John Doe"],
                "abstract": "Abstract",
                "session": "Session A",
                "poster_position": "A1",
                "paper_pdf_url": "https://example.com/paper.pdf",
                "poster_image_url": "https://example.com/poster.png",
                "url": "https://openreview.net/forum?id=abc",
                "room_name": "Hall A",
                "keywords": ["ML", "AI"],
                "starttime": "2025-12-10T10:00:00",
                "endtime": "2025-12-10T12:00:00",
                "year": 2025,
                "conference": "NeurIPS",
            }
        ]

        result = convert_to_lightweight_schema(papers)

        assert result[0]["paper_pdf_url"] == "https://example.com/paper.pdf"
        assert result[0]["poster_image_url"] == "https://example.com/poster.png"
        assert result[0]["url"] == "https://openreview.net/forum?id=abc"
        assert result[0]["room_name"] == "Hall A"
        assert result[0]["keywords"] == ["ML", "AI"]
        assert result[0]["starttime"] == "2025-12-10T10:00:00"
        assert result[0]["endtime"] == "2025-12-10T12:00:00"
        assert result[0]["year"] == 2025
        assert result[0]["conference"] == "NeurIPS"

    def test_keywords_as_string(self):
        """Test conversion of keywords from comma-separated string to list."""
        papers = [
            {
                "id": 1,
                "title": "Test Paper",
                "authors": ["John Doe"],
                "abstract": "Abstract",
                "session": "Session A",
                "poster_position": "A1",
                "keywords": "machine learning, deep learning, AI",
            }
        ]

        result = convert_to_lightweight_schema(papers)

        assert result[0]["keywords"] == ["machine learning", "deep learning", "AI"]

    def test_award_extraction(self):
        """Test extraction of award from decision field."""
        papers = [
            {
                "id": 1,
                "title": "Award Paper",
                "authors": ["John Doe"],
                "abstract": "Great work",
                "session": "Session A",
                "poster_position": "A1",
                "decision": "Best Paper Award",
            }
        ]

        result = convert_to_lightweight_schema(papers)

        assert result[0]["award"] == "Best Paper Award"

    def test_award_field_direct(self):
        """Test that direct award field is preserved."""
        papers = [
            {
                "id": 1,
                "title": "Award Paper",
                "authors": ["John Doe"],
                "abstract": "Great work",
                "session": "Session A",
                "poster_position": "A1",
                "award": "Outstanding Paper",
            }
        ]

        result = convert_to_lightweight_schema(papers)

        assert result[0]["award"] == "Outstanding Paper"

    def test_decision_without_award(self):
        """Test that regular decision is not converted to award."""
        papers = [
            {
                "id": 1,
                "title": "Test Paper",
                "authors": ["John Doe"],
                "abstract": "Abstract",
                "session": "Session A",
                "poster_position": "A1",
                "decision": "Accept (poster)",
            }
        ]

        result = convert_to_lightweight_schema(papers)

        assert "award" not in result[0]

    def test_decision_none_no_error(self):
        """Test that None decision value doesn't cause an error."""
        papers = [
            {
                "id": 1,
                "title": "Test Paper",
                "authors": ["John Doe"],
                "abstract": "Abstract",
                "session": "Session A",
                "poster_position": "A1",
                "decision": None,  # Decision is None, not missing
            }
        ]

        result = convert_to_lightweight_schema(papers)

        assert "award" not in result[0]
        # Should not raise AttributeError

    def test_none_field_values_converted_to_empty_strings(self):
        """Test that None field values are converted to empty strings."""
        papers = [
            {
                "id": 1,
                "title": "Test Paper",
                "authors": ["John Doe"],
                "abstract": None,  # None instead of string
                "session": None,  # None instead of string
                "poster_position": None,  # None instead of string
            }
        ]

        result = convert_to_lightweight_schema(papers)

        # All None values should be converted to appropriate defaults
        assert result[0]["abstract"] == ""
        assert result[0]["session"] == "No session"  # Default for None session
        assert result[0]["poster_position"] == ""
        # Should not raise validation errors when used with LightweightPaper

    def test_author_names_with_semicolons_sanitized(self):
        """Test that author names with semicolons are sanitized."""
        papers = [
            {
                "id": 1,
                "title": "Test Paper",
                "authors": [
                    {"fullname": "John; Doe"},  # Semicolon in name
                    {"fullname": "Jane;Smith"},  # Semicolon without space
                ],
                "abstract": "Abstract",
                "session": "Session A",
                "poster_position": "A1",
            }
        ]

        result = convert_to_lightweight_schema(papers)

        # Semicolons should be replaced with spaces
        assert result[0]["authors"] == ["John Doe", "Jane Smith"]
        # Should not raise validation errors when used with LightweightPaper

    def test_multiple_papers(self):
        """Test conversion of multiple papers at once."""
        papers = [
            {
                "id": 1,
                "title": "Paper 1",
                "authors": ["Alice"],
                "abstract": "Abstract 1",
                "session": "Session A",
                "poster_position": "A1",
            },
            {
                "id": 2,
                "title": "Paper 2",
                "authors": ["Bob"],
                "abstract": "Abstract 2",
                "session": "Session B",
                "poster_position": "B1",
            },
            {
                "id": 3,
                "title": "Paper 3",
                "authors": ["Charlie"],
                "abstract": "Abstract 3",
                "session": "Session C",
                "poster_position": "C1",
            },
        ]

        result = convert_to_lightweight_schema(papers)

        assert len(result) == 3
        assert result[0]["title"] == "Paper 1"
        assert result[1]["title"] == "Paper 2"
        assert result[2]["title"] == "Paper 3"

    def test_paper_without_title(self):
        """Test that papers without title are skipped."""
        papers = [
            {
                "id": 1,
                # No title or name field
                "authors": ["John Doe"],
                "abstract": "Abstract",
                "session": "Session A",
                "poster_position": "A1",
            },
            {
                "id": 2,
                "title": "Valid Paper",
                "authors": ["Jane Doe"],
                "abstract": "Abstract",
                "session": "Session B",
                "poster_position": "B1",
            },
        ]

        result = convert_to_lightweight_schema(papers)

        # First paper should be skipped
        assert len(result) == 1
        assert result[0]["title"] == "Valid Paper"

    def test_empty_title(self):
        """Test that papers with empty title are skipped."""
        papers = [
            {
                "id": 1,
                "title": "",  # Empty title
                "authors": ["John Doe"],
                "abstract": "Abstract",
                "session": "Session A",
                "poster_position": "A1",
            },
        ]

        result = convert_to_lightweight_schema(papers)

        assert len(result) == 0

    def test_author_with_name_field(self):
        """Test author dict with 'name' instead of 'fullname'."""
        papers = [
            {
                "id": 1,
                "title": "Test Paper",
                "authors": [
                    {"id": 1, "name": "John Doe"},  # Using 'name' instead of 'fullname'
                ],
                "abstract": "Abstract",
                "session": "Session A",
                "poster_position": "A1",
            }
        ]

        result = convert_to_lightweight_schema(papers)

        assert result[0]["authors"] == ["John Doe"]

    def test_author_without_name(self):
        """Test that authors without name/fullname are skipped."""
        papers = [
            {
                "id": 1,
                "title": "Test Paper",
                "authors": [
                    {"id": 1, "fullname": "John Doe"},
                    {"id": 2, "institution": "MIT"},  # No name
                    {"id": 3, "fullname": "Jane Smith"},
                ],
                "abstract": "Abstract",
                "session": "Session A",
                "poster_position": "A1",
            }
        ]

        result = convert_to_lightweight_schema(papers)

        assert result[0]["authors"] == ["John Doe", "Jane Smith"]

    def test_extra_neurips_fields_removed(self):
        """Test that NeurIPS-specific fields are not included in lightweight schema."""
        papers = [
            {
                "id": 1,
                "title": "Test Paper",
                "authors": ["John Doe"],
                "abstract": "Abstract",
                "session": "Session A",
                "poster_position": "A1",
                # Extra NeurIPS fields that should not appear in lightweight
                "topic": "Machine Learning",
                "eventtype": "Poster",
                "event_type": "poster_template",
                "virtualsite_url": "https://virtual.neurips.cc",
                "sourceid": 456,
                "sourceurl": "https://source.com",
                "diversity_event": False,
                "show_in_schedule_overview": True,
                "visible": True,
                "schedule_html": "<p>Schedule</p>",
            }
        ]

        result = convert_to_lightweight_schema(papers)

        # Ensure extra fields are not present
        assert "topic" not in result[0]
        assert "eventtype" not in result[0]
        assert "event_type" not in result[0]
        assert "virtualsite_url" not in result[0]
        assert "sourceid" not in result[0]
        assert "sourceurl" not in result[0]
        assert "diversity_event" not in result[0]
        assert "show_in_schedule_overview" not in result[0]
        assert "visible" not in result[0]
        assert "schedule_html" not in result[0]

    def test_empty_list(self):
        """Test conversion of empty paper list."""
        result = convert_to_lightweight_schema([])
        assert result == []


class TestIntegration:
    """Integration tests combining helper functions."""

    def test_sanitize_and_validate(self):
        """Test sanitizing authors and then validating the paper."""
        authors_with_semicolons = ["John Doe", "Jane; Smith", "Bob;Johnson"]
        sanitized = sanitize_author_names(authors_with_semicolons)

        paper = {
            "title": "Test Paper",
            "authors": sanitized,
            "abstract": "Test abstract",
            "session": "Session A",
            "poster_position": "A1",
            "year": 2025,
            "conference": "NeurIPS",
        }

        # Should validate successfully
        validated = validate_lightweight_paper(paper)
        assert validated.authors == ["John Doe", "Jane Smith", "Bob Johnson"]

    def test_convert_and_validate(self):
        """Test converting from NeurIPS schema and then validating."""
        papers = [
            {
                "id": 123,
                "title": "Deep Learning Paper",
                "authors": [
                    {"id": 1, "fullname": "Alice Johnson", "institution": "MIT"},
                    {"id": 2, "fullname": "Bob Smith", "institution": "Stanford"},
                ],
                "abstract": "A paper about deep learning",
                "session": "Session A",
                "poster_position": "A-42",
                "year": 2025,
                "conference": "NeurIPS",
            }
        ]

        lightweight = convert_to_lightweight_schema(papers)

        # Should validate successfully
        validated = validate_lightweight_paper(lightweight[0])
        assert validated.title == "Deep Learning Paper"
        assert validated.authors == ["Alice Johnson", "Bob Smith"]
        assert validated.year == 2025
        assert validated.conference == "NeurIPS"

    def test_convert_sanitize_and_validate(self):
        """Test full pipeline: convert, sanitize, validate."""
        papers = [
            {
                "id": 1,
                "title": "Test Paper",
                "authors": [{"fullname": "John; Doe"}, {"fullname": "Jane Smith"}],  # Has semicolon
                "abstract": "Abstract",
                "session": "Session A",
                "poster_position": "A1",
                "year": 2025,
                "conference": "NeurIPS",
            }
        ]

        # Convert
        lightweight = convert_to_lightweight_schema(papers)

        # Sanitize authors
        lightweight[0]["authors"] = sanitize_author_names(lightweight[0]["authors"])

        # Validate
        validated = validate_lightweight_paper(lightweight[0])
        assert validated.authors == ["John Doe", "Jane Smith"]


# ============================================================
# Tests from test_plugin_year_conference.py
# ============================================================

class TestConferencePluginYearFields:
    """Test that NeurIPS plugin sets year and conference fields."""

    @patch("abstracts_explorer.plugins.json_conference_downloader.requests.get")
    def test_neurips_plugin_adds_year_and_conference(self, mock_get):
        """Test that NeurIPS plugin adds year and conference to each paper."""
        # Mock the requests.get to return test data
        mock_response = Mock()
        mock_response.json.return_value = {
            "count": 2,
            "next": None,
            "previous": None,
            "results": [
                {
                    "id": 1,
                    "name": "Paper 1",
                    "abstract": "Abstract 1",
                    "authors": [{"fullname": "Author 1"}],
                    "session": "Session A",
                    "poster_position": "A1",
                },
                {
                    "id": 2,
                    "name": "Paper 2",
                    "abstract": "Abstract 2",
                    "authors": [{"fullname": "Author 2"}],
                    "session": "Session B",
                    "poster_position": "B1",
                },
            ],
        }
        mock_get.return_value = mock_response

        plugin = NeurIPSDownloaderPlugin()
        papers = plugin.download(year=2024)

        # Verify we got a list of LightweightPaper objects
        assert isinstance(papers, list)
        assert len(papers) == 2

        # Verify year and conference were set
        for paper in papers:
            assert paper.year == 2024
            assert paper.conference == "NeurIPS"

    @patch("abstracts_explorer.plugins.json_conference_downloader.requests.get")
    def test_neurips_plugin_preserves_existing_fields(self, mock_get):
        """Test that NeurIPS plugin preserves existing paper fields."""
        # Mock the requests.get to return test data with existing fields
        mock_response = Mock()
        mock_response.json.return_value = {
            "count": 1,
            "next": None,
            "previous": None,
            "results": [
                {
                    "id": 1,
                    "name": "Test Paper",
                    "abstract": "Test Abstract",
                    "authors": [{"fullname": "John Doe"}],
                    "keywords": ["ML", "AI"],
                    "session": "Session A",
                    "poster_position": "A1",
                },
            ],
        }
        mock_get.return_value = mock_response

        plugin = NeurIPSDownloaderPlugin()
        papers = plugin.download(year=2025)

        # Verify we got a list of LightweightPaper objects
        assert isinstance(papers, list)
        assert len(papers) == 1

        # Verify existing fields are preserved in LightweightPaper
        paper = papers[0]
        assert paper.title == "Test Paper"
        assert paper.abstract == "Test Abstract"
        assert "John Doe" in paper.authors
        assert paper.keywords == ["ML", "AI"]

        # Verify new fields were added
        assert paper.year == 2025
        assert paper.conference == "NeurIPS"


class TestICMLPluginYearConference:
    """Test that ICML plugin sets year and conference fields."""

    @patch("abstracts_explorer.plugins.json_conference_downloader.requests.get")
    def test_icml_plugin_adds_year_and_conference(self, mock_get):
        """Test that ICML plugin adds year and conference to each paper."""
        # Mock the requests.get to return test data
        mock_response = Mock()
        mock_response.json.return_value = {
            "count": 2,
            "next": None,
            "previous": None,
            "results": [
                {
                    "id": 1,
                    "name": "Paper 1",
                    "abstract": "Abstract 1",
                    "authors": [{"fullname": "Author 1"}],
                    "session": "Session A",
                    "poster_position": "A1",
                },
                {
                    "id": 2,
                    "name": "Paper 2",
                    "abstract": "Abstract 2",
                    "authors": [{"fullname": "Author 2"}],
                    "session": "Session B",
                    "poster_position": "B1",
                },
            ],
        }
        mock_get.return_value = mock_response

        plugin = ICMLDownloaderPlugin()
        papers = plugin.download(year=2025)

        # Verify we got a list of LightweightPaper objects
        assert isinstance(papers, list)
        assert len(papers) == 2

        # Verify year and conference were set
        for paper in papers:
            assert paper.year == 2025
            assert paper.conference == "ICML"

    @patch("abstracts_explorer.plugins.json_conference_downloader.requests.get")
    def test_icml_plugin_preserves_existing_fields(self, mock_get):
        """Test that ICML plugin preserves existing paper fields."""
        # Mock the requests.get to return test data with existing fields
        mock_response = Mock()
        mock_response.json.return_value = {
            "count": 1,
            "next": None,
            "previous": None,
            "results": [
                {
                    "id": 1,
                    "name": "Test Paper",
                    "abstract": "Test Abstract",
                    "authors": [{"fullname": "John Doe"}],
                    "keywords": ["ML", "AI"],
                    "session": "Session A",
                    "poster_position": "A1",
                },
            ],
        }
        mock_get.return_value = mock_response

        plugin = ICMLDownloaderPlugin()
        papers = plugin.download(year=2025)

        # Verify we got a list of LightweightPaper objects
        assert isinstance(papers, list)
        assert len(papers) == 1

        # Verify existing fields are preserved in LightweightPaper
        paper = papers[0]
        assert paper.title == "Test Paper"
        assert paper.abstract == "Test Abstract"
        assert "John Doe" in paper.authors
        assert paper.keywords == ["ML", "AI"]

        # Verify new fields were added
        assert paper.year == 2025
        assert paper.conference == "ICML"


class TestML4PSPluginYearConference:
    """Test that ML4PS plugin sets year and conference fields."""

    def test_ml4ps_lightweight_format_includes_year_and_conference(self):
        """Test that ML4PS plugin includes year and conference in lightweight format."""
        plugin = ML4PSDownloaderPlugin()

        # Create sample papers data
        papers = [
            {
                "id": 1,
                "title": "Test Paper",
                "authors_str": "John Doe, Jane Smith",
                "abstract": "Test abstract",
                "eventtype": "Poster",
                "awards": [],
            }
        ]

        # Convert to lightweight format
        lightweight_papers = plugin._convert_to_lightweight_format(papers)

        # Verify year and conference are set
        assert len(lightweight_papers) == 1
        paper = lightweight_papers[0]
        assert paper["year"] == 2025
        assert paper["conference"] == "ML4PS@Neurips"

    def test_ml4ps_lightweight_format_preserves_fields(self):
        """Test that ML4PS plugin preserves all required fields."""
        plugin = ML4PSDownloaderPlugin()

        papers = [
            {
                "id": 42,
                "title": "Amazing Paper",
                "authors_str": "Alice, Bob, Charlie",
                "abstract": "This is an amazing abstract",
                "eventtype": "Spotlight",
                "awards": ["Best Paper", "Outstanding Poster"],
                "paper_url": "https://example.com/paper.pdf",
                "openreview_url": "https://openreview.net/paper",
            }
        ]

        lightweight_papers = plugin._convert_to_lightweight_format(papers)

        paper = lightweight_papers[0]
        assert paper["title"] == "Amazing Paper"
        assert paper["authors"] == ["Alice", "Bob", "Charlie"]
        assert paper["abstract"] == "This is an amazing abstract"
        assert paper["session"] == "ML4PhysicalSciences 2025 Workshop - Spotlight"
        assert paper["id"] == 42
        assert paper["year"] == 2025
        assert paper["conference"] == "ML4PS@Neurips"
        assert paper["award"] == "Best Paper, Outstanding Poster"


class TestDatabaseYearConferenceIntegration:
    """Test that year and conference fields are properly stored in the database."""

    @patch("abstracts_explorer.plugins.json_conference_downloader.requests.get")
    def test_neurips_year_conference_in_database(self, mock_get, monkeypatch):
        """Test that year and conference are stored in database from NeurIPS plugin."""
        # Mock the requests.get to return test data
        mock_response = Mock()
        mock_response.json.return_value = {
            "count": 2,
            "next": None,
            "previous": None,
            "results": [
                {
                    "id": 1,
                    "name": "Test Paper 1",
                    "abstract": "Abstract 1",
                    "authors": [{"fullname": "Author 1"}],
                    "session": "Session A",
                    "eventtype": "Poster",
                    "poster_position": "A1",
                },
                {
                    "id": 2,
                    "name": "Test Paper 2",
                    "abstract": "Abstract 2",
                    "authors": [{"fullname": "Author 2"}],
                    "session": "Session B",
                    "eventtype": "Oral",
                    "poster_position": "B1",
                },
            ],
        }
        mock_get.return_value = mock_response

        plugin = NeurIPSDownloaderPlugin()
        data = plugin.download(year=2024)

        # Create temporary database and load data (data is now List[LightweightPaper])
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            monkeypatch.setenv("PAPER_DB", str(db_path))
            get_config(reload=True)
            with DatabaseManager() as db:
                db.create_tables()
                db.add_papers(data)

                # Query papers and verify year and conference
                papers = db.query("SELECT uid, title, year, conference FROM papers ORDER BY title")
                assert len(papers) == 2

                # Check first paper (Test Paper 1)
                assert papers[0]["title"] == "Test Paper 1"
                assert papers[0]["year"] == 2024
                assert papers[0]["conference"] == "NeurIPS"

                # Check second paper (Test Paper 2)
                assert papers[1]["title"] == "Test Paper 2"
                assert papers[1]["year"] == 2024
                assert papers[1]["conference"] == "NeurIPS"

                # Test filtering by year
                papers_2024 = db.query("SELECT * FROM papers WHERE year = 2024")
                assert len(papers_2024) == 2

                # Test filtering by conference
                papers = db.query("SELECT * FROM papers WHERE conference = 'NeurIPS'")
                assert len(papers) == 2

    def test_ml4ps_year_conference_in_database(self, monkeypatch):
        """Test that year and conference are stored in database from ML4PS plugin."""
        plugin = ML4PSDownloaderPlugin()

        # Create sample lightweight papers
        papers = [
            {
                "id": 1,
                "title": "ML4PS Paper 1",
                "authors_str": "Alice, Bob",
                "abstract": "Abstract for paper 1",
                "eventtype": "Poster",
                "awards": [],
            },
            {
                "id": 2,
                "title": "ML4PS Paper 2",
                "authors_str": "Charlie",
                "abstract": "Abstract for paper 2",
                "eventtype": "Spotlight",
                "awards": ["Best Paper"],
            },
        ]

        # Convert to lightweight format (which adds year and conference)
        lightweight_papers = plugin._convert_to_lightweight_format(papers)

        # Convert to LightweightPaper objects for insertion
        from abstracts_explorer.plugin import LightweightPaper

        papers_to_insert = [LightweightPaper(**paper_dict) for paper_dict in lightweight_papers]

        # Create temporary database and load data
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            monkeypatch.setenv("PAPER_DB", str(db_path))
            get_config(reload=True)
            with DatabaseManager() as db:
                db.create_tables()
                db.add_papers(papers_to_insert)

                # Query papers and verify year and conference
                papers = db.query("SELECT uid, title, year, conference FROM papers ORDER BY uid")
                assert len(papers) == 2

                # Check first paper
                assert papers[0]["title"] == "ML4PS Paper 1"
                assert papers[0]["year"] == 2025
                assert papers[0]["conference"] == "ML4PS@Neurips"

                # Check second paper
                assert papers[1]["title"] == "ML4PS Paper 2"
                assert papers[1]["year"] == 2025
                assert papers[1]["conference"] == "ML4PS@Neurips"

                # Test filtering by year
                papers_2025 = db.query("SELECT * FROM papers WHERE year = 2025")
                assert len(papers_2025) == 2

                # Test filtering by conference
                ml4ps_papers = db.query("SELECT * FROM papers WHERE conference = 'ML4PS@Neurips'")
                assert len(ml4ps_papers) == 2


# ============================================================
# Tests from test_plugins_models.py
# ============================================================

class TestLightweightPaper:
    """Tests for LightweightPaper model."""

    def test_valid_minimal_paper(self):
        """Test creating paper with minimal required fields."""
        paper = LightweightPaper(
            title="Test Paper",
            authors=["John Doe", "Jane Smith"],
            abstract="This is a test abstract.",
            session="Workshop 2025",
            poster_position="A1",
            year=2025,
            conference="NeurIPS",
        )
        assert paper.title == "Test Paper"
        assert len(paper.authors) == 2
        assert paper.abstract == "This is a test abstract."
        assert paper.session == "Workshop 2025"
        assert paper.poster_position == "A1"
        assert paper.year == 2025
        assert paper.conference == "NeurIPS"

    def test_valid_paper_with_optional_fields(self):
        """Test creating paper with optional fields."""
        paper = LightweightPaper(
            title="Test Paper",
            authors=["John Doe"],
            abstract="Abstract text",
            session="Workshop 2025",
            poster_position="A1",
            year=2025,
            conference="NeurIPS",
            original_id=123,
            paper_pdf_url="https://example.com/paper.pdf",
            poster_image_url="https://example.com/poster.png",
            url="https://openreview.net/forum?id=abc",
            keywords=["machine learning", "AI"],
            award="Best Paper Award",
        )
        assert paper.original_id == 123
        assert paper.paper_pdf_url == "https://example.com/paper.pdf"
        assert paper.award == "Best Paper Award"
        assert len(paper.keywords) == 2
        assert paper.year == 2025
        assert paper.conference == "NeurIPS"

    def test_empty_title_raises_error(self):
        """Test that empty title raises validation error."""
        with pytest.raises(ValidationError, match="Paper title cannot be empty"):
            LightweightPaper(
                title="",
                authors=["John Doe"],
                abstract="Abstract",
                session="Session",
                poster_position="A1",
                year=2025,
                conference="NeurIPS",
            )

    def test_empty_authors_raises_error(self):
        """Test that empty authors list raises validation error."""
        with pytest.raises(ValidationError, match="Authors list cannot be empty"):
            LightweightPaper(
                title="Test Paper",
                authors=[],  # Empty authors list should raise error
                abstract="Abstract",
                session="Session",
                poster_position="A1",
                year=2025,
                conference="NeurIPS",
            )

    def test_empty_session_raises_error(self):
        """Test that empty session raises validation error."""
        with pytest.raises(ValidationError, match="Session cannot be empty"):
            LightweightPaper(
                title="Test Paper",
                authors=["John Doe"],
                abstract="Abstract",
                session="",  # Empty session should raise error
                poster_position="A1",
                year=2025,
                conference="NeurIPS",
            )

    def test_year_and_conference_required(self):
        """Test that year and conference fields are required."""
        with pytest.raises(ValidationError, match="Field required"):
            LightweightPaper(
                title="Test Paper",
                authors=["John Doe"],
                abstract="Abstract",
                session="Session",
                poster_position="A1",
            )

    def test_year_and_conference_with_values(self):
        """Test that year and conference fields can be set."""
        paper = LightweightPaper(
            title="Test Paper",
            authors=["John Doe"],
            abstract="Abstract",
            session="Session",
            poster_position="A1",
            year=2025,
            conference="NeurIPS",
        )
        assert paper.year == 2025
        assert paper.conference == "NeurIPS"

    def test_empty_conference_raises_error(self):
        """Test that empty conference raises validation error."""
        with pytest.raises(ValidationError, match="Conference cannot be empty"):
            LightweightPaper(
                title="Test Paper",
                authors=["John Doe"],
                abstract="Abstract",
                session="Session",
                poster_position="A1",
                year=2025,
                conference="",
            )

    def test_invalid_year_raises_error(self):
        """Test that invalid year raises validation error."""
        with pytest.raises(ValidationError, match="Year .* is not reasonable"):
            LightweightPaper(
                title="Test Paper",
                authors=["John Doe"],
                abstract="Abstract",
                session="Session",
                poster_position="A1",
                year=1800,  # Too old
                conference="NeurIPS",
            )

        with pytest.raises(ValidationError, match="Year .* is not reasonable"):
            LightweightPaper(
                title="Test Paper",
                authors=["John Doe"],
                abstract="Abstract",
                session="Session",
                poster_position="A1",
                year=2200,  # Too far in future
                conference="NeurIPS",
            )


# ============================================================================
# Validation Helper Function Tests
# ============================================================================


class TestValidationHelpers:
    """Tests for validation helper functions."""

    def test_validate_lightweight_paper(self):
        """Test validate_lightweight_paper function."""
        paper_dict = {
            "title": "Test Paper",
            "authors": ["John Doe"],
            "abstract": "Abstract",
            "session": "Session",
            "poster_position": "A1",
            "year": 2025,
            "conference": "NeurIPS",
        }
        paper = validate_lightweight_paper(paper_dict)
        assert isinstance(paper, LightweightPaper)
        assert paper.title == "Test Paper"
        assert paper.year == 2025
        assert paper.conference == "NeurIPS"

    def test_validate_lightweight_paper_invalid(self):
        """Test validate_lightweight_paper with invalid data."""
        paper_dict = {
            "title": "",  # Empty title
            "authors": ["John Doe"],
            "abstract": "Abstract",
            "session": "Session",
            "poster_position": "A1",
            "year": 2025,
            "conference": "NeurIPS",
        }
        with pytest.raises(ValidationError):
            validate_lightweight_paper(paper_dict)

    def test_validate_lightweight_papers(self):
        """Test validate_lightweight_papers function."""
        papers = [
            {
                "title": "Paper 1",
                "authors": ["Alice"],
                "abstract": "Abstract 1",
                "session": "Session",
                "poster_position": "A1",
                "year": 2025,
                "conference": "NeurIPS",
            },
            {
                "title": "Paper 2",
                "authors": ["Bob"],
                "abstract": "Abstract 2",
                "session": "Session",
                "poster_position": "A2",
                "year": 2025,
                "conference": "NeurIPS",
            },
        ]
        validated = validate_lightweight_papers(papers)
        assert len(validated) == 2
        assert all(isinstance(p, LightweightPaper) for p in validated)
        assert all(p.year == 2025 for p in validated)
        assert all(p.conference == "NeurIPS" for p in validated)


# ============================================================================
# Integration Tests
# ============================================================================


class TestModelIntegration:
    """Integration tests for models."""

    def test_lightweight_to_dict(self):
        """Test converting lightweight paper to dict."""
        paper = LightweightPaper(
            title="Test Paper",
            authors=["John Doe"],
            abstract="Abstract",
            session="Session",
            poster_position="A1",
            year=2025,
            conference="NeurIPS",
            award="Best Paper",
        )
        paper_dict = paper.model_dump()
        assert paper_dict["title"] == "Test Paper"
        assert paper_dict["award"] == "Best Paper"
        assert paper_dict["year"] == 2025
        assert paper_dict["conference"] == "NeurIPS"


# ============================================================
# Tests from test_pydantic_validation.py
# ============================================================

class TestPydanticValidation:
    """Tests for Pydantic data validation."""

    def test_invalid_paper_id_type(self, connected_db):
        """Test that invalid paper_id type is handled."""
        # Note: paper_id is not part of LightweightPaper (it's original_id)
        # This test now validates that original_id accepts integers
        papers = [
            LightweightPaper(
                title="Test Paper",
                authors=["John Doe"],
                abstract="Test abstract",
                session="Test Session",
                poster_position="A1",
                year=2025,
                conference="NeurIPS",
                original_id=123456,  # Valid integer
            )
        ]
        count = connected_db.add_papers(papers)
        assert count == 1  # Should succeed with valid data

    def test_missing_required_fields(self, connected_db):
        """Test that missing required fields are rejected."""
        # Missing required 'title' field - will raise ValidationError
        with pytest.raises(Exception):  # Pydantic will raise validation error
            [
                LightweightPaper(
                    # Missing required 'title' field
                    authors=["John Doe"],
                    abstract="Test abstract",
                    session="Test Session",
                    poster_position="A1",
                    year=2025,
                    conference="NeurIPS",
                )
            ]

    def test_empty_paper_title(self, connected_db):
        """Test that empty paper title is rejected."""
        # Invalid: title cannot be empty - will raise ValidationError
        with pytest.raises(Exception):  # Pydantic will raise validation error
            [
                LightweightPaper(
                    title="",  # Invalid: title cannot be empty
                    authors=["John Doe"],
                    abstract="Test abstract",
                    session="Test Session",
                    poster_position="A1",
                    year=2025,
                    conference="NeurIPS",
                )
            ]

    def test_invalid_author_data(self, connected_db):
        """Test that invalid author data is handled gracefully."""
        # First author empty - will raise ValidationError
        with pytest.raises(Exception):  # Pydantic will raise validation error
            [
                LightweightPaper(
                    title="Valid Paper",
                    authors=["", "Jane Smith"],  # First author empty - invalid
                    abstract="Test abstract",
                    session="Test Session",
                    poster_position="A1",
                    year=2025,
                    conference="NeurIPS",
                )
            ]

    def test_valid_data_passes_validation(self, connected_db):
        """Test that valid data passes validation."""
        papers = [
            LightweightPaper(
                title="Valid Paper Title",
                authors=["John Doe", "Jane Smith"],
                abstract="This is a valid abstract",
                session="Test Session",
                poster_position="A1",
                keywords=["deep learning", "neural networks"],
                year=2025,
                conference="NeurIPS",
            )
        ]

        count = connected_db.add_papers(papers)
        assert count == 1

        # Verify data was inserted correctly
        papers_result = connected_db.search_papers(keyword="Valid")
        assert len(papers_result) == 1
        assert papers_result[0]["title"] == "Valid Paper Title"

        # Verify authors were stored as semicolon-separated string
        assert papers_result[0]["authors"] == "John Doe; Jane Smith"

    def test_extra_fields_allowed(self, connected_db):
        """Test that extra fields not in model are allowed."""
        papers = [
            LightweightPaper(
                title="Paper with Extra Fields",
                authors=["John Doe"],
                abstract="Test abstract",
                session="Test Session",
                poster_position="A1",
                year=2025,
                conference="NeurIPS",
                extra_field_1="This field is not in the model",
                extra_field_2=12345,
                nested_extra={"key": "value"},
            )
        ]

        # Should succeed because extra fields are allowed
        count = connected_db.add_papers(papers)
        assert count == 1

    def test_type_coercion(self, connected_db):
        """Test that Pydantic coerces compatible types."""
        papers = [
            LightweightPaper(
                title="Test Paper",
                authors=["John Doe"],
                abstract="Test abstract",
                session="Test Session",
                poster_position="A1",
                year="2025",  # String that can be converted to int
                conference="NeurIPS",
            )
        ]

        count = connected_db.add_papers(papers)
        assert count == 1

        papers_result = connected_db.search_papers(keyword="Test")
        assert papers_result[0]["uid"] is not None  # Should have valid UID

    def test_authors_with_semicolons_rejected(self, connected_db):
        """Test that author names with semicolons are rejected."""
        # Semicolon not allowed - will raise ValidationError
        with pytest.raises(Exception):  # Pydantic will raise validation error
            [
                LightweightPaper(
                    title="Test Paper",
                    authors=["John; Doe"],  # Semicolon not allowed
                    abstract="Test abstract",
                    session="Test Session",
                    poster_position="A1",
                    year=2025,
                    conference="NeurIPS",
                )
            ]

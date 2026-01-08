"""
Tests for plugin helper functions.
"""

from neurips_abstracts.plugin import (
    sanitize_author_names,
    convert_neurips_to_lightweight_schema,
    validate_lightweight_paper,
)


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
    """Tests for convert_neurips_to_lightweight_schema helper function."""

    def test_basic_conversion(self):
        """Test basic conversion with all required fields."""
        neurips_papers = [
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

        result = convert_neurips_to_lightweight_schema(neurips_papers)

        assert len(result) == 1
        assert result[0]["title"] == "Test Paper"
        assert result[0]["authors"] == ["John Doe", "Jane Smith"]
        assert result[0]["abstract"] == "Test abstract"
        assert result[0]["session"] == "Session A"
        assert result[0]["poster_position"] == "A-42"
        assert result[0]["original_id"] == 123

    def test_legacy_name_field(self):
        """Test conversion of legacy 'name' field to 'title'."""
        neurips_papers = [
            {
                "id": 1,
                "name": "Legacy Paper",  # Using 'name' instead of 'title'
                "authors": [{"fullname": "John Doe"}],
                "abstract": "Abstract",
                "session": "Session A",
                "poster_position": "A1",
            }
        ]

        result = convert_neurips_to_lightweight_schema(neurips_papers)

        assert result[0]["title"] == "Legacy Paper"
        assert "name" not in result[0]

    def test_authors_as_strings(self):
        """Test conversion when authors are already strings."""
        neurips_papers = [
            {
                "id": 1,
                "title": "Test Paper",
                "authors": ["John Doe", "Jane Smith"],  # Already strings
                "abstract": "Abstract",
                "session": "Session A",
                "poster_position": "A1",
            }
        ]

        result = convert_neurips_to_lightweight_schema(neurips_papers)

        assert result[0]["authors"] == ["John Doe", "Jane Smith"]

    def test_authors_as_semicolon_separated_string(self):
        """Test conversion when authors is a semicolon-separated string."""
        neurips_papers = [
            {
                "id": 1,
                "title": "Test Paper",
                "authors": "John Doe; Jane Smith; Bob Johnson",
                "abstract": "Abstract",
                "session": "Session A",
                "poster_position": "A1",
            }
        ]

        result = convert_neurips_to_lightweight_schema(neurips_papers)

        assert result[0]["authors"] == ["John Doe", "Jane Smith", "Bob Johnson"]

    def test_optional_fields(self):
        """Test that optional fields are included when present."""
        neurips_papers = [
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

        result = convert_neurips_to_lightweight_schema(neurips_papers)

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
        neurips_papers = [
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

        result = convert_neurips_to_lightweight_schema(neurips_papers)

        assert result[0]["keywords"] == ["machine learning", "deep learning", "AI"]

    def test_award_extraction(self):
        """Test extraction of award from decision field."""
        neurips_papers = [
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

        result = convert_neurips_to_lightweight_schema(neurips_papers)

        assert result[0]["award"] == "Best Paper Award"

    def test_award_field_direct(self):
        """Test that direct award field is preserved."""
        neurips_papers = [
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

        result = convert_neurips_to_lightweight_schema(neurips_papers)

        assert result[0]["award"] == "Outstanding Paper"

    def test_decision_without_award(self):
        """Test that regular decision is not converted to award."""
        neurips_papers = [
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

        result = convert_neurips_to_lightweight_schema(neurips_papers)

        assert "award" not in result[0]

    def test_decision_none_no_error(self):
        """Test that None decision value doesn't cause an error."""
        neurips_papers = [
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

        result = convert_neurips_to_lightweight_schema(neurips_papers)

        assert "award" not in result[0]
        # Should not raise AttributeError

    def test_none_field_values_converted_to_empty_strings(self):
        """Test that None field values are converted to empty strings."""
        neurips_papers = [
            {
                "id": 1,
                "title": "Test Paper",
                "authors": ["John Doe"],
                "abstract": None,  # None instead of string
                "session": None,  # None instead of string
                "poster_position": None,  # None instead of string
            }
        ]

        result = convert_neurips_to_lightweight_schema(neurips_papers)

        # All None values should be converted to appropriate defaults
        assert result[0]["abstract"] == ""
        assert result[0]["session"] == "No session"  # Default for None session
        assert result[0]["poster_position"] == ""
        # Should not raise validation errors when used with LightweightPaper

    def test_author_names_with_semicolons_sanitized(self):
        """Test that author names with semicolons are sanitized."""
        neurips_papers = [
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

        result = convert_neurips_to_lightweight_schema(neurips_papers)

        # Semicolons should be replaced with spaces
        assert result[0]["authors"] == ["John Doe", "Jane Smith"]
        # Should not raise validation errors when used with LightweightPaper

    def test_multiple_papers(self):
        """Test conversion of multiple papers at once."""
        neurips_papers = [
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

        result = convert_neurips_to_lightweight_schema(neurips_papers)

        assert len(result) == 3
        assert result[0]["title"] == "Paper 1"
        assert result[1]["title"] == "Paper 2"
        assert result[2]["title"] == "Paper 3"

    def test_paper_without_title(self):
        """Test that papers without title are skipped."""
        neurips_papers = [
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

        result = convert_neurips_to_lightweight_schema(neurips_papers)

        # First paper should be skipped
        assert len(result) == 1
        assert result[0]["title"] == "Valid Paper"

    def test_empty_title(self):
        """Test that papers with empty title are skipped."""
        neurips_papers = [
            {
                "id": 1,
                "title": "",  # Empty title
                "authors": ["John Doe"],
                "abstract": "Abstract",
                "session": "Session A",
                "poster_position": "A1",
            },
        ]

        result = convert_neurips_to_lightweight_schema(neurips_papers)

        assert len(result) == 0

    def test_author_with_name_field(self):
        """Test author dict with 'name' instead of 'fullname'."""
        neurips_papers = [
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

        result = convert_neurips_to_lightweight_schema(neurips_papers)

        assert result[0]["authors"] == ["John Doe"]

    def test_author_without_name(self):
        """Test that authors without name/fullname are skipped."""
        neurips_papers = [
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

        result = convert_neurips_to_lightweight_schema(neurips_papers)

        assert result[0]["authors"] == ["John Doe", "Jane Smith"]

    def test_extra_neurips_fields_removed(self):
        """Test that NeurIPS-specific fields are not included in lightweight schema."""
        neurips_papers = [
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

        result = convert_neurips_to_lightweight_schema(neurips_papers)

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
        result = convert_neurips_to_lightweight_schema([])
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
        neurips_papers = [
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

        lightweight = convert_neurips_to_lightweight_schema(neurips_papers)

        # Should validate successfully
        validated = validate_lightweight_paper(lightweight[0])
        assert validated.title == "Deep Learning Paper"
        assert validated.authors == ["Alice Johnson", "Bob Smith"]
        assert validated.year == 2025
        assert validated.conference == "NeurIPS"

    def test_convert_sanitize_and_validate(self):
        """Test full pipeline: convert, sanitize, validate."""
        neurips_papers = [
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
        lightweight = convert_neurips_to_lightweight_schema(neurips_papers)

        # Sanitize authors
        lightweight[0]["authors"] = sanitize_author_names(lightweight[0]["authors"])

        # Validate
        validated = validate_lightweight_paper(lightweight[0])
        assert validated.authors == ["John Doe", "Jane Smith"]

"""
Tests for Pydantic validation in database module.
"""

import pytest
from neurips_abstracts.database import DatabaseManager


class TestPydanticValidation:
    """Tests for Pydantic data validation."""

    def test_invalid_paper_id_type(self, connected_db):
        """Test that invalid paper ID type is rejected."""
        data = [
            {
                "id": "not-a-number",  # Invalid: should be integer
                "title": "Test Paper",
                "authors": ["John Doe"],
                "abstract": "Test abstract",
                "session": "Test Session",
                "poster_position": "A1",
            }
        ]

        # Should not raise, but should log warning and skip record
        count = connected_db.load_json_data(data)
        assert count == 0  # Record should be skipped due to validation error

    def test_missing_required_fields(self, connected_db):
        """Test that missing required fields are rejected."""
        data = [
            {
                "id": 123456,
                # Missing required 'title' field
                "authors": ["John Doe"],
                "abstract": "Test abstract",
                "session": "Test Session",
                "poster_position": "A1",
            }
        ]

        count = connected_db.load_json_data(data)
        assert count == 0  # Record should be skipped due to validation error

    def test_empty_paper_title(self, connected_db):
        """Test that empty paper title is rejected."""
        data = [
            {
                "id": 123456,
                "title": "",  # Invalid: title cannot be empty
                "authors": ["John Doe"],
                "abstract": "Test abstract",
                "session": "Test Session",
                "poster_position": "A1",
            }
        ]

        count = connected_db.load_json_data(data)
        assert count == 0  # Record should be skipped due to validation error

    def test_invalid_author_data(self, connected_db):
        """Test that invalid author data is handled gracefully."""
        data = [
            {
                "id": 123456,
                "title": "Valid Paper",
                "authors": ["", "Jane Smith"],  # First author empty - invalid
                "abstract": "Test abstract",
                "session": "Test Session",
                "poster_position": "A1",
            }
        ]

        # Paper should be skipped due to invalid author
        count = connected_db.load_json_data(data)
        assert count == 0  # Record skipped due to validation error

    def test_valid_data_passes_validation(self, connected_db):
        """Test that valid data passes validation."""
        data = [
            {
                "id": 123456,
                "title": "Valid Paper Title",
                "authors": ["John Doe", "Jane Smith"],
                "abstract": "This is a valid abstract",
                "session": "Test Session",
                "poster_position": "A1",
                "keywords": ["deep learning", "neural networks"],
                "year": 2025,
                "conference": "NeurIPS",
            }
        ]

        count = connected_db.load_json_data(data)
        assert count == 1

        # Verify data was inserted correctly
        papers = connected_db.search_papers(keyword="Valid")
        assert len(papers) == 1
        assert papers[0]["title"] == "Valid Paper Title"

        # Verify authors were stored as comma-separated string
        assert papers[0]["authors"] == "John Doe, Jane Smith"

    def test_extra_fields_allowed(self, connected_db):
        """Test that extra fields not in model are allowed."""
        data = [
            {
                "id": 123456,
                "title": "Paper with Extra Fields",
                "authors": ["John Doe"],
                "abstract": "Test abstract",
                "session": "Test Session",
                "poster_position": "A1",
                "extra_field_1": "This field is not in the model",
                "extra_field_2": 12345,
                "nested_extra": {"key": "value"},
            }
        ]

        # Should succeed because extra fields are allowed
        count = connected_db.load_json_data(data)
        assert count == 1

    def test_type_coercion(self, connected_db):
        """Test that Pydantic coerces compatible types."""
        data = [
            {
                "id": "123456",  # String that can be converted to int
                "title": "Test Paper",
                "authors": ["John Doe"],
                "abstract": "Test abstract",
                "session": "Test Session",
                "poster_position": "A1",
                "year": "2025",  # String that can be converted to int
            }
        ]

        count = connected_db.load_json_data(data)
        assert count == 1

        papers = connected_db.search_papers(keyword="Test")
        assert papers[0]["id"] is not None  # Should have valid ID

    def test_authors_with_semicolons_rejected(self, connected_db):
        """Test that author names with semicolons are rejected."""
        data = [
            {
                "id": 123456,
                "title": "Test Paper",
                "authors": ["John; Doe"],  # Semicolon not allowed
                "abstract": "Test abstract",
                "session": "Test Session",
                "poster_position": "A1",
            }
        ]

        count = connected_db.load_json_data(data)
        assert count == 0  # Record should be skipped due to validation error

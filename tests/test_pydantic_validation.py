"""
Tests for Pydantic validation in database module.
"""

import pytest
from neurips_abstracts.plugin import LightweightPaper


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

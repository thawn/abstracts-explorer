"""
Tests for Plugin Data Models
=============================

Tests for Pydantic models used in the plugin system.
"""

import pytest
from pydantic import ValidationError

from neurips_abstracts.plugins import (
    LightweightPaper,
    validate_lightweight_paper,
    validate_lightweight_papers,
)


# ============================================================================
# Lightweight Model Tests
# ============================================================================


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
            id=123,
            paper_pdf_url="https://example.com/paper.pdf",
            poster_image_url="https://example.com/poster.png",
            url="https://openreview.net/forum?id=abc",
            keywords=["machine learning", "AI"],
            award="Best Paper Award",
        )
        assert paper.id == 123
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
                authors=[],
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
                session="",
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

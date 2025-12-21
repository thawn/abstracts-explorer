"""
Tests for Plugin Data Models
=============================

Tests for Pydantic models used in the plugin system.
"""

import pytest
from pydantic import ValidationError

from neurips_abstracts.plugins import (
    LightweightAuthor,
    LightweightPaper,
    AuthorModel,
    PaperModel,
    EventMediaModel,
    validate_lightweight_paper,
    validate_lightweight_papers,
    validate_paper,
    validate_papers,
)


# ============================================================================
# Lightweight Model Tests
# ============================================================================


class TestLightweightAuthor:
    """Tests for LightweightAuthor model."""

    def test_valid_author_string(self):
        """Test creating author from simple string data."""
        author = LightweightAuthor(name="John Doe")
        assert author.name == "John Doe"
        assert author.affiliation is None

    def test_valid_author_with_affiliation(self):
        """Test creating author with affiliation."""
        author = LightweightAuthor(name="Jane Smith", affiliation="MIT")
        assert author.name == "Jane Smith"
        assert author.affiliation == "MIT"

    def test_empty_name_raises_error(self):
        """Test that empty name raises validation error."""
        with pytest.raises(ValidationError, match="Author name cannot be empty"):
            LightweightAuthor(name="")

    def test_whitespace_name_raises_error(self):
        """Test that whitespace-only name raises validation error."""
        with pytest.raises(ValidationError, match="Author name cannot be empty"):
            LightweightAuthor(name="   ")


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
        )
        assert paper.title == "Test Paper"
        assert len(paper.authors) == 2
        assert paper.abstract == "This is a test abstract."
        assert paper.session == "Workshop 2025"
        assert paper.poster_position == "A1"

    def test_valid_paper_with_optional_fields(self):
        """Test creating paper with optional fields."""
        paper = LightweightPaper(
            title="Test Paper",
            authors=["John Doe"],
            abstract="Abstract text",
            session="Workshop 2025",
            poster_position="A1",
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

    def test_empty_title_raises_error(self):
        """Test that empty title raises validation error."""
        with pytest.raises(ValidationError, match="Paper title cannot be empty"):
            LightweightPaper(
                title="",
                authors=["John Doe"],
                abstract="Abstract",
                session="Session",
                poster_position="A1",
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
            )

    def test_author_dicts_supported(self):
        """Test that author dicts are supported."""
        paper = LightweightPaper(
            title="Test Paper",
            authors=[
                {"name": "John Doe", "affiliation": "MIT"},
                {"name": "Jane Smith"},
            ],
            abstract="Abstract",
            session="Session",
            poster_position="A1",
        )
        assert len(paper.authors) == 2
        assert paper.authors[0]["name"] == "John Doe"

    def test_mixed_author_formats(self):
        """Test that mixed author formats (strings and dicts) work."""
        paper = LightweightPaper(
            title="Test Paper",
            authors=[
                "John Doe",
                {"name": "Jane Smith", "affiliation": "Stanford"},
            ],
            abstract="Abstract",
            session="Session",
            poster_position="A1",
        )
        assert len(paper.authors) == 2


# ============================================================================
# Full Schema Model Tests
# ============================================================================


class TestAuthorModel:
    """Tests for AuthorModel (full schema)."""

    def test_valid_author(self):
        """Test creating valid author."""
        author = AuthorModel(
            id=1,
            fullname="John Doe",
            url="https://example.com/john",
            institution="MIT",
        )
        assert author.id == 1
        assert author.fullname == "John Doe"
        assert author.institution == "MIT"

    def test_empty_fullname_raises_error(self):
        """Test that empty fullname raises validation error."""
        with pytest.raises(ValidationError, match="Author fullname cannot be empty"):
            AuthorModel(id=1, fullname="")


class TestPaperModel:
    """Tests for PaperModel (full schema)."""

    def test_valid_minimal_paper(self):
        """Test creating paper with minimal required fields."""
        paper = PaperModel(id=1, name="Test Paper")
        assert paper.id == 1
        assert paper.name == "Test Paper"

    def test_valid_paper_with_fields(self):
        """Test creating paper with many fields."""
        paper = PaperModel(
            id=1,
            name="Test Paper",
            authors=[{"id": 1, "fullname": "John Doe"}],
            abstract="Test abstract",
            session="Test Session",
            decision="Accept (poster)",
            keywords=["ML", "AI"],
        )
        assert paper.id == 1
        assert paper.name == "Test Paper"
        assert len(paper.authors) == 1
        assert paper.decision == "Accept (poster)"

    def test_empty_name_raises_error(self):
        """Test that empty name raises validation error."""
        with pytest.raises(ValidationError, match="Paper name cannot be empty"):
            PaperModel(id=1, name="")

    def test_none_id_raises_error(self):
        """Test that None ID raises validation error."""
        with pytest.raises(ValidationError, match="Input should be a valid integer"):
            PaperModel(id=None, name="Test Paper")

    def test_string_id_converted_to_int(self):
        """Test that string ID is converted to int."""
        paper = PaperModel(id="123", name="Test Paper")
        assert paper.id == 123
        assert isinstance(paper.id, int)


class TestEventMediaModel:
    """Tests for EventMediaModel."""

    def test_valid_media(self):
        """Test creating valid event media."""
        media = EventMediaModel(
            id=1,
            type="Poster",
            name="Paper Poster",
            uri="https://example.com/poster.pdf",
        )
        assert media.id == 1
        assert media.type == "Poster"


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
        }
        paper = validate_lightweight_paper(paper_dict)
        assert isinstance(paper, LightweightPaper)
        assert paper.title == "Test Paper"

    def test_validate_lightweight_paper_invalid(self):
        """Test validate_lightweight_paper with invalid data."""
        paper_dict = {
            "title": "",  # Empty title
            "authors": ["John Doe"],
            "abstract": "Abstract",
            "session": "Session",
            "poster_position": "A1",
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
            },
            {
                "title": "Paper 2",
                "authors": ["Bob"],
                "abstract": "Abstract 2",
                "session": "Session",
                "poster_position": "A2",
            },
        ]
        validated = validate_lightweight_papers(papers)
        assert len(validated) == 2
        assert all(isinstance(p, LightweightPaper) for p in validated)

    def test_validate_paper(self):
        """Test validate_paper function."""
        paper_dict = {
            "id": 1,
            "name": "Test Paper",
        }
        paper = validate_paper(paper_dict)
        assert isinstance(paper, PaperModel)
        assert paper.id == 1

    def test_validate_paper_invalid(self):
        """Test validate_paper with invalid data."""
        paper_dict = {
            "id": 1,
            "name": "",  # Empty name
        }
        with pytest.raises(ValidationError):
            validate_paper(paper_dict)

    def test_validate_papers(self):
        """Test validate_papers function."""
        papers = [
            {"id": 1, "name": "Paper 1"},
            {"id": 2, "name": "Paper 2"},
        ]
        validated = validate_papers(papers)
        assert len(validated) == 2
        assert all(isinstance(p, PaperModel) for p in validated)


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
            award="Best Paper",
        )
        paper_dict = paper.model_dump()
        assert paper_dict["title"] == "Test Paper"
        assert paper_dict["award"] == "Best Paper"

    def test_full_paper_to_dict(self):
        """Test converting full paper to dict."""
        paper = PaperModel(
            id=1,
            name="Test Paper",
            decision="Accept (poster)",
        )
        paper_dict = paper.model_dump()
        assert paper_dict["id"] == 1
        assert paper_dict["name"] == "Test Paper"
        assert paper_dict["decision"] == "Accept (poster)"

    def test_extra_fields_allowed(self):
        """Test that extra fields are allowed in models."""
        # Lightweight paper with extra field
        paper = LightweightPaper(
            title="Test Paper",
            authors=["John Doe"],
            abstract="Abstract",
            session="Session",
            poster_position="A1",
            custom_field="custom value",  # Extra field
        )
        paper_dict = paper.model_dump()
        assert "custom_field" in paper_dict

        # Full paper with extra field
        paper2 = PaperModel(
            id=1,
            name="Test Paper",
            custom_field="custom value",  # Extra field
        )
        paper2_dict = paper2.model_dump()
        assert "custom_field" in paper2_dict

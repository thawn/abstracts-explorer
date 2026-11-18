"""
Tests for export_utils module.

Tests the export utilities for generating markdown and zip files from papers,
including natural sorting, web scraping, and file generation.
"""

import zipfile
from io import BytesIO
from unittest.mock import patch

import pytest

from abstracts_explorer.export_utils import (
    natural_sort_key,
    get_poster_url,
    generate_all_papers_markdown,
    generate_search_term_markdown,
    generate_main_readme,
    generate_folder_structure_export,
    export_papers_to_zip,
)


class TestNaturalSortKey:
    """Tests for natural_sort_key function."""

    def test_natural_sort_basic(self):
        """Test basic natural sorting with numbers."""
        items = ["A10", "A2", "A1"]
        sorted_items = sorted(items, key=natural_sort_key)
        assert sorted_items == ["A1", "A2", "A10"]

    def test_natural_sort_mixed(self):
        """Test natural sorting with mixed alphanumeric strings."""
        items = ["file100", "file20", "file3", "file1"]
        sorted_items = sorted(items, key=natural_sort_key)
        assert sorted_items == ["file1", "file3", "file20", "file100"]

    def test_natural_sort_poster_positions(self):
        """Test natural sorting with poster position strings."""
        items = ["B100", "A10", "A2", "B5", "A1"]
        sorted_items = sorted(items, key=natural_sort_key)
        assert sorted_items == ["A1", "A2", "A10", "B5", "B100"]

    def test_natural_sort_no_numbers(self):
        """Test natural sorting with strings without numbers."""
        items = ["zebra", "apple", "banana"]
        sorted_items = sorted(items, key=natural_sort_key)
        assert sorted_items == ["apple", "banana", "zebra"]

    def test_natural_sort_empty_string(self):
        """Test natural sorting with empty string."""
        items = ["", "A1", "A2"]
        sorted_items = sorted(items, key=natural_sort_key)
        assert sorted_items == ["", "A1", "A2"]

    def test_natural_sort_only_numbers(self):
        """Test natural sorting with number-only strings."""
        items = ["100", "20", "3", "1"]
        sorted_items = sorted(items, key=natural_sort_key)
        assert sorted_items == ["1", "3", "20", "100"]


class TestGetPosterUrl:
    """Tests for get_poster_url function."""

    def test_get_poster_url_from_database(self):
        """Test getting poster URL from database field."""
        paper = {
            "poster_image_url": "https://example.com/poster.png",
            "original_id": "123",
            "conference": "NeurIPS",
            "year": 2025,
        }
        url = get_poster_url(paper)
        assert url == "https://example.com/poster.png"

    def test_get_poster_url_from_original_id(self):
        """Test constructing poster URL from original_id."""
        paper = {
            "original_id": "abc123",
            "conference": "NeurIPS",
            "year": 2025,
        }
        url = get_poster_url(paper)
        assert url == "https://neurips.cc/media/PosterPDFs/NeurIPS%202025/abc123.png"

    def test_get_poster_url_default_conference(self):
        """Test poster URL construction with default conference."""
        paper = {
            "original_id": "xyz789",
            "year": 2024,
        }
        url = get_poster_url(paper)
        assert "neurips.cc" in url
        assert "xyz789.png" in url

    def test_get_poster_url_no_data(self):
        """Test getting poster URL with no poster data."""
        paper = {}
        url = get_poster_url(paper)
        assert url is None

    def test_get_poster_url_empty_original_id(self):
        """Test getting poster URL with empty original_id."""
        paper = {
            "original_id": "",
            "poster_image_url": None,
        }
        url = get_poster_url(paper)
        assert url is None


class TestGenerateAllPapersMarkdown:
    """Tests for generate_all_papers_markdown function."""

    def test_generate_all_papers_basic(self):
        """Test basic markdown generation for all papers."""
        papers = [
            {
                "title": "Paper 1",
                "session": "Oral",
                "priority": 5,
                "authors": ["Alice", "Bob"],
                "abstract": "This is paper 1",
            }
        ]
        markdown = generate_all_papers_markdown(papers, "All Papers")

        assert "# All Papers" in markdown
        assert "**Papers:** 1" in markdown
        assert "## Oral" in markdown
        assert "### Paper 1" in markdown
        assert "Alice, Bob" in markdown
        assert "This is paper 1" in markdown
        assert "⭐⭐⭐⭐⭐" in markdown

    def test_generate_all_papers_multiple_sessions(self):
        """Test markdown generation with multiple sessions."""
        papers = [
            {"title": "Paper 1", "session": "Oral", "priority": 5},
            {"title": "Paper 2", "session": "Poster", "priority": 3},
            {"title": "Paper 3", "session": "Oral", "priority": 4},
        ]
        markdown = generate_all_papers_markdown(papers, "All Papers")

        assert "## Oral" in markdown
        assert "## Poster" in markdown
        assert "**Papers in this session:** 2" in markdown  # Oral session
        assert "### Paper 1" in markdown
        assert "### Paper 2" in markdown
        assert "### Paper 3" in markdown

    def test_generate_all_papers_no_session(self):
        """Test markdown generation with papers without session."""
        papers = [
            {"title": "Paper 1", "priority": 3},
        ]
        markdown = generate_all_papers_markdown(papers, "Papers")

        assert "## No Session" in markdown
        assert "### Paper 1" in markdown

    def test_generate_all_papers_with_urls(self):
        """Test markdown generation with paper URLs."""
        papers = [
            {
                "title": "Paper 1",
                "session": "Oral",
                "priority": 4,
                "paper_url": "https://openreview.net/forum?id=abc123",
                "paper_pdf_url": "https://openreview.net/pdf?id=abc123",
                "url": "https://example.com/source",
            }
        ]
        markdown = generate_all_papers_markdown(papers, "Papers")

        assert "**PDF:** [View on OpenReview](https://openreview.net/pdf?id=abc123)" in markdown
        assert "**Paper URL:** https://openreview.net/forum?id=abc123" in markdown
        assert "**Source URL:** https://example.com/source" in markdown

    def test_generate_all_papers_pdf_url_fallback(self):
        """Test PDF URL generation from paper_url."""
        papers = [
            {
                "title": "Paper 1",
                "session": "Poster",
                "priority": 3,
                "paper_url": "https://openreview.net/forum?id=xyz789",
            }
        ]
        markdown = generate_all_papers_markdown(papers, "Papers")

        assert "[View on OpenReview](https://openreview.net/pdf?id=xyz789)" in markdown

    def test_generate_all_papers_with_poster_position(self):
        """Test markdown generation with poster position."""
        papers = [
            {
                "title": "Paper 1",
                "session": "Poster",
                "priority": 3,
                "poster_position": "A-123",
            }
        ]
        markdown = generate_all_papers_markdown(papers, "Papers")

        assert "**Poster:** A-123" in markdown

    def test_generate_all_papers_with_search_term(self):
        """Test markdown generation with search term."""
        papers = [
            {
                "title": "Paper 1",
                "session": "Oral",
                "priority": 4,
                "searchTerm": "transformers",
            }
        ]
        markdown = generate_all_papers_markdown(papers, "Papers")

        assert "**Search Term:** transformers" in markdown

    def test_generate_all_papers_authors_list(self):
        """Test markdown with authors as list."""
        papers = [
            {
                "title": "Paper 1",
                "session": "Poster",
                "priority": 2,
                "authors": ["Alice", "Bob", "Charlie"],
            }
        ]
        markdown = generate_all_papers_markdown(papers, "Papers")

        assert "**Authors:** Alice, Bob, Charlie" in markdown

    def test_generate_all_papers_authors_string(self):
        """Test markdown with authors as string."""
        papers = [
            {
                "title": "Paper 1",
                "session": "Poster",
                "priority": 2,
                "authors": "Alice; Bob; Charlie",
            }
        ]
        markdown = generate_all_papers_markdown(papers, "Papers")

        assert "**Authors:** Alice; Bob; Charlie" in markdown

    @patch("abstracts_explorer.export.markdown.get_poster_url")
    def test_generate_all_papers_with_poster_image(self, mock_get_poster_url):
        """Test markdown generation with poster image."""
        mock_get_poster_url.return_value = "https://example.com/poster.png"
        papers = [
            {
                "title": "Paper 1",
                "session": "Poster",
                "priority": 3,
            }
        ]
        markdown = generate_all_papers_markdown(papers, "Papers")

        assert "**Poster Image:** ![Poster](https://example.com/poster.png)" in markdown


class TestGenerateSearchTermMarkdown:
    """Tests for generate_search_term_markdown function."""

    def test_generate_search_term_basic(self):
        """Test basic search term markdown generation."""
        papers = [
            {
                "title": "Paper 1",
                "session": "Oral",
                "priority": 4,
                "authors": ["Alice"],
                "abstract": "About transformers",
            }
        ]
        markdown = generate_search_term_markdown("transformers", papers)

        assert "# transformers" in markdown
        assert "**Papers:** 1" in markdown
        assert "## Oral" in markdown
        assert "### Paper 1" in markdown

    def test_generate_search_term_multiple_papers(self):
        """Test search term markdown with multiple papers."""
        papers = [
            {"title": "Paper 1", "session": "Oral", "priority": 5},
            {"title": "Paper 2", "session": "Poster", "priority": 4},
        ]
        markdown = generate_search_term_markdown("AI", papers)

        assert "# AI" in markdown
        assert "**Papers:** 2" in markdown
        assert "### Paper 1" in markdown
        assert "### Paper 2" in markdown

    def test_generate_search_term_no_session(self):
        """Test search term markdown with papers without session."""
        papers = [
            {"title": "Paper 1", "priority": 3},
        ]
        markdown = generate_search_term_markdown("ML", papers)

        assert "## No Session" in markdown

    def test_generate_search_term_with_all_fields(self):
        """Test search term markdown with all optional fields."""
        papers = [
            {
                "title": "Paper 1",
                "session": "Oral",
                "priority": 5,
                "authors": ["Alice", "Bob"],
                "poster_position": "O-1",
                "paper_url": "https://openreview.net/forum?id=test",
                "paper_pdf_url": "https://openreview.net/pdf?id=test",
                "url": "https://source.com",
                "abstract": "Test abstract",
            }
        ]
        markdown = generate_search_term_markdown("deep learning", papers)

        assert "**Authors:** Alice, Bob" in markdown
        assert "**Poster:** O-1" in markdown
        assert "**Paper URL:**" in markdown
        assert "**Abstract:**" in markdown

    def test_generate_search_term_pdf_url_conversion(self):
        """Test search term markdown with PDF URL conversion from paper_url."""
        papers = [
            {
                "title": "Paper 1",
                "session": "Poster",
                "priority": 3,
                "paper_url": "https://openreview.net/forum?id=abc123",
                # No paper_pdf_url provided, should be generated from paper_url
            }
        ]
        markdown = generate_search_term_markdown("AI", papers)

        # PDF URL should be generated by replacing forum with pdf
        assert "[View on OpenReview](https://openreview.net/pdf?id=abc123)" in markdown

    @patch("abstracts_explorer.export.markdown.get_poster_url")
    def test_generate_search_term_with_poster_image(self, mock_get_poster_url):
        """Test search term markdown with poster image."""
        mock_get_poster_url.return_value = "https://example.com/poster.png"
        papers = [
            {
                "title": "Paper 1",
                "session": "Poster",
                "priority": 4,
            }
        ]
        markdown = generate_search_term_markdown("ML", papers)

        assert "**Poster Image:** ![Poster](https://example.com/poster.png)" in markdown


class TestGenerateMainReadme:
    """Tests for generate_main_readme function."""

    def test_generate_main_readme_basic(self):
        """Test basic README generation."""
        papers = [
            {
                "title": "Paper 1",
                "searchTerm": "AI",
                "priority": 5,
                "session": "Oral",
                "conference": "NeurIPS",
                "year": 2025,
            }
        ]
        readme = generate_main_readme(papers, "search-rating-poster")

        assert "# NeurIPS 2025 - Interesting Papers" in readme
        assert "## Conference Information" in readme
        assert "## Export Information" in readme
        assert "**Total Papers:** 1" in readme
        assert "## Papers by Search Term" in readme

    def test_generate_main_readme_uses_paper_metadata(self):
        """Test README conference information comes from paper metadata."""
        papers = [
            {
                "title": "Paper 1",
                "searchTerm": "AI",
                "priority": 5,
                "session": "Oral",
                "conference": "NeurIPS",
                "year": 2025,
            }
        ]
        readme = generate_main_readme(papers, "search-rating-poster")

        assert "**Conference:** NeurIPS" in readme
        assert "**Year:** 2025" in readme

    def test_generate_main_readme_does_not_fetch_conference_website(self):
        """Test README generation never performs live conference lookups."""
        papers = [
            {
                "title": "Paper 1",
                "searchTerm": "AI",
                "priority": 5,
                "session": "Oral",
                "conference": "NeurIPS",
                "year": 2025,
            }
        ]
        with patch("requests.get") as mock_get:
            readme = generate_main_readme(papers, "search-rating-poster")

        mock_get.assert_not_called()
        assert "**Dates:**" not in readme
        assert "**Location:**" not in readme
        assert "**About:**" not in readme

    def test_generate_main_readme_sort_order_search_rating_poster(self):
        """Test README with search-rating-poster sort order."""
        papers = [
            {"title": "P1", "searchTerm": "AI", "priority": 5, "session": "Oral"},
            {"title": "P2", "searchTerm": "ML", "priority": 4, "session": "Poster"},
        ]
        readme = generate_main_readme(papers, "search-rating-poster")

        assert "**Sort Order:** Search Term → Rating → Poster #" in readme
        assert "## Papers by Search Term" in readme
        assert "| Search Term | Papers | Sessions | Avg Rating | File |" in readme

    def test_generate_main_readme_sort_order_rating_poster_search(self):
        """Test README with rating-poster-search sort order."""
        papers = [
            {"title": "P1", "searchTerm": "AI", "priority": 5, "session": "Oral"},
            {"title": "P2", "searchTerm": "AI", "priority": 3, "session": "Poster"},
        ]
        readme = generate_main_readme(papers, "rating-poster-search")

        assert "**Sort Order:** Rating → Poster # → Search Term" in readme
        assert "## Papers by Rating" in readme
        assert "| Rating | Papers | Search Terms | File |" in readme

    def test_generate_main_readme_sort_order_poster_search_rating(self):
        """Test README with poster-search-rating sort order."""
        papers = [
            {"title": "P1", "searchTerm": "AI", "priority": 5, "session": "Oral"},
        ]
        readme = generate_main_readme(papers, "poster-search-rating")

        assert "**Sort Order:** Poster # → Search Term → Rating" in readme
        assert "## Papers" in readme
        assert "All papers are organized in a single file" in readme

    def test_generate_main_readme_sessions_overview(self):
        """Test README sessions overview section."""
        papers = [
            {"title": "P1", "searchTerm": "AI", "priority": 5, "session": "Oral"},
            {"title": "P2", "searchTerm": "ML", "priority": 4, "session": "Oral"},
            {"title": "P3", "searchTerm": "CV", "priority": 3, "session": "Poster"},
        ]
        readme = generate_main_readme(papers, "search-rating-poster")

        assert "## Sessions Overview" in readme
        assert "### Oral" in readme
        assert "### Poster" in readme
        assert "**Papers:** 2" in readme  # Oral session

    def test_generate_main_readme_search_terms_summary(self):
        """Test README search terms summary."""
        papers = [
            {"title": "P1", "searchTerm": "AI", "priority": 5, "session": "Oral"},
            {"title": "P2", "searchTerm": "AI", "priority": 3, "session": "Poster"},
            {"title": "P3", "searchTerm": "ML", "priority": 4, "session": "Oral"},
        ]
        readme = generate_main_readme(papers, "search-rating-poster")

        # Search terms appear as links in the table
        assert "| [AI](AI.md) |" in readme
        assert "| [ML](ML.md) |" in readme
        # Check average rating for AI (5+3)/2 = 4.0
        assert "⭐⭐⭐⭐" in readme

    def test_generate_main_readme_unknown_search_term(self):
        """Test README with papers without search term."""
        papers = [
            {"title": "P1", "priority": 5, "session": "Oral"},
        ]
        readme = generate_main_readme(papers, "search-rating-poster")

        assert "Unknown" in readme

    def test_generate_main_readme_long_search_terms_list(self):
        """Test README with many search terms that need truncation in rating view."""
        # Create papers with many different search terms
        papers = []
        for i in range(10):
            papers.append(
                {
                    "title": f"P{i}",
                    "searchTerm": f"SearchTerm{i}",
                    "priority": 5,
                    "session": "Oral",
                }
            )
        readme = generate_main_readme(papers, "rating-poster-search")

        assert "## Papers by Rating" in readme
        # The search terms list should be truncated with "..."
        assert "..." in readme

    def test_generate_main_readme_empty_search_term_sanitization(self):
        """Test README with empty search term in default sort order."""
        papers = [
            {"title": "P1", "searchTerm": "", "priority": 5, "session": "Oral"},
        ]
        readme = generate_main_readme(papers, "search-rating-poster")

        # Empty search term becomes "Unknown" and gets sanitized to "unknown"
        assert "unknown.md" in readme or "Unknown.md" in readme


class TestGenerateFolderStructureExport:
    """Tests for generate_folder_structure_export function."""

    def test_generate_folder_structure_search_rating_poster(self):
        """Test folder structure with search-rating-poster order."""
        papers = [
            {"title": "P1", "searchTerm": "AI", "priority": 5, "session": "Oral"},
            {"title": "P2", "searchTerm": "ML", "priority": 4, "session": "Poster"},
        ]
        zip_buffer = generate_folder_structure_export(papers, "search-rating-poster")

        assert isinstance(zip_buffer, BytesIO)
        zip_buffer.seek(0)

        with zipfile.ZipFile(zip_buffer, "r") as zipf:
            files = zipf.namelist()
            assert "README.md" in files
            assert "AI.md" in files
            assert "ML.md" in files

    def test_generate_folder_structure_rating_poster_search(self):
        """Test folder structure with rating-poster-search order."""
        papers = [
            {"title": "P1", "searchTerm": "AI", "priority": 5, "session": "Oral"},
            {"title": "P2", "searchTerm": "ML", "priority": 3, "session": "Poster"},
        ]
        zip_buffer = generate_folder_structure_export(papers, "rating-poster-search")

        zip_buffer.seek(0)
        with zipfile.ZipFile(zip_buffer, "r") as zipf:
            files = zipf.namelist()
            assert "README.md" in files
            assert "5_stars.md" in files
            assert "3_stars.md" in files

    def test_generate_folder_structure_poster_search_rating(self):
        """Test folder structure with poster-search-rating order."""
        papers = [
            {"title": "P1", "searchTerm": "AI", "priority": 5, "session": "Oral"},
        ]
        zip_buffer = generate_folder_structure_export(papers, "poster-search-rating")

        zip_buffer.seek(0)
        with zipfile.ZipFile(zip_buffer, "r") as zipf:
            files = zipf.namelist()
            assert "README.md" in files
            assert "all_papers.md" in files

    def test_generate_folder_structure_search_term_sanitization(self):
        """Test search term filename sanitization."""
        papers = [
            {"title": "P1", "searchTerm": "AI/ML & DL!", "priority": 5, "session": "Oral"},
        ]
        zip_buffer = generate_folder_structure_export(papers, "search-rating-poster")

        zip_buffer.seek(0)
        with zipfile.ZipFile(zip_buffer, "r") as zipf:
            files = zipf.namelist()
            # Special characters should be removed, spaces become underscores
            # "AI/ML & DL!" -> "AIML__DL" (slashes and special chars removed, spaces become underscores)
            assert any("AIML__DL" in f for f in files)

    def test_generate_folder_structure_empty_search_term(self):
        """Test folder structure with empty/unknown search term."""
        papers = [
            {"title": "P1", "searchTerm": "", "priority": 5, "session": "Oral"},
        ]
        zip_buffer = generate_folder_structure_export(papers, "search-rating-poster")

        zip_buffer.seek(0)
        with zipfile.ZipFile(zip_buffer, "r") as zipf:
            files = zipf.namelist()
            # Empty search term becomes "Unknown" (capitalized because it's treated as a proper term)
            # Then sanitized to "Unknown.md"
            assert "Unknown.md" in files

    def test_generate_folder_structure_zero_priority(self):
        """Test folder structure with zero priority papers."""
        papers = [
            {"title": "P1", "searchTerm": "AI", "priority": 0, "session": "Oral"},
        ]
        zip_buffer = generate_folder_structure_export(papers, "rating-poster-search")

        zip_buffer.seek(0)
        with zipfile.ZipFile(zip_buffer, "r") as zipf:
            files = zipf.namelist()
            assert "0_stars.md" in files

    def test_generate_folder_structure_long_search_term(self):
        """Test search term filename length limiting."""
        papers = [
            {"title": "P1", "searchTerm": "a" * 100, "priority": 5, "session": "Oral"},
        ]
        zip_buffer = generate_folder_structure_export(papers, "search-rating-poster")

        zip_buffer.seek(0)
        with zipfile.ZipFile(zip_buffer, "r") as zipf:
            files = zipf.namelist()
            # Filename should be limited to 50 chars (plus .md)
            md_files = [f for f in files if f.endswith(".md") and f != "README.md"]
            assert len(md_files) > 0
            assert all(len(f) <= 53 for f in md_files)  # 50 + ".md"

    def test_generate_folder_structure_truly_empty_sanitized_name(self):
        """Test folder structure when sanitized name becomes empty after processing."""
        papers = [
            {"title": "P1", "searchTerm": "!!!", "priority": 5, "session": "Oral"},
        ]
        zip_buffer = generate_folder_structure_export(papers, "search-rating-poster")

        zip_buffer.seek(0)
        with zipfile.ZipFile(zip_buffer, "r") as zipf:
            files = zipf.namelist()
            # When all characters are special chars, it should become "unknown"
            assert "unknown.md" in files


class TestExportPapersToZip:
    """Tests for export_papers_to_zip function."""

    @staticmethod
    def _member(zip_buffer, name):
        zip_buffer.seek(0)
        with zipfile.ZipFile(zip_buffer, "r") as zipf:
            return zipf.read(name).decode("utf-8")

    def test_export_papers_to_zip_search_rating_poster(self):
        """Test export with search-rating-poster sorting."""
        papers = [
            {"title": "P1", "searchTerm": "B", "priority": 5, "poster_position": "A10"},
            {"title": "P2", "searchTerm": "A", "priority": 4, "poster_position": "A2"},
            {"title": "P3", "searchTerm": "A", "priority": 5, "poster_position": "A1"},
        ]
        original = [dict(paper) for paper in papers]
        zip_buffer = export_papers_to_zip(papers, "search-rating-poster")

        assert isinstance(zip_buffer, BytesIO)
        assert papers == original
        content = self._member(zip_buffer, "A.md")
        assert content.index("### P3") < content.index("### P2")

    def test_export_papers_to_zip_rating_poster_search(self):
        """Test export with rating-poster-search sorting."""
        papers = [
            {"title": "P1", "searchTerm": "B", "priority": 3, "poster_position": "A10"},
            {"title": "P2", "searchTerm": "A", "priority": 5, "poster_position": "A2"},
            {"title": "P3", "searchTerm": "C", "priority": 5, "poster_position": "A1"},
        ]
        original = [dict(paper) for paper in papers]
        zip_buffer = export_papers_to_zip(papers, "rating-poster-search")

        assert isinstance(zip_buffer, BytesIO)
        assert papers == original
        five_stars = self._member(zip_buffer, "5_stars.md")
        assert five_stars.index("### P3") < five_stars.index("### P2")

    def test_export_papers_to_zip_poster_search_rating(self):
        """Test export with poster-search-rating sorting."""
        papers = [
            {"title": "P1", "searchTerm": "B", "priority": 3, "poster_position": "B10"},
            {"title": "P2", "searchTerm": "A", "priority": 5, "poster_position": "A2"},
            {"title": "P3", "searchTerm": "C", "priority": 4, "poster_position": "A2"},
        ]
        original = [dict(paper) for paper in papers]
        zip_buffer = export_papers_to_zip(papers, "poster-search-rating")

        assert isinstance(zip_buffer, BytesIO)
        assert papers == original
        content = self._member(zip_buffer, "all_papers.md")
        assert content.index("### P2") < content.index("### P3") < content.index("### P1")

    def test_export_papers_to_zip_default_sort_order(self):
        """Test export with invalid sort order falls back to default."""
        papers = [
            {"title": "P1", "searchTerm": "B", "priority": 5},
            {"title": "P2", "searchTerm": "A", "priority": 4},
        ]
        original = [dict(paper) for paper in papers]
        zip_buffer = export_papers_to_zip(papers, "invalid-order")

        assert isinstance(zip_buffer, BytesIO)
        assert papers == original
        zip_buffer.seek(0)
        with zipfile.ZipFile(zip_buffer, "r") as zipf:
            assert {"A.md", "B.md"}.issubset(set(zipf.namelist()))

    def test_export_papers_to_zip_empty_poster_position(self):
        """Test export with missing poster positions."""
        papers = [
            {"title": "P1", "searchTerm": "A", "priority": 5, "poster_position": ""},
            {"title": "P2", "searchTerm": "A", "priority": 5, "poster_position": "A1"},
        ]
        original = [dict(paper) for paper in papers]
        zip_buffer = export_papers_to_zip(papers, "search-rating-poster")

        assert isinstance(zip_buffer, BytesIO)
        assert papers == original
        content = self._member(zip_buffer, "A.md")
        assert content.index("### P1") < content.index("### P2")

    def test_export_papers_to_zip_missing_fields(self):
        """Test export with papers missing optional fields."""
        papers = [
            {"title": "P1"},  # Missing searchTerm, priority, poster_position
            {"title": "P2", "searchTerm": "A"},
        ]
        zip_buffer = export_papers_to_zip(papers, "search-rating-poster")

        assert isinstance(zip_buffer, BytesIO)
        zip_buffer.seek(0)
        with zipfile.ZipFile(zip_buffer, "r") as zipf:
            assert "README.md" in zipf.namelist()

    def test_export_papers_to_zip_natural_sorting(self):
        """Test that export uses natural sorting for poster positions."""
        papers = [
            {"title": "P1", "searchTerm": "A", "priority": 5, "poster_position": "A10"},
            {"title": "P2", "searchTerm": "A", "priority": 5, "poster_position": "A2"},
            {"title": "P3", "searchTerm": "A", "priority": 5, "poster_position": "A1"},
        ]
        original = [dict(paper) for paper in papers]
        zip_buffer = export_papers_to_zip(papers, "search-rating-poster")

        assert isinstance(zip_buffer, BytesIO)
        assert papers == original
        content = self._member(zip_buffer, "A.md")
        assert content.index("### P3") < content.index("### P2") < content.index("### P1")


class TestExportHardening:
    """Tests for filename collisions, escaping, and priority validation."""

    def test_filename_collisions_and_reserved_names(self):
        """Colliding and reserved search terms produce unique linked filenames."""
        papers = [
            {"title": "P1", "searchTerm": "AI/ML", "priority": 5},
            {"title": "P2", "searchTerm": "AIML", "priority": 5},
            {"title": "P3", "searchTerm": "README", "priority": 5},
            {"title": "P4", "searchTerm": "All Papers", "priority": 5},
        ]
        zip_buffer = export_papers_to_zip(papers, "search-rating-poster")

        zip_buffer.seek(0)
        with zipfile.ZipFile(zip_buffer, "r") as zipf:
            names = set(zipf.namelist())
            readme = zipf.read("README.md").decode("utf-8")

        assert {"AIML.md", "AIML_2.md", "README_2.md", "All_Papers_2.md"}.issubset(names)
        assert "(AIML.md)" in readme
        assert "(AIML_2.md)" in readme
        assert "(README_2.md)" in readme
        assert "(All_Papers_2.md)" in readme

    def test_filename_collision_after_truncation(self):
        """Terms that collide after truncation still get unique filenames."""
        papers = [
            {"title": "P1", "searchTerm": "a" * 60, "priority": 5},
            {"title": "P2", "searchTerm": "a" * 60 + "b", "priority": 5},
        ]
        zip_buffer = export_papers_to_zip(papers, "search-rating-poster")

        zip_buffer.seek(0)
        with zipfile.ZipFile(zip_buffer, "r") as zipf:
            markdown_names = [name for name in zipf.namelist() if name != "README.md"]

        assert len(markdown_names) == 2
        assert len(set(markdown_names)) == 2
        assert all(len(name) <= 53 for name in markdown_names)

    def test_markdown_escaping_neutralizes_html_and_bad_urls(self):
        """User-provided text and URLs cannot inject HTML or dangerous links."""
        papers = [
            {
                "title": "Bad | Title\n<script>alert(1)</script>",
                "searchTerm": "AI | ML\nnext",
                "session": "S | 1\nnext",
                "authors": "A <b>B</b>",
                "priority": 3,
                "abstract": "Line1\n<script>x</script>\nLine2",
                "paper_url": "javascript:alert(1)",
                "poster_image_url": "javascript:alert(2)",
            }
        ]
        markdown = generate_search_term_markdown("AI | ML\nnext", papers)

        assert "<script>" not in markdown
        assert "javascript:" not in markdown
        assert "&lt;script&gt;" in markdown
        assert "Bad \\| Title" in markdown
        assert "AI \\| ML next" in markdown
        assert "S \\| 1 next" in markdown
        assert "A &lt;b&gt;B&lt;/b&gt;" in markdown
        assert "Line1\n&lt;script&gt;x&lt;/script&gt;\nLine2" in markdown

    def test_readme_table_escapes_pipes_and_newlines(self):
        """README table cells stay intact for search terms containing pipes."""
        papers = [{"title": "P1", "searchTerm": "A | B\nC", "priority": 5, "session": "Oral"}]
        readme = generate_main_readme(papers, "search-rating-poster")

        assert "| [A \\| B C](A__B_C.md) |" in readme

    @pytest.mark.parametrize(
        ("priority", "expected"),
        [("3", "3/5"), ("abc", "0/5"), (10, "5/5"), (-2, "0/5")],
    )
    def test_priority_values_are_coerced(self, priority, expected):
        """Direct export calls accept invalid/non-integer priorities safely."""
        papers = [{"title": "P1", "priority": priority}]
        markdown = generate_all_papers_markdown(papers, "Papers")

        assert expected in markdown

    def test_export_papers_to_zip_coerces_priority(self):
        """ZIP grouping uses a clamped integer priority."""
        papers = [
            {"title": "P1", "searchTerm": "A", "priority": "3"},
            {"title": "P2", "searchTerm": "A", "priority": "not-a-number"},
            {"title": "P3", "searchTerm": "A", "priority": 10},
        ]
        zip_buffer = export_papers_to_zip(papers, "rating-poster-search")

        zip_buffer.seek(0)
        with zipfile.ZipFile(zip_buffer, "r") as zipf:
            names = set(zipf.namelist())

        assert {"3_stars.md", "0_stars.md", "5_stars.md"}.issubset(names)

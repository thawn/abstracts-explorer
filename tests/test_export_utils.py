"""
Tests for export_utils module.

Tests the export utilities for generating markdown and zip files from papers,
including natural sorting, web scraping, and file generation.
"""

import zipfile
from io import BytesIO
from unittest.mock import Mock, patch

from abstracts_explorer.export_utils import (
    natural_sort_key,
    fetch_conference_info,
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


class TestFetchConferenceInfo:
    """Tests for fetch_conference_info function."""

    @patch("abstracts_explorer.export_utils.requests.get")
    def test_fetch_conference_info_success(self, mock_get):
        """Test successful conference info fetching."""
        # Mock successful HTML response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"""
        <html>
            <head><title>NeurIPS 2025 - Neural Information Processing Systems</title></head>
            <body>
                <h1>38th Conference on Neural Information Processing Systems (NeurIPS 2025)</h1>
                <p>December 9-15, 2025</p>
                <p>Vancouver Convention Centre, Vancouver, Canada</p>
                <div class="about">Leading conference in machine learning and AI</div>
            </body>
        </html>
        """
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        conf_info = fetch_conference_info()

        assert conf_info is not None
        assert "NeurIPS" in conf_info["name"]
        assert "December 9-15, 2025" in conf_info["dates"]
        assert "Vancouver" in conf_info["location"]
        mock_get.assert_called_once()

    @patch("abstracts_explorer.export_utils.requests.get")
    def test_fetch_conference_info_network_error(self, mock_get):
        """Test conference info fetching with network error."""
        mock_get.side_effect = Exception("Network error")

        conf_info = fetch_conference_info()

        assert conf_info is None

    @patch("abstracts_explorer.export_utils.requests.get")
    def test_fetch_conference_info_timeout(self, mock_get):
        """Test conference info fetching with timeout."""
        import requests
        mock_get.side_effect = requests.Timeout()

        conf_info = fetch_conference_info()

        assert conf_info is None

    @patch("abstracts_explorer.export_utils.requests.get")
    def test_fetch_conference_info_http_error(self, mock_get):
        """Test conference info fetching with HTTP error."""
        import requests
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError()
        mock_get.return_value = mock_response

        conf_info = fetch_conference_info()

        assert conf_info is None

    @patch("abstracts_explorer.export_utils.requests.get")
    def test_fetch_conference_info_missing_beautifulsoup(self, mock_get):
        """Test conference info fetching when BeautifulSoup is not available."""
        with patch.dict("sys.modules", {"bs4": None}):
            # This will cause ImportError when trying to import BeautifulSoup
            conf_info = fetch_conference_info()
            
            # Should return None when BeautifulSoup is not available
            # The function catches ImportError and returns None
            assert conf_info is None

    @patch("abstracts_explorer.export_utils.requests.get")
    def test_fetch_conference_info_partial_data(self, mock_get):
        """Test conference info with minimal HTML."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"""
        <html>
            <head><title>NeurIPS 2025</title></head>
            <body><p>Some content</p></body>
        </html>
        """
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        conf_info = fetch_conference_info()

        assert conf_info is not None
        assert "name" in conf_info
        # dates and location might be None with minimal HTML
        assert "dates" in conf_info
        assert "location" in conf_info

    @patch("abstracts_explorer.export_utils.requests.get")
    def test_fetch_conference_info_with_description(self, mock_get):
        """Test conference info with description section."""
        mock_response = Mock()
        mock_response.status_code = 200
        # Create a very long description to test truncation
        long_desc = "A" * 600  # Over 500 chars
        mock_response.content = f"""
        <html>
            <head><title>NeurIPS 2025</title></head>
            <body>
                <div class="about-section">{long_desc}</div>
            </body>
        </html>
        """.encode()
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        conf_info = fetch_conference_info()

        assert conf_info is not None
        assert conf_info["description"] is not None
        # Description should be truncated to 500 chars + "..."
        assert len(conf_info["description"]) == 503
        assert conf_info["description"].endswith("...")


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

    @patch("abstracts_explorer.export_utils.get_poster_url")
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

    @patch("abstracts_explorer.export_utils.get_poster_url")
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
            }
        ]
        readme = generate_main_readme(papers, "AI search", "search-rating-poster")
        
        assert "# NeurIPS 2025 - Interesting Papers" in readme
        assert "## Conference Information" in readme
        assert "## Export Information" in readme
        assert "**Total Papers:** 1" in readme
        assert "## Papers by Search Term" in readme

    @patch("abstracts_explorer.export_utils.fetch_conference_info")
    def test_generate_main_readme_with_conference_info(self, mock_fetch):
        """Test README with fetched conference info."""
        mock_fetch.return_value = {
            "name": "NeurIPS 2025",
            "dates": "December 9-15, 2025",
            "location": "Vancouver, Canada",
            "description": "Leading ML conference",
        }
        papers = [
            {"title": "Paper 1", "searchTerm": "AI", "priority": 5, "session": "Oral"}
        ]
        readme = generate_main_readme(papers, "search", "search-rating-poster")
        
        assert "**Conference:** NeurIPS 2025" in readme
        assert "**Dates:** December 9-15, 2025" in readme
        assert "**Location:** Vancouver, Canada" in readme
        assert "**About:** Leading ML conference" in readme

    @patch("abstracts_explorer.export_utils.fetch_conference_info")
    def test_generate_main_readme_fallback_conference_info(self, mock_fetch):
        """Test README with fallback conference info."""
        mock_fetch.return_value = None
        papers = [
            {"title": "Paper 1", "searchTerm": "AI", "priority": 5, "session": "Oral"}
        ]
        readme = generate_main_readme(papers, "search", "search-rating-poster")
        
        assert "**Conference:** 38th Conference on Neural Information Processing Systems (NeurIPS 2025)" in readme
        assert "**Dates:** December 9-15, 2025" in readme
        assert "**Location:** Vancouver Convention Centre" in readme

    def test_generate_main_readme_sort_order_search_rating_poster(self):
        """Test README with search-rating-poster sort order."""
        papers = [
            {"title": "P1", "searchTerm": "AI", "priority": 5, "session": "Oral"},
            {"title": "P2", "searchTerm": "ML", "priority": 4, "session": "Poster"},
        ]
        readme = generate_main_readme(papers, "search", "search-rating-poster")
        
        assert "**Sort Order:** Search Term → Rating → Poster #" in readme
        assert "## Papers by Search Term" in readme
        assert "| Search Term | Papers | Sessions | Avg Rating | File |" in readme

    def test_generate_main_readme_sort_order_rating_poster_search(self):
        """Test README with rating-poster-search sort order."""
        papers = [
            {"title": "P1", "searchTerm": "AI", "priority": 5, "session": "Oral"},
            {"title": "P2", "searchTerm": "AI", "priority": 3, "session": "Poster"},
        ]
        readme = generate_main_readme(papers, "search", "rating-poster-search")
        
        assert "**Sort Order:** Rating → Poster # → Search Term" in readme
        assert "## Papers by Rating" in readme
        assert "| Rating | Papers | Search Terms | File |" in readme

    def test_generate_main_readme_sort_order_poster_search_rating(self):
        """Test README with poster-search-rating sort order."""
        papers = [
            {"title": "P1", "searchTerm": "AI", "priority": 5, "session": "Oral"},
        ]
        readme = generate_main_readme(papers, "search", "poster-search-rating")
        
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
        readme = generate_main_readme(papers, "search", "search-rating-poster")
        
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
        readme = generate_main_readme(papers, "search", "search-rating-poster")
        
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
        readme = generate_main_readme(papers, "search", "search-rating-poster")
        
        assert "Unknown" in readme

    def test_generate_main_readme_long_search_terms_list(self):
        """Test README with many search terms that need truncation in rating view."""
        # Create papers with many different search terms
        papers = []
        for i in range(10):
            papers.append({
                "title": f"P{i}",
                "searchTerm": f"SearchTerm{i}",
                "priority": 5,
                "session": "Oral",
            })
        readme = generate_main_readme(papers, "search", "rating-poster-search")
        
        assert "## Papers by Rating" in readme
        # The search terms list should be truncated with "..."
        assert "..." in readme

    def test_generate_main_readme_empty_search_term_sanitization(self):
        """Test README with empty search term in default sort order."""
        papers = [
            {"title": "P1", "searchTerm": "", "priority": 5, "session": "Oral"},
        ]
        readme = generate_main_readme(papers, "search", "search-rating-poster")
        
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
        zip_buffer = generate_folder_structure_export(papers, "test", "search-rating-poster")
        
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
        zip_buffer = generate_folder_structure_export(papers, "test", "rating-poster-search")
        
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
        zip_buffer = generate_folder_structure_export(papers, "test", "poster-search-rating")
        
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
        zip_buffer = generate_folder_structure_export(papers, "test", "search-rating-poster")
        
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
        zip_buffer = generate_folder_structure_export(papers, "test", "search-rating-poster")
        
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
        zip_buffer = generate_folder_structure_export(papers, "test", "rating-poster-search")
        
        zip_buffer.seek(0)
        with zipfile.ZipFile(zip_buffer, "r") as zipf:
            files = zipf.namelist()
            assert "0_stars.md" in files

    def test_generate_folder_structure_long_search_term(self):
        """Test search term filename length limiting."""
        papers = [
            {"title": "P1", "searchTerm": "a" * 100, "priority": 5, "session": "Oral"},
        ]
        zip_buffer = generate_folder_structure_export(papers, "test", "search-rating-poster")
        
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
        zip_buffer = generate_folder_structure_export(papers, "test", "search-rating-poster")
        
        zip_buffer.seek(0)
        with zipfile.ZipFile(zip_buffer, "r") as zipf:
            files = zipf.namelist()
            # When all characters are special chars, it should become "unknown"
            assert "unknown.md" in files


class TestExportPapersToZip:
    """Tests for export_papers_to_zip function."""

    def test_export_papers_to_zip_search_rating_poster(self):
        """Test export with search-rating-poster sorting."""
        papers = [
            {"title": "P1", "searchTerm": "B", "priority": 5, "poster_position": "A10"},
            {"title": "P2", "searchTerm": "A", "priority": 4, "poster_position": "A2"},
            {"title": "P3", "searchTerm": "A", "priority": 5, "poster_position": "A1"},
        ]
        zip_buffer = export_papers_to_zip(papers, "test", "search-rating-poster")
        
        assert isinstance(zip_buffer, BytesIO)
        # Papers should be sorted by searchTerm, then priority (desc), then poster
        # Expected order: A-P3(5-A1), A-P2(4-A2), B-P1(5-A10)
        assert papers[0]["searchTerm"] == "A"
        assert papers[0]["priority"] == 5
        assert papers[1]["searchTerm"] == "A"
        assert papers[1]["priority"] == 4
        assert papers[2]["searchTerm"] == "B"

    def test_export_papers_to_zip_rating_poster_search(self):
        """Test export with rating-poster-search sorting."""
        papers = [
            {"title": "P1", "searchTerm": "B", "priority": 3, "poster_position": "A10"},
            {"title": "P2", "searchTerm": "A", "priority": 5, "poster_position": "A2"},
            {"title": "P3", "searchTerm": "C", "priority": 5, "poster_position": "A1"},
        ]
        zip_buffer = export_papers_to_zip(papers, "test", "rating-poster-search")
        
        assert isinstance(zip_buffer, BytesIO)
        # Papers should be sorted by priority (desc), then poster, then searchTerm
        # Expected: P3(5-A1-C), P2(5-A2-A), P1(3-A10-B)
        assert papers[0]["priority"] == 5
        assert papers[0]["poster_position"] == "A1"
        assert papers[1]["priority"] == 5
        assert papers[1]["poster_position"] == "A2"
        assert papers[2]["priority"] == 3

    def test_export_papers_to_zip_poster_search_rating(self):
        """Test export with poster-search-rating sorting."""
        papers = [
            {"title": "P1", "searchTerm": "B", "priority": 3, "poster_position": "B10"},
            {"title": "P2", "searchTerm": "A", "priority": 5, "poster_position": "A2"},
            {"title": "P3", "searchTerm": "C", "priority": 4, "poster_position": "A2"},
        ]
        zip_buffer = export_papers_to_zip(papers, "test", "poster-search-rating")
        
        assert isinstance(zip_buffer, BytesIO)
        # Papers should be sorted by poster, then searchTerm, then priority (desc)
        # Expected: P2(A2-A-5), P3(A2-C-4), P1(B10-B-3)
        assert papers[0]["poster_position"] == "A2"
        assert papers[0]["searchTerm"] == "A"
        assert papers[1]["poster_position"] == "A2"
        assert papers[1]["searchTerm"] == "C"

    def test_export_papers_to_zip_default_sort_order(self):
        """Test export with invalid sort order falls back to default."""
        papers = [
            {"title": "P1", "searchTerm": "B", "priority": 5},
            {"title": "P2", "searchTerm": "A", "priority": 4},
        ]
        zip_buffer = export_papers_to_zip(papers, "test", "invalid-order")
        
        assert isinstance(zip_buffer, BytesIO)
        # Should fall back to search-rating-poster
        assert papers[0]["searchTerm"] == "A"
        assert papers[1]["searchTerm"] == "B"

    def test_export_papers_to_zip_empty_poster_position(self):
        """Test export with missing poster positions."""
        papers = [
            {"title": "P1", "searchTerm": "A", "priority": 5, "poster_position": ""},
            {"title": "P2", "searchTerm": "A", "priority": 5, "poster_position": "A1"},
        ]
        zip_buffer = export_papers_to_zip(papers, "test", "search-rating-poster")
        
        assert isinstance(zip_buffer, BytesIO)
        # Paper with empty poster_position should come before A1
        assert papers[0]["poster_position"] == ""
        assert papers[1]["poster_position"] == "A1"

    def test_export_papers_to_zip_missing_fields(self):
        """Test export with papers missing optional fields."""
        papers = [
            {"title": "P1"},  # Missing searchTerm, priority, poster_position
            {"title": "P2", "searchTerm": "A"},
        ]
        zip_buffer = export_papers_to_zip(papers, "test", "search-rating-poster")
        
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
        zip_buffer = export_papers_to_zip(papers, "test", "search-rating-poster")
        
        assert isinstance(zip_buffer, BytesIO)
        # Natural sort: A1, A2, A10 (not A1, A10, A2)
        assert papers[0]["poster_position"] == "A1"
        assert papers[1]["poster_position"] == "A2"
        assert papers[2]["poster_position"] == "A10"

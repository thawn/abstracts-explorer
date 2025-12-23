"""
Tests for the ML4PS downloader plugin.

This module contains unit tests and end-to-end tests for the ML4PS
workshop downloader plugin, including the lightweight API conversion.
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import requests

from neurips_abstracts.plugins.ml4ps_downloader import ML4PSDownloaderPlugin
from neurips_abstracts.plugins import (
    LightweightDownloaderPlugin,
    DownloaderPlugin,
    get_plugin,
    list_plugins,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def ml4ps_plugin():
    """Create an ML4PS plugin instance."""
    return ML4PSDownloaderPlugin()


@pytest.fixture
def mock_html_page():
    """Mock HTML page content from ML4PS website."""
    return """
    <html>
    <body>
        <h2>Papers</h2>
        <table>
            <tr>
                <td>1</td>
                <td>
                    <strong>Test Paper Title One</strong><br/>
                    John Doe, Jane Smith<br/>
                    <a href="/files/paper1.pdf">paper</a>
                    <a href="/assets/posters/123456.png">poster</a>
                </td>
            </tr>
            <tr>
                <td>2</td>
                <td>
                    <strong>Test Paper Title Two [Best Poster]</strong><br/>
                    Alice Johnson, Bob Wilson<br/>
                    <a href="/files/paper2.pdf">paper</a>
                    <a href="/assets/posters/123457.png">poster</a>
                    Spotlight Talk
                </td>
            </tr>
            <tr>
                <td>3</td>
                <td>
                    <strong>Test Paper Title Three</strong><br/>
                    Charlie Brown<br/>
                    <a href="/files/paper3.pdf">paper</a>
                </td>
            </tr>
        </table>
    </body>
    </html>
    """


@pytest.fixture
def mock_neurips_virtual_page():
    """Mock NeurIPS virtual page with abstract and OpenReview link."""
    return """
    <html>
    <body>
        <h3>Abstract</h3>
        <p>This is a test abstract about machine learning applications in physics.</p>
        <a class="action-btn" href="https://openreview.net/forum?id=abc123">OpenReview</a>
    </body>
    </html>
    """


@pytest.fixture
def sample_scraped_papers():
    """Sample scraped papers in internal format."""
    return [
        {
            "id": 1,
            "title": "Test Paper Title One",
            "authors_str": "John Doe, Jane Smith",
            "paper_url": "https://ml4physicalsciences.github.io/2025/files/paper1.pdf",
            "poster_url": "https://ml4physicalsciences.github.io/2025/assets/posters/123456.png",
            "video_url": None,
            "awards": [],
            "abstract": "This is a test abstract.",
            "neurips_paper_id": "123456",
            "openreview_url": "https://openreview.net/forum?id=abc123",
            "eventtype": "Poster",
            "decision": "Accept (poster)",
        },
        {
            "id": 2,
            "title": "Test Paper Title Two",
            "authors_str": "Alice Johnson, Bob Wilson",
            "paper_url": "https://ml4physicalsciences.github.io/2025/files/paper2.pdf",
            "poster_url": "https://ml4physicalsciences.github.io/2025/assets/posters/123457.png",
            "video_url": None,
            "awards": ["Best Poster", "Spotlight Talk"],
            "abstract": "Another test abstract.",
            "neurips_paper_id": "123457",
            "openreview_url": "https://openreview.net/forum?id=def456",
            "eventtype": "Spotlight",
            "decision": "Accept (spotlight)",
        },
    ]


@pytest.fixture
def sample_lightweight_papers():
    """Sample papers in lightweight format."""
    return [
        {
            "title": "Test Paper Title One",
            "authors": ["John Doe", "Jane Smith"],
            "abstract": "This is a test abstract.",
            "session": "ML4PhysicalSciences 2025 Workshop",
            "poster_position": "1",
            "id": 1,
            "paper_pdf_url": "https://ml4physicalsciences.github.io/2025/files/paper1.pdf",
            "poster_image_url": "https://ml4physicalsciences.github.io/2025/assets/posters/123456.png",
            "url": "https://openreview.net/forum?id=abc123",
            "keywords": [],
        },
        {
            "title": "Test Paper Title Two",
            "authors": ["Alice Johnson", "Bob Wilson"],
            "abstract": "Another test abstract.",
            "session": "ML4PhysicalSciences 2025 Workshop - Spotlight",
            "poster_position": "2",
            "id": 2,
            "paper_pdf_url": "https://ml4physicalsciences.github.io/2025/files/paper2.pdf",
            "poster_image_url": "https://ml4physicalsciences.github.io/2025/assets/posters/123457.png",
            "url": "https://openreview.net/forum?id=def456",
            "keywords": ["Best Poster", "Spotlight Talk"],
        },
    ]


# ============================================================================
# Unit Tests - Plugin Properties
# ============================================================================


class TestML4PSPluginProperties:
    """Test basic plugin properties and metadata."""

    def test_plugin_instantiation(self, ml4ps_plugin):
        """Test that plugin can be instantiated."""
        assert ml4ps_plugin is not None
        assert isinstance(ml4ps_plugin, ML4PSDownloaderPlugin)

    def test_plugin_inheritance(self, ml4ps_plugin):
        """Test that plugin inherits from LightweightDownloaderPlugin."""
        assert isinstance(ml4ps_plugin, LightweightDownloaderPlugin)
        assert isinstance(ml4ps_plugin, DownloaderPlugin)

    def test_plugin_metadata(self, ml4ps_plugin):
        """Test plugin metadata."""
        metadata = ml4ps_plugin.get_metadata()
        assert metadata["name"] == "ml4ps"
        assert metadata["description"] == "ML4PS (Machine Learning for Physical Sciences) workshop downloader"
        assert 2025 in metadata["supported_years"]
        assert "parameters" in metadata

    def test_plugin_name(self, ml4ps_plugin):
        """Test plugin name attribute."""
        assert ml4ps_plugin.plugin_name == "ml4ps"

    def test_plugin_supported_years(self, ml4ps_plugin):
        """Test supported years."""
        assert ml4ps_plugin.supported_years == [2025]

    def test_plugin_base_url(self, ml4ps_plugin):
        """Test base URL configuration."""
        assert ml4ps_plugin.BASE_URL == "https://ml4physicalsciences.github.io/2025/"
        assert "neurips.cc" in ml4ps_plugin.NEURIPS_VIRTUAL_BASE


# ============================================================================
# Unit Tests - Validation
# ============================================================================


class TestML4PSPluginValidation:
    """Test plugin validation methods."""

    def test_validate_year_valid(self, ml4ps_plugin):
        """Test validation with valid year."""
        # Should not raise
        ml4ps_plugin.validate_year(2025)

    def test_validate_year_invalid(self, ml4ps_plugin):
        """Test validation with invalid year."""
        with pytest.raises(ValueError, match="not supported"):
            ml4ps_plugin.validate_year(2024)

    def test_validate_year_none(self, ml4ps_plugin):
        """Test validation with None year."""
        # Should not raise
        ml4ps_plugin.validate_year(None)


# ============================================================================
# Unit Tests - Helper Methods
# ============================================================================


class TestML4PSPluginHelpers:
    """Test plugin helper methods."""

    def test_clean_text(self, ml4ps_plugin):
        """Test text cleaning."""
        text = "  Test   text [paper] [poster]  🏅  "
        cleaned = ml4ps_plugin._clean_text(text)
        assert cleaned == "Test text"

    def test_clean_text_awards(self, ml4ps_plugin):
        """Test cleaning text with link markers."""
        text = "Title [paper] [poster] [video]"
        cleaned = ml4ps_plugin._clean_text(text)
        assert cleaned == "Title"
        assert "[" not in cleaned
        assert "]" not in cleaned

    def test_extract_paper_id_from_poster_url(self, ml4ps_plugin):
        """Test extracting paper ID from poster URL."""
        url = "https://ml4physicalsciences.github.io/2025/assets/posters/123456.png"
        paper_id = ml4ps_plugin._extract_paper_id_from_poster_url(url)
        assert paper_id == "123456"

    def test_extract_paper_id_no_match(self, ml4ps_plugin):
        """Test extracting paper ID with no match."""
        url = "https://example.com/poster.png"
        paper_id = ml4ps_plugin._extract_paper_id_from_poster_url(url)
        assert paper_id is None


# ============================================================================
# Unit Tests - Lightweight Conversion
# ============================================================================


class TestML4PSLightweightConversion:
    """Test conversion to lightweight format."""

    def test_convert_to_lightweight_format(self, ml4ps_plugin, sample_scraped_papers):
        """Test conversion to lightweight format."""
        lightweight = ml4ps_plugin._convert_to_lightweight_format(sample_scraped_papers)

        assert len(lightweight) == 2

        # Check first paper
        paper1 = lightweight[0]
        assert paper1["title"] == "Test Paper Title One"
        assert paper1["authors"] == ["John Doe", "Jane Smith"]
        assert paper1["abstract"] == "This is a test abstract."
        assert paper1["session"] == "ML4PhysicalSciences 2025 Workshop"
        assert paper1["poster_position"] == "1"
        assert paper1["id"] == 1

    def test_convert_spotlight_session(self, ml4ps_plugin, sample_scraped_papers):
        """Test that spotlight papers get special session name."""
        lightweight = ml4ps_plugin._convert_to_lightweight_format(sample_scraped_papers)

        paper2 = lightweight[1]
        assert "Spotlight" in paper2["session"]

    def test_convert_includes_optional_fields(self, ml4ps_plugin, sample_scraped_papers):
        """Test that optional fields are included."""
        lightweight = ml4ps_plugin._convert_to_lightweight_format(sample_scraped_papers)

        paper1 = lightweight[0]
        assert "paper_pdf_url" in paper1
        assert "poster_image_url" in paper1
        assert "url" in paper1
        # awards field is not set for paper1 in test data

    def test_convert_awards_to_award_field(self, ml4ps_plugin, sample_scraped_papers):
        """Test that awards become award string."""
        lightweight = ml4ps_plugin._convert_to_lightweight_format(sample_scraped_papers)

        paper2 = lightweight[1]
        assert paper2["award"] == "Best Poster, Spotlight Talk"


# ============================================================================
# Unit Tests - Web Scraping (Mocked)
# ============================================================================


class TestML4PSWebScraping:
    """Test web scraping functionality with mocks."""

    def test_fetch_page_success(self, ml4ps_plugin, mock_html_page):
        """Test successful page fetching."""
        mock_response = Mock()
        mock_response.content = mock_html_page.encode()
        mock_response.raise_for_status = Mock()

        with patch.object(ml4ps_plugin.session, "get", return_value=mock_response):
            soup = ml4ps_plugin._fetch_page("https://example.com")
            assert soup is not None
            assert soup.find("h2", string=lambda x: x and "Papers" in x) is not None

    def test_fetch_page_failure(self, ml4ps_plugin):
        """Test page fetching with network error."""
        with patch.object(ml4ps_plugin.session, "get", side_effect=requests.RequestException("Network error")):
            soup = ml4ps_plugin._fetch_page("https://example.com")
            assert soup is None

    @pytest.mark.slow
    def test_fetch_page_retries(self, ml4ps_plugin):
        """Test that fetch_page retries on failure."""
        with patch.object(
            ml4ps_plugin.session, "get", side_effect=requests.RequestException("Network error")
        ) as mock_get:
            ml4ps_plugin._fetch_page("https://example.com", max_retries=3)
            assert mock_get.call_count == 3


# ============================================================================
# Unit Tests - Abstract Fetching
# ============================================================================


class TestML4PSAbstractFetching:
    """Test abstract and OpenReview URL fetching."""

    def test_fetch_abstract_and_openreview(self, ml4ps_plugin, mock_neurips_virtual_page):
        """Test fetching abstract and OpenReview URL."""
        mock_response = Mock()
        mock_response.content = mock_neurips_virtual_page.encode()
        mock_response.raise_for_status = Mock()

        with patch.object(ml4ps_plugin.session, "get", return_value=mock_response):
            abstract, openreview_url = ml4ps_plugin._fetch_abstract_and_openreview("123456")

            assert abstract is not None
            assert "machine learning applications in physics" in abstract.lower()
            assert openreview_url == "https://openreview.net/forum?id=abc123"

    def test_fetch_single_abstract_success(self, ml4ps_plugin, mock_neurips_virtual_page):
        """Test fetching abstract for single paper."""
        paper = {
            "id": 1,
            "poster_url": "https://example.com/posters/123456.png",
        }

        mock_response = Mock()
        mock_response.content = mock_neurips_virtual_page.encode()
        mock_response.raise_for_status = Mock()

        with patch.object(ml4ps_plugin.session, "get", return_value=mock_response):
            updated_paper, success = ml4ps_plugin._fetch_single_abstract(paper)

            assert success
            assert "abstract" in updated_paper
            assert "openreview_url" in updated_paper
            assert updated_paper["neurips_paper_id"] == "123456"

    def test_fetch_single_abstract_no_poster_url(self, ml4ps_plugin):
        """Test fetching abstract when poster URL is missing."""
        paper = {"id": 1}

        updated_paper, success = ml4ps_plugin._fetch_single_abstract(paper)

        assert not success
        assert updated_paper == paper


# ============================================================================
# Integration Tests - Full Pipeline
# ============================================================================


class TestML4PSFullPipeline:
    """Test full download pipeline with mocks."""

    @patch.object(ML4PSDownloaderPlugin, "_scrape_papers")
    @patch.object(ML4PSDownloaderPlugin, "_fetch_abstracts_for_papers")
    def test_download_without_abstracts_mock(self, mock_fetch, mock_scrape, ml4ps_plugin, sample_scraped_papers):
        """Test download with abstracts mocked (abstracts are always fetched)."""
        mock_scrape.return_value = sample_scraped_papers

        result = ml4ps_plugin.download(year=2025)

        # Result is now a list of LightweightPaper objects
        assert isinstance(result, list)
        assert len(result) == 2
        mock_scrape.assert_called_once()
        mock_fetch.assert_called_once()

    @patch.object(ML4PSDownloaderPlugin, "_scrape_papers")
    @patch.object(ML4PSDownloaderPlugin, "_fetch_abstracts_for_papers")
    def test_download_with_abstracts(self, mock_fetch, mock_scrape, ml4ps_plugin, sample_scraped_papers):
        """Test download with abstract fetching."""
        mock_scrape.return_value = sample_scraped_papers

        result = ml4ps_plugin.download(year=2025)

        # Result is now a list of LightweightPaper objects
        assert isinstance(result, list)
        assert len(result) == 2
        mock_scrape.assert_called_once()
        mock_fetch.assert_called_once()

    @patch.object(ML4PSDownloaderPlugin, "_scrape_papers")
    @patch.object(ML4PSDownloaderPlugin, "_fetch_abstracts_for_papers")
    def test_download_saves_to_file(self, mock_fetch, mock_scrape, ml4ps_plugin, sample_scraped_papers, tmp_path):
        """Test that download saves to file."""
        mock_scrape.return_value = sample_scraped_papers
        output_file = tmp_path / "ml4ps_output.json"

        result = ml4ps_plugin.download(year=2025, output_path=str(output_file))

        assert output_file.exists()
        with open(output_file) as f:
            saved_data = json.load(f)
        # saved_data is a list of paper dicts, result is a list of LightweightPaper objects
        assert len(saved_data) == len(result)

    @patch.object(ML4PSDownloaderPlugin, "_scrape_papers")
    def test_download_loads_from_existing_file(self, mock_scrape, ml4ps_plugin, tmp_path):
        """Test that download loads from existing file when force_download=False."""
        output_file = tmp_path / "ml4ps_output.json"

        # Create existing file with lightweight format (list of paper dicts)
        existing_data = [
            {
                "title": "Cached Paper",
                "abstract": "Test abstract",
                "authors": ["Test Author"],
                "session": "Test Session",
                "poster_position": "1",
                "year": 2025,
                "conference": "NeurIPS",
            }
        ]
        with open(output_file, "w") as f:
            json.dump(existing_data, f)

        result = ml4ps_plugin.download(year=2025, output_path=str(output_file), force_download=False)

        # Should load from file, not scrape
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].title == "Cached Paper"
        mock_scrape.assert_not_called()


# ============================================================================
# End-to-End Tests
# ============================================================================


class TestML4PSEndToEnd:
    """End-to-end tests for ML4PS plugin (slow, marked for optional execution)."""

    def test_plugin_registration(self):
        """Test that plugin is registered correctly."""
        # Import to trigger registration
        import neurips_abstracts.plugins.ml4ps_downloader  # noqa: F401

        plugins = list_plugins()
        plugin_names = [p["name"] for p in plugins]
        assert "ml4ps" in plugin_names

    def test_get_plugin(self):
        """Test retrieving plugin from registry."""
        # Import to trigger registration
        import neurips_abstracts.plugins.ml4ps_downloader  # noqa: F401

        plugin = get_plugin("ml4ps")
        assert plugin is not None
        assert isinstance(plugin, ML4PSDownloaderPlugin)

    @pytest.mark.slow
    def test_download_real_data(self, tmp_path):
        """Test downloading real data from ML4PS website."""
        plugin = ML4PSDownloaderPlugin()
        output_file = tmp_path / "ml4ps_real.json"

        result = plugin.download(
            year=2025,
            output_path=str(output_file),
            max_workers=10,
        )

        # Result is now a list of LightweightPaper objects
        assert isinstance(result, list)
        assert len(result) > 0
        assert output_file.exists()

        # Verify schema - papers are LightweightPaper objects
        first_paper = result[0]
        assert hasattr(first_paper, "title")
        assert hasattr(first_paper, "authors")
        assert hasattr(first_paper, "session")
        assert hasattr(first_paper, "year")
        assert hasattr(first_paper, "conference")


# ============================================================================
# Regression Tests
# ============================================================================


class TestML4PSRegression:
    """Regression tests to ensure compatibility."""

    @pytest.mark.slow
    @patch.object(ML4PSDownloaderPlugin, "_scrape_papers")
    def test_output_schema_matches_lightweight_format(self, mock_scrape, ml4ps_plugin, sample_scraped_papers):
        """Test that output schema matches the lightweight format."""
        mock_scrape.return_value = sample_scraped_papers

        result = ml4ps_plugin.download(year=2025)

        # Result should be a list of LightweightPaper objects
        assert isinstance(result, list)
        assert len(result) > 0

        # Check first paper has required lightweight schema fields
        first_paper = result[0]

        # Required fields in LightweightPaper model
        assert hasattr(first_paper, "title")
        assert hasattr(first_paper, "authors")
        assert hasattr(first_paper, "abstract")
        assert hasattr(first_paper, "session")
        assert hasattr(first_paper, "poster_position")
        assert hasattr(first_paper, "year")
        assert hasattr(first_paper, "conference")

        # Verify data types
        assert isinstance(first_paper.title, str)
        assert isinstance(first_paper.authors, list)
        assert isinstance(first_paper.abstract, str)
        assert isinstance(first_paper.session, str)
        assert isinstance(first_paper.poster_position, str)
        assert isinstance(first_paper.year, int)
        assert isinstance(first_paper.conference, str)

        # Optional fields
        assert hasattr(first_paper, "paper_pdf_url")
        assert hasattr(first_paper, "poster_image_url")
        assert hasattr(first_paper, "url")
        assert hasattr(first_paper, "keywords")

    def test_author_format_compatibility(self, ml4ps_plugin, sample_scraped_papers):
        """Test that authors are formatted correctly in lightweight schema."""
        lightweight = ml4ps_plugin._convert_to_lightweight_format(sample_scraped_papers)

        # Lightweight schema now uses simple list of author names
        assert isinstance(lightweight, list)
        assert len(lightweight) > 0

        paper = lightweight[0]
        assert "authors" in paper
        assert isinstance(paper["authors"], list)
        assert len(paper["authors"]) > 0
        # Authors are now simple strings, not dicts
        assert isinstance(paper["authors"][0], str)

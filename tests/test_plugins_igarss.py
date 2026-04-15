"""
Tests for IGARSS Downloader Plugin
====================================

Test suite for the IGARSS conference data downloader plugin.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock, call
import requests

from abstracts_explorer.plugins.igarss_downloader import (
    IGARSSDownloaderPlugin,
    _strip_highlight_tags,
)
from abstracts_explorer.database import DatabaseManager
from tests.conftest import set_test_db

# ---------------------------------------------------------------------------
# Sample data helpers
# ---------------------------------------------------------------------------

SAMPLE_RECORD_1 = {
    "articleNumber": "9876543",
    "articleTitle": "Satellite Image Classification Using Deep Learning",
    "abstract": "This paper presents a novel approach for satellite image classification.",
    "authors": [
        {
            "preferredName": "Alice Smith",
            "authorName": "Smith, Alice",
            "normalizedName": "A. Smith",
            "id": "37085001",
            "affiliation": "MIT",
        },
        {
            "preferredName": "Bob Jones",
            "authorName": "Jones, Bob",
            "normalizedName": "B. Jones",
            "id": "37085002",
            "affiliation": "Stanford",
        },
    ],
    "publicationTitle": "IGARSS 2025 - 2025 IEEE International Geoscience and Remote Sensing Symposium",
    "doi": "10.1109/IGARSS52108.2025.0000001",
    "startPage": "1234",
    "endPage": "1237",
    "publicationYear": "2025",
    "contentType": "Conferences",
    "indexTerms": {
        "ieee_terms": {"terms": ["Remote sensing", "Deep learning"]},
        "author_terms": {"terms": ["SAR", "Image classification"]},
    },
}

SAMPLE_RECORD_2 = {
    "articleNumber": "9876544",
    "articleTitle": "Radar Remote Sensing for Forest Monitoring",
    "abstract": "We propose a radar-based method for monitoring forest cover changes.",
    "authors": [
        {
            "preferredName": "Charlie Brown",
            "authorName": "Brown, Charlie",
            "normalizedName": "C. Brown",
            "id": "37085003",
        },
    ],
    "publicationTitle": "IGARSS 2025 - 2025 IEEE International Geoscience and Remote Sensing Symposium",
    "doi": "10.1109/IGARSS52108.2025.0000002",
    "pdfUrl": "/stamp/stamp.jsp?arnumber=9876544",
    "startPage": "1238",
    "endPage": "1241",
    "publicationYear": "2025",
    "contentType": "Conferences",
    "indexTerms": {
        "ieee_terms": {"terms": ["Radar", "Forestry"]},
    },
}

SAMPLE_RECORD_HIGHLIGHTED = {
    "articleNumber": "9876545",
    "articleTitle": "<highlight>IGARSS</highlight> Study on Urban Areas",
    "abstract": "An <highlight>IGARSS</highlight> study of urban heat islands.",
    "authors": [
        {
            "preferredName": "Diana Prince",
            "authorName": "Prince, Diana",
        },
    ],
    "publicationTitle": "IGARSS 2025",
    "doi": "10.1109/IGARSS52108.2025.0000003",
}


def _make_search_response(records, total_records=None):
    """Return a Mock that behaves like a successful IEEE Xplore search response."""
    if total_records is None:
        total_records = len(records)
    mock_response = Mock()
    mock_response.json.return_value = {
        "totalRecords": total_records,
        "records": records,
    }
    mock_response.raise_for_status = Mock()
    return mock_response


# ---------------------------------------------------------------------------
# _strip_highlight_tags tests
# ---------------------------------------------------------------------------


class TestStripHighlightTags:
    """Test the highlight tag stripping utility."""

    def test_removes_highlight_tags(self):
        assert _strip_highlight_tags("<highlight>IGARSS</highlight>") == "IGARSS"

    def test_removes_multiple_highlight_tags(self):
        text = "<highlight>Remote</highlight> <highlight>Sensing</highlight>"
        assert _strip_highlight_tags(text) == "Remote Sensing"

    def test_plain_text_unchanged(self):
        assert _strip_highlight_tags("No tags here") == "No tags here"

    def test_empty_string(self):
        assert _strip_highlight_tags("") == ""

    def test_nested_content(self):
        text = "Analysis of <highlight>SAR</highlight> data"
        assert _strip_highlight_tags(text) == "Analysis of SAR data"


# ---------------------------------------------------------------------------
# Plugin metadata tests
# ---------------------------------------------------------------------------


class TestIGARSSPluginMetadata:
    """Test plugin metadata and initialization."""

    def test_plugin_name(self):
        plugin = IGARSSDownloaderPlugin()
        assert plugin.plugin_name == "igarss"

    def test_plugin_description(self):
        plugin = IGARSSDownloaderPlugin()
        assert "IGARSS" in plugin.plugin_description
        assert "IEEE" in plugin.plugin_description

    def test_conference_name(self):
        plugin = IGARSSDownloaderPlugin()
        assert plugin.conference_name == "IGARSS"

    def test_supported_years_range(self):
        plugin = IGARSSDownloaderPlugin()
        # _start_year is 1994 so years from 1994 onward should be included
        assert 1994 in plugin.supported_years
        assert 2024 in plugin.supported_years
        assert 1993 not in plugin.supported_years

    def test_year_2005_excluded(self):
        """Year 2005 is excluded because IGARSS 2005 records lack abstracts."""
        plugin = IGARSSDownloaderPlugin()
        assert 2005 not in plugin.supported_years

    def test_validate_year_2005_raises(self):
        """validate_year should reject 2005 because it is excluded."""
        plugin = IGARSSDownloaderPlugin()
        with pytest.raises(ValueError, match="not supported"):
            plugin.validate_year(2005)

    def test_get_metadata(self):
        plugin = IGARSSDownloaderPlugin()
        metadata = plugin.get_metadata()
        assert metadata["name"] == "igarss"
        assert "IGARSS" in metadata["description"]
        assert metadata["conference_name"] == "IGARSS"
        assert "year" in metadata["parameters"]
        assert "output_path" in metadata["parameters"]
        assert "force_download" in metadata["parameters"]
        assert "timeout" in metadata["parameters"]
        assert "page_delay" in metadata["parameters"]

    def test_initialization_defaults(self):
        plugin = IGARSSDownloaderPlugin()
        assert plugin.timeout == 30
        assert plugin.verify_ssl is True

    def test_initialization_with_custom_params(self):
        plugin = IGARSSDownloaderPlugin(timeout=60, verify_ssl=False)
        assert plugin.timeout == 60
        assert plugin.verify_ssl is False


# ---------------------------------------------------------------------------
# URL construction tests
# ---------------------------------------------------------------------------


class TestIGARSSPluginURL:
    """Test URL construction."""

    def test_get_url_2025(self):
        plugin = IGARSSDownloaderPlugin()
        url = plugin.get_url(2025)
        assert "ieeexplore.ieee.org" in url
        assert "igarss" in url.lower()
        assert "2025" in url

    def test_get_url_2024(self):
        plugin = IGARSSDownloaderPlugin()
        url = plugin.get_url(2024)
        assert "2024" in url


# ---------------------------------------------------------------------------
# Year validation tests
# ---------------------------------------------------------------------------


class TestIGARSSPluginYearValidation:
    """Test year validation."""

    def test_validate_supported_year(self):
        plugin = IGARSSDownloaderPlugin()
        plugin.validate_year(2024)  # should not raise

    def test_validate_unsupported_year(self):
        plugin = IGARSSDownloaderPlugin()
        with pytest.raises(ValueError, match="Year 1990 not supported"):
            plugin.validate_year(1990)


# ---------------------------------------------------------------------------
# Search payload tests
# ---------------------------------------------------------------------------


class TestIGARSSBuildSearchPayload:
    """Test the search payload construction."""

    def test_basic_payload(self):
        plugin = IGARSSDownloaderPlugin()
        payload = plugin._build_search_payload(2025)
        assert payload["queryText"] == '("Publication Title":igarss 2025)'
        assert payload["newsearch"] is True
        assert payload["matchPubs"] is True
        assert payload["rowsPerPage"] == plugin._ROWS_PER_PAGE
        assert payload["pageNumber"] == 1

    def test_pagination(self):
        plugin = IGARSSDownloaderPlugin()
        payload = plugin._build_search_payload(2025, page_number=3)
        assert payload["pageNumber"] == 3


# ---------------------------------------------------------------------------
# _convert_paper tests
# ---------------------------------------------------------------------------


class TestIGARSSConvertPaper:
    """Test the internal _convert_paper helper."""

    def test_basic_conversion(self):
        plugin = IGARSSDownloaderPlugin()
        paper = plugin._convert_paper(SAMPLE_RECORD_1, 2025)

        assert paper is not None
        assert paper.title == "Satellite Image Classification Using Deep Learning"
        assert paper.abstract == "This paper presents a novel approach for satellite image classification."
        assert paper.year == 2025
        assert paper.conference == "IGARSS"
        assert "Alice Smith" in paper.authors
        assert "Bob Jones" in paper.authors

    def test_doi_converted_to_url(self):
        plugin = IGARSSDownloaderPlugin()
        paper = plugin._convert_paper(SAMPLE_RECORD_1, 2025)
        assert paper is not None
        assert paper.url == "https://doi.org/10.1109/IGARSS52108.2025.0000001"

    def test_pdf_url_mapped(self):
        plugin = IGARSSDownloaderPlugin()
        paper = plugin._convert_paper(SAMPLE_RECORD_2, 2025)
        assert paper is not None
        assert paper.paper_pdf_url == "https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9876544"

    def test_keywords_extracted(self):
        plugin = IGARSSDownloaderPlugin()
        paper = plugin._convert_paper(SAMPLE_RECORD_1, 2025)
        assert paper is not None
        assert paper.keywords is not None
        assert "Remote sensing" in paper.keywords
        assert "Deep learning" in paper.keywords
        assert "SAR" in paper.keywords
        assert "Image classification" in paper.keywords

    def test_original_id_from_article_number(self):
        plugin = IGARSSDownloaderPlugin()
        paper = plugin._convert_paper(SAMPLE_RECORD_1, 2025)
        assert paper is not None
        assert paper.original_id == 9876543

    def test_page_range_as_poster_position(self):
        plugin = IGARSSDownloaderPlugin()
        paper = plugin._convert_paper(SAMPLE_RECORD_1, 2025)
        assert paper is not None
        assert paper.poster_position == "pp. 1234-1237"

    def test_session_from_publication_title(self):
        plugin = IGARSSDownloaderPlugin()
        paper = plugin._convert_paper(SAMPLE_RECORD_1, 2025)
        assert paper is not None
        assert "IGARSS" in paper.session

    def test_highlight_tags_stripped(self):
        plugin = IGARSSDownloaderPlugin()
        paper = plugin._convert_paper(SAMPLE_RECORD_HIGHLIGHTED, 2025)
        assert paper is not None
        assert "<highlight>" not in paper.title
        assert "</highlight>" not in paper.title
        assert paper.title == "IGARSS Study on Urban Areas"
        assert "<highlight>" not in paper.abstract

    def test_missing_title_returns_none(self):
        plugin = IGARSSDownloaderPlugin()
        record = dict(SAMPLE_RECORD_1)
        record["articleTitle"] = ""
        assert plugin._convert_paper(record, 2025) is None

    def test_missing_authors_returns_none(self):
        plugin = IGARSSDownloaderPlugin()
        record = dict(SAMPLE_RECORD_1)
        record["authors"] = []
        assert plugin._convert_paper(record, 2025) is None

    def test_no_doi(self):
        plugin = IGARSSDownloaderPlugin()
        record = dict(SAMPLE_RECORD_1)
        del record["doi"]
        paper = plugin._convert_paper(record, 2025)
        assert paper is not None
        assert paper.url is None

    def test_no_index_terms(self):
        plugin = IGARSSDownloaderPlugin()
        record = dict(SAMPLE_RECORD_1)
        del record["indexTerms"]
        paper = plugin._convert_paper(record, 2025)
        assert paper is not None
        assert paper.keywords is None

    def test_authors_as_strings(self):
        """Test handling of authors given as plain strings."""
        plugin = IGARSSDownloaderPlugin()
        record = dict(SAMPLE_RECORD_1)
        record["authors"] = ["Alice Smith", "Bob Jones"]
        paper = plugin._convert_paper(record, 2025)
        assert paper is not None
        assert "Alice Smith" in paper.authors
        assert "Bob Jones" in paper.authors

    def test_fallback_author_name_fields(self):
        """Test fallback to authorName when preferredName is missing."""
        plugin = IGARSSDownloaderPlugin()
        record = dict(SAMPLE_RECORD_1)
        record["authors"] = [{"authorName": "Smith, Alice"}]
        paper = plugin._convert_paper(record, 2025)
        assert paper is not None
        assert "Smith, Alice" in paper.authors

    def test_no_pages_uses_article_number(self):
        """When start/end pages are missing, poster_position falls back to article number."""
        plugin = IGARSSDownloaderPlugin()
        record = dict(SAMPLE_RECORD_1)
        del record["startPage"]
        del record["endPage"]
        paper = plugin._convert_paper(record, 2025)
        assert paper is not None
        assert paper.poster_position == "9876543"


# ---------------------------------------------------------------------------
# _extract_keywords tests
# ---------------------------------------------------------------------------


class TestIGARSSExtractKeywords:
    """Test keyword extraction from IEEE Xplore records."""

    def test_extracts_ieee_and_author_terms(self):
        plugin = IGARSSDownloaderPlugin()
        keywords = plugin._extract_keywords(SAMPLE_RECORD_1)
        assert keywords is not None
        assert "Remote sensing" in keywords
        assert "SAR" in keywords

    def test_no_duplicate_keywords(self):
        plugin = IGARSSDownloaderPlugin()
        record = {
            "indexTerms": {
                "ieee_terms": {"terms": ["Remote sensing"]},
                "author_terms": {"terms": ["Remote sensing", "SAR"]},
            }
        }
        keywords = plugin._extract_keywords(record)
        assert keywords is not None
        assert keywords.count("Remote sensing") == 1

    def test_empty_index_terms_returns_none(self):
        plugin = IGARSSDownloaderPlugin()
        record = {"indexTerms": {}}
        assert plugin._extract_keywords(record) is None

    def test_no_index_terms_returns_none(self):
        plugin = IGARSSDownloaderPlugin()
        assert plugin._extract_keywords({}) is None


# ---------------------------------------------------------------------------
# _fetch_page tests
# ---------------------------------------------------------------------------


class TestIGARSSFetchPage:
    """Test the _fetch_page method."""

    def test_fetch_page_success(self):
        plugin = IGARSSDownloaderPlugin()
        mock_response = _make_search_response([SAMPLE_RECORD_1])

        with patch.object(plugin._session, "post", return_value=mock_response) as mock_post:
            result = plugin._fetch_page(2025, 1, 30, True)

        mock_post.assert_called_once()
        assert result["totalRecords"] == 1
        assert len(result["records"]) == 1

    def test_fetch_page_request_exception(self):
        plugin = IGARSSDownloaderPlugin()
        with patch.object(
            plugin._session,
            "post",
            side_effect=requests.exceptions.RequestException("Connection error"),
        ):
            with pytest.raises(RuntimeError, match="Failed to fetch IGARSS"):
                plugin._fetch_page(2025, 1, 30, True)

    def test_fetch_page_invalid_json(self):
        plugin = IGARSSDownloaderPlugin()
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)

        with patch.object(plugin._session, "post", return_value=mock_response):
            with pytest.raises(RuntimeError, match="Invalid JSON response"):
                plugin._fetch_page(2025, 1, 30, True)


# ---------------------------------------------------------------------------
# Download method tests
# ---------------------------------------------------------------------------


class TestIGARSSDownload:
    """Test the download() method."""

    @patch.object(IGARSSDownloaderPlugin, "_fetch_page")
    def test_download_success(self, mock_fetch):
        mock_fetch.return_value = {
            "totalRecords": 2,
            "records": [SAMPLE_RECORD_1, SAMPLE_RECORD_2],
        }

        plugin = IGARSSDownloaderPlugin()
        papers = plugin.download(year=2025)

        mock_fetch.assert_called_once_with(2025, 1, 30, True)

        assert isinstance(papers, list)
        assert len(papers) == 2
        for p in papers:
            assert p.year == 2025
            assert p.conference == "IGARSS"

    @patch.object(IGARSSDownloaderPlugin, "_fetch_page")
    def test_download_default_year(self, mock_fetch):
        mock_fetch.return_value = {
            "totalRecords": 1,
            "records": [SAMPLE_RECORD_1],
        }

        plugin = IGARSSDownloaderPlugin()
        # Pin supported years so the test is independent of the current date
        plugin.supported_years = [2024, 2025]
        papers = plugin.download()  # No year specified

        expected_year = max(plugin.supported_years)
        assert len(papers) == 1
        assert papers[0].year == expected_year

    @patch.object(IGARSSDownloaderPlugin, "_fetch_page")
    def test_download_saves_to_file(self, mock_fetch):
        mock_fetch.return_value = {
            "totalRecords": 1,
            "records": [SAMPLE_RECORD_1],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "igarss_2025.json"

            plugin = IGARSSDownloaderPlugin()
            plugin.download(year=2025, output_path=str(output_path))

            assert output_path.exists()
            with open(output_path, "r") as f:
                saved = json.load(f)

            assert isinstance(saved, list)
            assert len(saved) == 1
            assert saved[0]["year"] == 2025
            assert saved[0]["conference"] == "IGARSS"

    def test_download_loads_from_existing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "igarss_2025.json"

            cached = [
                {
                    "title": "Cached IGARSS Paper",
                    "abstract": "Cached abstract about remote sensing",
                    "authors": ["Cached Author"],
                    "session": "IGARSS Session",
                    "poster_position": "",
                    "year": 2025,
                    "conference": "IGARSS",
                }
            ]
            with open(output_path, "w") as f:
                json.dump(cached, f)

            plugin = IGARSSDownloaderPlugin()
            with patch.object(plugin, "_fetch_page") as mock_fetch:
                papers = plugin.download(year=2025, output_path=str(output_path))
                mock_fetch.assert_not_called()

            assert len(papers) == 1
            assert papers[0].title == "Cached IGARSS Paper"

    @patch.object(IGARSSDownloaderPlugin, "_fetch_page")
    def test_download_force_redownload(self, mock_fetch):
        mock_fetch.return_value = {
            "totalRecords": 1,
            "records": [SAMPLE_RECORD_2],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "igarss_2025.json"
            # Write stale file
            with open(output_path, "w") as f:
                json.dump([], f)

            plugin = IGARSSDownloaderPlugin()
            papers = plugin.download(year=2025, output_path=str(output_path), force_download=True)

            mock_fetch.assert_called_once()
            assert len(papers) == 1
            assert papers[0].title == "Radar Remote Sensing for Forest Monitoring"

    @patch.object(IGARSSDownloaderPlugin, "_fetch_page")
    def test_download_pagination(self, mock_fetch):
        """Test that multiple pages are fetched when total records exceed page size."""
        plugin = IGARSSDownloaderPlugin()

        # First page returns records with total indicating more pages
        page1_response = {
            "totalRecords": plugin._ROWS_PER_PAGE + 1,
            "records": [SAMPLE_RECORD_1] * plugin._ROWS_PER_PAGE,
        }
        page2_response = {
            "totalRecords": plugin._ROWS_PER_PAGE + 1,
            "records": [SAMPLE_RECORD_2],
        }

        mock_fetch.side_effect = [page1_response, page2_response]

        papers = plugin.download(year=2025, page_delay=0)

        assert mock_fetch.call_count == 2
        # First call is page 1, second call is page 2
        assert mock_fetch.call_args_list[0] == call(2025, 1, 30, True)
        assert mock_fetch.call_args_list[1] == call(2025, 2, 30, True)

        assert len(papers) == plugin._ROWS_PER_PAGE + 1

    @patch.object(IGARSSDownloaderPlugin, "_fetch_page")
    def test_download_pagination_stops_on_empty_page(self, mock_fetch):
        """Test that pagination stops when a page returns no records."""
        plugin = IGARSSDownloaderPlugin()

        page1_response = {
            "totalRecords": 200,
            "records": [SAMPLE_RECORD_1],
        }
        page2_response = {
            "totalRecords": 200,
            "records": [],
        }

        mock_fetch.side_effect = [page1_response, page2_response]

        papers = plugin.download(year=2025, page_delay=0)
        assert mock_fetch.call_count == 2
        assert len(papers) == 1

    @patch.object(IGARSSDownloaderPlugin, "_fetch_page")
    def test_download_request_exception(self, mock_fetch):
        mock_fetch.side_effect = RuntimeError("Failed to fetch IGARSS")

        plugin = IGARSSDownloaderPlugin()
        with pytest.raises(RuntimeError, match="Failed to fetch IGARSS"):
            plugin.download(year=2025)

    @patch.object(IGARSSDownloaderPlugin, "_fetch_page")
    def test_download_skips_papers_without_title(self, mock_fetch):
        no_title = dict(SAMPLE_RECORD_1)
        no_title["articleTitle"] = ""
        mock_fetch.return_value = {
            "totalRecords": 2,
            "records": [no_title, SAMPLE_RECORD_2],
        }

        plugin = IGARSSDownloaderPlugin()
        papers = plugin.download(year=2025)
        assert len(papers) == 1
        assert papers[0].title == "Radar Remote Sensing for Forest Monitoring"

    @patch.object(IGARSSDownloaderPlugin, "_fetch_page")
    def test_download_custom_timeout(self, mock_fetch):
        mock_fetch.return_value = {"totalRecords": 0, "records": []}

        plugin = IGARSSDownloaderPlugin(timeout=60)
        plugin.download(year=2025)

        assert mock_fetch.call_args[0][2] == 60  # timeout arg

    @patch.object(IGARSSDownloaderPlugin, "_fetch_page")
    def test_download_kwargs_override(self, mock_fetch):
        mock_fetch.return_value = {"totalRecords": 0, "records": []}

        plugin = IGARSSDownloaderPlugin(timeout=30, verify_ssl=True)
        plugin.download(year=2025, timeout=90, verify_ssl=False)

        assert mock_fetch.call_args[0][2] == 90  # timeout
        assert mock_fetch.call_args[0][3] is False  # verify_ssl

    @patch.object(IGARSSDownloaderPlugin, "_fetch_page")
    def test_download_no_records(self, mock_fetch):
        mock_fetch.return_value = {"totalRecords": 0, "records": []}

        plugin = IGARSSDownloaderPlugin()
        papers = plugin.download(year=2025)
        assert papers == []


# ---------------------------------------------------------------------------
# Database integration tests
# ---------------------------------------------------------------------------


class TestIGARSSPluginDatabaseIntegration:
    """Test that IGARSS plugin data can be stored in the database."""

    @patch.object(IGARSSDownloaderPlugin, "_fetch_page")
    def test_igarss_data_in_database(self, mock_fetch):
        mock_fetch.return_value = {
            "totalRecords": 2,
            "records": [SAMPLE_RECORD_1, SAMPLE_RECORD_2],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_igarss.db"

            plugin = IGARSSDownloaderPlugin()
            papers = plugin.download(year=2025)

            set_test_db(db_path)
            with DatabaseManager() as db:
                db.create_tables()
                db.add_papers(papers)

                rows = db.query("SELECT uid, title, abstract, year, conference, authors FROM papers")
                assert len(rows) == 2

                titles = {r["title"] for r in rows}
                assert "Satellite Image Classification Using Deep Learning" in titles
                assert "Radar Remote Sensing for Forest Monitoring" in titles

                for row in rows:
                    assert row["year"] == 2025
                    assert row["conference"] == "IGARSS"


# ---------------------------------------------------------------------------
# Plugin registration tests
# ---------------------------------------------------------------------------


class TestIGARSSPluginRegistration:
    """Test plugin auto-registration."""

    def test_plugin_auto_registers(self):
        from abstracts_explorer.plugins import get_plugin

        plugin = get_plugin("igarss")
        assert plugin is not None
        assert isinstance(plugin, IGARSSDownloaderPlugin)

    def test_plugin_in_list(self):
        from abstracts_explorer.plugins import list_plugin_names

        assert "igarss" in list_plugin_names()

"""
Tests for IEEE VIS Downloader Plugin
======================================

Test suite for the IEEE VIS conference data downloader plugin.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
import requests

from abstracts_explorer.plugins.ieeevis_downloader import IEEEVISDownloaderPlugin
from abstracts_explorer.database import DatabaseManager
from tests.conftest import set_test_db

# ---------------------------------------------------------------------------
# Sample data helpers
# ---------------------------------------------------------------------------

SAMPLE_PAPER_1 = {
    "UID": "013c31af-6df4-4dea-b01c-3f4cd7af88df",
    "title": "Test VIS Paper One",
    "abstract": "Abstract for paper one.",
    "authors": [
        {"email": None, "name": "Alice Smith"},
        {"email": None, "name": "Bob Jones"},
    ],
    "keywords": ["Visualization", "Deep learning"],
    "session_title": "Visual Analytics Methods",
    "session_room": "Hall E1",
    "event_title": "VIS Full Papers",
    "paper_type": "full",
    "doi": "10.1109/TVCG.2025.0000001",
    "pdf_url": None,
    "award": "",
    "time_stamp": "2025-11-06T11:03:00.000Z",
}

SAMPLE_PAPER_2 = {
    "UID": "02afa2f8-af68-4eb3-b1e6-9f20791ee73d",
    "title": "Test VIS Paper Two",
    "abstract": "Abstract for paper two.",
    "authors": [
        {"email": None, "name": "Charlie Brown"},
    ],
    "keywords": ["Information visualization"],
    "session_title": "Graph Visualization",
    "session_room": "Room 101",
    "event_title": "VIS Full Papers",
    "paper_type": "full",
    "doi": "10.1109/TVCG.2025.0000002",
    "pdf_url": "https://example.com/paper2.pdf",
    "award": "Best Paper Award",
    "time_stamp": "2025-11-06T14:00:00.000Z",
}


def _make_mock_response(papers):
    """Return a Mock that behaves like a successful requests.Response with a list payload."""
    mock_response = Mock()
    mock_response.json.return_value = papers
    return mock_response


# ---------------------------------------------------------------------------
# Plugin metadata tests
# ---------------------------------------------------------------------------


class TestIEEEVISPluginMetadata:
    """Test plugin metadata and initialization."""

    def test_plugin_name(self):
        plugin = IEEEVISDownloaderPlugin()
        assert plugin.plugin_name == "ieeevis"

    def test_plugin_description(self):
        plugin = IEEEVISDownloaderPlugin()
        assert "IEEE VIS" in plugin.plugin_description

    def test_conference_name(self):
        plugin = IEEEVISDownloaderPlugin()
        assert plugin.conference_name == "IEEE VIS"

    def test_supported_years_contains_2025(self):
        plugin = IEEEVISDownloaderPlugin()
        assert 2025 in plugin.supported_years

    def test_get_metadata(self):
        plugin = IEEEVISDownloaderPlugin()
        metadata = plugin.get_metadata()
        assert metadata["name"] == "ieeevis"
        assert "IEEE VIS" in metadata["description"]
        assert "year" in metadata["parameters"]
        assert "output_path" in metadata["parameters"]
        assert "force_download" in metadata["parameters"]

    def test_initialization_with_custom_params(self):
        plugin = IEEEVISDownloaderPlugin(timeout=60, verify_ssl=False)
        assert plugin.timeout == 60
        assert plugin.verify_ssl is False


# ---------------------------------------------------------------------------
# URL construction tests
# ---------------------------------------------------------------------------


class TestIEEEVISPluginURL:
    """Test URL construction."""

    def test_get_url_2025(self):
        plugin = IEEEVISDownloaderPlugin()
        url = plugin.get_url(2025)
        assert url == "https://ieeevis.org/year/2025/program/papers.json"

    def test_get_url_other_year(self):
        plugin = IEEEVISDownloaderPlugin()
        url = plugin.get_url(2024)
        assert url == "https://ieeevis.org/year/2024/program/papers.json"


# ---------------------------------------------------------------------------
# Year validation tests
# ---------------------------------------------------------------------------


class TestIEEEVISPluginYearValidation:
    """Test year validation."""

    def test_validate_supported_year(self):
        plugin = IEEEVISDownloaderPlugin()
        plugin.validate_year(2025)  # should not raise

    def test_validate_unsupported_year(self):
        plugin = IEEEVISDownloaderPlugin()
        with pytest.raises(ValueError, match="Year 1800 not supported"):
            plugin.validate_year(1800)


# ---------------------------------------------------------------------------
# _convert_paper tests
# ---------------------------------------------------------------------------


class TestIEEEVISConvertPaper:
    """Test the internal _convert_paper helper."""

    def test_basic_conversion(self):
        plugin = IEEEVISDownloaderPlugin()
        paper = plugin._convert_paper(SAMPLE_PAPER_1, 2025)

        assert paper is not None
        assert paper.title == "Test VIS Paper One"
        assert paper.abstract == "Abstract for paper one."
        assert paper.year == 2025
        assert paper.conference == "IEEE VIS"
        assert "Alice Smith" in paper.authors
        assert "Bob Jones" in paper.authors

    def test_session_from_session_title(self):
        plugin = IEEEVISDownloaderPlugin()
        paper = plugin._convert_paper(SAMPLE_PAPER_1, 2025)
        assert paper is not None
        assert paper.session == "Visual Analytics Methods"

    def test_session_falls_back_to_event_title(self):
        plugin = IEEEVISDownloaderPlugin()
        item = dict(SAMPLE_PAPER_1)
        del item["session_title"]
        paper = plugin._convert_paper(item, 2025)
        assert paper is not None
        assert paper.session == "VIS Full Papers"

    def test_session_falls_back_to_no_session(self):
        plugin = IEEEVISDownloaderPlugin()
        item = {k: v for k, v in SAMPLE_PAPER_1.items() if k not in ("session_title", "event_title")}
        paper = plugin._convert_paper(item, 2025)
        assert paper is not None
        assert paper.session == "No session"

    def test_room_name_mapped(self):
        plugin = IEEEVISDownloaderPlugin()
        paper = plugin._convert_paper(SAMPLE_PAPER_1, 2025)
        assert paper is not None
        assert paper.room_name == "Hall E1"

    def test_doi_converted_to_url(self):
        plugin = IEEEVISDownloaderPlugin()
        paper = plugin._convert_paper(SAMPLE_PAPER_1, 2025)
        assert paper is not None
        assert paper.url == "https://doi.org/10.1109/TVCG.2025.0000001"

    def test_pdf_url_mapped(self):
        plugin = IEEEVISDownloaderPlugin()
        paper = plugin._convert_paper(SAMPLE_PAPER_2, 2025)
        assert paper is not None
        assert paper.paper_pdf_url == "https://example.com/paper2.pdf"

    def test_award_mapped(self):
        plugin = IEEEVISDownloaderPlugin()
        paper = plugin._convert_paper(SAMPLE_PAPER_2, 2025)
        assert paper is not None
        assert paper.award == "Best Paper Award"

    def test_empty_award_not_stored(self):
        plugin = IEEEVISDownloaderPlugin()
        paper = plugin._convert_paper(SAMPLE_PAPER_1, 2025)
        assert paper is not None
        assert paper.award is None

    def test_starttime_mapped(self):
        plugin = IEEEVISDownloaderPlugin()
        paper = plugin._convert_paper(SAMPLE_PAPER_1, 2025)
        assert paper is not None
        assert paper.starttime == "2025-11-06T11:03:00.000Z"

    def test_keywords_mapped(self):
        plugin = IEEEVISDownloaderPlugin()
        paper = plugin._convert_paper(SAMPLE_PAPER_1, 2025)
        assert paper is not None
        assert paper.keywords == ["Visualization", "Deep learning"]

    def test_missing_title_returns_none(self):
        plugin = IEEEVISDownloaderPlugin()
        item = dict(SAMPLE_PAPER_1)
        item["title"] = ""
        assert plugin._convert_paper(item, 2025) is None

    def test_missing_authors_returns_none(self):
        plugin = IEEEVISDownloaderPlugin()
        item = dict(SAMPLE_PAPER_1)
        item["authors"] = []
        assert plugin._convert_paper(item, 2025) is None

    def test_poster_position_is_empty_string(self):
        plugin = IEEEVISDownloaderPlugin()
        paper = plugin._convert_paper(SAMPLE_PAPER_1, 2025)
        assert paper is not None
        assert paper.poster_position == ""


# ---------------------------------------------------------------------------
# Download method tests
# ---------------------------------------------------------------------------


class TestIEEEVISDownload:
    """Test the download() method."""

    @patch("abstracts_explorer.plugins.ieeevis_downloader.requests.get")
    def test_download_success(self, mock_get):
        mock_get.return_value = _make_mock_response([SAMPLE_PAPER_1, SAMPLE_PAPER_2])

        plugin = IEEEVISDownloaderPlugin()
        papers = plugin.download(year=2025)

        mock_get.assert_called_once()
        assert mock_get.call_args[0][0] == "https://ieeevis.org/year/2025/program/papers.json"

        assert isinstance(papers, list)
        assert len(papers) == 2
        for p in papers:
            assert p.year == 2025
            assert p.conference == "IEEE VIS"

    @patch("abstracts_explorer.plugins.ieeevis_downloader.requests.get")
    def test_download_default_year(self, mock_get):
        mock_get.return_value = _make_mock_response([SAMPLE_PAPER_1])

        plugin = IEEEVISDownloaderPlugin()
        papers = plugin.download()  # No year specified

        assert "2025" in mock_get.call_args[0][0]
        assert papers[0].year == 2025

    @patch("abstracts_explorer.plugins.ieeevis_downloader.requests.get")
    def test_download_saves_to_file(self, mock_get):
        mock_get.return_value = _make_mock_response([SAMPLE_PAPER_1])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "ieeevis_2025.json"

            plugin = IEEEVISDownloaderPlugin()
            plugin.download(year=2025, output_path=str(output_path))

            assert output_path.exists()
            with open(output_path, "r") as f:
                saved = json.load(f)

            assert isinstance(saved, list)
            assert len(saved) == 1
            assert saved[0]["year"] == 2025
            assert saved[0]["conference"] == "IEEE VIS"

    def test_download_loads_from_existing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "ieeevis_2025.json"

            cached = [
                {
                    "title": "Cached Paper",
                    "abstract": "Cached abstract",
                    "authors": ["Cached Author"],
                    "session": "Cached Session",
                    "poster_position": "",
                    "year": 2025,
                    "conference": "IEEE VIS",
                }
            ]
            with open(output_path, "w") as f:
                json.dump(cached, f)

            plugin = IEEEVISDownloaderPlugin()
            with patch("abstracts_explorer.plugins.ieeevis_downloader.requests.get") as mock_get:
                papers = plugin.download(year=2025, output_path=str(output_path))
                mock_get.assert_not_called()

            assert len(papers) == 1
            assert papers[0].title == "Cached Paper"

    @patch("abstracts_explorer.plugins.ieeevis_downloader.requests.get")
    def test_download_force_redownload(self, mock_get):
        mock_get.return_value = _make_mock_response([SAMPLE_PAPER_2])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "ieeevis_2025.json"
            # Write stale file
            with open(output_path, "w") as f:
                json.dump([], f)

            plugin = IEEEVISDownloaderPlugin()
            papers = plugin.download(year=2025, output_path=str(output_path), force_download=True)

            mock_get.assert_called_once()
            assert len(papers) == 1
            assert papers[0].title == "Test VIS Paper Two"

    @patch("abstracts_explorer.plugins.ieeevis_downloader.requests.get")
    def test_download_request_exception(self, mock_get):
        mock_get.side_effect = requests.exceptions.RequestException("Connection error")

        plugin = IEEEVISDownloaderPlugin()
        with pytest.raises(RuntimeError, match="Failed to download"):
            plugin.download(year=2025)

    @patch("abstracts_explorer.plugins.ieeevis_downloader.requests.get")
    def test_download_invalid_json(self, mock_get):
        mock_response = Mock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_get.return_value = mock_response

        plugin = IEEEVISDownloaderPlugin()
        with pytest.raises(RuntimeError, match="Invalid JSON response"):
            plugin.download(year=2025)

    @patch("abstracts_explorer.plugins.ieeevis_downloader.requests.get")
    def test_download_unexpected_json_structure(self, mock_get):
        """When the API returns a dict instead of a list, return an empty list gracefully."""
        mock_response = Mock()
        mock_response.json.return_value = {"error": "not found"}
        mock_get.return_value = mock_response

        plugin = IEEEVISDownloaderPlugin()
        papers = plugin.download(year=2025)
        assert papers == []

    @patch("abstracts_explorer.plugins.ieeevis_downloader.requests.get")
    def test_download_skips_papers_without_title(self, mock_get):
        no_title = dict(SAMPLE_PAPER_1)
        no_title["title"] = ""
        mock_get.return_value = _make_mock_response([no_title, SAMPLE_PAPER_2])

        plugin = IEEEVISDownloaderPlugin()
        papers = plugin.download(year=2025)

        assert len(papers) == 1
        assert papers[0].title == "Test VIS Paper Two"

    @patch("abstracts_explorer.plugins.ieeevis_downloader.requests.get")
    def test_download_custom_timeout(self, mock_get):
        mock_get.return_value = _make_mock_response([])

        plugin = IEEEVISDownloaderPlugin(timeout=60)
        plugin.download(year=2025)

        assert mock_get.call_args[1]["timeout"] == 60

    @patch("abstracts_explorer.plugins.ieeevis_downloader.requests.get")
    def test_download_kwargs_override(self, mock_get):
        mock_get.return_value = _make_mock_response([])

        plugin = IEEEVISDownloaderPlugin(timeout=30, verify_ssl=True)
        plugin.download(year=2025, timeout=90, verify_ssl=False)

        assert mock_get.call_args[1]["timeout"] == 90
        assert mock_get.call_args[1]["verify"] is False


# ---------------------------------------------------------------------------
# Database integration tests
# ---------------------------------------------------------------------------


class TestIEEEVISPluginDatabaseIntegration:
    """Test that IEEE VIS plugin data can be stored in the database."""

    @patch("abstracts_explorer.plugins.ieeevis_downloader.requests.get")
    def test_ieeevis_data_in_database(self, mock_get):
        mock_get.return_value = _make_mock_response([SAMPLE_PAPER_1, SAMPLE_PAPER_2])

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_ieeevis.db"

            plugin = IEEEVISDownloaderPlugin()
            papers = plugin.download(year=2025)

            set_test_db(db_path)
            with DatabaseManager() as db:
                db.create_tables()
                db.add_papers(papers)

                rows = db.query("SELECT uid, title, abstract, year, conference, authors FROM papers")
                assert len(rows) == 2

                titles = {r["title"] for r in rows}
                assert "Test VIS Paper One" in titles
                assert "Test VIS Paper Two" in titles

                for row in rows:
                    assert row["year"] == 2025
                    assert row["conference"] == "IEEE VIS"


# ---------------------------------------------------------------------------
# Plugin registration tests
# ---------------------------------------------------------------------------


class TestIEEEVISPluginRegistration:
    """Test plugin auto-registration."""

    def test_plugin_auto_registers(self):
        from abstracts_explorer.plugins import get_plugin

        plugin = get_plugin("ieeevis")
        assert plugin is not None
        assert isinstance(plugin, IEEEVISDownloaderPlugin)

    def test_plugin_in_list(self):
        from abstracts_explorer.plugins import list_plugin_names

        assert "ieeevis" in list_plugin_names()

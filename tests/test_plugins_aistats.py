"""
Tests for AISTATS Downloader Plugin
====================================

Test suite for the AISTATS conference data downloader plugin.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
import requests

from abstracts_explorer.plugins.aistats_downloader import AISTATSDownloaderPlugin
from abstracts_explorer.database import DatabaseManager
from tests.conftest import set_test_db


class TestAISTATSPlugin:
    """Test suite for AISTATS downloader plugin."""

    def test_plugin_metadata(self):
        """Test that plugin provides correct metadata."""
        plugin = AISTATSDownloaderPlugin()

        assert plugin.plugin_name == "aistats"
        assert plugin.plugin_description == "Official AISTATS conference data downloader"
        assert 2025 in plugin.supported_years

        metadata = plugin.get_metadata()
        assert metadata["name"] == "aistats"
        assert metadata["description"] == "Official AISTATS conference data downloader"
        assert "year" in metadata["parameters"]
        assert "output_path" in metadata["parameters"]
        assert "force_download" in metadata["parameters"]

    def test_plugin_initialization(self):
        """Test plugin initialization with custom parameters."""
        plugin = AISTATSDownloaderPlugin(timeout=60, verify_ssl=False)

        assert plugin.timeout == 60
        assert plugin.verify_ssl is False

    def test_validate_year_success(self):
        """Test year validation with supported year."""
        plugin = AISTATSDownloaderPlugin()

        # Should not raise exception
        plugin.validate_year(2025)

    def test_validate_year_failure(self):
        """Test year validation with unsupported year."""
        plugin = AISTATSDownloaderPlugin()

        with pytest.raises(ValueError, match="Year 1800 not supported"):
            plugin.validate_year(1800)

    def test_get_url(self):
        """Test URL generation for AISTATS data."""
        plugin = AISTATSDownloaderPlugin()

        assert (
            plugin.get_url(2021) == "https://virtual.aistats.org/static/virtual/data/aistats-2021-orals-posters.json"
        )
        assert (
            plugin.get_url(2024) == "https://virtual.aistats.org/static/virtual/data/aistats-2024-orals-posters.json"
        )
        assert (
            plugin.get_url(2025) == "https://virtual.aistats.org/static/virtual/data/aistats-2025-orals-posters.json"
        )

    @patch("abstracts_explorer.plugins.json_conference_downloader.requests.get")
    def test_download_success(self, mock_get):
        """Test successful download of AISTATS data."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "count": 2,
            "next": None,
            "previous": None,
            "results": [
                {
                    "id": 1,
                    "name": "Test Paper 1",
                    "abstract": "Abstract 1",
                    "authors": [{"fullname": "Author 1"}],
                },
                {
                    "id": 2,
                    "name": "Test Paper 2",
                    "abstract": "Abstract 2",
                    "authors": [{"fullname": "Author 2"}],
                },
            ],
        }
        mock_get.return_value = mock_response

        plugin = AISTATSDownloaderPlugin()
        data = plugin.download(year=2025)

        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args[0][0] == "https://virtual.aistats.org/static/virtual/data/aistats-2025-orals-posters.json"

        assert isinstance(data, list)
        assert len(data) == 2

        for paper in data:
            assert paper.year == 2025
            assert paper.conference == "AISTATS"

    @patch("abstracts_explorer.plugins.json_conference_downloader.requests.get")
    def test_download_with_default_year(self, mock_get):
        """Test download with default year (2025)."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "count": 1,
            "next": None,
            "previous": None,
            "results": [
                {
                    "id": 1,
                    "name": "Test Paper",
                    "abstract": "A non-empty test abstract.",
                    "authors": [{"fullname": "Test Author"}],
                }
            ],
        }
        mock_get.return_value = mock_response

        plugin = AISTATSDownloaderPlugin()
        data = plugin.download()

        call_args = mock_get.call_args
        assert "aistats-2025-orals-posters.json" in call_args[0][0]

        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0].year == 2025

    @patch("abstracts_explorer.plugins.json_conference_downloader.requests.get")
    def test_download_with_save_to_file(self, mock_get):
        """Test download with saving to file."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "count": 1,
            "next": None,
            "previous": None,
            "results": [
                {"id": 1, "name": "Test Paper", "abstract": "Test abstract", "authors": [{"fullname": "Test Author"}]}
            ],
        }
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "aistats_2025.json"

            plugin = AISTATSDownloaderPlugin()
            plugin.download(year=2025, output_path=str(output_path))

            assert output_path.exists()

            with open(output_path, "r") as f:
                saved_data = json.load(f)

            assert isinstance(saved_data, list)
            assert len(saved_data) == 1
            assert saved_data[0]["year"] == 2025
            assert saved_data[0]["conference"] == "AISTATS"

    def test_download_load_from_existing_file(self):
        """Test loading data from existing file without re-downloading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "aistats_2025.json"

            test_data = [
                {
                    "title": "Cached Paper",
                    "abstract": "Test abstract",
                    "authors": ["Test Author"],
                    "session": "Test Session",
                    "poster_position": "A1",
                    "year": 2025,
                    "conference": "AISTATS",
                }
            ]

            with open(output_path, "w") as f:
                json.dump(test_data, f)

            plugin = AISTATSDownloaderPlugin()
            with patch("abstracts_explorer.plugins.json_conference_downloader.requests.get") as mock_get:
                data = plugin.download(year=2025, output_path=str(output_path))

                mock_get.assert_not_called()

                assert isinstance(data, list)
                assert len(data) == 1
                assert data[0].title == "Cached Paper"

    @patch("abstracts_explorer.plugins.json_conference_downloader.requests.get")
    def test_download_force_redownload(self, mock_get):
        """Test force re-download even when file exists."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "count": 1,
            "next": None,
            "previous": None,
            "results": [
                {
                    "id": 1,
                    "name": "Fresh Paper",
                    "abstract": "A fresh test abstract.",
                    "authors": [{"fullname": "Test Author"}],
                }
            ],
        }
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "aistats_2025.json"

            with open(output_path, "w") as f:
                json.dump({"count": 1, "results": [{"id": 1, "name": "Old Paper"}]}, f)

            plugin = AISTATSDownloaderPlugin()
            data = plugin.download(year=2025, output_path=str(output_path), force_download=True)

            mock_get.assert_called_once()

            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0].title == "Fresh Paper"

    @patch("abstracts_explorer.plugins.json_conference_downloader.requests.get")
    def test_download_request_exception(self, mock_get):
        """Test handling of request exceptions."""
        mock_get.side_effect = requests.exceptions.RequestException("Connection error")

        plugin = AISTATSDownloaderPlugin()

        with pytest.raises(RuntimeError, match="Failed to download"):
            plugin.download(year=2025)

    @patch("abstracts_explorer.plugins.json_conference_downloader.requests.get")
    def test_download_invalid_json(self, mock_get):
        """Test handling of invalid JSON response."""
        mock_response = Mock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_get.return_value = mock_response

        plugin = AISTATSDownloaderPlugin()

        with pytest.raises(RuntimeError, match="Invalid JSON response"):
            plugin.download(year=2025)

    @patch("abstracts_explorer.plugins.json_conference_downloader.requests.get")
    def test_download_custom_timeout(self, mock_get):
        """Test download with custom timeout."""
        mock_response = Mock()
        mock_response.json.return_value = {"count": 0, "next": None, "previous": None, "results": []}
        mock_get.return_value = mock_response

        plugin = AISTATSDownloaderPlugin(timeout=60)
        plugin.download(year=2025)

        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["timeout"] == 60

    @patch("abstracts_explorer.plugins.json_conference_downloader.requests.get")
    def test_download_kwargs_override(self, mock_get):
        """Test that kwargs can override default timeout and verify_ssl."""
        mock_response = Mock()
        mock_response.json.return_value = {"count": 0, "next": None, "previous": None, "results": []}
        mock_get.return_value = mock_response

        plugin = AISTATSDownloaderPlugin(timeout=30, verify_ssl=True)
        plugin.download(year=2025, timeout=90, verify_ssl=False)

        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["timeout"] == 90
        assert call_kwargs["verify"] is False

    @patch("abstracts_explorer.plugins.json_conference_downloader.requests.get")
    def test_download_full_aistats_record(self, mock_get):
        """Test that optional AISTATS fields are correctly mapped."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "count": 1,
            "next": None,
            "previous": None,
            "results": [
                {
                    "id": 9908,
                    "name": "Test AISTATS Paper",
                    "abstract": "Test abstract text.",
                    "authors": [
                        {"id": 101, "fullname": "Alice Smith", "institution": "MIT"},
                        {"id": 102, "fullname": "Bob Jones", "institution": "Stanford"},
                    ],
                    "session": "Poster Session 3",
                    "poster_position": "36",
                    "room_name": "Hall A-E",
                    "keywords": ["bayesian", "optimization"],
                    "starttime": "2025-05-03T01:00:00-07:00",
                    "endtime": "2025-05-03T04:00:00-07:00",
                    "paper_pdf_url": "https://example.com/paper.pdf",
                    "paper_url": "https://openreview.net/forum?id=test",
                    "url": "https://virtual.aistats.org/virtual/2025/poster/9908",
                    "decision": "Accept (Best Paper)",
                }
            ],
        }
        mock_get.return_value = mock_response

        plugin = AISTATSDownloaderPlugin()
        data = plugin.download(year=2025)

        assert len(data) == 1
        paper = data[0]

        assert paper.title == "Test AISTATS Paper"
        assert paper.authors == ["Alice Smith", "Bob Jones"]
        assert paper.abstract == "Test abstract text."
        assert paper.session == "Poster Session 3"
        assert paper.poster_position == "36"
        assert paper.room_name == "Hall A-E"
        assert paper.keywords == ["bayesian", "optimization"]
        assert paper.starttime == "2025-05-03T01:00:00-07:00"
        assert paper.endtime == "2025-05-03T04:00:00-07:00"
        assert paper.original_id == 9908
        assert paper.conference == "AISTATS"
        assert paper.year == 2025


class TestAISTATSPluginDatabaseIntegration:
    """Test AISTATS plugin integration with database."""

    @patch("abstracts_explorer.plugins.json_conference_downloader.requests.get")
    def test_aistats_data_in_database(self, mock_get):
        """Test that AISTATS data can be stored in the database."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "count": 1,
            "next": None,
            "previous": None,
            "results": [
                {
                    "id": 1,
                    "name": "Test AISTATS Paper",
                    "abstract": "This is a test abstract for AISTATS",
                    "authors": [
                        {"id": 101, "fullname": "Alice Smith", "institution": "MIT"},
                        {"id": 102, "fullname": "Bob Jones", "institution": "Stanford"},
                    ],
                    "keywords": ["statistics", "machine learning"],
                    "decision": "Accept (Poster)",
                    "session": "Poster Session 1",
                }
            ],
        }
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_aistats.db"

            plugin = AISTATSDownloaderPlugin()
            data = plugin.download(year=2025)

            set_test_db(db_path)
            with DatabaseManager() as db:
                db.create_tables()
                db.add_papers(data)

                papers = db.query("SELECT uid, title, abstract, year, conference, authors FROM papers")
                assert len(papers) == 1

                paper = papers[0]
                assert paper["title"] == "Test AISTATS Paper"
                assert paper["abstract"] == "This is a test abstract for AISTATS"
                assert paper["year"] == 2025
                assert paper["conference"] == "AISTATS"

                authors_str = paper["authors"]
                assert "Alice Smith" in authors_str
                assert "Bob Jones" in authors_str
                assert ";" in authors_str


class TestAISTATSPluginRegistration:
    """Test AISTATS plugin registration."""

    def test_plugin_auto_registers(self):
        """Test that AISTATS plugin auto-registers on import."""
        from abstracts_explorer.plugins import get_plugin

        plugin = get_plugin("aistats")
        assert plugin is not None

        assert plugin.plugin_name == "aistats"

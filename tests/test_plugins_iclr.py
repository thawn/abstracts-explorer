"""
Tests for ICLR Downloader Plugin
=================================

Test suite for the ICLR conference data downloader plugin.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
import requests

from abstracts_explorer.plugins.iclr_downloader import ICLRDownloaderPlugin
from abstracts_explorer.database import DatabaseManager


class TestICLRPlugin:
    """Test suite for ICLR downloader plugin."""

    def test_plugin_metadata(self):
        """Test that plugin provides correct metadata."""
        plugin = ICLRDownloaderPlugin()

        assert plugin.plugin_name == "iclr"
        assert plugin.plugin_description == "Official ICLR conference data downloader"
        assert 2025 in plugin.supported_years

        metadata = plugin.get_metadata()
        assert metadata["name"] == "iclr"
        assert metadata["description"] == "Official ICLR conference data downloader"
        assert "year" in metadata["parameters"]
        assert "output_path" in metadata["parameters"]
        assert "force_download" in metadata["parameters"]

    def test_plugin_initialization(self):
        """Test plugin initialization with custom parameters."""
        plugin = ICLRDownloaderPlugin(timeout=60, verify_ssl=False)

        assert plugin.timeout == 60
        assert plugin.verify_ssl is False

    def test_validate_year_success(self):
        """Test year validation with supported year."""
        plugin = ICLRDownloaderPlugin()

        # Should not raise exception
        plugin.validate_year(2025)

    def test_validate_year_failure(self):
        """Test year validation with unsupported year."""
        plugin = ICLRDownloaderPlugin()

        with pytest.raises(ValueError, match="Year 1800 not supported"):
            plugin.validate_year(1800)

    @patch("abstracts_explorer.plugins.json_conference_downloader.requests.get")
    def test_download_success(self, mock_get):
        """Test successful download of ICLR data."""
        # Mock the requests.get response
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

        plugin = ICLRDownloaderPlugin()
        data = plugin.download(year=2025)

        # Verify the request was made to the correct URL
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args[0][0] == "https://iclr.cc/static/virtual/data/iclr-2025-orals-posters.json"

        # Verify the data structure - now returns List[LightweightPaper]
        assert isinstance(data, list)
        assert len(data) == 2

        # Verify year and conference were added
        for paper in data:
            assert paper.year == 2025
            assert paper.conference == "ICLR"

    @patch("abstracts_explorer.plugins.json_conference_downloader.requests.get")
    def test_download_with_default_year(self, mock_get):
        """Test download with default year (2025)."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "count": 1,
            "next": None,
            "previous": None,
            "results": [{"id": 1, "name": "Test Paper", "authors": [{"fullname": "Test Author"}]}],
        }
        mock_get.return_value = mock_response

        plugin = ICLRDownloaderPlugin()
        data = plugin.download()  # No year specified

        # Should default to 2025
        call_args = mock_get.call_args
        assert "iclr-2025-orals-posters.json" in call_args[0][0]

        # Verify year was set to 2025 - returns List[LightweightPaper]
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
            output_path = Path(tmpdir) / "iclr_2025.json"

            plugin = ICLRDownloaderPlugin()
            plugin.download(year=2025, output_path=str(output_path))

            # Verify file was created
            assert output_path.exists()

            # Verify file contents - now saves as list of paper dicts
            with open(output_path, "r") as f:
                saved_data = json.load(f)

            assert isinstance(saved_data, list)
            assert len(saved_data) == 1
            assert saved_data[0]["year"] == 2025
            assert saved_data[0]["conference"] == "ICLR"

    def test_download_load_from_existing_file(self):
        """Test loading data from existing file without re-downloading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "iclr_2025.json"

            # Create a mock file with lightweight format (list of paper dicts)
            test_data = [
                {
                    "title": "Cached Paper",
                    "abstract": "Test abstract",
                    "authors": ["Test Author"],
                    "session": "Test Session",
                    "poster_position": "A1",  # Required field
                    "year": 2025,
                    "conference": "ICLR",
                }
            ]

            with open(output_path, "w") as f:
                json.dump(test_data, f)

            # Download without force_download should load from file
            plugin = ICLRDownloaderPlugin()
            with patch("abstracts_explorer.plugins.json_conference_downloader.requests.get") as mock_get:
                data = plugin.download(year=2025, output_path=str(output_path))

                # requests.get should NOT have been called
                mock_get.assert_not_called()

                # Data should be List[LightweightPaper]
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
            "results": [{"id": 1, "name": "Fresh Paper", "authors": [{"fullname": "Test Author"}]}],
        }
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "iclr_2025.json"

            # Create existing file
            with open(output_path, "w") as f:
                json.dump({"count": 1, "results": [{"id": 1, "name": "Old Paper"}]}, f)

            # Download with force_download=True
            plugin = ICLRDownloaderPlugin()
            data = plugin.download(year=2025, output_path=str(output_path), force_download=True)

            # Should have made the request
            mock_get.assert_called_once()

            # Should have the new data - returns List[LightweightPaper]
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0].title == "Fresh Paper"

    @patch("abstracts_explorer.plugins.json_conference_downloader.requests.get")
    def test_download_request_exception(self, mock_get):
        """Test handling of request exceptions."""
        mock_get.side_effect = requests.exceptions.RequestException("Connection error")

        plugin = ICLRDownloaderPlugin()

        with pytest.raises(RuntimeError, match="Failed to download"):
            plugin.download(year=2025)

    @patch("abstracts_explorer.plugins.json_conference_downloader.requests.get")
    def test_download_invalid_json(self, mock_get):
        """Test handling of invalid JSON response."""
        mock_response = Mock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_get.return_value = mock_response

        plugin = ICLRDownloaderPlugin()

        with pytest.raises(RuntimeError, match="Invalid JSON response"):
            plugin.download(year=2025)

    @patch("abstracts_explorer.plugins.json_conference_downloader.requests.get")
    def test_download_custom_timeout(self, mock_get):
        """Test download with custom timeout."""
        mock_response = Mock()
        mock_response.json.return_value = {"count": 0, "next": None, "previous": None, "results": []}
        mock_get.return_value = mock_response

        plugin = ICLRDownloaderPlugin(timeout=60)
        plugin.download(year=2025)

        # Verify timeout was passed to requests.get
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["timeout"] == 60

    @patch("abstracts_explorer.plugins.json_conference_downloader.requests.get")
    def test_download_kwargs_override(self, mock_get):
        """Test that kwargs can override default timeout and verify_ssl."""
        mock_response = Mock()
        mock_response.json.return_value = {"count": 0, "next": None, "previous": None, "results": []}
        mock_get.return_value = mock_response

        plugin = ICLRDownloaderPlugin(timeout=30, verify_ssl=True)
        plugin.download(year=2025, timeout=90, verify_ssl=False)

        # Verify kwargs overrode defaults
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["timeout"] == 90
        assert call_kwargs["verify"] is False


class TestICLRPluginDatabaseIntegration:
    """Test ICLR plugin integration with database."""

    @patch("abstracts_explorer.plugins.json_conference_downloader.requests.get")
    def test_iclr_data_in_database(self, mock_get, monkeypatch):
        """Test that ICLR data can be stored in the database."""
        # Mock the ICLR API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "count": 1,
            "next": None,
            "previous": None,
            "results": [
                {
                    "id": 1,
                    "name": "Test ICLR Paper",
                    "abstract": "This is a test abstract for ICLR",
                    "authors": [
                        {"id": 101, "fullname": "Alice Smith", "institution": "MIT"},
                        {"id": 102, "fullname": "Bob Jones", "institution": "Stanford"},
                    ],
                    "keywords": ["deep learning", "transformers"],
                    "decision": "Accept (Poster)",
                    "session": "Poster Session 1",
                }
            ],
        }
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_iclr.db"

            # Download data (returns List[LightweightPaper])
            plugin = ICLRDownloaderPlugin()
            data = plugin.download(year=2025)

            # Create database and load data
            monkeypatch.setenv("PAPER_DB", str(db_path))
            from abstracts_explorer.config import get_config
            get_config(reload=True)
            with DatabaseManager() as db:
                db.create_tables()
                db.add_papers(data)

                # Verify paper was stored (lightweight schema uses 'title' not 'name', and 'uid' not 'id')
                papers = db.query("SELECT uid, title, abstract, year, conference, authors FROM papers")
                assert len(papers) == 1

                paper = papers[0]
                assert paper["title"] == "Test ICLR Paper"
                assert paper["abstract"] == "This is a test abstract for ICLR"
                assert paper["year"] == 2025
                assert paper["conference"] == "ICLR"

                # Verify authors (lightweight schema stores as semicolon-separated string)
                authors_str = paper["authors"]
                assert "Alice Smith" in authors_str
                assert "Bob Jones" in authors_str
                # Verify semicolon separator is used
                assert ";" in authors_str


class TestICLRPluginRegistration:
    """Test ICLR plugin registration."""

    def test_plugin_auto_registers(self):
        """Test that ICLR plugin auto-registers on import."""
        from abstracts_explorer.plugins import get_plugin

        plugin = get_plugin("iclr")
        assert plugin is not None
        assert isinstance(plugin, ICLRDownloaderPlugin)

    def test_plugin_in_list(self):
        """Test that ICLR plugin appears in plugin list."""
        from abstracts_explorer.plugins import list_plugin_names

        plugin_names = list_plugin_names()
        assert "iclr" in plugin_names

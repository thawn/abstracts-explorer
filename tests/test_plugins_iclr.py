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
from tests.conftest import set_test_db
from tests.helpers import raw_query

pytestmark = pytest.mark.usefixtures("mock_current_year_probe")


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

    def test_get_url_uses_eventhosts_for_modern_years(self):
        """Modern years use the EventHosts JSON endpoint."""
        plugin = ICLRDownloaderPlugin()

        assert plugin.get_url(2025) == "https://iclr.cc/static/virtual/data/iclr-2025-orals-posters.json"

    def test_get_url_uses_openreview_for_legacy_years(self):
        """Pre-2020 years use the OpenReview API endpoint."""
        plugin = ICLRDownloaderPlugin()

        assert plugin.get_url(2019) == "https://api.openreview.net/notes?content.venueid=ICLR.cc/2019/Conference"

    def test_validate_year_rejects_pre_start_year(self):
        """Years before 2017 are unsupported."""
        plugin = ICLRDownloaderPlugin()

        with pytest.raises(ValueError, match="Year 2016 not supported"):
            plugin.validate_year(2016)

    @patch("abstracts_explorer.config.get_config")
    @patch("getpass.getpass", return_value="secret")
    @patch("builtins.input", return_value="user@example.org")
    @patch("requests.Session.post")
    @patch("requests.Session.get")
    def test_download_legacy_prompts_for_login_on_auth_error(
        self,
        mock_get,
        mock_post,
        mock_input,
        mock_getpass,
        mock_get_config,
        monkeypatch,
    ):
        """Prompt for OpenReview credentials after an authentication failure."""
        mock_get_config.return_value = Mock(
            openreview_token="",
            openreview_email="",
            openreview_password="",
        )
        monkeypatch.setattr(ICLRDownloaderPlugin, "_prompt_is_available", staticmethod(lambda: True))
        auth_error = requests.exceptions.HTTPError("403 Client Error", response=Mock(status_code=403))
        login_response = Mock()
        login_response.json.return_value = {"token": "openreview-token"}
        mock_post.return_value = login_response
        notes_response = Mock()
        notes_response.json.return_value = {
            "notes": [
                {
                    "id": "forum1",
                    "forum": "forum1",
                    "content": {
                        "title": "OpenReview Paper",
                        "abstract": "OpenReview abstract.",
                        "authors": ["Alice Smith"],
                    },
                }
            ]
        }
        mock_get.side_effect = [auth_error, notes_response]

        papers = ICLRDownloaderPlugin().download(year=2019)

        assert len(papers) == 1
        mock_input.assert_called_once()
        mock_getpass.assert_called_once()
        assert mock_post.call_args.kwargs["json"] == {"id": "user@example.org", "password": "secret"}
        assert mock_get.call_args.kwargs["headers"] == {"Authorization": "Bearer openreview-token"}

    @patch("abstracts_explorer.config.get_config")
    @patch("builtins.input", return_value="")
    @patch("requests.Session.post")
    @patch("requests.Session.get")
    def test_download_legacy_declined_prompt_falls_back(
        self,
        mock_get,
        mock_post,
        mock_input,
        mock_get_config,
        monkeypatch,
    ):
        """Fall back to Semantic Scholar when the user declines the prompt."""
        mock_get_config.return_value = Mock(
            openreview_token="",
            openreview_email="",
            openreview_password="",
        )
        monkeypatch.setattr(ICLRDownloaderPlugin, "_prompt_is_available", staticmethod(lambda: True))
        auth_error = requests.exceptions.HTTPError("403 Client Error", response=Mock(status_code=403))
        s2_response = Mock()
        s2_response.json.return_value = {
            "data": [
                {
                    "paperId": "s2-1",
                    "title": "S2 Paper",
                    "abstract": "S2 abstract.",
                    "authors": [{"name": "Carol White"}],
                }
            ],
            "token": None,
        }
        mock_get.side_effect = [auth_error, s2_response]

        papers = ICLRDownloaderPlugin().download(year=2019)

        assert len(papers) == 1
        assert papers[0].title == "S2 Paper"
        mock_post.assert_not_called()
        assert "api.semanticscholar.org" in mock_get.call_args.args[0]

    @patch("abstracts_explorer.config.get_config")
    @patch("requests.Session.post")
    @patch("requests.Session.get")
    def test_download_legacy_uses_config_token(self, mock_get, mock_post, mock_get_config):
        """Use an OpenReview token from configuration when supplied there."""
        mock_get_config.return_value = Mock(
            openreview_token="config-token",
            openreview_email="",
            openreview_password="",
        )
        notes_response = Mock()
        notes_response.json.return_value = {
            "notes": [
                {
                    "id": "forum1",
                    "forum": "forum1",
                    "content": {
                        "title": "OpenReview Paper",
                        "abstract": "OpenReview abstract.",
                        "authors": ["Alice Smith"],
                    },
                }
            ]
        }
        mock_get.return_value = notes_response

        papers = ICLRDownloaderPlugin().download(year=2019)

        assert len(papers) == 1
        mock_post.assert_not_called()
        assert mock_get.call_args.kwargs["headers"] == {"Authorization": "Bearer config-token"}

    @patch("requests.Session.post")
    @patch("requests.Session.get")
    def test_download_legacy_logs_in_with_credentials(self, mock_get, mock_post):
        """Log in with an OpenReview account before querying notes."""
        login_response = Mock()
        login_response.json.return_value = {"token": "openreview-token"}
        mock_post.return_value = login_response

        notes_response = Mock()
        notes_response.json.return_value = {
            "notes": [
                {
                    "id": "forum1",
                    "forum": "forum1",
                    "content": {
                        "title": "OpenReview Paper",
                        "abstract": "OpenReview abstract.",
                        "authors": ["Alice Smith"],
                    },
                }
            ]
        }
        mock_get.return_value = notes_response

        papers = ICLRDownloaderPlugin().download(
            year=2019,
            openreview_email="user@example.org",
            openreview_password="secret",
        )

        assert len(papers) == 1
        mock_post.assert_called_once()
        post_args, post_kwargs = mock_post.call_args
        assert post_args[0].endswith("/login")
        assert post_kwargs["json"] == {"id": "user@example.org", "password": "secret"}
        assert mock_get.call_args.kwargs["headers"] == {"Authorization": "Bearer openreview-token"}

    @patch("requests.Session.post")
    @patch("requests.Session.get")
    def test_download_legacy_uses_existing_token(self, mock_get, mock_post):
        """Use a supplied bearer token without logging in."""
        notes_response = Mock()
        notes_response.json.return_value = {
            "notes": [
                {
                    "id": "forum1",
                    "forum": "forum1",
                    "content": {
                        "title": "OpenReview Paper",
                        "abstract": "OpenReview abstract.",
                        "authors": ["Alice Smith"],
                    },
                }
            ]
        }
        mock_get.return_value = notes_response

        papers = ICLRDownloaderPlugin().download(year=2019, openreview_token="given-token")

        assert len(papers) == 1
        mock_post.assert_not_called()
        assert mock_get.call_args.kwargs["headers"] == {"Authorization": "Bearer given-token"}

    @patch("requests.Session.post")
    @patch("requests.Session.get")
    def test_download_legacy_falls_back_when_login_fails(self, mock_get, mock_post):
        """Fall back to Semantic Scholar when OpenReview login fails."""
        mock_post.side_effect = requests.exceptions.RequestException("bad credentials")
        s2_response = Mock()
        s2_response.json.return_value = {
            "data": [
                {
                    "paperId": "s2-1",
                    "title": "S2 Paper",
                    "abstract": "S2 abstract.",
                    "authors": [{"name": "Carol White"}],
                }
            ],
            "token": None,
        }
        mock_get.return_value = s2_response

        papers = ICLRDownloaderPlugin().download(
            year=2019,
            openreview_email="user@example.org",
            openreview_password="wrong",
        )

        assert len(papers) == 1
        assert papers[0].title == "S2 Paper"
        assert "api.semanticscholar.org" in mock_get.call_args.args[0]

    @patch("requests.Session.get")
    def test_download_legacy_uses_openreview(self, mock_get):
        """Prefer OpenReview when it returns records."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "notes": [
                {
                    "id": "forum1",
                    "forum": "forum1",
                    "content": {
                        "title": "OpenReview Paper",
                        "abstract": "OpenReview abstract.",
                        "authors": ["Alice Smith", "Bob Jones"],
                    },
                }
            ]
        }
        mock_get.return_value = mock_response

        papers = ICLRDownloaderPlugin().download(year=2019)

        assert len(papers) == 1
        paper = papers[0]
        assert paper.title == "OpenReview Paper"
        assert paper.abstract == "OpenReview abstract."
        assert paper.authors == ["Alice Smith", "Bob Jones"]
        assert paper.original_id == "forum1"
        assert paper.url == "https://openreview.net/forum?id=forum1"
        assert paper.session == "ICLR 2019"
        assert "api.openreview.net" in mock_get.call_args.args[0]

    @patch("requests.Session.get")
    def test_download_legacy_falls_back_to_semantic_scholar(self, mock_get):
        """Fall back to Semantic Scholar when OpenReview fails."""
        s2_response = Mock()
        s2_response.json.return_value = {
            "data": [
                {
                    "paperId": "s2-1",
                    "title": "S2 Paper",
                    "abstract": "S2 abstract.",
                    "authors": [{"name": "Carol White"}],
                    "url": "https://www.semanticscholar.org/paper/s2-1",
                }
            ],
            "token": None,
        }
        mock_get.side_effect = [
            requests.exceptions.RequestException("403"),
            s2_response,
        ]

        papers = ICLRDownloaderPlugin().download(year=2019)

        assert len(papers) == 1
        paper = papers[0]
        assert paper.title == "S2 Paper"
        assert paper.abstract == "S2 abstract."
        assert paper.authors == ["Carol White"]
        assert paper.original_id == "s2-1"
        assert paper.url == "https://www.semanticscholar.org/paper/s2-1"
        assert "api.semanticscholar.org" in mock_get.call_args.args[0]

    @patch("requests.Session.get")
    def test_download_legacy_raises_when_all_sources_empty(self, mock_get):
        """Raise when both OpenReview and Semantic Scholar return nothing."""
        empty_response = Mock()
        empty_response.json.return_value = {"notes": [], "data": [], "token": None}
        mock_get.return_value = empty_response

        with pytest.raises(RuntimeError, match="No valid ICLR papers"):
            ICLRDownloaderPlugin().download(year=2019)

    @patch("requests.Session.get")
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

    @patch("requests.Session.get")
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

        plugin = ICLRDownloaderPlugin()
        plugin.supported_years = [2025]
        data = plugin.download()  # No year specified

        # Should default to the newest supported year
        call_args = mock_get.call_args
        assert "iclr-2025-orals-posters.json" in call_args[0][0]

        # Verify year was set to 2025 - returns List[LightweightPaper]
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0].year == 2025

    @patch("requests.Session.get")
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
            with patch("requests.Session.get") as mock_get:
                data = plugin.download(year=2025, output_path=str(output_path))

                # requests.get should NOT have been called
                mock_get.assert_not_called()

                # Data should be List[LightweightPaper]
                assert isinstance(data, list)
                assert len(data) == 1
                assert data[0].title == "Cached Paper"

    @patch("requests.Session.get")
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

    @patch("requests.Session.get")
    def test_download_request_exception(self, mock_get):
        """Test handling of request exceptions."""
        mock_get.side_effect = requests.exceptions.RequestException("Connection error")

        plugin = ICLRDownloaderPlugin()

        with pytest.raises(RuntimeError, match="Failed to download"):
            plugin.download(year=2025)

    @patch("requests.Session.get")
    def test_download_invalid_json(self, mock_get):
        """Test handling of invalid JSON response."""
        mock_response = Mock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_get.return_value = mock_response

        plugin = ICLRDownloaderPlugin()

        with pytest.raises(RuntimeError, match="Invalid JSON response"):
            plugin.download(year=2025)

    @patch("requests.Session.get")
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

    @patch("requests.Session.get")
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

    @patch("requests.Session.get")
    def test_iclr_data_in_database(self, mock_get):
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
            set_test_db(db_path)
            with DatabaseManager() as db:
                db.create_tables()
                db.add_papers(data)

                # Verify paper was stored (lightweight schema uses 'title' not 'name', and 'uid' not 'id')
                papers = raw_query(db, "SELECT uid, title, abstract, year, conference, authors FROM papers")
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

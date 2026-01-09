"""
Tests for the downloader module.
"""

import json
import pytest
from unittest.mock import Mock, patch
import requests

from abstracts_explorer.downloader import download_json, download_conference_data, DownloadError


@pytest.fixture
def sample_json_data():
    """Sample JSON data for testing."""
    return {
        "papers": [
            {
                "id": "paper1",
                "title": "Test Paper 1",
                "abstract": "This is a test abstract",
                "authors": ["Author 1", "Author 2"],
            },
            {"id": "paper2", "title": "Test Paper 2", "abstract": "Another test abstract", "authors": ["Author 3"]},
        ]
    }


@pytest.fixture
def mock_response(sample_json_data):
    """Mock requests response."""
    mock = Mock()
    mock.status_code = 200
    mock.json.return_value = sample_json_data
    mock.raise_for_status = Mock()
    return mock


class TestDownloadJson:
    """Tests for download_json function."""

    def test_download_json_success(self, mock_response, sample_json_data):
        """Test successful JSON download."""
        with patch("abstracts_explorer.downloader.requests.get", return_value=mock_response):
            result = download_json("https://example.com/data.json")
            assert result == sample_json_data

    def test_download_json_with_output_path(self, mock_response, sample_json_data, tmp_path):
        """Test downloading JSON with output path."""
        output_file = tmp_path / "output.json"

        with patch("abstracts_explorer.downloader.requests.get", return_value=mock_response):
            result = download_json("https://example.com/data.json", output_path=output_file)

            assert result == sample_json_data
            assert output_file.exists()

            with open(output_file, "r") as f:
                saved_data = json.load(f)
            assert saved_data == sample_json_data

    def test_download_json_creates_directories(self, mock_response, sample_json_data, tmp_path):
        """Test that download_json creates parent directories."""
        output_file = tmp_path / "subdir" / "another" / "output.json"

        with patch("abstracts_explorer.downloader.requests.get", return_value=mock_response):
            download_json("https://example.com/data.json", output_path=output_file)

            assert output_file.exists()
            assert output_file.parent.exists()

    def test_download_json_empty_url(self):
        """Test download_json with empty URL."""
        with pytest.raises(ValueError, match="URL must be a non-empty string"):
            download_json("")

    def test_download_json_none_url(self):
        """Test download_json with None URL."""
        with pytest.raises(ValueError, match="URL must be a non-empty string"):
            download_json(None)

    def test_download_json_request_exception(self):
        """Test download_json when request fails."""
        with patch("abstracts_explorer.downloader.requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.RequestException("Connection error")

            with pytest.raises(DownloadError, match="Failed to download"):
                download_json("https://example.com/data.json")

    def test_download_json_http_error(self):
        """Test download_json when HTTP error occurs."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")

        with patch("abstracts_explorer.downloader.requests.get", return_value=mock_response):
            with pytest.raises(DownloadError, match="Failed to download"):
                download_json("https://example.com/data.json")

    def test_download_json_invalid_json(self):
        """Test download_json when response is not valid JSON."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)

        with patch("abstracts_explorer.downloader.requests.get", return_value=mock_response):
            with pytest.raises(DownloadError, match="Invalid JSON response"):
                download_json("https://example.com/data.json")

    def test_download_json_custom_timeout(self, mock_response):
        """Test download_json with custom timeout."""
        with patch("abstracts_explorer.downloader.requests.get", return_value=mock_response) as mock_get:
            download_json("https://example.com/data.json", timeout=60)
            mock_get.assert_called_once_with("https://example.com/data.json", timeout=60, verify=True)

    def test_download_json_verify_ssl_false(self, mock_response):
        """Test download_json with SSL verification disabled."""
        with patch("abstracts_explorer.downloader.requests.get", return_value=mock_response) as mock_get:
            download_json("https://example.com/data.json", verify_ssl=False)
            mock_get.assert_called_once_with("https://example.com/data.json", timeout=30, verify=False)


class TestDownloadConferenceData:
    """Tests for download_conference_data function."""

    def test_download_conference_data_default_year(self, mock_response, sample_json_data):
        """Test downloading NeurIPS data with default year."""
        with patch("abstracts_explorer.downloader.requests.get", return_value=mock_response) as mock_get:
            result = download_conference_data()

            assert result == sample_json_data
            expected_url = "https://neurips.cc/static/virtual/data/neurips-2025-orals-posters.json"
            mock_get.assert_called_once_with(expected_url, timeout=30, verify=True)

    def test_download_conference_data_custom_year(self, mock_response, sample_json_data):
        """Test downloading NeurIPS data with custom year."""
        with patch("abstracts_explorer.downloader.requests.get", return_value=mock_response) as mock_get:
            result = download_conference_data(year=2024)

            assert result == sample_json_data
            expected_url = "https://neurips.cc/static/virtual/data/neurips-2024-orals-posters.json"
            mock_get.assert_called_once_with(expected_url, timeout=30, verify=True)

    def test_download_conference_data_with_output(self, mock_response, sample_json_data, tmp_path):
        """Test downloading NeurIPS data with output path."""
        output_file = tmp_path / "neurips_2025.json"

        with patch("abstracts_explorer.downloader.requests.get", return_value=mock_response):
            result = download_conference_data(output_path=output_file)

            assert result == sample_json_data
            assert output_file.exists()

    def test_download_conference_data_custom_timeout(self, mock_response):
        """Test downloading NeurIPS data with custom timeout."""
        with patch("abstracts_explorer.downloader.requests.get", return_value=mock_response) as mock_get:
            download_conference_data(timeout=120)

            expected_url = "https://neurips.cc/static/virtual/data/neurips-2025-orals-posters.json"
            mock_get.assert_called_once_with(expected_url, timeout=120, verify=True)

    def test_download_conference_data_failure(self):
        """Test download_conference_data when download fails."""
        with patch("abstracts_explorer.downloader.requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.RequestException("Network error")

            with pytest.raises(DownloadError):
                download_conference_data()


class TestLocalCaching:
    """Tests for local file caching functionality."""

    def test_download_json_loads_from_existing_file(self, sample_json_data, tmp_path):
        """Test that download_json loads from existing local file without downloading."""
        output_file = tmp_path / "cached_data.json"

        # Create a local file with data
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(sample_json_data, f)

        # Download should load from local file, not make HTTP request
        with patch("abstracts_explorer.downloader.requests.get") as mock_get:
            result = download_json("https://example.com/data.json", output_path=output_file)

            assert result == sample_json_data
            # Verify no HTTP request was made
            mock_get.assert_not_called()

    def test_download_json_force_download_ignores_cache(self, mock_response, sample_json_data, tmp_path):
        """Test that force_download=True downloads even if file exists."""
        output_file = tmp_path / "cached_data.json"

        # Create a local file with different data
        old_data = {"old": "data"}
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(old_data, f)

        # Force download should ignore local file and download fresh data
        with patch("abstracts_explorer.downloader.requests.get", return_value=mock_response):
            result = download_json("https://example.com/data.json", output_path=output_file, force_download=True)

            assert result == sample_json_data
            assert result != old_data

            # Verify file was updated
            with open(output_file, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
            assert saved_data == sample_json_data

    def test_download_json_downloads_if_cache_corrupted(self, mock_response, sample_json_data, tmp_path):
        """Test that download happens if local file is corrupted."""
        output_file = tmp_path / "corrupted_data.json"

        # Create a corrupted JSON file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("{invalid json content")

        # Should detect corrupted file and download fresh data
        with patch("abstracts_explorer.downloader.requests.get", return_value=mock_response):
            result = download_json("https://example.com/data.json", output_path=output_file)

            assert result == sample_json_data

            # Verify file was fixed
            with open(output_file, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
            assert saved_data == sample_json_data

    def test_download_conference_data_uses_cache(self, sample_json_data, tmp_path):
        """Test that download_conference_data uses cached file."""
        output_file = tmp_path / "neurips_2025.json"

        # Create a cached file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(sample_json_data, f)

        # Should load from cache
        with patch("abstracts_explorer.downloader.requests.get") as mock_get:
            result = download_conference_data(output_path=output_file)

            assert result == sample_json_data
            # No HTTP request should be made
            mock_get.assert_not_called()

    def test_download_conference_data_force_download(self, mock_response, sample_json_data, tmp_path):
        """Test that download_conference_data with force_download ignores cache."""
        output_file = tmp_path / "neurips_2025.json"

        # Create a cached file
        old_data = {"cached": "data"}
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(old_data, f)

        # Force download should ignore cache
        with patch("abstracts_explorer.downloader.requests.get", return_value=mock_response):
            result = download_conference_data(output_path=output_file, force_download=True)

            assert result == sample_json_data
            assert result != old_data

    def test_download_json_no_output_path_always_downloads(self, mock_response, sample_json_data):
        """Test that without output_path, data is always downloaded."""
        # Without output_path, should always download (no caching)
        with patch("abstracts_explorer.downloader.requests.get", return_value=mock_response) as mock_get:
            result = download_json("https://example.com/data.json")

            assert result == sample_json_data
            # Should make HTTP request
            mock_get.assert_called_once()

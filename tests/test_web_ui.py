"""
Tests for web_ui/app.py module.

This module contains unit tests for the web UI application including:
- Flask app configuration and initialization
- API endpoints (search, chat, stats, paper details)
- Semantic search functionality
- Error handling and edge cases
- Response formatting
"""

import json
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from tests.conftest import set_test_db

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from abstracts_explorer.web_ui import app as flask_app
from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.config import get_config

# ============================================================
# Tests from test_web.py
# ============================================================


@pytest.fixture
def app():
    """
    Create Flask app for testing.

    Returns
    -------
    Flask
        Flask application instance
    """
    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture
def client(app):
    """
    Create test client.

    Parameters
    ----------
    app : Flask
        Flask application

    Returns
    -------
    FlaskClient
        Test client for making requests
    """
    return app.test_client()


# Note: Flask app fixtures (app, client) are kept in this file as they're specific
# to web testing and not shared with other test modules.


@pytest.fixture
def test_db(tmp_path, web_test_papers):
    """
    Create a test database with sample papers for web testing.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory path
    web_test_papers : list
        List of test papers from shared fixture

    Returns
    -------
    DatabaseManager
        Database manager with test data

    Notes
    -----
    Uses the shared web_test_papers fixture from conftest.py to ensure
    consistency across web-related tests.
    """
    db_path = tmp_path / "test.db"
    set_test_db(str(db_path))
    db = DatabaseManager()

    with db:
        db.create_tables()
        db.add_papers(web_test_papers)

    return db


class TestWebInterface:
    """Test web interface endpoints."""

    def test_index_route(self, client):
        """Test that the main page loads."""
        response = client.get("/")
        assert response.status_code == 200
        assert b"Abstracts Explorer" in response.data

    def test_index_version_display(self, client):
        """Test that the version is displayed in the footer."""
        response = client.get("/")
        assert response.status_code == 200
        # Check that version is present in footer
        assert b"Abstracts Explorer" in response.data
        # Ensure "Abstracts Explorer" links to the GitHub project page
        assert b"https://github.com/thawn/abstracts-explorer" in response.data
        # Ensure the template variable was replaced (not left as {{ version }})
        assert b"{{ version }}" not in response.data

    def test_index_documentation_link(self, client):
        """Test that the documentation link is present in the header."""
        response = client.get("/")
        assert response.status_code == 200
        assert b"https://thawn.github.io/abstracts-explorer/web_ui.html" in response.data

    def test_index_no_imprint_link_by_default(self, client):
        """Test that the imprint link is not shown by default."""
        response = client.get("/")
        assert response.status_code == 200
        assert b"Imprint" not in response.data

    def test_index_imprint_link_shown_when_configured(self, client):
        """Test that the imprint link is shown when IMPRINT_LINK is configured."""
        import importlib

        web_app_module = importlib.import_module("abstracts_explorer.web_ui.app")

        original_imprint_link = web_app_module._config.imprint_link
        try:
            web_app_module._config.imprint_link = "https://example.com/imprint"
            response = client.get("/")
            assert response.status_code == 200
            assert b"Imprint" in response.data
            assert b"https://example.com/imprint" in response.data
        finally:
            web_app_module._config.imprint_link = original_imprint_link

    def test_stats_endpoint_no_db(self, client):
        """Test stats endpoint when database doesn't exist."""
        response = client.get("/api/stats")
        # Should return error if DB doesn't exist
        assert response.status_code in [200, 500]

    def test_filters_endpoint_no_db(self, client):
        """Test filters endpoint when database doesn't exist."""
        response = client.get("/api/filters")
        # Should return error if DB doesn't exist
        assert response.status_code in [200, 500]

    def test_years_endpoint_no_db(self, client):
        """Test years endpoint when database doesn't exist."""
        response = client.get("/api/years")
        # Should return error if DB doesn't exist
        assert response.status_code in [200, 500]

    def _mock_db_for_available_filters(self, app_module, conferences_with_years):
        """
        Helper to create a mock database for available-filters tests.

        Parameters
        ----------
        app_module : module
            The web_ui.app module.
        conferences_with_years : dict
            Mapping of conference name to list of years (descending), e.g.
            {"NeurIPS": [2025, 2024], "ICLR": [2024]}.

        Returns
        -------
        MagicMock
            Configured mock database.
        """
        from abstracts_explorer.database import DatabaseManager

        mock_db = MagicMock()
        mock_db.get_conference_years_from_db.return_value = conferences_with_years
        mock_db.get_conferences.return_value = sorted(conferences_with_years.keys())
        mock_db.get_years.side_effect = lambda conference=None: (
            conferences_with_years.get(conference, [])
            if conference
            else sorted({y for years in conferences_with_years.values() for y in years}, reverse=True)
        )
        # Delegate resolve_default_conference_year to the real implementation so
        # the business logic is exercised via database.py.
        mock_db.resolve_default_conference_year.side_effect = (
            lambda conf, year: DatabaseManager.resolve_default_conference_year(mock_db, conf, year)
        )
        return mock_db

    def test_available_filters_includes_defaults(self, client):
        """Test that /api/available-filters honours configured defaults when DB has matching data."""
        import sys

        app_module = sys.modules["abstracts_explorer.web_ui.app"]
        mock_db = self._mock_db_for_available_filters(app_module, {"NeurIPS": [2025, 2024]})

        with patch.object(app_module, "get_database", return_value=mock_db):
            with patch("abstracts_explorer.web_ui.app.get_config") as mock_cfg:
                mock_cfg.return_value.default_conference = "NeurIPS"
                mock_cfg.return_value.default_year = 2024
                response = client.get("/api/available-filters")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["default_conference"] == "NeurIPS"
        assert data["default_year"] == 2024

    def test_available_filters_empty_defaults(self, client):
        """Test that /api/available-filters returns empty defaults when DB is empty and nothing is configured."""
        import sys

        app_module = sys.modules["abstracts_explorer.web_ui.app"]
        mock_db = self._mock_db_for_available_filters(app_module, {})

        with patch.object(app_module, "get_database", return_value=mock_db):
            with patch("abstracts_explorer.web_ui.app.get_config") as mock_cfg:
                mock_cfg.return_value.default_conference = ""
                mock_cfg.return_value.default_year = 0
                response = client.get("/api/available-filters")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["default_conference"] == ""
        assert data["default_year"] is None

    def test_available_filters_includes_conference_years(self, client):
        """Test that /api/available-filters includes conference_years reflecting actual DB data."""
        import sys

        app_module = sys.modules["abstracts_explorer.web_ui.app"]
        mock_db = self._mock_db_for_available_filters(app_module, {"NeurIPS": [2025, 2024], "ICLR": [2024]})

        with patch.object(app_module, "get_database", return_value=mock_db):
            with patch("abstracts_explorer.web_ui.app.get_config") as mock_cfg:
                mock_cfg.return_value.default_conference = ""
                mock_cfg.return_value.default_year = 0
                response = client.get("/api/available-filters")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert "conference_years" in data
        assert data["conference_years"]["NeurIPS"] == [2025, 2024]
        assert data["conference_years"]["ICLR"] == [2024]
        assert sorted(data["conferences"]) == ["ICLR", "NeurIPS"]
        assert data["years"] == [2025, 2024]

    def test_available_filters_fallback_when_default_conf_no_data(self, client):
        """Test that default_conference falls back to most recent DB entry when configured default has no data."""
        import sys

        app_module = sys.modules["abstracts_explorer.web_ui.app"]
        # DB only has ICLR/2024 – configured default "ICML" has no data
        mock_db = self._mock_db_for_available_filters(app_module, {"ICLR": [2024]})

        with patch.object(app_module, "get_database", return_value=mock_db):
            with patch("abstracts_explorer.web_ui.app.get_config") as mock_cfg:
                mock_cfg.return_value.default_conference = "ICML"
                mock_cfg.return_value.default_year = 2024
                response = client.get("/api/available-filters")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["default_conference"] == "ICLR"
        assert data["default_year"] == 2024

    def test_available_filters_fallback_when_no_default_configured(self, client):
        """Test that the most recent conference/year from DB is returned when no default is configured."""
        import sys

        app_module = sys.modules["abstracts_explorer.web_ui.app"]
        # DB has NeurIPS/2024 and ICLR/2025 – ICLR/2025 is the most recent
        mock_db = self._mock_db_for_available_filters(app_module, {"NeurIPS": [2024], "ICLR": [2025]})

        with patch.object(app_module, "get_database", return_value=mock_db):
            with patch("abstracts_explorer.web_ui.app.get_config") as mock_cfg:
                mock_cfg.return_value.default_conference = ""
                mock_cfg.return_value.default_year = 0
                response = client.get("/api/available-filters")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["default_conference"] == "ICLR"
        assert data["default_year"] == 2025

    def test_available_filters_fallback_year_for_default_conf(self, client):
        """Test that default_year falls back to the most recent DB year when configured year has no data."""
        import sys

        app_module = sys.modules["abstracts_explorer.web_ui.app"]
        # DB has NeurIPS 2024 and 2025 but not 2020
        mock_db = self._mock_db_for_available_filters(app_module, {"NeurIPS": [2025, 2024]})

        with patch.object(app_module, "get_database", return_value=mock_db):
            with patch("abstracts_explorer.web_ui.app.get_config") as mock_cfg:
                mock_cfg.return_value.default_conference = "NeurIPS"
                mock_cfg.return_value.default_year = 2020
                response = client.get("/api/available-filters")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["default_conference"] == "NeurIPS"
        assert data["default_year"] == 2025

    def test_available_filters_includes_default_distance_threshold(self, client):
        """Test that /api/available-filters includes the module-level default distance threshold (1.2)."""
        import sys

        app_module = sys.modules["abstracts_explorer.web_ui.app"]
        mock_db = self._mock_db_for_available_filters(app_module, {"NeurIPS": [2025]})

        with patch.object(app_module, "get_database", return_value=mock_db):
            with patch("abstracts_explorer.web_ui.app.get_config") as mock_cfg:
                mock_cfg.return_value.default_conference = "NeurIPS"
                mock_cfg.return_value.default_year = 2025
                response = client.get("/api/available-filters")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["default_distance_threshold"] == 1.2

    def test_index_llm_backend_not_shown_for_unknown_url(self, client):
        """Test that no LLM backend reference is shown for an unknown backend URL."""
        import sys

        app_module = sys.modules["abstracts_explorer.web_ui.app"]
        original_url = app_module._config.llm_backend_url
        try:
            app_module._config.llm_backend_url = "http://unknown.example.com:5678"
            response = client.get("/")
            assert response.status_code == 200
            assert b"LLM provided by" not in response.data
        finally:
            app_module._config.llm_backend_url = original_url

    def test_index_llm_backend_blablador_shown(self, client):
        """Test that the Blablador backend link is shown when configured."""
        import sys

        app_module = sys.modules["abstracts_explorer.web_ui.app"]
        original_url = app_module._config.llm_backend_url
        try:
            app_module._config.llm_backend_url = "https://api.helmholtz-blablador.fz-juelich.de/v1"
            response = client.get("/")
            assert response.status_code == 200
            assert b"LLM provided by" in response.data
            assert b"BLABLADOR" in response.data
            assert b"helmholtz-blablador.fz-juelich.de" in response.data
            assert b"blablador-logo.png" in response.data
        finally:
            app_module._config.llm_backend_url = original_url

    def test_index_llm_backend_lmstudio_shown(self, client):
        """Test that the LM Studio backend link is shown when configured."""
        import sys

        app_module = sys.modules["abstracts_explorer.web_ui.app"]
        original_url = app_module._config.llm_backend_url
        try:
            app_module._config.llm_backend_url = "http://localhost:1234/v1"
            response = client.get("/")
            assert response.status_code == 200
            assert b"LLM provided by" in response.data
            assert b"LM Studio" in response.data
            assert b"lmstudio.ai" in response.data
        finally:
            app_module._config.llm_backend_url = original_url

    def test_index_llm_backend_rossendorf_shown(self, client):
        """Test that the chat.fz-rossendorf.de backend link is shown when configured."""
        import sys

        app_module = sys.modules["abstracts_explorer.web_ui.app"]
        original_url = app_module._config.llm_backend_url
        try:
            app_module._config.llm_backend_url = "https://chat.fz-rossendorf.de/v1"
            response = client.get("/")
            assert response.status_code == 200
            assert b"LLM provided by" in response.data
            assert b"chat.fz-rossendorf.de" in response.data
        finally:
            app_module._config.llm_backend_url = original_url


class TestGetLLMBackendInfo:
    """Test get_llm_backend_info() helper function."""

    def setup_method(self):
        """Import the function once for all tests."""
        from abstracts_explorer.web_ui.app import get_llm_backend_info

        self.get_llm_backend_info = get_llm_backend_info

    def test_blablador_url_detected(self):
        """Test that the Blablador API URL is recognised."""
        result = self.get_llm_backend_info("https://api.helmholtz-blablador.fz-juelich.de/v1")
        assert result["name"] == "BLABLADOR"
        assert result["homepage"] == "https://helmholtz-blablador.fz-juelich.de"
        assert result["logo"] == "blablador-logo.png"

    def test_blablador_homepage_url_detected(self):
        """Test that the Blablador homepage URL is also recognised."""
        result = self.get_llm_backend_info("https://helmholtz-blablador.fz-juelich.de")
        assert result["name"] == "BLABLADOR"

    def test_lmstudio_localhost_detected(self):
        """Test that the LM Studio localhost URL is recognised."""
        result = self.get_llm_backend_info("http://localhost:1234")
        assert result["name"] == "LM Studio"
        assert result["homepage"] == "https://lmstudio.ai"
        assert result["logo"] is None

    def test_lmstudio_127001_detected(self):
        """Test that the LM Studio 127.0.0.1 URL is recognised."""
        result = self.get_llm_backend_info("http://127.0.0.1:1234/v1")
        assert result["name"] == "LM Studio"

    def test_rossendorf_url_detected(self):
        """Test that the chat.fz-rossendorf.de URL is recognised."""
        result = self.get_llm_backend_info("https://chat.fz-rossendorf.de/v1")
        assert result["name"] == "chat.fz-rossendorf.de"
        assert result["homepage"] == "https://chat.fz-rossendorf.de"
        assert result["logo"] is None

    def test_unknown_url_returns_none_fields(self):
        """Test that an unknown URL returns None for all metadata fields."""
        result = self.get_llm_backend_info("http://unknown.example.com:5678")
        assert result["name"] is None
        assert result["homepage"] is None
        assert result["logo"] is None

    def test_empty_url_returns_none_fields(self):
        """Test that an empty URL returns None for all metadata fields."""
        result = self.get_llm_backend_info("")
        assert result["name"] is None
        assert result["homepage"] is None
        assert result["logo"] is None


class TestConferenceURLRoute:
    """Test direct URL paths to conferences (e.g., /icml, /neurips)."""

    def test_valid_conference_url_by_plugin_name(self, client):
        """Test that navigating to /neurips selects the NeurIPS conference."""
        import sys

        app_module = sys.modules["abstracts_explorer.web_ui.app"]
        mock_db = MagicMock()
        mock_db.resolve_conference_for_url.return_value = {"conference": "NeurIPS", "error": None}

        with patch.object(app_module, "get_database", return_value=mock_db):
            response = client.get("/neurips")

        assert response.status_code == 200
        assert b"Abstracts Explorer" in response.data
        # Check that urlConference is set in the page
        assert b"window.urlConference" in response.data
        assert b"NeurIPS" in response.data

    def test_valid_conference_url_case_insensitive(self, client):
        """Test that conference URLs are case-insensitive (e.g., /ICML, /icml, /Icml)."""
        import sys

        app_module = sys.modules["abstracts_explorer.web_ui.app"]
        mock_db = MagicMock()
        mock_db.resolve_conference_for_url.return_value = {"conference": "ICML", "error": None}

        with patch.object(app_module, "get_database", return_value=mock_db):
            response = client.get("/ICML")

        assert response.status_code == 200
        assert b"window.urlConference" in response.data

    def test_valid_conference_url_by_conference_name(self, client):
        """Test that navigating to the full conference name works (e.g., /NeurIPS)."""
        import sys

        app_module = sys.modules["abstracts_explorer.web_ui.app"]
        mock_db = MagicMock()
        mock_db.resolve_conference_for_url.return_value = {"conference": "NeurIPS", "error": None}

        with patch.object(app_module, "get_database", return_value=mock_db):
            response = client.get("/NeurIPS")

        assert response.status_code == 200
        assert b"window.urlConference" in response.data
        assert b"NeurIPS" in response.data

    def test_unknown_conference_url_shows_error(self, client):
        """Test that an unknown conference path shows an error with available conferences."""
        import sys

        app_module = sys.modules["abstracts_explorer.web_ui.app"]
        mock_db = MagicMock()
        mock_db.resolve_conference_for_url.return_value = {
            "conference": None,
            "error": {
                "message": "Conference 'unknownconf' not found.",
                "available_conferences": ["ICLR", "NeurIPS"],
            },
        }

        with patch.object(app_module, "get_database", return_value=mock_db):
            response = client.get("/unknownconf")

        assert response.status_code == 200
        assert b"Abstracts Explorer" in response.data
        # Should contain error info
        assert b"window.urlConferenceError" in response.data
        assert b"unknownconf" in response.data
        # Should NOT contain urlConference (valid conference)
        assert b"window.urlConference =" not in response.data

    def test_conference_with_no_data_shows_error(self, client):
        """Test that a known plugin conference with no DB data shows a helpful error."""
        import sys

        app_module = sys.modules["abstracts_explorer.web_ui.app"]
        mock_db = MagicMock()
        mock_db.resolve_conference_for_url.return_value = {
            "conference": None,
            "error": {
                "message": "No data available for conference 'ICML'. Please download data first.",
                "available_conferences": ["NeurIPS"],
            },
        }

        with patch.object(app_module, "get_database", return_value=mock_db):
            response = client.get("/icml")

        assert response.status_code == 200
        assert b"window.urlConferenceError" in response.data
        assert b"No data available" in response.data

    def test_conference_url_does_not_conflict_with_api(self, client):
        """Test that /api paths still work and are not caught by the conference route."""
        response = client.get("/api/stats")
        # Should be handled by the stats endpoint, not the conference route
        assert response.status_code in [200, 500]

    def test_conference_url_does_not_conflict_with_health(self, client):
        """Test that /health still works."""
        response = client.get("/health")
        # Should be handled by the health endpoint
        assert response.status_code in [200, 503]

    def test_root_still_works(self, client):
        """Test that the root URL / still returns the main page without conference context."""
        response = client.get("/")
        assert response.status_code == 200
        assert b"Abstracts Explorer" in response.data
        # Should NOT contain conference URL overrides
        assert b"window.urlConference" not in response.data
        assert b"window.urlConferenceError" not in response.data

    def test_well_known_path_returns_404(self, client):
        """Test that /.well-known paths are not intercepted by the conference route."""
        response = client.get("/.well-known")
        assert response.status_code == 404


class TestSearchEndpoint:
    """Test the search endpoint specifically."""

    def test_search_without_query(self, client):
        """Test search endpoint with missing query."""
        response = client.post("/api/search", data=json.dumps({}), content_type="application/json")
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_search_with_empty_query(self, client):
        """Test search endpoint with empty query."""
        response = client.post("/api/search", data=json.dumps({"query": ""}), content_type="application/json")
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_search_keyword_parameters(self, client):
        """
        Test that keyword search uses correct parameters.

        This test ensures the bug where 'title' and 'abstract' were
        passed to search_papers() doesn't happen again. It verifies
        that search_papers_keyword is called with 'query' parameter only.
        """
        from unittest.mock import MagicMock, patch
        import sys

        # Get the actual app module (not the Flask app object)
        app_module = sys.modules["abstracts_explorer.web_ui.app"]

        # Create a mock database
        mock_db = MagicMock()
        mock_papers = [{"id": 1, "name": "Test Paper", "abstract": "Test abstract", "uid": "test1", "authors": []}]

        # Setup the mock to return our test data
        mock_db.search_papers_keyword.return_value = mock_papers

        # Patch the get_database function to return our mock
        with patch.object(app_module, "get_database", return_value=mock_db):
            # Make search request
            response = client.post(
                "/api/search",
                data=json.dumps({"query": "test", "use_embeddings": False, "limit": 10}),
                content_type="application/json",
            )

        # Verify search_papers_keyword was called with correct parameters
        # This is the key test - it should use 'query' parameter
        mock_db.search_papers_keyword.assert_called_once()
        call_args = mock_db.search_papers_keyword.call_args

        # Check that it was called with 'query' parameter
        assert "query" in call_args.kwargs, "search_papers_keyword should be called with 'query' parameter"
        assert call_args.kwargs["query"] == "test"
        assert call_args.kwargs["limit"] == 10

        # Verify it was NOT called with invalid parameters
        assert "title" not in call_args.kwargs, "search_papers_keyword should NOT be called with 'title' parameter"
        assert (
            "abstract" not in call_args.kwargs
        ), "search_papers_keyword should NOT be called with 'abstract' parameter"

        # Check response
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "papers" in data
        assert "count" in data
        assert "query" in data
        assert data["query"] == "test"

    def test_search_with_limit(self, client):
        """Test that limit parameter is passed correctly."""
        from unittest.mock import MagicMock, patch
        import sys

        # Get the actual app module (not the Flask app object)
        app_module = sys.modules["abstracts_explorer.web_ui.app"]

        # Create a mock database
        mock_db = MagicMock()
        mock_papers = [
            {"id": i, "name": f"Paper {i}", "abstract": "About transformers", "authors": []} for i in range(5)
        ]

        mock_db.search_papers_keyword.return_value = mock_papers[:2]  # Return limited results

        # Patch the get_database function
        with patch.object(app_module, "get_database", return_value=mock_db):
            # Search with limit
            response = client.post(
                "/api/search",
                data=json.dumps({"query": "transformers", "use_embeddings": False, "limit": 2}),
                content_type="application/json",
            )

        # Verify limit was passed
        call_args = mock_db.search_papers_keyword.call_args
        assert call_args.kwargs["limit"] == 2

        assert response.status_code == 200
        data = json.loads(response.data)

        # Should respect the limit
        assert len(data["papers"]) <= 2

    def test_search_response_format(self, client):
        """Test that search response has correct format."""
        response = client.post(
            "/api/search",
            data=json.dumps({"query": "test", "use_embeddings": False, "limit": 10}),
            content_type="application/json",
        )

        # Even if it fails, check response structure
        data = json.loads(response.data)

        if response.status_code == 200:
            assert "papers" in data
            assert "count" in data
            assert "query" in data
            assert "use_embeddings" in data
            assert isinstance(data["papers"], list)
            assert isinstance(data["count"], int)
        else:
            assert "error" in data

    def test_search_keyword_field_filter(self, client):
        """Test that keyword search supports field:"value" syntax."""
        from unittest.mock import MagicMock, patch
        import sys

        app_module = sys.modules["abstracts_explorer.web_ui.app"]

        mock_db = MagicMock()
        mock_papers = [
            {"id": 1, "name": "Test Paper", "abstract": "Test abstract", "uid": "test1", "authors": ["John Smith"]}
        ]
        mock_db.search_papers_keyword.return_value = mock_papers

        with patch.object(app_module, "get_database", return_value=mock_db):
            response = client.post(
                "/api/search",
                data=json.dumps({"query": 'authors:"John Smith"', "use_embeddings": False, "limit": 10}),
                content_type="application/json",
            )

        mock_db.search_papers_keyword.assert_called_once()
        call_args = mock_db.search_papers_keyword.call_args
        assert call_args.kwargs["query"] == 'authors:"John Smith"'

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["count"] == 1

    def test_search_semantic_field_filter_only(self, client):
        """Test semantic search with field-filter-only query bypasses embeddings."""
        from unittest.mock import MagicMock, patch
        import sys

        app_module = sys.modules["abstracts_explorer.web_ui.app"]

        mock_em = MagicMock()
        mock_db = MagicMock()
        mock_papers = [{"uid": "test1", "title": "Test Paper", "abstract": "Test", "authors": ["John Smith"]}]
        mock_em.search_papers_semantic.return_value = mock_papers

        with patch.object(app_module, "get_embeddings_manager", return_value=mock_em):
            with patch.object(app_module, "get_database", return_value=mock_db):
                response = client.post(
                    "/api/search",
                    data=json.dumps({"query": 'authors:"John Smith"', "use_embeddings": True, "limit": 10}),
                    content_type="application/json",
                )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["count"] == 1
        # Field-filter-only query should not trigger count_papers_within_distance
        mock_em.count_papers_within_distance.assert_not_called()
        # total_similar should not be in response for field-filter-only queries
        assert "total_similar" not in data


class TestExtractTopKeywords:
    """Tests for the extract_top_keywords helper function."""

    def test_returns_empty_list_for_empty_input(self):
        """Should return empty list when no papers are provided."""
        from abstracts_explorer.paper_utils import extract_top_keywords

        result = extract_top_keywords([])
        assert result == []

    def test_returns_empty_list_for_papers_without_text(self):
        """Should return empty list when all papers have no title or abstract."""
        from abstracts_explorer.paper_utils import extract_top_keywords

        result = extract_top_keywords([{"uid": "p1"}, {"uid": "p2"}])
        assert result == []

    def test_extracts_keywords_from_single_paper(self):
        """Should extract keywords from a single paper."""
        from abstracts_explorer.paper_utils import extract_top_keywords

        papers = [
            {
                "title": "Neural Network Optimization",
                "abstract": "We study gradient descent optimization for neural networks.",
            }
        ]
        result = extract_top_keywords(papers, n_keywords=3)
        assert isinstance(result, list)
        assert len(result) <= 3
        assert all(isinstance(kw, str) for kw in result)

    def test_extracts_keywords_from_multiple_papers(self):
        """Should extract keywords from multiple papers."""
        from abstracts_explorer.paper_utils import extract_top_keywords

        papers = [
            {
                "title": "Deep Learning for Vision",
                "abstract": "Convolutional neural networks for image classification.",
            },
            {"title": "Transformer Models", "abstract": "Attention mechanism in deep learning for NLP tasks."},
            {
                "title": "Reinforcement Learning",
                "abstract": "Policy gradient methods for deep reinforcement learning agents.",
            },
        ]
        result = extract_top_keywords(papers, n_keywords=5)
        assert isinstance(result, list)
        assert len(result) <= 5
        assert all(isinstance(kw, str) for kw in result)
        # All keywords should be multi-word phrases (bigrams or trigrams)
        assert all(len(kw.split()) >= 2 for kw in result)
        # "deep learning" should appear in keywords given its frequency
        all_kw = " ".join(result).lower()
        assert "deep learning" in all_kw

    def test_respects_n_keywords_parameter(self):
        """Should return at most n_keywords results."""
        from abstracts_explorer.paper_utils import extract_top_keywords

        papers = [
            {"title": "Neural Networks", "abstract": "Deep learning optimization with gradient descent algorithms."},
            {
                "title": "Convolutional Networks",
                "abstract": "Image recognition with convolutional neural network layers.",
            },
        ]
        for n in [1, 3, 5]:
            result = extract_top_keywords(papers, n_keywords=n)
            assert len(result) <= n

    def test_search_response_includes_related_topics(self, client):
        """Test that search response includes related_topics field."""
        from unittest.mock import MagicMock, patch
        import sys

        app_module = sys.modules["abstracts_explorer.web_ui.app"]

        mock_db = MagicMock()
        mock_papers = [
            {"uid": "p1", "title": "Deep Learning", "abstract": "Neural network optimization with gradient descent."},
            {
                "uid": "p2",
                "title": "Neural Networks",
                "abstract": "Convolutional deep learning for image classification.",
            },
        ]
        mock_db.search_papers_keyword.return_value = mock_papers

        with patch.object(app_module, "get_database", return_value=mock_db):
            response = client.post(
                "/api/search",
                data=json.dumps({"query": "deep learning", "use_embeddings": False, "limit": 10}),
                content_type="application/json",
            )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert "related_topics" in data
        assert isinstance(data["related_topics"], list)


class TestChatEndpoint:
    """Test the chat endpoint."""

    def test_chat_without_message(self, client):
        """Test chat endpoint with missing message."""
        response = client.post("/api/chat", data=json.dumps({}), content_type="application/json")
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_chat_with_empty_message(self, client):
        """Test chat endpoint with empty message."""
        response = client.post("/api/chat", data=json.dumps({"message": ""}), content_type="application/json")
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_chat_reset(self, client):
        """Test chat reset endpoint."""
        response = client.post("/api/chat/reset")
        # May fail if LM Studio not running, but should not crash
        assert response.status_code in [200, 500]


class TestPaperEndpoint:
    """Test the paper details endpoint."""

    def test_get_paper_invalid_id(self, client):
        """Test getting a paper with invalid ID."""
        response = client.get("/api/paper/99999")
        # Should return 404 or 500
        assert response.status_code in [404, 500]


class TestDatabaseSearchIntegration:
    """Test database search_papers method directly."""

    def test_search_papers_with_keyword(self, test_db):
        """Test search_papers with keyword parameter."""
        with test_db:
            results = test_db.search_papers(keyword="transformer")
            assert len(results) > 0
            # Should find papers with "transformer" in name or abstract

    def test_search_papers_limit(self, test_db):
        """Test search_papers limit parameter."""
        with test_db:
            results = test_db.search_papers(keyword="", limit=2)
            assert len(results) <= 2

    def test_search_papers_no_invalid_params(self, test_db):
        """
        Test that search_papers rejects invalid parameters.

        This ensures we don't accidentally pass invalid parameters
        that don't exist in the function signature.
        """
        with test_db:
            # These should raise TypeError if we try to pass invalid params
            with pytest.raises(TypeError):
                test_db.search_papers(title="test")

            with pytest.raises(TypeError):
                test_db.search_papers(abstract="test")

            with pytest.raises(TypeError):
                test_db.search_papers(invalid_param="test")

    def test_search_papers_valid_params_only(self, test_db):
        """Test that only valid parameters work."""
        with test_db:
            # These should work
            results = test_db.search_papers(keyword="attention")
            assert isinstance(results, list)

            results = test_db.search_papers(year=2017)
            assert isinstance(results, list)

            results = test_db.search_papers(limit=5)
            assert isinstance(results, list)


class TestErrorHandling:
    """Test error handling in web interface."""

    def test_404_handler(self, client):
        """Test 404 error handler."""
        response = client.get("/api/nonexistent")
        assert response.status_code == 404
        data = json.loads(response.data)
        assert "error" in data

    def test_search_json_error(self, client):
        """Test search with invalid JSON."""
        response = client.post("/api/search", data="invalid json", content_type="application/json")
        assert response.status_code in [400, 500]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ============================================================
# Tests from test_web_ui_unit.py
# ============================================================


class TestWebUISemanticSearchDetails:
    """Test semantic search result processing (lines 165-183)."""

    def test_semantic_search_transforms_chromadb_results(self):
        """Test that semantic search correctly transforms ChromaDB results to paper format."""
        # Import inside test to avoid import-time issues
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            # Mock the get_embeddings_manager function
            with patch("abstracts_explorer.web_ui.app.get_embeddings_manager") as mock_get_em:
                with patch("abstracts_explorer.web_ui.app.get_database") as mock_get_db:
                    # Setup mock embeddings manager
                    mock_em = Mock()
                    mock_db = Mock()

                    # Mock the search_papers_semantic method to return proper papers
                    mock_papers = [
                        {
                            "uid": "uid_123",
                            "title": "Test Paper 1",
                            "abstract": "Abstract 1",
                            "authors": ["Author A", "Author B"],
                            "similarity": 0.9,
                            "distance": 0.1,
                        },
                        {
                            "uid": "uid_456",
                            "title": "Test Paper 2",
                            "abstract": "Abstract 2",
                            "authors": ["Author C", "Author D"],
                            "similarity": 0.8,
                            "distance": 0.2,
                        },
                    ]
                    mock_em.search_papers_semantic.return_value = mock_papers
                    mock_em.count_papers_within_distance.return_value = 25
                    mock_get_em.return_value = mock_em
                    mock_get_db.return_value = mock_db

                    # Make request
                    response = client.post(
                        "/api/search",
                        json={"query": "test query", "use_embeddings": True, "limit": 5},
                    )

                    assert response.status_code == 200
                    data = response.get_json()

                    # Verify transformation happened
                    assert "papers" in data
                    assert len(data["papers"]) == 2

                    # Check similarity scores were added
                    assert "similarity" in data["papers"][0]
                    assert "similarity" in data["papers"][1]

                    # Verify similarity calculation (1 - distance)
                    assert data["papers"][0]["similarity"] == pytest.approx(0.9, 0.01)
                    assert data["papers"][1]["similarity"] == pytest.approx(0.8, 0.01)

                    # Verify 'uid' field is present
                    assert data["papers"][0]["uid"] == "uid_123"
                    assert data["papers"][1]["uid"] == "uid_456"

    def test_semantic_search_handles_empty_results(self):
        """Test semantic search with no results."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_embeddings_manager") as mock_get_em:
                with patch("abstracts_explorer.web_ui.app.get_database") as mock_get_db:
                    mock_em = Mock()
                    mock_db = Mock()
                    mock_em.search_papers_semantic.return_value = []
                    mock_em.count_papers_within_distance.return_value = 0
                    mock_get_em.return_value = mock_em
                    mock_get_db.return_value = mock_db

                    response = client.post(
                        "/api/search",
                        json={"query": "nonexistent", "use_embeddings": True, "limit": 5},
                    )

                    assert response.status_code == 200
                    data = response.get_json()
                    assert data["papers"] == []
                    assert data["count"] == 0

    def test_semantic_search_uses_module_default_distance_threshold(self):
        """Test that semantic search uses the module-level default distance threshold (1.2)."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_embeddings_manager") as mock_get_em:
                with patch("abstracts_explorer.web_ui.app.get_database") as mock_get_db:
                    with patch("abstracts_explorer.web_ui.app._SIMILAR_DISTANCE_THRESHOLD", 1.2):
                        mock_em = Mock()
                        mock_db = Mock()
                        mock_em.search_papers_semantic.return_value = []
                        mock_em.count_papers_within_distance.return_value = 0
                        mock_get_em.return_value = mock_em
                        mock_get_db.return_value = mock_db

                        response = client.post(
                            "/api/search",
                            json={"query": "transformers", "use_embeddings": True, "limit": 5},
                        )

                        assert response.status_code == 200
                        # Verify search_papers_semantic was called with the module default threshold
                        call_kwargs = mock_em.search_papers_semantic.call_args
                        assert call_kwargs.kwargs.get("distance_threshold") == 1.2

    def test_semantic_search_accepts_per_request_distance_threshold(self):
        """Test that a distance_threshold in the request body overrides the module default."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_embeddings_manager") as mock_get_em:
                with patch("abstracts_explorer.web_ui.app.get_database") as mock_get_db:
                    mock_em = Mock()
                    mock_db = Mock()
                    mock_em.search_papers_semantic.return_value = []
                    mock_em.count_papers_within_distance.return_value = 0
                    mock_get_em.return_value = mock_em
                    mock_get_db.return_value = mock_db

                    response = client.post(
                        "/api/search",
                        json={
                            "query": "transformers",
                            "use_embeddings": True,
                            "limit": 5,
                            "distance_threshold": 0.7,  # per-request override
                        },
                    )

                    assert response.status_code == 200
                    # Verify the per-request value was used, not the module default
                    call_kwargs = mock_em.search_papers_semantic.call_args
                    assert call_kwargs.kwargs.get("distance_threshold") == 0.7


class TestWebUIChatEndpointSuccess:
    """Test chat endpoint success paths (lines 219-227, 255-270)."""

    def test_chat_with_valid_message_success(self):
        """Test successful chat response."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_rag_chat") as mock_get_rag:
                with patch("abstracts_explorer.web_ui.app.get_config") as mock_get_config:
                    # Setup mocks
                    mock_config = Mock()
                    mock_config.max_context_papers = 3
                    mock_get_config.return_value = mock_config

                    mock_rag = Mock()
                    mock_rag.query.return_value = {
                        "response": "This is a test response",
                        "papers": [{"title": "Paper 1"}],
                        "metadata": {"n_papers": 1},
                    }
                    mock_get_rag.return_value = mock_rag

                    # Make request
                    response = client.post(
                        "/api/chat",
                        json={"message": "What is a transformer?"},
                    )

                    assert response.status_code == 200
                    data = response.get_json()

                    # Verify response structure (lines 255-260)
                    assert "response" in data
                    assert "message" in data
                    assert data["message"] == "What is a transformer?"

                    # Verify query was called
                    mock_rag.query.assert_called_once()

    def test_chat_with_custom_n_papers(self):
        """Test chat with custom n_papers parameter."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_rag_chat") as mock_get_rag:
                with patch("abstracts_explorer.web_ui.app.get_config") as mock_get_config:
                    mock_config = Mock()
                    mock_config.max_context_papers = 3
                    mock_get_config.return_value = mock_config

                    mock_rag = Mock()
                    mock_rag.query.return_value = {
                        "response": "Test",
                        "papers": [],
                        "metadata": {},
                    }
                    mock_get_rag.return_value = mock_rag

                    response = client.post(
                        "/api/chat",
                        json={"message": "Test question", "n_papers": 5},
                    )

                    assert response.status_code == 200
                    # Verify n_papers was used
                    call_kwargs = mock_rag.query.call_args
                    assert call_kwargs.args[0] == "Test question"
                    assert call_kwargs.kwargs["n_results"] == 5
                    assert call_kwargs.kwargs["metadata_filter"] is None
                    assert call_kwargs.kwargs["conferences"] is None
                    assert call_kwargs.kwargs["years"] is None

    def test_chat_with_reset_flag(self):
        """Test chat with reset=True (lines 249-253)."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_rag_chat") as mock_get_rag:
                with patch("abstracts_explorer.web_ui.app.get_config") as mock_get_config:
                    mock_config = Mock()
                    mock_config.max_context_papers = 3
                    mock_get_config.return_value = mock_config

                    mock_rag = Mock()
                    mock_rag.query.return_value = {
                        "response": "Test",
                        "papers": [],
                        "metadata": {},
                    }
                    mock_get_rag.return_value = mock_rag

                    response = client.post(
                        "/api/chat",
                        json={"message": "Test", "reset": True},
                    )

                    assert response.status_code == 200
                    # Verify reset_conversation was called
                    mock_rag.reset_conversation.assert_called_once()


class TestWebUIErrorHandlingDetails:
    """Test error handling paths (lines 287-288, 304-305)."""

    def test_chat_reset_exception_handling(self):
        """Test chat reset endpoint handles exceptions (lines 287-288)."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_rag_chat") as mock_get_rag:
                mock_rag = Mock()
                mock_rag.reset_conversation.side_effect = Exception("Reset failed")
                mock_get_rag.return_value = mock_rag

                response = client.post("/api/chat/reset")

                assert response.status_code == 500
                data = response.get_json()
                assert "error" in data
                assert "Reset failed" in data["error"]

    def test_stats_endpoint_exception_handling(self):
        """Test stats endpoint handles exceptions (lines 304-305)."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_database") as mock_get_db:
                mock_db = Mock()
                mock_db.get_stats.side_effect = Exception("Database error")
                mock_get_db.return_value = mock_db

                response = client.get("/api/stats")

                assert response.status_code == 500
                data = response.get_json()
                assert "error" in data
                assert "Database error" in data["error"]


class TestWebUIGetPaperDetails:
    """Test get_paper endpoint (lines 219-227)."""

    def test_get_paper_with_authors_list(self):
        """Test that paper details include authors as list.

        DatabaseManager.get_paper_by_uid() already returns authors as a list
        (via _paper_to_dict), so the endpoint returns them as-is.
        """
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_database") as mock_get_db:
                mock_db = Mock()

                # DatabaseManager.get_paper_by_uid returns authors already as a list
                paper_row = {
                    "uid": "test_uid_123",
                    "title": "Test Paper",
                    "abstract": "Test abstract",
                    "authors": ["Author 1", "Author 2"],
                    "session": "Poster Session 1",
                    "poster_position": "123",
                }
                mock_db.get_paper_by_uid.return_value = paper_row
                mock_get_db.return_value = mock_db

                # Use string UID (not integer ID)
                response = client.get("/api/paper/test_uid_123")

                assert response.status_code == 200
                data = response.get_json()

                # Verify authors are included as list (formatting done by DatabaseManager)
                assert "authors" in data
                assert data["authors"] == ["Author 1", "Author 2"]
                # Verify 'uid' field is present
                assert data["uid"] == "test_uid_123"


class TestWebUIStatsEndpoint:
    """Test stats endpoint details (line 319)."""

    def test_stats_returns_paper_count(self):
        """Test that stats endpoint returns paper count."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_database") as mock_get_db:
                mock_db = Mock()
                mock_db.get_stats.return_value = {
                    "total_papers": 42,
                    "year": None,
                    "conference": None,
                }
                mock_get_db.return_value = mock_db

                response = client.get("/api/stats")

                assert response.status_code == 200
                data = response.get_json()

                assert "total_papers" in data
                assert data["total_papers"] == 42


class TestWebUISearchExceptionHandling:
    """Test search endpoint exception handling (line 195)."""

    def test_search_handles_database_exception(self):
        """Test that search endpoint handles database exceptions."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_database") as mock_get_db:
                mock_db = Mock()
                mock_db.search_papers.side_effect = Exception("Database connection failed")
                mock_db.get_paper_authors.return_value = []
                mock_get_db.return_value = mock_db

                response = client.post(
                    "/api/search",
                    json={"query": "test", "use_embeddings": False, "limit": 10},
                )

                assert response.status_code == 500
                data = response.get_json()
                assert "error" in data

    def test_search_handles_embeddings_exception(self):
        """Test that semantic search handles embeddings exceptions."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_embeddings_manager") as mock_get_em:
                mock_em = Mock()
                mock_em.search_similar.side_effect = Exception("Embeddings error")
                mock_get_em.return_value = mock_em

                response = client.post(
                    "/api/search",
                    json={"query": "test", "use_embeddings": True, "limit": 10},
                )

                assert response.status_code == 500
                data = response.get_json()
                assert "error" in data


class TestWebUIDatabaseNotFound:
    """Test database file not found handling for both local and URL databases."""

    def test_get_database_sqlite_url_not_found(self, tmp_path, monkeypatch):
        """Test that get_database works with SQLite database URL."""
        from abstracts_explorer.web_ui.app import app

        # Create a valid SQLite database
        db_path = tmp_path / "test.db"
        set_test_db(db_path)

        with app.test_client() as client:
            # Database will be created automatically, so /api/stats should work
            response = client.get("/api/stats")

            # Should succeed now that databases are auto-created
            assert response.status_code == 200
            data = response.get_json()
            assert "total_papers" in data

    def test_get_database_postgresql_url_connection_failed(self, monkeypatch):
        """Test that get_database uses fallback SQLite when PostgreSQL URL cannot connect."""
        from abstracts_explorer.web_ui.app import app
        from tests.conftest import get_env_test_path

        # Set invalid PostgreSQL URL (but no PAPER_DB, so it falls back to default SQLite)
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@nonexistent-host:5432/db")
        get_config(reload=True, env_path=get_env_test_path())

        with app.test_client() as client:
            # Try to access endpoint that uses database
            response = client.get("/api/stats")

            # Should succeed with fallback to SQLite (since PAPER_DB not set)
            # The config system falls back to default SQLite when DATABASE_URL doesn't work
            assert response.status_code == 200
            data = response.get_json()
            assert "total_papers" in data


class TestWebUIDatabaseModes:
    """Test database initialization with both local SQLite and database URL modes."""

    def test_get_database_with_sqlite_path(self, tmp_path):
        """Test that get_database works with local SQLite database path."""
        from abstracts_explorer.web_ui.app import app, get_database
        from abstracts_explorer.database import DatabaseManager

        # Create a real SQLite database
        db_path = tmp_path / "test.db"
        set_test_db(db_path)
        db = DatabaseManager()
        with db:
            db.create_tables()

        with app.test_client():
            with patch("abstracts_explorer.web_ui.app.get_config") as mock_get_config:
                mock_config = Mock()
                # Simulate PAPER_DB_PATH mode (legacy)
                mock_config.paper_db_path = str(db_path)
                mock_config.database_url = f"sqlite:///{str(db_path)}"
                mock_get_config.return_value = mock_config

                # Access database within app context
                with app.app_context():
                    database = get_database()

                    # Verify database is connected and functional
                    assert database is not None
                    assert database.database_url == f"sqlite:///{str(db_path)}"

                    # Verify we can query the database
                    count = database.get_paper_count()
                    assert count == 0  # Empty database

    def test_get_database_with_sqlite_url(self, tmp_path):
        """Test that get_database works with SQLite database URL (converted from path)."""
        from abstracts_explorer.web_ui.app import app, get_database
        from abstracts_explorer.database import DatabaseManager

        # Create a real SQLite database
        db_path = tmp_path / "test.db"
        set_test_db(db_path)
        db = DatabaseManager()
        with db:
            db.create_tables()

        with app.test_client():
            with patch("abstracts_explorer.web_ui.app.get_config") as mock_get_config:
                mock_config = Mock()
                # Simulate DATABASE_URL with SQLite
                mock_config.paper_db_path = ""  # Not used when DATABASE_URL is set
                mock_config.database_url = f"sqlite:///{str(db_path)}"
                mock_get_config.return_value = mock_config

                # Access database within app context
                with app.app_context():
                    database = get_database()

                    # Verify database is connected and functional
                    assert database is not None
                    assert database.database_url == f"sqlite:///{str(db_path)}"

                    # Verify we can query the database
                    count = database.get_paper_count()
                    assert count == 0  # Empty database

    def test_stats_endpoint_with_sqlite_database(self, tmp_path):
        """Test that stats endpoint works with local SQLite database."""
        from abstracts_explorer.web_ui.app import app
        from abstracts_explorer.database import DatabaseManager
        from abstracts_explorer.plugin import LightweightPaper

        # Create a real SQLite database with test data
        db_path = tmp_path / "test.db"
        set_test_db(db_path)
        db = DatabaseManager()
        with db:
            db.create_tables()
            # Add test papers
            paper = LightweightPaper(
                title="Test Paper",
                authors=["Author 1"],
                abstract="Test abstract",
                session="Session 1",
                poster_position="P1",
                year=2025,
                conference="Test",
            )
            db.add_paper(paper)

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_config") as mock_get_config:
                mock_config = Mock()
                mock_config.database_url = f"sqlite:///{str(db_path)}"
                mock_config.embedding_db_path = "chroma_db"
                mock_config.collection_name = "papers"
                mock_get_config.return_value = mock_config

                response = client.get("/api/stats")

                assert response.status_code == 200
                data = response.get_json()
                assert "total_papers" in data
                assert data["total_papers"] == 1


class TestWebUIChatExceptionLines:
    """Test specific exception handling lines in chat endpoint."""

    def test_chat_exception_returns_500(self):
        """Test that chat exceptions return 500 with error message (lines 269-270)."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_rag_chat") as mock_get_rag:
                with patch("abstracts_explorer.web_ui.app.get_config") as mock_get_config:
                    mock_config = Mock()
                    mock_config.max_context_papers = 3
                    mock_get_config.return_value = mock_config

                    mock_rag = Mock()
                    mock_rag.query.side_effect = Exception("RAG system error")
                    mock_get_rag.return_value = mock_rag

                    response = client.post(
                        "/api/chat",
                        json={"message": "test"},
                    )

                    assert response.status_code == 500
                    data = response.get_json()
                    assert "error" in data
                    assert "RAG system error" in data["error"]


class TestWebUIGetPaperException:
    """Test get_paper exception handling."""

    def test_get_paper_not_found_returns_404(self):
        """Test that missing paper returns 404."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_database") as mock_get_db:
                mock_db = Mock()
                mock_db.get_paper_by_uid.return_value = None  # Paper not found
                mock_get_db.return_value = mock_db

                response = client.get("/api/paper/999")

                assert response.status_code == 404
                data = response.get_json()
                assert "error" in data
                assert "not found" in data["error"].lower()

    def test_get_paper_database_error_returns_500(self):
        """Test that database exceptions return 500."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_database") as mock_get_db:
                mock_db = Mock()
                # Simulate a database exception
                mock_db.get_paper_by_uid.side_effect = RuntimeError("Database connection lost")
                mock_get_db.return_value = mock_db

                response = client.get("/api/paper/1")

                assert response.status_code == 500
                data = response.get_json()
                assert "error" in data


class TestWebUIStatsExceptionHandling:
    """Test stats endpoint exception handling more thoroughly."""

    def test_stats_paper_count_calculation(self):
        """Test that stats correctly calls get_stats (line 319)."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_database") as mock_get_db:
                mock_db = Mock()
                mock_db.get_stats.return_value = {
                    "total_papers": 100,
                    "year": None,
                    "conference": None,
                }
                mock_get_db.return_value = mock_db

                response = client.get("/api/stats")

                assert response.status_code == 200
                data = response.get_json()
                assert data["total_papers"] == 100

                # Verify get_stats was actually called
                mock_db.get_stats.assert_called_once()


class TestWebUIRunServer:
    """Test run_server function with Waitress production server."""

    def test_run_server_starts_waitress_server(self):
        """Test that run_server starts Waitress server by default."""
        from abstracts_explorer.web_ui import run_server

        with patch("abstracts_explorer.web_ui.app.os.path.exists", return_value=True):
            with patch("abstracts_explorer.web_ui.app.get_config") as mock_config:
                mock_cfg = Mock()
                mock_cfg.paper_db_path = "/some/path/test.db"
                mock_cfg.embedding_db_path = "/some/path/chroma_db"
                mock_config.return_value = mock_cfg

                # Mock Waitress serve with timeout
                with patch("waitress.serve") as mock_serve:
                    mock_serve.side_effect = KeyboardInterrupt()  # Simulate server stop

                    with pytest.raises(KeyboardInterrupt):
                        run_server(host="127.0.0.1", port=5000, debug=False)

                    # Verify Waitress serve was called with correct parameters
                    mock_serve.assert_called_once()
                    call_args = mock_serve.call_args
                    assert call_args[1]["host"] == "127.0.0.1"
                    assert call_args[1]["port"] == 5000

    def test_run_server_with_debug_mode(self):
        """Test run_server with debug=True uses Waitress with debug enabled."""
        from abstracts_explorer.web_ui import run_server, app

        with patch("abstracts_explorer.web_ui.app.os.path.exists", return_value=True):
            with patch("abstracts_explorer.web_ui.app.get_config") as mock_config:
                mock_cfg = Mock()
                mock_cfg.paper_db_path = "/some/path/test.db"
                mock_cfg.embedding_db_path = "/some/path/chroma_db"
                mock_config.return_value = mock_cfg

                # Mock Waitress serve with timeout (debug mode now works with Waitress)
                with patch("waitress.serve") as mock_serve:
                    mock_serve.side_effect = KeyboardInterrupt()  # Simulate server stop

                    with pytest.raises(KeyboardInterrupt):
                        run_server(host="0.0.0.0", port=8080, debug=True)

                    # Verify Waitress was called (debug mode works with production server)
                    mock_serve.assert_called_once()
                    call_args = mock_serve.call_args
                    assert call_args[1]["host"] == "0.0.0.0"
                    assert call_args[1]["port"] == 8080
                    # Verify Flask debug flag was set
                    assert app.debug

    def test_run_server_database_not_found(self, capsys):
        """Test that run_server raises FileNotFoundError when database doesn't exist."""
        from abstracts_explorer.web_ui import run_server

        # Set non-existent database path
        nonexistent_path = "/nonexistent/test.db"
        set_test_db(nonexistent_path)

        with patch("abstracts_explorer.web_ui.app.os.path.exists", return_value=False):
            with pytest.raises(FileNotFoundError) as exc_info:
                run_server(host="127.0.0.1", port=5000, debug=False)

            # Verify error message includes database filename (platform-independent check)
            # On Unix: "/nonexistent/test.db", on Windows: "D:\nonexistent\test.db"
            assert "test.db" in str(exc_info.value)

        # Verify helpful error message was printed
        captured = capsys.readouterr()
        assert "Database not found" in captured.err
        assert "neurips-abstracts download" in captured.err
        assert "create-embeddings" in captured.err

    def test_run_server_registers_sigterm_handler(self):
        """Test that run_server registers a SIGTERM handler for graceful shutdown."""
        import signal as signal_module
        from abstracts_explorer.web_ui import run_server

        with patch("abstracts_explorer.web_ui.app.os.path.exists", return_value=True):
            with patch("abstracts_explorer.web_ui.app.get_config") as mock_config:
                mock_cfg = Mock()
                mock_cfg.paper_db_path = "/some/path/test.db"
                mock_cfg.embedding_db_path = "/some/path/chroma_db"
                mock_config.return_value = mock_cfg

                sigterm_handler = None

                def capture_signal(signum, handler):
                    nonlocal sigterm_handler
                    if signum == signal_module.SIGTERM:
                        sigterm_handler = handler

                with patch("abstracts_explorer.web_ui.app.signal.signal", side_effect=capture_signal):
                    with patch("waitress.serve") as mock_serve:
                        mock_serve.side_effect = KeyboardInterrupt()

                        with pytest.raises(KeyboardInterrupt):
                            run_server(host="127.0.0.1", port=5000)

                # Verify a SIGTERM handler was registered
                assert sigterm_handler is not None, "SIGTERM handler was not registered"
                # Verify the handler raises SystemExit(0) when called
                with pytest.raises(SystemExit) as exc_info:
                    sigterm_handler(signal_module.SIGTERM, None)
                assert exc_info.value.code == 0


class TestConferencePluginMapping:
    """Test conference name to plugin name mapping in download endpoint."""

    def test_ml4ps_conference_name_maps_to_plugin(self):
        """Test that 'ML4PS@Neurips' conference name correctly maps to 'ml4ps' plugin."""
        from abstracts_explorer.plugins import get_plugin, list_plugins

        # Get all plugins and build mapping
        plugins = list_plugins()
        conference_to_plugin = {}
        for plugin_info in plugins:
            conf_name = plugin_info.get("conference_name")
            plug_name = plugin_info.get("name")
            if conf_name and plug_name:
                conference_to_plugin[conf_name] = plug_name

        # Verify ML4PS@Neurips maps to ml4ps
        assert "ML4PS@Neurips" in conference_to_plugin
        assert conference_to_plugin["ML4PS@Neurips"] == "ml4ps"

        # Verify we can get the plugin
        plugin_name = conference_to_plugin["ML4PS@Neurips"]
        plugin = get_plugin(plugin_name)
        assert plugin is not None
        assert plugin.plugin_name == "ml4ps"

    def test_all_conferences_map_to_plugins(self):
        """Test that all conference names can be mapped to valid plugins."""
        from abstracts_explorer.plugins import get_plugin, list_plugins

        # Get all plugins
        plugins = list_plugins()

        # Build mapping and verify each conference has a valid plugin
        for plugin_info in plugins:
            conf_name = plugin_info.get("conference_name")
            plug_name = plugin_info.get("name")

            if conf_name and plug_name:
                # Verify we can get the plugin
                plugin = get_plugin(plug_name)
                assert plugin is not None, f"Plugin '{plug_name}' for conference '{conf_name}' not found"

                # Verify conference name and plugin name match
                assert plugin.conference_name == conf_name


class TestServerInitialization:
    """Test server initialization with production and development modes."""

    def test_run_server_with_waitress_available(self, tmp_path):
        """Test that run_server uses Waitress by default when available."""
        from abstracts_explorer.web_ui.app import run_server
        from abstracts_explorer.database import DatabaseManager

        # Create a test database
        db_path = tmp_path / "test.db"
        set_test_db(db_path)
        db = DatabaseManager()
        with db:
            db.create_tables()

        # Mock Waitress serve - need to patch the import inside the function
        with patch("waitress.serve") as mock_serve:
            mock_serve.side_effect = KeyboardInterrupt()  # Simulate Ctrl+C

            with pytest.raises(KeyboardInterrupt):
                run_server(host="127.0.0.1", port=5000, debug=False, dev=False)

            # Verify Waitress was called
            mock_serve.assert_called_once()
            call_args = mock_serve.call_args
            assert call_args[1]["host"] == "127.0.0.1"
            assert call_args[1]["port"] == 5000

    def test_run_server_with_dev_flag(self, tmp_path):
        """Test that run_server uses Flask dev server when dev=True."""
        from abstracts_explorer.web_ui.app import run_server, app
        from abstracts_explorer.database import DatabaseManager

        # Create a test database
        db_path = tmp_path / "test.db"
        set_test_db(db_path)
        db = DatabaseManager()
        with db:
            db.create_tables()

        # Mock config to use our test database
        with patch("abstracts_explorer.web_ui.app.get_config") as mock_config:
            mock_config.return_value = Mock(database_url=f"sqlite:///{str(db_path)}", embedding_db_path="chroma_db")

            # Mock app.run
            with patch.object(app, "run") as mock_run:
                mock_run.side_effect = KeyboardInterrupt()  # Simulate Ctrl+C

                with pytest.raises(KeyboardInterrupt):
                    run_server(host="127.0.0.1", port=5000, debug=False, dev=True)

                # Verify Flask dev server was called
                mock_run.assert_called_once_with(host="127.0.0.1", port=5000, debug=False)

    def test_run_server_with_debug_flag(self, tmp_path):
        """Test that run_server uses Waitress with debug enabled when debug=True."""
        from abstracts_explorer.web_ui.app import run_server, app
        from abstracts_explorer.database import DatabaseManager

        # Create a test database
        db_path = tmp_path / "test.db"
        set_test_db(db_path)
        db = DatabaseManager()
        with db:
            db.create_tables()

        # Mock config to use our test database
        with patch("abstracts_explorer.web_ui.app.get_config") as mock_config:
            mock_config.return_value = Mock(database_url=f"sqlite:///{str(db_path)}", embedding_db_path="chroma_db")

            # Mock Waitress serve (debug mode now works with Waitress)
            with patch("waitress.serve") as mock_serve:
                mock_serve.side_effect = KeyboardInterrupt()  # Simulate Ctrl+C

                with pytest.raises(KeyboardInterrupt):
                    run_server(host="127.0.0.1", port=5000, debug=True, dev=False)

                # Verify Waitress was called (debug mode works with production server)
                mock_serve.assert_called_once()
                call_args = mock_serve.call_args
                assert call_args[1]["host"] == "127.0.0.1"
                assert call_args[1]["port"] == 5000
                # Verify Flask debug flag was set
                assert app.debug

    def test_run_server_waitress_not_available(self, tmp_path):
        """Test that run_server falls back to Flask when Waitress is not available."""
        from abstracts_explorer.web_ui.app import run_server, app
        from abstracts_explorer.database import DatabaseManager

        # Create a test database
        db_path = tmp_path / "test.db"
        set_test_db(db_path)
        db = DatabaseManager()
        with db:
            db.create_tables()

        # Mock config to use our test database
        with patch("abstracts_explorer.web_ui.app.get_config") as mock_config:
            mock_config.return_value = Mock(database_url=f"sqlite:///{str(db_path)}", embedding_db_path="chroma_db")

            # Mock waitress import to fail
            import builtins

            real_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "waitress":
                    raise ImportError("Waitress not installed")
                return real_import(name, *args, **kwargs)

            # Mock app.run
            with patch.object(app, "run") as mock_run:
                mock_run.side_effect = KeyboardInterrupt()

                with patch("builtins.__import__", side_effect=mock_import):
                    with pytest.raises(KeyboardInterrupt):
                        run_server(host="127.0.0.1", port=5000, debug=False, dev=False)

                # Verify Flask was called as fallback
                mock_run.assert_called_once_with(host="127.0.0.1", port=5000, debug=False)

    def test_run_server_missing_database(self, tmp_path):
        """Test that run_server raises FileNotFoundError when database is missing."""
        from abstracts_explorer.web_ui.app import run_server

        # Use a non-existent database path
        db_path = tmp_path / "nonexistent.db"
        set_test_db(db_path)

        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError, match="Database not found"):
            run_server(host="127.0.0.1", port=5000)

    def test_run_server_default_threads(self, tmp_path):
        """Test that run_server passes default threads=6 to waitress.serve."""
        from abstracts_explorer.web_ui.app import run_server
        from abstracts_explorer.database import DatabaseManager

        db_path = tmp_path / "test.db"
        set_test_db(db_path)
        db = DatabaseManager()
        with db:
            db.create_tables()

        with patch("waitress.serve") as mock_serve:
            mock_serve.side_effect = KeyboardInterrupt()

            with pytest.raises(KeyboardInterrupt):
                run_server(host="127.0.0.1", port=5000)

            mock_serve.assert_called_once()
            call_args = mock_serve.call_args
            assert call_args[1]["threads"] == 6

    def test_run_server_custom_threads(self, tmp_path):
        """Test that run_server forwards a custom threads value to waitress.serve."""
        from abstracts_explorer.web_ui.app import run_server
        from abstracts_explorer.database import DatabaseManager

        db_path = tmp_path / "test.db"
        set_test_db(db_path)
        db = DatabaseManager()
        with db:
            db.create_tables()

        with patch("waitress.serve") as mock_serve:
            mock_serve.side_effect = KeyboardInterrupt()

            with pytest.raises(KeyboardInterrupt):
                run_server(host="127.0.0.1", port=5000, threads=12)

            mock_serve.assert_called_once()
            call_args = mock_serve.call_args
            assert call_args[1]["threads"] == 12

    def test_run_server_invalid_threads_raises_value_error(self, tmp_path):
        """Test that run_server raises ValueError when threads < 1."""
        from abstracts_explorer.web_ui.app import run_server

        db_path = tmp_path / "test.db"
        set_test_db(db_path)

        with pytest.raises(ValueError, match="threads must be >= 1"):
            run_server(host="127.0.0.1", port=5000, threads=0)

        with pytest.raises(ValueError, match="threads must be >= 1"):
            run_server(host="127.0.0.1", port=5000, threads=-5)

    """Test clustering endpoints and caching functionality."""

    def test_clustering_colors_defined_in_javascript(self):
        """Test that the clustering visualization JavaScript has explicit color definitions."""
        # Read the JavaScript file
        # Check constants file for PLOTLY_COLORS
        constants_path = (
            Path(__file__).parent.parent
            / "src"
            / "abstracts_explorer"
            / "web_ui"
            / "static"
            / "modules"
            / "utils"
            / "constants.js"
        )
        with open(constants_path, "r", encoding="utf-8") as f:
            constants_content = f.read()

        # Check that Plotly colors are defined (using constant naming convention)
        assert "PLOTLY_COLORS = [" in constants_content, "Plotly colors should be explicitly defined in constants.js"

        # Check clustering module for color assignment
        clustering_path = (
            Path(__file__).parent.parent
            / "src"
            / "abstracts_explorer"
            / "web_ui"
            / "static"
            / "modules"
            / "clustering.js"
        )
        with open(clustering_path, "r", encoding="utf-8") as f:
            clustering_content = f.read()

        # Check that colors are explicitly assigned to markers
        assert "color: clusterColor" in clustering_content, "Cluster color should be explicitly assigned to markers"

        # Verify the Plotly Dark24 and Light24 color palettes are present
        # Check a few representative colors from both palettes
        dark24_sample_colors = ["#2E91E5", "#E15F99", "#1CA71C", "#FB0D0D"]
        light24_sample_colors = ["#83BCFF", "#FFC3E0", "#8DFFB7", "#FF8F8F"]

        for color in dark24_sample_colors + light24_sample_colors:
            assert color in constants_content, f"Plotly color {color} should be in the constants.js"

    def test_compute_clusters_creates_tables_on_first_connection(self, tmp_path):
        """Test that create_tables is called when connecting to database."""
        from abstracts_explorer.web_ui.app import app, get_database
        from abstracts_explorer.database import DatabaseManager
        from abstracts_explorer.plugin import LightweightPaper

        # Create a database with only old tables (simulate migration scenario)
        db_path = tmp_path / "test.db"
        set_test_db(db_path)
        db = DatabaseManager()
        with db:
            db.create_tables()
            # Add a test paper
            paper = LightweightPaper(
                title="Test Paper",
                authors=["Test Author"],
                abstract="Test abstract",
                session="Session 1",
                poster_position="P1",
                year=2025,
                conference="Test",
            )
            db.add_paper(paper)

        # Now test that get_database calls create_tables
        with app.test_client():
            with patch("abstracts_explorer.web_ui.app.get_config") as mock_config:
                mock_config.return_value = Mock(
                    database_url=f"sqlite:///{str(db_path)}", embedding_db_path="chroma_db", collection_name="papers"
                )

                # Mock create_tables to verify it's called
                with patch.object(DatabaseManager, "create_tables") as mock_create_tables:
                    # Access an endpoint that uses get_database
                    with app.app_context():
                        db = get_database()
                        # Verify create_tables was called
                        mock_create_tables.assert_called_once()

    def test_compute_clusters_returns_error_without_embeddings(self):
        """Test that compute_clusters returns error when embeddings manager fails."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_embeddings_manager") as mock_get_em:
                with patch("abstracts_explorer.web_ui.app.get_database") as mock_get_db:
                    with patch("abstracts_explorer.web_ui.app.get_config") as mock_config:
                        mock_config.return_value = Mock(embedding_model="test-model")
                        mock_get_db.return_value = Mock()
                        mock_em = Mock()
                        mock_em.collection = None
                        mock_get_em.return_value = mock_em

                        response = client.post(
                            "/api/clusters/compute", json={"reduction_method": "pca", "n_components": 2}
                        )

                        assert response.status_code == 500
                        data = response.get_json()
                        assert "error" in data

    def test_compute_clusters_uses_cache_when_available(self, tmp_path):
        """Test that compute_clusters uses cached results when available."""
        from abstracts_explorer.web_ui.app import app

        # Note: JSON serialization converts int keys to strings
        cached_results = {
            "points": [{"id": "test", "x": 1.0, "y": 2.0, "cluster": 0}],
            "statistics": {"n_clusters": 1, "total_papers": 1},
            "cluster_centers": {"0": {"x": 1.0, "y": 2.0}},
        }

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_database") as mock_get_db:
                with patch("abstracts_explorer.web_ui.app.get_config") as mock_config:
                    mock_config.return_value = Mock(embedding_model="test-model")
                    mock_db = Mock()
                    mock_db.get_clustering_cache.return_value = cached_results
                    mock_get_db.return_value = mock_db

                    response = client.post(
                        "/api/clusters/compute",
                        json={},
                    )

                    assert response.status_code == 200
                    data = response.get_json()
                    assert data == cached_results
                    # Verify cache was queried with fixed parameters
                    mock_db.get_clustering_cache.assert_called_once()
                    call_kwargs = mock_db.get_clustering_cache.call_args[1]
                    assert call_kwargs["clustering_method"] == "agglomerative"
                    assert call_kwargs["reduction_method"] == "tsne"
                    assert call_kwargs["n_components"] == 2
                    assert call_kwargs["n_clusters"] is None
                    assert call_kwargs["conference"] is None
                    assert call_kwargs["year"] is None

    def test_compute_clusters_returns_404_when_no_cache(self, tmp_path):
        """Test that compute_clusters returns 404 when no cached data is available."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_database") as mock_get_db:
                with patch("abstracts_explorer.web_ui.app.get_config") as mock_config:
                    mock_config.return_value = Mock(embedding_model="test-model")
                    mock_db = Mock()
                    mock_db.get_clustering_cache.return_value = None
                    mock_get_db.return_value = mock_db

                    response = client.post(
                        "/api/clusters/compute",
                        json={},
                    )

                    assert response.status_code == 404
                    data = response.get_json()
                    assert "error" in data
                    assert "pre-computed" in data["error"].lower() or "pre-generate" in data["error"].lower()

    def test_compute_clusters_includes_conference_year_in_cache_params(self):
        """Test that compute_clusters passes conference/year filters to cache lookup."""
        from abstracts_explorer.web_ui.app import app

        cached_results = {
            "points": [],
            "statistics": {"n_clusters": 0, "total_papers": 0},
        }

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_database") as mock_get_db:
                with patch("abstracts_explorer.web_ui.app.get_config") as mock_config:
                    mock_config.return_value = Mock(embedding_model="test-model")
                    mock_db = Mock()
                    mock_db.get_clustering_cache.return_value = cached_results
                    mock_get_db.return_value = mock_db

                    response = client.post(
                        "/api/clusters/compute",
                        json={
                            "conferences": ["NeurIPS"],
                            "years": [2024],
                        },
                    )

                    assert response.status_code == 200
                    call_kwargs = mock_db.get_clustering_cache.call_args[1]
                    assert call_kwargs["conference"] == "NeurIPS"
                    assert call_kwargs["year"] == 2024
                    # conferences/years should NOT be in clustering_params
                    assert "conferences" not in call_kwargs["clustering_params"]
                    assert "years" not in call_kwargs["clustering_params"]

    def test_get_default_cluster_count(self):
        """Test getting default cluster count based on embeddings."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_embeddings_manager") as mock_get_em:
                mock_em = Mock()
                mock_em.get_collection_stats.return_value = {"count": 250}
                mock_get_em.return_value = mock_em

                response = client.get("/api/clusters/default-count")

                assert response.status_code == 200
                data = response.get_json()

                assert "n_clusters" in data
                assert "n_papers" in data
                assert data["n_papers"] == 250
                # For 250 papers: max(2, min(50, 250 // 100)) = max(2, min(500, 2)) = max(2, 2) = 2
                assert data["n_clusters"] == 2

    def test_get_default_cluster_count_large_dataset(self):
        """Test default cluster count with large dataset."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_embeddings_manager") as mock_get_em:
                mock_em = Mock()
                mock_em.get_collection_stats.return_value = {"count": 100000}
                mock_get_em.return_value = mock_em

                response = client.get("/api/clusters/default-count")

                assert response.status_code == 200
                data = response.get_json()

                assert data["n_papers"] == 100000
                # For 100000 papers: max(2, min(50, 100000 // 100)) = max(2, min(1000, 500)) = 500
                assert data["n_clusters"] == 500

    def test_get_default_cluster_count_error(self):
        """Test error handling when getting default cluster count."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_embeddings_manager", side_effect=Exception("Test error")):
                response = client.get("/api/clusters/default-count")

                assert response.status_code == 500
                data = response.get_json()
                assert "error" in data

    def test_search_custom_cluster_success(self):
        """Test successful custom cluster search."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_embeddings_manager") as mock_get_em:
                with patch("abstracts_explorer.web_ui.app.get_database") as mock_get_db:
                    # Mock embeddings manager
                    mock_em = Mock()
                    mock_query_emb = [0.1] * 100  # Mock 100-dim embedding

                    # Mock the find_papers_within_distance method directly
                    mock_em.find_papers_within_distance.return_value = {
                        "query": "Uncertainty quantification",
                        "query_embedding": mock_query_emb,
                        "distance": 150,
                        "papers": [
                            {
                                "uid": "paper1",
                                "title": "Paper 1",
                                "abstract": "Abstract for paper1",
                                "year": 2025,
                                "authors": ["Author 1"],
                                "distance": 0.5,
                            },
                            {
                                "uid": "paper2",
                                "title": "Paper 2",
                                "abstract": "Abstract for paper2",
                                "year": 2025,
                                "authors": ["Author 2"],
                                "distance": 1.0,
                            },
                        ],
                        "count": 2,
                    }

                    mock_get_em.return_value = mock_em

                    # Mock database
                    mock_db = Mock()
                    mock_get_db.return_value = mock_db

                    response = client.post(
                        "/api/clusters/search", json={"query": "Uncertainty quantification", "distance": 150}
                    )

                    assert response.status_code == 200
                    data = response.get_json()

                    assert data["query"] == "Uncertainty quantification"
                    assert data["distance"] == 150
                    assert "query_embedding" in data
                    assert "papers" in data
                    assert data["count"] == 2

                    # Check papers are sorted by distance
                    if len(data["papers"]) > 1:
                        for i in range(len(data["papers"]) - 1):
                            assert data["papers"][i]["distance"] <= data["papers"][i + 1]["distance"]

    def test_search_custom_cluster_missing_query(self):
        """Test custom cluster search with missing query parameter."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            response = client.post("/api/clusters/search", json={"distance": 150})  # Missing query

            assert response.status_code == 400
            data = response.get_json()
            assert "error" in data
            assert "query" in data["error"].lower()

    def test_search_custom_cluster_default_distance(self):
        """Test custom cluster search uses default distance if not provided."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_embeddings_manager") as mock_get_em:
                with patch("abstracts_explorer.web_ui.app.get_database") as mock_get_db:
                    # Mock embeddings manager
                    mock_em = Mock()

                    # Mock find_papers_within_distance to return result with default distance
                    mock_em.find_papers_within_distance.return_value = {
                        "query": "Test query",
                        "query_embedding": [0.1] * 100,
                        "distance": 1.1,
                        "papers": [],
                        "count": 0,
                    }

                    mock_get_em.return_value = mock_em

                    mock_db = Mock()
                    mock_get_db.return_value = mock_db

                    response = client.post(
                        "/api/clusters/search", json={"query": "Test query"}  # No distance specified
                    )

                    assert response.status_code == 200
                    data = response.get_json()
                    assert data["distance"] == 1.1  # New default value
                    assert data["query"] == "Test query"

    def test_search_custom_cluster_no_embeddings(self):
        """Test custom cluster search when no embeddings exist."""
        from abstracts_explorer.web_ui.app import app
        from abstracts_explorer.embeddings import EmbeddingsError

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_embeddings_manager") as mock_get_em:
                with patch("abstracts_explorer.web_ui.app.get_database") as mock_get_db:
                    mock_em = Mock()

                    # Mock find_papers_within_distance to raise EmbeddingsError
                    mock_em.find_papers_within_distance.side_effect = EmbeddingsError("No papers in collection")

                    mock_get_em.return_value = mock_em

                    mock_db = Mock()
                    mock_get_db.return_value = mock_db

                    response = client.post("/api/clusters/search", json={"query": "Test query", "distance": 100})

                    # Returns 500 since EmbeddingsError is raised and caught as general exception
                    assert response.status_code == 500
                    data = response.get_json()
                    assert "error" in data

    def test_search_custom_cluster_error_handling(self):
        """Test error handling in custom cluster search."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_embeddings_manager", side_effect=Exception("Test error")):
                response = client.post("/api/clusters/search", json={"query": "Test query", "distance": 100})

                assert response.status_code == 500
                data = response.get_json()
                assert "error" in data


class TestDataDonationEndpoint:
    """Test the data donation endpoint."""

    def test_donate_data_success(self, tmp_path):
        """Test successful data donation."""
        from abstracts_explorer.web_ui.app import app
        from abstracts_explorer.plugin import LightweightPaper

        # Create a real test database
        db_path = tmp_path / "test_donate.db"
        set_test_db(str(db_path))

        db = DatabaseManager()
        with db:
            db.create_tables()

            # Add a test paper using LightweightPaper
            paper = LightweightPaper(
                title="Test Paper 1",
                authors=["Test Author"],
                abstract="Test abstract",
                session="Test Session",
                poster_position="P1",
                year=2025,
                conference="NeurIPS",
            )
            db.add_papers([paper])

        # Get the actual UID that was generated
        with db:
            all_papers = db.query("SELECT uid FROM papers LIMIT 1")
            paper_uid = all_papers[0]["uid"]

        paper_priorities = {paper_uid: {"priority": 5, "searchTerm": "machine learning"}}

        with app.test_client() as client:
            response = client.post(
                "/api/donate-data", json={"paperPriorities": paper_priorities}, content_type="application/json"
            )

            assert response.status_code == 200
            data = response.get_json()
            assert data["success"] is True
            assert data["count"] == 1
            assert "Successfully donated" in data["message"]

            # Verify the data was actually stored by creating a fresh database manager
            db_verify = DatabaseManager()
            with db_verify:
                # Query using the database manager's query method
                results = db_verify.query("SELECT * FROM validation_data")
                assert len(results) == 1
                assert results[0]["paper_uid"] == paper_uid
                assert results[0]["priority"] == 5
                assert results[0]["search_term"] == "machine learning"

    def test_donate_data_no_data(self, client):
        """Test data donation with no data provided."""
        response = client.post("/api/donate-data", json={}, content_type="application/json")

        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "No data provided" in data["error"]

    def test_donate_data_empty_priorities(self, client):
        """Test data donation with empty priorities."""
        response = client.post("/api/donate-data", json={"paperPriorities": {}}, content_type="application/json")

        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_donate_data_invalid_format(self, tmp_path):
        """Test data donation with invalid integer format (should be rejected)."""
        from abstracts_explorer.web_ui.app import app

        # Create a real test database
        db_path = tmp_path / "test_donate_invalid.db"
        set_test_db(str(db_path))

        db = DatabaseManager()
        with db:
            db.create_tables()

        # Old format: paper_id -> priority (integer) - should now be rejected
        paper_priorities = {"test1": 5, "test2": 3}

        with app.test_client() as client:
            response = client.post(
                "/api/donate-data", json={"paperPriorities": paper_priorities}, content_type="application/json"
            )

            # Should return 400 error for invalid format
            assert response.status_code == 400
            data = response.get_json()
            assert "error" in data
            assert "Invalid data format" in data["error"]

    def test_donate_data_database_error(self, client):
        """Test data donation with database error."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_database") as mock_get_db:
                mock_db = Mock()
                # Simulate engine creation failure
                mock_db.engine.side_effect = Exception("Database connection error")
                mock_get_db.return_value = mock_db

                response = client.post(
                    "/api/donate-data",
                    json={"paperPriorities": {"test1": {"priority": 5, "searchTerm": "test"}}},
                    content_type="application/json",
                )

                # Should catch the exception and return 500
                assert response.status_code == 500
                data = response.get_json()
                assert "error" in data


class TestChatDonationEndpoint:
    """Tests for the /api/donate-chat endpoint."""

    def test_donate_chat_success(self, tmp_path):
        """Test successful chat transcript donation."""
        from abstracts_explorer.web_ui.app import app

        db_path = tmp_path / "test_chat_donate.db"
        set_test_db(str(db_path))

        db = DatabaseManager()
        with db:
            db.create_tables()

        transcript = [
            {"role": "user", "text": "What papers discuss transformers?"},
            {"role": "assistant", "text": "Here are some relevant papers..."},
        ]

        with app.test_client() as client:
            response = client.post(
                "/api/donate-chat",
                json={"rating": "up", "transcript": transcript},
                content_type="application/json",
            )

            assert response.status_code == 200
            data = response.get_json()
            assert data["success"] is True
            assert "Thank you" in data["message"]
            assert "id" in data

    def test_donate_chat_thumbs_down(self, tmp_path):
        """Test chat donation with thumbs down rating."""
        from abstracts_explorer.web_ui.app import app

        db_path = tmp_path / "test_chat_donate_down.db"
        set_test_db(str(db_path))

        db = DatabaseManager()
        with db:
            db.create_tables()

        transcript = [{"role": "user", "text": "test"}, {"role": "assistant", "text": "response"}]

        with app.test_client() as client:
            response = client.post(
                "/api/donate-chat",
                json={"rating": "down", "transcript": transcript},
                content_type="application/json",
            )

            assert response.status_code == 200
            data = response.get_json()
            assert data["success"] is True

    def test_donate_chat_missing_rating(self, client):
        """Test chat donation with missing rating."""
        response = client.post(
            "/api/donate-chat",
            json={"transcript": [{"role": "user", "text": "test"}]},
            content_type="application/json",
        )

        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_donate_chat_missing_transcript(self, client):
        """Test chat donation with missing transcript."""
        response = client.post(
            "/api/donate-chat",
            json={"rating": "up"},
            content_type="application/json",
        )

        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_donate_chat_invalid_rating(self, tmp_path):
        """Test chat donation with invalid rating value."""
        from abstracts_explorer.web_ui.app import app

        db_path = tmp_path / "test_chat_donate_invalid.db"
        set_test_db(str(db_path))

        db = DatabaseManager()
        with db:
            db.create_tables()

        with app.test_client() as client:
            response = client.post(
                "/api/donate-chat",
                json={"rating": "neutral", "transcript": [{"role": "user", "text": "test"}]},
                content_type="application/json",
            )

            assert response.status_code == 400
            data = response.get_json()
            assert "error" in data
            assert "Invalid rating" in data["error"]

    def test_donate_chat_empty_body(self, client):
        """Test chat donation with empty body."""
        response = client.post(
            "/api/donate-chat",
            json={},
            content_type="application/json",
        )

        assert response.status_code == 400


class TestValidationDataModel:
    """Test the ValidationData database model."""

    def test_validation_data_creation(self, tmp_path):
        """Test creating a validation data entry."""
        from abstracts_explorer.db_models import ValidationData
        from abstracts_explorer.database import DatabaseManager
        from sqlalchemy.orm import Session

        # Create test database
        db_path = tmp_path / "test_validation.db"
        set_test_db(str(db_path))
        db = DatabaseManager()

        with db:
            db.create_tables()

            # Create session
            session = Session(db.engine)

            try:
                # Create validation data entry
                validation_entry = ValidationData(paper_uid="test123", priority=5, search_term="machine learning")
                session.add(validation_entry)
                session.commit()

                # Query back
                result = session.query(ValidationData).filter_by(paper_uid="test123").first()
                assert result is not None
                assert result.paper_uid == "test123"
                assert result.priority == 5
                assert result.search_term == "machine learning"
                assert result.donated_at is not None

            finally:
                session.close()

    def test_validation_data_repr(self):
        """Test ValidationData string representation."""
        from abstracts_explorer.db_models import ValidationData

        entry = ValidationData(paper_uid="test123", priority=5, search_term="test")
        entry.id = 1

        repr_str = repr(entry)
        assert "ValidationData" in repr_str
        assert "test123" in repr_str
        assert "5" in repr_str


# ============================================================
# Paper card display field tests
# ============================================================


class TestPaperCardDisplayFieldsUnit:
    """
    Unit tests verifying that API endpoints return correct fields for paper card display.

    Paper cards (formatPaperCard in paper-card.js) require specific fields:
    - uid: for star ratings and the paper details modal onclick handler
    - title: displayed as the card heading
    - authors: must be a list (not a semicolon-separated string)
    - conference: shown as indigo badge on every card
    - abstract: shown in the card body

    The showPaperDetails() modal also requires uid, title, authors (list), conference, abstract.
    """

    # Sample papers that represent what the DB returns after author parsing
    _SAMPLE_PAPERS = [
        {
            "uid": "paper-uid-1",
            "title": "Attention is All You Need",
            "authors": ["Ashish Vaswani", "Noam Shazeer"],
            "abstract": "We propose the Transformer architecture.",
            "conference": "NeurIPS",
            "session": "Oral Session 1",
            "poster_position": "O1",
            "year": 2017,
            "url": "https://papers.nips.cc/paper/1",
        },
        {
            "uid": "paper-uid-2",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "authors": ["Jacob Devlin", "Ming-Wei Chang"],
            "abstract": "We introduce BERT, a new language representation model.",
            "conference": "NeurIPS",
            "session": "Poster Session 2",
            "poster_position": "P42",
            "year": 2019,
            "url": "https://papers.nips.cc/paper/2",
        },
    ]

    def _make_mock_db(self, papers=None):
        """Create a mock DatabaseManager that returns sample papers."""
        mock_db = MagicMock()
        if papers is None:
            papers = self._SAMPLE_PAPERS
        mock_db.search_papers_keyword.return_value = papers
        return mock_db

    @pytest.fixture
    def app_client(self):
        """Create a Flask test client and return (client, app_module) tuple."""
        app_module = sys.modules["abstracts_explorer.web_ui.app"]
        flask_app.config["TESTING"] = True
        client = flask_app.test_client()
        return client, app_module

    def test_keyword_search_includes_conference_field(self, app_client):
        """
        Keyword search results must include the 'conference' field.

        The conference badge in paper cards is only shown when paper.conference is set.
        """
        client, app_module = app_client
        mock_db = self._make_mock_db()

        with patch.object(app_module, "get_database", return_value=mock_db):
            response = client.post(
                "/api/search",
                json={"query": "attention", "use_embeddings": False, "limit": 5},
            )

        assert response.status_code == 200
        data = response.get_json()
        assert "papers" in data
        assert len(data["papers"]) > 0

        for paper in data["papers"]:
            assert (
                "conference" in paper
            ), f"Paper '{paper.get('title')}' missing 'conference' field needed for paper card badge"
            assert paper["conference"] == "NeurIPS"

    def test_keyword_search_authors_is_list(self, app_client):
        """
        Keyword search results must return 'authors' as a list.

        formatPaperCard() calls paper.authors.join(', ') which requires a list.
        A string would cause TypeError: 'authors must be an array'.
        """
        client, app_module = app_client
        mock_db = self._make_mock_db()

        with patch.object(app_module, "get_database", return_value=mock_db):
            response = client.post(
                "/api/search",
                json={"query": "attention", "use_embeddings": False, "limit": 5},
            )

        assert response.status_code == 200
        data = response.get_json()

        for paper in data["papers"]:
            authors = paper.get("authors")
            assert isinstance(authors, list), (
                f"Paper '{paper.get('title')}' has authors as "
                f"{type(authors).__name__}, must be list for paper card"
            )

    def test_keyword_search_authors_no_semicolons(self, app_client):
        """
        Authors from keyword search must not contain semicolons.

        The database stores authors as semicolon-separated strings.
        The API must split them into individual author names.
        """
        client, app_module = app_client

        # Return a paper with authors as already-parsed list (as search_papers_keyword returns)
        papers_with_authors = [
            {**self._SAMPLE_PAPERS[0], "authors": ["Alice Smith", "Bob Jones"]},
        ]
        mock_db = self._make_mock_db(papers=papers_with_authors)

        with patch.object(app_module, "get_database", return_value=mock_db):
            response = client.post(
                "/api/search",
                json={"query": "attention", "use_embeddings": False, "limit": 5},
            )

        assert response.status_code == 200
        data = response.get_json()

        for paper in data["papers"]:
            for author in paper.get("authors", []):
                assert ";" not in author, f"Author '{author}' contains semicolons; authors were not split correctly"

    def test_paper_detail_includes_conference_field(self, app_client):
        """
        The paper detail endpoint must include 'conference' for the modal badge.

        The showPaperDetails() modal renders a conference badge using paper.conference.
        """
        client, app_module = app_client

        expected_paper = {
            "uid": "detail-uid-1",
            "title": "Test Paper for Detail",
            "authors": ["Author One", "Author Two"],
            "abstract": "Test abstract text.",
            "conference": "NeurIPS",
            "session": "Oral Session 1",
            "poster_position": "O1",
            "year": 2024,
            "url": "https://papers.nips.cc/paper/detail",
        }

        with patch.object(app_module, "get_database") as mock_get_db:
            mock_db = Mock()
            mock_db.get_paper_by_uid.return_value = expected_paper
            mock_get_db.return_value = mock_db

            response = client.get("/api/paper/detail-uid-1")

        assert response.status_code == 200
        paper = response.get_json()

        assert "conference" in paper, "Paper detail modal requires 'conference' field"
        assert paper["conference"] == "NeurIPS"

    def test_paper_detail_authors_is_list(self, app_client):
        """
        The paper detail endpoint must return authors as a list.

        showPaperDetails() joins authors with ', ' which requires a list.
        """
        client, app_module = app_client

        expected_paper = {
            "uid": "detail-uid-2",
            "title": "Test Paper for Author Check",
            "authors": ["First Author", "Second Author", "Third Author"],
            "abstract": "Abstract text.",
            "conference": "ICML",
            "year": 2025,
        }

        with patch.object(app_module, "get_database") as mock_get_db:
            mock_db = Mock()
            mock_db.get_paper_by_uid.return_value = expected_paper
            mock_get_db.return_value = mock_db

            response = client.get("/api/paper/detail-uid-2")

        assert response.status_code == 200
        paper = response.get_json()

        authors = paper.get("authors")
        assert isinstance(authors, list), f"Paper detail modal requires authors as list, got {type(authors).__name__}"
        assert authors == ["First Author", "Second Author", "Third Author"]

    def test_batch_endpoint_includes_conference_field(self, app_client):
        """
        The /api/papers/batch endpoint must include 'conference' for paper cards.

        The interesting papers tab and count filter use this endpoint to
        render paper cards with the conference badge.
        """
        client, app_module = app_client

        batch_papers = [
            {
                "uid": "batch-uid-1",
                "title": "Batch Paper 1",
                "authors": ["Author A"],
                "abstract": "Abstract A",
                "conference": "NeurIPS",
                "year": 2024,
            },
            {
                "uid": "batch-uid-2",
                "title": "Batch Paper 2",
                "authors": ["Author B", "Author C"],
                "abstract": "Abstract B",
                "conference": "ICLR",
                "year": 2024,
            },
        ]

        paper_map = {p["uid"]: p for p in batch_papers}

        with patch.object(app_module, "get_database") as mock_get_db:
            mock_db = Mock()
            mock_db.get_paper_by_uid.side_effect = lambda uid: paper_map.get(uid)
            mock_get_db.return_value = mock_db

            response = client.post(
                "/api/papers/batch",
                json={"paper_ids": ["batch-uid-1", "batch-uid-2"]},
            )

        assert response.status_code == 200
        data = response.get_json()
        assert "papers" in data
        assert len(data["papers"]) == 2

        for paper in data["papers"]:
            assert "conference" in paper, "Batch endpoint must return 'conference' for paper card badge"
            assert isinstance(
                paper.get("authors"), list
            ), f"Batch endpoint must return authors as list, got {type(paper.get('authors')).__name__}"


class TestPapersPerYearEndpoint:
    """Test /api/papers-per-year endpoint."""

    def test_papers_per_year_returns_counts(self):
        """Test that papers-per-year returns year counts."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_database") as mock_get_db:
                mock_db = Mock()
                mock_db.get_years_for_conference.return_value = [2023, 2024]
                mock_db.get_stats.side_effect = [
                    {"total_papers": 100, "year": 2023, "conference": "NeurIPS"},
                    {"total_papers": 150, "year": 2024, "conference": "NeurIPS"},
                ]
                mock_get_db.return_value = mock_db

                response = client.get("/api/papers-per-year?conference=NeurIPS")

                assert response.status_code == 200
                data = response.get_json()
                assert "year_counts" in data
                assert data["year_counts"]["2023"] == 100
                assert data["year_counts"]["2024"] == 150
                assert data["conference"] == "NeurIPS"

    def test_papers_per_year_no_conference(self):
        """Test papers-per-year without conference filter."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_database") as mock_get_db:
                mock_db = Mock()
                mock_db.get_years.return_value = [2024, 2023]
                mock_db.get_stats.side_effect = [
                    {"total_papers": 200, "year": 2023, "conference": None},
                    {"total_papers": 300, "year": 2024, "conference": None},
                ]
                mock_get_db.return_value = mock_db

                response = client.get("/api/papers-per-year")

                assert response.status_code == 200
                data = response.get_json()
                assert "year_counts" in data
                assert data["conference"] is None

    def test_papers_per_year_error_handling(self):
        """Test papers-per-year handles errors."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.get_database") as mock_get_db:
                mock_get_db.side_effect = Exception("DB error")

                response = client.get("/api/papers-per-year")

                assert response.status_code == 500
                data = response.get_json()
                assert "error" in data


class TestTopicEvolutionEndpoint:
    """Test /api/topic-evolution endpoint."""

    def test_topic_evolution_returns_data(self):
        """Test that topic-evolution returns evolution data."""
        from abstracts_explorer.web_ui.app import app

        mock_result = {
            "topic": "transformers",
            "conferences": ["NeurIPS"],
            "distance_threshold": 1.1,
            "total_papers": 10,
            "year_range": {"start": 2022, "end": 2024},
            "conference_data": {
                "NeurIPS": {
                    "year_counts": {"2022": 3, "2023": 5, "2024": 7},
                    "year_relative": {"2022": 1.5, "2023": 2.5, "2024": 3.5},
                    "year_totals": {"2022": 200, "2023": 200, "2024": 200},
                }
            },
        }

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.json") as mock_json:
                mock_json.loads.return_value = mock_result
                with patch(
                    "abstracts_explorer.mcp_server.get_topic_evolution",
                    return_value='{"topic": "transformers"}',
                ):
                    response = client.post(
                        "/api/topic-evolution",
                        json={"topic_keywords": "transformers", "conferences": ["NeurIPS"]},
                    )

                    assert response.status_code == 200

    def test_topic_evolution_missing_keywords(self):
        """Test topic-evolution returns 400 for missing topic_keywords."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            response = client.post(
                "/api/topic-evolution",
                json={"conferences": ["NeurIPS"]},
            )

            assert response.status_code == 400
            data = response.get_json()
            assert "error" in data

    def test_topic_evolution_error_handling(self):
        """Test topic-evolution handles errors."""
        from abstracts_explorer.web_ui.app import app

        with app.test_client() as client:
            with patch("abstracts_explorer.web_ui.app.json") as mock_json:
                mock_json.loads.side_effect = Exception("Parse error")
                with patch(
                    "abstracts_explorer.mcp_server.get_topic_evolution",
                    return_value="{}",
                ):
                    response = client.post(
                        "/api/topic-evolution",
                        json={"topic_keywords": "test"},
                    )

                    assert response.status_code == 500
                    data = response.get_json()
                    assert "error" in data

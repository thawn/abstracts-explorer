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


SAMPLE_HTML_2026 = """
<html><body>
<article class="program-card"
    data-program-card="true"
    data-paper-id="paper-1"
    data-type="full"
    data-title="lowercase title"
    data-authors="alice smith,bob jones"
    data-keywords-label="Visualization, LLM agents"
    data-keywords="visualization, llm agents"
    data-session="me, myself, and ai"
    data-event="vis full papers"
    data-abstract="A full paper abstract."
    data-award="best paper"
    data-start="2026-11-09T14:00:00.000Z">
  <a class="program-card__title-link" href="/year/2026/program/content/paper-1">Proper Title One</a>
  <a class="program-card__person-link" href="/year/2026/program/content/?filter=author&amp;search=Alice">Alice Smith</a>
  <a class="program-card__person-link" href="/year/2026/program/content/?filter=author&amp;search=Bob">Bob Jones</a>
  <a class="program-card__person-link" href="/year/2026/program/content/?filter=author&amp;search=Başak">Başak Oral</a>
  <a class="program-card__action-link" href="/year/2026/program/session/v-full-1">
    Mon, Nov 9, 09:00 - Me, Myself, and AI
  </a>
</article>
<article class="program-card"
    data-program-card="true"
    data-paper-id="paper-2"
    data-type="workshop"
    data-title="workshop title"
    data-authors="carol lee"
    data-keywords-label="No keywords listed"
    data-session="workshop session"
    data-event="some workshop"
    data-abstract="Workshop abstract."
    data-start="2026-11-10T10:00:00.000Z">
  <a class="program-card__title-link" href="/year/2026/program/content/paper-2">Workshop Paper</a>
  <a class="program-card__person-link" href="/year/2026/program/content/?filter=author&amp;search=Carol">Carol Lee</a>
  <a class="program-card__action-link" href="/year/2026/program/session/v-ws-1">
    Tue, Nov 10, 10:00 - Workshop Session
  </a>
</article>
<article class="program-card"
    data-program-card="true"
    data-paper-id="paper-3"
    data-type="poster"
    data-title="poster without abstract"
    data-authors="dave kim"
    data-abstract=""
    data-session="poster session"
    data-event="vis posters">
  <a class="program-card__title-link" href="/year/2026/program/content/paper-3">Poster Without Abstract</a>
  <a class="program-card__person-link" href="/year/2026/program/content/?filter=author&amp;search=Dave">Dave Kim</a>
</article>
<article class="program-card"
    data-program-card="true"
    data-paper-id="paper-4"
    data-type="full"
    data-authors="erin ng"
    data-abstract="An abstract with no title."
    data-session="session"
    data-event="vis full papers">
  <span class="program-card__title-link"></span>
  <a class="program-card__person-link" href="/year/2026/program/content/?filter=author&amp;search=Erin">Erin Ng</a>
</article>
<article class="program-card" data-session-role="Chair" data-session-id="session-1">
  <span class="program-card__person-link">Session Chair</span>
</article>
</body></html>
"""


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

    @pytest.fixture(autouse=True)
    def _mock_current_year_probe(self, monkeypatch):
        """Avoid live IEEE VIS requests while testing metadata."""
        monkeypatch.setattr(
            IEEEVISDownloaderPlugin,
            "_check_current_year_available",
            lambda self, year: True,
        )

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

    def test_supported_years_contains_2026(self):
        plugin = IEEEVISDownloaderPlugin()
        assert 2026 in plugin.supported_years

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

    def test_get_url_2026_uses_html_program(self):
        plugin = IEEEVISDownloaderPlugin()
        url = plugin.get_url(2026)
        assert url == "https://ieeevis.org/year/2026/program/content"


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

    def test_explicit_url_takes_precedence_over_doi(self):
        plugin = IEEEVISDownloaderPlugin()
        item = dict(SAMPLE_PAPER_1)
        item["url"] = "https://ieeevis.org/year/2026/program/content/paper-1"
        paper = plugin._convert_paper(item, 2026)
        assert paper is not None
        assert paper.url == "https://ieeevis.org/year/2026/program/content/paper-1"

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
# HTML program tests (2026+)
# ---------------------------------------------------------------------------


class TestIEEEVISHTMLProgram:
    """Test the 2026+ HTML program parser and download path."""

    def test_parse_program_html_includes_title_and_abstract(self):
        plugin = IEEEVISDownloaderPlugin()
        papers = plugin._parse_program_html(SAMPLE_HTML_2026, 2026)
        assert len(papers) == 2

        by_title = {paper.title: paper for paper in papers}
        assert "Proper Title One" in by_title
        assert "Workshop Paper" in by_title
        assert "Poster Without Abstract" not in by_title

        first = by_title["Proper Title One"]
        assert first.authors == ["Alice Smith", "Bob Jones", "Başak Oral"]
        assert first.abstract == "A full paper abstract."
        assert first.keywords == ["Visualization", "LLM agents"]
        assert first.session == "Me, Myself, and AI"
        assert first.award == "Best Paper"
        assert first.starttime == "2026-11-09T14:00:00.000Z"
        assert first.url == "https://ieeevis.org/year/2026/program/content/paper-1"
        assert first.year == 2026
        assert first.conference == "IEEE VIS"

        workshop = by_title["Workshop Paper"]
        assert workshop.keywords is None
        assert workshop.award is None
        assert workshop.session == "Workshop Session"

    def test_parse_program_html_skips_missing_title_or_abstract(self):
        plugin = IEEEVISDownloaderPlugin()
        papers = plugin._parse_program_html(SAMPLE_HTML_2026, 2026)
        assert all(paper.title for paper in papers)
        assert all(paper.abstract for paper in papers)
        assert len(papers) == 2

    @patch("abstracts_explorer.plugins.ieeevis_downloader.requests.get")
    def test_download_2026_uses_html_program(self, mock_get):
        mock_response = Mock()
        mock_response.text = SAMPLE_HTML_2026
        mock_get.return_value = mock_response

        plugin = IEEEVISDownloaderPlugin()
        with patch.object(plugin, "_check_current_year_available", return_value=True):
            papers = plugin.download(year=2026)

        assert mock_get.call_args[0][0] == "https://ieeevis.org/year/2026/program/content"
        assert len(papers) == 2
        assert all(paper.year == 2026 for paper in papers)

    @patch("abstracts_explorer.plugins.ieeevis_downloader.requests.get")
    def test_download_2026_decodes_utf8(self, mock_get):
        response = requests.Response()
        response.status_code = 200
        response._content = SAMPLE_HTML_2026.encode("utf-8")
        response.headers["Content-Type"] = "text/html"
        response.encoding = "ISO-8859-1"
        mock_get.return_value = response

        plugin = IEEEVISDownloaderPlugin()
        with patch.object(plugin, "_check_current_year_available", return_value=True):
            papers = plugin.download(year=2026)

        by_title = {paper.title: paper for paper in papers}
        assert "Başak Oral" in by_title["Proper Title One"].authors

    @patch("abstracts_explorer.plugins.ieeevis_downloader.requests.get")
    def test_download_2026_saves_to_file(self, mock_get):
        mock_response = Mock()
        mock_response.text = SAMPLE_HTML_2026
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "ieeevis_2026.json"
            plugin = IEEEVISDownloaderPlugin()
            with patch.object(plugin, "_check_current_year_available", return_value=True):
                plugin.download(year=2026, output_path=str(output_path))

            assert output_path.exists()
            saved = json.loads(output_path.read_text())
            assert len(saved) == 2
            assert all(paper["year"] == 2026 for paper in saved)
            assert all(paper["conference"] == "IEEE VIS" for paper in saved)

    @patch("abstracts_explorer.plugins.ieeevis_downloader.requests.get")
    def test_download_2026_request_exception(self, mock_get):
        mock_get.side_effect = requests.exceptions.RequestException("Connection error")

        plugin = IEEEVISDownloaderPlugin()
        with patch.object(plugin, "_check_current_year_available", return_value=True):
            with pytest.raises(RuntimeError, match="Failed to download"):
                plugin.download(year=2026)

    def test_parse_program_html_empty_page(self):
        plugin = IEEEVISDownloaderPlugin()
        assert plugin._parse_program_html("<html><body></body></html>", 2026) == []


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

                rows = db._query("SELECT uid, title, abstract, year, conference, authors FROM papers")
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

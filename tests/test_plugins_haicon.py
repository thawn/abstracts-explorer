"""
Tests for HAICON Downloader Plugin
=====================================

Test suite for the HAICON (Helmholtz AI Conference) data downloader plugin.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
import requests

from abstracts_explorer.plugins.haicon_downloader import HAICONDownloaderPlugin
from abstracts_explorer.plugins import (
    LightweightDownloaderPlugin,
    DownloaderPlugin,
    get_plugin,
    list_plugin_names,
)

# ============================================================================
# Sample HTML fixtures
# ============================================================================


SESSIONS_OVERVIEW_HTML = """
<!DOCTYPE html>
<html>
<body>
<a href="index.php?page=browseSessions&form_date=all&form_session=8&presentations=show">Registration</a>
<a href="index.php?page=browseSessions&form_date=all&form_session=14&presentations=show">Session 1a</a>
<a href="index.php?page=browseSessions&form_date=all&form_session=19&presentations=show">Poster Session I</a>
<a href="index.php?page=browseSessions&form_date=all&form_session=21&presentations=show">Session 1b</a>
</body>
</html>
"""

SESSION_19_HTML = """
<!DOCTYPE html>
<html>
<body>
<span class='font11'><b>Poster Session I</b></span>
<div id='paperID358'>
<div class="paper">
<span class="paper_id">ID: 358</span>
<span class="paper_session"> / Poster No. # 9: 001</span><br />
<span class="paper_label">Modalities: </span><span class="paper_topics">Image</span><br>
<span class="paper_label">Methods: </span><span class="paper_topics">Foundation Models, Other</span><br>
<span class="paper_label">Application Domain: </span><span class="paper_topics">Health</span><br>
<p class="paper_title">Interpretable Representations for Hematology</p>
<p class="paper_author"> <u>Muhammed Furkan Dasdelen</u><sup>1</sup>, Hyesu Lim<sup>1</sup>, Paul Pommer<sup>1</sup>, Michele Buck<sup>2</sup>, Steffen Schneider<sup>1</sup>, Carsten Marr<sup>1</sup></p>
<p class="paper_organisation"><sup>1</sup>Helmholtz Munich, Germany; <sup>2</sup>Medical Department for Hematology and Oncology, Technical University Munich</p>
<div ><div class="paper">
<p class="paper_abstract">Sparse autoencoders (SAEs) emerged as a promising tool for mechanistic interpretability.</p>
<p class="paper_abstract">In this work, we show the applicability of SAEs for hematology.</p>
</div></div>
</div>
</div>
<hr>
<div id='paperID255'>
<div class="paper">
<span class="paper_id">ID: 255</span>
<span class="paper_session"> / Poster No. # 9: 002</span><br />
<span class="paper_label">Modalities: </span><span class="paper_topics">Graphs, Multimodal Data</span><br>
<span class="paper_label">Methods: </span><span class="paper_topics">Graph Neural Networks</span><br>
<span class="paper_label">Application Domain: </span><span class="paper_topics">Health</span><br>
<p class="paper_title">Analysis of multi-omics data using graph neural networks</p>
<p class="paper_author"> <u>Svitlana Oleshko</u><sup>1,2</sup>, Samuele Firmani<sup>1</sup>, Matthias Heinig<sup>1</sup></p>
<p class="paper_organisation"><sup>1</sup>Helmholtz Munich, Germany; <sup>2</sup>Boehringer Ingelheim Pharma, Germany</p>
<div ><div class="paper">
<p class="paper_abstract">Identifying novel therapeutic targets for psychiatric disorders requires understanding altered molecular neurobiology.</p>
</div></div>
</div>
</div>
</body>
</html>
"""

SESSION_14_HTML = """
<!DOCTYPE html>
<html>
<body>
<span class='font11'><b>Session 1a: Benchmarking &amp; Testing</b></span>
<div id='paperID402'>
<div class="paper">
<span class="paper_time_value">2:00pm - 2:20pm</span><br />
<span class='paper_status'><b><font color="#880000"> Invited talk </font></b></span><br>
<span class="paper_id">ID: 402</span>
<span class="paper_session"> / Tue | LAB 14h Parallel S 1a: 001</span><br />
<span class="paper_label">Modalities: </span><span class="paper_topics">Graphs, Multimodal Data</span><br>
<span class="paper_label">Methods: </span><span class="paper_topics">Physics-informed Machine Learning</span><br>
<span class="paper_label">Application Domain: </span><span class="paper_topics">Core Machine Learning</span><br>
<p class="paper_title">Embracing the Tyranny of Testing</p>
<p class="paper_author"> <u>Moritz Hardt</u></p>
<p class="paper_organisation">Max Planck Institute for Intelligent Systems, Germany</p>
<div ><div class="paper">
<p class="paper_abstract">We all remember cramming for a test, scrambling to prepare in the final stretch.</p>
<p class="paper_abstract">But what if the problem also charted a path forward?</p>
</div></div>
</div>
</div>
<hr>
<div id='paperID246'>
<div class="paper">
<span class="paper_time_value">2:20pm - 2:34pm</span><br />
<span class="paper_id">ID: 246</span>
<span class="paper_session"> / Tue | LAB 14h Parallel S 1a: 002</span><br />
<span class="paper_label">Modalities: </span><span class="paper_topics">Image, Video</span><br>
<span class="paper_label">Methods: </span><span class="paper_topics">Other</span><br>
<span class="paper_label">Application Domain: </span><span class="paper_topics">Core Machine Learning</span><br>
<p class="paper_title">Bridging Perception and Logic</p>
<p class="paper_author"> <u>Jens Johannes Gebele</u><sup>1,2</sup>, Philipp Brune<sup>1</sup>, Frank Schwab<sup>2</sup></p>
<p class="paper_organisation"><sup>1</sup>Neu-Ulm University of Applied Sciences; <sup>2</sup>University of Würzburg</p>
<div ><div class="paper">
<p class="paper_abstract">Automated recognition of emotions and facial expressions is key for human-computer interaction.</p>
</div></div>
</div>
</div>
</body>
</html>
"""

SESSION_EMPTY_HTML = """
<!DOCTYPE html>
<html>
<body>
<span class='font11'><b>Registration</b></span>
</body>
</html>
"""

SESSION_NO_ABSTRACT_HTML = """
<!DOCTYPE html>
<html>
<body>
<span class='font11'><b>Keynote</b></span>
<div id='paperID100'>
<div class="paper">
<span class="paper_id">ID: 100</span>
<span class="paper_session"> / Keynote: 001</span><br />
<p class="paper_title">Some Keynote Title</p>
<p class="paper_author"> <u>Some Speaker</u></p>
<p class="paper_organisation">Some Institution</p>
</div>
</div>
</body>
</html>
"""


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def haicon_plugin():
    """Create a HAICON plugin instance."""
    return HAICONDownloaderPlugin()


def _make_mock_response(html: str) -> Mock:
    """Create a mock requests.Response for the given HTML content."""
    mock_response = Mock()
    mock_response.text = html
    mock_response.raise_for_status = Mock()
    return mock_response


# ============================================================================
# Unit Tests – Plugin Properties
# ============================================================================


class TestHAICONPluginProperties:
    """Test basic plugin properties and metadata."""

    def test_plugin_instantiation(self, haicon_plugin):
        """Test that the plugin can be instantiated."""
        assert haicon_plugin is not None
        assert isinstance(haicon_plugin, HAICONDownloaderPlugin)

    def test_plugin_inherits_lightweight(self, haicon_plugin):
        """Test that the plugin inherits from LightweightDownloaderPlugin."""
        assert isinstance(haicon_plugin, LightweightDownloaderPlugin)
        assert isinstance(haicon_plugin, DownloaderPlugin)

    def test_plugin_name(self, haicon_plugin):
        """Test plugin name attribute."""
        assert haicon_plugin.plugin_name == "haicon"

    def test_plugin_conference_name(self, haicon_plugin):
        """Test conference name attribute."""
        assert haicon_plugin.conference_name == "HAICON"

    def test_plugin_description(self, haicon_plugin):
        """Test plugin description."""
        assert "HAICON" in haicon_plugin.plugin_description
        assert "ConfTool" in haicon_plugin.plugin_description

    def test_plugin_supported_years(self, haicon_plugin):
        """Test that 2026 is in supported years."""
        assert 2026 in haicon_plugin.supported_years
        assert all(y >= 2026 for y in haicon_plugin.supported_years)

    def test_plugin_metadata(self, haicon_plugin):
        """Test plugin get_metadata() output."""
        meta = haicon_plugin.get_metadata()
        assert meta["name"] == "haicon"
        assert "HAICON" in meta["description"]
        assert "HAICON" in meta["conference_name"]
        assert 2026 in meta["supported_years"]
        assert "year" in meta["parameters"]
        assert "output_path" in meta["parameters"]
        assert "force_download" in meta["parameters"]

    def test_plugin_initialization_custom(self):
        """Test plugin initialization with custom parameters."""
        plugin = HAICONDownloaderPlugin(timeout=60, verify_ssl=False)
        assert plugin.timeout == 60
        assert plugin.verify_ssl is False

    def test_get_url(self, haicon_plugin):
        """Test that get_url returns the correct ConfTool URL."""
        url = haicon_plugin.get_url(2026)
        assert "haicon26" in url
        assert "sessions.php" in url

    def test_get_url_different_year(self, haicon_plugin):
        """Test get_url for different years."""
        assert "haicon27" in haicon_plugin.get_url(2027)
        assert "haicon30" in haicon_plugin.get_url(2030)


# ============================================================================
# Unit Tests – Validation
# ============================================================================


class TestHAICONValidation:
    """Test plugin year validation."""

    def test_validate_year_valid(self, haicon_plugin):
        """Test validation with a supported year."""
        haicon_plugin.validate_year(2026)  # Should not raise

    def test_validate_year_invalid(self, haicon_plugin):
        """Test validation with an unsupported year."""
        with pytest.raises(ValueError, match="not supported"):
            haicon_plugin.validate_year(2000)

    def test_validate_year_none(self, haicon_plugin):
        """Test validation with None (should not raise)."""
        haicon_plugin.validate_year(None)


# ============================================================================
# Unit Tests – HTML Parsing Helpers
# ============================================================================


class TestHAICONHTMLParsing:
    """Test internal HTML parsing methods."""

    def test_fetch_session_ids(self, haicon_plugin):
        """Test extraction of session IDs from the overview page."""
        with patch.object(haicon_plugin._session, "get") as mock_get:
            mock_get.return_value = _make_mock_response(SESSIONS_OVERVIEW_HTML)
            ids = haicon_plugin._fetch_session_ids("https://www.conftool.pro/haicon26", 30, True)
        assert 8 in ids
        assert 14 in ids
        assert 19 in ids
        assert 21 in ids
        assert len(ids) == 4

    def test_fetch_session_ids_sorted(self, haicon_plugin):
        """Test that returned session IDs are sorted."""
        with patch.object(haicon_plugin._session, "get") as mock_get:
            mock_get.return_value = _make_mock_response(SESSIONS_OVERVIEW_HTML)
            ids = haicon_plugin._fetch_session_ids("https://www.conftool.pro/haicon26", 30, True)
        assert ids == sorted(ids)

    def test_fetch_session_ids_request_error(self, haicon_plugin):
        """Test that a request error raises RuntimeError."""
        with patch.object(haicon_plugin._session, "get") as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")
            with pytest.raises(RuntimeError, match="Failed to fetch HAICON sessions"):
                haicon_plugin._fetch_session_ids("https://www.conftool.pro/haicon26", 30, True)

    def test_extract_session_name(self, haicon_plugin):
        """Test extraction of session name from page HTML."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(SESSION_19_HTML, "html.parser")
        name = haicon_plugin._extract_session_name(soup)
        assert name == "Poster Session I"

    def test_extract_session_name_fallback(self, haicon_plugin):
        """Test session name fallback when no font11 span exists."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup("<html><body></body></html>", "html.parser")
        name = haicon_plugin._extract_session_name(soup)
        assert name == "unknown session at HAICON"

    def test_extract_authors_simple(self, haicon_plugin):
        """Test author extraction without superscripts."""
        from bs4 import BeautifulSoup

        html = "<p class='paper_author'>Alice Smith, Bob Jones</p>"
        soup = BeautifulSoup(html, "html.parser")
        tag = soup.find("p")
        authors = haicon_plugin._extract_authors(tag)
        assert "Alice Smith" in authors
        assert "Bob Jones" in authors

    def test_extract_authors_with_superscripts(self, haicon_plugin):
        """Test author extraction with superscript affiliation indices."""
        from bs4 import BeautifulSoup

        html = "<p class='paper_author'>" "<u>Alice Smith</u><sup>1</sup>, Bob Jones<sup>1,2</sup>" "</p>"
        soup = BeautifulSoup(html, "html.parser")
        tag = soup.find("p")
        authors = haicon_plugin._extract_authors(tag)
        assert "Alice Smith" in authors
        assert "Bob Jones" in authors
        # No affiliation index numbers should remain
        for name in authors:
            assert not any(c.isdigit() for c in name)

    def test_extract_authors_single_author(self, haicon_plugin):
        """Test author extraction for a paper with a single author."""
        from bs4 import BeautifulSoup

        html = "<p class='paper_author'><u>Moritz Hardt</u></p>"
        soup = BeautifulSoup(html, "html.parser")
        tag = soup.find("p")
        authors = haicon_plugin._extract_authors(tag)
        assert len(authors) == 1
        assert authors[0] == "Moritz Hardt"

    def test_parse_paper_div_full(self, haicon_plugin):
        """Test parsing a full paper div."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(SESSION_19_HTML, "html.parser")
        paper_div = soup.find("div", id="paperID358")
        paper = haicon_plugin._parse_paper_div(paper_div, "Poster Session I", 2026)

        assert paper is not None
        assert paper.title == "Interpretable Representations for Hematology"
        assert paper.year == 2026
        assert paper.conference == "HAICON"
        assert paper.session == "Poster Session I"
        assert paper.poster_position == "Poster No. # 9: 001"
        assert paper.original_id == 358
        assert "Muhammed Furkan Dasdelen" in paper.authors
        assert "Hyesu Lim" in paper.authors
        assert "Sparse autoencoders" in paper.abstract

    def test_parse_paper_div_with_time(self, haicon_plugin):
        """Test parsing a paper div that includes presentation time."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(SESSION_14_HTML, "html.parser")
        paper_div = soup.find("div", id="paperID402")
        paper = haicon_plugin._parse_paper_div(paper_div, "Session 1a", 2026)

        assert paper is not None
        assert paper.title == "Embracing the Tyranny of Testing"
        assert paper.starttime == "2:00pm - 2:20pm"
        assert paper.original_id == 402

    def test_parse_paper_div_keywords(self, haicon_plugin):
        """Test that keywords are extracted from topics."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(SESSION_19_HTML, "html.parser")
        paper_div = soup.find("div", id="paperID358")
        paper = haicon_plugin._parse_paper_div(paper_div, "Poster Session I", 2026)

        assert paper is not None
        assert paper.keywords is not None
        # Keywords include modalities, methods, and application domain
        kw_text = " ".join(paper.keywords)
        assert "Image" in kw_text
        assert "Foundation Models" in kw_text
        assert "Health" in kw_text

    def test_parse_paper_div_no_abstract_returns_none(self, haicon_plugin):
        """Test that a paper div without an abstract returns None."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(SESSION_NO_ABSTRACT_HTML, "html.parser")
        paper_div = soup.find("div", id="paperID100")
        paper = haicon_plugin._parse_paper_div(paper_div, "Keynote", 2026)

        assert paper is None

    def test_parse_paper_div_no_title_returns_none(self, haicon_plugin):
        """Test that a paper div without a title returns None."""
        from bs4 import BeautifulSoup

        html = """
        <div id='paperID999'>
        <div class="paper">
          <span class="paper_id">ID: 999</span>
          <p class="paper_author"><u>Some Author</u></p>
          <div><div class="paper">
            <p class="paper_abstract">Some abstract text here.</p>
          </div></div>
        </div>
        </div>
        """
        soup = BeautifulSoup(html, "html.parser")
        paper_div = soup.find("div", id="paperID999")
        paper = haicon_plugin._parse_paper_div(paper_div, "Test Session", 2026)
        assert paper is None

    def test_parse_paper_div_no_authors_returns_none(self, haicon_plugin):
        """Test that a paper div without authors returns None."""
        from bs4 import BeautifulSoup

        html = """
        <div id='paperID999'>
        <div class="paper">
          <span class="paper_id">ID: 999</span>
          <p class="paper_title">A Paper Title</p>
          <div><div class="paper">
            <p class="paper_abstract">Some abstract text here.</p>
          </div></div>
        </div>
        </div>
        """
        soup = BeautifulSoup(html, "html.parser")
        paper_div = soup.find("div", id="paperID999")
        paper = haicon_plugin._parse_paper_div(paper_div, "Test Session", 2026)
        assert paper is None

    def test_fetch_session_papers(self, haicon_plugin):
        """Test that papers are fetched and parsed for a session."""
        with patch.object(haicon_plugin._session, "get") as mock_get:
            mock_get.return_value = _make_mock_response(SESSION_19_HTML)
            papers = haicon_plugin._fetch_session_papers("https://www.conftool.pro/haicon26", 19, 2026, 30, True)

        assert len(papers) == 2
        titles = [p.title for p in papers]
        assert "Interpretable Representations for Hematology" in titles

    def test_fetch_session_papers_empty_session(self, haicon_plugin):
        """Test that an empty session returns no papers."""
        with patch.object(haicon_plugin._session, "get") as mock_get:
            mock_get.return_value = _make_mock_response(SESSION_EMPTY_HTML)
            papers = haicon_plugin._fetch_session_papers("https://www.conftool.pro/haicon26", 8, 2026, 30, True)

        assert papers == []

    def test_fetch_session_papers_request_error(self, haicon_plugin):
        """Test that a request error logs a warning and returns empty list."""
        with patch.object(haicon_plugin._session, "get") as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")
            papers = haicon_plugin._fetch_session_papers("https://www.conftool.pro/haicon26", 99, 2026, 30, True)

        assert papers == []

    def test_multiline_abstract_joined(self, haicon_plugin):
        """Test that multi-paragraph abstracts are joined with newlines."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(SESSION_14_HTML, "html.parser")
        paper_div = soup.find("div", id="paperID402")
        paper = haicon_plugin._parse_paper_div(paper_div, "Session 1a", 2026)

        assert paper is not None
        assert "\n\n" in paper.abstract
        assert "cramming" in paper.abstract
        assert "path forward" in paper.abstract


# ============================================================================
# Unit Tests – Download Method
# ============================================================================


class TestHAICONDownload:
    """Test the main download() method."""

    def _setup_session_mock(self, mock_get):
        """Configure mock_get to return appropriate HTML for each URL."""

        def side_effect(url, **kwargs):
            if "sessions.php" in url:
                return _make_mock_response(SESSIONS_OVERVIEW_HTML)
            elif "form_session=14" in url:
                return _make_mock_response(SESSION_14_HTML)
            elif "form_session=19" in url:
                return _make_mock_response(SESSION_19_HTML)
            elif "form_session=21" in url:
                return _make_mock_response(SESSION_EMPTY_HTML)
            else:
                return _make_mock_response(SESSION_EMPTY_HTML)

        mock_get.side_effect = side_effect

    def test_download_returns_list(self, haicon_plugin):
        """Test that download() returns a list of LightweightPaper objects."""
        with patch.object(haicon_plugin._session, "get") as mock_get:
            self._setup_session_mock(mock_get)
            papers = haicon_plugin.download(year=2026, request_delay=0)

        assert isinstance(papers, list)
        assert len(papers) > 0

    def test_download_paper_fields(self, haicon_plugin):
        """Test that downloaded papers have required fields."""
        with patch.object(haicon_plugin._session, "get") as mock_get:
            self._setup_session_mock(mock_get)
            papers = haicon_plugin.download(year=2026, request_delay=0)

        for paper in papers:
            assert paper.title
            assert paper.authors
            assert paper.abstract
            assert paper.year == 2026
            assert paper.conference == "HAICON"

    def test_download_deduplication(self, haicon_plugin):
        """Test that duplicate papers (same ID) are de-duplicated."""
        # Both SESSION_14_HTML and SESSION_19_HTML have different paper IDs,
        # so no duplicates. But we can simulate duplicates by using the same
        # session HTML for multiple sessions.
        duplicate_html_map = {
            "sessions.php": SESSIONS_OVERVIEW_HTML,
            "form_session=8": SESSION_19_HTML,  # same papers appear in two sessions
            "form_session=14": SESSION_19_HTML,
            "form_session=19": SESSION_19_HTML,
            "form_session=21": SESSION_EMPTY_HTML,
        }

        def side_effect(url, **kwargs):
            for key, html in duplicate_html_map.items():
                if key in url:
                    return _make_mock_response(html)
            return _make_mock_response(SESSION_EMPTY_HTML)

        with patch.object(haicon_plugin._session, "get") as mock_get:
            mock_get.side_effect = side_effect
            papers = haicon_plugin.download(year=2026, request_delay=0)

        # Papers 358 and 255 should appear only once despite appearing in
        # multiple session responses.
        titles = [p.title for p in papers]
        assert titles.count("Interpretable Representations for Hematology") == 1
        assert titles.count("Analysis of multi-omics data using graph neural networks") == 1

    def test_download_default_year(self, haicon_plugin):
        """Test that the default year is 2026."""
        with patch.object(haicon_plugin._session, "get") as mock_get:
            self._setup_session_mock(mock_get)
            papers = haicon_plugin.download(request_delay=0)  # no year specified

        for paper in papers:
            assert paper.year == 2026

    def test_download_invalid_year(self, haicon_plugin):
        """Test that an unsupported year raises ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            haicon_plugin.download(year=2000)

    def test_download_saves_to_file(self, haicon_plugin):
        """Test that download() saves results to a JSON file when path given."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "haicon_2026.json"

            with patch.object(haicon_plugin._session, "get") as mock_get:
                self._setup_session_mock(mock_get)
                haicon_plugin.download(year=2026, output_path=str(output_path), request_delay=0)

            assert output_path.exists()
            with open(output_path, "r", encoding="utf-8") as fh:
                saved = json.load(fh)

            assert isinstance(saved, list)
            assert len(saved) > 0
            assert saved[0]["year"] == 2026
            assert saved[0]["conference"] == "HAICON"

    def test_download_loads_from_cache(self, haicon_plugin):
        """Test that download() loads from a cached file without HTTP requests."""
        cached_papers = [
            {
                "title": "Cached Paper",
                "abstract": "Cached abstract text.",
                "authors": ["Alice Test"],
                "session": "Poster Session I",
                "poster_position": "001",
                "year": 2026,
                "conference": "HAICON",
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "haicon_2026.json"
            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(cached_papers, fh)

            with patch.object(haicon_plugin._session, "get") as mock_get:
                papers = haicon_plugin.download(year=2026, output_path=str(output_path), request_delay=0)
                mock_get.assert_not_called()

            assert len(papers) == 1
            assert papers[0].title == "Cached Paper"

    def test_download_force_redownload(self, haicon_plugin):
        """Test that force_download=True ignores cached file."""
        cached_papers = [
            {
                "title": "Old Cached Paper",
                "abstract": "Old abstract.",
                "authors": ["Old Author"],
                "session": "Session",
                "poster_position": "",
                "year": 2026,
                "conference": "HAICON",
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "haicon_2026.json"
            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(cached_papers, fh)

            with patch.object(haicon_plugin._session, "get") as mock_get:
                self._setup_session_mock(mock_get)
                papers = haicon_plugin.download(
                    year=2026,
                    output_path=str(output_path),
                    force_download=True,
                    request_delay=0,
                )
                assert mock_get.called

            # Should have fresh data, not the cached "Old Cached Paper"
            titles = [p.title for p in papers]
            assert "Old Cached Paper" not in titles

    def test_download_request_error(self, haicon_plugin):
        """Test that a request error on the overview page raises RuntimeError."""
        with patch.object(haicon_plugin._session, "get") as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError("Network error")
            with pytest.raises(RuntimeError, match="Failed to fetch HAICON sessions"):
                haicon_plugin.download(year=2026, request_delay=0)

    def test_download_timeout_kwarg(self, haicon_plugin):
        """Test that a custom timeout kwarg is passed to requests."""
        with patch.object(haicon_plugin._session, "get") as mock_get:
            self._setup_session_mock(mock_get)
            haicon_plugin.download(year=2026, timeout=90, request_delay=0)

        # Check that at least one call used timeout=90
        call_kwargs = [call[1] for call in mock_get.call_args_list]
        assert any(kw.get("timeout") == 90 for kw in call_kwargs)


# ============================================================================
# Unit Tests – Plugin Registration
# ============================================================================


class TestHAICONPluginRegistration:
    """Test plugin auto-registration."""

    def test_plugin_auto_registers(self):
        """Test that the HAICON plugin auto-registers on import."""
        plugin = get_plugin("haicon")
        assert plugin is not None
        assert isinstance(plugin, HAICONDownloaderPlugin)

    def test_plugin_in_list(self):
        """Test that the HAICON plugin appears in the plugin list."""
        names = list_plugin_names()
        assert "haicon" in names

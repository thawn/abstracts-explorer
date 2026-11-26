"""
Tests for NeurIPS Downloader Plugin
====================================

Test suite for the NeurIPS conference data downloader plugin, including the
pre-2020 proceedings scraper.
"""

import json
import logging
from unittest.mock import Mock, patch

import pytest
import requests

from abstracts_explorer.plugin.persistence import FailureSidecar
from abstracts_explorer.plugins.neurips_downloader import NeurIPSDownloaderPlugin

pytestmark = pytest.mark.usefixtures("mock_current_year_probe")


def make_text_response(text: str) -> Mock:
    """Return a mock HTTP response with *text* as its body."""
    response = Mock()
    response.text = text
    return response


def make_json_response(payload) -> Mock:
    """Return a mock HTTP response with *payload* as JSON."""
    response = Mock()
    response.json.return_value = payload
    return response


def make_cache_paper(url: str = "https://papers.nips.cc/paper_files/paper/2019/hash/aaa-Abstract.html") -> dict:
    """Return a cached lightweight paper dictionary."""
    return {
        "title": "Cached Paper",
        "authors": ["Cached Author"],
        "abstract": "Cached abstract.",
        "session": "NeurIPS 2019",
        "poster_position": "",
        "year": 2019,
        "conference": "NeurIPS",
        "url": url,
    }


INDEX_HTML = """
<html><body><ul class="paper-list">
<li><div class="paper-content">
  <a href="/paper_files/paper/2019/hash/abc123-Abstract.html">Paper One</a>
  <span class="paper-authors">Alice Smith, Bob Jones</span>
</div></li>
<li><div class="paper-content">
  <a href="/paper_files/paper/2019/hash/def456-Abstract.html">Paper Two</a>
  <span class="paper-authors">Carol White</span>
</div></li>
</ul></body></html>
"""


def abstract_html(title: str, abstract: str, pdf: str) -> str:
    """Return a NeurIPS abstract page fixture."""
    return f"""
<html><head>
<meta name="citation_title" content="{title}">
<meta name="citation_pdf_url" content="{pdf}">
</head><body>
<p class="paper-abstract">{abstract}</p>
</body></html>
"""


class TestNeurIPSPluginMetadata:
    """Metadata and year handling tests."""

    def test_plugin_metadata(self):
        plugin = NeurIPSDownloaderPlugin()

        assert plugin.plugin_name == "neurips"
        assert 2025 in plugin.supported_years
        assert 1987 in plugin.supported_years
        metadata = plugin.get_metadata()
        assert metadata["name"] == "neurips"
        assert metadata["conference_name"] == "NeurIPS"

    def test_get_url_uses_eventhosts_for_modern_years(self):
        plugin = NeurIPSDownloaderPlugin()

        assert plugin.get_url(2025) == "https://neurips.cc/static/virtual/data/neurips-2025-orals-posters.json"

    def test_get_url_uses_proceedings_for_legacy_years(self):
        plugin = NeurIPSDownloaderPlugin()

        assert plugin.get_url(2019) == "https://papers.nips.cc/paper_files/paper/2019"

    def test_validate_year_rejects_pre_start_year(self):
        plugin = NeurIPSDownloaderPlugin()

        with pytest.raises(ValueError, match="Year 1986 not supported"):
            plugin.validate_year(1986)


class TestNeurIPSLegacyDownload:
    """Tests for the pre-2020 proceedings scraper."""

    @patch("requests.Session.get")
    def test_download_legacy_scrapes_index_and_abstracts(self, mock_get):
        mock_get.side_effect = [
            make_text_response(INDEX_HTML),
            make_text_response(
                abstract_html(
                    "Paper One",
                    "First abstract.",
                    "https://papers.nips.cc/paper_files/paper/2019/file/abc123-Paper.pdf",
                )
            ),
            make_text_response(
                abstract_html(
                    "Paper Two",
                    "Second abstract.",
                    "https://papers.nips.cc/paper_files/paper/2019/file/def456-Paper.pdf",
                )
            ),
        ]

        plugin = NeurIPSDownloaderPlugin()
        papers = plugin.download(year=2019, max_workers=1)

        assert len(papers) == 2
        by_title = {paper.title: paper for paper in papers}
        assert by_title["Paper One"].abstract == "First abstract."
        assert by_title["Paper One"].authors == ["Alice Smith", "Bob Jones"]
        assert by_title["Paper One"].original_id == "abc123"
        assert by_title["Paper One"].paper_pdf_url
        assert by_title["Paper One"].paper_pdf_url.endswith("abc123-Paper.pdf")
        assert by_title["Paper One"].session == "NeurIPS 2019"
        assert by_title["Paper One"].conference == "NeurIPS"
        assert by_title["Paper One"].year == 2019

    @patch("requests.Session.get")
    def test_download_legacy_uses_metadata_authors_when_index_missing(self, mock_get):
        index_html = """
        <ul class="paper-list"><li><div class="paper-content">
          <a href="/paper_files/paper/2019/hash/abc123-Abstract.html">Paper One</a>
          <span class="paper-authors"></span>
        </div></li></ul>
        """
        abstract_page = """
        <html><head>
        <meta name="citation_title" content="Paper One">
        <meta name="citation_author" content="Smith, Alice">
        <meta name="citation_author" content="Jones, Bob">
        </head><body><p class="paper-abstract">An abstract.</p></body></html>
        """
        mock_get.side_effect = [make_text_response(index_html), make_text_response(abstract_page)]

        papers = NeurIPSDownloaderPlugin().download(year=2019, max_workers=1)

        assert papers[0].authors == ["Alice Smith", "Bob Jones"]

    @patch("requests.Session.get")
    def test_download_legacy_records_failures(self, mock_get, tmp_path):
        output_path = tmp_path / "neurips_2019.json"
        mock_get.side_effect = [
            make_text_response(INDEX_HTML),
            make_text_response(abstract_html("Paper One", "First abstract.", "")),
            requests.exceptions.RequestException("boom"),
        ]

        papers = NeurIPSDownloaderPlugin().download(year=2019, output_path=str(output_path), max_workers=1)

        assert [paper.title for paper in papers] == ["Paper One"]
        failures = FailureSidecar(str(output_path), "NeurIPS", 2019).load()
        assert len(failures) == 1
        assert failures[0]["type"] == "paper"
        assert "def456" in failures[0]["url"]

    @patch("requests.Session.get")
    def test_download_legacy_skips_papers_without_authors(self, mock_get, tmp_path, caplog):
        """Skip records with no authors without recording a retriable failure."""
        output_path = tmp_path / "neurips_2019.json"
        index_html = """
        <ul class="paper-list">
        <li><div class="paper-content">
          <a href="/paper_files/paper/2019/hash/abc123-Abstract.html">Paper One</a>
          <span class="paper-authors">Alice Smith</span>
        </div></li>
        <li><div class="paper-content">
          <a href="/paper_files/paper/2019/hash/def456-Abstract.html">Paper Two</a>
          <span class="paper-authors"></span>
        </div></li>
        </ul>
        """
        mock_get.side_effect = [
            make_text_response(index_html),
            make_text_response(abstract_html("Paper One", "First abstract.", "")),
            make_text_response(abstract_html("Paper Two", "Second abstract.", "")),
        ]

        with caplog.at_level(logging.WARNING):
            papers = NeurIPSDownloaderPlugin().download(
                year=2019,
                output_path=str(output_path),
                max_workers=1,
            )

        assert [paper.title for paper in papers] == ["Paper One"]
        assert FailureSidecar(str(output_path), "NeurIPS", 2019).load() == []
        assert "Authors list cannot be empty" not in caplog.text

    @patch("requests.Session.get")
    def test_download_legacy_retries_failures_from_cache(self, mock_get, tmp_path):
        output_path = tmp_path / "neurips_2019.json"
        output_path.write_text(json.dumps([make_cache_paper()]))
        failure_url = "https://papers.nips.cc/paper_files/paper/2019/hash/def456-Abstract.html"
        FailureSidecar(str(output_path), "NeurIPS", 2019).save(
            [{"url": failure_url, "type": "paper", "title": "Paper Two", "authors": ["Carol White"]}]
        )
        mock_get.return_value = make_text_response(
            abstract_html(
                "Paper Two",
                "Recovered abstract.",
                "https://papers.nips.cc/paper_files/paper/2019/file/def456-Paper.pdf",
            )
        )

        papers = NeurIPSDownloaderPlugin().download(year=2019, output_path=str(output_path))

        assert len(papers) == 2
        assert any(paper.abstract == "Recovered abstract." for paper in papers)
        assert FailureSidecar(str(output_path), "NeurIPS", 2019).load() == []

    @patch("requests.Session.get")
    def test_download_modern_year_still_uses_json(self, mock_get):
        mock_get.return_value = make_json_response(
            {
                "results": [
                    {
                        "id": 1,
                        "name": "Modern Paper",
                        "abstract": "Modern abstract.",
                        "authors": [{"fullname": "Modern Author"}],
                        "session": "Poster Session",
                        "poster_position": "1",
                    }
                ]
            }
        )

        papers = NeurIPSDownloaderPlugin().download(year=2025)

        assert len(papers) == 1
        assert papers[0].title == "Modern Paper"
        requested_urls = [call.args[0] for call in mock_get.call_args_list]
        assert any("neurips-2025-orals-posters.json" in url for url in requested_urls)


class TestNeurIPSPluginRegistration:
    """Plugin registration tests."""

    def test_plugin_auto_registers(self):
        from abstracts_explorer.plugins import get_plugin

        plugin = get_plugin("neurips")
        assert isinstance(plugin, NeurIPSDownloaderPlugin)

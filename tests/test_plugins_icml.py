"""
Tests for ICML Downloader Plugin
=================================

Test suite for the ICML conference data downloader plugin, including the
pre-2020 PMLR proceedings loader.
"""

from unittest.mock import Mock, patch

import pytest

from abstracts_explorer.plugins.icml_downloader import ICMLDownloaderPlugin

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


CITEPROC_YAML = """
- title: 'Paper One'
  abstract: 'First abstract.'
  authors: 'Smith, Alice and Jones, Bob'
  url: 'https://proceedings.mlr.press/v97/smith19a.html'
  pdf: 'http://proceedings.mlr.press/v97/smith19a/smith19a.pdf'
- title: 'Paper Two'
  abstract: 'Second abstract.'
  authors: 'White, Carol'
  url: 'https://proceedings.mlr.press/v97/white19a.html'
  pdf: 'http://proceedings.mlr.press/v97/white19a/white19a.pdf'
"""


class TestICMLPluginMetadata:
    """Metadata and year handling tests."""

    def test_plugin_metadata(self):
        plugin = ICMLDownloaderPlugin()

        assert plugin.plugin_name == "icml"
        assert 2025 in plugin.supported_years
        assert 2013 in plugin.supported_years
        metadata = plugin.get_metadata()
        assert metadata["name"] == "icml"
        assert metadata["conference_name"] == "ICML"

    def test_get_url_uses_eventhosts_for_modern_years(self):
        plugin = ICMLDownloaderPlugin()

        assert plugin.get_url(2025) == "https://icml.cc/static/virtual/data/icml-2025-orals-posters.json"

    @pytest.mark.parametrize(
        "year,volume",
        [(2013, 28), (2014, 32), (2015, 37), (2016, 48), (2017, 70), (2018, 80), (2019, 97)],
    )
    def test_get_url_maps_legacy_year_to_pmlr_volume(self, year, volume):
        plugin = ICMLDownloaderPlugin()

        assert plugin.get_url(year) == f"https://proceedings.mlr.press/v{volume}/"

    def test_validate_year_rejects_pre_start_year(self):
        plugin = ICMLDownloaderPlugin()

        with pytest.raises(ValueError, match="Year 2012 not supported"):
            plugin.validate_year(2012)


class TestICMLLegacyDownload:
    """Tests for the pre-2020 PMLR loader."""

    @patch("requests.Session.get")
    def test_download_legacy_parses_citeproc_yaml(self, mock_get):
        mock_get.return_value = make_text_response(CITEPROC_YAML)

        papers = ICMLDownloaderPlugin().download(year=2019)

        assert len(papers) == 2
        by_title = {paper.title: paper for paper in papers}
        assert by_title["Paper One"].abstract == "First abstract."
        assert by_title["Paper One"].authors == ["Alice Smith", "Bob Jones"]
        assert by_title["Paper One"].original_id == "smith19a"
        assert by_title["Paper One"].paper_pdf_url
        assert by_title["Paper One"].paper_pdf_url.endswith("smith19a.pdf")
        assert by_title["Paper One"].url == "https://proceedings.mlr.press/v97/smith19a.html"
        assert by_title["Paper One"].session == "ICML 2019"
        assert by_title["Paper One"].conference == "ICML"
        assert by_title["Paper One"].year == 2019
        assert mock_get.call_args.args[0].endswith("v97/assets/bib/citeproc.yaml")

    @patch("requests.Session.get")
    def test_download_legacy_parses_native_citeproc_shape(self, mock_get):
        """Support PMLR's native citeproc keys (author/URL/PDF/id)."""
        mock_get.return_value = make_text_response("""
- title: 'Paper One'
  abstract: 'First abstract.'
  author:
    - given: 'Alice'
      family: 'Smith'
    - given: 'Bob'
      family: 'Jones'
  URL: 'https://proceedings.mlr.press/v97/smith19a.html'
  PDF: 'http://proceedings.mlr.press/v97/smith19a/smith19a.pdf'
  id: 'smith19a'
""")

        papers = ICMLDownloaderPlugin().download(year=2019)

        assert len(papers) == 1
        paper = papers[0]
        assert paper.authors == ["Alice Smith", "Bob Jones"]
        assert paper.original_id == "smith19a"
        assert paper.url == "https://proceedings.mlr.press/v97/smith19a.html"
        assert paper.paper_pdf_url == "http://proceedings.mlr.press/v97/smith19a/smith19a.pdf"

    @patch("requests.Session.get")
    def test_download_legacy_skips_incomplete_records(self, mock_get):
        mock_get.return_value = make_text_response("""
- title: 'Paper One'
  abstract: 'First abstract.'
  authors: 'Smith, Alice'
  url: 'https://proceedings.mlr.press/v97/smith19a.html'
- title: 'Missing Abstract'
  authors: 'Jones, Bob'
  url: 'https://proceedings.mlr.press/v97/jones19a.html'
""")

        papers = ICMLDownloaderPlugin().download(year=2019)

        assert [paper.title for paper in papers] == ["Paper One"]

    @patch("requests.Session.get")
    def test_download_legacy_invalid_yaml_raises(self, mock_get):
        mock_get.return_value = make_text_response("foo: [unclosed")

        with pytest.raises(RuntimeError, match="Invalid PMLR citation YAML"):
            ICMLDownloaderPlugin().download(year=2019)

    @patch("requests.Session.get")
    def test_download_legacy_empty_yaml_raises(self, mock_get):
        mock_get.return_value = make_text_response("[]")

        with pytest.raises(RuntimeError, match="No PMLR citation records"):
            ICMLDownloaderPlugin().download(year=2019)

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

        papers = ICMLDownloaderPlugin().download(year=2025)

        assert len(papers) == 1
        assert papers[0].title == "Modern Paper"
        assert "icml-2025-orals-posters.json" in mock_get.call_args.args[0]


class TestICMLPluginRegistration:
    """Plugin registration tests."""

    def test_plugin_auto_registers(self):
        from abstracts_explorer.plugins import get_plugin

        plugin = get_plugin("icml")
        assert isinstance(plugin, ICMLDownloaderPlugin)

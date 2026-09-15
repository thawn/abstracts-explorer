"""Tests for the shared JSON conference downloader."""

import json
import logging
from unittest.mock import Mock, patch

import pytest
import requests

from abstracts_explorer.plugins.json_conference_downloader import (
    JSONConferenceDownloaderPlugin,
)


class ExampleJSONDownloader(JSONConferenceDownloaderPlugin):
    """Minimal downloader used to exercise the shared implementation."""

    plugin_name = "example-json"
    plugin_description = "Example JSON downloader"
    _start_year = 2020
    conference_name = "Example"

    def get_url(self, year: int) -> str:
        """Return an EventHosts-style paper URL."""
        return f"https://example.org/data/example-{year}-orals-posters.json"


def make_response(payload):
    """Return a mock HTTP response containing *payload*."""
    response = Mock()
    response.json.return_value = payload
    return response


def make_invalid_json_response():
    """Return a mock HTTP response whose body is not valid JSON."""
    response = Mock()
    response.json.side_effect = json.JSONDecodeError("invalid JSON", "not JSON", 0)
    return response


def make_paper(paper_id: int, abstract=None):
    """Return a minimal EventHosts paper record."""
    paper = {
        "id": paper_id,
        "name": f"Paper {paper_id}",
        "authors": [{"fullname": "Test Author"}],
        "session": "Poster Session",
        "poster_position": str(paper_id),
    }
    if abstract is not None:
        paper["abstract"] = abstract
    return paper


class TestJSONConferenceDownloader:
    """Test companion abstract handling in the shared downloader."""

    def test_derives_companion_abstract_url(self):
        """Derive the companion URL from the EventHosts paper URL."""
        plugin = ExampleJSONDownloader()
        expected_url = "https://example.org/data/example-2025-abstracts.json"

        assert plugin.get_companion_abstracts_url(2025) == expected_url

    @patch("abstracts_explorer.plugins.json_conference_downloader.requests.get")
    def test_inline_abstracts_do_not_fetch_companion(self, mock_get):
        """Keep historical inline abstracts without another request."""
        mock_get.return_value = make_response({"results": [make_paper(1, "Existing abstract")]})

        papers = ExampleJSONDownloader().download(year=2025)

        assert [paper.abstract for paper in papers] == ["Existing abstract"]
        mock_get.assert_called_once()

    @patch("abstracts_explorer.plugins.json_conference_downloader.requests.get")
    def test_merges_companion_abstracts(self, mock_get):
        """Join split abstracts to paper records by stringified ID."""
        mock_get.side_effect = [
            make_response({"results": [make_paper(1), make_paper(2, "   ")]}),
            make_response({"1": "First abstract", "2": "Second abstract"}),
        ]

        papers = ExampleJSONDownloader().download(year=2025)

        assert [paper.abstract for paper in papers] == [
            "First abstract",
            "Second abstract",
        ]
        companion_url = mock_get.call_args_list[1].args[0]
        assert companion_url.endswith("example-2025-abstracts.json")

    @patch("abstracts_explorer.plugins.json_conference_downloader.requests.get")
    def test_partial_companion_coverage_keeps_valid_papers(self, mock_get, caplog):
        """Preserve inline and matched papers while reporting unmatched records."""
        mock_get.side_effect = [
            make_response(
                {
                    "results": [
                        make_paper(1, "Existing abstract"),
                        make_paper(2),
                        make_paper(3),
                    ]
                }
            ),
            make_response(
                {
                    "1": "Must not replace the inline abstract",
                    "2": "Companion abstract",
                }
            ),
        ]

        with caplog.at_level(logging.WARNING):
            papers = ExampleJSONDownloader().download(year=2025)

        assert [paper.abstract for paper in papers] == [
            "Existing abstract",
            "Companion abstract",
        ]
        assert "No companion abstract found for 1 of 2 records" in caplog.text

    @pytest.mark.parametrize(
        "companion_response",
        [
            requests.exceptions.RequestException("not found"),
            make_invalid_json_response(),
            make_response([]),
        ],
    )
    @patch("abstracts_explorer.plugins.json_conference_downloader.requests.get")
    def test_reports_unusable_companion_response(
        self,
        mock_get,
        companion_response,
        caplog,
    ):
        """Report unavailable, invalid JSON, and malformed companion responses."""
        mock_get.side_effect = [
            make_response({"results": [make_paper(1, "Existing abstract"), make_paper(2)]}),
            companion_response,
        ]

        with caplog.at_level(logging.WARNING):
            papers = ExampleJSONDownloader().download(year=2025)

        assert [paper.abstract for paper in papers] == ["Existing abstract"]
        assert "companion abstracts" in caplog.text

    @patch("abstracts_explorer.plugins.json_conference_downloader.requests.get")
    def test_empty_event_payload_does_not_fetch_companion(self, mock_get):
        """Do not request companion data for an empty event payload."""
        mock_get.return_value = make_response({"results": []})

        assert ExampleJSONDownloader().download(year=2025) == []
        mock_get.assert_called_once()

    @patch("abstracts_explorer.plugins.json_conference_downloader.requests.get")
    def test_reports_when_no_source_records_are_valid(self, mock_get, caplog):
        """Report a non-empty source that produces no valid papers."""
        mock_get.side_effect = [
            make_response({"results": [make_paper(1)]}),
            requests.exceptions.RequestException("not found"),
        ]

        with caplog.at_level(logging.WARNING):
            papers = ExampleJSONDownloader().download(year=2025)

        assert papers == []
        assert "No valid papers produced from 1 source records" in caplog.text

    @patch("abstracts_explorer.plugins.json_conference_downloader.requests.get")
    def test_reports_when_conversion_drops_all_source_records(self, mock_get, caplog):
        """Report a non-empty source even when conversion drops every record."""
        mock_get.return_value = make_response({"results": [{"id": 1, "abstract": "Existing abstract"}]})

        with caplog.at_level(logging.WARNING):
            papers = ExampleJSONDownloader().download(year=2025)

        assert papers == []
        assert "No valid papers produced from 1 source records" in caplog.text
        mock_get.assert_called_once()

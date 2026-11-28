"""Tests for automatic keyword generation during download."""

import argparse
from unittest.mock import MagicMock, patch

from abstracts_explorer.cli.download import download_command
from abstracts_explorer.plugin import LightweightPaper


def _paper(conference="NeurIPS", year=2025):
    return LightweightPaper(
        title="Neural Networks",
        authors=["Alice"],
        abstract="Neural network optimization.",
        session="Session",
        poster_position="P1",
        year=year,
        conference=conference,
    )


def _args(tmp_path):
    return argparse.Namespace(
        conference=None,
        year=2025,
        output=str(tmp_path / "papers.json"),
        force=False,
        verbose=0,
        list_plugins=False,
    )


def _plugin():
    plugin = MagicMock()
    plugin.plugin_name = "neurips"
    plugin.plugin_description = "NeurIPS"
    plugin.supported_years = {2025}
    plugin.requires_manual_input = False
    return plugin


class TestDownloadKeywordGeneration:
    def test_generates_keywords_after_adding_papers(self, tmp_path):
        db = MagicMock()
        db.__enter__.return_value = db
        db.add_papers.return_value = 1

        with (
            patch("abstracts_explorer.cli.download.get_all_plugins", return_value=[_plugin()]),
            patch("abstracts_explorer.cli.download._download_single", return_value=[_paper()]),
            patch("abstracts_explorer.cli.download.DatabaseManager", return_value=db),
            patch("abstracts_explorer.cli.download.generate_keywords_for_scope") as generate,
        ):
            result = download_command(_args(tmp_path))

        assert result == 0
        generate.assert_called_once_with(db, "NeurIPS", 2025)

    def test_generation_failure_does_not_fail_download(self, tmp_path):
        db = MagicMock()
        db.__enter__.return_value = db
        db.add_papers.return_value = 1

        with (
            patch("abstracts_explorer.cli.download.get_all_plugins", return_value=[_plugin()]),
            patch("abstracts_explorer.cli.download._download_single", return_value=[_paper()]),
            patch("abstracts_explorer.cli.download.DatabaseManager", return_value=db),
            patch(
                "abstracts_explorer.cli.download.generate_keywords_for_scope",
                side_effect=RuntimeError("boom"),
            ),
        ):
            result = download_command(_args(tmp_path))

        assert result == 0

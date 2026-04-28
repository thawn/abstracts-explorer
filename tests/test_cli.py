"""
Unit tests for the CLI module.

This module tests the command-line interface for neurips-abstracts,
including the download and create-embeddings commands.
"""

import sys
import logging
import argparse
import contextlib
from unittest.mock import Mock, patch
import pytest
from abstracts_explorer.cli import (
    main,
    search_command,
    setup_logging,
    _build_embeddings_where_clause,
    add_conference_year_args,
    pre_process_command,
)
from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.plugin import LightweightPaper
from tests.conftest import set_test_db
from abstracts_explorer.config import get_config


def patch_get_config_for_test(monkeypatch, embeddings_path):
    """
    Patch get_config to use test environment variables.

    This ensures that get_config reads the EMBEDDING_DB environment variable
    set by monkeypatch, by forcing a config reload with .env.test.
    """
    from tests.conftest import get_env_test_path

    monkeypatch.setenv("EMBEDDING_DB", str(embeddings_path))

    # Get path to .env.test
    env_test = get_env_test_path()

    # Create a wrapper that always reloads config with .env.test
    original_get_config = get_config

    def get_config_with_reload(reload=False):
        return original_get_config(reload=True, env_path=env_test)

    # Patch get_config in the cli module
    monkeypatch.setattr("abstracts_explorer.cli.get_config", get_config_with_reload)


class TestBuildEmbeddingsWhereClause:
    """Test cases for _build_embeddings_where_clause helper."""

    def test_no_filters(self):
        """No filters returns None."""
        args = argparse.Namespace(conference=None, year=None, where=None)
        assert _build_embeddings_where_clause(args) is None

    def test_conference_only(self):
        """Conference filter produces an exact-match WHERE clause."""
        args = argparse.Namespace(conference="NeurIPS", year=None, where=None)
        result = _build_embeddings_where_clause(args)
        assert "conference = 'NeurIPS'" in result

    def test_year_only(self):
        """Year filter produces correct WHERE clause."""
        args = argparse.Namespace(conference=None, year=2024, where=None)
        assert _build_embeddings_where_clause(args) == "year = 2024"

    def test_conference_and_year(self):
        """Both conference and year produce combined WHERE clause."""
        args = argparse.Namespace(conference="ICLR", year=2025, where=None)
        result = _build_embeddings_where_clause(args)
        assert "conference = 'ICLR'" in result
        assert "year = 2025" in result
        assert " AND " in result

    def test_where_only(self):
        """Raw --where produces parenthesised clause."""
        args = argparse.Namespace(conference=None, year=None, where="award IS NOT NULL")
        assert _build_embeddings_where_clause(args) == "(award IS NOT NULL)"

    def test_conference_and_where(self):
        """Conference + --where are combined with AND."""
        args = argparse.Namespace(conference="NeurIPS", year=None, where="award IS NOT NULL")
        result = _build_embeddings_where_clause(args)
        assert "conference = 'NeurIPS'" in result
        assert "(award IS NOT NULL)" in result
        assert " AND " in result

    def test_all_filters(self):
        """All three filters are combined with AND."""
        args = argparse.Namespace(conference="NeurIPS", year=2024, where="award IS NOT NULL")
        result = _build_embeddings_where_clause(args)
        assert "conference = 'NeurIPS'" in result
        assert "year = 2024" in result
        assert "(award IS NOT NULL)" in result

    def test_conference_with_quote(self):
        """Single quotes in conference name are escaped."""
        args = argparse.Namespace(conference="O'Reilly", year=None, where=None)
        result = _build_embeddings_where_clause(args)
        assert "O''Reilly" in result

    def test_conference_exact_match(self):
        """Conference name is used as-is (already resolved to canonical form)."""
        args_lower = argparse.Namespace(conference="neurips", year=None, where=None)
        args_upper = argparse.Namespace(conference="NeurIPS", year=None, where=None)
        result_lower = _build_embeddings_where_clause(args_lower)
        result_upper = _build_embeddings_where_clause(args_upper)
        # Exact match — the literal value must match what was passed
        assert "conference = 'neurips'" in result_lower
        assert "conference = 'NeurIPS'" in result_upper


class TestAddConferenceYearArgs:
    """Test cases for the unified add_conference_year_args helper."""

    def test_adds_conference_arg(self):
        """add_conference_year_args adds --conference with correct defaults."""
        parser = argparse.ArgumentParser()
        add_conference_year_args(parser)
        args = parser.parse_args([])
        assert args.conference is None

    def test_adds_year_arg(self):
        """add_conference_year_args adds --year with correct defaults."""
        parser = argparse.ArgumentParser()
        add_conference_year_args(parser)
        args = parser.parse_args([])
        assert args.year is None

    def test_conference_set(self):
        """--conference value is passed through unchanged."""
        parser = argparse.ArgumentParser()
        add_conference_year_args(parser)
        args = parser.parse_args(["--conference", "NeurIPS"])
        assert args.conference == "NeurIPS"

    def test_conference_lowercase(self):
        """--conference accepts lowercase values."""
        parser = argparse.ArgumentParser()
        add_conference_year_args(parser)
        args = parser.parse_args(["--conference", "neurips"])
        assert args.conference == "neurips"

    def test_year_set(self):
        """--year value is converted to int."""
        parser = argparse.ArgumentParser()
        add_conference_year_args(parser)
        args = parser.parse_args(["--year", "2025"])
        assert args.year == 2025


class TestCLI:
    """Test cases for the CLI module."""

    def test_main_no_command(self, capsys):
        """Test main() with no command shows help."""
        with patch.object(sys, "argv", ["neurips-abstracts"]):
            exit_code = main()
            assert exit_code == 1

            captured = capsys.readouterr()
            assert "usage:" in captured.out
            assert "Available commands" in captured.out

    def test_main_help(self, capsys):
        """Test main() with --help shows help."""
        with patch.object(sys, "argv", ["neurips-abstracts", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

            captured = capsys.readouterr()
            assert "usage:" in captured.out
            assert "create-embeddings" in captured.out

    def test_argcomplete_autocomplete_called(self):
        """Test that argcomplete.autocomplete is called during CLI startup."""
        with patch.object(sys, "argv", ["neurips-abstracts"]):
            with patch("abstracts_explorer.cli.argcomplete.autocomplete") as mock_autocomplete:
                main()
                assert mock_autocomplete.called, "argcomplete.autocomplete should be called in main()"
                # Verify it received the argument parser as its first argument
                call_args = mock_autocomplete.call_args
                import argparse

                assert isinstance(call_args[0][0], argparse.ArgumentParser)

    def test_download_command_success(self, tmp_path, capsys):
        """Test download command completes successfully."""
        output_db = tmp_path / "test.db"

        # Set PAPER_DB to output location
        set_test_db(output_db)

        # Mock the plugin and its download method to return LightweightPaper objects
        mock_plugin = Mock()
        mock_plugin.plugin_name = "neurips"
        mock_plugin.plugin_description = "NeurIPS Test Plugin"
        mock_papers = [
            LightweightPaper(
                title="Paper 1",
                abstract="Abstract 1",
                authors=["Author 1"],
                session="Session 1",
                poster_position="P1",
                year=2025,
                conference="NeurIPS",
            ),
            LightweightPaper(
                title="Paper 2",
                abstract="Abstract 2",
                authors=["Author 2"],
                session="Session 2",
                poster_position="P2",
                year=2025,
                conference="NeurIPS",
            ),
        ]
        mock_plugin.download.return_value = mock_papers

        with patch("abstracts_explorer.cli.get_plugin") as mock_get_plugin:
            mock_get_plugin.return_value = mock_plugin

            with patch.object(
                sys,
                "argv",
                [
                    "neurips-abstracts",
                    "download",
                    "--conference",
                    "neurips",
                    "--year",
                    "2025",
                    "--output",
                    str(output_db),
                ],
            ):
                exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Downloaded 2 papers" in captured.out
        assert "Database updated:" in captured.out
        assert output_db.exists()

    def test_download_command_failure(self, tmp_path, capsys):
        """Test download command handles errors gracefully."""
        output_db = tmp_path / "test.db"

        # Set PAPER_DB to output location
        set_test_db(output_db)

        # Mock the plugin to raise an exception
        mock_plugin = Mock()
        mock_plugin.plugin_name = "neurips"
        mock_plugin.plugin_description = "NeurIPS Test Plugin"
        mock_plugin.supported_years = [2025]
        mock_plugin.download.side_effect = Exception("Network error")

        with patch("abstracts_explorer.cli.get_plugin") as mock_get_plugin:
            mock_get_plugin.return_value = mock_plugin

            with patch.object(
                sys,
                "argv",
                ["neurips-abstracts", "download", "--conference", "neurips", "--output", str(output_db)],
            ):
                exit_code = main()

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Error downloading neurips" in captured.err

    def test_download_command_with_database_url(self, tmp_path, capsys):
        """Test download command uses PAPER_DB when set."""
        # Create a temporary SQLite database for testing
        db_path = tmp_path / "test.db"

        # Set PAPER_DB environment variable
        set_test_db(db_path)

        # Mock the plugin and its download method
        mock_plugin = Mock()
        mock_plugin.plugin_name = "neurips"
        mock_plugin.plugin_description = "NeurIPS Test Plugin"
        mock_papers = [
            LightweightPaper(
                title="Paper 1",
                abstract="Abstract 1",
                authors=["Author 1"],
                session="Session 1",
                poster_position="P1",
                year=2025,
                conference="NeurIPS",
            ),
        ]
        mock_plugin.download.return_value = mock_papers

        # Mock get_plugin to return the plugin
        with patch("abstracts_explorer.cli.get_plugin") as mock_get_plugin:
            mock_get_plugin.return_value = mock_plugin

            with patch.object(
                sys,
                "argv",
                [
                    "neurips-abstracts",
                    "download",
                    "--conference",
                    "neurips",
                    "--year",
                    "2025",
                    "--output",
                    "ignored_path.db",
                ],
            ):
                exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Downloaded 1 papers" in captured.out or "Downloaded 1 paper" in captured.out
        # Should use PAPER_DB, not the --output path
        assert "Database updated:" in captured.out
        # Verify database was created
        assert db_path.exists()

    def test_download_all_conferences(self, tmp_path, capsys):
        """Test download command without --conference downloads all plugins."""
        output_db = tmp_path / "test.db"
        set_test_db(output_db)

        mock_plugin_a = Mock()
        mock_plugin_a.plugin_name = "plugina"
        mock_plugin_a.plugin_description = "Plugin A"
        mock_plugin_a.supported_years = [2024, 2025]
        mock_plugin_a.download.return_value = [
            LightweightPaper(
                title="Paper A",
                abstract="Abstract A",
                authors=["Author A"],
                session="S1",
                poster_position="P1",
                year=2024,
                conference="A",
            ),
        ]

        mock_plugin_b = Mock()
        mock_plugin_b.plugin_name = "pluginb"
        mock_plugin_b.plugin_description = "Plugin B"
        mock_plugin_b.supported_years = [2025]
        mock_plugin_b.download.return_value = [
            LightweightPaper(
                title="Paper B",
                abstract="Abstract B",
                authors=["Author B"],
                session="S2",
                poster_position="P2",
                year=2025,
                conference="B",
            ),
        ]

        with patch("abstracts_explorer.cli.get_all_plugins", return_value=[mock_plugin_a, mock_plugin_b]):
            with patch.object(
                sys,
                "argv",
                ["abstracts-explorer", "download", "--output", str(output_db)],
            ):
                exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        # Should download from both plugins
        assert "plugina" in captured.out
        assert "pluginb" in captured.out
        # Plugin A should be called for both 2024 and 2025
        assert mock_plugin_a.download.call_count == 2
        # Plugin B should be called once for 2025
        assert mock_plugin_b.download.call_count == 1
        assert "Total papers downloaded:" in captured.out

    def test_download_conference_all_years(self, tmp_path, capsys):
        """Test download with --conference but no --year downloads all supported years."""
        output_db = tmp_path / "test.db"
        set_test_db(output_db)

        mock_plugin = Mock()
        mock_plugin.plugin_name = "neurips"
        mock_plugin.plugin_description = "NeurIPS Test Plugin"
        mock_plugin.supported_years = [2023, 2024, 2025]
        mock_plugin.download.return_value = [
            LightweightPaper(
                title="Paper 1",
                abstract="Abstract 1",
                authors=["Author 1"],
                session="S1",
                poster_position="P1",
                year=2025,
                conference="NeurIPS",
            ),
        ]

        with patch("abstracts_explorer.cli.get_plugin") as mock_get_plugin:
            mock_get_plugin.return_value = mock_plugin

            with patch.object(
                sys,
                "argv",
                ["abstracts-explorer", "download", "--conference", "neurips", "--output", str(output_db)],
            ):
                exit_code = main()

        assert exit_code == 0
        # Should download for all 3 years
        assert mock_plugin.download.call_count == 3
        captured = capsys.readouterr()
        assert "2023, 2024, 2025" in captured.out

    def test_download_conference_with_year(self, tmp_path, capsys):
        """Test that --conference with --year downloads a specific conference/year."""
        output_db = tmp_path / "test.db"
        set_test_db(output_db)

        mock_plugin = Mock()
        mock_plugin.plugin_name = "neurips"
        mock_plugin.plugin_description = "NeurIPS Test Plugin"
        mock_plugin.supported_years = [2025]
        mock_plugin.download.return_value = [
            LightweightPaper(
                title="Paper 1",
                abstract="Abstract 1",
                authors=["Author 1"],
                session="S1",
                poster_position="P1",
                year=2025,
                conference="NeurIPS",
            ),
        ]

        with patch("abstracts_explorer.cli.get_plugin") as mock_get_plugin:
            mock_get_plugin.return_value = mock_plugin

            with patch.object(
                sys,
                "argv",
                [
                    "abstracts-explorer",
                    "download",
                    "--conference",
                    "neurips",
                    "--year",
                    "2025",
                    "--output",
                    str(output_db),
                ],
            ):
                exit_code = main()

        assert exit_code == 0
        mock_get_plugin.assert_called_once_with("neurips")

    def test_download_partial_failure(self, tmp_path, capsys):
        """Test that download continues after a single year fails."""
        output_db = tmp_path / "test.db"
        set_test_db(output_db)

        mock_plugin = Mock()
        mock_plugin.plugin_name = "neurips"
        mock_plugin.plugin_description = "NeurIPS Test Plugin"
        mock_plugin.supported_years = [2024, 2025]

        papers_2025 = [
            LightweightPaper(
                title="Paper 1",
                abstract="Abstract 1",
                authors=["Author 1"],
                session="S1",
                poster_position="P1",
                year=2025,
                conference="NeurIPS",
            ),
        ]
        # First call (2024) fails, second call (2025) succeeds
        mock_plugin.download.side_effect = [Exception("Network error"), papers_2025]

        with patch("abstracts_explorer.cli.get_plugin") as mock_get_plugin:
            mock_get_plugin.return_value = mock_plugin

            with patch.object(
                sys,
                "argv",
                ["abstracts-explorer", "download", "--conference", "neurips", "--output", str(output_db)],
            ):
                exit_code = main()

        # Should return 1 because there was at least one error
        assert exit_code == 1
        captured = capsys.readouterr()
        # But the 2025 download should have succeeded
        assert "Downloaded 1 papers" in captured.out
        assert "error(s) occurred" in captured.err

    def test_download_conference_case_insensitive(self, tmp_path, capsys):
        """Test download command accepts conference names in any case."""
        output_db = tmp_path / "test.db"
        set_test_db(output_db)

        mock_plugin = Mock()
        mock_plugin.plugin_name = "neurips"
        mock_plugin.plugin_description = "NeurIPS Test Plugin"
        mock_plugin.download.return_value = [
            LightweightPaper(
                title="Paper 1",
                abstract="Abstract 1",
                authors=["Author 1"],
                session="S1",
                poster_position="P1",
                year=2025,
                conference="NeurIPS",
            )
        ]

        # Pass "NeurIPS" (mixed case) – should still find the lowercase plugin "neurips"
        with patch("abstracts_explorer.cli.get_plugin") as mock_get_plugin:
            mock_get_plugin.return_value = mock_plugin

            with patch.object(
                sys,
                "argv",
                [
                    "abstracts-explorer",
                    "download",
                    "--conference",
                    "NeurIPS",
                    "--year",
                    "2025",
                    "--output",
                    str(output_db),
                ],
            ):
                exit_code = main()

        assert exit_code == 0
        # get_plugin must have been called with the lowercased name
        mock_get_plugin.assert_called_once_with("neurips")

    def test_pre_process_command_success(self, tmp_path, capsys, monkeypatch):
        """Test pre-process command chains download → embeddings → clustering."""
        patch_get_config_for_test(monkeypatch, tmp_path / "embeddings")

        with (
            patch("abstracts_explorer.cli.download_command") as mock_download,
            patch("abstracts_explorer.cli.create_embeddings_command") as mock_embed,
            patch("abstracts_explorer.cli.pre_generate_clustering_command") as mock_cluster,
        ):
            mock_download.return_value = 0
            mock_embed.return_value = 0
            mock_cluster.return_value = 0

            args = argparse.Namespace(conference="neurips", year=2025, force=False, verbose=0)
            rc = pre_process_command(args)

        assert rc == 0
        captured = capsys.readouterr()
        assert "Step 1/3" in captured.out
        assert "Step 2/3" in captured.out
        assert "Step 3/3" in captured.out
        assert "completed successfully" in captured.out

        # Verify each sub-command was called
        mock_download.assert_called_once()
        mock_embed.assert_called_once()
        mock_cluster.assert_called_once()

        # Conference and year must be forwarded
        download_args = mock_download.call_args[0][0]
        assert download_args.conference == "neurips"
        assert download_args.year == 2025

    def test_pre_process_command_download_fails(self, tmp_path, capsys, monkeypatch):
        """Test pre-process command stops on download failure."""
        patch_get_config_for_test(monkeypatch, tmp_path / "embeddings")

        with (
            patch("abstracts_explorer.cli.download_command") as mock_download,
            patch("abstracts_explorer.cli.create_embeddings_command") as mock_embed,
            patch("abstracts_explorer.cli.pre_generate_clustering_command") as mock_cluster,
        ):
            mock_download.return_value = 1  # failure
            mock_embed.return_value = 0
            mock_cluster.return_value = 0

            args = argparse.Namespace(conference=None, year=None, force=False, verbose=0)
            rc = pre_process_command(args)

        assert rc == 1
        mock_embed.assert_not_called()
        mock_cluster.assert_not_called()

    def test_pre_process_command_via_main(self, tmp_path, capsys, monkeypatch):
        """Test pre-process command is dispatched correctly via main()."""
        patch_get_config_for_test(monkeypatch, tmp_path / "embeddings")

        with (
            patch("abstracts_explorer.cli.download_command") as mock_download,
            patch("abstracts_explorer.cli.create_embeddings_command") as mock_embed,
            patch("abstracts_explorer.cli.pre_generate_clustering_command") as mock_cluster,
        ):
            mock_download.return_value = 0
            mock_embed.return_value = 0
            mock_cluster.return_value = 0

            with patch.object(
                sys, "argv", ["abstracts-explorer", "pre-process", "--conference", "iclr", "--year", "2024"]
            ):
                rc = main()

        assert rc == 0
        mock_download.assert_called_once()
        mock_embed.assert_called_once()
        mock_cluster.assert_called_once()

        download_args = mock_download.call_args[0][0]
        assert download_args.conference == "iclr"
        assert download_args.year == 2024
        """Test create-embeddings with non-existent database."""
        nonexistent_db = tmp_path / "nonexistent.db"
        set_test_db(nonexistent_db)

        # The exception will be raised and not caught, so we expect it to propagate
        with patch.object(
            sys,
            "argv",
            ["neurips-abstracts", "create-embeddings"],
        ):
            # Should raise DatabaseError because database/tables don't exist
            with pytest.raises(Exception) as exc_info:
                main()

        # Verify it's a database-related error
        assert "table" in str(exc_info.value).lower() or "database" in str(exc_info.value).lower()

    def test_create_embeddings_lm_studio_not_available(self, tmp_path, capsys, monkeypatch):
        """Test create-embeddings when OpenAI API is not available."""
        # Create a test database
        from abstracts_explorer import DatabaseManager

        db_path = tmp_path / "test.db"
        set_test_db(db_path)
        with DatabaseManager() as db:
            db.create_tables()
            papers = [
                LightweightPaper(
                    title="Test",
                    abstract="Abstract",
                    authors=["Test Author"],
                    session="Test Session",
                    poster_position="P1",
                    year=2025,
                    conference="NeurIPS",
                )
            ]
            db.add_papers(papers)

        # Set EMBEDDING_DB and patch config reload
        patch_get_config_for_test(monkeypatch, tmp_path / "embeddings")

        # Mock OpenAI API connection failure
        with patch("abstracts_explorer.cli.EmbeddingsManager") as MockEM:
            mock_em = Mock()
            mock_em.check_model_compatibility.return_value = (True, None, "test-model")
            mock_em.test_lm_studio_connection.return_value = False
            MockEM.return_value = mock_em

            with patch.object(
                sys,
                "argv",
                [
                    "neurips-abstracts",
                    "create-embeddings",
                ],
            ):
                exit_code = main()

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Failed to connect to OpenAI API" in captured.err

    def test_create_embeddings_success(self, tmp_path, capsys, monkeypatch):
        """Test create-embeddings command completes successfully."""
        from abstracts_explorer import DatabaseManager

        # Create a test database
        db_path = tmp_path / "test.db"
        set_test_db(db_path)
        with DatabaseManager() as db:
            db.create_tables()
            papers = [
                LightweightPaper(
                    title="Paper 1",
                    abstract="Abstract 1",
                    authors=["Test Author"],
                    session="Test Session",
                    poster_position="P1",
                    year=2025,
                    conference="NeurIPS",
                ),
                LightweightPaper(
                    title="Paper 2",
                    abstract="Abstract 2",
                    authors=["Test Author"],
                    session="Test Session",
                    poster_position="P1",
                    year=2025,
                    conference="NeurIPS",
                ),
            ]
            db.add_papers(papers)

        # Set EMBEDDING_DB and patch config reload
        patch_get_config_for_test(monkeypatch, tmp_path / "embeddings")

        # Mock embeddings manager
        with patch("abstracts_explorer.cli.EmbeddingsManager") as MockEM:
            mock_em = Mock()
            mock_em.check_model_compatibility.return_value = (True, None, "test-model")
            mock_em.test_lm_studio_connection.return_value = True
            mock_em.embed_from_database.return_value = 2
            mock_em.get_collection_stats.return_value = {"name": "test", "count": 2}
            mock_em.__enter__ = Mock(return_value=mock_em)
            mock_em.__exit__ = Mock(return_value=False)
            MockEM.return_value = mock_em

            with patch.object(
                sys,
                "argv",
                [
                    "neurips-abstracts",
                    "create-embeddings",
                ],
            ):
                exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Successfully generated embeddings for 2 abstracts" in captured.out
        assert "Vector database saved to" in captured.out

    def test_create_embeddings_with_where_clause(self, tmp_path, capsys, monkeypatch):
        """Test create-embeddings with WHERE clause filter."""
        from abstracts_explorer import DatabaseManager

        # Create a test database
        db_path = tmp_path / "test.db"
        set_test_db(db_path)
        with DatabaseManager() as db:
            db.create_tables()
            papers = [
                LightweightPaper(
                    title="Paper 1",
                    authors=["Author 1"],
                    abstract="Abstract 1",
                    session="Session 1",
                    poster_position="P1",
                    year=2025,
                    conference="NeurIPS",
                    award="Best Paper",
                ),
                LightweightPaper(
                    title="Paper 2",
                    authors=["Author 2"],
                    abstract="Abstract 2",
                    session="Session 2",
                    poster_position="P2",
                    year=2025,
                    conference="NeurIPS",
                    award=None,
                ),
            ]
            db.add_papers(papers)

        # Set EMBEDDING_DB and patch config reload
        patch_get_config_for_test(monkeypatch, tmp_path / "embeddings")

        # Mock embeddings manager
        with patch("abstracts_explorer.cli.EmbeddingsManager") as MockEM:
            mock_em = Mock()
            mock_em.check_model_compatibility.return_value = (True, None, "test-model")
            mock_em.test_lm_studio_connection.return_value = True
            mock_em.embed_from_database.return_value = 1
            mock_em.get_collection_stats.return_value = {"name": "test", "count": 1}
            mock_em.__enter__ = Mock(return_value=mock_em)
            mock_em.__exit__ = Mock(return_value=False)
            MockEM.return_value = mock_em

            with patch.object(
                sys,
                "argv",
                [
                    "neurips-abstracts",
                    "create-embeddings",
                    "--where",
                    "award IS NOT NULL",
                ],
            ):
                exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Filter will process 1 abstracts" in captured.out
        assert "Successfully generated embeddings for 1 abstracts" in captured.out

    def test_create_embeddings_force_flag(self, tmp_path, capsys, monkeypatch):
        """Test create-embeddings with --force flag."""
        from abstracts_explorer import DatabaseManager

        # Create a test database
        db_path = tmp_path / "test.db"
        set_test_db(db_path)
        with DatabaseManager() as db:
            db.create_tables()
            papers = [
                LightweightPaper(
                    title="Test",
                    abstract="Abstract",
                    authors=["Test Author"],
                    session="Test Session",
                    poster_position="P1",
                    year=2025,
                    conference="NeurIPS",
                )
            ]
            db.add_papers(papers)

        embeddings_path = tmp_path / "embeddings"
        embeddings_path.mkdir()

        # Set EMBEDDING_DB and patch config reload
        patch_get_config_for_test(monkeypatch, embeddings_path)

        # Mock embeddings manager
        with patch("abstracts_explorer.cli.EmbeddingsManager") as MockEM:
            mock_em = Mock()
            mock_em.check_model_compatibility.return_value = (True, None, "test-model")
            mock_em.test_lm_studio_connection.return_value = True
            mock_em.embed_from_database.return_value = 1
            mock_em.get_collection_stats.return_value = {"name": "test", "count": 1}
            mock_em.__enter__ = Mock(return_value=mock_em)
            mock_em.__exit__ = Mock(return_value=False)
            MockEM.return_value = mock_em

            with patch.object(
                sys,
                "argv",
                [
                    "neurips-abstracts",
                    "create-embeddings",
                    "--force",
                ],
            ):
                exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Resetting existing collection" in captured.out

        # Verify create_collection was called with reset=True
        mock_em.create_collection.assert_called_once_with(reset=True)

    def test_create_embeddings_force_with_conference_does_not_reset_collection(self, tmp_path, capsys, monkeypatch):
        """--force --conference should delete only matching embeddings, not reset whole collection."""
        from abstracts_explorer import DatabaseManager

        db_path = tmp_path / "test.db"
        set_test_db(db_path)
        with DatabaseManager() as db:
            db.create_tables()
            papers = [
                LightweightPaper(
                    title="NeurIPS Paper",
                    abstract="Abstract 1",
                    authors=["Author"],
                    session="S1",
                    poster_position="P1",
                    year=2025,
                    conference="NeurIPS",
                ),
            ]
            db.add_papers(papers)

        patch_get_config_for_test(monkeypatch, tmp_path / "embeddings")

        with patch("abstracts_explorer.cli.EmbeddingsManager") as MockEM:
            mock_em = Mock()
            mock_em.check_model_compatibility.return_value = (True, None, "test-model")
            mock_em.test_lm_studio_connection.return_value = True
            mock_em.embed_from_database.return_value = 1
            mock_em.get_collection_stats.return_value = {"name": "test", "count": 1}
            mock_em.delete_embeddings_by_filter.return_value = 0
            mock_em.__enter__ = Mock(return_value=mock_em)
            mock_em.__exit__ = Mock(return_value=False)
            MockEM.return_value = mock_em

            with patch.object(
                sys,
                "argv",
                [
                    "neurips-abstracts",
                    "create-embeddings",
                    "--force",
                    "--conference",
                    "NeurIPS",
                ],
            ):
                exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        # Should NOT say "Resetting existing collection" (that implies full reset)
        assert "Resetting existing collection" not in captured.out
        # Should print scoped removal message
        assert "Removing existing embeddings" in captured.out
        assert "conference=NeurIPS" in captured.out

        # create_collection must be called with reset=False (not wipe the collection)
        mock_em.create_collection.assert_called_once_with(reset=False)
        # delete_embeddings_by_filter must be called with the conference
        mock_em.delete_embeddings_by_filter.assert_called_once_with(conference="NeurIPS", year=None)

    def test_create_embeddings_force_with_year_does_not_reset_collection(self, tmp_path, capsys, monkeypatch):
        """--force --year should delete only matching embeddings, not reset whole collection."""
        from abstracts_explorer import DatabaseManager

        db_path = tmp_path / "test.db"
        set_test_db(db_path)
        with DatabaseManager() as db:
            db.create_tables()
            papers = [
                LightweightPaper(
                    title="Paper 2024",
                    abstract="Abstract",
                    authors=["Author"],
                    session="S1",
                    poster_position="P1",
                    year=2024,
                    conference="NeurIPS",
                ),
            ]
            db.add_papers(papers)

        patch_get_config_for_test(monkeypatch, tmp_path / "embeddings")

        with patch("abstracts_explorer.cli.EmbeddingsManager") as MockEM:
            mock_em = Mock()
            mock_em.check_model_compatibility.return_value = (True, None, "test-model")
            mock_em.test_lm_studio_connection.return_value = True
            mock_em.embed_from_database.return_value = 1
            mock_em.get_collection_stats.return_value = {"name": "test", "count": 1}
            mock_em.delete_embeddings_by_filter.return_value = 2
            mock_em.__enter__ = Mock(return_value=mock_em)
            mock_em.__exit__ = Mock(return_value=False)
            MockEM.return_value = mock_em

            with patch.object(
                sys,
                "argv",
                [
                    "neurips-abstracts",
                    "create-embeddings",
                    "--force",
                    "--year",
                    "2024",
                ],
            ):
                exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Resetting existing collection" not in captured.out
        assert "Removing existing embeddings" in captured.out
        assert "year=2024" in captured.out
        # Reports the number removed
        assert "Removed 2" in captured.out

        mock_em.create_collection.assert_called_once_with(reset=False)
        mock_em.delete_embeddings_by_filter.assert_called_once_with(conference=None, year=2024)

    def test_create_embeddings_force_with_conference_and_year(self, tmp_path, capsys, monkeypatch):
        """--force --conference X --year Y deletes only that conference+year slice."""
        from abstracts_explorer import DatabaseManager

        db_path = tmp_path / "test.db"
        set_test_db(db_path)
        with DatabaseManager() as db:
            db.create_tables()
            papers = [
                LightweightPaper(
                    title="NeurIPS 2024",
                    abstract="Abstract",
                    authors=["Author"],
                    session="S1",
                    poster_position="P1",
                    year=2024,
                    conference="NeurIPS",
                ),
            ]
            db.add_papers(papers)

        patch_get_config_for_test(monkeypatch, tmp_path / "embeddings")

        with patch("abstracts_explorer.cli.EmbeddingsManager") as MockEM:
            mock_em = Mock()
            mock_em.check_model_compatibility.return_value = (True, None, "test-model")
            mock_em.test_lm_studio_connection.return_value = True
            mock_em.embed_from_database.return_value = 1
            mock_em.get_collection_stats.return_value = {"name": "test", "count": 1}
            mock_em.delete_embeddings_by_filter.return_value = 5
            mock_em.__enter__ = Mock(return_value=mock_em)
            mock_em.__exit__ = Mock(return_value=False)
            MockEM.return_value = mock_em

            with patch.object(
                sys,
                "argv",
                [
                    "neurips-abstracts",
                    "create-embeddings",
                    "--force",
                    "--conference",
                    "NeurIPS",
                    "--year",
                    "2024",
                ],
            ):
                exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Resetting existing collection" not in captured.out
        assert "Removing existing embeddings" in captured.out
        assert "conference=NeurIPS" in captured.out
        assert "year=2024" in captured.out

        mock_em.create_collection.assert_called_once_with(reset=False)
        mock_em.delete_embeddings_by_filter.assert_called_once_with(conference="NeurIPS", year=2024)

        """Test create-embeddings with --requests-per-minute flag."""
        from abstracts_explorer import DatabaseManager

        # Create a test database
        db_path = tmp_path / "test.db"
        set_test_db(db_path)
        with DatabaseManager() as db:
            db.create_tables()
            papers = [
                LightweightPaper(
                    title="Test",
                    abstract="Abstract",
                    authors=["Test Author"],
                    session="Test Session",
                    poster_position="P1",
                    year=2025,
                    conference="NeurIPS",
                )
            ]
            db.add_papers(papers)

        patch_get_config_for_test(monkeypatch, tmp_path / "embeddings")

        with patch("abstracts_explorer.cli.EmbeddingsManager") as MockEM:
            mock_em = Mock()
            mock_em.check_model_compatibility.return_value = (True, None, "test-model")
            mock_em.test_lm_studio_connection.return_value = True
            mock_em.embed_from_database.return_value = 1
            mock_em.get_collection_stats.return_value = {"name": "test", "count": 1}
            mock_em.__enter__ = Mock(return_value=mock_em)
            mock_em.__exit__ = Mock(return_value=False)
            MockEM.return_value = mock_em

            with patch.object(
                sys,
                "argv",
                [
                    "neurips-abstracts",
                    "create-embeddings",
                    "--requests-per-minute",
                    "30",
                ],
            ):
                exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "30 req/min" in captured.out

        # Verify EmbeddingsManager was initialized with requests_per_minute=30
        MockEM.assert_called_once()
        call_kwargs = MockEM.call_args.kwargs
        assert call_kwargs["requests_per_minute"] == 30

    def test_create_embeddings_requests_per_minute_zero(self, tmp_path, capsys, monkeypatch):
        """Test create-embeddings with --requests-per-minute 0 disables rate limiting."""
        from abstracts_explorer import DatabaseManager

        db_path = tmp_path / "test.db"
        set_test_db(db_path)
        with DatabaseManager() as db:
            db.create_tables()
            papers = [
                LightweightPaper(
                    title="Test",
                    abstract="Abstract",
                    authors=["Test Author"],
                    session="Test Session",
                    poster_position="P1",
                    year=2025,
                    conference="NeurIPS",
                )
            ]
            db.add_papers(papers)

        patch_get_config_for_test(monkeypatch, tmp_path / "embeddings")

        with patch("abstracts_explorer.cli.EmbeddingsManager") as MockEM:
            mock_em = Mock()
            mock_em.check_model_compatibility.return_value = (True, None, "test-model")
            mock_em.test_lm_studio_connection.return_value = True
            mock_em.embed_from_database.return_value = 1
            mock_em.get_collection_stats.return_value = {"name": "test", "count": 1}
            mock_em.__enter__ = Mock(return_value=mock_em)
            mock_em.__exit__ = Mock(return_value=False)
            MockEM.return_value = mock_em

            with patch.object(
                sys,
                "argv",
                [
                    "neurips-abstracts",
                    "create-embeddings",
                    "--requests-per-minute",
                    "0",
                ],
            ):
                exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "disabled" in captured.out

        MockEM.assert_called_once()
        call_kwargs = MockEM.call_args.kwargs
        assert call_kwargs["requests_per_minute"] == 0

    def test_create_embeddings_custom_model(self, tmp_path, capsys):
        """Test create-embeddings with custom model settings."""
        from abstracts_explorer import DatabaseManager

        # Create a test database
        db_path = tmp_path / "test.db"
        set_test_db(db_path)
        with DatabaseManager() as db:
            db.create_tables()
            papers = [
                LightweightPaper(
                    title="Test",
                    abstract="Abstract",
                    authors=["Test Author"],
                    session="Test Session",
                    poster_position="P1",
                    year=2025,
                    conference="NeurIPS",
                )
            ]
            db.add_papers(papers)

        custom_url = "http://localhost:5000"
        custom_model = "custom-embedding-model"

        # Mock embeddings manager
        with patch("abstracts_explorer.cli.EmbeddingsManager") as MockEM:
            mock_em = Mock()
            mock_em.check_model_compatibility.return_value = (True, None, custom_model)
            mock_em.test_lm_studio_connection.return_value = True
            mock_em.embed_from_database.return_value = 1
            mock_em.get_collection_stats.return_value = {"name": "test", "count": 1}
            mock_em.__enter__ = Mock(return_value=mock_em)
            mock_em.__exit__ = Mock(return_value=False)
            MockEM.return_value = mock_em

            with patch.object(
                sys,
                "argv",
                [
                    "neurips-abstracts",
                    "create-embeddings",
                    "--lm-studio-url",
                    custom_url,
                    "--model",
                    custom_model,
                ],
            ):
                exit_code = main()

        assert exit_code == 0

        # Verify EmbeddingsManager was initialized with custom settings
        MockEM.assert_called_once()
        call_kwargs = MockEM.call_args.kwargs
        assert call_kwargs["lm_studio_url"] == custom_url
        assert call_kwargs["model_name"] == custom_model

    def test_create_embeddings_embeddings_error(self, tmp_path, capsys):
        """Test create-embeddings handles EmbeddingsError gracefully."""
        from abstracts_explorer import DatabaseManager
        from abstracts_explorer.embeddings import EmbeddingsError

        # Create a test database
        db_path = tmp_path / "test.db"
        set_test_db(db_path)
        with DatabaseManager() as db:
            db.create_tables()
            papers = [
                LightweightPaper(
                    title="Test",
                    abstract="Abstract",
                    authors=["Test Author"],
                    session="Test Session",
                    poster_position="P1",
                    year=2025,
                    conference="NeurIPS",
                )
            ]
            db.add_papers(papers)

        # Mock embeddings manager to raise error
        with patch("abstracts_explorer.cli.EmbeddingsManager") as MockEM:
            mock_em = Mock()
            mock_em.check_model_compatibility.return_value = (True, None, "test-model")
            mock_em.test_lm_studio_connection.return_value = True
            mock_em.connect.side_effect = EmbeddingsError("Connection failed")
            MockEM.return_value = mock_em

            with patch.object(
                sys,
                "argv",
                ["neurips-abstracts", "create-embeddings"],
            ):
                exit_code = main()

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Embeddings error:" in captured.err

    def test_create_embeddings_with_conference(self, tmp_path, capsys, monkeypatch):
        """Test create-embeddings with --conference flag filters to one conference."""
        from abstracts_explorer import DatabaseManager

        db_path = tmp_path / "test.db"
        set_test_db(db_path)
        with DatabaseManager() as db:
            db.create_tables()
            papers = [
                LightweightPaper(
                    title="NeurIPS Paper",
                    abstract="Abstract 1",
                    authors=["Author"],
                    session="S1",
                    poster_position="P1",
                    year=2025,
                    conference="NeurIPS",
                ),
                LightweightPaper(
                    title="ICLR Paper",
                    abstract="Abstract 2",
                    authors=["Author"],
                    session="S2",
                    poster_position="P2",
                    year=2025,
                    conference="ICLR",
                ),
            ]
            db.add_papers(papers)

        patch_get_config_for_test(monkeypatch, tmp_path / "embeddings")

        with patch("abstracts_explorer.cli.EmbeddingsManager") as MockEM:
            mock_em = Mock()
            mock_em.check_model_compatibility.return_value = (True, None, "test-model")
            mock_em.test_lm_studio_connection.return_value = True
            mock_em.embed_from_database.return_value = 1
            mock_em.get_collection_stats.return_value = {"name": "test", "count": 1}
            mock_em.__enter__ = Mock(return_value=mock_em)
            mock_em.__exit__ = Mock(return_value=False)
            MockEM.return_value = mock_em

            with patch.object(
                sys,
                "argv",
                [
                    "neurips-abstracts",
                    "create-embeddings",
                    "--conference",
                    "NeurIPS",
                ],
            ):
                exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Conference: NeurIPS" in captured.out
        assert "Filter will process 1 abstracts" in captured.out

        # Verify the WHERE clause was passed to embed_from_database
        call_kwargs = mock_em.embed_from_database.call_args.kwargs
        assert "conference = 'NeurIPS'" in call_kwargs["where_clause"]

    def test_create_embeddings_with_year(self, tmp_path, capsys, monkeypatch):
        """Test create-embeddings with --year flag filters to one year."""
        from abstracts_explorer import DatabaseManager

        db_path = tmp_path / "test.db"
        set_test_db(db_path)
        with DatabaseManager() as db:
            db.create_tables()
            papers = [
                LightweightPaper(
                    title="Paper 2024",
                    abstract="Abstract 1",
                    authors=["Author"],
                    session="S1",
                    poster_position="P1",
                    year=2024,
                    conference="NeurIPS",
                ),
                LightweightPaper(
                    title="Paper 2025",
                    abstract="Abstract 2",
                    authors=["Author"],
                    session="S2",
                    poster_position="P2",
                    year=2025,
                    conference="NeurIPS",
                ),
            ]
            db.add_papers(papers)

        patch_get_config_for_test(monkeypatch, tmp_path / "embeddings")

        with patch("abstracts_explorer.cli.EmbeddingsManager") as MockEM:
            mock_em = Mock()
            mock_em.check_model_compatibility.return_value = (True, None, "test-model")
            mock_em.test_lm_studio_connection.return_value = True
            mock_em.embed_from_database.return_value = 1
            mock_em.get_collection_stats.return_value = {"name": "test", "count": 1}
            mock_em.__enter__ = Mock(return_value=mock_em)
            mock_em.__exit__ = Mock(return_value=False)
            MockEM.return_value = mock_em

            with patch.object(
                sys,
                "argv",
                [
                    "neurips-abstracts",
                    "create-embeddings",
                    "--year",
                    "2024",
                ],
            ):
                exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Year:       2024" in captured.out
        assert "Filter will process 1 abstracts" in captured.out

        call_kwargs = mock_em.embed_from_database.call_args.kwargs
        assert "year = 2024" in call_kwargs["where_clause"]

    def test_create_embeddings_with_conference_and_year(self, tmp_path, capsys, monkeypatch):
        """Test create-embeddings with both --conference and --year flags."""
        from abstracts_explorer import DatabaseManager

        db_path = tmp_path / "test.db"
        set_test_db(db_path)
        with DatabaseManager() as db:
            db.create_tables()
            papers = [
                LightweightPaper(
                    title="NeurIPS 2024",
                    abstract="Abstract 1",
                    authors=["Author"],
                    session="S1",
                    poster_position="P1",
                    year=2024,
                    conference="NeurIPS",
                ),
                LightweightPaper(
                    title="NeurIPS 2025",
                    abstract="Abstract 2",
                    authors=["Author"],
                    session="S2",
                    poster_position="P2",
                    year=2025,
                    conference="NeurIPS",
                ),
                LightweightPaper(
                    title="ICLR 2024",
                    abstract="Abstract 3",
                    authors=["Author"],
                    session="S3",
                    poster_position="P3",
                    year=2024,
                    conference="ICLR",
                ),
            ]
            db.add_papers(papers)

        patch_get_config_for_test(monkeypatch, tmp_path / "embeddings")

        with patch("abstracts_explorer.cli.EmbeddingsManager") as MockEM:
            mock_em = Mock()
            mock_em.check_model_compatibility.return_value = (True, None, "test-model")
            mock_em.test_lm_studio_connection.return_value = True
            mock_em.embed_from_database.return_value = 1
            mock_em.get_collection_stats.return_value = {"name": "test", "count": 1}
            mock_em.__enter__ = Mock(return_value=mock_em)
            mock_em.__exit__ = Mock(return_value=False)
            MockEM.return_value = mock_em

            with patch.object(
                sys,
                "argv",
                [
                    "neurips-abstracts",
                    "create-embeddings",
                    "--conference",
                    "NeurIPS",
                    "--year",
                    "2024",
                ],
            ):
                exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Conference: NeurIPS" in captured.out
        assert "Year:       2024" in captured.out
        assert "Filter will process 1 abstracts" in captured.out

        call_kwargs = mock_em.embed_from_database.call_args.kwargs
        where = call_kwargs["where_clause"]
        assert "conference = 'NeurIPS'" in where
        assert "year = 2024" in where

    def test_create_embeddings_conference_with_where(self, tmp_path, capsys, monkeypatch):
        """Test create-embeddings combining --conference with --where clause."""
        from abstracts_explorer import DatabaseManager

        db_path = tmp_path / "test.db"
        set_test_db(db_path)
        with DatabaseManager() as db:
            db.create_tables()
            papers = [
                LightweightPaper(
                    title="Award Paper",
                    abstract="Abstract 1",
                    authors=["Author"],
                    session="S1",
                    poster_position="P1",
                    year=2025,
                    conference="NeurIPS",
                    award="Best Paper",
                ),
                LightweightPaper(
                    title="Regular Paper",
                    abstract="Abstract 2",
                    authors=["Author"],
                    session="S2",
                    poster_position="P2",
                    year=2025,
                    conference="NeurIPS",
                ),
            ]
            db.add_papers(papers)

        patch_get_config_for_test(monkeypatch, tmp_path / "embeddings")

        with patch("abstracts_explorer.cli.EmbeddingsManager") as MockEM:
            mock_em = Mock()
            mock_em.check_model_compatibility.return_value = (True, None, "test-model")
            mock_em.test_lm_studio_connection.return_value = True
            mock_em.embed_from_database.return_value = 1
            mock_em.get_collection_stats.return_value = {"name": "test", "count": 1}
            mock_em.__enter__ = Mock(return_value=mock_em)
            mock_em.__exit__ = Mock(return_value=False)
            MockEM.return_value = mock_em

            with patch.object(
                sys,
                "argv",
                [
                    "neurips-abstracts",
                    "create-embeddings",
                    "--conference",
                    "NeurIPS",
                    "--where",
                    "award IS NOT NULL",
                ],
            ):
                exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Filter will process 1 abstracts" in captured.out

        call_kwargs = mock_em.embed_from_database.call_args.kwargs
        where = call_kwargs["where_clause"]
        assert "conference = 'NeurIPS'" in where
        assert "(award IS NOT NULL)" in where

    def test_create_embeddings_default_embeds_all(self, tmp_path, capsys, monkeypatch):
        """Test create-embeddings without arguments embeds all papers."""
        from abstracts_explorer import DatabaseManager

        db_path = tmp_path / "test.db"
        set_test_db(db_path)
        with DatabaseManager() as db:
            db.create_tables()
            papers = [
                LightweightPaper(
                    title="NeurIPS Paper",
                    abstract="Abstract 1",
                    authors=["Author"],
                    session="S1",
                    poster_position="P1",
                    year=2024,
                    conference="NeurIPS",
                ),
                LightweightPaper(
                    title="ICLR Paper",
                    abstract="Abstract 2",
                    authors=["Author"],
                    session="S2",
                    poster_position="P2",
                    year=2025,
                    conference="ICLR",
                ),
            ]
            db.add_papers(papers)

        patch_get_config_for_test(monkeypatch, tmp_path / "embeddings")

        with patch("abstracts_explorer.cli.EmbeddingsManager") as MockEM:
            mock_em = Mock()
            mock_em.check_model_compatibility.return_value = (True, None, "test-model")
            mock_em.test_lm_studio_connection.return_value = True
            mock_em.embed_from_database.return_value = 2
            mock_em.get_collection_stats.return_value = {"name": "test", "count": 2}
            mock_em.__enter__ = Mock(return_value=mock_em)
            mock_em.__exit__ = Mock(return_value=False)
            MockEM.return_value = mock_em

            with patch.object(
                sys,
                "argv",
                [
                    "neurips-abstracts",
                    "create-embeddings",
                ],
            ):
                exit_code = main()

        assert exit_code == 0

        # Verify no WHERE clause was passed (embeds all papers)
        call_kwargs = mock_em.embed_from_database.call_args.kwargs
        assert call_kwargs["where_clause"] is None

    def test_search_embeddings_not_found(self, tmp_path, capsys, monkeypatch):
        """Test search command with non-existent embeddings database."""
        nonexistent_path = tmp_path / "nonexistent_embeddings"

        # Set EMBEDDING_DB and patch config reload
        patch_get_config_for_test(monkeypatch, nonexistent_path)

        with patch.object(
            sys,
            "argv",
            [
                "neurips-abstracts",
                "search",
                "test query",
            ],
        ):
            exit_code = main()

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Embeddings database not found" in captured.err

    def test_search_lm_studio_not_available(self, tmp_path, capsys, monkeypatch):
        """Test search command when OpenAI API is not available."""
        # Create embeddings directory
        embeddings_path = tmp_path / "embeddings"
        embeddings_path.mkdir()

        # Set EMBEDDING_DB and patch config reload
        patch_get_config_for_test(monkeypatch, embeddings_path)

        # Mock embeddings manager with OpenAI API unavailable
        with patch("abstracts_explorer.cli.EmbeddingsManager") as MockEM:
            mock_em = Mock()
            mock_em.test_lm_studio_connection.return_value = False
            MockEM.return_value = mock_em

            with patch.object(
                sys,
                "argv",
                [
                    "neurips-abstracts",
                    "search",
                    "test query",
                ],
            ):
                exit_code = main()

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Failed to connect to OpenAI API" in captured.err

    def test_search_success(self, tmp_path, capsys, monkeypatch):
        """Test search command completes successfully."""
        embeddings_path = tmp_path / "embeddings"
        embeddings_path.mkdir()

        # Set EMBEDDING_DB and patch config reload
        patch_get_config_for_test(monkeypatch, embeddings_path)

        # Mock embeddings manager with results
        with patch("abstracts_explorer.cli.EmbeddingsManager") as MockEM:
            mock_em = Mock()
            mock_em.test_lm_studio_connection.return_value = True
            mock_em.get_collection_stats.return_value = {"name": "test", "count": 100}
            mock_em.search_similar.return_value = {
                "ids": [["123", "456"]],
                "distances": [[0.1, 0.2]],
                "metadatas": [
                    [
                        {
                            "title": "Paper 1",
                            "authors": "Author 1",
                            "decision": "Accept",
                            "topic": "ML",
                        },
                        {
                            "title": "Paper 2",
                            "authors": "Author 2",
                            "decision": "Accept",
                            "topic": "DL",
                        },
                    ]
                ],
                "documents": [["Abstract 1", "Abstract 2"]],
            }
            mock_em.__enter__ = Mock(return_value=mock_em)
            mock_em.__exit__ = Mock(return_value=False)
            MockEM.return_value = mock_em

            with patch.object(
                sys,
                "argv",
                [
                    "neurips-abstracts",
                    "search",
                    "test query",
                    "--n-results",
                    "2",
                ],
            ):
                exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Found 2 similar paper(s)" in captured.out
        assert "Paper 1" in captured.out
        assert "Paper 2" in captured.out

    def test_search_with_abstract(self, tmp_path, capsys, monkeypatch):
        """Test search command with --show-abstract flag."""
        embeddings_path = tmp_path / "embeddings"
        embeddings_path.mkdir()

        # Set EMBEDDING_DB and patch config reload
        patch_get_config_for_test(monkeypatch, embeddings_path)

        # Mock embeddings manager
        with patch("abstracts_explorer.cli.EmbeddingsManager") as MockEM:
            mock_em = Mock()
            mock_em.test_lm_studio_connection.return_value = True
            mock_em.get_collection_stats.return_value = {"name": "test", "count": 10}
            mock_em.search_similar.return_value = {
                "ids": [["789"]],
                "distances": [[0.15]],
                "metadatas": [[{"title": "Test Paper", "authors": "Test Author", "decision": "Accept"}]],
                "documents": [["This is a test abstract about machine learning."]],
            }
            mock_em.__enter__ = Mock(return_value=mock_em)
            mock_em.__exit__ = Mock(return_value=False)
            MockEM.return_value = mock_em

            with patch.object(
                sys,
                "argv",
                [
                    "neurips-abstracts",
                    "search",
                    "machine learning",
                    "--show-abstract",
                ],
            ):
                exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Test Paper" in captured.out
        assert "Abstract: This is a test abstract" in captured.out

    def test_search_with_filter(self, tmp_path, capsys, monkeypatch):
        """Test search command with metadata filter."""
        embeddings_path = tmp_path / "embeddings"
        embeddings_path.mkdir()

        # Set EMBEDDING_DB and patch config reload
        patch_get_config_for_test(monkeypatch, embeddings_path)

        # Mock embeddings manager
        with patch("abstracts_explorer.cli.EmbeddingsManager") as MockEM:
            mock_em = Mock()
            mock_em.test_lm_studio_connection.return_value = True
            mock_em.get_collection_stats.return_value = {"name": "test", "count": 50}
            mock_em.search_similar.return_value = {
                "ids": [["111"]],
                "distances": [[0.05]],
                "metadatas": [
                    [
                        {
                            "title": "Filtered Paper",
                            "authors": "Author",
                            "decision": "Accept (poster)",
                        }
                    ]
                ],
                "documents": [["Abstract"]],
            }
            mock_em.__enter__ = Mock(return_value=mock_em)
            mock_em.__exit__ = Mock(return_value=False)
            MockEM.return_value = mock_em

            with patch.object(
                sys,
                "argv",
                [
                    "neurips-abstracts",
                    "search",
                    "neural networks",
                    "--where",
                    "decision=Accept (poster)",
                ],
            ):
                exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Filter: {'decision': 'Accept (poster)'}" in captured.out
        assert "Filtered Paper" in captured.out

    def test_search_no_results(self, tmp_path, capsys, monkeypatch):
        """Test search command with no results."""
        embeddings_path = tmp_path / "embeddings"
        embeddings_path.mkdir()

        # Set EMBEDDING_DB and patch config reload
        patch_get_config_for_test(monkeypatch, embeddings_path)

        # Mock embeddings manager with empty results
        with patch("abstracts_explorer.cli.EmbeddingsManager") as MockEM:
            mock_em = Mock()
            mock_em.test_lm_studio_connection.return_value = True
            mock_em.get_collection_stats.return_value = {"name": "test", "count": 100}
            mock_em.search_similar.return_value = {"ids": [[]], "distances": [[]], "metadatas": [[]]}
            mock_em.__enter__ = Mock(return_value=mock_em)
            mock_em.__exit__ = Mock(return_value=False)
            MockEM.return_value = mock_em

            with patch.object(
                sys,
                "argv",
                [
                    "neurips-abstracts",
                    "search",
                    "nonexistent topic",
                ],
            ):
                exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "No results found" in captured.out

    def test_search_with_db_path_author_names(self, tmp_path, capsys, monkeypatch):
        """Test search command with database to resolve author names."""
        from abstracts_explorer import DatabaseManager

        # Create a test database with lightweight schema
        db_path = tmp_path / "test.db"
        set_test_db(db_path)
        with DatabaseManager() as db:
            db.create_tables()
            papers = [
                LightweightPaper(
                    title="Test Paper",
                    authors=["John Doe", "Jane Smith"],
                    abstract="Test abstract",
                    session="Session 1",
                    poster_position="A12",
                    year=2025,
                    conference="NeurIPS",
                    url="https://example.com/paper/1",
                )
            ]
            db.add_papers(papers)

        embeddings_path = tmp_path / "embeddings"
        embeddings_path.mkdir()

        # Set EMBEDDING_DB and patch config reload
        patch_get_config_for_test(monkeypatch, embeddings_path)

        # Mock embeddings manager
        with patch("abstracts_explorer.cli.EmbeddingsManager") as MockEM:
            mock_em = Mock()
            mock_em.test_lm_studio_connection.return_value = True
            mock_em.get_collection_stats.return_value = {"name": "test", "count": 1}
            mock_em.search_similar.return_value = {
                "ids": [["1"]],
                "distances": [[0.1]],
                "metadatas": [
                    [
                        {
                            "title": "Test Paper",
                            "authors": "John Doe; Jane Smith",
                            "session": "Session 1",
                            "year": 2025,
                            "conference": "NeurIPS",
                            "paper_url": "https://example.com/paper/1",
                            "poster_position": "A12",
                        }
                    ]
                ],
                "documents": [["Test abstract"]],
            }
            mock_em.__enter__ = Mock(return_value=mock_em)
            mock_em.__exit__ = Mock(return_value=False)
            MockEM.return_value = mock_em

            with patch.object(
                sys,
                "argv",
                [
                    "neurips-abstracts",
                    "search",
                    "test",
                ],
            ):
                exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        # Should show author names (already in proper format in lightweight schema)
        assert "John Doe; Jane Smith" in captured.out
        assert "https://example.com/paper/1" in captured.out
        assert "A12" in captured.out

    def test_search_with_db_path_missing_database(self, tmp_path, capsys, monkeypatch):
        """Test search command with non-existent database."""
        embeddings_path = tmp_path / "embeddings"
        embeddings_path.mkdir()
        nonexistent_db = tmp_path / "nonexistent.db"
        set_test_db(nonexistent_db)

        # Set EMBEDDING_DB and patch config reload
        patch_get_config_for_test(monkeypatch, embeddings_path)

        # Mock embeddings manager
        with patch("abstracts_explorer.cli.EmbeddingsManager") as MockEM:
            mock_em = Mock()
            mock_em.test_lm_studio_connection.return_value = True
            mock_em.get_collection_stats.return_value = {"name": "test", "count": 1}
            mock_em.search_similar.return_value = {
                "ids": [["1"]],
                "distances": [[0.1]],
                "metadatas": [[{"title": "Test", "authors": "101,102", "decision": "Accept"}]],
            }
            mock_em.__enter__ = Mock(return_value=mock_em)
            mock_em.__exit__ = Mock(return_value=False)
            MockEM.return_value = mock_em

            with patch.object(
                sys,
                "argv",
                [
                    "neurips-abstracts",
                    "search",
                    "test",
                ],
            ):
                exit_code = main()

        # Should succeed but fall back to author IDs (no author name resolution)
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "101,102" in captured.out  # Shows IDs as fallback

    def test_search_with_db_path_lookup_error(self, tmp_path, capsys, monkeypatch):
        """Test search command when database connection fails (no longer relevant)."""
        from abstracts_explorer.database import DatabaseManager

        embeddings_path = tmp_path / "embeddings"
        embeddings_path.mkdir()

        # Create an empty database using DatabaseManager
        db_path = tmp_path / "test.db"
        set_test_db(db_path)
        db = DatabaseManager()
        db.connect()
        db.close()

        # Set EMBEDDING_DB and patch config reload
        patch_get_config_for_test(monkeypatch, embeddings_path)

        # Mock embeddings manager
        with patch("abstracts_explorer.cli.EmbeddingsManager") as MockEM:
            mock_em = Mock()
            mock_em.test_lm_studio_connection.return_value = True
            mock_em.get_collection_stats.return_value = {"name": "test", "count": 1}
            mock_em.search_similar.return_value = {
                "ids": [["1"]],
                "distances": [[0.1]],
                "metadatas": [[{"title": "Test", "authors": "101", "decision": "Accept"}]],
            }
            mock_em.__enter__ = Mock(return_value=mock_em)
            mock_em.__exit__ = Mock(return_value=False)
            MockEM.return_value = mock_em

            with patch.object(
                sys,
                "argv",
                [
                    "neurips-abstracts",
                    "search",
                    "test",
                ],
            ):
                exit_code = main()

        # Should succeed (database connection is no longer attempted from search command)
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "101" in captured.out  # Shows author ID as-is since author name resolution is not in search

    def test_search_unexpected_exception(self, tmp_path, capsys, monkeypatch):
        """Test search command with unexpected exception."""
        embeddings_path = tmp_path / "embeddings"
        embeddings_path.mkdir()

        # Set EMBEDDING_DB and patch config reload
        patch_get_config_for_test(monkeypatch, embeddings_path)

        # Mock embeddings manager to raise unexpected exception
        with patch("abstracts_explorer.cli.EmbeddingsManager") as MockEM:
            mock_em = Mock()
            mock_em.test_lm_studio_connection.return_value = True
            mock_em.get_collection_stats.side_effect = Exception("Unexpected error")
            mock_em.__enter__ = Mock(return_value=mock_em)
            mock_em.__exit__ = Mock(return_value=False)
            MockEM.return_value = mock_em

            with patch.object(
                sys,
                "argv",
                [
                    "neurips-abstracts",
                    "search",
                    "test",
                ],
            ):
                exit_code = main()

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Unexpected error" in captured.err

    def test_clear_clustering_cache_success(self, tmp_path, capsys):
        """Test clear-clustering-cache command clears all cache entries."""
        output_db = tmp_path / "test.db"

        # Set PAPER_DB to output location
        set_test_db(output_db)

        # Create database and add some cache entries
        from abstracts_explorer.database import DatabaseManager

        with DatabaseManager() as db:
            db.create_tables()
            # Add some test cache entries
            db.save_clustering_cache(
                embedding_model="test-model-1",
                reduction_method="pca",
                n_components=2,
                clustering_method="kmeans",
                n_clusters=5,
                results={
                    "points": [{"id": "p1", "x": 1, "y": 2, "cluster": 0}],
                    "statistics": {"total_papers": 1, "n_clusters": 1},
                },
            )
            db.save_clustering_cache(
                embedding_model="test-model-2",
                reduction_method="pca",
                n_components=2,
                clustering_method="kmeans",
                n_clusters=5,
                results={
                    "points": [{"id": "p1", "x": 1, "y": 2, "cluster": 0}],
                    "statistics": {"total_papers": 1, "n_clusters": 1},
                },
            )

        # Run clear-clustering-cache command
        with patch.object(
            sys,
            "argv",
            ["abstracts-explorer", "clustering", "clear-cache"],
        ):
            exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Cleared all 2 clustering cache entries" in captured.out

        # Verify cache is cleared
        with DatabaseManager() as db:
            cached = db.get_clustering_cache(
                embedding_model="test-model-1",
                reduction_method="pca",
                n_components=2,
                clustering_method="kmeans",
                n_clusters=5,
            )
            assert cached is None

    def test_clear_clustering_cache_by_model(self, tmp_path, capsys):
        """Test clear-clustering-cache command with --embedding-model filter."""
        output_db = tmp_path / "test.db"

        # Set PAPER_DB to output location
        set_test_db(output_db)

        # Create database and add cache entries for different models
        from abstracts_explorer.database import DatabaseManager

        with DatabaseManager() as db:
            db.create_tables()
            # Add cache for model1
            db.save_clustering_cache(
                embedding_model="model1",
                reduction_method="pca",
                n_components=2,
                clustering_method="kmeans",
                n_clusters=5,
                results={
                    "points": [{"id": "p1", "x": 1, "y": 2, "cluster": 0}],
                    "statistics": {"total_papers": 1, "n_clusters": 1},
                },
            )

            # Add cache for model2
            db.save_clustering_cache(
                embedding_model="model2",
                reduction_method="pca",
                n_components=2,
                clustering_method="kmeans",
                n_clusters=5,
                results={
                    "points": [{"id": "p1", "x": 1, "y": 2, "cluster": 0}],
                    "statistics": {"total_papers": 1, "n_clusters": 1},
                },
            )

        # Clear cache only for model1
        with patch.object(
            sys,
            "argv",
            ["abstracts-explorer", "clustering", "clear-cache", "--embedding-model", "model1"],
        ):
            exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Cleared 1 clustering cache entry for model: model1" in captured.out

        # Verify model1 cache is cleared but model2 remains
        with DatabaseManager() as db:
            cached1 = db.get_clustering_cache(
                embedding_model="model1",
                reduction_method="pca",
                n_components=2,
                clustering_method="kmeans",
                n_clusters=5,
            )
            assert cached1 is None

            cached2 = db.get_clustering_cache(
                embedding_model="model2",
                reduction_method="pca",
                n_components=2,
                clustering_method="kmeans",
                n_clusters=5,
            )
            assert cached2 is not None

    def test_clear_clustering_cache_no_entries(self, tmp_path, capsys):
        """Test clear-clustering-cache command when no cache entries exist."""
        output_db = tmp_path / "test.db"

        # Set PAPER_DB to output location
        set_test_db(output_db)

        # Create empty database
        from abstracts_explorer.database import DatabaseManager

        with DatabaseManager() as db:
            db.create_tables()

        # Run clear-clustering-cache command
        with patch.object(
            sys,
            "argv",
            ["abstracts-explorer", "clustering", "clear-cache"],
        ):
            exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "No cache entries found to clear" in captured.out

    def test_clear_clustering_cache_error(self, tmp_path, capsys):
        """Test clear-clustering-cache command handles errors gracefully."""
        # Set a database path that doesn't exist and can't be created
        set_test_db("/nonexistent/path/test.db")

        # Run clear-clustering-cache command
        with patch.object(
            sys,
            "argv",
            ["abstracts-explorer", "clustering", "clear-cache"],
        ):
            exit_code = main()

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Error clearing clustering cache" in captured.err

    def test_pre_generate_clustering_missing_embeddings(self, tmp_path, capsys, monkeypatch):
        """Test pre-generate-clustering fails when embeddings DB doesn't exist."""
        patch_get_config_for_test(monkeypatch, tmp_path / "nonexistent")

        with patch.object(
            sys,
            "argv",
            ["abstracts-explorer", "clustering", "pre-generate"],
        ):
            exit_code = main()

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Embeddings database not found" in captured.err

    def test_pre_generate_clustering_calls_create_collection(self, tmp_path, capsys, monkeypatch):
        """Test pre-generate-clustering calls create_collection() before compute_clusters_with_cache.

        This is a regression test: previously the command called em.connect() but
        forgot em.create_collection(), causing ``get_collection_stats()`` to raise
        ``EmbeddingsError: Collection not initialized. Call create_collection() first.``
        """

        embeddings_path = tmp_path / "chroma_db"
        embeddings_path.mkdir()
        patch_get_config_for_test(monkeypatch, embeddings_path)
        set_test_db(tmp_path / "test.db")

        call_order = []

        with (
            patch("abstracts_explorer.cli.EmbeddingsManager") as mock_em_class,
            patch("abstracts_explorer.cli.compute_clusters_with_cache") as mock_compute,
            patch("abstracts_explorer.cli.DatabaseManager") as mock_db_class,
        ):
            mock_em = Mock()

            def track_connect():
                call_order.append("connect")

            def track_create_collection(reset=False):
                call_order.append("create_collection")

            mock_em.connect.side_effect = track_connect
            mock_em.create_collection.side_effect = track_create_collection
            mock_em_class.return_value = mock_em
            mock_compute.return_value = {
                "points": [],
                "statistics": {"total_papers": 5, "n_clusters": 2, "n_noise": 0, "cluster_sizes": {}},
            }

            mock_db_instance = Mock()
            mock_db_instance.get_conferences.return_value = ["NeurIPS"]
            mock_db_instance.get_years.side_effect = lambda conference=None: [2023]
            mock_db_instance.__enter__ = Mock(return_value=mock_db_instance)
            mock_db_instance.__exit__ = Mock(return_value=False)
            mock_db_class.return_value = mock_db_instance

            with patch.object(sys, "argv", ["abstracts-explorer", "clustering", "pre-generate"]):
                exit_code = main()

        assert exit_code == 0
        # create_collection must be called after connect and before compute_clusters_with_cache
        assert "connect" in call_order, "em.connect() was not called"
        assert "create_collection" in call_order, "em.create_collection() was not called (regression)"
        assert call_order.index("connect") < call_order.index(
            "create_collection"
        ), "em.connect() must be called before em.create_collection()"
        mock_compute.assert_called_once()

    def test_pre_generate_clustering_success(self, tmp_path, capsys, monkeypatch):
        """Test pre-generate-clustering without args generates all conference/year combos."""
        embeddings_path = tmp_path / "chroma_db"
        embeddings_path.mkdir()
        patch_get_config_for_test(monkeypatch, embeddings_path)

        set_test_db(tmp_path / "test.db")

        mock_results = {
            "points": [],
            "statistics": {"total_papers": 10, "n_clusters": 2, "n_noise": 0, "cluster_sizes": {}},
        }

        # DB returns one conference with two years
        with (
            patch("abstracts_explorer.cli.EmbeddingsManager") as mock_em_class,
            patch("abstracts_explorer.cli.compute_clusters_with_cache") as mock_compute,
            patch("abstracts_explorer.cli.DatabaseManager") as mock_db_class,
        ):
            mock_em = Mock()
            mock_em_class.return_value = mock_em
            mock_compute.return_value = mock_results

            mock_db_instance = Mock()
            mock_db_instance.get_conferences.return_value = ["NeurIPS"]
            mock_db_instance.get_years.side_effect = lambda conference=None: [2023, 2024]
            mock_db_instance.__enter__ = Mock(return_value=mock_db_instance)
            mock_db_instance.__exit__ = Mock(return_value=False)
            mock_db_class.return_value = mock_db_instance

            with patch.object(
                sys,
                "argv",
                ["abstracts-explorer", "clustering", "pre-generate"],
            ):
                exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Pre-generation complete" in captured.out
        # 2 combos: NeurIPS 2023 + NeurIPS 2024
        assert mock_compute.call_count == 2
        # Verify fixed agglomerative/t-SNE parameters on all calls
        for call in mock_compute.call_args_list:
            kw = call[1]
            assert kw["clustering_method"] == "agglomerative"
            assert kw["linkage"] == "ward"
            assert kw["distance_threshold"] == 150.0
            assert kw["reduction_method"] == "tsne"
            assert kw["n_clusters"] is None

    def test_pre_generate_clustering_no_conferences_in_db(self, tmp_path, capsys, monkeypatch):
        """Test pre-generate-clustering without args fails gracefully when DB is empty."""
        embeddings_path = tmp_path / "chroma_db"
        embeddings_path.mkdir()
        patch_get_config_for_test(monkeypatch, embeddings_path)

        set_test_db(tmp_path / "test.db")

        with (
            patch("abstracts_explorer.cli.EmbeddingsManager") as mock_em_class,
            patch("abstracts_explorer.cli.DatabaseManager") as mock_db_class,
        ):
            mock_em_class.return_value = Mock()

            mock_db_instance = Mock()
            mock_db_instance.get_conferences.return_value = []
            mock_db_instance.get_years.return_value = []
            mock_db_instance.__enter__ = Mock(return_value=mock_db_instance)
            mock_db_instance.__exit__ = Mock(return_value=False)
            mock_db_class.return_value = mock_db_instance

            with patch.object(
                sys,
                "argv",
                ["abstracts-explorer", "clustering", "pre-generate"],
            ):
                exit_code = main()

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "No conferences found" in captured.err

    def test_pre_generate_clustering_with_force_flag(self, tmp_path, capsys, monkeypatch):
        """Test pre-generate-clustering with --force flag."""
        embeddings_path = tmp_path / "chroma_db"
        embeddings_path.mkdir()
        patch_get_config_for_test(monkeypatch, embeddings_path)

        set_test_db(tmp_path / "test.db")

        mock_results = {
            "points": [],
            "statistics": {"total_papers": 5, "n_clusters": 3, "n_noise": 0, "cluster_sizes": {}},
        }

        with (
            patch("abstracts_explorer.cli.EmbeddingsManager") as mock_em_class,
            patch("abstracts_explorer.cli.compute_clusters_with_cache") as mock_compute,
            patch("abstracts_explorer.cli.DatabaseManager") as mock_db_class,
        ):
            mock_em = Mock()
            mock_em_class.return_value = mock_em
            mock_compute.return_value = mock_results

            mock_db_instance = Mock()
            mock_db_instance.get_conferences.return_value = ["NeurIPS"]
            mock_db_instance.get_years.side_effect = lambda conference=None: [2024, 2025]
            mock_db_instance.__enter__ = Mock(return_value=mock_db_instance)
            mock_db_instance.__exit__ = Mock(return_value=False)
            mock_db_class.return_value = mock_db_instance

            with patch.object(
                sys,
                "argv",
                [
                    "abstracts-explorer",
                    "clustering",
                    "pre-generate",
                    "--force",
                ],
            ):
                exit_code = main()

        assert exit_code == 0
        call_kwargs = mock_compute.call_args[1]
        assert call_kwargs["force"] is True

    def test_pre_generate_clustering_requests_per_minute(self, tmp_path, capsys, monkeypatch):
        """Test pre-generate-clustering with --requests-per-minute flag."""
        embeddings_path = tmp_path / "chroma_db"
        embeddings_path.mkdir()
        patch_get_config_for_test(monkeypatch, embeddings_path)

        set_test_db(tmp_path / "test.db")

        mock_results = {
            "points": [],
            "statistics": {"total_papers": 5, "n_clusters": 3, "n_noise": 0, "cluster_sizes": {}},
        }

        with (
            patch("abstracts_explorer.cli.EmbeddingsManager") as mock_em_class,
            patch("abstracts_explorer.cli.compute_clusters_with_cache") as mock_compute,
            patch("abstracts_explorer.cli.DatabaseManager") as mock_db_class,
        ):
            mock_em = Mock()
            mock_em_class.return_value = mock_em
            mock_compute.return_value = mock_results

            mock_db_instance = Mock()
            mock_db_instance.get_conferences.return_value = ["NeurIPS"]
            mock_db_instance.get_years.side_effect = lambda conference=None: [2024, 2025]
            mock_db_instance.__enter__ = Mock(return_value=mock_db_instance)
            mock_db_instance.__exit__ = Mock(return_value=False)
            mock_db_class.return_value = mock_db_instance

            with patch.object(
                sys,
                "argv",
                [
                    "abstracts-explorer",
                    "clustering",
                    "pre-generate",
                    "--requests-per-minute",
                    "30",
                ],
            ):
                exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "30 req/min" in captured.out

        # Verify EmbeddingsManager was initialized with requests_per_minute=30
        mock_em_class.assert_called_once()
        call_kwargs = mock_em_class.call_args.kwargs
        assert call_kwargs["requests_per_minute"] == 30

    def test_pre_generate_clustering_requests_per_minute_zero(self, tmp_path, capsys, monkeypatch):
        """Test pre-generate-clustering with --requests-per-minute 0 disables rate limiting."""
        embeddings_path = tmp_path / "chroma_db"
        embeddings_path.mkdir()
        patch_get_config_for_test(monkeypatch, embeddings_path)

        set_test_db(tmp_path / "test.db")

        mock_results = {
            "points": [],
            "statistics": {"total_papers": 5, "n_clusters": 3, "n_noise": 0, "cluster_sizes": {}},
        }

        with (
            patch("abstracts_explorer.cli.EmbeddingsManager") as mock_em_class,
            patch("abstracts_explorer.cli.compute_clusters_with_cache") as mock_compute,
            patch("abstracts_explorer.cli.DatabaseManager") as mock_db_class,
        ):
            mock_em = Mock()
            mock_em_class.return_value = mock_em
            mock_compute.return_value = mock_results

            mock_db_instance = Mock()
            mock_db_instance.get_conferences.return_value = ["NeurIPS"]
            mock_db_instance.get_years.side_effect = lambda conference=None: [2024, 2025]
            mock_db_instance.__enter__ = Mock(return_value=mock_db_instance)
            mock_db_instance.__exit__ = Mock(return_value=False)
            mock_db_class.return_value = mock_db_instance

            with patch.object(
                sys,
                "argv",
                [
                    "abstracts-explorer",
                    "clustering",
                    "pre-generate",
                    "--requests-per-minute",
                    "0",
                ],
            ):
                exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "disabled" in captured.out

        mock_em_class.assert_called_once()
        call_kwargs = mock_em_class.call_args.kwargs
        assert call_kwargs["requests_per_minute"] == 0

    def test_pre_generate_clustering_error(self, tmp_path, capsys, monkeypatch):
        """Test pre-generate-clustering handles errors gracefully."""
        embeddings_path = tmp_path / "chroma_db"
        embeddings_path.mkdir()
        patch_get_config_for_test(monkeypatch, embeddings_path)

        set_test_db(tmp_path / "test.db")

        with (
            patch("abstracts_explorer.cli.EmbeddingsManager") as mock_em_class,
            patch("abstracts_explorer.cli.compute_clusters_with_cache") as mock_compute,
            patch("abstracts_explorer.cli.DatabaseManager") as mock_db_class,
        ):
            mock_em_class.return_value = Mock()
            mock_compute.side_effect = Exception("Clustering failed")

            mock_db_instance = Mock()
            mock_db_instance.get_conferences.return_value = ["NeurIPS"]
            mock_db_instance.get_years.side_effect = lambda conference=None: [2025]
            mock_db_instance.__enter__ = Mock(return_value=mock_db_instance)
            mock_db_instance.__exit__ = Mock(return_value=False)
            mock_db_class.return_value = mock_db_instance

            with patch.object(
                sys,
                "argv",
                ["abstracts-explorer", "clustering", "pre-generate"],
            ):
                exit_code = main()

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "0 succeeded, 1 failed" in captured.out

    def test_pre_generate_clustering_with_conference(self, tmp_path, capsys, monkeypatch):
        """Test --conference generates all-years combo and each individual year."""
        embeddings_path = tmp_path / "chroma_db"
        embeddings_path.mkdir()
        patch_get_config_for_test(monkeypatch, embeddings_path)
        set_test_db(tmp_path / "test.db")

        mock_results = {
            "points": [],
            "statistics": {"total_papers": 50, "n_clusters": 3, "n_noise": 0, "cluster_sizes": {}},
        }

        with (
            patch("abstracts_explorer.cli.EmbeddingsManager") as mock_em_class,
            patch("abstracts_explorer.cli.compute_clusters_with_cache") as mock_compute,
            patch("abstracts_explorer.cli.DatabaseManager") as mock_db_class,
        ):
            mock_em = Mock()
            mock_em_class.return_value = mock_em
            mock_compute.return_value = mock_results

            mock_db_instance = Mock()
            mock_db_instance.get_conferences.return_value = ["ML4PS@NeurIPS"]
            mock_db_instance.get_years.side_effect = lambda conference=None: [2023, 2024]
            mock_db_instance.resolve_conference_name.return_value = "ML4PS@NeurIPS"
            mock_db_instance.__enter__ = Mock(return_value=mock_db_instance)
            mock_db_instance.__exit__ = Mock(return_value=False)
            mock_db_class.return_value = mock_db_instance

            with patch.object(
                sys,
                "argv",
                ["abstracts-explorer", "clustering", "pre-generate", "--conference", "ML4PS@NeurIPS"],
            ):
                exit_code = main()

        assert exit_code == 0
        # 2 combos: 2023 + 2024
        assert mock_compute.call_count == 2
        all_calls = [(c[1]["conferences"], c[1]["years"]) for c in mock_compute.call_args_list]
        assert (["ML4PS@NeurIPS"], [2023]) in all_calls
        assert (["ML4PS@NeurIPS"], [2024]) in all_calls
        captured = capsys.readouterr()
        assert "ML4PS@NeurIPS" in captured.out

    def test_pre_generate_clustering_all_db_years_used(self, tmp_path, capsys, monkeypatch):
        """Test pre-generate-clustering uses all years from the database (no plugin filtering)."""
        embeddings_path = tmp_path / "chroma_db"
        embeddings_path.mkdir()
        patch_get_config_for_test(monkeypatch, embeddings_path)
        set_test_db(tmp_path / "test.db")

        mock_results = {
            "points": [],
            "statistics": {"total_papers": 50, "n_clusters": 3, "n_noise": 0, "cluster_sizes": {}},
        }

        with (
            patch("abstracts_explorer.cli.EmbeddingsManager") as mock_em_class,
            patch("abstracts_explorer.cli.compute_clusters_with_cache") as mock_compute,
            patch("abstracts_explorer.cli.DatabaseManager") as mock_db_class,
        ):
            mock_em = Mock()
            mock_em_class.return_value = mock_em
            mock_compute.return_value = mock_results

            mock_db_instance = Mock()
            mock_db_instance.get_conferences.return_value = ["TestConf"]
            mock_db_instance.get_years.side_effect = lambda conference=None: [2022, 2023, 2024, 2025]
            mock_db_instance.__enter__ = Mock(return_value=mock_db_instance)
            mock_db_instance.__exit__ = Mock(return_value=False)
            mock_db_class.return_value = mock_db_instance

            with patch.object(
                sys,
                "argv",
                ["abstracts-explorer", "clustering", "pre-generate"],
            ):
                exit_code = main()

        assert exit_code == 0
        # All 4 DB years are used (no plugin filtering)
        assert mock_compute.call_count == 4
        all_calls = [(c[1]["conferences"], c[1]["years"]) for c in mock_compute.call_args_list]
        assert (["TestConf"], [2022]) in all_calls
        assert (["TestConf"], [2023]) in all_calls
        assert (["TestConf"], [2024]) in all_calls
        assert (["TestConf"], [2025]) in all_calls

    def test_pre_generate_clustering_with_year(self, tmp_path, capsys, monkeypatch):
        """Test pre-generate-clustering with --conference and --year (single combo)."""
        embeddings_path = tmp_path / "chroma_db"
        embeddings_path.mkdir()
        patch_get_config_for_test(monkeypatch, embeddings_path)
        set_test_db(tmp_path / "test.db")

        mock_results = {
            "points": [],
            "statistics": {"total_papers": 30, "n_clusters": 2, "n_noise": 0, "cluster_sizes": {}},
        }

        with (
            patch("abstracts_explorer.cli.EmbeddingsManager") as mock_em_class,
            patch("abstracts_explorer.cli.compute_clusters_with_cache") as mock_compute,
        ):
            mock_em = Mock()
            mock_em_class.return_value = mock_em
            mock_compute.return_value = mock_results

            with patch.object(
                sys,
                "argv",
                [
                    "abstracts-explorer",
                    "clustering",
                    "pre-generate",
                    "--conference",
                    "NeurIPS",
                    "--year",
                    "2024",
                ],
            ):
                exit_code = main()

        assert exit_code == 0
        # Only one combo when both --conference and --year are specified
        assert mock_compute.call_count == 1
        call_kwargs = mock_compute.call_args[1]
        assert call_kwargs["conferences"] == ["NeurIPS"]
        assert call_kwargs["years"] == [2024]
        captured = capsys.readouterr()
        assert "2024" in captured.out

    def test_pre_generate_clustering_conference_case_insensitive(self, tmp_path, capsys, monkeypatch):
        """Test that --conference is resolved case-insensitively from stored names."""
        embeddings_path = tmp_path / "chroma_db"
        embeddings_path.mkdir()
        patch_get_config_for_test(monkeypatch, embeddings_path)
        set_test_db(tmp_path / "test.db")

        mock_results = {
            "points": [],
            "statistics": {"total_papers": 50, "n_clusters": 3, "n_noise": 0, "cluster_sizes": {}},
        }

        # DatabaseManager methods return the canonical spelling
        with (
            patch("abstracts_explorer.cli.EmbeddingsManager") as mock_em_class,
            patch("abstracts_explorer.cli.compute_clusters_with_cache") as mock_compute,
            patch("abstracts_explorer.cli.DatabaseManager") as mock_db_class,
        ):
            mock_em = Mock()
            mock_em_class.return_value = mock_em
            mock_compute.return_value = mock_results

            # DatabaseManager used as context manager for conference resolution
            mock_db_instance = Mock()
            mock_db_instance.get_conferences.return_value = ["ML4PS@NeurIPS", "NeurIPS"]
            mock_db_instance.get_years.side_effect = lambda conference=None: [2024]
            # resolve_conference_name returns the canonical form for "ml4ps@neurips"
            mock_db_instance.resolve_conference_name.return_value = "ML4PS@NeurIPS"
            mock_db_instance.__enter__ = Mock(return_value=mock_db_instance)
            mock_db_instance.__exit__ = Mock(return_value=False)
            mock_db_class.return_value = mock_db_instance

            # User types the conference name in wrong case, with --year to get single combo
            with patch.object(
                sys,
                "argv",
                [
                    "abstracts-explorer",
                    "clustering",
                    "pre-generate",
                    "--conference",
                    "ml4ps@neurips",
                    "--year",
                    "2024",
                ],
            ):
                exit_code = main()

        assert exit_code == 0
        call_kwargs = mock_compute.call_args[1]
        # Should use the canonical spelling, not the user-supplied lower-case version
        assert call_kwargs["conferences"] == ["ML4PS@NeurIPS"]
        captured = capsys.readouterr()
        # Should print a resolution notice
        assert "ml4ps@neurips" in captured.out.lower()

    """Test cases for the chat command."""

    def test_chat_embeddings_not_found(self, tmp_path, capsys, monkeypatch):
        """Test chat command when embeddings don't exist."""
        from abstracts_explorer.cli import chat_command
        import argparse

        # Set EMBEDDING_DB and patch config reload
        patch_get_config_for_test(monkeypatch, tmp_path / "nonexistent")

        args = argparse.Namespace(
            embeddings_path=str(tmp_path / "nonexistent"),
            collection="test",
            model="test-model",
            embedding_model="test-embedding-model",
            lm_studio_url="http://localhost:1234",
            max_context=5,
            temperature=0.7,
            show_sources=False,
            export=None,
        )

        exit_code = chat_command(args)
        assert exit_code == 1

        captured = capsys.readouterr()
        assert "Embeddings database not found" in captured.err

    def test_chat_lm_studio_not_available(self, tmp_path, capsys, monkeypatch):
        """Test chat command when OpenAI API is not available."""
        from abstracts_explorer.cli import chat_command
        import argparse

        embeddings_path = tmp_path / "chroma_db"
        embeddings_path.mkdir()

        # Set EMBEDDING_DB and patch config reload
        patch_get_config_for_test(monkeypatch, embeddings_path)

        args = argparse.Namespace(
            embeddings_path=str(embeddings_path),
            collection="test",
            model="test-model",
            embedding_model="test-embedding-model",
            lm_studio_url="http://localhost:1234",
            max_context=5,
            temperature=0.7,
            show_sources=False,
            export=None,
        )

        with patch("abstracts_explorer.cli.EmbeddingsManager") as MockEM:
            mock_em = Mock()
            mock_em.test_lm_studio_connection.return_value = False
            MockEM.return_value = mock_em

            exit_code = chat_command(args)

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Failed to connect to OpenAI API" in captured.err

    def test_chat_rag_error(self, tmp_path, capsys, monkeypatch):
        """Test chat command with RAG error."""
        from abstracts_explorer.cli import chat_command
        from abstracts_explorer.rag import RAGError
        import argparse

        embeddings_path = tmp_path / "chroma_db"
        embeddings_path.mkdir()

        # Set up database path
        db_path = tmp_path / "test.db"
        set_test_db(db_path)

        # Set EMBEDDING_DB and patch config reload
        patch_get_config_for_test(monkeypatch, embeddings_path)

        args = argparse.Namespace(
            embeddings_path=str(embeddings_path),
            collection="test",
            model="test-model",
            embedding_model="test-embedding-model",
            lm_studio_url="http://localhost:1234",
            max_context=5,
            temperature=0.7,
            show_sources=False,
            export=None,
        )

        with patch("abstracts_explorer.cli.EmbeddingsManager") as MockEM:
            mock_em = Mock()
            mock_em.test_lm_studio_connection.return_value = True
            mock_em.get_collection_stats.return_value = {"name": "test", "count": 100}
            MockEM.return_value = mock_em

            with patch("abstracts_explorer.cli.RAGChat") as MockRAG:
                MockRAG.side_effect = RAGError("Test RAG error")

                exit_code = chat_command(args)

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "RAG error" in captured.err


class TestWebUICommand:
    """Test cases for the web-ui command."""

    def test_web_ui_flask_not_installed(self, capsys):
        """Test web-ui command when Flask is not installed."""
        from abstracts_explorer.cli import web_ui_command
        import argparse

        args = argparse.Namespace(host="127.0.0.1", port=5000, debug=False)

        # Mock the import at the location where it happens (inside web_ui_command)
        with patch.dict("sys.modules", {"abstracts_explorer.web_ui": None}):
            # Make importing web_ui raise ImportError

            def mock_import(name, *args, **kwargs):
                if name == "abstracts_explorer.web_ui":
                    raise ImportError("No module named 'flask'")
                return __import__(name, *args, **kwargs)

            with patch("builtins.__import__", wraps=mock_import):
                exit_code = web_ui_command(args)

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Web UI dependencies not installed" in captured.err

    def test_web_ui_keyboard_interrupt(self, capsys):
        """Test web-ui command handles keyboard interrupt gracefully."""
        from abstracts_explorer.cli import web_ui_command
        import argparse

        args = argparse.Namespace(host="127.0.0.1", port=5000, debug=False)

        # Mock run_server at the location where it's used after import
        with patch("abstracts_explorer.web_ui.run_server", side_effect=KeyboardInterrupt()):
            exit_code = web_ui_command(args)

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Server stopped" in captured.out

    def test_web_ui_unexpected_error(self, capsys):
        """Test web-ui command handles unexpected errors."""
        from abstracts_explorer.cli import web_ui_command
        import argparse

        args = argparse.Namespace(host="127.0.0.1", port=5000, debug=False)

        # Mock run_server at the location where it's used after import
        with patch("abstracts_explorer.web_ui.run_server", side_effect=Exception("Test error")):
            exit_code = web_ui_command(args)

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Error starting web server" in captured.err

    def test_web_ui_database_not_found(self, capsys):
        """Test web-ui command handles database not found error gracefully."""
        from abstracts_explorer.cli import web_ui_command
        import argparse

        args = argparse.Namespace(host="127.0.0.1", port=5000, debug=False)

        # Mock run_server to raise FileNotFoundError (as it would when database is missing)
        with patch(
            "abstracts_explorer.web_ui.run_server",
            side_effect=FileNotFoundError("Database not found: /nonexistent/test.db"),
        ):
            exit_code = web_ui_command(args)

        # Should exit with code 1 but not show traceback
        assert exit_code == 1
        captured = capsys.readouterr()
        # Should NOT show "Error starting web server" or traceback
        # The error message is printed by run_server itself before raising the exception
        assert "Error starting web server" not in captured.err

    def test_web_ui_passes_threads_to_run_server(self, capsys):
        """Test that web-ui command forwards --threads to run_server."""
        from abstracts_explorer.cli import web_ui_command
        import argparse

        args = argparse.Namespace(host="127.0.0.1", port=5000, verbose=0, dev=False, threads=10)

        with patch("abstracts_explorer.web_ui.run_server") as mock_run_server:
            mock_run_server.return_value = None
            exit_code = web_ui_command(args)

        assert exit_code == 0
        mock_run_server.assert_called_once_with(host="127.0.0.1", port=5000, debug=False, dev=False, threads=10)

    def test_web_ui_invalid_threads_returns_error(self, capsys):
        """Test that web-ui command returns exit code 1 when threads < 1."""
        from abstracts_explorer.cli import web_ui_command
        import argparse

        args = argparse.Namespace(host="127.0.0.1", port=5000, verbose=0, dev=False, threads=0)

        with patch("abstracts_explorer.web_ui.run_server", side_effect=ValueError("threads must be >= 1, got 0")):
            exit_code = web_ui_command(args)

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Invalid configuration" in captured.err


class TestMainDispatch:
    """Test main() command dispatch."""

    def test_main_chat_command(self, tmp_path, monkeypatch):
        """Test main() dispatches chat command."""
        embeddings_path = tmp_path / "chroma_db"
        embeddings_path.mkdir()

        # Set EMBEDDING_DB and patch config reload
        patch_get_config_for_test(monkeypatch, embeddings_path)

        with patch.object(sys, "argv", ["neurips-abstracts", "chat"]):
            with patch("abstracts_explorer.cli.chat_command") as mock_chat:
                mock_chat.return_value = 0
                exit_code = main()

        assert exit_code == 0
        mock_chat.assert_called_once()

    def test_main_web_ui_command(self):
        """Test main() dispatches web-ui command."""
        with patch.object(sys, "argv", ["neurips-abstracts", "web-ui"]):
            with patch("abstracts_explorer.cli.web_ui_command") as mock_web:
                mock_web.return_value = 0
                exit_code = main()

        assert exit_code == 0
        mock_web.assert_called_once()

    def test_main_version_flag(self, capsys):
        """Test --version flag prints version and exits with code 0."""
        with patch.object(sys, "argv", ["abstracts-explorer", "--version"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "abstracts-explorer" in captured.out
        # Version string should be present in output
        from abstracts_explorer.cli import __version__

        assert __version__ in captured.out


class TestCLISearchErrorHandling:
    """Test search command error handling paths."""

    def test_search_command_where_parse_warning(self, tmp_path, capsys, monkeypatch):
        """Test warning when where clause cannot be parsed (lines 229-230)."""
        import argparse

        embeddings_path = tmp_path / "chroma_db"
        embeddings_path.mkdir()

        # Set EMBEDDING_DB and patch config reload
        patch_get_config_for_test(monkeypatch, embeddings_path)

        args = argparse.Namespace(
            query="test",
            embeddings_path=str(embeddings_path),
            lm_studio_url="http://localhost:1234",
            model="test-model",
            collection="test_collection",
            n_results=5,
            show_abstract=False,
            where="invalid_no_equals_sign",  # Will fail parsing - no = sign
            db_path=None,
        )

        with patch("abstracts_explorer.cli.EmbeddingsManager") as MockEM:
            mock_em = MockEM.return_value
            mock_em.connect.return_value = None
            mock_em.create_collection.return_value = None
            mock_em.get_collection_stats.return_value = {"name": "test_collection", "count": 100}
            mock_em.search_similar.return_value = {
                "ids": [[]],
                "distances": [[]],
                "documents": [[]],
            }

            exit_code = search_command(args)

        assert exit_code == 0  # Should still succeed
        captured = capsys.readouterr()
        # Should show warning about filter parsing
        assert "Warning" in captured.err or "Could not parse" in captured.err

    def test_search_command_general_exception(self, tmp_path, capsys, monkeypatch):
        """Test general exception handling in search (lines 308-309)."""
        import argparse

        embeddings_path = tmp_path / "chroma_db"
        embeddings_path.mkdir()

        # Set EMBEDDING_DB and patch config reload
        patch_get_config_for_test(monkeypatch, embeddings_path)

        args = argparse.Namespace(
            query="test",
            embeddings_path=str(embeddings_path),
            lm_studio_url="http://localhost:1234",
            model="test-model",
            collection="test_collection",
            n_results=5,
            show_abstract=False,
            where=None,
            db_path=None,
        )

        with patch("abstracts_explorer.cli.EmbeddingsManager") as MockEM:
            # Make EmbeddingsManager raise unexpected exception
            MockEM.side_effect = RuntimeError("Unexpected error")

            exit_code = search_command(args)

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Unexpected error" in captured.err


class TestCLIEmbeddingsProgressAndStats:
    """Test embeddings command progress and stats display."""

    def test_create_embeddings_success_displays_stats(self, tmp_path, capsys, monkeypatch):
        """Test that successful embedding displays stats (lines 131-136, 147-152)."""
        from abstracts_explorer.cli import main

        db_path = tmp_path / "test.db"

        # Create test database
        from abstracts_explorer.database import DatabaseManager
        from abstracts_explorer.plugin import LightweightPaper

        set_test_db(db_path)
        db = DatabaseManager()
        with db:
            db.create_tables()
            for i in range(3):
                paper = LightweightPaper(
                    title=f"Paper {i}",
                    abstract=f"Abstract {i}",
                    authors=["Test Author"],
                    session="Session",
                    poster_position="P1",
                    year=2025,
                    conference="TestConf",
                )
                db.add_paper(paper)

        # Set EMBEDDING_DB and patch config reload
        patch_get_config_for_test(monkeypatch, tmp_path / "chroma_db")

        with patch("abstracts_explorer.cli.EmbeddingsManager") as MockEM:
            mock_em = MockEM.return_value
            mock_em.check_model_compatibility.return_value = (True, None, "test-model")
            mock_em.test_lm_studio_connection.return_value = True
            mock_em.collection_exists.return_value = False

            # Mock embed_from_database to simulate progress callbacks

            def mock_embed(batch_size=10, where_clause=None, progress_callback=None, force_recreate=False):
                # Simulate calling progress callback with (current, total) arguments
                if progress_callback:
                    progress_callback(1, 3)
                    progress_callback(2, 3)
                    progress_callback(3, 3)
                return 3

            mock_em.embed_from_database.side_effect = mock_embed
            mock_em.get_collection_stats.return_value = {
                "name": "papers",
                "count": 3,
            }

            with patch.object(
                sys,
                "argv",
                [
                    "neurips-abstracts",
                    "create-embeddings",
                ],
            ):
                exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        # Check stats are displayed (lines 132-136)
        assert "Collection Statistics" in captured.out
        assert "papers" in captured.out
        assert "3" in captured.out


class TestCLIChatInteractiveLoop:
    """Test chat command interactive loop paths."""

    def test_chat_empty_input_continues(self, tmp_path, capsys, monkeypatch):
        """Test that empty input is skipped in chat loop."""
        from abstracts_explorer.cli import chat_command
        import argparse

        embeddings_path = tmp_path / "chroma_db"
        embeddings_path.mkdir()

        # Set EMBEDDING_DB and patch config reload
        patch_get_config_for_test(monkeypatch, embeddings_path)

        args = argparse.Namespace(
            embeddings_path=str(embeddings_path),
            lm_studio_url="http://localhost:1234",
            model="test-model",
            embedding_model="test-embedding-model",
            collection="test_collection",
            max_context=3,
            temperature=0.7,
            show_sources=False,
            export=None,
        )

        with patch("abstracts_explorer.cli.EmbeddingsManager") as MockEM:
            mock_em = MockEM.return_value
            mock_em.connect.return_value = None
            mock_em.create_collection.return_value = None
            mock_em.get_collection_stats.return_value = {"name": "test_collection", "count": 100}

            with patch("abstracts_explorer.cli.RAGChat"):

                # Simulate: empty input, then exit
                with patch("builtins.input", side_effect=["", "   ", "exit"]):
                    exit_code = chat_command(args)

        assert exit_code == 0
        # Chat should have exited cleanly without processing empty inputs

    def test_chat_quit_command(self, tmp_path, capsys, monkeypatch):
        """Test chat exits on 'quit' command."""
        from abstracts_explorer.cli import chat_command
        import argparse

        embeddings_path = tmp_path / "chroma_db"
        embeddings_path.mkdir()

        # Set EMBEDDING_DB and patch config reload
        patch_get_config_for_test(monkeypatch, embeddings_path)

        args = argparse.Namespace(
            embeddings_path=str(embeddings_path),
            lm_studio_url="http://localhost:1234",
            model="test-model",
            embedding_model="test-embedding-model",
            collection="test_collection",
            max_context=3,
            temperature=0.7,
            show_sources=False,
            export=None,
        )

        with patch("abstracts_explorer.cli.EmbeddingsManager") as MockEM:
            mock_em = MockEM.return_value
            mock_em.connect.return_value = None
            mock_em.create_collection.return_value = None
            mock_em.get_collection_stats.return_value = {"name": "test_collection", "count": 100}

            with patch("abstracts_explorer.cli.RAGChat"):
                # Test 'quit' command
                with patch("builtins.input", side_effect=["quit"]):
                    exit_code = chat_command(args)

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Goodbye" in captured.out

    def test_chat_q_command(self, tmp_path, capsys, monkeypatch):
        """Test chat exits on 'q' command."""
        from abstracts_explorer.cli import chat_command
        import argparse

        embeddings_path = tmp_path / "chroma_db"
        embeddings_path.mkdir()

        # Set EMBEDDING_DB and patch config reload
        patch_get_config_for_test(monkeypatch, embeddings_path)

        args = argparse.Namespace(
            embeddings_path=str(embeddings_path),
            lm_studio_url="http://localhost:1234",
            model="test-model",
            embedding_model="test-embedding-model",
            collection="test_collection",
            max_context=3,
            temperature=0.7,
            show_sources=False,
            export=None,
        )

        with patch("abstracts_explorer.cli.EmbeddingsManager") as MockEM:
            mock_em = MockEM.return_value
            mock_em.connect.return_value = None
            mock_em.create_collection.return_value = None
            mock_em.get_collection_stats.return_value = {"name": "test_collection", "count": 100}

            with patch("abstracts_explorer.cli.RAGChat"):
                # Test 'q' command
                with patch("builtins.input", side_effect=["q"]):
                    exit_code = chat_command(args)

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Goodbye" in captured.out

    def test_chat_reset_command(self, tmp_path, capsys, monkeypatch):
        """Test chat reset command works."""
        from abstracts_explorer.cli import chat_command
        import argparse

        embeddings_path = tmp_path / "chroma_db"
        embeddings_path.mkdir()

        # Set EMBEDDING_DB and patch config reload
        patch_get_config_for_test(monkeypatch, embeddings_path)

        args = argparse.Namespace(
            embeddings_path=str(embeddings_path),
            lm_studio_url="http://localhost:1234",
            model="test-model",
            embedding_model="test-embedding-model",
            collection="test_collection",
            max_context=3,
            temperature=0.7,
            show_sources=False,
            export=None,
        )

        with patch("abstracts_explorer.cli.EmbeddingsManager") as MockEM:
            mock_em = MockEM.return_value
            mock_em.connect.return_value = None
            mock_em.create_collection.return_value = None
            mock_em.get_collection_stats.return_value = {"name": "test_collection", "count": 100}

            with patch("abstracts_explorer.cli.RAGChat") as MockRAG:
                mock_rag = MockRAG.return_value

                # Test reset then exit
                with patch("builtins.input", side_effect=["reset", "exit"]):
                    exit_code = chat_command(args)

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Conversation history cleared" in captured.out
        mock_rag.reset_conversation.assert_called_once()

    def test_chat_help_command(self, tmp_path, capsys, monkeypatch):
        """Test chat help command displays help."""
        from abstracts_explorer.cli import chat_command
        import argparse

        embeddings_path = tmp_path / "chroma_db"
        embeddings_path.mkdir()

        # Set EMBEDDING_DB and patch config reload
        patch_get_config_for_test(monkeypatch, embeddings_path)

        args = argparse.Namespace(
            embeddings_path=str(embeddings_path),
            lm_studio_url="http://localhost:1234",
            model="test-model",
            embedding_model="test-embedding-model",
            collection="test_collection",
            max_context=3,
            temperature=0.7,
            show_sources=False,
            export=None,
        )

        with patch("abstracts_explorer.cli.EmbeddingsManager") as MockEM:
            mock_em = MockEM.return_value
            mock_em.connect.return_value = None
            mock_em.create_collection.return_value = None
            mock_em.get_collection_stats.return_value = {"name": "test_collection", "count": 100}

            with patch("abstracts_explorer.cli.RAGChat"):
                # Test help then exit
                with patch("builtins.input", side_effect=["help", "exit"]):
                    exit_code = chat_command(args)

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Available commands" in captured.out
        assert "exit" in captured.out.lower()
        assert "reset" in captured.out.lower()

    def test_chat_with_query_and_show_sources(self, tmp_path, capsys, monkeypatch):
        """Test chat processes query and shows sources."""
        from abstracts_explorer.cli import chat_command
        import argparse

        embeddings_path = tmp_path / "chroma_db"
        embeddings_path.mkdir()

        # Set EMBEDDING_DB and patch config reload
        patch_get_config_for_test(monkeypatch, embeddings_path)

        args = argparse.Namespace(
            embeddings_path=str(embeddings_path),
            lm_studio_url="http://localhost:1234",
            model="test-model",
            embedding_model="test-embedding-model",
            collection="test_collection",
            max_context=3,
            temperature=0.7,
            show_sources=True,  # Enable source display
            export=None,
        )

        with patch("abstracts_explorer.cli.EmbeddingsManager") as MockEM:
            mock_em = MockEM.return_value
            mock_em.connect.return_value = None
            mock_em.create_collection.return_value = None
            mock_em.get_collection_stats.return_value = {"name": "test_collection", "count": 100}

            with patch("abstracts_explorer.cli.RAGChat") as MockRAG:
                mock_rag = MockRAG.return_value
                mock_rag.query.return_value = {
                    "response": "Test response about transformers",
                    "metadata": {"n_papers": 2},
                    "papers": [
                        {"title": "Attention Is All You Need", "similarity": 0.95},
                        {"title": "BERT", "similarity": 0.89},
                    ],
                }

                # Ask question then exit
                with patch("builtins.input", side_effect=["What are transformers?", "exit"]):
                    exit_code = chat_command(args)

        assert exit_code == 0
        captured = capsys.readouterr()
        # Check that response was displayed
        assert "Test response about transformers" in captured.out
        # Check that sources were displayed
        assert "Source papers" in captured.out
        assert "Attention Is All You Need" in captured.out

    def test_chat_with_export(self, tmp_path, capsys, monkeypatch):
        """Test chat exports conversation at end."""
        from abstracts_explorer.cli import chat_command
        import argparse

        embeddings_path = tmp_path / "chroma_db"
        embeddings_path.mkdir()
        export_file = tmp_path / "conversation.json"

        # Set EMBEDDING_DB and patch config reload
        patch_get_config_for_test(monkeypatch, embeddings_path)

        args = argparse.Namespace(
            embeddings_path=str(embeddings_path),
            lm_studio_url="http://localhost:1234",
            model="test-model",
            embedding_model="test-embedding-model",
            collection="test_collection",
            max_context=3,
            temperature=0.7,
            show_sources=False,
            export=str(export_file),  # Export enabled
        )

        with patch("abstracts_explorer.cli.EmbeddingsManager") as MockEM:
            mock_em = MockEM.return_value
            mock_em.connect.return_value = None
            mock_em.create_collection.return_value = None
            mock_em.get_collection_stats.return_value = {"name": "test_collection", "count": 100}

            with patch("abstracts_explorer.cli.RAGChat") as MockRAG:
                mock_rag = MockRAG.return_value

                # Exit immediately
                with patch("builtins.input", side_effect=["exit"]):
                    exit_code = chat_command(args)

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Conversation exported" in captured.out
        mock_rag.export_conversation.assert_called_once()

    def test_chat_keyboard_interrupt(self, tmp_path, capsys, monkeypatch):
        """Test chat handles Ctrl+C gracefully."""
        from abstracts_explorer.cli import chat_command
        import argparse

        embeddings_path = tmp_path / "chroma_db"
        embeddings_path.mkdir()

        # Set EMBEDDING_DB and patch config reload
        patch_get_config_for_test(monkeypatch, embeddings_path)

        args = argparse.Namespace(
            embeddings_path=str(embeddings_path),
            lm_studio_url="http://localhost:1234",
            model="test-model",
            embedding_model="test-embedding-model",
            collection="test_collection",
            max_context=3,
            temperature=0.7,
            show_sources=False,
            export=None,
        )

        with patch("abstracts_explorer.cli.EmbeddingsManager") as MockEM:
            mock_em = MockEM.return_value
            mock_em.connect.return_value = None
            mock_em.create_collection.return_value = None
            mock_em.get_collection_stats.return_value = {"name": "test_collection", "count": 100}

            with patch("abstracts_explorer.cli.RAGChat"):
                # Simulate Ctrl+C
                with patch("builtins.input", side_effect=KeyboardInterrupt()):
                    exit_code = chat_command(args)

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Goodbye" in captured.out

    def test_chat_eoferror(self, tmp_path, capsys, monkeypatch):
        """Test chat handles EOF (Ctrl+D) gracefully."""
        from abstracts_explorer.cli import chat_command
        import argparse

        embeddings_path = tmp_path / "chroma_db"
        embeddings_path.mkdir()

        # Set EMBEDDING_DB and patch config reload
        patch_get_config_for_test(monkeypatch, embeddings_path)

        args = argparse.Namespace(
            embeddings_path=str(embeddings_path),
            lm_studio_url="http://localhost:1234",
            model="test-model",
            embedding_model="test-embedding-model",
            collection="test_collection",
            max_context=3,
            temperature=0.7,
            show_sources=False,
            export=None,
        )

        with patch("abstracts_explorer.cli.EmbeddingsManager") as MockEM:
            mock_em = MockEM.return_value
            mock_em.connect.return_value = None
            mock_em.create_collection.return_value = None
            mock_em.get_collection_stats.return_value = {"name": "test_collection", "count": 100}

            with patch("abstracts_explorer.cli.RAGChat"):
                # Simulate EOF
                with patch("builtins.input", side_effect=EOFError()):
                    exit_code = chat_command(args)

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Goodbye" in captured.out


class TestDeleteDataCommand:
    """Tests for the delete-data command."""

    def _create_paper(self, conference: str, year: int) -> LightweightPaper:
        return LightweightPaper(
            title=f"{conference} {year} Paper",
            authors=["Author"],
            abstract="Test abstract",
            session="s",
            poster_position="p",
            year=year,
            conference=conference,
        )

    def test_requires_conference(self, tmp_path, capsys):
        """delete-data exits with error when --conference is missing."""
        set_test_db(tmp_path / "test.db")

        with patch.object(sys, "argv", ["abstracts-explorer", "delete-data", "--year", "2024"]):
            # argparse will error because --conference is required
            with pytest.raises(SystemExit):
                main()

    def test_requires_year(self, tmp_path, capsys):
        """delete-data exits with error when --year is missing."""
        set_test_db(tmp_path / "test.db")

        with patch.object(sys, "argv", ["abstracts-explorer", "delete-data", "--conference", "NeurIPS"]):
            with pytest.raises(SystemExit):
                main()

    def test_aborts_on_no_confirmation(self, tmp_path, capsys):
        """delete-data aborts when user does not type 'yes'."""
        output_db = tmp_path / "test.db"
        set_test_db(output_db)

        from abstracts_explorer.database import DatabaseManager

        with DatabaseManager() as db:
            db.create_tables()
            db.add_paper(self._create_paper("NeurIPS", 2024))

        with (
            patch.object(
                sys,
                "argv",
                ["abstracts-explorer", "delete-data", "--conference", "NeurIPS", "--year", "2024"],
            ),
            patch("abstracts_explorer.cli.EmbeddingsManager") as mock_em_class,
            patch("builtins.input", return_value="no"),
        ):
            mock_em = Mock()
            mock_em.collection.get.return_value = {"ids": []}
            mock_em_class.return_value = mock_em

            exit_code = main()

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Aborted" in captured.out

    def test_deletes_data_with_yes_flag(self, tmp_path, capsys):
        """delete-data --yes removes papers, embeddings, and clustering cache."""
        output_db = tmp_path / "test.db"
        set_test_db(output_db)

        from abstracts_explorer.database import DatabaseManager

        with DatabaseManager() as db:
            db.create_tables()
            db.add_paper(self._create_paper("NeurIPS", 2024))
            db.save_clustering_cache(
                embedding_model="model",
                reduction_method="pca",
                n_components=2,
                clustering_method="kmeans",
                n_clusters=3,
                results={
                    "points": [{"id": "p1", "x": 1, "y": 2, "cluster": 0}],
                    "statistics": {"total_papers": 1, "n_clusters": 3},
                },
                conference="NeurIPS",
                year=2024,
            )

        with (
            patch.object(
                sys,
                "argv",
                [
                    "abstracts-explorer",
                    "delete-data",
                    "--conference",
                    "NeurIPS",
                    "--year",
                    "2024",
                    "--yes",
                ],
            ),
            patch("abstracts_explorer.cli.EmbeddingsManager") as mock_em_class,
        ):
            mock_em = Mock()
            mock_em.collection.get.return_value = {"ids": ["emb1", "emb2"]}
            mock_em.delete_embeddings_by_filter.return_value = 2
            mock_em_class.return_value = mock_em

            exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Deleted" in captured.out
        assert "paper" in captured.out.lower()

        # Verify delete_embeddings_by_filter was called with correct parameters
        mock_em.delete_embeddings_by_filter.assert_called_once_with(conference="NeurIPS", year=2024)

        # Verify papers deleted from DB
        with DatabaseManager() as db:
            papers = db.search_papers(conference="NeurIPS", year=2024, limit=0)
            assert len(papers) == 0

            # Verify clustering cache deleted
            count = db.count_clustering_cache_by_conference_year("NeurIPS", 2024)
            assert count == 0

    def test_nothing_to_delete(self, tmp_path, capsys):
        """delete-data reports nothing to delete when no data exists."""
        output_db = tmp_path / "test.db"
        set_test_db(output_db)

        from abstracts_explorer.database import DatabaseManager

        with DatabaseManager() as db:
            db.create_tables()

        with (
            patch.object(
                sys,
                "argv",
                [
                    "abstracts-explorer",
                    "delete-data",
                    "--conference",
                    "NeurIPS",
                    "--year",
                    "2024",
                    "--yes",
                ],
            ),
            patch("abstracts_explorer.cli.EmbeddingsManager") as mock_em_class,
        ):
            mock_em = Mock()
            mock_em.collection.get.return_value = {"ids": []}
            mock_em_class.return_value = mock_em

            exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Nothing to delete" in captured.out

    def test_confirms_with_yes_input(self, tmp_path, capsys):
        """delete-data proceeds when user types 'yes' interactively."""
        output_db = tmp_path / "test.db"
        set_test_db(output_db)

        from abstracts_explorer.database import DatabaseManager

        with DatabaseManager() as db:
            db.create_tables()
            db.add_paper(self._create_paper("ICLR", 2023))

        with (
            patch.object(
                sys,
                "argv",
                ["abstracts-explorer", "delete-data", "--conference", "ICLR", "--year", "2023"],
            ),
            patch("abstracts_explorer.cli.EmbeddingsManager") as mock_em_class,
            patch("builtins.input", return_value="yes"),
        ):
            mock_em = Mock()
            mock_em.collection.get.return_value = {"ids": []}
            mock_em.delete_embeddings_by_filter.return_value = 0
            mock_em_class.return_value = mock_em

            exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "deleted" in captured.out.lower()

    def test_shows_counts_before_confirmation(self, tmp_path, capsys):
        """delete-data shows what will be deleted before asking for confirmation."""
        output_db = tmp_path / "test.db"
        set_test_db(output_db)

        from abstracts_explorer.database import DatabaseManager

        with DatabaseManager() as db:
            db.create_tables()
            db.add_paper(self._create_paper("NeurIPS", 2025))
            db.add_paper(self._create_paper("NeurIPS", 2025))

        with (
            patch.object(
                sys,
                "argv",
                [
                    "abstracts-explorer",
                    "delete-data",
                    "--conference",
                    "NeurIPS",
                    "--year",
                    "2025",
                ],
            ),
            patch("abstracts_explorer.cli.EmbeddingsManager") as mock_em_class,
            patch("builtins.input", return_value="no"),
        ):
            mock_em = Mock()
            mock_em.collection.get.return_value = {"ids": ["e1", "e2", "e3"]}
            mock_em_class.return_value = mock_em

            exit_code = main()

        assert exit_code == 1
        captured = capsys.readouterr()
        # Should show paper count (2), embedding count (3), cache count (0)
        assert "2" in captured.out
        assert "3" in captured.out
        assert "permanently" in captured.out.lower() or "⚠️" in captured.out

    def test_partial_failure_returns_nonzero(self, tmp_path, capsys):
        """delete-data returns exit code 1 when one step fails."""
        output_db = tmp_path / "test.db"
        set_test_db(output_db)

        from abstracts_explorer.database import DatabaseManager

        with DatabaseManager() as db:
            db.create_tables()
            db.add_paper(self._create_paper("NeurIPS", 2024))

        with (
            patch.object(
                sys,
                "argv",
                [
                    "abstracts-explorer",
                    "delete-data",
                    "--conference",
                    "NeurIPS",
                    "--year",
                    "2024",
                    "--yes",
                ],
            ),
            patch("abstracts_explorer.cli.EmbeddingsManager") as mock_em_class,
        ):
            mock_em = Mock()
            mock_em.collection.get.return_value = {"ids": ["e1"]}
            mock_em.delete_embeddings_by_filter.side_effect = Exception("ChromaDB failure")
            mock_em_class.return_value = mock_em

            exit_code = main()

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "ChromaDB failure" in captured.err


class TestLogging:
    """Test cases for logging configuration."""

    def test_setup_logging_default_warning(self, monkeypatch):
        """Test that default logging level is WARNING when no flags or env var is set."""
        from tests.conftest import get_env_test_path

        # Clear any LOG_LEVEL env var
        monkeypatch.delenv("LOG_LEVEL", raising=False)

        # Force config reload to pick up environment changes
        get_config(reload=True, env_path=get_env_test_path())

        # Setup logging with verbosity=0 (default)
        setup_logging(0)

        # Check that root logger is at WARNING level
        assert logging.root.level == logging.WARNING

    def test_setup_logging_verbosity_info(self, monkeypatch):
        """Test that -v flag sets INFO level."""
        from tests.conftest import get_env_test_path

        # Clear any LOG_LEVEL env var
        monkeypatch.delenv("LOG_LEVEL", raising=False)

        # Force config reload
        get_config(reload=True, env_path=get_env_test_path())

        # Setup logging with verbosity=1 (-v)
        setup_logging(1)

        # Check that root logger is at INFO level
        assert logging.root.level == logging.INFO

    def test_setup_logging_verbosity_debug(self, monkeypatch):
        """Test that -vv flag sets DEBUG level."""
        from tests.conftest import get_env_test_path

        # Clear any LOG_LEVEL env var
        monkeypatch.delenv("LOG_LEVEL", raising=False)

        # Force config reload
        get_config(reload=True, env_path=get_env_test_path())

        # Setup logging with verbosity=2 (-vv)
        setup_logging(2)

        # Check that root logger is at DEBUG level
        assert logging.root.level == logging.DEBUG

    def test_setup_logging_env_var_info(self, monkeypatch):
        """Test that LOG_LEVEL=INFO env var sets INFO level."""
        from tests.conftest import get_env_test_path

        # Set LOG_LEVEL env var
        monkeypatch.setenv("LOG_LEVEL", "INFO")

        # Force config reload to pick up environment changes
        get_config(reload=True, env_path=get_env_test_path())

        # Setup logging with verbosity=0 (no flags)
        setup_logging(0)

        # Check that root logger is at INFO level
        assert logging.root.level == logging.INFO

    def test_setup_logging_env_var_debug(self, monkeypatch):
        """Test that LOG_LEVEL=DEBUG env var sets DEBUG level."""
        from tests.conftest import get_env_test_path

        # Set LOG_LEVEL env var
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")

        # Force config reload
        get_config(reload=True, env_path=get_env_test_path())

        # Setup logging with verbosity=0 (no flags)
        setup_logging(0)

        # Check that root logger is at DEBUG level
        assert logging.root.level == logging.DEBUG

    def test_setup_logging_flag_overrides_env_var(self, monkeypatch):
        """Test that -v flag overrides LOG_LEVEL env var."""
        from tests.conftest import get_env_test_path

        # Set LOG_LEVEL to WARNING
        monkeypatch.setenv("LOG_LEVEL", "WARNING")

        # Force config reload
        get_config(reload=True, env_path=get_env_test_path())

        # Setup logging with verbosity=2 (-vv for DEBUG)
        setup_logging(2)

        # Check that root logger is at DEBUG level (flag overrides env var)
        assert logging.root.level == logging.DEBUG

    def test_setup_logging_invalid_env_var(self, monkeypatch):
        """Test that invalid LOG_LEVEL env var falls back to WARNING."""
        from tests.conftest import get_env_test_path

        # Set invalid LOG_LEVEL
        monkeypatch.setenv("LOG_LEVEL", "INVALID")

        # Force config reload
        get_config(reload=True, env_path=get_env_test_path())

        # Setup logging with verbosity=0
        setup_logging(0)

        # Check that root logger falls back to WARNING
        assert logging.root.level == logging.WARNING

    def test_setup_logging_resets_package_logger(self, monkeypatch):
        """Test that setup_logging resets the abstracts_explorer package logger to NOTSET."""
        from tests.conftest import get_env_test_path

        monkeypatch.delenv("LOG_LEVEL", raising=False)
        get_config(reload=True, env_path=get_env_test_path())

        setup_logging(1)  # INFO

        # Package logger should be NOTSET so it inherits from root
        assert logging.getLogger("abstracts_explorer").level == logging.NOTSET

    def test_setup_logging_package_logger_inherits_root(self, monkeypatch):
        """Test that after setup_logging, the package logger inherits the root level."""
        from tests.conftest import get_env_test_path

        monkeypatch.delenv("LOG_LEVEL", raising=False)
        get_config(reload=True, env_path=get_env_test_path())

        setup_logging(1)  # INFO

        # Effective level of package logger should match root (INFO)
        assert logging.getLogger("abstracts_explorer").getEffectiveLevel() == logging.INFO


class TestPackageLogging:
    """Test cases for package-level logging configuration at import time."""

    def test_configure_package_logging_default_warning(self, monkeypatch):
        """Test that package logger is set to WARNING by default."""
        from tests.conftest import get_env_test_path
        from abstracts_explorer import _configure_package_logging

        monkeypatch.delenv("LOG_LEVEL", raising=False)
        get_config(reload=True, env_path=get_env_test_path())

        _configure_package_logging()

        assert logging.getLogger("abstracts_explorer").level == logging.WARNING

    def test_configure_package_logging_env_var_info(self, monkeypatch):
        """Test that LOG_LEVEL=INFO env var is respected at import time."""
        from abstracts_explorer import _configure_package_logging

        monkeypatch.setenv("LOG_LEVEL", "INFO")
        get_config(reload=True)

        _configure_package_logging()

        assert logging.getLogger("abstracts_explorer").level == logging.INFO

    def test_configure_package_logging_env_var_debug(self, monkeypatch):
        """Test that LOG_LEVEL=DEBUG env var is respected at import time."""
        from abstracts_explorer import _configure_package_logging

        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        get_config(reload=True)

        _configure_package_logging()

        assert logging.getLogger("abstracts_explorer").level == logging.DEBUG

    def test_configure_package_logging_env_var_invalid(self, monkeypatch):
        """Test that an invalid LOG_LEVEL falls back to WARNING."""
        from abstracts_explorer import _configure_package_logging

        monkeypatch.setenv("LOG_LEVEL", "INVALID")
        get_config(reload=True)

        _configure_package_logging()

        assert logging.getLogger("abstracts_explorer").level == logging.WARNING

    def test_configure_package_logging_env_file(self, monkeypatch, tmp_path):
        """Test that LOG_LEVEL in .env file is respected at import time."""
        from abstracts_explorer import _configure_package_logging

        # No LOG_LEVEL in environment
        monkeypatch.delenv("LOG_LEVEL", raising=False)

        # Create a .env file in a temp dir and reload config from it
        env_file = tmp_path / ".env"
        env_file.write_text("LOG_LEVEL=DEBUG\n")
        get_config(reload=True, env_path=env_file)

        _configure_package_logging()

        assert logging.getLogger("abstracts_explorer").level == logging.DEBUG

    def test_plugin_info_messages_suppressed_at_default_level(self, monkeypatch):
        """Test that plugin registration INFO messages are suppressed at default WARNING level."""
        monkeypatch.delenv("LOG_LEVEL", raising=False)

        # Ensure package logger is at WARNING
        logging.getLogger("abstracts_explorer").setLevel(logging.WARNING)

        # Capture log records from the plugin logger
        plugin_logger = logging.getLogger("abstracts_explorer.plugin")
        with self._capture_logs(plugin_logger, logging.INFO) as captured:
            # Simulate a plugin registration
            plugin_logger.info("Registered plugin: test")

        # INFO message should NOT have been captured (WARNING level blocks it)
        assert len(captured) == 0

    def test_plugin_info_messages_visible_at_info_level(self, monkeypatch):
        """Test that plugin registration INFO messages are shown when LOG_LEVEL=INFO."""
        monkeypatch.setenv("LOG_LEVEL", "INFO")

        # Ensure package logger is at INFO
        logging.getLogger("abstracts_explorer").setLevel(logging.INFO)

        # Capture log records from the plugin logger
        plugin_logger = logging.getLogger("abstracts_explorer.plugin")
        with self._capture_logs(plugin_logger, logging.INFO) as captured:
            plugin_logger.info("Registered plugin: test")

        # INFO message should have been captured
        assert len(captured) == 1
        assert "Registered plugin: test" in captured[0].getMessage()

    @staticmethod
    @contextlib.contextmanager
    def _capture_logs(logger, level):
        """Context manager to capture log records from a logger."""
        records = []

        class _Handler(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = _Handler(level)
        logger.addHandler(handler)
        try:
            yield records
        finally:
            logger.removeHandler(handler)


class TestFeedbackCommand:
    """Tests for the 'feedback' CLI command group and sub-commands."""

    def _seed_data(self, db_path):
        """Seed the test database with chat donations and validation data."""
        from abstracts_explorer.database import DatabaseManager

        with DatabaseManager() as db:
            db.create_tables()
            db.donate_chat_transcript("up", [{"role": "user", "text": "q"}, {"role": "assistant", "text": "a"}])
            db.donate_chat_transcript("down", [{"role": "user", "text": "q2"}, {"role": "assistant", "text": "a2"}])
            db.donate_validation_data(
                {"uid1": {"priority": 5, "searchTerm": "ml"}, "uid2": {"priority": 3, "searchTerm": None}}
            )

    # ------------------------------------------------------------------ stats
    def test_feedback_stats_empty(self, tmp_path, capsys):
        """feedback stats shows zeros when no data exists."""
        set_test_db(tmp_path / "test.db")

        with patch.object(sys, "argv", ["abstracts-explorer", "feedback", "stats"]):
            from abstracts_explorer.database import DatabaseManager

            with DatabaseManager() as db:
                db.create_tables()

            exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Total" in captured.out
        assert "0" in captured.out

    def test_feedback_stats_with_data(self, tmp_path, capsys):
        """feedback stats shows correct counts with data."""
        set_test_db(tmp_path / "test.db")
        self._seed_data(tmp_path / "test.db")

        with patch.object(sys, "argv", ["abstracts-explorer", "feedback", "stats"]):
            exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "2" in captured.out  # 2 chat donations
        assert "2" in captured.out  # 2 data donations

    # ------------------------------------------------------------------ list
    def test_feedback_list_all(self, tmp_path, capsys):
        """feedback list --type all shows both chat and data donations."""
        set_test_db(tmp_path / "test.db")
        self._seed_data(tmp_path / "test.db")

        with patch.object(sys, "argv", ["abstracts-explorer", "feedback", "list", "--type", "all"]):
            exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Chat Donations" in captured.out
        assert "Data Donations" in captured.out

    def test_feedback_list_chat_only(self, tmp_path, capsys):
        """feedback list --type chat shows only chat donations."""
        set_test_db(tmp_path / "test.db")
        self._seed_data(tmp_path / "test.db")

        with patch.object(sys, "argv", ["abstracts-explorer", "feedback", "list", "--type", "chat"]):
            exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Chat Donations" in captured.out
        assert "Data Donations" not in captured.out

    def test_feedback_list_chat_rating_filter(self, tmp_path, capsys):
        """feedback list --type chat --rating up lists only thumbs-up donations."""
        set_test_db(tmp_path / "test.db")
        self._seed_data(tmp_path / "test.db")

        with patch.object(
            sys, "argv", ["abstracts-explorer", "feedback", "list", "--type", "chat", "--rating", "up"]
        ):
            exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "up" in captured.out
        # Only 1 up entry expected
        assert captured.out.count("👍") == 1

    def test_feedback_list_data_only(self, tmp_path, capsys):
        """feedback list --type data shows only data donations."""
        set_test_db(tmp_path / "test.db")
        self._seed_data(tmp_path / "test.db")

        with patch.object(sys, "argv", ["abstracts-explorer", "feedback", "list", "--type", "data"]):
            exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Data Donations" in captured.out
        assert "Chat Donations" not in captured.out

    def test_feedback_list_empty(self, tmp_path, capsys):
        """feedback list shows 'No … found' when no data exists."""
        set_test_db(tmp_path / "test.db")
        with DatabaseManager() as db:
            db.create_tables()

        with patch.object(sys, "argv", ["abstracts-explorer", "feedback", "list"]):
            exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "No chat donations found" in captured.out
        assert "No data donations found" in captured.out

    # ------------------------------------------------------------------ browse
    def test_feedback_browse_chat_quit(self, tmp_path, capsys):
        """feedback browse --type chat quits gracefully on 'q'."""
        set_test_db(tmp_path / "test.db")
        self._seed_data(tmp_path / "test.db")

        with (
            patch.object(sys, "argv", ["abstracts-explorer", "feedback", "browse", "--type", "chat"]),
            patch("builtins.input", return_value="q"),
        ):
            exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Browsing" in captured.out

    def test_feedback_browse_data_quit(self, tmp_path, capsys):
        """feedback browse --type data quits gracefully on 'q'."""
        set_test_db(tmp_path / "test.db")
        self._seed_data(tmp_path / "test.db")

        with (
            patch.object(sys, "argv", ["abstracts-explorer", "feedback", "browse", "--type", "data"]),
            patch("builtins.input", return_value="q"),
        ):
            exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Browsing" in captured.out

    def test_feedback_browse_empty_chat(self, tmp_path, capsys):
        """feedback browse shows message when no chat donations exist."""
        set_test_db(tmp_path / "test.db")
        with DatabaseManager() as db:
            db.create_tables()

        with patch.object(sys, "argv", ["abstracts-explorer", "feedback", "browse", "--type", "chat"]):
            exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "No chat donations found" in captured.out

    def test_feedback_browse_eoferror(self, tmp_path, capsys):
        """feedback browse handles EOFError gracefully."""
        set_test_db(tmp_path / "test.db")
        self._seed_data(tmp_path / "test.db")

        with (
            patch.object(sys, "argv", ["abstracts-explorer", "feedback", "browse", "--type", "chat"]),
            patch("builtins.input", side_effect=EOFError),
        ):
            exit_code = main()

        assert exit_code == 0

    def test_feedback_browse_sample(self, tmp_path, capsys):
        """feedback browse --sample limits entries to random sample."""
        set_test_db(tmp_path / "test.db")
        self._seed_data(tmp_path / "test.db")

        with (
            patch.object(
                sys, "argv", ["abstracts-explorer", "feedback", "browse", "--type", "chat", "--sample", "1"]
            ),
            patch("builtins.input", return_value="q"),
        ):
            exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Browsing 1 chat donation(s)" in captured.out

    # ------------------------------------------------------------------ clear
    def test_feedback_clear_aborted(self, tmp_path, capsys):
        """feedback clear aborts when user does not confirm."""
        set_test_db(tmp_path / "test.db")
        self._seed_data(tmp_path / "test.db")

        with (
            patch.object(sys, "argv", ["abstracts-explorer", "feedback", "clear"]),
            patch("builtins.input", return_value="no"),
        ):
            exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Aborted" in captured.out

    def test_feedback_clear_yes_flag(self, tmp_path, capsys):
        """feedback clear --yes deletes all feedback without prompting."""
        set_test_db(tmp_path / "test.db")
        self._seed_data(tmp_path / "test.db")

        with patch.object(sys, "argv", ["abstracts-explorer", "feedback", "clear", "--yes"]):
            exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Deleted" in captured.out

    def test_feedback_clear_chat_only(self, tmp_path, capsys):
        """feedback clear --type chat removes only chat donations."""
        set_test_db(tmp_path / "test.db")
        self._seed_data(tmp_path / "test.db")

        with patch.object(sys, "argv", ["abstracts-explorer", "feedback", "clear", "--type", "chat", "--yes"]):
            exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "chat donation" in captured.out

        # Data donations should still exist
        with DatabaseManager() as db:
            data = db.get_validation_data()
        assert len(data) > 0

    def test_feedback_clear_nothing_to_clear(self, tmp_path, capsys):
        """feedback clear shows message when no feedback data exists."""
        set_test_db(tmp_path / "test.db")
        with DatabaseManager() as db:
            db.create_tables()

        with patch.object(sys, "argv", ["abstracts-explorer", "feedback", "clear"]):
            exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "nothing to clear" in captured.out

    # ------------------------------------------------------------------ no subcommand
    def test_feedback_no_subcommand_prints_help(self, tmp_path, capsys):
        """feedback with no sub-command prints help and returns 1."""
        set_test_db(tmp_path / "test.db")

        with patch.object(sys, "argv", ["abstracts-explorer", "feedback"]):
            exit_code = main()

        assert exit_code == 1

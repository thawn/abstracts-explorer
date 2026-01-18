"""
Unit tests for the CLI module.

This module tests the command-line interface for neurips-abstracts,
including the download and create-embeddings commands.
"""

import sys
from unittest.mock import Mock, patch
import pytest
from abstracts_explorer.cli import (
    main,
    search_command,
)
from abstracts_explorer.plugin import LightweightPaper
from abstracts_explorer.config import get_config
from tests.conftest import set_test_db


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

    def test_download_command_success(self, tmp_path, capsys, monkeypatch):
        """Test download command completes successfully."""
        output_db = tmp_path / "test.db"
        
        # Set PAPER_DB to output location
        set_test_db(monkeypatch, output_db)

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
                ["neurips-abstracts", "download", "--year", "2025", "--output", str(output_db)],
            ):
                exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Downloaded 2 papers" in captured.out
        assert "Database updated:" in captured.out
        assert output_db.exists()

    def test_download_command_failure(self, tmp_path, capsys, monkeypatch):
        """Test download command handles errors gracefully."""
        output_db = tmp_path / "test.db"
        
        # Set PAPER_DB to output location
        set_test_db(monkeypatch, output_db)

        # Mock the plugin to raise an exception
        mock_plugin = Mock()
        mock_plugin.plugin_name = "neurips"
        mock_plugin.plugin_description = "NeurIPS Test Plugin"
        mock_plugin.download.side_effect = Exception("Network error")

        with patch("abstracts_explorer.cli.get_plugin") as mock_get_plugin:
            mock_get_plugin.return_value = mock_plugin

            with patch.object(
                sys,
                "argv",
                ["neurips-abstracts", "download", "--output", str(output_db)],
            ):
                exit_code = main()

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err

    def test_download_command_with_database_url(self, tmp_path, capsys, monkeypatch):
        """Test download command uses PAPER_DB when set."""
        # Create a temporary SQLite database for testing
        db_path = tmp_path / "test.db"
        
        # Set PAPER_DB environment variable
        set_test_db(monkeypatch, db_path)
        
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
                ["neurips-abstracts", "download", "--year", "2025", "--output", "ignored_path.db"],
            ):
                exit_code = main()
        
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Downloaded 1 papers" in captured.out or "Downloaded 1 paper" in captured.out
        # Should use PAPER_DB, not the --output path
        assert "Database updated:" in captured.out
        # Verify database was created
        assert db_path.exists()

    def test_create_embeddings_db_not_found(self, tmp_path, capsys, monkeypatch):
        """Test create-embeddings with non-existent database."""
        nonexistent_db = tmp_path / "nonexistent.db"
        set_test_db(monkeypatch, nonexistent_db)

        # The exception will be raised and not caught, so we expect it to propagate
        with patch.object(
            sys,
            "argv",
            ["neurips-abstracts", "create-embeddings"],
        ):
            # Should raise DatabaseError because database/tables don't exist
            with pytest.raises(Exception) as exc_info:
                exit_code = main()
        
        # Verify it's a database-related error
        assert "table" in str(exc_info.value).lower() or "database" in str(exc_info.value).lower()

    def test_create_embeddings_lm_studio_not_available(self, tmp_path, capsys, monkeypatch):
        """Test create-embeddings when OpenAI API is not available."""
        # Create a test database
        from abstracts_explorer import DatabaseManager

        db_path = tmp_path / "test.db"
        set_test_db(monkeypatch, db_path)
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
                    "--output",
                    str(tmp_path / "embeddings"),
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
        set_test_db(monkeypatch, db_path)
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
                    "--output",
                    str(tmp_path / "embeddings"),
                ],
            ):
                exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Successfully generated embeddings for 2 papers" in captured.out
        assert "Vector database saved to" in captured.out

    def test_create_embeddings_with_where_clause(self, tmp_path, capsys, monkeypatch):
        """Test create-embeddings with WHERE clause filter."""
        from abstracts_explorer import DatabaseManager

        # Create a test database
        db_path = tmp_path / "test.db"
        set_test_db(monkeypatch, db_path)
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
                    "--output",
                    str(tmp_path / "embeddings"),
                    "--where",
                    "award IS NOT NULL",
                ],
            ):
                exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Filter will process 1 papers" in captured.out
        assert "Successfully generated embeddings for 1 papers" in captured.out

    def test_create_embeddings_force_flag(self, tmp_path, capsys, monkeypatch):
        """Test create-embeddings with --force flag."""
        from abstracts_explorer import DatabaseManager

        # Create a test database
        db_path = tmp_path / "test.db"
        set_test_db(monkeypatch, db_path)
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
                    "--output",
                    str(embeddings_path),
                    "--force",
                ],
            ):
                exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Resetting existing collection" in captured.out

        # Verify create_collection was called with reset=True
        mock_em.create_collection.assert_called_once_with(reset=True)

    def test_create_embeddings_custom_model(self, tmp_path, capsys, monkeypatch):
        """Test create-embeddings with custom model settings."""
        from abstracts_explorer import DatabaseManager

        # Create a test database
        db_path = tmp_path / "test.db"
        set_test_db(monkeypatch, db_path)
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

    def test_create_embeddings_embeddings_error(self, tmp_path, capsys, monkeypatch):
        """Test create-embeddings handles EmbeddingsError gracefully."""
        from abstracts_explorer import DatabaseManager
        from abstracts_explorer.embeddings import EmbeddingsError

        # Create a test database
        db_path = tmp_path / "test.db"
        set_test_db(monkeypatch, db_path)
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

    def test_search_embeddings_not_found(self, tmp_path, capsys):
        """Test search command with non-existent embeddings database."""
        nonexistent_path = tmp_path / "nonexistent_embeddings"

        with patch.object(
            sys,
            "argv",
            [
                "neurips-abstracts",
                "search",
                "test query",
                "--embeddings-path",
                str(nonexistent_path),
            ],
        ):
            exit_code = main()

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Embeddings database not found" in captured.err

    def test_search_lm_studio_not_available(self, tmp_path, capsys):
        """Test search command when OpenAI API is not available."""
        # Create embeddings directory
        embeddings_path = tmp_path / "embeddings"
        embeddings_path.mkdir()

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
                    "--embeddings-path",
                    str(embeddings_path),
                ],
            ):
                exit_code = main()

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Failed to connect to OpenAI API" in captured.err

    def test_search_success(self, tmp_path, capsys):
        """Test search command completes successfully."""
        embeddings_path = tmp_path / "embeddings"
        embeddings_path.mkdir()

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
                    "--embeddings-path",
                    str(embeddings_path),
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

    def test_search_with_abstract(self, tmp_path, capsys):
        """Test search command with --show-abstract flag."""
        embeddings_path = tmp_path / "embeddings"
        embeddings_path.mkdir()

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
                    "--embeddings-path",
                    str(embeddings_path),
                    "--show-abstract",
                ],
            ):
                exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Test Paper" in captured.out
        assert "Abstract: This is a test abstract" in captured.out

    def test_search_with_filter(self, tmp_path, capsys):
        """Test search command with metadata filter."""
        embeddings_path = tmp_path / "embeddings"
        embeddings_path.mkdir()

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
                    "--embeddings-path",
                    str(embeddings_path),
                    "--where",
                    "decision=Accept (poster)",
                ],
            ):
                exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Filter: {'decision': 'Accept (poster)'}" in captured.out
        assert "Filtered Paper" in captured.out

    def test_search_no_results(self, tmp_path, capsys):
        """Test search command with no results."""
        embeddings_path = tmp_path / "embeddings"
        embeddings_path.mkdir()

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
                    "--embeddings-path",
                    str(embeddings_path),
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
        set_test_db(monkeypatch, db_path)
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
                    "--embeddings-path",
                    str(embeddings_path),
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
        set_test_db(monkeypatch, nonexistent_db)

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
                    "--embeddings-path",
                    str(embeddings_path),
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
        set_test_db(monkeypatch, db_path)
        db = DatabaseManager()
        db.connect()
        db.close()

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
                    "--embeddings-path",
                    str(embeddings_path),
                ],
            ):
                exit_code = main()

        # Should succeed (database connection is no longer attempted from search command)
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "101" in captured.out  # Shows author ID as-is since author name resolution is not in search

    def test_search_unexpected_exception(self, tmp_path, capsys):
        """Test search command with unexpected exception."""
        embeddings_path = tmp_path / "embeddings"
        embeddings_path.mkdir()

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
                    "--embeddings-path",
                    str(embeddings_path),
                ],
            ):
                exit_code = main()

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Unexpected error" in captured.err


class TestChatCommand:
    """Test cases for the chat command."""

    def test_chat_embeddings_not_found(self, tmp_path, capsys):
        """Test chat command when embeddings don't exist."""
        from abstracts_explorer.cli import chat_command
        import argparse

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

    def test_chat_lm_studio_not_available(self, tmp_path, capsys):
        """Test chat command when OpenAI API is not available."""
        from abstracts_explorer.cli import chat_command
        import argparse

        embeddings_path = tmp_path / "chroma_db"
        embeddings_path.mkdir()

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

    def test_chat_rag_error(self, tmp_path, capsys):
        """Test chat command with RAG error."""
        from abstracts_explorer.cli import chat_command
        from abstracts_explorer.rag import RAGError
        import argparse

        embeddings_path = tmp_path / "chroma_db"
        embeddings_path.mkdir()

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
        with patch("abstracts_explorer.web_ui.run_server", side_effect=FileNotFoundError("Database not found: /nonexistent/test.db")):
            exit_code = web_ui_command(args)

        # Should exit with code 1 but not show traceback
        assert exit_code == 1
        captured = capsys.readouterr()
        # Should NOT show "Error starting web server" or traceback
        # The error message is printed by run_server itself before raising the exception
        assert "Error starting web server" not in captured.err


class TestMainDispatch:
    """Test main() command dispatch."""

    def test_main_chat_command(self, tmp_path):
        """Test main() dispatches chat command."""
        embeddings_path = tmp_path / "chroma_db"
        embeddings_path.mkdir()

        with patch.object(sys, "argv", ["neurips-abstracts", "chat", "--embeddings-path", str(embeddings_path)]):
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


class TestCLISearchErrorHandling:
    """Test search command error handling paths."""

    def test_search_command_where_parse_warning(self, tmp_path, capsys):
        """Test warning when where clause cannot be parsed (lines 229-230)."""
        import argparse

        embeddings_path = tmp_path / "chroma_db"
        embeddings_path.mkdir()

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

    def test_search_command_general_exception(self, tmp_path, capsys):
        """Test general exception handling in search (lines 308-309)."""
        import argparse

        embeddings_path = tmp_path / "chroma_db"
        embeddings_path.mkdir()

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

        set_test_db(monkeypatch, db_path)
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
                    "--output",
                    str(tmp_path / "chroma_db"),
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

    def test_chat_empty_input_continues(self, tmp_path, capsys):
        """Test that empty input is skipped in chat loop."""
        from abstracts_explorer.cli import chat_command
        import argparse

        embeddings_path = tmp_path / "chroma_db"
        embeddings_path.mkdir()

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

    def test_chat_quit_command(self, tmp_path, capsys):
        """Test chat exits on 'quit' command."""
        from abstracts_explorer.cli import chat_command
        import argparse

        embeddings_path = tmp_path / "chroma_db"
        embeddings_path.mkdir()

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

    def test_chat_q_command(self, tmp_path, capsys):
        """Test chat exits on 'q' command."""
        from abstracts_explorer.cli import chat_command
        import argparse

        embeddings_path = tmp_path / "chroma_db"
        embeddings_path.mkdir()

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

    def test_chat_reset_command(self, tmp_path, capsys):
        """Test chat reset command works."""
        from abstracts_explorer.cli import chat_command
        import argparse

        embeddings_path = tmp_path / "chroma_db"
        embeddings_path.mkdir()

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

    def test_chat_help_command(self, tmp_path, capsys):
        """Test chat help command displays help."""
        from abstracts_explorer.cli import chat_command
        import argparse

        embeddings_path = tmp_path / "chroma_db"
        embeddings_path.mkdir()

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

    def test_chat_with_query_and_show_sources(self, tmp_path, capsys):
        """Test chat processes query and shows sources."""
        from abstracts_explorer.cli import chat_command
        import argparse

        embeddings_path = tmp_path / "chroma_db"
        embeddings_path.mkdir()

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

    def test_chat_with_export(self, tmp_path, capsys):
        """Test chat exports conversation at end."""
        from abstracts_explorer.cli import chat_command
        import argparse

        embeddings_path = tmp_path / "chroma_db"
        embeddings_path.mkdir()
        export_file = tmp_path / "conversation.json"

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

    def test_chat_keyboard_interrupt(self, tmp_path, capsys):
        """Test chat handles Ctrl+C gracefully."""
        from abstracts_explorer.cli import chat_command
        import argparse

        embeddings_path = tmp_path / "chroma_db"
        embeddings_path.mkdir()

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

    def test_chat_eoferror(self, tmp_path, capsys):
        """Test chat handles EOF (Ctrl+D) gracefully."""
        from abstracts_explorer.cli import chat_command
        import argparse

        embeddings_path = tmp_path / "chroma_db"
        embeddings_path.mkdir()

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

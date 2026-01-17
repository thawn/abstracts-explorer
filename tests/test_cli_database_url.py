"""
Test DATABASE_URL configuration support in CLI commands.

This module tests that CLI commands properly respect the DATABASE_URL
environment variable for PostgreSQL and other database backends.
"""

import sys
from unittest.mock import Mock, patch
from abstracts_explorer.cli import main


class TestCLIDatabaseURLConfiguration:
    """Test cases for PAPER_DB configuration support."""

    def test_create_embeddings_with_database_url_from_env(self, tmp_path, monkeypatch):
        """Test that create-embeddings respects PAPER_DB environment variable."""
        from abstracts_explorer.database import DatabaseManager
        from abstracts_explorer.plugin import LightweightPaper

        # Create test database
        db_path = tmp_path / "test.db"
        database_url = f"sqlite:///{db_path.absolute()}"
        db = DatabaseManager(database_url=database_url)
        with db:
            db.create_tables()
            paper = LightweightPaper(
                title="Test Paper",
                abstract="Test abstract",
                authors=["Test Author"],
                session="Session",
                poster_position="P1",
                year=2025,
                conference="TestConf",
            )
            db.add_paper(paper)

        # Set PAPER_DB environment variable (can be URL or path)
        database_url = f"sqlite:///{db_path.absolute()}"
        monkeypatch.setenv("PAPER_DB", database_url)

        # Mock embeddings manager
        with patch("abstracts_explorer.cli.EmbeddingsManager") as MockEM:
            mock_em = Mock()
            mock_em.check_model_compatibility.return_value = (True, None, "test-model")
            mock_em.test_lm_studio_connection.return_value = True
            mock_em.embed_from_database.return_value = 1
            mock_em.get_collection_stats.return_value = {"name": "test", "count": 1}
            MockEM.return_value = mock_em

            with patch.object(
                sys,
                "argv",
                [
                    "abstracts-explorer",
                    "create-embeddings",
                    "--output",
                    str(tmp_path / "chroma_db"),
                ],
            ):
                # Force config reload to pick up PAPER_DB
                from abstracts_explorer.config import get_config
                get_config(reload=True)
                
                exit_code = main()

        assert exit_code == 0
        # Verify that embed_from_database was called with database_url parameter
        mock_em.embed_from_database.assert_called_once()
        call_kwargs = mock_em.embed_from_database.call_args.kwargs
        assert "database_url" in call_kwargs
        assert database_url in call_kwargs["database_url"]

    def test_create_embeddings_paper_db_with_file_path(self, tmp_path, monkeypatch):
        """Test that PAPER_DB works with file path (not just URL)."""
        from abstracts_explorer.database import DatabaseManager
        from abstracts_explorer.plugin import LightweightPaper

        # Create test database
        db_path = tmp_path / "test.db"
        database_url = f"sqlite:///{db_path.absolute()}"
        db = DatabaseManager(database_url=database_url)
        with db:
            db.create_tables()
            paper = LightweightPaper(
                title="Test Paper",
                abstract="Test abstract",
                authors=["Test Author"],
                session="Session",
                poster_position="P1",
                year=2025,
                conference="TestConf",
            )
            db.add_paper(paper)

        # Set PAPER_DB to file path (not URL) - should still work
        monkeypatch.setenv("PAPER_DB", str(db_path))

        # Mock embeddings manager
        with patch("abstracts_explorer.cli.EmbeddingsManager") as MockEM:
            mock_em = Mock()
            mock_em.check_model_compatibility.return_value = (True, None, "test-model")
            mock_em.test_lm_studio_connection.return_value = True
            mock_em.embed_from_database.return_value = 1
            mock_em.get_collection_stats.return_value = {"name": "test", "count": 1}
            MockEM.return_value = mock_em

            with patch.object(
                sys,
                "argv",
                [
                    "abstracts-explorer",
                    "create-embeddings",
                    "--output",
                    str(tmp_path / "chroma_db"),
                ],
            ):
                # Force config reload to pick up PAPER_DB
                from abstracts_explorer.config import get_config
                get_config(reload=True)
                
                exit_code = main()

        assert exit_code == 0
        # Verify that embed_from_database was called with database_url parameter
        mock_em.embed_from_database.assert_called_once()
        call_kwargs = mock_em.embed_from_database.call_args.kwargs
        # The database_url should be automatically converted from path
        assert "database_url" in call_kwargs
        assert str(db_path.absolute()) in call_kwargs["database_url"]

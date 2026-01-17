"""
Test PAPER_DB configuration support in CLI commands.

This module tests that CLI commands properly respect the PAPER_DB
environment variable for database configuration (supports both SQLite paths
and PostgreSQL URLs).
"""

import sys
from unittest.mock import Mock, patch
from abstracts_explorer.cli import main


class TestCLIPaperDBConfiguration:
    """Test cases for PAPER_DB configuration support."""

    def test_create_embeddings_with_database_url_from_env(self, tmp_path, monkeypatch):
        """Test that create-embeddings respects PAPER_DB environment variable."""
        from abstracts_explorer.database import DatabaseManager
        from abstracts_explorer.plugin import LightweightPaper
        from abstracts_explorer.config import get_config

        # Create test database
        db_path = tmp_path / "test.db"
        database_url = f"sqlite:///{db_path.absolute()}"
        
        # Set environment variable and reload config for db creation
        monkeypatch.setenv("PAPER_DB", database_url)
        get_config(reload=True)
        
        db = DatabaseManager()
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

        # PAPER_DB is already set via monkeypatch
        # Also set EMBEDDING_DB for the ChromaDB location
        monkeypatch.setenv("EMBEDDING_DB", str(tmp_path / "chroma_db"))

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
                ],
            ):
                # Force config reload to pick up env vars
                get_config(reload=True)
                
                exit_code = main()

        assert exit_code == 0
        # Verify that embed_from_database was called
        mock_em.embed_from_database.assert_called_once()
        call_kwargs = mock_em.embed_from_database.call_args.kwargs
        assert "database_url" in call_kwargs
        assert database_url in call_kwargs["database_url"]

    def test_create_embeddings_paper_db_with_file_path(self, tmp_path, monkeypatch):
        """Test that PAPER_DB works with file path (not just URL)."""
        from abstracts_explorer.database import DatabaseManager
        from abstracts_explorer.plugin import LightweightPaper
        from abstracts_explorer.config import get_config

        # Create test database
        db_path = tmp_path / "test.db"
        database_url = f"sqlite:///{db_path.absolute()}"
        
        # Set environment variable and reload config for db creation
        monkeypatch.setenv("PAPER_DB", database_url)
        get_config(reload=True)
        
        db = DatabaseManager()
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
        # Also set EMBEDDING_DB
        monkeypatch.setenv("EMBEDDING_DB", str(tmp_path / "chroma_db"))

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
                ],
            ):
                # Force config reload to pick up PAPER_DB
                get_config(reload=True)
                
                exit_code = main()

        assert exit_code == 0
        # Verify that embed_from_database was called with database_url parameter
        mock_em.embed_from_database.assert_called_once()
        call_kwargs = mock_em.embed_from_database.call_args.kwargs
        # The database_url should be automatically converted from path
        assert "database_url" in call_kwargs
        assert str(db_path.absolute()) in call_kwargs["database_url"]

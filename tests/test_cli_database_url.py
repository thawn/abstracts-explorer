"""
Test DATABASE_URL configuration support in CLI commands.

This module tests that CLI commands properly respect the DATABASE_URL
environment variable for PostgreSQL and other database backends.
"""

import sys
from unittest.mock import Mock, patch
from abstracts_explorer.cli import main


class TestCLIDatabaseURLConfiguration:
    """Test cases for DATABASE_URL configuration support."""

    def test_create_embeddings_with_database_url_from_env(self, tmp_path, monkeypatch):
        """Test that create-embeddings respects DATABASE_URL environment variable."""
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

        # Set DATABASE_URL environment variable
        database_url = f"sqlite:///{db_path.absolute()}"
        monkeypatch.setenv("DATABASE_URL", database_url)

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
                # Force config reload to pick up DATABASE_URL
                from abstracts_explorer.config import get_config
                get_config(reload=True)
                
                exit_code = main()

        assert exit_code == 0
        # Verify that embed_from_database was called with database_url parameter
        mock_em.embed_from_database.assert_called_once()
        call_kwargs = mock_em.embed_from_database.call_args.kwargs
        assert "database_url" in call_kwargs
        assert database_url in call_kwargs["database_url"]

    def test_create_embeddings_database_url_takes_precedence(self, tmp_path, monkeypatch):
        """Test that DATABASE_URL takes precedence over --db-path argument."""
        from abstracts_explorer.database import DatabaseManager
        from abstracts_explorer.plugin import LightweightPaper

        # Create two databases
        db_path1 = tmp_path / "test1.db"
        db_path2 = tmp_path / "test2.db"
        
        for db_path in [db_path1, db_path2]:
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

        # Set DATABASE_URL to db_path1 (should take precedence)
        database_url = f"sqlite:///{db_path1.absolute()}"
        monkeypatch.setenv("DATABASE_URL", database_url)

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
                    "--db-path",
                    str(db_path2),  # This should be ignored in favor of DATABASE_URL
                    "--output",
                    str(tmp_path / "chroma_db"),
                ],
            ):
                # Force config reload to pick up DATABASE_URL
                from abstracts_explorer.config import get_config
                get_config(reload=True)
                
                exit_code = main()

        assert exit_code == 0
        # Verify that database_url from env was used (not db_path2 from CLI)
        mock_em.embed_from_database.assert_called_once()
        call_kwargs = mock_em.embed_from_database.call_args.kwargs
        # The database_url should be from environment (db_path1)
        assert "database_url" in call_kwargs
        assert str(db_path1.absolute()) in call_kwargs["database_url"]

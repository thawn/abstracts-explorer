"""
Tests for multi-database backend support.

Tests that verify DatabaseManager works with different database backends
including PostgreSQL. PostgreSQL tests are skipped if PostgreSQL is not available.
"""

import pytest
import os

from abstracts_explorer.database import DatabaseManager, DatabaseError
from abstracts_explorer.plugin import LightweightPaper


# Check if PostgreSQL is available via environment variable
# Set POSTGRES_TEST_URL to test PostgreSQL backend
# Example: POSTGRES_TEST_URL=postgresql://user:password@localhost/test_db
POSTGRES_TEST_URL = os.getenv("POSTGRES_TEST_URL")
HAS_POSTGRES = POSTGRES_TEST_URL is not None

# Skip marker for PostgreSQL tests
skip_without_postgres = pytest.mark.skipif(
    not HAS_POSTGRES,
    reason="PostgreSQL not available (set POSTGRES_TEST_URL to test)"
)


class TestMultiDatabaseBackend:
    """Test multi-database backend support."""

    def test_sqlite_via_database_url(self, tmp_path):
        """Test using SQLite via DATABASE_URL parameter."""
        db_path = tmp_path / "test.db"
        database_url = f"sqlite:///{db_path}"
        
        db = DatabaseManager(database_url=database_url)
        with db:
            db.create_tables()
            
            # Test basic operations
            paper = LightweightPaper(
                title="Test Paper",
                authors=["Author"],
                abstract="Test abstract",
                session="Session A",
                poster_position="P1",
                year=2025,
                conference="TestConf",
            )
            
            paper_uid = db.add_paper(paper)
            assert paper_uid is not None
            
            count = db.get_paper_count()
            assert count == 1

    def test_sqlite_legacy_path(self, tmp_path):
        """Test using SQLite via legacy db_path parameter."""
        db_path = tmp_path / "test.db"
        
        db = DatabaseManager(db_path=db_path)
        with db:
            db.create_tables()
            
            # Test basic operations
            paper = LightweightPaper(
                title="Test Paper",
                authors=["Author"],
                abstract="Test abstract",
                session="Session A",
                poster_position="P1",
                year=2025,
                conference="TestConf",
            )
            
            paper_uid = db.add_paper(paper)
            assert paper_uid is not None
            
            count = db.get_paper_count()
            assert count == 1

    def test_must_provide_db_path_or_url(self):
        """Test that either db_path or database_url must be provided."""
        with pytest.raises(DatabaseError, match="Either db_path or database_url must be provided"):
            DatabaseManager()

    def test_database_url_takes_precedence(self, tmp_path):
        """Test that database_url takes precedence over db_path."""
        db_path1 = tmp_path / "test1.db"
        db_path2 = tmp_path / "test2.db"
        database_url = f"sqlite:///{db_path2}"
        
        db = DatabaseManager(db_path=db_path1, database_url=database_url)
        
        # database_url should take precedence
        assert db.database_url == database_url
        assert db_path2.stem in db.database_url

    @skip_without_postgres
    def test_postgresql_basic_operations(self):
        """Test basic operations with PostgreSQL backend."""
        # This test requires PostgreSQL to be available
        db = DatabaseManager(database_url=POSTGRES_TEST_URL)
        
        with db:
            # Create tables
            db.create_tables()
            
            # Test adding a paper
            paper = LightweightPaper(
                title="PostgreSQL Test Paper",
                authors=["Author 1", "Author 2"],
                abstract="Testing PostgreSQL backend",
                session="Session A",
                poster_position="P1",
                year=2025,
                conference="TestConf",
                keywords=["test", "postgresql"],
            )
            
            paper_uid = db.add_paper(paper)
            assert paper_uid is not None
            
            # Test retrieving count
            count = db.get_paper_count()
            assert count >= 1  # Might have papers from previous tests
            
            # Test search
            results = db.search_papers(keyword="PostgreSQL", limit=10)
            assert len(results) > 0
            assert any("PostgreSQL" in r["title"] for r in results)
            
            # Test filter options
            filters = db.get_filter_options()
            assert "sessions" in filters
            assert "years" in filters
            assert "conferences" in filters

    @skip_without_postgres
    def test_postgresql_multiple_papers(self):
        """Test adding multiple papers with PostgreSQL."""
        db = DatabaseManager(database_url=POSTGRES_TEST_URL)
        
        with db:
            db.create_tables()
            
            papers = [
                LightweightPaper(
                    title=f"Paper {i}",
                    authors=[f"Author {i}"],
                    abstract=f"Abstract {i}",
                    session=f"Session {i % 3}",
                    poster_position=f"P{i}",
                    year=2025,
                    conference="TestConf",
                )
                for i in range(5)
            ]
            
            count = db.add_papers(papers)
            assert count == 5
            
            total = db.get_paper_count()
            assert total >= 5

    @skip_without_postgres
    def test_postgresql_embedding_model_metadata(self):
        """Test embedding model metadata with PostgreSQL."""
        db = DatabaseManager(database_url=POSTGRES_TEST_URL)
        
        with db:
            db.create_tables()
            
            # Set embedding model
            model_name = "test-embedding-model"
            db.set_embedding_model(model_name)
            
            # Retrieve embedding model
            retrieved_model = db.get_embedding_model()
            assert retrieved_model == model_name
            
            # Update embedding model
            new_model = "updated-embedding-model"
            db.set_embedding_model(new_model)
            
            retrieved_model = db.get_embedding_model()
            assert retrieved_model == new_model

    @skip_without_postgres
    def test_postgresql_idempotent_create_tables(self):
        """Test that create_tables can be called multiple times without error."""
        db = DatabaseManager(database_url=POSTGRES_TEST_URL)
        
        with db:
            # Call create_tables multiple times - should not raise errors
            db.create_tables()
            db.create_tables()
            db.create_tables()
            
            # Verify tables still work
            paper = LightweightPaper(
                title="Idempotent Test Paper",
                authors=["Test Author"],
                abstract="Testing idempotent table creation",
                session="Session A",
                poster_position="P1",
                year=2025,
                conference="TestConf",
            )
            
            paper_uid = db.add_paper(paper)
            assert paper_uid is not None


def test_database_url_in_config(tmp_path, monkeypatch):
    """Test that DATABASE_URL is properly loaded from config."""
    from abstracts_explorer.config import Config
    
    db_path = tmp_path / "test.db"
    database_url = f"sqlite:///{db_path}"
    
    # Set environment variable
    monkeypatch.setenv("DATABASE_URL", database_url)
    
    config = Config()
    assert config.database_url == database_url


def test_legacy_paper_db_path_in_config(tmp_path, monkeypatch):
    """Test that PAPER_DB_PATH still works for backward compatibility."""
    from abstracts_explorer.config import Config
    
    db_path = tmp_path / "test.db"
    
    # Set environment variable
    monkeypatch.setenv("PAPER_DB_PATH", str(db_path))
    
    config = Config()
    # Should be converted to SQLite URL
    assert "sqlite:///" in config.database_url
    assert str(db_path) in config.database_url

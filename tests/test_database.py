"""
Tests for the database module.

Note: This test file contains tests for core database functionality (connection,
table creation, context managers, etc.). Tests using the old schema (string IDs,
old field names) have been removed. See test_authors.py for comprehensive tests
using the new schema with integer IDs and proper author relationships.
"""

import json
import pytest
import sqlite3
from pathlib import Path

from neurips_abstracts.database import DatabaseManager, DatabaseError

# Fixtures are now imported from conftest.py:
# - db_manager: DatabaseManager instance with temporary database
# - connected_db: Connected database with tables created


class TestDatabaseManager:
    """Tests for DatabaseManager class."""

    def test_init(self, tmp_path):
        """Test DatabaseManager initialization."""
        db_path = tmp_path / "test.db"
        db = DatabaseManager(db_path)

        assert db.db_path == db_path
        assert db.connection is None

    def test_connect(self, db_manager):
        """Test database connection."""
        db_manager.connect()

        assert db_manager.connection is not None
        assert isinstance(db_manager.connection, sqlite3.Connection)

        db_manager.close()

    def test_connect_creates_directories(self, tmp_path):
        """Test that connect creates parent directories."""
        db_path = tmp_path / "subdir" / "another" / "test.db"
        db = DatabaseManager(db_path)
        db.connect()

        assert db_path.parent.exists()
        assert db.connection is not None

        db.close()

    def test_close(self, db_manager):
        """Test database close."""
        db_manager.connect()
        db_manager.close()

        assert db_manager.connection is None

    def test_close_without_connection(self, db_manager):
        """Test closing without connection doesn't raise error."""
        db_manager.close()  # Should not raise
        assert db_manager.connection is None

    def test_context_manager(self, tmp_path):
        """Test DatabaseManager as context manager."""
        db_path = tmp_path / "test.db"

        with DatabaseManager(db_path) as db:
            assert db.connection is not None

        # Connection should be closed after exiting context
        assert db.connection is None

    def test_create_tables(self, db_manager):
        """Test table creation."""
        db_manager.connect()
        db_manager.create_tables()

        # Check if tables exist
        cursor = db_manager.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        assert "papers" in tables

        db_manager.close()

    def test_create_tables_without_connection(self, db_manager):
        """Test create_tables raises error when not connected."""
        with pytest.raises(DatabaseError, match="Not connected to database"):
            db_manager.create_tables()

    def test_load_json_data_without_connection(self, db_manager):
        """Test load_json_data raises error when not connected."""
        with pytest.raises(DatabaseError, match="Not connected to database"):
            db_manager.load_json_data([])

    def test_load_json_data_invalid_type(self, connected_db):
        """Test load_json_data with invalid data type."""
        with pytest.raises(ValueError, match="Data must be a dictionary or list"):
            connected_db.load_json_data("invalid data")

    def test_query_without_connection(self, db_manager):
        """Test query raises error when not connected."""
        with pytest.raises(DatabaseError, match="Not connected to database"):
            db_manager.query("SELECT * FROM papers")

    def test_query_with_invalid_sql(self, connected_db):
        """Test query with invalid SQL."""
        with pytest.raises(DatabaseError, match="Query failed"):
            connected_db.query("SELECT * FROM nonexistent_table")

    def test_get_paper_count_empty(self, connected_db):
        """Test getting paper count on empty database."""
        assert connected_db.get_paper_count() == 0

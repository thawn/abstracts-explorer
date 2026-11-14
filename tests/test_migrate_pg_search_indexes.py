"""
Tests for the PostgreSQL search index migration script.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple
from unittest.mock import MagicMock, Mock

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "migrate_pg_search_indexes.py"
POSTGRES_TEST_URL = os.getenv("POSTGRES_TEST_URL")
skip_without_postgres = pytest.mark.skipif(
    not POSTGRES_TEST_URL,
    reason="PostgreSQL not available (set POSTGRES_TEST_URL to test)",
)


@pytest.fixture(scope="module")
def migration():
    """Load the migration script as a module."""
    spec = importlib.util.spec_from_file_location("migrate_pg_search_indexes", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_db(
    migration: Any,
    *,
    dialect: str = "postgresql",
    installed: Any = 1,
    status: Optional[List[Tuple[str, bool]]] = None,
    exec_side_effect: Optional[Any] = None,
) -> Tuple[Any, MagicMock]:
    """Build a mocked DatabaseManager and connection."""
    db = Mock()
    db.database_url = "postgresql://user:pass@host/db"
    db.engine = Mock()
    db.engine.dialect.name = dialect

    connection = MagicMock()
    connection.execute.return_value = status or []
    connection.scalar.return_value = installed
    connection.execution_options.return_value = connection
    if exec_side_effect is not None:
        connection.exec_driver_sql.side_effect = exec_side_effect
    db.engine.connect.return_value = connection
    return db, connection


class TestStatements:
    """Tests for DDL statement builders."""

    def test_create_statement_concurrent(self, migration):
        index = migration.INDEXES[0]
        assert migration.create_statement(index) == (
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {index.name} ON {index.target}"
        )

    def test_create_statement_blocking(self, migration):
        index = migration.INDEXES[0]
        assert migration.create_statement(index, concurrently=False) == (
            f"CREATE INDEX IF NOT EXISTS {index.name} ON {index.target}"
        )

    def test_drop_statement_concurrent(self, migration):
        index = migration.INDEXES[0]
        assert migration.drop_statement(index) == f"DROP INDEX CONCURRENTLY IF EXISTS {index.name}"

    def test_drop_statement_blocking(self, migration):
        index = migration.INDEXES[0]
        assert migration.drop_statement(index, concurrently=False) == f"DROP INDEX IF EXISTS {index.name}"


class TestRunMigration:
    """Tests for the migration runner."""

    def test_rejects_non_postgresql(self, migration):
        db, _ = make_db(migration, dialect="sqlite")
        with pytest.raises(migration.DatabaseError, match="only supports PostgreSQL"):
            migration.run_migration(db)

    def test_creates_all_indexes_and_analyzes(self, migration):
        db, connection = make_db(migration, status=[])

        statements = migration.run_migration(db)

        assert len(statements) == len(migration.INDEXES) + 1
        assert statements[-1] == "ANALYZE papers"
        assert connection.exec_driver_sql.call_count == len(migration.INDEXES) + 1
        assert any("gin_trgm_ops" in statement for statement in statements)
        assert all("CONCURRENTLY" in statement for statement in statements[:-1])

    def test_creates_extension_when_missing(self, migration):
        db, connection = make_db(migration, installed=None, status=[])

        statements = migration.run_migration(db)

        assert statements[0] == "CREATE EXTENSION IF NOT EXISTS pg_trgm"
        assert connection.exec_driver_sql.call_args_list[0].args[0] == statements[0]

    def test_skips_extension_when_present(self, migration):
        db, connection = make_db(migration, installed=1, status=[])

        statements = migration.run_migration(db)

        assert not any("CREATE EXTENSION" in statement for statement in statements)
        assert connection.exec_driver_sql.call_args_list[0].args[0].startswith("CREATE INDEX")

    def test_skip_extension_option(self, migration):
        db, connection = make_db(migration, installed=None, status=[])

        statements = migration.run_migration(db, skip_extension=True)

        connection.scalar.assert_not_called()
        assert not any("CREATE EXTENSION" in statement for statement in statements)

    def test_extension_permission_error(self, migration):
        db, _ = make_db(
            migration,
            installed=None,
            status=[],
            exec_side_effect=RuntimeError("permission denied"),
        )

        with pytest.raises(migration.DatabaseError, match="pg_trgm"):
            migration.run_migration(db)

    def test_dry_run_executes_nothing(self, migration):
        db, connection = make_db(migration, installed=None, status=[])

        statements = migration.run_migration(db, dry_run=True)

        connection.exec_driver_sql.assert_not_called()
        assert statements[0] == "CREATE EXTENSION IF NOT EXISTS pg_trgm"
        assert statements[-1] == "ANALYZE papers"
        assert len(statements) == len(migration.INDEXES) + 2

    def test_skips_valid_index(self, migration):
        title_index = migration.INDEXES[0]
        db, _ = make_db(migration, status=[(title_index.name, True)])

        statements = migration.run_migration(db)

        assert not any(
            statement.startswith("CREATE INDEX") and title_index.name in statement for statement in statements
        )

    def test_recreates_invalid_index(self, migration):
        title_index = migration.INDEXES[0]
        db, connection = make_db(migration, status=[(title_index.name, False)])

        statements = migration.run_migration(db)

        drop_position = next(
            index for index, statement in enumerate(statements) if statement.startswith("DROP INDEX")
        )
        create_position = next(
            index
            for index, statement in enumerate(statements)
            if statement.startswith("CREATE INDEX") and title_index.name in statement
        )
        assert drop_position < create_position
        assert connection.exec_driver_sql.call_count == len(migration.INDEXES) + 2

    def test_blocking_build(self, migration):
        db, _ = make_db(migration, status=[])

        statements = migration.run_migration(db, concurrently=False)

        assert all("CONCURRENTLY" not in statement for statement in statements)

    def test_drop_indexes(self, migration):
        status = [(index.name, True) for index in migration.INDEXES]
        db, connection = make_db(migration, status=status)

        statements = migration.run_migration(db, drop=True)

        assert len(statements) == len(migration.INDEXES)
        assert all(statement.startswith("DROP INDEX") for statement in statements)
        assert "ANALYZE papers" not in statements
        assert connection.scalar.call_count == 0

    def test_drop_skips_missing_indexes(self, migration):
        db, connection = make_db(migration, status=[])

        statements = migration.run_migration(db, drop=True)

        assert statements == []
        connection.exec_driver_sql.assert_not_called()


@skip_without_postgres
def test_postgresql_index_migration(migration):
    """Create and verify the managed indexes on a real PostgreSQL database."""
    from abstracts_explorer.database import DatabaseManager
    from tests.conftest import set_test_db

    assert POSTGRES_TEST_URL is not None
    set_test_db(POSTGRES_TEST_URL)

    db = DatabaseManager()
    try:
        db.connect()
        assert db.engine is not None
        if db.engine.dialect.name != "postgresql":
            pytest.skip("Configured test database is not PostgreSQL")

        migration.run_migration(db, drop=True, verbose=False)
        try:
            migration.run_migration(db, verbose=False)
        except migration.DatabaseError as exc:
            if "pg_trgm" in str(exc):
                pytest.skip(str(exc))
            raise

        with db.engine.connect() as connection:
            status = migration.read_index_status(
                connection,
                [index.name for index in migration.INDEXES],
            )

        assert set(status) == {index.name for index in migration.INDEXES}
        assert all(status.values())
    finally:
        try:
            if db.engine is not None:
                migration.run_migration(db, drop=True, verbose=False)
        except Exception:
            pass
        db.close()

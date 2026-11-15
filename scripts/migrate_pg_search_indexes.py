#!/usr/bin/env python3
"""Create PostgreSQL search indexes for the papers table.

This is a manual migration script.  It is intentionally host-only: do not copy
it into the runtime image and do not run it automatically at application
startup.

The migration is idempotent and can be run repeatedly.  It:

1. Ensures the ``pg_trgm`` extension is available.
2. Creates GIN trigram indexes for title, abstract, keywords, and authors.
3. Creates composite B-tree indexes for conference/year/session metadata.
4. Runs ``ANALYZE papers`` so the planner can use the new indexes.

Usage::

    python scripts/migrate_pg_search_indexes.py --dry-run
    python scripts/migrate_pg_search_indexes.py
    python scripts/migrate_pg_search_indexes.py --drop

Inside the running container (the database host is not exposed to the host)::

    podman exec -i abstracts-explorer sh -c '
      curl -fsSL https://raw.githubusercontent.com/thawn/abstracts-explorer/<branch>/scripts/migrate_pg_search_indexes.py \
      | python - --dry-run
    '
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence

from sqlalchemy import bindparam, text
from sqlalchemy.engine import Connection

from abstracts_explorer.database import DatabaseError, DatabaseManager


@dataclass(frozen=True)
class IndexDefinition:
    """
    A single index managed by this migration.

    Attributes
    ----------
    name : str
        Index name.
    target : str
        SQL target clause, e.g. ``papers USING gin (title gin_trgm_ops)``.
    """

    name: str
    target: str


INDEXES: tuple[IndexDefinition, ...] = (
    IndexDefinition("ix_papers_title_trgm", "papers USING gin (title gin_trgm_ops)"),
    IndexDefinition("ix_papers_abstract_trgm", "papers USING gin (abstract gin_trgm_ops)"),
    IndexDefinition("ix_papers_keywords_trgm", "papers USING gin (keywords gin_trgm_ops)"),
    IndexDefinition("ix_papers_authors_trgm", "papers USING gin (authors gin_trgm_ops)"),
    IndexDefinition("ix_papers_conference_year", "papers (conference, year)"),
    IndexDefinition("ix_papers_conference_year_session", "papers (conference, year, session)"),
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=("Create or drop PostgreSQL trigram and composite search indexes on the papers table.")
    )
    parser.add_argument(
        "--paper-db",
        help="Override PAPER_DB for this command only. Accepts a PostgreSQL URL or SQLite path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned DDL and index state without executing anything.",
    )
    parser.add_argument(
        "--no-concurrently",
        action="store_true",
        help="Use blocking index builds instead of CREATE INDEX CONCURRENTLY.",
    )
    parser.add_argument(
        "--skip-extension",
        action="store_true",
        help="Assume pg_trgm is already installed by a database administrator.",
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop the managed indexes instead of creating them.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-index progress.",
    )
    return parser.parse_args()


def create_statement(index: IndexDefinition, concurrently: bool = True) -> str:
    """
    Build a CREATE INDEX statement.

    Parameters
    ----------
    index : IndexDefinition
        Index to create.
    concurrently : bool, optional
        Whether to use ``CONCURRENTLY``.  Defaults to ``True``.

    Returns
    -------
    str
        SQL statement.
    """
    concurrent = "CONCURRENTLY " if concurrently else ""
    return f"CREATE INDEX {concurrent}IF NOT EXISTS {index.name} ON {index.target}"


def drop_statement(index: IndexDefinition, concurrently: bool = True) -> str:
    """
    Build a DROP INDEX statement.

    Parameters
    ----------
    index : IndexDefinition
        Index to drop.
    concurrently : bool, optional
        Whether to use ``CONCURRENTLY``.  Defaults to ``True``.

    Returns
    -------
    str
        SQL statement.
    """
    concurrent = "CONCURRENTLY " if concurrently else ""
    return f"DROP INDEX {concurrent}IF EXISTS {index.name}"


def read_index_status(connection: Connection, names: Sequence[str]) -> Dict[str, bool]:
    """
    Read the validity of managed indexes.

    Parameters
    ----------
    connection : Connection
        PostgreSQL connection.
    names : sequence of str
        Index names to check.

    Returns
    -------
    dict
        Mapping of existing index name to ``indisvalid``.
    """
    if not names:
        return {}

    statement = text("""
        SELECT c.relname, i.indisvalid
        FROM pg_index i
        JOIN pg_class c ON c.oid = i.indexrelid
        WHERE c.relname IN :names
        """).bindparams(bindparam("names", expanding=True))
    result = connection.execute(statement, {"names": list(names)})
    return {str(row[0]): bool(row[1]) for row in result}


def ensure_trgm_extension(
    connection: Connection,
    *,
    dry_run: bool,
    skip_extension: bool,
    verbose: bool,
) -> List[str]:
    """
    Ensure the ``pg_trgm`` extension is installed.

    Parameters
    ----------
    connection : Connection
        Autocommit PostgreSQL connection.
    dry_run : bool
        Only print the planned statement.
    skip_extension : bool
        Assume the extension is installed by a DBA.
    verbose : bool
        Print progress.

    Returns
    -------
    list of str
        Statements that were executed or planned.

    Raises
    ------
    DatabaseError
        If the extension cannot be created.
    """
    if skip_extension:
        if verbose:
            print("Skipping pg_trgm extension check (--skip-extension)")
        return []

    installed = connection.scalar(text("SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm'"))
    if installed:
        return []

    statement = "CREATE EXTENSION IF NOT EXISTS pg_trgm"
    if dry_run:
        print(statement)
        return [statement]

    try:
        connection.exec_driver_sql(statement)
    except Exception as exc:
        raise DatabaseError(
            "Failed to create the pg_trgm extension. Install it as a database superuser "
            "with 'CREATE EXTENSION pg_trgm;', or rerun with --skip-extension after a DBA "
            "installs it."
        ) from exc

    if verbose:
        print("Created pg_trgm extension")
    return [statement]


def _execute(connection: Connection, statement: str, *, dry_run: bool) -> None:
    """
    Print and optionally execute one SQL statement.

    Parameters
    ----------
    connection : Connection
        Autocommit PostgreSQL connection.
    statement : str
        SQL statement.
    dry_run : bool
        When ``True``, only print the statement.
    """
    print(statement)
    if not dry_run:
        connection.exec_driver_sql(statement)


def run_migration(
    db: DatabaseManager,
    *,
    dry_run: bool = False,
    concurrently: bool = True,
    skip_extension: bool = False,
    drop: bool = False,
    verbose: bool = False,
) -> List[str]:
    """
    Create or drop the managed PostgreSQL indexes.

    Parameters
    ----------
    db : DatabaseManager
        Database manager connected to the target database.
    dry_run : bool, optional
        Only print planned statements.  Defaults to ``False``.
    concurrently : bool, optional
        Use concurrent index builds.  Defaults to ``True``.
    skip_extension : bool, optional
        Assume ``pg_trgm`` is installed.  Defaults to ``False``.
    drop : bool, optional
        Drop the managed indexes instead of creating them.  Defaults to ``False``.
    verbose : bool, optional
        Print per-index progress.  Defaults to ``False``.

    Returns
    -------
    list of str
        Statements that were executed or planned.

    Raises
    ------
    DatabaseError
        If the database is not PostgreSQL or an operation fails.
    """
    if db.engine is None:
        db.connect()
    assert db.engine is not None

    if db.engine.dialect.name != "postgresql":
        raise DatabaseError("This migration only supports PostgreSQL databases.")

    executed: List[str] = []
    connection = db.engine.connect().execution_options(isolation_level="AUTOCOMMIT")
    try:
        if not drop:
            executed.extend(
                ensure_trgm_extension(
                    connection,
                    dry_run=dry_run,
                    skip_extension=skip_extension,
                    verbose=verbose,
                )
            )

        status = read_index_status(connection, [index.name for index in INDEXES])

        for index in INDEXES:
            exists = index.name in status
            valid = status.get(index.name, False)

            if drop:
                if not exists:
                    if verbose:
                        print(f"Skipping {index.name}: not present")
                    continue
                statement = drop_statement(index, concurrently)
                _execute(connection, statement, dry_run=dry_run)
                executed.append(statement)
                continue

            if valid:
                if verbose:
                    print(f"Skipping {index.name}: already valid")
                continue

            if exists and not valid:
                statement = drop_statement(index, concurrently)
                _execute(connection, statement, dry_run=dry_run)
                executed.append(statement)

            statement = create_statement(index, concurrently)
            _execute(connection, statement, dry_run=dry_run)
            executed.append(statement)

        if not drop:
            statement = "ANALYZE papers"
            _execute(connection, statement, dry_run=dry_run)
            executed.append(statement)
    finally:
        connection.close()

    return executed


def main() -> int:
    """
    Run the migration command.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).
    """
    args = parse_args()

    if args.paper_db:
        os.environ["PAPER_DB"] = args.paper_db
        from abstracts_explorer.config import get_config

        get_config(reload=True)

    db = DatabaseManager()
    try:
        print(f"Database: {db._mask_url(db.database_url)}")
        if args.drop:
            print("Mode: DROP indexes")
        elif args.dry_run:
            print("Mode: dry run")
        else:
            print("Mode: CREATE indexes")

        statements = run_migration(
            db,
            dry_run=args.dry_run,
            concurrently=not args.no_concurrently,
            skip_extension=args.skip_extension,
            drop=args.drop,
            verbose=args.verbose,
        )

        action = "Rollback" if args.drop else "Migration"
        state = "planned" if args.dry_run else "completed"
        print(f"{action} {state}: {len(statements)} statement(s)")
        return 0
    except Exception as exc:
        print(f"Migration failed: {exc}", file=sys.stderr)
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())

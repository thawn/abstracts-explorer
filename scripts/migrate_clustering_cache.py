#!/usr/bin/env python3
"""Migrate existing clustering cache entries to the new table structure.

This one-time migration script:

1. Adds the ``conference``, ``year``, and ``n_clusters`` columns to the
   ``clustering_cache`` table if they are missing (schema migration).
2. Fills the new columns from existing data in ``clustering_params`` and
   ``results_json`` (data migration).

It reads ``conferences`` and ``years`` from the ``clustering_params`` JSON
and copies them into the dedicated columns.  It also fills ``n_clusters``
from ``results_json["statistics"]["n_clusters"]`` when it is currently
``NULL``.  After migration the ``conferences`` and ``years`` keys are
removed from ``clustering_params``.

Usage::

    python scripts/migrate_clustering_cache.py
    python scripts/migrate_clustering_cache.py --paper-db /path/to/abstracts.db
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from sqlalchemy import inspect, select, text

from abstracts_explorer.database import DatabaseError, DatabaseManager
from abstracts_explorer.db_models import ClusteringCache


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Migrate existing clustering cache entries to fill the new "
            "conference, year, and n_clusters columns."
        )
    )
    parser.add_argument(
        "--paper-db",
        help="Override PAPER_DB for this command only. Accepts a SQLite path or full database URL.",
    )
    return parser.parse_args()


def add_missing_columns(db: DatabaseManager) -> list[str]:
    """Add missing columns to the clustering_cache table.

    Uses ``ALTER TABLE`` to add ``conference`` (TEXT), ``year`` (INTEGER),
    and ``n_clusters`` (INTEGER) columns when they are absent.  This handles
    databases that were created before this schema change.

    Parameters
    ----------
    db : DatabaseManager
        Open database connection.

    Returns
    -------
    list[str]
        Names of columns that were added.

    Raises
    ------
    DatabaseError
        If a column cannot be added.
    """
    if not db.engine:
        raise DatabaseError("Not connected to database")

    inspector = inspect(db.engine)

    # If the table doesn't exist at all, create it via the ORM metadata
    if "clustering_cache" not in inspector.get_table_names():
        db.create_tables()
        return []

    existing = {col["name"] for col in inspector.get_columns("clustering_cache")}
    added: list[str] = []

    column_ddl = {
        "conference": "ALTER TABLE clustering_cache ADD COLUMN conference TEXT",
        "year": "ALTER TABLE clustering_cache ADD COLUMN year INTEGER",
        "n_clusters": "ALTER TABLE clustering_cache ADD COLUMN n_clusters INTEGER",
    }

    try:
        with db.engine.begin() as conn:
            for col_name, ddl in column_ddl.items():
                if col_name not in existing:
                    conn.execute(text(ddl))
                    added.append(col_name)
    except Exception as e:
        raise DatabaseError(f"Failed to add missing columns: {e}") from e

    return added


def migrate_clustering_cache(db: DatabaseManager) -> int:
    """Migrate existing clustering cache entries to fill conference, year, and n_clusters columns.

    Reads ``conferences`` and ``years`` from the ``clustering_params`` JSON
    and copies them into the dedicated ``conference`` and ``year`` columns.
    Also fills ``n_clusters`` from ``results_json["statistics"]["n_clusters"]``
    when it is currently ``NULL``.  After migration the ``conferences`` and
    ``years`` keys are removed from ``clustering_params``.

    Parameters
    ----------
    db : DatabaseManager
        Open database connection.

    Returns
    -------
    int
        Number of cache entries migrated.

    Raises
    ------
    DatabaseError
        If the migration fails.
    """
    if not db._session:
        raise DatabaseError("Not connected to database")

    try:
        entries = db._session.execute(select(ClusteringCache)).scalars().all()
        migrated = 0
        for entry in entries:
            changed = False

            # --- Migrate conference/year from clustering_params ---
            if entry.conference is None and entry.clustering_params:
                try:
                    params = json.loads(entry.clustering_params)
                except (ValueError, TypeError):
                    params = {}
                conferences = params.get("conferences", [])
                years = params.get("years", [])
                if len(conferences) == 1:
                    entry.conference = conferences[0]
                    changed = True
                if len(years) == 1:
                    entry.year = years[0]
                    changed = True
                # Remove conferences/years keys from clustering_params
                if "conferences" in params or "years" in params:
                    params.pop("conferences", None)
                    params.pop("years", None)
                    entry.clustering_params = json.dumps(params) if params else None
                    changed = True

            # --- Fill n_clusters from results_json ---
            if entry.n_clusters is None and entry.results_json:
                try:
                    results = json.loads(entry.results_json)
                except (ValueError, TypeError):
                    results = {}
                actual = results.get("statistics", {}).get("n_clusters")
                if actual is not None:
                    entry.n_clusters = int(actual)
                    changed = True

            if changed:
                migrated += 1

        db._session.commit()
        return migrated

    except DatabaseError:
        raise
    except Exception as e:
        db._session.rollback()
        raise DatabaseError(f"Failed to migrate clustering cache: {str(e)}") from e


def main() -> int:
    """Run the migration.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).
    """
    args = parse_args()

    if args.paper_db:
        os.environ["PAPER_DB"] = args.paper_db
        # Force config reload so DatabaseManager picks up the override
        from abstracts_explorer.config import get_config

        get_config(reload=True)

    try:
        with DatabaseManager() as db:
            added = add_missing_columns(db)
            if added:
                print(f"Added missing columns: {', '.join(added)}")
            # Expire all ORM state so SQLAlchemy re-reads the updated schema
            db._session.expire_all()
            count = migrate_clustering_cache(db)
            if count == 0:
                print("✅ No cache entries needed migration.")
            else:
                entry_word = "entry" if count == 1 else "entries"
                print(f"✅ Migrated {count} clustering cache {entry_word}.")
        return 0
    except DatabaseError as exc:
        print(f"Failed to migrate clustering cache: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


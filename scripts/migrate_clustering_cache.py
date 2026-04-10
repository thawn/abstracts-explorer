#!/usr/bin/env python3
"""Migrate existing clustering cache entries to the new table structure.

This one-time migration script fills the new ``conference``, ``year``, and
``n_clusters`` columns on existing :class:`ClusteringCache` rows.

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

from sqlalchemy import select

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
            db.create_tables()
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

#!/usr/bin/env python3
"""Backfill the ChromaDB collection name into existing clustering-cache rows.

Clustering cache rows written before collection scoping do not contain the
``__collection__`` key in ``clustering_params``.  The web UI and MCP tools now
look up cache entries by collection, so those rows are invisible until they
are either backfilled or regenerated.

This idempotent migration adds the configured collection to unscoped rows.
Rows whose ``clustering_params`` is ``NULL`` cannot be mapped safely and are
reported so they can be regenerated with ``clustering cache populate --force``.

Usage::

    python scripts/backfill_clustering_cache_collection.py --dry-run
    python scripts/backfill_clustering_cache_collection.py
    python scripts/backfill_clustering_cache_collection.py \
        --paper-db /path/to/abstracts.db --collection papers
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional, Tuple

from sqlalchemy import select

from abstracts_explorer.config import get_config
from abstracts_explorer.database import DatabaseError, DatabaseManager
from abstracts_explorer.db_models import ClusteringCache

COLLECTION_KEY = "__collection__"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Add the ChromaDB collection name to clustering-cache rows that were "
            "written before collection scoping."
        )
    )
    parser.add_argument(
        "--paper-db",
        help="Override PAPER_DB for this command only. Accepts a SQLite path or full database URL.",
    )
    parser.add_argument(
        "--collection",
        help="ChromaDB collection name to add (defaults to COLLECTION_NAME from config).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without committing.",
    )
    return parser.parse_args()


def backfill_clustering_cache_collection(
    db: DatabaseManager,
    collection_name: str,
    dry_run: bool = False,
) -> Tuple[int, int, int, int]:
    """
    Add ``__collection__`` to unscoped clustering-cache rows.

    Parameters
    ----------
    db : DatabaseManager
        Connected database manager.
    collection_name : str
        ChromaDB collection name to store in the cache key.
    dry_run : bool, optional
        When True, do not commit changes.

    Returns
    -------
    tuple of (updated, already_scoped, null_params, malformed)
        Counts describing the rows encountered.

    Raises
    ------
    DatabaseError
        If the database operation fails.
    """
    updated = 0
    already_scoped = 0
    null_params = 0
    malformed = 0

    rows = db.session.execute(select(ClusteringCache)).scalars().all()
    for row in rows:
        if row.clustering_params is None:
            null_params += 1
            continue
        try:
            params = json.loads(str(row.clustering_params))
        except (TypeError, ValueError):
            malformed += 1
            continue
        if not isinstance(params, dict):
            malformed += 1
            continue
        if COLLECTION_KEY in params:
            already_scoped += 1
            continue
        params[COLLECTION_KEY] = collection_name
        setattr(row, "clustering_params", json.dumps(params))
        updated += 1

    if updated and not dry_run:
        db.session.commit()

    return updated, already_scoped, null_params, malformed


def main() -> int:
    """Run the backfill migration.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).
    """
    args = parse_args()

    if args.paper_db:
        os.environ["PAPER_DB"] = args.paper_db
    config = get_config(reload=True)

    collection_name: Optional[str] = args.collection or config.collection_name
    if not collection_name:
        print("No collection name configured; pass --collection.", file=sys.stderr)
        return 1

    try:
        with DatabaseManager() as db:
            db.create_tables()
            updated, already_scoped, null_params, malformed = backfill_clustering_cache_collection(
                db,
                collection_name,
                dry_run=args.dry_run,
            )
    except DatabaseError as exc:
        print(f"Failed to backfill clustering cache: {exc}", file=sys.stderr)
        return 1

    action = "Would update" if args.dry_run else "Updated"
    print(f"{action} {updated} clustering-cache row(s) with collection '{collection_name}'.")
    print(f"Already scoped: {already_scoped}")
    if null_params:
        print(
            f"Skipped {null_params} row(s) with NULL clustering_params. "
            "Regenerate those with 'abstracts-explorer clustering cache populate --force'.",
            file=sys.stderr,
        )
    if malformed:
        print(f"Skipped {malformed} row(s) with malformed clustering_params.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

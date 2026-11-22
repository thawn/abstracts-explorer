#!/usr/bin/env python3
"""Remove legacy hierarchy levels and obsolete label caches.

This one-off cleanup:

1. Verifies every clustering cache row contains ``hierarchy.tree``.
2. Removes ``levels``, ``nr_levels``, and ``level_scale`` from cached
   hierarchy payloads.
3. Removes ``nr_levels`` and ``level_scale`` from ``clustering_params``.
4. Drops the obsolete ``flat_label_cache`` and ``hierarchical_label_cache``
   tables.

Usage::

    python scripts/prune_legacy_hierarchy_cache.py --dry-run
    python scripts/prune_legacy_hierarchy_cache.py
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from abstracts_explorer.config import get_config
from abstracts_explorer.database import DatabaseManager

LEGACY_HIERARCHY_KEYS = ("levels", "nr_levels", "level_scale")
LEGACY_PARAM_KEYS = ("nr_levels", "level_scale")
OBSOLETE_TABLES = ("flat_label_cache", "hierarchical_label_cache")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Remove legacy hierarchy levels and obsolete label caches.")
    parser.add_argument("--paper-db", help="Override PAPER_DB for this command only.")
    parser.add_argument("--conference", help="Only clean cache entries for this conference.")
    parser.add_argument("--year", type=int, help="Only clean cache entries for this year.")
    parser.add_argument("--collection", help="Only clean cache entries for this ChromaDB collection.")
    parser.add_argument("--embedding-model", help="Only clean cache entries for this embedding model.")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing.")
    parser.add_argument(
        "--keep-label-tables",
        action="store_true",
        help="Do not drop the obsolete flat/hierarchical label cache tables.",
    )
    return parser.parse_args()


def _collection_matches(entry: Dict[str, Any], collection: Optional[str]) -> bool:
    """Return whether a cache entry belongs to the requested collection."""
    if collection is None:
        return True
    params = entry.get("clustering_params") or {}
    return params.get("__collection__") == collection


def _clean_results(results: Dict[str, Any]) -> bool:
    """Remove legacy hierarchy keys from results in place; return whether changed."""
    hierarchy = results.get("hierarchy")
    if not isinstance(hierarchy, dict):
        return False
    changed = False
    for key in LEGACY_HIERARCHY_KEYS:
        if key in hierarchy:
            hierarchy.pop(key, None)
            changed = True
    return changed


def _clean_params(params: Optional[Dict[str, Any]]) -> bool:
    """Remove legacy level keys from clustering params in place; return whether changed."""
    if not isinstance(params, dict):
        return False
    changed = False
    for key in LEGACY_PARAM_KEYS:
        if key in params:
            params.pop(key, None)
            changed = True
    return changed


def _table_exists(db: DatabaseManager, table: str) -> bool:
    """Return whether a table exists in the connected database."""
    try:
        db.session.execute(text(f"SELECT 1 FROM {table} LIMIT 1"))
        return True
    except Exception:
        db.session.rollback()
        return False


def _drop_table(db: DatabaseManager, table: str) -> None:
    """Drop a table if it exists."""
    db.session.execute(text(f"DROP TABLE IF EXISTS {table}"))
    db.session.commit()


def main() -> int:
    """Run the legacy hierarchy cleanup.

    Returns
    -------
    int
        Exit code (0 for success, non-zero when any row failed).
    """
    args = parse_args()

    if args.paper_db:
        os.environ["PAPER_DB"] = args.paper_db
    get_config(reload=True)

    processed = 0
    updated = 0
    skipped = 0
    failed = 0
    dropped_tables: List[str] = []

    try:
        with DatabaseManager() as db:
            entries = db.list_clustering_cache_entries(
                conference=args.conference,
                year=args.year,
                embedding_model=args.embedding_model,
            )
            entries = [entry for entry in entries if _collection_matches(entry, args.collection)]
            if not entries:
                print("No matching clustering cache entries found.", file=sys.stderr)
                return 1

            print(f"Found {len(entries)} matching clustering cache entr{'y' if len(entries) == 1 else 'ies'}.")
            for entry in entries:
                processed += 1
                results = entry["results"]
                params = entry.get("clustering_params")
                label = f"{entry.get('conference') or 'all'} {entry.get('year') or ''}".strip()

                if not (results.get("hierarchy") or {}).get("tree"):
                    failed += 1
                    print(
                        f"[{entry['id']}] {label}: FAILED: missing hierarchy tree; "
                        "regenerate with 'abstracts-explorer clustering pre-generate --force'",
                        file=sys.stderr,
                    )
                    continue

                results_changed = _clean_results(results)
                params_changed = _clean_params(params)
                if not results_changed and not params_changed:
                    skipped += 1
                    print(f"[{entry['id']}] {label}: already clean")
                    continue

                if not args.dry_run:
                    db.update_clustering_cache_entry(
                        entry["id"],
                        results=results if results_changed else None,
                        clustering_params=params if params_changed else None,
                    )
                updated += 1
                action = "would clean" if args.dry_run else "cleaned"
                print(f"[{entry['id']}] {label}: {action} legacy hierarchy cache")

            if not args.keep_label_tables:
                for table in OBSOLETE_TABLES:
                    exists = _table_exists(db, table)
                    if not exists:
                        continue
                    if args.dry_run:
                        dropped_tables.append(table)
                    else:
                        _drop_table(db, table)
                        dropped_tables.append(table)

    except Exception as exc:
        print(f"Cleanup failed: {exc}", file=sys.stderr)
        return 1

    print("\nSummary:")
    print(f"  processed: {processed}")
    print(f"  updated:   {updated}")
    print(f"  skipped:   {skipped}")
    print(f"  failed:    {failed}")
    if dropped_tables:
        verb = "would drop" if args.dry_run else "dropped"
        print(f"  {verb} tables: {', '.join(dropped_tables)}")
    else:
        print("  obsolete label tables: none found")
    if args.dry_run:
        print("  dry run: no changes were written")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

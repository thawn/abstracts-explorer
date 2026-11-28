#!/usr/bin/env python3
"""
Backfill generated keywords in the papers table.

The script ensures the ``generated_keywords`` column exists, then generates
TF-IDF keywords per ``(conference, year)`` corpus for papers that do not have
keywords yet.  Existing values are kept unless ``--force`` is supplied.

This is a script-only operation; the normal download command generates
keywords automatically for downloaded units.

Usage
-----
    python scripts/backfill_generated_keywords.py
    python scripts/backfill_generated_keywords.py --conference NeurIPS --year 2025
    python scripts/backfill_generated_keywords.py --dry-run --json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from sqlalchemy import inspect, text

from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.keywords.generation import generate_keywords_for_scope


def ensure_generated_keywords_column(db: DatabaseManager) -> bool:
    """
    Ensure the ``papers.generated_keywords`` column exists.

    Parameters
    ----------
    db : DatabaseManager
        Open database manager.

    Returns
    -------
    bool
        ``True`` when the column was added, ``False`` when it already existed.
    """
    if db.engine is None:
        db.connect()
    assert db.engine is not None

    inspector = inspect(db.engine)
    if not inspector.has_table("papers"):
        return False
    columns = {column["name"] for column in inspector.get_columns("papers")}
    if "generated_keywords" in columns:
        return False

    dialect = db.engine.dialect.name
    with db.engine.begin() as connection:
        if dialect == "sqlite":
            rows = connection.execute(text("PRAGMA table_info(papers)")).all()
            names = {row[1] for row in rows}
            if "generated_keywords" not in names:
                connection.execute(text("ALTER TABLE papers ADD COLUMN generated_keywords TEXT"))
        else:
            connection.execute(text("ALTER TABLE papers ADD COLUMN IF NOT EXISTS generated_keywords TEXT"))
    return True


def _iter_scopes(
    db: DatabaseManager,
    conference: Optional[str],
    year: Optional[int],
) -> List[Dict[str, Any]]:
    """Return the ``(conference, year)`` scopes to process."""
    conditions = ["conference IS NOT NULL", "year IS NOT NULL"]
    params: Dict[str, Any] = {}
    if conference is not None:
        conditions.append("conference = :conference")
        params["conference"] = conference
    if year is not None:
        conditions.append("year = :year")
        params["year"] = year
    where = " AND ".join(conditions)
    stmt = text(
        f"SELECT conference, year, COUNT(*) AS total FROM papers WHERE {where} "
        "GROUP BY conference, year ORDER BY conference, year"
    )
    return [
        {"conference": str(row.conference), "year": int(row.year), "total": int(row.total)}
        for row in db.session.execute(stmt, params).all()
    ]


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Backfill generated keywords")
    parser.add_argument("--conference", default=None, help="Only process this conference")
    parser.add_argument("--year", type=int, default=None, help="Only process this year")
    parser.add_argument("--force", action="store_true", help="Regenerate existing keywords")
    parser.add_argument("--dry-run", action="store_true", help="Compute without writing")
    parser.add_argument("--json", action="store_true", help="Print a JSON summary")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """
    Run the keyword backfill.

    Parameters
    ----------
    argv : list of str, optional
        Argument list for testing.  Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code.
    """
    args = _parse_args(argv)

    summary: Dict[str, Any] = {
        "updated": 0,
        "dry_run": bool(args.dry_run),
        "scopes": [],
    }

    with DatabaseManager() as db:
        db.create_tables()
        ensure_generated_keywords_column(db)
        scopes = _iter_scopes(db, args.conference, args.year)
        for scope in scopes:
            updated = generate_keywords_for_scope(
                db,
                scope["conference"],
                scope["year"],
                force=args.force,
                dry_run=args.dry_run,
            )
            scope["updated"] = updated
            summary["updated"] += updated
            summary["scopes"].append(scope)
            if not args.json:
                action = "would update" if args.dry_run else "updated"
                print(
                    f"{scope['conference']} {scope['year']}: {action} {updated:,} " f"of {scope['total']:,} paper(s)"
                )

    if args.json:
        print(json.dumps(summary, sort_keys=True))
    else:
        print(f"Total {'would update' if args.dry_run else 'updated'}: {summary['updated']:,}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

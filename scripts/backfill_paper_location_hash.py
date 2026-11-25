#!/usr/bin/env python3
"""
Backfill paper location and content hash columns.

Adds the ``location`` and ``content_hash`` columns to the ``papers`` table when
missing, computes the content hash for every row, and derives presentation
locations for multi-site NeurIPS years from the EventHosts virtual site.

The script is idempotent and can be re-run safely.

Usage
-----
    python scripts/backfill_paper_location_hash.py
    python scripts/backfill_paper_location_hash.py --conference NeurIPS --year 2026
    python scripts/backfill_paper_location_hash.py --skip-locations
"""

from __future__ import annotations

import argparse
import sys
from typing import Iterable, Optional, Tuple

import requests
from sqlalchemy import inspect, text

from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.utils.hashing import paper_content_hash
from abstracts_explorer.plugin.locations import LocationMap, build_neurips_location_map

_USER_AGENT = "abstracts-explorer location backfill"
_BATCH_SIZE = 1000


def _ensure_columns(db: DatabaseManager) -> None:
    """Add missing paper columns and indexes."""
    assert db.engine is not None
    inspector = inspect(db.engine)
    columns = {column["name"] for column in inspector.get_columns("papers")}
    with db.engine.begin() as connection:
        if "location" not in columns:
            connection.execute(text("ALTER TABLE papers ADD COLUMN location VARCHAR"))
            print("Added papers.location column")
        if "content_hash" not in columns:
            connection.execute(text("ALTER TABLE papers ADD COLUMN content_hash VARCHAR(71)"))
            print("Added papers.content_hash column")
        connection.execute(text("CREATE INDEX IF NOT EXISTS ix_papers_location ON papers (location)"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS ix_papers_content_hash ON papers (content_hash)"))


def _backfill_content_hashes(
    db: DatabaseManager,
    force: bool,
    conference: Optional[str] = None,
    year: Optional[int] = None,
) -> int:
    """Compute content hashes for paper rows and return the update count."""
    conditions = []
    params = {}
    if conference is not None:
        conditions.append("conference = :conference")
        params["conference"] = conference
    if year is not None:
        conditions.append("year = :year")
        params["year"] = year
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    updated = 0
    pending = []
    rows = db.session.execute(
        text(f"SELECT uid, title, abstract, content_hash FROM papers {where} ORDER BY uid"), params
    ).all()
    for uid, title, abstract, existing in rows:
        expected = paper_content_hash(title, abstract)
        if not force and existing == expected:
            continue
        pending.append({"hash": expected, "uid": uid})
        updated += 1
        if len(pending) >= _BATCH_SIZE:
            db.session.execute(text("UPDATE papers SET content_hash = :hash WHERE uid = :uid"), pending)
            db.session.commit()
            pending.clear()
    if pending:
        db.session.execute(text("UPDATE papers SET content_hash = :hash WHERE uid = :uid"), pending)
    db.session.commit()
    return updated


def _fetch_text(url: str, timeout: float) -> str:
    """Fetch a URL and return decoded text."""
    response = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=timeout)
    response.raise_for_status()
    return response.text


def _iter_scope(db: DatabaseManager, conference: Optional[str], year: Optional[int]) -> Iterable[Tuple[str, int]]:
    """Yield ``(conference, year)`` scopes to process."""
    conditions = []
    params = {}
    if conference is not None:
        conditions.append("conference = :conference")
        params["conference"] = conference
    if year is not None:
        conditions.append("year = :year")
        params["year"] = year
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    stmt = text(f"SELECT DISTINCT conference, year FROM papers {where} ORDER BY conference, year")
    for row_conference, row_year in db.session.execute(stmt, params).all():
        if row_conference is None or row_year is None:
            continue
        yield str(row_conference), int(row_year)


def _backfill_locations(
    db: DatabaseManager,
    conference: Optional[str],
    year: Optional[int],
    timeout: float,
) -> int:
    """Derive and persist locations for multi-site NeurIPS years."""
    total = 0
    for scope_conference, scope_year in _iter_scope(db, conference, year):
        if scope_conference.lower() != "neurips":
            continue
        try:
            location_map: LocationMap = build_neurips_location_map(scope_year, lambda url: _fetch_text(url, timeout))
        except Exception as exc:
            print(f"Skipping {scope_conference} {scope_year}: {exc}", file=sys.stderr)
            continue
        if not location_map.is_multi_location:
            continue

        rows = db.session.execute(
            text("SELECT uid, session, room_name FROM papers " "WHERE conference = :conference AND year = :year"),
            {"conference": scope_conference, "year": scope_year},
        ).all()
        updated = 0
        pending = []
        for uid, session, room_name in rows:
            location = location_map.assign(session, room_name)
            if location is None:
                continue
            pending.append({"location": location, "uid": uid})
            updated += 1
            if len(pending) >= _BATCH_SIZE:
                db.session.execute(text("UPDATE papers SET location = :location WHERE uid = :uid"), pending)
                db.session.commit()
                pending.clear()
        if pending:
            db.session.execute(text("UPDATE papers SET location = :location WHERE uid = :uid"), pending)
        db.session.commit()
        total += updated
        print(f"{scope_conference} {scope_year}: {updated} presentations across {', '.join(location_map.locations)}")
    return total


def main() -> int:
    """Run the backfill."""
    parser = argparse.ArgumentParser(description="Backfill paper location and content hash columns")
    parser.add_argument("--paper-db", dest="paper_db", default=None, help="Path or URL of the paper database")
    parser.add_argument("--conference", default=None, help="Only process this conference")
    parser.add_argument("--year", type=int, default=None, help="Only process this year")
    parser.add_argument("--skip-locations", action="store_true", help="Do not derive locations")
    parser.add_argument("--force-hashes", action="store_true", help="Recompute hashes even when already set")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds")
    args = parser.parse_args()

    with DatabaseManager() as db:
        if args.paper_db:
            db.database_url = args.paper_db
        db.connect()
        _ensure_columns(db)
        updated_hashes = _backfill_content_hashes(
            db, force=args.force_hashes, conference=args.conference, year=args.year
        )
        print(f"Content hash updated for {updated_hashes} row(s)")
        if not args.skip_locations:
            updated_locations = _backfill_locations(db, args.conference, args.year, args.timeout)
            print(f"Location updated for {updated_locations} row(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

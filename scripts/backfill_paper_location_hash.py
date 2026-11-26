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
from tqdm import tqdm

from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.utils.hashing import paper_content_hash
from abstracts_explorer.plugin.locations import LocationMap
from abstracts_explorer.plugins.neurips_downloader import build_neurips_location_map

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

    scope = "all papers"
    if conference or year:
        scope = f"{conference or 'all conferences'} / {year or 'all years'}"
    print(f"Content hashes: scanning {scope}...")

    updated = 0
    skipped = 0
    pending = []
    rows = db.session.execute(
        text(f"SELECT uid, title, abstract, content_hash FROM papers {where} ORDER BY uid"), params
    ).all()
    print(f"Content hashes: {len(rows):,} row(s) to inspect")

    progress = tqdm(total=len(rows), desc="Content hashes", unit="row")
    try:
        for uid, title, abstract, existing in rows:
            expected = paper_content_hash(title, abstract)
            if not force and existing == expected:
                skipped += 1
                progress.update(1)
                continue
            pending.append({"hash": expected, "uid": uid})
            updated += 1
            progress.update(1)
            progress.set_postfix(updated=f"{updated:,}", unchanged=f"{skipped:,}")
            if len(pending) >= _BATCH_SIZE:
                db.session.execute(text("UPDATE papers SET content_hash = :hash WHERE uid = :uid"), pending)
                db.session.commit()
                pending.clear()
        if pending:
            db.session.execute(text("UPDATE papers SET content_hash = :hash WHERE uid = :uid"), pending)
        db.session.commit()
    finally:
        progress.close()

    print(f"Content hashes: updated {updated:,}, unchanged {skipped:,}")
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
    scopes = list(_iter_scope(db, conference, year))
    print(f"Locations: inspecting {len(scopes)} conference/year scope(s)")

    for scope_conference, scope_year in tqdm(scopes, desc="Location scopes", unit="scope"):
        if scope_conference.lower() != "neurips":
            print(f"Locations: skipping {scope_conference} {scope_year} (not NeurIPS)")
            continue
        print(f"Locations: discovering sites for {scope_conference} {scope_year} from neurips.cc...")
        try:
            location_map: LocationMap = build_neurips_location_map(scope_year, lambda url: _fetch_text(url, timeout))
        except Exception as exc:
            print(f"Locations: skipping {scope_conference} {scope_year}: {exc}", file=sys.stderr)
            continue
        if not location_map.is_multi_location:
            print(f"Locations: skipping {scope_conference} {scope_year} (single site)")
            continue

        print(f"Locations: {scope_conference} {scope_year} -> {', '.join(location_map.locations)}")
        rows = db.session.execute(
            text("SELECT uid, session, room_name FROM papers WHERE conference = :conference AND year = :year"),
            {"conference": scope_conference, "year": scope_year},
        ).all()
        updated = 0
        pending = []
        progress = tqdm(total=len(rows), desc=f"{scope_conference} {scope_year}", unit="row")
        try:
            for uid, session, room_name in rows:
                location = location_map.assign(session, room_name)
                progress.update(1)
                if location is None:
                    continue
                pending.append({"location": location, "uid": uid})
                updated += 1
                progress.set_postfix(updated=f"{updated:,}")
                if len(pending) >= _BATCH_SIZE:
                    db.session.execute(text("UPDATE papers SET location = :location WHERE uid = :uid"), pending)
                    db.session.commit()
                    pending.clear()
            if pending:
                db.session.execute(text("UPDATE papers SET location = :location WHERE uid = :uid"), pending)
            db.session.commit()
        finally:
            progress.close()
        total += updated
        print(
            f"Locations: {scope_conference} {scope_year} updated {updated:,} of {len(rows):,} "
            f"presentation(s) across {', '.join(location_map.locations)}"
        )
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
        print(f"Backfill target: {db.database_url}")
        if args.conference or args.year:
            print(f"Scope: conference={args.conference or '*'} year={args.year or '*'}")
        else:
            print("Scope: all conferences and years")
        _ensure_columns(db)
        updated_hashes = _backfill_content_hashes(
            db, force=args.force_hashes, conference=args.conference, year=args.year
        )
        print("")
        print(f"Content hash updated for {updated_hashes:,} row(s)")
        if not args.skip_locations:
            print("")
            updated_locations = _backfill_locations(db, args.conference, args.year, args.timeout)
            print(f"Location updated for {updated_locations:,} row(s)")
        print("")
        print("Backfill complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

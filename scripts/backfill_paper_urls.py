#!/usr/bin/env python3
"""Re-download NeurIPS/ICLR/ICML data and backfill missing paper URLs.

Before the ``paper_url`` → ``paper_pdf_url`` fallback was added in
:func:`convert_to_lightweight_schema`, papers that only had a ``paper_url``
field in the source JSON ended up with a NULL ``paper_pdf_url`` in the
database.  This script re-downloads the conference JSON for every
conference/year combination already present in the database, recomputes the
lightweight schema (which now includes the fallback), and updates any paper
whose ``paper_pdf_url`` is currently NULL.

Usage::

    python scripts/backfill_paper_urls.py
    python scripts/backfill_paper_urls.py --paper-db /path/to/abstracts.db
    python scripts/backfill_paper_urls.py --yes          # skip confirmation
    python scripts/backfill_paper_urls.py --dry-run      # preview only
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, List, Tuple

from sqlalchemy import select

from abstracts_explorer.database import DatabaseError, DatabaseManager
from abstracts_explorer.db_models import Paper
from abstracts_explorer.plugin import LightweightPaper
from abstracts_explorer.plugins.neurips_downloader import NeurIPSDownloaderPlugin
from abstracts_explorer.plugins.iclr_downloader import ICLRDownloaderPlugin
from abstracts_explorer.plugins.icml_downloader import ICMLDownloaderPlugin

logger = logging.getLogger(__name__)

# Map conference name (as stored in DB) to downloader class
CONFERENCE_DOWNLOADERS = {
    "NeurIPS": NeurIPSDownloaderPlugin,
    "ICLR": ICLRDownloaderPlugin,
    "ICML": ICMLDownloaderPlugin,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Re-download NeurIPS/ICLR/ICML conference data and backfill "
            "missing paper_pdf_url values in the database."
        )
    )
    parser.add_argument(
        "--paper-db",
        help="Override PAPER_DB for this command only. Accepts a SQLite path or full database URL.",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip the interactive confirmation prompt.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes.",
    )
    return parser.parse_args()


def get_conference_years_in_db(db: DatabaseManager) -> Dict[str, List[int]]:
    """Query the database for distinct conference/year pairs.

    Parameters
    ----------
    db : DatabaseManager
        Open database connection.

    Returns
    -------
    dict
        Mapping of conference name to list of years present in the database.
        Only includes conferences that use the NeurIPS JSON schema
        (NeurIPS, ICLR, ICML).

    Raises
    ------
    DatabaseError
        If the query fails.
    """
    if not db._session:
        raise DatabaseError("Not connected to database")

    try:
        stmt = (
            select(Paper.conference, Paper.year)
            .where(Paper.conference.in_(list(CONFERENCE_DOWNLOADERS.keys())))
            .distinct()
        )
        rows = db._session.execute(stmt).all()

        result: Dict[str, List[int]] = {}
        for conference, year in rows:
            if conference and year:
                result.setdefault(conference, []).append(year)

        # Sort years for consistent ordering
        for years in result.values():
            years.sort()

        return result
    except DatabaseError:
        raise
    except Exception as e:
        raise DatabaseError(f"Failed to query conference/year pairs: {e}") from e


def download_and_build_url_map(conference: str, year: int) -> Dict[str, Dict[str, str | None]]:
    """Download conference data and build a UID → URL mapping.

    Parameters
    ----------
    conference : str
        Conference name (e.g. "NeurIPS").
    year : int
        Conference year.

    Returns
    -------
    dict
        Mapping of UID to dict with URL fields (``paper_pdf_url``,
        ``poster_image_url``, ``url``).  Values may be ``None``.
    """
    downloader_cls = CONFERENCE_DOWNLOADERS[conference]
    downloader = downloader_cls()

    # Download and convert to lightweight papers
    papers: List[LightweightPaper] = downloader.download(year=year, force_download=True)

    url_map: Dict[str, Dict[str, str | None]] = {}
    for paper in papers:
        original_id = str(paper.original_id) if paper.original_id else None
        uid = DatabaseManager.compute_uid(paper.title, original_id, paper.conference, paper.year)
        url_map[uid] = {
            "paper_pdf_url": paper.paper_pdf_url,
            "poster_image_url": paper.poster_image_url,
            "url": paper.url,
        }
    return url_map


def find_papers_missing_urls(db: DatabaseManager, conference: str, year: int) -> List[Paper]:
    """Find papers that have a NULL paper_pdf_url.

    Parameters
    ----------
    db : DatabaseManager
        Open database connection.
    conference : str
        Conference name.
    year : int
        Conference year.

    Returns
    -------
    list of Paper
        Papers with missing ``paper_pdf_url``.

    Raises
    ------
    DatabaseError
        If the query fails.
    """
    if not db._session:
        raise DatabaseError("Not connected to database")

    try:
        stmt = (
            select(Paper)
            .where(Paper.conference == conference)
            .where(Paper.year == year)
            .where(Paper.paper_pdf_url.is_(None))
        )
        return list(db._session.execute(stmt).scalars().all())
    except DatabaseError:
        raise
    except Exception as e:
        raise DatabaseError(f"Failed to query papers with missing URLs: {e}") from e


def backfill_urls(
    db: DatabaseManager,
    conference: str,
    year: int,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """Download data and update papers with missing URLs.

    Parameters
    ----------
    db : DatabaseManager
        Open database connection.
    conference : str
        Conference name (e.g. "NeurIPS").
    year : int
        Conference year.
    dry_run : bool
        If True, only report what would change without modifying the database.

    Returns
    -------
    tuple of (int, int)
        (number of papers updated, number of papers that still have no URL).

    Raises
    ------
    DatabaseError
        If a database operation fails.
    RuntimeError
        If the download fails.
    """
    papers_missing = find_papers_missing_urls(db, conference, year)
    if not papers_missing:
        return 0, 0

    print(f"  Found {len(papers_missing)} paper(s) with missing URLs")

    # Download fresh data
    url_map = download_and_build_url_map(conference, year)
    print(f"  Downloaded {len(url_map)} paper(s) from {conference} {year}")

    updated = 0
    still_missing = 0

    for paper in papers_missing:
        urls = url_map.get(paper.uid)
        if urls and urls.get("paper_pdf_url"):
            if not dry_run:
                paper.paper_pdf_url = urls["paper_pdf_url"]
                # Also backfill poster_image_url and url if missing
                if not paper.poster_image_url and urls.get("poster_image_url"):
                    paper.poster_image_url = urls["poster_image_url"]
                if not paper.url and urls.get("url"):
                    paper.url = urls["url"]
            updated += 1
        else:
            still_missing += 1

    if not dry_run and updated > 0:
        try:
            db._session.commit()
        except Exception as e:
            db._session.rollback()
            raise DatabaseError(f"Failed to commit URL updates: {e}") from e

    return updated, still_missing


def main() -> int:
    """Run the backfill.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).
    """
    args = parse_args()

    logging.basicConfig(level=logging.WARNING)

    if args.paper_db:
        os.environ["PAPER_DB"] = args.paper_db
        from abstracts_explorer.config import get_config

        get_config(reload=True)

    # ------------------------------------------------------------------
    # Step 1: find which conferences/years are in the database
    # ------------------------------------------------------------------
    try:
        with DatabaseManager() as db:
            db.create_tables()
            conf_years = get_conference_years_in_db(db)
    except DatabaseError as exc:
        print(f"❌ Failed to query database: {exc}", file=sys.stderr)
        return 1

    if not conf_years:
        print("No NeurIPS/ICLR/ICML papers found in the database — nothing to do.")
        return 0

    total_pairs = sum(len(years) for years in conf_years.values())
    print(f"Found {total_pairs} conference/year combination(s) in the database:")
    for conf, years in sorted(conf_years.items()):
        print(f"  {conf}: {', '.join(str(y) for y in years)}")

    # ------------------------------------------------------------------
    # Step 2: confirm
    # ------------------------------------------------------------------
    if not args.yes and not args.dry_run:
        print(
            "\nThis will re-download data from the conference websites and "
            "update papers with missing paper_pdf_url values."
        )
        confirm = input("Type 'yes' to confirm: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            return 1

    if args.dry_run:
        print("\n🔍 Dry run — no changes will be made.\n")

    # ------------------------------------------------------------------
    # Step 3: process each conference/year
    # ------------------------------------------------------------------
    total_updated = 0
    total_still_missing = 0
    errors: list[str] = []

    try:
        with DatabaseManager() as db:
            db.create_tables()
            for conference, years in sorted(conf_years.items()):
                for year in years:
                    print(f"\nProcessing {conference} {year}...")
                    try:
                        updated, still_missing = backfill_urls(db, conference, year, dry_run=args.dry_run)
                        total_updated += updated
                        total_still_missing += still_missing
                        if updated > 0:
                            action = "Would update" if args.dry_run else "Updated"
                            print(f"  ✅ {action} {updated} paper(s)")
                        else:
                            print("  ✅ No updates needed")
                        if still_missing > 0:
                            print(f"  ⚠️  {still_missing} paper(s) still have no URL in source data")
                    except RuntimeError as exc:
                        msg = f"{conference} {year}: download failed — {exc}"
                        print(f"  ❌ {msg}", file=sys.stderr)
                        errors.append(msg)
                    except DatabaseError as exc:
                        msg = f"{conference} {year}: database error — {exc}"
                        print(f"  ❌ {msg}", file=sys.stderr)
                        errors.append(msg)
    except DatabaseError as exc:
        print(f"\n❌ Failed to open database: {exc}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Step 4: summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 50}")
    action = "Would update" if args.dry_run else "Updated"
    print(f"{action} {total_updated} paper(s) total.")
    if total_still_missing > 0:
        print(f"⚠️  {total_still_missing} paper(s) have no URL in source data either.")
    if errors:
        print(f"\n⚠️  Finished with {len(errors)} error(s):", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

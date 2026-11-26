#!/usr/bin/env python3
"""
Regenerate clustering caches for scopes with many duplicate presentations.

This is the second half of the hybrid cache cleanup: existing caches are
migrated in place by ``migrate_clustering_cache_locations.py``, and scopes
where the original clustering was most distorted by duplicate rows are
recomputed from scratch.

By default the script only reports what it would regenerate. Pass ``--execute``
to run the clustering pipeline. Regeneration uses the configured LLM backend
and can take a long time for large scopes.

Usage
-----
    python scripts/regenerate_high_duplicate_scopes.py --dry-run
    python scripts/regenerate_high_duplicate_scopes.py --threshold 200
    python scripts/regenerate_high_duplicate_scopes.py --execute
    python scripts/regenerate_high_duplicate_scopes.py --conference ICML --year 2022 --execute
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Tuple

from sqlalchemy import text
from tqdm import tqdm

from abstracts_explorer.clustering.pipeline import compute_clusters_with_cache
from abstracts_explorer.config import get_config
from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.embeddings import EmbeddingsManager


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Regenerate clustering caches for high-duplicate scopes.")
    parser.add_argument("--paper-db", help="Override PAPER_DB for this command only.")
    parser.add_argument("--conference", help="Only process this conference.")
    parser.add_argument("--year", type=int, help="Only process this year.")
    parser.add_argument("--collection", help="Override the ChromaDB collection name.")
    parser.add_argument(
        "--threshold",
        type=int,
        default=200,
        help="Minimum duplicate rows in a scope to trigger regeneration (default: 200).",
    )
    parser.add_argument("--max-scopes", type=int, default=None, help="Process at most this many scopes.")
    parser.add_argument("--execute", action="store_true", help="Actually regenerate (default: report only).")
    parser.add_argument(
        "--no-llm-labels",
        action="store_true",
        help="Use c-TF-IDF labels instead of LLM labels (not compatible with the web cache).",
    )
    parser.add_argument("--show-progress", action="store_true", help="Show BERTopic/LLM progress output.")
    return parser.parse_args()


def high_duplicate_scopes(
    db: DatabaseManager,
    conference: str | None,
    year: int | None,
    threshold: int,
) -> List[Tuple[str, int, int, int]]:
    """
    Return scopes with at least *threshold* duplicate presentation rows.

    Parameters
    ----------
    db : DatabaseManager
        Connected database manager.
    conference : str or None
        Optional conference filter.
    year : int or None
        Optional year filter.
    threshold : int
        Minimum number of duplicate rows.

    Returns
    -------
    list of tuple
        ``(conference, year, rows, duplicates)`` sorted by duplicates descending.
    """
    conditions = ["conference IS NOT NULL", "year IS NOT NULL"]
    params: Dict[str, Any] = {}
    if conference is not None:
        conditions.append("conference = :conference")
        params["conference"] = conference
    if year is not None:
        conditions.append("year = :year")
        params["year"] = year
    where = " AND ".join(conditions)
    rows = db.session.execute(
        text(
            "SELECT conference, year, COUNT(*) AS rows, "
            "COUNT(DISTINCT coalesce(content_hash, uid)) AS unique_rows "
            f"FROM papers WHERE {where} GROUP BY conference, year"
        ),
        params,
    ).all()

    scopes: List[Tuple[str, int, int, int]] = []
    for row_conference, row_year, row_count, unique_count in rows:
        duplicates = int(row_count) - int(unique_count)
        if duplicates >= threshold:
            scopes.append((str(row_conference), int(row_year), int(row_count), duplicates))
    scopes.sort(key=lambda item: item[3], reverse=True)
    return scopes


def main() -> int:
    """
    Run the hybrid regeneration report or execution.

    Returns
    -------
    int
        Exit code (0 on success, 1 when any scope failed).
    """
    args = parse_args()
    config = get_config()
    collection_name = args.collection or config.collection_name

    with DatabaseManager() as db:
        if args.paper_db:
            db.database_url = args.paper_db
        db.connect()
        scopes = high_duplicate_scopes(db, args.conference, args.year, args.threshold)

    if args.max_scopes is not None:
        scopes = scopes[: args.max_scopes]

    print(f"Duplicate threshold: {args.threshold:,} extra row(s)")
    print(f"Scopes selected: {len(scopes):,}")
    for scope_conference, scope_year, row_count, duplicates in scopes:
        print(f"  {scope_conference} {scope_year}: {row_count:,} rows, " f"{duplicates:,} duplicate(s)")

    if not scopes:
        print("Nothing to regenerate.")
        return 0

    print("")
    print("Commands:")
    for scope_conference, scope_year, _, _ in scopes:
        print(
            f"  abstracts-explorer clustering pre-generate --conference {scope_conference} --year {scope_year} --force"
        )

    if not args.execute:
        print("")
        print("Dry run only. Re-run with --execute to regenerate the selected scopes.")
        return 0

    failures = 0
    with EmbeddingsManager(collection_name=collection_name) as embeddings_manager:
        for scope_conference, scope_year, _, duplicates in tqdm(scopes, desc="Regenerating scopes", unit="scope"):
            print("")
            print(f"Regenerating {scope_conference} {scope_year} ({duplicates:,} duplicates removed)...")
            try:
                with DatabaseManager() as db:
                    if args.paper_db:
                        db.database_url = args.paper_db
                    db.connect()
                    results = compute_clusters_with_cache(
                        embeddings_manager=embeddings_manager,
                        database=db,
                        embedding_model=config.embedding_model,
                        force=True,
                        conferences=[scope_conference],
                        years=[scope_year],
                        use_llm_labels=not args.no_llm_labels,
                        show_progress=args.show_progress,
                        backfill_missing_labels=True,
                    )
                statistics = results.get("statistics", {})
                print(
                    f"  Done: {statistics.get('total_papers', 0):,} papers, "
                    f"{statistics.get('n_clusters', 0):,} clusters"
                )
            except Exception as exc:
                failures += 1
                print(f"  ERROR: {exc}", file=sys.stderr)

    print("")
    print(f"Regeneration complete: {len(scopes) - failures:,} succeeded, {failures:,} failed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

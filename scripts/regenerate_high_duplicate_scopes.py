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

Completed scopes are recorded in a resume state file, so an interrupted run can
be restarted without recomputing scopes that already finished.  Pass
``--no-resume`` to ignore the state file and regenerate everything.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from sqlalchemy import text
from tqdm import tqdm

from abstracts_explorer.clustering.pipeline import compute_clusters_with_cache
from abstracts_explorer.config import get_config
from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.embeddings import EmbeddingsManager

STATE_FILENAME = "regenerate_high_duplicate_scopes.state.json"


def _scope_key(conference: str, year: int) -> str:
    """Return the resume-state key for a scope."""
    return f"{conference}::{year}"


def _default_state_path(config: Any) -> Path:
    """Return the default resume state-file path inside the data directory."""
    return Path(config.data_dir) / STATE_FILENAME


def load_state(path: Path) -> Dict[str, Any]:
    """
    Load resume state from *path*.

    Parameters
    ----------
    path : Path
        State-file location.

    Returns
    -------
    dict
        Mapping with ``meta`` and ``completed`` keys.  An empty mapping is
        returned when the file is missing, unreadable, or malformed.
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    completed = data.get("completed")
    if not isinstance(completed, list):
        completed = []
    meta = data.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    return {"meta": meta, "completed": [str(item) for item in completed]}


def save_state(path: Path, state: Dict[str, Any]) -> None:
    """
    Atomically write resume state to *path*.

    Parameters
    ----------
    path : Path
        State-file location.
    state : dict
        State mapping to persist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def _state_meta(
    database_url: str,
    collection_name: str,
    embedding_model: str,
    threshold: int,
) -> Dict[str, Any]:
    """Return metadata used to validate a resume state file against a run."""
    return {
        "database_url": database_url,
        "collection": collection_name,
        "embedding_model": embedding_model,
        "threshold": threshold,
    }


def cleanup_scope_cache(
    db: DatabaseManager,
    conference: str,
    year: int,
    embedding_model: str,
    collection_name: str,
) -> int:
    """
    Keep only the newest cache row for a scope and delete older duplicates.

    Parameters
    ----------
    db : DatabaseManager
        Connected database manager.
    conference : str
        Conference of the scope.
    year : int
        Year of the scope.
    embedding_model : str
        Embedding model used by the cache rows.
    collection_name : str
        ChromaDB collection encoded in the cache key.

    Returns
    -------
    int
        Number of stale cache rows deleted.
    """
    rows = db.session.execute(
        text(
            "SELECT id, clustering_params FROM clustering_cache "
            "WHERE conference = :conference AND year = :year AND embedding_model = :embedding_model "
            "ORDER BY created_at DESC, id DESC"
        ),
        {"conference": conference, "year": year, "embedding_model": embedding_model},
    ).all()
    stale_ids: List[int] = []
    kept_newest = False
    for cache_id, params_json in rows:
        try:
            params = json.loads(params_json) if params_json else {}
        except ValueError:
            params = {}
        if params.get("__collection__") != collection_name:
            continue
        if not kept_newest:
            kept_newest = True
            continue
        stale_ids.append(int(cache_id))
    if not stale_ids:
        return 0
    return db.delete_clustering_cache_ids(stale_ids)


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
    parser.add_argument(
        "--state-file",
        help=("Resume state file (default: " f"<DATA_DIR>/{STATE_FILENAME})."),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore completed scopes in the state file and regenerate everything.",
    )
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
    embedding_model = config.embedding_model
    database_url = args.paper_db or config.database_url
    state_path = Path(args.state_file) if args.state_file else _default_state_path(config)
    meta = _state_meta(database_url, collection_name, embedding_model, args.threshold)

    state = load_state(state_path)
    completed: Set[str] = set()
    if not args.no_resume and state.get("meta") == meta:
        completed = set(state.get("completed", []))

    with DatabaseManager() as db:
        if args.paper_db:
            db.database_url = args.paper_db
        db.connect()
        scopes = high_duplicate_scopes(db, args.conference, args.year, args.threshold)

    skipped = 0
    if completed:
        remaining = [scope for scope in scopes if _scope_key(scope[0], scope[1]) not in completed]
        skipped = len(scopes) - len(remaining)
        scopes = remaining

    if args.max_scopes is not None:
        scopes = scopes[: args.max_scopes]

    print(f"Duplicate threshold: {args.threshold:,} extra row(s)")
    if skipped:
        print(f"Skipped {skipped:,} already-regenerated scope(s) from {state_path}")
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
                        embedding_model=embedding_model,
                        force=True,
                        conferences=[scope_conference],
                        years=[scope_year],
                        use_llm_labels=not args.no_llm_labels,
                        show_progress=args.show_progress,
                        backfill_missing_labels=True,
                    )
                    cleanup_scope_cache(db, scope_conference, scope_year, embedding_model, collection_name)
                statistics = results.get("statistics", {})
                completed.add(_scope_key(scope_conference, scope_year))
                save_state(state_path, {"meta": meta, "completed": sorted(completed)})
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

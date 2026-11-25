#!/usr/bin/env python3
"""Remove duplicate conference data left behind by a rename.

The ML4PS plugin originally stored papers under ``ML4PS@Neurips``.  After the
canonical name changed to ``ML4PS``, stale per-year JSON caches could cause the
old rows to be reinserted alongside the canonical ones.  This script removes
the resulting duplicates from the paper database, ChromaDB, clustering caches,
and the per-year JSON caches.

Papers without a canonical counterpart are migrated instead of deleted, so the
script is safe to run when only part of the data has already been renamed.

Examples
--------
>>> uv run python scripts/cleanup_ml4ps_duplicates.py --dry-run
>>> uv run python scripts/cleanup_ml4ps_duplicates.py --apply
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast

from sqlalchemy import delete, func, select, update

from abstracts_explorer.database import DatabaseError, DatabaseManager
from abstracts_explorer.db_models import ClusteringCache, Paper, ValidationData

_BATCH_SIZE = 400


@dataclass
class MigrationPlan:
    """Mapping of legacy UIDs to canonical UIDs and rows that can be removed."""

    mapping: Dict[str, str] = field(default_factory=dict)
    duplicates: List[str] = field(default_factory=list)
    renames: List[str] = field(default_factory=list)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Parameters
    ----------
    argv : list of str, optional
        Arguments to parse. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Remove duplicate legacy conference data.")
    parser.add_argument("--old-name", default="ML4PS@Neurips", help="Legacy conference name (case-insensitive).")
    parser.add_argument("--new-name", default="ML4PS", help="Canonical conference name.")
    parser.add_argument("--paper-db", help="Override PAPER_DB for this run.")
    parser.add_argument("--data-dir", default="data", help="Directory containing per-year JSON caches.")
    parser.add_argument("--data-pattern", default="ml4ps_*.json", help="Glob for per-year JSON cache files.")
    parser.add_argument("--skip-embeddings", action="store_true", help="Do not update the ChromaDB collection.")
    parser.add_argument("--skip-json", action="store_true", help="Do not rewrite per-year JSON cache files.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Write changes.")
    mode.add_argument("--dry-run", action="store_true", help="Report changes without writing anything (default).")
    return parser.parse_args(argv)


def _batched(values: Iterable[str], size: int = _BATCH_SIZE) -> Iterable[List[str]]:
    """Yield *values* in batches of at most *size*."""
    batch: List[str] = []
    for value in values:
        batch.append(value)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_plan(papers: Sequence[Paper], existing_uids: set[str], new_name: str) -> MigrationPlan:
    """
    Build the legacy-to-canonical UID mapping for *papers*.

    Parameters
    ----------
    papers : list of Paper
        Papers stored under the legacy conference name.
    existing_uids : set of str
        Every UID currently present in the database.
    new_name : str
        Canonical conference name.

    Returns
    -------
    MigrationPlan
        Mapping plus the subsets that are duplicates or plain renames.
    """
    plan = MigrationPlan()
    for paper in papers:
        old_uid = cast(str, paper.uid)
        title = cast(str, paper.title)
        original_id = cast(Optional[str], paper.original_id)
        year = cast(int, paper.year)
        new_uid = DatabaseManager.compute_uid(title, original_id, new_name, year)
        plan.mapping[old_uid] = new_uid
        if new_uid != old_uid and new_uid in existing_uids:
            plan.duplicates.append(old_uid)
        else:
            plan.renames.append(old_uid)
    return plan


def _delete_papers(session: Any, uids: List[str]) -> int:
    """Delete paper rows in batches and return the number deleted."""
    deleted = 0
    for batch in _batched(uids):
        result = session.execute(delete(Paper).where(Paper.uid.in_(batch)))
        deleted += getattr(result, "rowcount", 0) or 0
    return deleted


def _migrate_database(
    db: DatabaseManager,
    old_name: str,
    new_name: str,
    dry_run: bool,
) -> MigrationPlan:
    """
    Remove duplicate legacy papers and migrate the remaining ones.

    Parameters
    ----------
    db : DatabaseManager
        Open database manager.
    old_name : str
        Legacy conference name.
    new_name : str
        Canonical conference name.
    dry_run : bool
        When ``True``, report the plan without writing changes.

    Returns
    -------
    MigrationPlan
        The computed migration plan.
    """
    session = db.session
    papers = session.execute(select(Paper).where(func.lower(Paper.conference) == old_name.lower())).scalars().all()
    if not papers:
        print(f"No papers found for conference {old_name!r}.")
        return MigrationPlan()

    existing_uids = set(session.execute(select(Paper.uid)).scalars().all())
    plan = build_plan(papers, existing_uids, new_name)

    print(f"Found {len(papers)} paper(s) for {old_name!r}:")
    print(f"  {len(plan.duplicates)} duplicate(s) with an existing {new_name!r} counterpart")
    print(f"  {len(plan.renames)} paper(s) to rename to {new_name!r}")

    if dry_run:
        return plan

    for old_uid in plan.renames:
        session.execute(
            update(Paper).where(Paper.uid == old_uid).values(uid=plan.mapping[old_uid], conference=new_name)
        )
    deleted = _delete_papers(session, plan.duplicates)
    session.flush()

    validation_count = 0
    for old_uid, new_uid in plan.mapping.items():
        result = session.execute(
            update(ValidationData).where(ValidationData.paper_uid == old_uid).values(paper_uid=new_uid)
        )
        validation_count += getattr(result, "rowcount", 0) or 0

    cache_deleted, cache_renamed = _migrate_clustering_cache(session, old_name, new_name, plan.mapping)
    session.commit()

    print(f"Deleted {deleted} duplicate paper row(s).")
    print(f"Renamed {len(plan.renames)} paper row(s).")
    print(f"Updated {validation_count} validation_data row(s).")
    print(f"Deleted {cache_deleted} duplicate clustering cache entry(ies).")
    print(f"Renamed {cache_renamed} clustering cache entry(ies).")
    return plan


def _cache_key(entry: ClusteringCache) -> Tuple[Any, ...]:
    """Return the fields that identify an equivalent clustering cache entry.

    ``n_clusters`` is intentionally excluded: re-clustering the same papers under
    the legacy UIDs can produce a slightly different cluster count, but the
    canonical cache still supersedes the legacy one.
    """
    return (
        entry.embedding_model,
        entry.year,
        entry.reduction_method,
        entry.n_components,
        entry.clustering_method,
        entry.clustering_params,
    )


def _replace_uids(value: Any, mapping: Dict[str, str]) -> Any:
    """Recursively replace legacy UIDs in a JSON-compatible structure."""
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [_replace_uids(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: _replace_uids(item, mapping) for key, item in value.items()}
    return value


def _migrate_clustering_cache(
    session: Any,
    old_name: str,
    new_name: str,
    mapping: Dict[str, str],
) -> Tuple[int, int]:
    """
    Delete or rename clustering cache entries for the legacy conference.

    Parameters
    ----------
    session : Any
        Active SQLAlchemy session.
    old_name : str
        Legacy conference name.
    new_name : str
        Canonical conference name.
    mapping : dict
        Legacy-to-canonical UID mapping used to rewrite result JSON.

    Returns
    -------
    tuple of int
        Number of deleted and renamed cache entries.
    """
    old_entries = (
        session.execute(select(ClusteringCache).where(func.lower(ClusteringCache.conference) == old_name.lower()))
        .scalars()
        .all()
    )
    canonical_keys = {
        _cache_key(entry)
        for entry in session.execute(select(ClusteringCache).where(ClusteringCache.conference == new_name))
        .scalars()
        .all()
    }

    deleted = 0
    renamed = 0
    for entry in old_entries:
        if _cache_key(entry) in canonical_keys:
            session.delete(entry)
            deleted += 1
            continue
        entry.conference = new_name
        try:
            results = json.loads(entry.results_json)
        except (TypeError, ValueError):
            results = None
        if results is not None:
            migrated = _replace_uids(results, mapping)
            if migrated != results:
                entry.results_json = json.dumps(migrated)
        renamed += 1
    return deleted, renamed


def _chroma_contains(collection: Any, uid: str) -> bool:
    """Return whether *uid* already exists in the ChromaDB collection."""
    return bool(collection.get(ids=[uid], include=[]).get("ids"))


def _migrate_embeddings(mapping: Dict[str, str], new_name: str, dry_run: bool) -> None:
    """
    Delete legacy embeddings, moving them when no canonical copy exists.

    Parameters
    ----------
    mapping : dict
        Legacy-to-canonical UID mapping.
    new_name : str
        Canonical conference name.
    dry_run : bool
        When ``True``, only report how many embeddings would be processed.
    """
    if not mapping:
        return
    if dry_run:
        print(f"Would process {len(mapping)} embedding(s) in ChromaDB.")
        return

    from abstracts_explorer.embeddings import EmbeddingsManager

    em = EmbeddingsManager()
    migrated = 0
    deleted = 0
    missing = 0
    try:
        for old_uid, new_uid in mapping.items():
            result = em.collection.get(
                ids=[old_uid],
                include=["documents", "embeddings", "metadatas"],
            )
            if not result.get("ids"):
                missing += 1
                continue

            if _chroma_contains(em.collection, new_uid):
                em.delete_embeddings_by_ids([old_uid])
                deleted += 1
                continue

            metadata = dict((result.get("metadatas") or [{}])[0])
            metadata["conference"] = new_name
            add_kwargs: Dict[str, Any] = {
                "ids": [new_uid],
                "documents": result.get("documents") or [""],
                "metadatas": [metadata],
            }
            embeddings = result.get("embeddings")
            if embeddings is not None:
                add_kwargs["embeddings"] = embeddings
            em.collection.add(**add_kwargs)
            em.delete_embeddings_by_ids([old_uid])
            migrated += 1
    finally:
        em.close()

    print(f"Deleted {deleted} duplicate embedding(s).")
    print(f"Migrated {migrated} embedding(s).")
    print(f"{missing} legacy embedding(s) were not present in ChromaDB.")


def normalize_json_caches(
    data_dir: Path,
    pattern: str,
    old_name: str,
    new_name: str,
    dry_run: bool,
) -> int:
    """
    Rewrite the conference label in matching per-year JSON caches.

    Parameters
    ----------
    data_dir : Path
        Directory containing the cache files.
    pattern : str
        Glob pattern selecting the caches.
    old_name : str
        Legacy conference label.
    new_name : str
        Canonical conference label.
    dry_run : bool
        When ``True``, report changes without writing files.

    Returns
    -------
    int
        Number of files that contain the legacy label.
    """
    changed = 0
    for path in sorted(data_dir.glob(pattern)):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Skipping {path}: {exc}", file=sys.stderr)
            continue
        if not isinstance(data, list):
            continue
        touched = False
        for paper in data:
            if isinstance(paper, dict) and paper.get("conference") == old_name:
                paper["conference"] = new_name
                touched = True
        if touched:
            changed += 1
            print(f"{'Would update' if dry_run else 'Updated'} {path}")
            if not dry_run:
                path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return changed


def main(argv: Optional[List[str]] = None) -> int:
    """
    Run the duplicate cleanup.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    dry_run = not args.apply

    if args.old_name.lower() == args.new_name.lower():
        print("Old and new conference names are identical; nothing to do.")
        return 0

    if args.paper_db:
        os.environ["PAPER_DB"] = args.paper_db

    try:
        with DatabaseManager() as db:
            plan = _migrate_database(db, args.old_name, args.new_name, dry_run)
    except DatabaseError as exc:
        print(f"Database migration failed: {exc}", file=sys.stderr)
        return 1

    if not args.skip_embeddings:
        try:
            _migrate_embeddings(plan.mapping, args.new_name, dry_run)
        except Exception as exc:
            print(f"Embedding migration failed: {exc}", file=sys.stderr)
            print(
                "The SQL database was already migrated. Re-run with --skip-embeddings "
                "after fixing the embeddings problem, or recreate embeddings.",
                file=sys.stderr,
            )
            return 1

    if not args.skip_json:
        changed = normalize_json_caches(
            Path(args.data_dir),
            args.data_pattern,
            args.old_name,
            args.new_name,
            dry_run,
        )
        if changed == 0:
            print("No per-year JSON cache files needed updates.")

    if dry_run:
        print("Dry run complete; no changes were written. Re-run with --apply to apply them.")
    else:
        print(f"Cleanup complete. Legacy {args.old_name!r} data was removed or migrated to {args.new_name!r}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

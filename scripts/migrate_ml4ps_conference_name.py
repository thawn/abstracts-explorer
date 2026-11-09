#!/usr/bin/env python3
"""Rename the ML4PS conference and migrate all related data.

The ML4PS plugin used to store papers under ``ML4PS@Neurips``.  This script
renames those papers to ``ML4PS``, recomputes their UIDs (the UID is derived
from the conference name), and updates every known reference:

- ``papers.uid`` / ``papers.conference``
- ``validation_data.paper_uid``
- ``clustering_cache.conference`` and paper IDs in ``results_json``
- ChromaDB embedding IDs and metadata (unless ``--skip-embeddings`` is given)

Registry artifacts are **not** modified.  Tags change from ``ml4ps-neurips-*``
to ``ml4ps-*``; re-upload or delete old artifacts after running this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

from sqlalchemy import func, select, update

from abstracts_explorer.database import DatabaseError, DatabaseManager
from abstracts_explorer.db_models import ClusteringCache, Paper, ValidationData


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Rename ML4PS conference data.")
    parser.add_argument("--old-name", default="ML4PS@Neurips", help="Old conference name (case-insensitive).")
    parser.add_argument("--new-name", default="ML4PS", help="New conference name.")
    parser.add_argument("--paper-db", help="Override PAPER_DB for this run.")
    parser.add_argument("--skip-embeddings", action="store_true", help="Do not update the ChromaDB collection.")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing anything.")
    return parser.parse_args()


def _replace_uids(value: Any, mapping: Dict[str, str]) -> Any:
    """Recursively replace old UIDs in a JSON-compatible structure."""
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [_replace_uids(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: _replace_uids(item, mapping) for key, item in value.items()}
    return value


def _migrate_database(
    db: DatabaseManager,
    old_name: str,
    new_name: str,
    dry_run: bool,
) -> Dict[str, str]:
    """Migrate papers and related SQL rows.  Returns the old->new UID mapping."""
    session = db.session

    papers = session.execute(select(Paper).where(func.lower(Paper.conference) == old_name.lower())).scalars().all()

    if not papers:
        print(f"No papers found for conference {old_name!r}.")
        return {}

    existing_uids = set(session.execute(select(Paper.uid)).scalars().all())
    mapping: Dict[str, str] = {}
    skipped: List[str] = []

    for paper in papers:
        new_uid = DatabaseManager.compute_uid(paper.title, paper.original_id, new_name, paper.year)
        if new_uid != paper.uid and new_uid in existing_uids:
            skipped.append(paper.uid)
            print(f"⚠️  Skipping UID collision: {paper.uid} -> {new_uid} (target already exists)")
            continue
        mapping[paper.uid] = new_uid

    print(f"Found {len(papers)} paper(s) for {old_name!r}; {len(mapping)} can be migrated.")
    if skipped:
        print(f"Skipped {len(skipped)} paper(s) due to UID collisions.")

    if dry_run:
        return mapping

    for old_uid, new_uid in mapping.items():
        session.execute(
            update(Paper)
            .where(Paper.uid == old_uid)
            .values(uid=new_uid, conference=new_name)
        )

    session.flush()

    validation_count = 0
    for old_uid, new_uid in mapping.items():
        result = session.execute(
            update(ValidationData).where(ValidationData.paper_uid == old_uid).values(paper_uid=new_uid)
        )
        validation_count += result.rowcount or 0

    cache_entries = (
        session.execute(select(ClusteringCache).where(func.lower(ClusteringCache.conference) == old_name.lower()))
        .scalars()
        .all()
    )

    cache_uid_count = 0
    for entry in cache_entries:
        entry.conference = new_name
        try:
            results = json.loads(entry.results_json)
            migrated = _replace_uids(results, mapping)
            if migrated != results:
                entry.results_json = json.dumps(migrated)
                cache_uid_count += 1
        except (TypeError, ValueError):
            pass

    session.commit()

    print(f"Updated {len(mapping)} paper row(s).")
    print(f"Updated {validation_count} validation_data row(s).")
    print(f"Updated {len(cache_entries)} clustering cache entry(ies); {cache_uid_count} contained paper IDs.")
    return mapping


def _migrate_embeddings(mapping: Dict[str, str], new_name: str, dry_run: bool) -> None:
    """Update ChromaDB embedding IDs and metadata."""
    if not mapping:
        return
    if dry_run:
        print(f"Would migrate {len(mapping)} embedding(s) in ChromaDB.")
        return

    from abstracts_explorer.embeddings import EmbeddingsManager

    em = EmbeddingsManager()
    try:
        migrated = 0
        missing = 0
        for old_uid, new_uid in mapping.items():
            result = em.collection.get(
                ids=[old_uid],
                include=["documents", "embeddings", "metadatas"],
            )
            if not result.get("ids"):
                missing += 1
                continue

            documents = result.get("documents") or [""]
            embeddings = result.get("embeddings")
            metadatas = result.get("metadatas") or [{}]
            metadata = dict(metadatas[0])
            metadata["conference"] = new_name

            add_kwargs: Dict[str, Any] = {
                "ids": [new_uid],
                "documents": documents,
                "metadatas": [metadata],
            }
            if embeddings is not None:
                add_kwargs["embeddings"] = embeddings
            em.collection.add(**add_kwargs)
            em.collection.delete(ids=[old_uid])
            migrated += 1

        print(f"Migrated {migrated} embedding(s); {missing} were not present in ChromaDB.")
    finally:
        em.close()


def main() -> int:
    """Run the migration."""
    args = parse_args()

    if args.old_name.lower() == args.new_name.lower():
        print("Old and new conference names are identical; nothing to do.")
        return 0

    if args.paper_db:
        import os

        os.environ["PAPER_DB"] = args.paper_db

    try:
        with DatabaseManager() as db:
            mapping = _migrate_database(db, args.old_name, args.new_name, args.dry_run)
    except DatabaseError as exc:
        print(f"Database migration failed: {exc}", file=sys.stderr)
        return 1

    if not args.skip_embeddings:
        try:
            _migrate_embeddings(mapping, args.new_name, args.dry_run)
        except Exception as exc:
            print(f"Embedding migration failed: {exc}", file=sys.stderr)
            print(
                "The SQL database was already migrated. Re-run with --skip-embeddings "
                "after fixing the embeddings problem, or recreate embeddings.",
                file=sys.stderr,
            )
            return 1

    if args.dry_run:
        print("Dry run complete; no changes were written.")
    else:
        print("Migration complete. Remember to re-upload/delete registry artifacts using ml4ps-neurips-* tags.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

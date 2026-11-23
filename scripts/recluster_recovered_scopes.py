#!/usr/bin/env python3
"""Regenerate clustering caches for scopes whose embeddings were rebuilt.

After restoring a ChromaDB snapshot and re-embedding the missing papers, the
clustering cache for every scope whose vectors changed is stale: the cache key
contains only the embedding model name, which does not change during recovery.
This script deletes and recomputes the cached clustering results for the known
affected scopes.

Affected scopes (from the 2026-11-23 embedding recovery):

- AGU 2016-2021
- ICML 2013-2019
- NeurIPS 1987-2003 and 2019

Run this only after ``scripts/migrate_embeddings_compaction.py`` and
``abstracts-explorer create-embeddings`` have completed. Run it as a single
process; the local ChromaDB lock rejects parallel runs.

Usage::

    python scripts/recluster_recovered_scopes.py --dry-run
    python scripts/recluster_recovered_scopes.py
    python scripts/recluster_recovered_scopes.py --no-llm-labels
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Sequence, Tuple

from abstracts_explorer.clustering import compute_clusters_with_cache
from abstracts_explorer.config import get_config
from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.embeddings import EmbeddingsManager

AFFECTED_SCOPES: Tuple[Tuple[str, int], ...] = (
    tuple(("AGU", year) for year in range(2016, 2022))
    + tuple(("ICML", year) for year in range(2013, 2020))
    + tuple(("NeurIPS", year) for year in (*range(1987, 2004), 2019))
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments to parse. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Delete and regenerate clustering caches for the conference/year "
            "scopes affected by the embedding recovery."
        )
    )
    parser.add_argument(
        "--paper-db",
        help="Override PAPER_DB for this command only. Accepts a SQLite path or full database URL.",
    )
    parser.add_argument(
        "--collection",
        help="ChromaDB collection name (defaults to COLLECTION_NAME from config).",
    )
    parser.add_argument(
        "--no-llm-labels",
        action="store_true",
        help="Use keyword labels instead of calling the LLM for cluster labels.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the affected scopes without deleting or recomputing anything.",
    )
    return parser.parse_args(argv)


def recluster_scope(
    embeddings_manager: EmbeddingsManager,
    database: DatabaseManager,
    embedding_model: str,
    conference: str,
    year: int,
    *,
    use_llm_labels: bool = True,
) -> dict:
    """
    Delete and recompute the clustering cache for one conference/year scope.

    Parameters
    ----------
    embeddings_manager : EmbeddingsManager
        Connected embeddings manager.
    database : DatabaseManager
        Connected database manager.
    embedding_model : str
        Current embedding model name used as part of the cache key.
    conference : str
        Conference name to regenerate.
    year : int
        Conference year to regenerate.
    use_llm_labels : bool, optional
        Whether to generate cluster labels with the LLM (default True).

    Returns
    -------
    dict
        New clustering results for the scope.
    """
    deleted = database.delete_clustering_cache_by_conference_year(conference, year)
    if deleted:
        print(f"   Deleted {deleted} stale cache row(s)")
    return compute_clusters_with_cache(
        embeddings_manager=embeddings_manager,
        database=database,
        embedding_model=embedding_model,
        limit=None,
        force=True,
        conferences=[conference],
        years=[year],
        use_llm_labels=use_llm_labels,
        show_progress=True,
        backfill_missing_labels=True,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Regenerate the clustering cache for all affected scopes.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code (0 for success, non-zero if any scope failed).
    """
    args = parse_args(argv)

    if args.paper_db:
        os.environ["PAPER_DB"] = args.paper_db
    config = get_config(reload=True)

    collection_name: Optional[str] = args.collection or config.collection_name
    if not collection_name:
        print("No collection name configured; pass --collection.", file=sys.stderr)
        return 1

    scopes: List[Tuple[str, int]] = list(AFFECTED_SCOPES)

    if args.dry_run:
        print(f"Would regenerate {len(scopes)} clustering cache scope(s):")
        for conference, year in scopes:
            print(f"   {conference} {year}")
        return 0

    failures: List[Tuple[str, int, str]] = []
    succeeded = 0

    with DatabaseManager() as database, EmbeddingsManager(collection_name=collection_name) as embeddings_manager:
        for conference, year in scopes:
            print(f"\n🔄 Re-clustering {conference} {year}...")
            try:
                results = recluster_scope(
                    embeddings_manager,
                    database,
                    config.embedding_model,
                    conference,
                    year,
                    use_llm_labels=not args.no_llm_labels,
                )
                stats = results.get("statistics", {})
                n_papers = stats.get("total_papers", "N/A")
                n_clusters = stats.get("n_clusters", "N/A")
                print(f"   ✅ {conference} {year}: {n_papers} papers → {n_clusters} clusters")
                succeeded += 1
            except Exception as exc:
                print(f"   ❌ {conference} {year}: {exc}", file=sys.stderr)
                failures.append((conference, year, str(exc)))

    print(f"\n{'=' * 70}")
    print(f"✅ Re-clustering complete: {succeeded} succeeded, {len(failures)} failed")
    for conference, year, message in failures:
        print(f"   ❌ {conference} {year}: {message}", file=sys.stderr)

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())

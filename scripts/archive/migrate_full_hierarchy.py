#!/usr/bin/env python3
"""Migrate clustering caches to a full hierarchy tree.

This one-off migration reconstructs the complete BERTopic merge tree from the
cached points and paper metadata.  It reuses every LLM-generated hierarchy
label/description that can be recovered from the existing hierarchy levels and
stores the result as ``hierarchy.tree``.  Missing descriptions are left blank
so the normal backfill can repair them later.

Usage::

    python scripts/migrate_full_hierarchy.py --dry-run
    python scripts/migrate_full_hierarchy.py
    python scripts/migrate_full_hierarchy.py --conference AAAI --year 2026
    python scripts/migrate_full_hierarchy.py --prune-levels
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity

from abstracts_explorer.clustering.hierarchy_levels import (
    level_parent_descriptions,
    level_parent_labels,
)
from abstracts_explorer.clustering.results import build_hierarchy_tree
from abstracts_explorer.config import get_config
from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.embeddings import EmbeddingsManager


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Reconstruct full clustering hierarchy trees in the cache.")
    parser.add_argument("--paper-db", help="Override PAPER_DB for this command only.")
    parser.add_argument("--conference", help="Only migrate cache entries for this conference.")
    parser.add_argument("--year", type=int, help="Only migrate cache entries for this year.")
    parser.add_argument("--collection", help="Only migrate cache entries for this ChromaDB collection.")
    parser.add_argument("--embedding-model", help="Only migrate cache entries for this embedding model.")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing to the database.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild rows that already contain a tree.")
    parser.add_argument(
        "--prune-levels", action="store_true", help="Remove legacy hierarchy levels after building a tree."
    )
    return parser.parse_args()


def _has_text(value: Any) -> bool:
    """Return whether a value contains non-whitespace text."""
    return bool(str(value or "").strip())


def _member_set(node: Mapping[str, Any]) -> frozenset[int]:
    """Return the positive flat-topic members for a hierarchy node."""
    return frozenset(int(topic) for topic in node.get("topics") or [] if int(topic) >= 0)


def _recover_tree_labels(results: Mapping[str, Any]) -> Dict[frozenset[int], Tuple[str, str]]:
    """Return recoverable labels/descriptions from an existing tree."""
    recovered: Dict[frozenset[int], Tuple[str, str]] = {}
    hierarchy = results.get("hierarchy") or {}
    for node in hierarchy.get("tree") or []:
        members = _member_set(node)
        label = str(node.get("label") or "")
        description = str(node.get("description") or "")
        if len(members) <= 1 or (not _has_text(label) and not _has_text(description)):
            continue
        recovered[members] = (label, description)
    return recovered


def _recover_level_labels(
    results: Mapping[str, Any],
    warnings: List[str],
) -> Dict[frozenset[int], Tuple[str, str]]:
    """Return recoverable labels/descriptions from legacy hierarchy levels."""
    recovered: Dict[frozenset[int], Tuple[str, str]] = {}
    hierarchy = results.get("hierarchy") or {}
    for level in hierarchy.get("levels") or []:
        members_by_cluster: Dict[int, List[int]] = defaultdict(list)
        for flat_id, cluster_id in (level.get("mapping") or {}).items():
            if int(flat_id) < 0:
                continue
            members_by_cluster[int(cluster_id)].append(int(flat_id))
        for cluster in level.get("clusters") or []:
            cluster_id = int(cluster.get("id", -1))
            members = frozenset(members_by_cluster.get(cluster_id, {cluster_id}))
            label = str(cluster.get("label") or "")
            description = str(cluster.get("description") or "")
            if len(members) <= 1 or (not _has_text(label) and not _has_text(description)):
                continue
            candidate = (label, description)
            previous = recovered.get(members)
            if previous is not None and previous != candidate:
                warnings.append(f"conflicting labels for members {sorted(members)}; keeping first")
                continue
            recovered.setdefault(members, candidate)
    return recovered


def _recover_existing_labels(
    results: Mapping[str, Any],
    warnings: List[str],
) -> Dict[frozenset[int], Tuple[str, str]]:
    """Recover labels from the existing tree and legacy levels."""
    recovered = _recover_level_labels(results, warnings)
    for members, value in _recover_tree_labels(results).items():
        existing = recovered.get(members)
        if _has_text(value[1]) or existing is None:
            recovered[members] = value
    return recovered


def _load_documents(
    results: Mapping[str, Any],
    database: DatabaseManager,
) -> Tuple[List[str], List[int]]:
    """Rebuild documents and cached topic assignments for a clustering result."""
    points = list(results.get("points") or [])
    if not points:
        raise ValueError("cached results contain no points")

    paper_ids = [str(point["id"]) for point in points if point.get("id") is not None]
    if len(paper_ids) != len(points):
        raise ValueError("a cached point has no paper ID")

    papers = database.get_paper_summaries_by_uids(paper_ids)
    papers_by_uid = {str(paper.get("uid")): paper for paper in papers}
    missing = [paper_id for paper_id in paper_ids if paper_id not in papers_by_uid]
    if missing:
        raise ValueError(f"missing paper metadata for {len(missing)} cached point(s)")

    documents = [EmbeddingsManager.embedding_text_from_paper(papers_by_uid[paper_id]) for paper_id in paper_ids]
    topics = [int(point.get("cluster", -1)) for point in points]
    return documents, topics


def _distance_function(matrix: Any) -> Any:
    """Return the distance function used during clustering."""
    return np.clip(1 - cosine_similarity(matrix), 0, None)


def _rebuild_tree(
    results: Mapping[str, Any],
    database: DatabaseManager,
    recovered: Mapping[frozenset[int], Tuple[str, str]],
    warnings: List[str],
) -> Tuple[List[Dict[str, Any]], int]:
    """Rebuild the full hierarchy tree and overlay recovered labels."""
    documents, topics = _load_documents(results, database)
    topic_model = BERTopic(embedding_model=None, representation_model=None, verbose=False)
    topic_model.topics_ = list(topics)
    topic_model.update_topics(documents, topics=list(topics), representation_model=None)
    hierarchical_topics = topic_model.hierarchical_topics(documents, distance_function=_distance_function)

    tree = build_hierarchy_tree(
        hierarchical_topics,
        level_parent_labels(hierarchical_topics),
        level_parent_descriptions(hierarchical_topics),
    )

    nodes_by_members = {_member_set(node): node for node in tree}
    recovered_count = 0
    for members, (label, description) in recovered.items():
        node = nodes_by_members.get(members)
        if node is None:
            warnings.append(f"recovered members {sorted(members)} not found in rebuilt tree")
            continue
        node["label"] = label or str(node.get("label") or "")
        node["description"] = description
        recovered_count += 1
    return tree, recovered_count


def _store_tree(
    results: Dict[str, Any],
    tree: Sequence[Mapping[str, Any]],
    prune_levels: bool,
) -> None:
    """Store the tree in the result payload, optionally pruning legacy levels."""
    hierarchy = results.setdefault("hierarchy", {})
    statistics = results.get("statistics") or {}
    hierarchy["method"] = "bertopic"
    hierarchy["n_topics"] = int(statistics.get("n_clusters", 0))
    hierarchy["tree"] = list(tree)
    if prune_levels:
        hierarchy.pop("levels", None)
        hierarchy.pop("nr_levels", None)
        hierarchy.pop("level_scale", None)


def _cache_has_tree(results: Mapping[str, Any]) -> bool:
    """Return whether cached results already contain a hierarchy tree."""
    return bool((results.get("hierarchy") or {}).get("tree"))


def _cache_has_legacy_levels(results: Mapping[str, Any]) -> bool:
    """Return whether cached results contain legacy hierarchy levels."""
    return bool((results.get("hierarchy") or {}).get("levels"))


def _collection_matches(entry: Mapping[str, Any], collection: Optional[str]) -> bool:
    """Return whether a cache entry belongs to the requested collection."""
    if collection is None:
        return True
    params = entry.get("clustering_params") or {}
    return params.get("__collection__") == collection


def main() -> int:
    """Run the hierarchy migration.

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
    rebuilt = 0
    pruned = 0
    skipped = 0
    failed = 0
    recovered_total = 0
    missing_total = 0

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
                label = f"{entry.get('conference') or 'all'} {entry.get('year') or ''}".strip()
                has_tree = _cache_has_tree(results)

                if has_tree and not args.rebuild:
                    if args.prune_levels and _cache_has_legacy_levels(results):
                        hierarchy = results.setdefault("hierarchy", {})
                        hierarchy.pop("levels", None)
                        hierarchy.pop("nr_levels", None)
                        hierarchy.pop("level_scale", None)
                        if not args.dry_run:
                            db.update_clustering_cache_results(entry["id"], results)
                        pruned += 1
                        print(f"[{entry['id']}] {label}: pruned legacy levels")
                    else:
                        skipped += 1
                    continue

                warnings: List[str] = []
                try:
                    recovered = _recover_existing_labels(results, warnings)
                    tree, recovered_count = _rebuild_tree(results, db, recovered, warnings)
                except Exception as exc:
                    failed += 1
                    print(f"[{entry['id']}] {label}: FAILED: {exc}", file=sys.stderr)
                    continue

                nodes = len(tree)
                missing = sum(1 for node in tree if not _has_text(node.get("description")))
                recovered_total += recovered_count
                missing_total += missing
                rebuilt += 1
                for warning in warnings:
                    print(f"[{entry['id']}] {label}: warning: {warning}", file=sys.stderr)

                _store_tree(results, tree, prune_levels=args.prune_levels)
                if args.prune_levels:
                    pruned += 1
                if not args.dry_run:
                    db.update_clustering_cache_results(entry["id"], results)

                action = "would rebuild" if args.dry_run else "rebuilt"
                print(
                    f"[{entry['id']}] {label}: {action} {nodes} tree nodes, "
                    f"{recovered_count} labels recovered, {missing} missing descriptions"
                )

    except Exception as exc:
        print(f"Migration failed: {exc}", file=sys.stderr)
        return 1

    print("\nSummary:")
    print(f"  processed: {processed}")
    print(f"  rebuilt:   {rebuilt}")
    print(f"  pruned:    {pruned}")
    print(f"  skipped:   {skipped}")
    print(f"  failed:    {failed}")
    print(f"  recovered hierarchy labels: {recovered_total}")
    print(f"  missing hierarchy descriptions: {missing_total}")
    if args.dry_run:
        print("  dry run: no changes were written")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

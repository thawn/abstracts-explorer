#!/usr/bin/env python3
"""Backfill robust (median) cluster centers into existing clustering-cache rows.

Flat ``cluster_centers`` and per-tree-node hierarchy centers written before the
median change are means. This idempotent migration recomputes both from the
cached points so the web UI and MCP tools can read centers directly instead of
recomputing them on every request.

Usage::

    python scripts/backfill_cluster_centers_median.py --dry-run
    python scripts/backfill_cluster_centers_median.py
    python scripts/backfill_cluster_centers_median.py \
        --paper-db /path/to/abstracts.db
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Mapping, Sequence, Tuple

from abstracts_explorer.config import get_config
from abstracts_explorer.database import DatabaseError, DatabaseManager
from abstracts_explorer.clustering.results import (
    attach_tree_centers,
    build_cluster_centers_from_points,
)

_TOLERANCE = 1e-9


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Recompute median flat and hierarchy-node cluster centers for " "existing clustering-cache rows."
        )
    )
    parser.add_argument(
        "--paper-db",
        help="Override PAPER_DB for this command only. Accepts a SQLite path or full database URL.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without committing.",
    )
    return parser.parse_args()


def _normalize_centers(values: Mapping[Any, Any] | None) -> Dict[int, Dict[str, float]]:
    """Normalize cached center keys to integers."""
    if not values:
        return {}
    return {int(key): value for key, value in values.items()}


def _centers_equal(
    left: Mapping[int, Mapping[str, float]] | None,
    right: Mapping[int, Mapping[str, float]] | None,
) -> bool:
    """Return whether two cluster-center maps are numerically equal."""
    left_map = _normalize_centers(left)
    right_map = _normalize_centers(right)
    if set(left_map) != set(right_map):
        return False
    for cluster_id, left_center in left_map.items():
        right_center = right_map[cluster_id]
        for axis in ("x", "y", "z"):
            if axis not in left_center and axis not in right_center:
                continue
            if abs(float(left_center.get(axis, 0.0)) - float(right_center.get(axis, 0.0))) > _TOLERANCE:
                return False
    return True


def _tree_centers_equal(
    old_tree: Sequence[Mapping[str, Any]],
    new_tree: Sequence[Mapping[str, Any]],
) -> bool:
    """Return whether two tree node lists have equal centers."""
    old_by_id: Dict[int, Mapping[str, float]] = {}
    new_by_id: Dict[int, Mapping[str, float]] = {}
    for node in old_tree:
        center = node.get("center")
        if isinstance(center, Mapping):
            old_by_id[int(node["id"])] = center
    for node in new_tree:
        center = node.get("center")
        if isinstance(center, Mapping):
            new_by_id[int(node["id"])] = center
    return _centers_equal(old_by_id, new_by_id)


def backfill_cluster_centers_median(
    db: DatabaseManager,
    dry_run: bool = False,
) -> Tuple[int, int, int, int]:
    """
    Recompute median flat and hierarchy-node centers for every cache row.

    Parameters
    ----------
    db : DatabaseManager
        Connected database manager.
    dry_run : bool, optional
        When True, do not persist changes.

    Returns
    -------
    tuple of (updated, unchanged, no_points, missing_tree)
        Counts describing the rows encountered.

    Raises
    ------
    DatabaseError
        If a database operation fails.
    """
    updated = 0
    unchanged = 0
    no_points = 0
    missing_tree = 0

    entries = db.list_clustering_cache_entries()
    for entry in entries:
        results = entry["results"]
        points = results.get("points") or []
        if not points:
            no_points += 1
            continue

        new_flat_centers = build_cluster_centers_from_points(points)
        tree = (results.get("hierarchy") or {}).get("tree")
        changed = not _centers_equal(results.get("cluster_centers"), new_flat_centers)
        new_tree = None
        if tree:
            new_tree = attach_tree_centers(tree, points)
            if not _tree_centers_equal(tree, new_tree):
                changed = True
        else:
            missing_tree += 1

        if not changed:
            unchanged += 1
            continue

        updated += 1
        if dry_run:
            continue

        results["cluster_centers"] = new_flat_centers
        if new_tree is not None:
            results.setdefault("hierarchy", {})["tree"] = new_tree
        db.update_clustering_cache_results(int(entry["id"]), results)

    return updated, unchanged, no_points, missing_tree


def main() -> int:
    """Run the backfill migration.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).
    """
    args = parse_args()

    if args.paper_db:
        os.environ["PAPER_DB"] = args.paper_db
    get_config(reload=True)

    try:
        with DatabaseManager() as db:
            updated, unchanged, no_points, missing_tree = backfill_cluster_centers_median(
                db,
                dry_run=args.dry_run,
            )
    except DatabaseError as exc:
        print(f"Error: {exc}")
        return 1

    action = "Would update" if args.dry_run else "Updated"
    print(f"{action} {updated} cache row(s)")
    print(f"Unchanged: {unchanged}")
    if no_points:
        print(f"Skipped (no points): {no_points}")
    if missing_tree:
        print(f"Rows missing hierarchy tree: {missing_tree}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

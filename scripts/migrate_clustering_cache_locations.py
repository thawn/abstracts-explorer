#!/usr/bin/env python3
"""
Migrate clustering caches to content-hash dedupe and location metadata.

The migration upgrades cache entries created before locations were added:

* collapses duplicate presentation rows by ``(conference, year, content_hash)``
* attaches ``location``/``locations`` from the paper database
* recomputes cluster statistics and centers for the deduped points
* updates the cache parameters to the current pipeline version

It does not re-run BERTopic, UMAP, or LLM labeling. Cluster assignments, labels,
keywords, and the hierarchy tree are preserved.

Usage
-----
    python scripts/migrate_clustering_cache_locations.py --dry-run
    python scripts/migrate_clustering_cache_locations.py
    python scripts/migrate_clustering_cache_locations.py --conference NeurIPS --year 2026
    python scripts/migrate_clustering_cache_locations.py --collection papers --force
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from typing import Any, Dict, List, Tuple

from sqlalchemy import text
from tqdm import tqdm

from abstracts_explorer.clustering.constants import DEFAULT_CLUSTERING_PARAMS
from abstracts_explorer.config import get_config
from abstracts_explorer.database import DatabaseManager


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Migrate clustering caches to location-aware, deduped results.")
    parser.add_argument("--paper-db", help="Override PAPER_DB for this command only.")
    parser.add_argument("--conference", help="Only migrate cache entries for this conference.")
    parser.add_argument("--year", type=int, help="Only migrate cache entries for this year.")
    parser.add_argument("--collection", help="Override the ChromaDB collection name stored in the cache key.")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing to the database.")
    parser.add_argument("--force", action="store_true", help="Re-migrate entries already at the current version.")
    return parser.parse_args()


def target_params(existing: Dict[str, Any], collection_name: str) -> Dict[str, Any]:
    """
    Build the cache parameter dictionary for the current pipeline version.

    Parameters
    ----------
    existing : dict
        Parameters currently stored with the cache entry.
    collection_name : str
        ChromaDB collection name used by the web UI.

    Returns
    -------
    dict
        Current parameter dictionary including ``__collection__``.
    """
    params = dict(DEFAULT_CLUSTERING_PARAMS)
    representation = existing.get("representation")
    if representation:
        params["representation"] = representation
    params["__collection__"] = collection_name
    return params


def migrate_results(
    results: Dict[str, Any],
    summaries: Dict[str, Dict[str, Any]],
    locations_by_hash: Dict[str, List[str]],
) -> Tuple[Dict[str, Any], int, int]:
    """
    Deduplicate cached points and attach location metadata.

    Parameters
    ----------
    results : dict
        Cached clustering payload.
    summaries : dict
        Mapping of paper UID to the database summary (includes ``content_hash``
        and ``location``).
    locations_by_hash : dict
        Mapping of content hash to all known presentation locations.

    Returns
    -------
    tuple
        ``(new_results, removed_points, missing_summaries)``.

    Raises
    ------
    ValueError
        If no points remain or a flat cluster loses all its points.
    """
    old_points = results.get("points") or []
    kept: List[Dict[str, Any]] = []
    seen: set[tuple] = set()
    removed = 0
    missing = 0

    for point in old_points:
        uid = str(point.get("id")) if point.get("id") is not None else None
        summary = summaries.get(uid) if uid else None
        content_hash = str((summary or {}).get("content_hash") or uid or "")
        dedupe_key = (point.get("conference"), point.get("year"), content_hash)
        if dedupe_key in seen:
            removed += 1
            continue
        seen.add(dedupe_key)

        new_point = dict(point)
        if summary is None:
            missing += 1
            new_point.setdefault("location", "")
            new_point.setdefault("locations", [])
        else:
            locations = locations_by_hash.get(content_hash)
            if not locations:
                single = summary.get("location")
                locations = [single] if single else []
            new_point["content_hash"] = summary.get("content_hash")
            new_point["location"] = locations[0] if locations else (summary.get("location") or "")
            new_point["locations"] = locations
        kept.append(new_point)

    if not kept:
        raise ValueError("migration left no points")

    old_statistics = results.get("statistics") or {}
    old_cluster_ids = {int(key) for key in (old_statistics.get("cluster_sizes") or {}) if int(key) >= 0}
    counts: Counter = Counter()
    for point in kept:
        cluster_id = point.get("cluster")
        if cluster_id is None:
            continue
        counts[int(cluster_id)] += 1

    empty_clusters = sorted(old_cluster_ids - {cluster_id for cluster_id in counts if cluster_id >= 0})
    if empty_clusters:
        sample = ", ".join(str(cluster_id) for cluster_id in empty_clusters[:5])
        raise ValueError(f"clusters lost all points after dedupe: {sample}")

    cluster_sizes = {cluster_id: int(size) for cluster_id, size in counts.items() if cluster_id >= 0}
    statistics = dict(old_statistics)
    statistics.update(
        {
            "cluster_sizes": cluster_sizes,
            "n_clusters": len(cluster_sizes),
            "n_noise": sum(1 for point in kept if point.get("cluster") == -1),
            "total_papers": len(kept),
        }
    )

    centers: Dict[int, Dict[str, float]] = {}
    for cluster_id in cluster_sizes:
        cluster_points = [point for point in kept if point.get("cluster") == cluster_id]
        if not cluster_points:
            continue
        center = {
            "x": sum(float(point.get("x", 0.0)) for point in cluster_points) / len(cluster_points),
            "y": sum(float(point.get("y", 0.0)) for point in cluster_points) / len(cluster_points),
        }
        if any("z" in point for point in cluster_points):
            center["z"] = sum(float(point.get("z", 0.0)) for point in cluster_points) / len(cluster_points)
        centers[cluster_id] = center

    new_results = dict(results)
    new_results["points"] = kept
    new_results["statistics"] = statistics
    new_results["cluster_centers"] = centers
    hierarchy = dict(new_results.get("hierarchy") or {})
    if hierarchy:
        hierarchy["n_topics"] = len(cluster_sizes)
        new_results["hierarchy"] = hierarchy
    return new_results, removed, missing


def migrate_row(
    db: DatabaseManager,
    row: Any,
    collection_name: str,
    target_version: int,
    force: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    """
    Migrate a single cache row.

    Parameters
    ----------
    db : DatabaseManager
        Connected database manager.
    row : Row
        Cache row with ``id``, ``conference``, ``year``, ``clustering_params``,
        and ``results_json`` fields.
    collection_name : str
        ChromaDB collection name used in the cache key.
    target_version : int
        Current pipeline version.
    force : bool
        Re-migrate rows already at *target_version*.
    dry_run : bool
        When ``True``, do not write changes.

    Returns
    -------
    dict
        Summary of the row migration.
    """
    cache_id, conference, year, params_json, results_json = row
    existing_params = json.loads(params_json or "{}")
    if not force and int(existing_params.get("pipeline_version") or 0) == target_version:
        return {
            "status": "already_current",
            "id": int(cache_id),
            "conference": conference,
            "year": year,
            "points_before": None,
            "points_after": None,
            "removed": 0,
            "missing": 0,
        }

    results = json.loads(results_json)
    points = results.get("points") or []
    uids = [str(point.get("id")) for point in points if point.get("id") is not None]
    summaries = {summary["uid"]: summary for summary in db.get_paper_summaries_by_uids(uids)}
    hashes = [str((summaries.get(uid) or {}).get("content_hash") or uid) for uid in uids]
    locations_by_hash = db.get_locations_by_content_hashes(hashes)

    try:
        new_results, removed, missing = migrate_results(results, summaries, locations_by_hash)
    except ValueError as exc:
        return {
            "status": "error",
            "id": int(cache_id),
            "conference": conference,
            "year": year,
            "error": str(exc),
            "points_before": len(points),
            "points_after": None,
            "removed": 0,
            "missing": 0,
        }

    new_params = target_params(existing_params, collection_name)
    status = "would_migrate" if dry_run else "migrated"

    if not dry_run:
        db.session.execute(
            text(
                "UPDATE clustering_cache "
                "SET results_json = :results, clustering_params = :params, n_clusters = :n_clusters "
                "WHERE id = :id"
            ),
            {
                "results": json.dumps(new_results, ensure_ascii=False),
                "params": json.dumps(new_params, sort_keys=True),
                "n_clusters": int(new_results["statistics"]["n_clusters"]),
                "id": int(cache_id),
            },
        )
        db.session.commit()

    return {
        "status": status,
        "id": int(cache_id),
        "conference": conference,
        "year": year,
        "points_before": len(points),
        "points_after": len(new_results["points"]),
        "removed": removed,
        "missing": missing,
        "old_version": existing_params.get("pipeline_version"),
        "new_version": new_params["pipeline_version"],
    }


def _scope_label(conference: Any, year: Any) -> str:
    """Return a printable cache scope label."""
    return f"{conference or 'all'} {year if year is not None else 'all'}"


def main() -> int:
    """
    Run the cache migration.

    Returns
    -------
    int
        Exit code (0 on success, 1 when any row failed).
    """
    args = parse_args()
    config = get_config()
    collection_name = args.collection or config.collection_name
    target_version = int(DEFAULT_CLUSTERING_PARAMS["pipeline_version"])

    conditions = []
    params: Dict[str, Any] = {}
    if args.conference is not None:
        conditions.append("conference = :conference")
        params["conference"] = args.conference
    if args.year is not None:
        conditions.append("year = :year")
        params["year"] = args.year
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    with DatabaseManager() as db:
        if args.paper_db:
            db.database_url = args.paper_db
        db.connect()
        print(f"Cache migration target: {db.database_url}")
        print(f"Collection: {collection_name}")
        print(f"Target pipeline version: {target_version}")
        print(f"Mode: {'dry run' if args.dry_run else 'write'}{' (force)' if args.force else ''}")

        rows = db.session.execute(
            text(
                "SELECT id, conference, year, clustering_params, results_json "
                f"FROM clustering_cache {where} ORDER BY conference, year, id"
            ),
            params,
        ).all()
        print(f"Cache entries to inspect: {len(rows):,}")

        summaries: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        progress = tqdm(rows, desc="Clustering caches", unit="row")
        try:
            for row in progress:
                summary = migrate_row(
                    db,
                    row,
                    collection_name,
                    target_version,
                    force=args.force,
                    dry_run=args.dry_run,
                )
                summaries.append(summary)
                if summary["status"] == "error":
                    errors.append(summary)
                    progress.set_postfix(errors=len(errors))
                elif summary["status"] in ("migrated", "would_migrate"):
                    progress.set_postfix(removed=sum(item["removed"] for item in summaries))
        finally:
            progress.close()

    migrated = [item for item in summaries if item["status"] in ("migrated", "would_migrate")]
    current = [item for item in summaries if item["status"] == "already_current"]
    total_before = sum(item["points_before"] or 0 for item in migrated)
    total_after = sum(item["points_after"] or 0 for item in migrated)
    total_removed = sum(item["removed"] for item in migrated)

    print("")
    print(f"Migrated (or would migrate): {len(migrated):,}")
    print(f"Already current: {len(current):,}")
    print(f"Errors: {len(errors):,}")
    print(f"Points: {total_before:,} -> {total_after:,} (removed {total_removed:,})")
    if migrated:
        print("")
        for item in sorted(migrated, key=lambda entry: entry["removed"] or 0, reverse=True):
            removed = item["removed"] or 0
            if removed:
                print(
                    f"  {_scope_label(item['conference'], item['year'])}: "
                    f"{item['points_before']:,} -> {item['points_after']:,} (removed {removed:,})"
                )
    if errors:
        print("")
        for item in errors:
            print(
                f"  ERROR {_scope_label(item['conference'], item['year'])}: {item.get('error')}",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

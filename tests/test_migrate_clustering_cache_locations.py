"""Tests for the clustering-cache location migration script."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from abstracts_explorer.clustering.constants import DEFAULT_CLUSTERING_PARAMS

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "migrate_clustering_cache_locations.py"
SPEC = importlib.util.spec_from_file_location("migrate_clustering_cache_locations", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
migration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(migration)

HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64


def _summary(uid: str, content_hash: str | None, location: str | None = None, year: int = 2026) -> dict:
    return {
        "uid": uid,
        "title": f"Title {uid}",
        "abstract": f"Abstract {uid}",
        "year": year,
        "conference": "NeurIPS",
        "session": "Poster Session",
        "location": location,
        "content_hash": content_hash,
    }


def _point(uid: str, cluster: int, x: float, y: float, *, year: int = 2026, conference: str = "NeurIPS") -> dict:
    return {
        "id": uid,
        "x": x,
        "y": y,
        "cluster": cluster,
        "title": f"Title {uid}",
        "year": year,
        "conference": conference,
        "session": "Poster Session",
    }


def _results(points: list[dict]) -> dict:
    return {
        "points": points,
        "statistics": {
            "n_clusters": 2,
            "n_noise": 0,
            "cluster_sizes": {"0": 3, "1": 1},
            "total_papers": 4,
        },
        "cluster_centers": {"0": {"x": 0.0, "y": 0.0}, "1": {"x": 1.0, "y": 1.0}},
        "cluster_labels": {0: "A", 1: "B"},
        "cluster_descriptions": {0: "A desc.", 1: "B desc."},
        "cluster_keywords": {0: ["a"], 1: ["b"]},
        "hierarchy": {"method": "bertopic", "n_topics": 2, "tree": [{"id": 2, "topics": [0, 1]}]},
        "n_dimensions": 2,
    }


def test_target_params_uses_current_version_and_collection():
    params = migration.target_params({"representation": "ctfidf"}, "custom_collection")
    assert params["pipeline_version"] == DEFAULT_CLUSTERING_PARAMS["pipeline_version"]
    assert params["representation"] == "ctfidf"
    assert params["__collection__"] == "custom_collection"


def test_migrate_results_dedupes_and_attaches_locations():
    results = _results(
        [
            _point("p1", 0, 0.0, 0.0),
            _point("p2", 0, 10.0, 10.0),
            _point("p3", 0, 2.0, 2.0),
            _point("p4", 1, 1.0, 1.0),
        ]
    )
    hash_c = "sha256:" + "c" * 64
    summaries = {
        "p1": _summary("p1", HASH_A, "Sydney"),
        "p2": _summary("p2", HASH_A, "Paris"),
        "p3": _summary("p3", HASH_B, "Sydney"),
        "p4": _summary("p4", hash_c, "Sydney"),
    }
    locations = {HASH_A: ["Paris", "Sydney"], HASH_B: ["Sydney"], hash_c: ["Sydney"]}

    new_results, removed, missing = migration.migrate_results(results, summaries, locations)

    assert removed == 1  # p2 duplicates p1
    assert missing == 0
    assert [point["id"] for point in new_results["points"]] == ["p1", "p3", "p4"]
    assert new_results["points"][0]["location"] == "Paris"
    assert new_results["points"][0]["locations"] == ["Paris", "Sydney"]
    assert new_results["points"][0]["content_hash"] == HASH_A
    assert new_results["statistics"]["total_papers"] == 3
    assert new_results["statistics"]["cluster_sizes"] == {0: 2, 1: 1}
    assert new_results["statistics"]["n_clusters"] == 2
    assert new_results["cluster_centers"][0] == {"x": 1.0, "y": 1.0}
    assert new_results["cluster_centers"][1] == {"x": 1.0, "y": 1.0}
    assert new_results["hierarchy"]["n_topics"] == 2
    assert new_results["cluster_labels"] == results["cluster_labels"]


def test_migrate_results_does_not_merge_across_years():
    results = _results(
        [
            _point("p1", 0, 0.0, 0.0, year=2025),
            _point("p2", 0, 1.0, 1.0, year=2026),
        ]
    )
    results["statistics"]["cluster_sizes"] = {"0": 2}
    summaries = {"p1": _summary("p1", HASH_A, year=2025), "p2": _summary("p2", HASH_A, year=2026)}

    new_results, removed, missing = migration.migrate_results(results, summaries, {HASH_A: ["Sydney"]})

    assert removed == 0
    assert missing == 0
    assert len(new_results["points"]) == 2


def test_migrate_results_uses_uid_fallback_for_missing_hash():
    results = _results([_point("p1", 0, 0.0, 0.0)])
    results["statistics"]["cluster_sizes"] = {"0": 1}
    summaries = {"p1": _summary("p1", None, "Sydney")}

    new_results, _, _ = migration.migrate_results(results, summaries, {})

    assert new_results["points"][0]["content_hash"] is None
    assert new_results["points"][0]["locations"] == ["Sydney"]


def test_migrate_results_aborts_when_cluster_loses_all_points():
    results = _results(
        [
            _point("p1", 0, 0.0, 0.0),
            _point("p2", 1, 1.0, 1.0),
        ]
    )
    results["statistics"]["cluster_sizes"] = {"0": 1, "1": 1}
    summaries = {"p1": _summary("p1", HASH_A, "Sydney"), "p2": _summary("p2", HASH_A, "Sydney")}

    try:
        migration.migrate_results(results, summaries, {HASH_A: ["Sydney"]})
    except ValueError as exc:
        assert "clusters lost all points" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError")

"""Tests for the full-hierarchy migration script."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from unittest.mock import Mock, patch

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "migrate_full_hierarchy.py"
SPEC = importlib.util.spec_from_file_location("migrate_full_hierarchy", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
migration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(migration)


def _legacy_results():
    return {
        "points": [
            {"id": "p0", "cluster": 0},
            {"id": "p1", "cluster": 1},
        ],
        "statistics": {"n_clusters": 2, "cluster_sizes": {0: 1, 1: 1}},
        "cluster_labels": {0: "A", 1: "B"},
        "cluster_descriptions": {0: "A desc.", 1: "B desc."},
        "cluster_keywords": {0: ["a"], 1: ["b"]},
        "hierarchy": {
            "method": "bertopic",
            "nr_levels": 1,
            "level_scale": "linear",
            "levels": [
                {
                    "level": 0,
                    "max_distance": 0.1,
                    "mapping": {"0": 2, "1": 2},
                    "clusters": [
                        {
                            "id": 2,
                            "label": "Parent",
                            "description": "Parent description.",
                            "size": 2,
                            "keywords": ["a", "b"],
                            "source_topic_count": 2,
                        }
                    ],
                }
            ],
        },
    }


def test_recover_level_labels():
    warnings = []
    recovered = migration._recover_level_labels(_legacy_results(), warnings)

    assert recovered == {frozenset({0, 1}): ("Parent", "Parent description.")}
    assert warnings == []


def test_recover_level_labels_reports_conflicts():
    results = _legacy_results()
    results["hierarchy"]["levels"].append(
        {
            "level": 1,
            "max_distance": 0.2,
            "mapping": {"0": 4, "1": 4},
            "clusters": [
                {
                    "id": 4,
                    "label": "Other Parent",
                    "description": "Other description.",
                    "size": 2,
                    "keywords": ["a", "b"],
                    "source_topic_count": 2,
                }
            ],
        }
    )
    warnings = []

    recovered = migration._recover_level_labels(results, warnings)

    assert recovered == {frozenset({0, 1}): ("Parent", "Parent description.")}
    assert warnings and "conflicting labels" in warnings[0]


def test_recover_existing_labels_prefers_tree_labels():
    results = _legacy_results()
    results["hierarchy"]["tree"] = [
        {
            "id": 2,
            "topics": [0, 1],
            "label": "Tree Parent",
            "description": "Tree description.",
            "distance": 0.1,
            "child_left_id": 0,
            "child_right_id": 1,
        }
    ]
    warnings = []

    recovered = migration._recover_existing_labels(results, warnings)

    assert recovered == {frozenset({0, 1}): ("Tree Parent", "Tree description.")}


def test_recover_existing_labels_prefers_level_label_over_keyword_tree_label():
    results = _legacy_results()
    results["hierarchy"]["levels"][0]["clusters"][0]["description"] = ""
    results["hierarchy"]["levels"][0]["clusters"][0]["label"] = "Useful Parent Label"
    results["hierarchy"]["tree"] = [
        {
            "id": 2,
            "topics": [0, 1],
            "label": "keyword fallback",
            "description": "",
            "distance": 0.1,
            "child_left_id": 0,
            "child_right_id": 1,
        }
    ]

    recovered = migration._recover_existing_labels(results, [])

    assert recovered == {frozenset({0, 1}): ("Useful Parent Label", "")}


def test_store_tree_keeps_levels_by_default():
    results = _legacy_results()
    tree = [{"id": 2, "topics": [0, 1], "label": "Parent", "description": "", "distance": 0.1}]

    migration._store_tree(results, tree, prune_levels=False)

    assert results["hierarchy"]["tree"] == tree
    assert "levels" in results["hierarchy"]
    assert results["hierarchy"]["nr_levels"] == 1


def test_store_tree_prunes_levels():
    results = _legacy_results()
    tree = [{"id": 2, "topics": [0, 1], "label": "Parent", "description": "", "distance": 0.1}]

    migration._store_tree(results, tree, prune_levels=True)

    assert results["hierarchy"]["tree"] == tree
    assert "levels" not in results["hierarchy"]
    assert "nr_levels" not in results["hierarchy"]
    assert "level_scale" not in results["hierarchy"]


def test_main_rebuilds_and_saves_rows():
    results = _legacy_results()
    entry = {
        "id": 7,
        "conference": "AAAI",
        "year": 2026,
        "embedding_model": "model",
        "clustering_method": "bertopic-agglomerative",
        "clustering_params": {"__collection__": "papers"},
        "results": results,
    }
    tree = [
        {
            "id": 2,
            "topics": [0, 1],
            "label": "Parent",
            "description": "Parent description.",
            "distance": 0.1,
            "child_left_id": 0,
            "child_right_id": 1,
        }
    ]

    mock_db = Mock()
    mock_db.__enter__ = Mock(return_value=mock_db)
    mock_db.__exit__ = Mock(return_value=False)
    mock_db.list_clustering_cache_entries.return_value = [entry]

    args = argparse.Namespace(
        paper_db=None,
        conference=None,
        year=None,
        collection=None,
        embedding_model=None,
        dry_run=False,
        rebuild=False,
        prune_levels=False,
    )

    with (
        patch.object(migration, "parse_args", return_value=args),
        patch.object(migration, "get_config"),
        patch.object(migration, "DatabaseManager", return_value=mock_db),
        patch.object(migration, "_rebuild_tree", return_value=(tree, 1)),
    ):
        exit_code = migration.main()

    assert exit_code == 0
    mock_db.update_clustering_cache_results.assert_called_once()
    saved_id, saved_results = mock_db.update_clustering_cache_results.call_args.args
    assert saved_id == 7
    assert saved_results["hierarchy"]["tree"] == tree
    assert "levels" in saved_results["hierarchy"]


def test_main_skips_existing_tree():
    results = _legacy_results()
    results["hierarchy"]["tree"] = [{"id": 2, "topics": [0, 1]}]
    entry = {
        "id": 7,
        "conference": "AAAI",
        "year": 2026,
        "embedding_model": "model",
        "clustering_method": "bertopic-agglomerative",
        "clustering_params": {"__collection__": "papers"},
        "results": results,
    }
    mock_db = Mock()
    mock_db.__enter__ = Mock(return_value=mock_db)
    mock_db.__exit__ = Mock(return_value=False)
    mock_db.list_clustering_cache_entries.return_value = [entry]

    args = argparse.Namespace(
        paper_db=None,
        conference=None,
        year=None,
        collection=None,
        embedding_model=None,
        dry_run=False,
        rebuild=False,
        prune_levels=False,
    )

    with (
        patch.object(migration, "parse_args", return_value=args),
        patch.object(migration, "get_config"),
        patch.object(migration, "DatabaseManager", return_value=mock_db),
        patch.object(migration, "_rebuild_tree") as rebuild_tree,
    ):
        exit_code = migration.main()

    assert exit_code == 0
    rebuild_tree.assert_not_called()
    mock_db.update_clustering_cache_results.assert_not_called()

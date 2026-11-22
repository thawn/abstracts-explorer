"""Tests for the legacy hierarchy cache prune script."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from unittest.mock import Mock, patch

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "archive" / "prune_legacy_hierarchy_cache.py"
SPEC = importlib.util.spec_from_file_location("prune_legacy_hierarchy_cache", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
migration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(migration)


def _entry():
    return {
        "id": 7,
        "conference": "AAAI",
        "year": 2026,
        "embedding_model": "model",
        "clustering_method": "bertopic-agglomerative",
        "clustering_params": {"distance_threshold": 0.5, "nr_levels": 10, "level_scale": "linear"},
        "results": {
            "points": [],
            "statistics": {"n_clusters": 1},
            "hierarchy": {
                "method": "bertopic",
                "n_topics": 1,
                "nr_levels": 10,
                "level_scale": "linear",
                "levels": [],
                "tree": [{"id": 2, "topics": [0, 1], "label": "Parent", "description": "Desc."}],
            },
        },
    }


def _args(**overrides):
    values = {
        "paper_db": None,
        "conference": None,
        "year": None,
        "collection": None,
        "embedding_model": None,
        "dry_run": False,
        "keep_label_tables": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_clean_results_removes_level_keys():
    results = _entry()["results"]

    assert migration._clean_results(results) is True
    assert "tree" in results["hierarchy"]
    assert "levels" not in results["hierarchy"]
    assert "nr_levels" not in results["hierarchy"]
    assert "level_scale" not in results["hierarchy"]
    assert migration._clean_results(results) is False


def test_clean_params_removes_level_keys():
    params = _entry()["clustering_params"]

    assert migration._clean_params(params) is True
    assert params == {"distance_threshold": 0.5}
    assert migration._clean_params(params) is False


def test_main_cleans_rows_and_drops_tables():
    entry = _entry()
    mock_db = Mock()
    mock_db.__enter__ = Mock(return_value=mock_db)
    mock_db.__exit__ = Mock(return_value=False)
    mock_db.list_clustering_cache_entries.return_value = [entry]

    with (
        patch.object(migration, "parse_args", return_value=_args()),
        patch.object(migration, "get_config"),
        patch.object(migration, "DatabaseManager", return_value=mock_db),
        patch.object(migration, "_table_exists", return_value=True),
        patch.object(migration, "_drop_table") as drop_table,
    ):
        exit_code = migration.main()

    assert exit_code == 0
    mock_db.update_clustering_cache_entry.assert_called_once()
    call_kwargs = mock_db.update_clustering_cache_entry.call_args.kwargs
    assert "levels" not in call_kwargs["results"]["hierarchy"]
    assert "nr_levels" not in call_kwargs["clustering_params"]
    assert [call.args[1] for call in drop_table.call_args_list] == list(migration.OBSOLETE_TABLES)


def test_main_dry_run_does_not_write():
    entry = _entry()
    mock_db = Mock()
    mock_db.__enter__ = Mock(return_value=mock_db)
    mock_db.__exit__ = Mock(return_value=False)
    mock_db.list_clustering_cache_entries.return_value = [entry]

    with (
        patch.object(migration, "parse_args", return_value=_args(dry_run=True)),
        patch.object(migration, "get_config"),
        patch.object(migration, "DatabaseManager", return_value=mock_db),
        patch.object(migration, "_table_exists", return_value=True),
        patch.object(migration, "_drop_table") as drop_table,
    ):
        exit_code = migration.main()

    assert exit_code == 0
    mock_db.update_clustering_cache_entry.assert_not_called()
    drop_table.assert_not_called()


def test_main_fails_rows_without_tree():
    entry = _entry()
    entry["results"]["hierarchy"].pop("tree")
    mock_db = Mock()
    mock_db.__enter__ = Mock(return_value=mock_db)
    mock_db.__exit__ = Mock(return_value=False)
    mock_db.list_clustering_cache_entries.return_value = [entry]

    with (
        patch.object(migration, "parse_args", return_value=_args()),
        patch.object(migration, "get_config"),
        patch.object(migration, "DatabaseManager", return_value=mock_db),
    ):
        exit_code = migration.main()

    assert exit_code == 1
    mock_db.update_clustering_cache_entry.assert_not_called()

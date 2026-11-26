"""
Tests for the median cluster-center backfill script.
"""

import importlib.util
import json
from pathlib import Path

from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.db_models import ClusteringCache
from tests.conftest import set_test_db

_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "backfill_cluster_centers_median.py"
_spec = importlib.util.spec_from_file_location("backfill_cluster_centers_median", _SCRIPT_PATH)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
backfill_cluster_centers_median = _module.backfill_cluster_centers_median


def _results_with_outlier() -> dict:
    """Return cached results whose mean and median centers differ."""
    return {
        "points": [
            {"id": "p1", "cluster": 0, "x": 0.0, "y": 0.0},
            {"id": "p2", "cluster": 0, "x": 2.0, "y": 0.0},
            {"id": "p3", "cluster": 0, "x": 100.0, "y": 100.0},
            {"id": "p4", "cluster": 1, "x": 10.0, "y": 10.0},
        ],
        "statistics": {"n_clusters": 2, "cluster_sizes": {0: 3, 1: 1}},
        "cluster_centers": {"0": {"x": 34.0, "y": 33.3333333333}, "1": {"x": 10.0, "y": 10.0}},
        "hierarchy": {
            "method": "bertopic",
            "n_topics": 2,
            "tree": [
                {
                    "id": 10,
                    "topics": [0, 1],
                    "label": "Merged",
                    "description": "",
                    "distance": 0.5,
                }
            ],
        },
    }


def _add_cache_row(db: DatabaseManager, results: dict) -> ClusteringCache:
    """Insert one clustering-cache row carrying *results*."""
    row = ClusteringCache(
        embedding_model="test-model",
        reduction_method="tsne",
        n_components=2,
        clustering_method="agglomerative",
        n_clusters=2,
        clustering_params=json.dumps({"linkage": "ward"}),
        results_json=json.dumps(results),
    )
    db.session.add(row)
    db.session.commit()
    return row


def test_backfill_recomputes_flat_and_node_centers(tmp_path):
    """Outlier-shifted mean centers are replaced by medians."""
    set_test_db(tmp_path / "test.db")
    with DatabaseManager() as db:
        db.create_tables()
        row = _add_cache_row(db, _results_with_outlier())

        updated, unchanged, no_points, missing_tree = backfill_cluster_centers_median(db)

        assert (updated, unchanged, no_points, missing_tree) == (1, 0, 0, 0)
        db.session.expire_all()
        stored = json.loads(row.results_json)
        assert stored["cluster_centers"]["0"] == {"x": 2.0, "y": 0.0}
        assert stored["cluster_centers"]["1"] == {"x": 10.0, "y": 10.0}
        assert stored["hierarchy"]["tree"][0]["center"] == {"x": 6.0, "y": 5.0}


def test_backfill_dry_run_does_not_commit(tmp_path):
    """Dry-run reports the update but leaves the stored JSON unchanged."""
    set_test_db(tmp_path / "test.db")
    with DatabaseManager() as db:
        db.create_tables()
        row = _add_cache_row(db, _results_with_outlier())
        original = row.results_json

        updated, _, _, _ = backfill_cluster_centers_median(db, dry_run=True)

        assert updated == 1
        db.session.expire_all()
        assert row.results_json == original


def test_backfill_is_idempotent(tmp_path):
    """A second run leaves already-backfilled rows unchanged."""
    set_test_db(tmp_path / "test.db")
    with DatabaseManager() as db:
        db.create_tables()
        _add_cache_row(db, _results_with_outlier())

        backfill_cluster_centers_median(db)
        updated, unchanged, _, _ = backfill_cluster_centers_median(db)

        assert updated == 0
        assert unchanged == 1


def test_backfill_skips_rows_without_points(tmp_path):
    """Rows without points are skipped; missing trees are reported."""
    set_test_db(tmp_path / "test.db")
    with DatabaseManager() as db:
        db.create_tables()
        _add_cache_row(db, {"points": [], "hierarchy": {"tree": []}})
        _add_cache_row(
            db,
            {
                "points": [{"id": "p1", "cluster": 0, "x": 1.0, "y": 1.0}],
                "statistics": {"n_clusters": 1, "cluster_sizes": {0: 1}},
                "hierarchy": {"method": "bertopic"},
            },
        )

        updated, unchanged, no_points, missing_tree = backfill_cluster_centers_median(db)

        assert no_points == 1
        assert missing_tree == 1
        assert updated == 1
        assert unchanged == 0

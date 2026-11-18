"""
Tests for the clustering-cache collection backfill script.
"""

import importlib.util
import json
from pathlib import Path

from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.db_models import ClusteringCache
from tests.conftest import set_test_db

_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "backfill_clustering_cache_collection.py"
_spec = importlib.util.spec_from_file_location("backfill_clustering_cache_collection", _SCRIPT_PATH)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
backfill_clustering_cache_collection = _module.backfill_clustering_cache_collection


def _add_cache_row(db: DatabaseManager, params, results_json: str = "{}") -> ClusteringCache:
    """Insert one clustering-cache row with the given params."""
    row = ClusteringCache(
        embedding_model="test-model",
        reduction_method="tsne",
        n_components=2,
        clustering_method="agglomerative",
        n_clusters=2,
        clustering_params=params,
        results_json=results_json,
    )
    db.session.add(row)
    db.session.commit()
    return row


def test_backfill_adds_collection_and_reports_counts(tmp_path):
    """Unscoped rows are updated; scoped/null/malformed rows are reported."""
    set_test_db(tmp_path / "test.db")
    with DatabaseManager() as db:
        db.create_tables()
        unscoped = _add_cache_row(db, json.dumps({"linkage": "ward"}))
        scoped = _add_cache_row(
            db,
            json.dumps({"linkage": "ward", "__collection__": "already"}),
        )
        null_params = _add_cache_row(db, None)
        malformed = _add_cache_row(db, "not-json")

        updated, already_scoped, null_count, malformed_count = backfill_clustering_cache_collection(
            db,
            "papers",
        )

        assert (updated, already_scoped, null_count, malformed_count) == (1, 1, 1, 1)

        db.session.expire_all()
        assert json.loads(unscoped.clustering_params)["__collection__"] == "papers"
        assert json.loads(scoped.clustering_params)["__collection__"] == "already"
        assert null_params.clustering_params is None
        assert malformed.clustering_params == "not-json"


def test_backfill_dry_run_does_not_commit(tmp_path):
    """Dry-run reports the update but leaves the stored JSON unchanged."""
    set_test_db(tmp_path / "test.db")
    with DatabaseManager() as db:
        db.create_tables()
        row = _add_cache_row(db, json.dumps({"linkage": "ward"}))

        updated, _, _, _ = backfill_clustering_cache_collection(db, "papers", dry_run=True)

        assert updated == 1
        db.session.rollback()
        db.session.expire_all()
        assert "__collection__" not in json.loads(row.clustering_params)


def test_backfill_is_idempotent(tmp_path):
    """A second run does not change already-backfilled rows."""
    set_test_db(tmp_path / "test.db")
    with DatabaseManager() as db:
        db.create_tables()
        row = _add_cache_row(db, json.dumps({"linkage": "ward"}))

        backfill_clustering_cache_collection(db, "papers")
        db.session.expire_all()
        first_value = row.clustering_params

        updated, already_scoped, _, _ = backfill_clustering_cache_collection(db, "papers")

        assert updated == 0
        assert already_scoped == 1
        db.session.expire_all()
        assert row.clustering_params == first_value

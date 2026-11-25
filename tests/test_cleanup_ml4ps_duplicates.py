"""Tests for the ML4PS duplicate cleanup script."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.db_models import ClusteringCache, Paper, ValidationData
from abstracts_explorer.plugin import LightweightPaper
from tests.conftest import set_test_db

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "cleanup_ml4ps_duplicates.py"
SPEC = importlib.util.spec_from_file_location("cleanup_ml4ps_duplicates", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
cleanup = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cleanup
SPEC.loader.exec_module(cleanup)


def _paper(uid: str, title: str, conference: str, year: int = 2024, original_id: str | None = None) -> Paper:
    return Paper(
        uid=uid,
        original_id=original_id,
        title=title,
        authors="Alice",
        abstract="Abstract",
        session="Session",
        poster_position="1",
        year=year,
        conference=conference,
    )


def _cache(conference: str, results: dict, params: str = "{}", n_clusters: int = 2) -> ClusteringCache:
    return ClusteringCache(
        embedding_model="model",
        conference=conference,
        year=2024,
        reduction_method="umap",
        n_components=2,
        clustering_method="hdbscan",
        n_clusters=n_clusters,
        clustering_params=params,
        results_json=json.dumps(results),
    )


def test_build_plan_classifies_duplicates_and_renames():
    duplicate = _paper("old-dup", "Duplicate", "ML4PS@Neurips")
    rename = _paper("old-new", "Rename", "ML4PS@Neurips")
    canonical_dup_uid = DatabaseManager.compute_uid("Duplicate", None, "ML4PS", 2024)

    plan = cleanup.build_plan([duplicate, rename], {canonical_dup_uid}, "ML4PS")

    assert plan.mapping["old-dup"] == canonical_dup_uid
    assert plan.duplicates == ["old-dup"]
    assert plan.renames == ["old-new"]


def test_migrate_database_removes_duplicates_and_renames_rest(tmp_path):
    set_test_db(tmp_path / "test.db")
    with DatabaseManager() as db:
        db.create_tables()
        db.add_papers(
            [
                LightweightPaper(
                    title="Duplicate",
                    authors=["Alice"],
                    abstract="Abstract",
                    session="Session",
                    poster_position="1",
                    year=2024,
                    conference="ML4PS",
                )
            ]
        )
        canonical_uid = DatabaseManager.compute_uid("Duplicate", None, "ML4PS", 2024)
        legacy_dup_uid = DatabaseManager.compute_uid("Duplicate", None, "ML4PS@Neurips", 2024)
        legacy_rename_uid = DatabaseManager.compute_uid("Rename", None, "ML4PS@Neurips", 2024)
        db.session.add_all(
            [
                _paper(legacy_dup_uid, "Duplicate", "ML4PS@Neurips"),
                _paper(legacy_rename_uid, "Rename", "ML4PS@Neurips"),
                ValidationData(paper_uid=legacy_dup_uid, priority=3),
                _cache(
                    "ML4PS@Neurips",
                    {"points": [{"id": legacy_dup_uid, "x": 1.0}]},
                    n_clusters=3,
                ),
                _cache("ML4PS", {"points": [{"id": canonical_uid, "x": 1.0}]}),
                _cache(
                    "ML4PS@Neurips",
                    {"points": [{"id": legacy_rename_uid, "x": 2.0}]},
                    params='{"different": true}',
                ),
            ]
        )
        db.session.commit()

        plan = cleanup._migrate_database(db, "ML4PS@Neurips", "ML4PS", dry_run=False)

        new_rename_uid = plan.mapping[legacy_rename_uid]
        uids = {row[0] for row in db.session.query(Paper.uid)}
        assert legacy_dup_uid not in uids
        assert legacy_rename_uid not in uids
        assert new_rename_uid in uids
        assert db.session.get(Paper, canonical_uid).conference == "ML4PS"
        assert db.session.get(Paper, new_rename_uid).conference == "ML4PS"

        validation = db.session.query(ValidationData).one()
        assert validation.paper_uid == canonical_uid

        caches = db.session.query(ClusteringCache).all()
        assert len(caches) == 2
        assert {cache.conference for cache in caches} == {"ML4PS"}
        migrated_results = [json.loads(cache.results_json) for cache in caches]
        assert any(entry["points"][0]["id"] == new_rename_uid for entry in migrated_results)


def test_migrate_database_dry_run_writes_nothing(tmp_path):
    set_test_db(tmp_path / "test.db")
    with DatabaseManager() as db:
        db.create_tables()
        legacy_uid = DatabaseManager.compute_uid("Rename", None, "ML4PS@Neurips", 2024)
        db.session.add(_paper(legacy_uid, "Rename", "ML4PS@Neurips"))
        db.session.commit()

        plan = cleanup._migrate_database(db, "ML4PS@Neurips", "ML4PS", dry_run=True)

        assert plan.mapping == {legacy_uid: DatabaseManager.compute_uid("Rename", None, "ML4PS", 2024)}
        assert db.session.get(Paper, legacy_uid).conference == "ML4PS@Neurips"


def test_migrate_embeddings_deletes_duplicate_and_moves_unique():
    collection = Mock()
    collection.get.side_effect = lambda ids, include: (
        {"ids": ids, "documents": ["text"], "embeddings": [[0.1]], "metadatas": [{"conference": "old"}]}
        if len(include) > 0
        else {"ids": ["new-dup"] if ids == ["new-dup"] else []}
    )
    em = Mock(collection=collection)
    em.close = Mock()

    with patch("abstracts_explorer.embeddings.EmbeddingsManager", return_value=em):
        cleanup._migrate_embeddings(
            {"old-dup": "new-dup", "old-new": "new-new"},
            "ML4PS",
            dry_run=False,
        )

    collection.add.assert_called_once()
    add_kwargs = collection.add.call_args.kwargs
    assert add_kwargs["ids"] == ["new-new"]
    assert add_kwargs["metadatas"] == [{"conference": "ML4PS"}]
    assert em.delete_embeddings_by_ids.call_count == 2
    em.delete_embeddings_by_ids.assert_any_call(["old-dup"])
    em.delete_embeddings_by_ids.assert_any_call(["old-new"])
    em.close.assert_called_once()


def test_normalize_json_caches_updates_conference(tmp_path):
    cache_path = tmp_path / "ml4ps_2024.json"
    cache_path.write_text(
        json.dumps([{"title": "Paper", "conference": "ML4PS@Neurips"}]),
        encoding="utf-8",
    )

    changed = cleanup.normalize_json_caches(tmp_path, "ml4ps_*.json", "ML4PS@Neurips", "ML4PS", dry_run=True)
    assert changed == 1
    assert json.loads(cache_path.read_text(encoding="utf-8"))[0]["conference"] == "ML4PS@Neurips"

    changed = cleanup.normalize_json_caches(tmp_path, "ml4ps_*.json", "ML4PS@Neurips", "ML4PS", dry_run=False)
    assert changed == 1
    assert json.loads(cache_path.read_text(encoding="utf-8"))[0]["conference"] == "ML4PS"


@pytest.mark.parametrize("apply", [False, True])
def test_main_respects_apply_flag(tmp_path, apply):
    set_test_db(tmp_path / "test.db")
    with DatabaseManager() as db:
        db.create_tables()
        legacy_uid = DatabaseManager.compute_uid("Rename", None, "ML4PS@Neurips", 2024)
        db.session.add(_paper(legacy_uid, "Rename", "ML4PS@Neurips"))
        db.session.commit()

    args = [
        "--paper-db",
        str(tmp_path / "test.db"),
        "--data-dir",
        str(tmp_path),
        "--skip-embeddings",
        "--skip-json",
    ]
    if apply:
        args.append("--apply")

    assert cleanup.main(args) == 0

    with DatabaseManager() as db:
        remaining = db.session.get(Paper, legacy_uid)
        assert (remaining is None) is apply

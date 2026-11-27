"""
Tests for the cross-database consistency checker.
"""

from __future__ import annotations

import json

import pytest

from abstracts_explorer.consistency import (
    ConsistencyIssue,
    ConsistencyReport,
    apply_cache_fixes,
    apply_embedding_fixes,
    check_consistency,
)
from abstracts_explorer.embeddings.hashing import hash_text
from abstracts_explorer.plugin import LightweightPaper


class FakeCollection:
    """Minimal stand-in for a ChromaDB collection backed by a dict."""

    def __init__(self, name: str, records: dict[str, str]) -> None:
        self.name = name
        self.records = dict(records)

    def count(self) -> int:
        return len(self.records)

    def get(self, ids=None, include=None, limit=None, offset=None):
        include = include or []
        keys = list(self.records) if ids is None else [str(uid) for uid in ids if str(uid) in self.records]
        if offset:
            keys = keys[offset:]
        if limit is not None:
            keys = keys[:limit]
        result: dict = {"ids": keys}
        if "documents" in include:
            result["documents"] = [self.records[uid] for uid in keys]
        if "embeddings" in include:
            result["embeddings"] = [[0.0] for _ in keys]
        return result

    def delete(self, ids=None, **kwargs):
        removed = 0
        for uid in ids or []:
            if str(uid) in self.records:
                del self.records[str(uid)]
                removed += 1
        return removed


class FakeCollectionMeta:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeClient:
    def __init__(self, names) -> None:
        self._names = list(names)

    def list_collections(self):
        return [FakeCollectionMeta(name) for name in self._names]


class FakeEmbeddingsManager:
    def __init__(self, records: dict[str, str], *, collection_name="papers", collections=None) -> None:
        self.collection_name = collection_name
        self.collection = FakeCollection(collection_name, records)
        self.client = FakeClient(collections or [collection_name])

    def delete_embeddings_by_ids(self, ids):
        if not ids:
            return 0
        return self.collection.delete(ids=list(ids))


def _add_paper(db, uid_title, *, conference="NeurIPS", year=2025):
    paper = LightweightPaper(
        title=uid_title,
        abstract=f"Abstract for {uid_title}",
        authors=["Author"],
        session="Session 1",
        poster_position="P1",
        year=year,
        conference=conference,
    )
    uid = db.add_paper(paper)
    text = f"{paper.title}\n\n{paper.abstract}".strip()
    return uid, hash_text(text)


def _cache(db, conference, year, points, total=None, collection="papers", model="test-model", **params):
    results = {
        "points": points,
        "statistics": {
            "n_clusters": 1,
            "total_papers": len(points) if total is None else total,
        },
    }
    db.save_clustering_cache(
        embedding_model=model,
        reduction_method="umap",
        n_components=2,
        clustering_method="bertopic-agglomerative",
        results=results,
        clustering_params={"distance_threshold": 0.5, **params},
        conference=conference,
        year=year,
        collection_name=collection,
    )


def _point(uid):
    return {"id": uid, "x": 1.0, "y": 2.0, "cluster": 0}


@pytest.fixture
def setup(connected_db):
    uid, doc = _add_paper(connected_db, "Paper One")
    connected_db.set_embedding_model("test-model")
    _cache(connected_db, "NeurIPS", 2025, [_point(uid)])
    em = FakeEmbeddingsManager({uid: doc})
    return connected_db, em, uid, doc


def _categories(report):
    return {issue.category for issue in report.issues}


class TestConsistentData:
    def test_clean_state_has_no_errors_or_warnings(self, setup):
        db, em, *_ = setup
        report = check_consistency(db, em, embedding_model="test-model")

        assert report.errors == []
        assert report.warnings == []
        assert report.scopes_checked == 1
        assert report.embeddings_checked == 1

    def test_progress_callback_reports_stages(self, setup):
        db, em, *_ = setup
        events: list[tuple[str, int, int]] = []

        check_consistency(db, em, embedding_model="test-model", progress=lambda *a: events.append(a))

        stages = {stage for stage, _, _ in events}
        assert "Checking embeddings" in stages
        assert "Checking clustering cache" in stages
        assert all(0 <= completed <= total for _, completed, total in events)


class TestEmbeddingChecks:
    def test_missing_embedding_is_an_error(self, setup):
        db, em, uid, _ = setup
        em.collection.records.clear()

        report = check_consistency(db, em, embedding_model="test-model")

        assert "embeddings" in _categories(report)
        assert any("missing" in issue.message.lower() for issue in report.errors)

    def test_orphan_embedding_is_an_error(self, setup):
        db, em, *_ = setup
        em.collection.records["deadbeefdeadbeef"] = hash_text("orphan")

        report = check_consistency(db, em, embedding_model="test-model")

        assert any("orphan" in issue.message.lower() for issue in report.errors)

    def test_orphan_embedding_marks_for_fix(self, setup):
        db, em, *_ = setup
        em.collection.records["deadbeefdeadbeef"] = hash_text("orphan")

        report = check_consistency(db, em, embedding_model="test-model")

        assert report.fix_embedding_ids == ["deadbeefdeadbeef"]

    def test_apply_embedding_fixes_deletes_orphans(self, setup):
        db, em, *_ = setup
        em.collection.records["deadbeefdeadbeef"] = hash_text("orphan")
        report = check_consistency(db, em, embedding_model="test-model")

        deleted = apply_embedding_fixes(em, report)

        assert deleted == 1
        assert "deadbeefdeadbeef" not in em.collection.records

    def test_clean_state_has_no_embedding_fixes(self, setup):
        db, em, *_ = setup

        report = check_consistency(db, em, embedding_model="test-model")

        assert report.fix_embedding_ids == []
        assert apply_embedding_fixes(em, ConsistencyReport()) == 0

    def test_stale_content_hash_is_an_error(self, setup):
        db, em, uid, _ = setup
        em.collection.records[uid] = hash_text("outdated text")

        report = check_consistency(db, em, embedding_model="test-model")

        assert any("stale" in issue.message.lower() for issue in report.errors)

    def test_embedding_model_mismatch_is_an_error(self, setup):
        db, em, *_ = setup
        db.set_embedding_model("other-model")

        report = check_consistency(db, em, embedding_model="test-model")

        assert any("model" in issue.message.lower() for issue in report.errors)


class TestCacheCoverage:
    def test_missing_cache_is_reported_but_not_fixed(self, connected_db):
        _add_paper(connected_db, "Uncached Paper")
        connected_db.set_embedding_model("test-model")
        em = FakeEmbeddingsManager({})

        report = check_consistency(connected_db, em, embedding_model="test-model")

        assert any(issue.category == "cache-coverage" for issue in report.issues)
        assert report.fix_cache_ids == []
        assert any("clustering cache populate" in (issue.fix or "") for issue in report.issues)


class TestCacheIntegrity:
    def test_duplicate_cache_rows_mark_older_for_fix(self, setup):
        db, em, uid, _ = setup
        _cache(db, "NeurIPS", 2025, [_point(uid)])

        report = check_consistency(db, em, embedding_model="test-model")

        duplicates = [issue for issue in report.issues if issue.category == "cache-duplicate"]
        assert duplicates
        assert len(report.fix_cache_ids) == 1

    def test_apply_cache_fixes_deletes_duplicates(self, setup):
        db, em, uid, _ = setup
        _cache(db, "NeurIPS", 2025, [_point(uid)])
        report = check_consistency(db, em, embedding_model="test-model")

        deleted = apply_cache_fixes(db, report)

        assert deleted == 1
        assert len(db.list_clustering_cache_entries()) == 1

    def test_cache_for_unknown_scope_is_stale(self, setup):
        db, em, *_ = setup
        _cache(db, "GhostConf", 1999, [])

        report = check_consistency(db, em, embedding_model="test-model")

        assert any(issue.category == "cache-stale" for issue in report.issues)
        assert report.fix_cache_ids

    def test_point_count_mismatch_is_an_error(self, setup):
        db, em, uid, _ = setup
        db.clear_clustering_cache()
        _cache(db, "NeurIPS", 2025, [_point(uid)], total=99)

        report = check_consistency(db, em, embedding_model="test-model")

        assert any(issue.category == "cache-integrity" for issue in report.errors)

    def test_cache_points_from_wrong_scope_is_an_error(self, setup):
        db, em, uid, _ = setup
        other_uid, _ = _add_paper(db, "Other Scope Paper", conference="ICLR", year=2024)
        db.clear_clustering_cache()
        _cache(db, "NeurIPS", 2025, [_point(other_uid)])

        report = check_consistency(db, em, embedding_model="test-model")

        assert any(issue.category == "cache-integrity" for issue in report.errors)

    def test_duplicate_presentations_do_not_break_cache_integrity(self, connected_db):
        connected_db.add_papers(
            [
                LightweightPaper(
                    title="Shared abstract",
                    authors=["Author"],
                    abstract="Same abstract",
                    session="Poster Session",
                    poster_position="P1",
                    location="Sydney",
                    year=2025,
                    conference="NeurIPS",
                    original_id="presentation-1",
                ),
                LightweightPaper(
                    title="Shared abstract",
                    authors=["Author"],
                    abstract="Same abstract",
                    session="Oral Session",
                    poster_position="O1",
                    location="Paris",
                    year=2025,
                    conference="NeurIPS",
                    original_id="presentation-2",
                ),
            ]
        )
        connected_db.set_embedding_model("test-model")
        uids = [paper["uid"] for paper in connected_db.get_paper()]
        doc = hash_text("Shared abstract\n\nSame abstract")
        em = FakeEmbeddingsManager({uid: doc for uid in uids})
        connected_db.clear_clustering_cache()
        _cache(connected_db, "NeurIPS", 2025, [_point(uids[0])])

        report = check_consistency(connected_db, em, embedding_model="test-model")

        assert not any(issue.category == "cache-integrity" for issue in report.errors)
        assert report.papers_checked == 1

    def test_collection_mismatch_is_a_warning(self, setup):
        db, em, uid, _ = setup
        db.clear_clustering_cache()
        _cache(db, "NeurIPS", 2025, [_point(uid)], collection="other")

        report = check_consistency(db, em, embedding_model="test-model")

        assert any(issue.category == "cache-collection" for issue in report.warnings)

    def test_stale_chroma_collection_is_reported(self, setup):
        db, em, *_ = setup
        em.client = FakeClient(["papers", "leftover"])

        report = check_consistency(db, em, embedding_model="test-model")

        assert any(issue.category == "collection" for issue in report.warnings)
        assert report.fix_cache_ids == []


class TestReportSerialization:
    def test_to_dict_round_trips(self, setup):
        db, em, *_ = setup
        report = check_consistency(db, em, embedding_model="test-model")

        payload = report.to_dict()

        assert json.loads(json.dumps(payload))["summary"]["scopes_checked"] == 1

    def test_issue_can_be_constructed(self):
        issue = ConsistencyIssue(category="x", severity="info", message="hello")
        assert issue.severity == "info"

    def test_report_counts(self):
        report = ConsistencyReport(scopes_checked=1, papers_checked=2, embeddings_checked=3)
        report.add(ConsistencyIssue(category="a", severity="error", message="e"))
        report.add(ConsistencyIssue(category="b", severity="warning", message="w"))

        assert len(report.errors) == 1
        assert len(report.warnings) == 1

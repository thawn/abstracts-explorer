"""Tests for the high-duplicate scope regeneration script."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from abstracts_explorer.config import get_config
from abstracts_explorer.plugin import LightweightPaper

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "regenerate_high_duplicate_scopes.py"
SPEC = importlib.util.spec_from_file_location("regenerate_high_duplicate_scopes", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
regen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(regen)


def _paper(title: str, year: int, session: str) -> LightweightPaper:
    return LightweightPaper(
        title=title,
        authors=["Author"],
        abstract=f"Abstract for {title}",
        session=session,
        poster_position="1",
        year=year,
        conference="ICML",
        original_id=f"{title}:{session}",
    )


def test_high_duplicate_scopes_selects_and_sorts(connected_db):
    connected_db.add_papers(
        [
            _paper("A", 2022, "Poster"),
            _paper("A", 2022, "Talk"),
            _paper("B", 2022, "Poster"),
            _paper("B", 2022, "Talk"),
            _paper("C", 2022, "Poster"),
            _paper("D", 2023, "Poster"),
            _paper("D", 2023, "Talk"),
        ]
    )

    scopes = regen.high_duplicate_scopes(connected_db, conference=None, year=None, threshold=2)

    assert scopes == [("ICML", 2022, 5, 2)]
    assert regen.high_duplicate_scopes(connected_db, conference=None, year=None, threshold=1) == [
        ("ICML", 2022, 5, 2),
        ("ICML", 2023, 2, 1),
    ]
    assert regen.high_duplicate_scopes(connected_db, conference="ICML", year=2023, threshold=1) == [
        ("ICML", 2023, 2, 1)
    ]
    assert regen.high_duplicate_scopes(connected_db, conference=None, year=None, threshold=99) == []


def _duplicate_papers() -> List[LightweightPaper]:
    return [
        _paper("A", 2022, "Poster"),
        _paper("A", 2022, "Talk"),
        _paper("B", 2022, "Poster"),
        _paper("B", 2022, "Talk"),
        _paper("D", 2023, "Poster"),
        _paper("D", 2023, "Talk"),
    ]


class _DummyEmbeddingsManager:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __enter__(self) -> "_DummyEmbeddingsManager":
        return self

    def __exit__(self, *args: Any) -> bool:
        return False


def _args(state_file: Path, **overrides: Any) -> argparse.Namespace:
    values: Dict[str, Any] = {
        "paper_db": None,
        "conference": None,
        "year": None,
        "collection": None,
        "threshold": 1,
        "max_scopes": None,
        "execute": True,
        "no_llm_labels": False,
        "show_progress": False,
        "state_file": str(state_file),
        "no_resume": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _run_meta(threshold: int = 1) -> Dict[str, Any]:
    config = get_config()
    return regen._state_meta(
        config.database_url,
        config.collection_name,
        config.embedding_model,
        threshold,
    )


def test_load_and_save_state_roundtrip(tmp_path):
    path = tmp_path / "state.json"
    state = {"meta": {"threshold": 1}, "completed": ["ICML::2023", "ICML::2022"]}

    regen.save_state(path, state)

    assert regen.load_state(path) == state
    assert not path.with_name(f"{path.name}.tmp").exists()


def test_load_state_missing_or_malformed(tmp_path):
    assert regen.load_state(tmp_path / "missing.json") == {}

    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    assert regen.load_state(bad) == {}

    wrong_shape = tmp_path / "wrong.json"
    wrong_shape.write_text('{"meta": [], "completed": "x"}', encoding="utf-8")
    assert regen.load_state(wrong_shape) == {"meta": {}, "completed": []}


def test_cleanup_scope_cache_keeps_newest_row(connected_db):
    results = {"statistics": {"n_clusters": 1}}
    params = {"pipeline_version": 1}

    def add(year: int, collection: str, embedding_model: str = "model") -> None:
        connected_db.save_clustering_cache(
            embedding_model=embedding_model,
            reduction_method="umap",
            n_components=2,
            clustering_method="agglomerative",
            results=results,
            n_clusters=1,
            clustering_params=params,
            conference="ICML",
            year=year,
            collection_name=collection,
        )

    add(2022, "papers")
    add(2022, "papers")
    add(2022, "other")
    add(2023, "papers")
    add(2022, "papers", embedding_model="other-model")

    removed = regen.cleanup_scope_cache(connected_db, "ICML", 2022, "model", "papers")

    assert removed == 1
    rows = connected_db.session.execute(
        regen.text("SELECT id, clustering_params FROM clustering_cache ORDER BY id")
    ).all()
    remaining = {int(row.id) for row in rows}
    assert remaining == {2, 3, 4, 5}
    params_by_id = {int(row.id): json.loads(row.clustering_params) for row in rows}
    assert params_by_id[2]["__collection__"] == "papers"


def test_main_skips_completed_scopes(tmp_path, connected_db, monkeypatch):
    connected_db.add_papers(_duplicate_papers())
    state_path = tmp_path / "state.json"
    regen.save_state(
        state_path,
        {"meta": _run_meta(), "completed": ["ICML::2022"]},
    )

    calls: List[Tuple[str, int]] = []

    def fake_compute(**kwargs: Any) -> Dict[str, Any]:
        calls.append((kwargs["conferences"][0], kwargs["years"][0]))
        return {"statistics": {"total_papers": 1, "n_clusters": 1}}

    monkeypatch.setattr(regen, "EmbeddingsManager", _DummyEmbeddingsManager)
    monkeypatch.setattr(regen, "compute_clusters_with_cache", fake_compute)
    monkeypatch.setattr(regen, "parse_args", lambda: _args(state_path))

    assert regen.main() == 0
    assert calls == [("ICML", 2023)]
    assert regen.load_state(state_path)["completed"] == ["ICML::2022", "ICML::2023"]

    calls.clear()
    assert regen.main() == 0
    assert calls == []


def test_main_no_resume_recomputes_all(tmp_path, connected_db, monkeypatch):
    connected_db.add_papers(_duplicate_papers())
    state_path = tmp_path / "state.json"
    regen.save_state(
        state_path,
        {"meta": _run_meta(), "completed": ["ICML::2022", "ICML::2023"]},
    )

    calls: List[Tuple[str, int]] = []

    def fake_compute(**kwargs: Any) -> Dict[str, Any]:
        calls.append((kwargs["conferences"][0], kwargs["years"][0]))
        return {"statistics": {"total_papers": 1, "n_clusters": 1}}

    monkeypatch.setattr(regen, "EmbeddingsManager", _DummyEmbeddingsManager)
    monkeypatch.setattr(regen, "compute_clusters_with_cache", fake_compute)
    monkeypatch.setattr(regen, "parse_args", lambda: _args(state_path, no_resume=True))

    assert regen.main() == 0
    assert sorted(calls) == [("ICML", 2022), ("ICML", 2023)]


def test_main_meta_mismatch_ignores_state(tmp_path, connected_db, monkeypatch):
    connected_db.add_papers(_duplicate_papers())
    state_path = tmp_path / "state.json"
    regen.save_state(
        state_path,
        {"meta": {**_run_meta(), "threshold": 999}, "completed": ["ICML::2022", "ICML::2023"]},
    )

    calls: List[Tuple[str, int]] = []

    def fake_compute(**kwargs: Any) -> Dict[str, Any]:
        calls.append((kwargs["conferences"][0], kwargs["years"][0]))
        return {"statistics": {"total_papers": 1, "n_clusters": 1}}

    monkeypatch.setattr(regen, "EmbeddingsManager", _DummyEmbeddingsManager)
    monkeypatch.setattr(regen, "compute_clusters_with_cache", fake_compute)
    monkeypatch.setattr(regen, "parse_args", lambda: _args(state_path))

    assert regen.main() == 0
    assert sorted(calls) == [("ICML", 2022), ("ICML", 2023)]


def test_main_failure_is_not_recorded(tmp_path, connected_db, monkeypatch):
    connected_db.add_papers(_duplicate_papers())
    state_path = tmp_path / "state.json"

    def failing_compute(**kwargs: Any) -> Dict[str, Any]:
        raise RuntimeError("boom")

    monkeypatch.setattr(regen, "EmbeddingsManager", _DummyEmbeddingsManager)
    monkeypatch.setattr(regen, "compute_clusters_with_cache", failing_compute)
    monkeypatch.setattr(regen, "parse_args", lambda: _args(state_path))

    assert regen.main() == 1
    assert not state_path.exists()

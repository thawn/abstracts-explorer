"""
Tests for the recovered-scopes clustering script.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "recluster_recovered_scopes.py"


@pytest.fixture(scope="module")
def script():
    """Load the recluster script as a module."""
    spec = importlib.util.spec_from_file_location("recluster_recovered_scopes", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runtime(script, monkeypatch):
    """Patch the script's runtime dependencies and return the mocks."""
    config = Mock(collection_name="papers", embedding_model="test-model")
    monkeypatch.setattr(script, "get_config", Mock(return_value=config))

    database = Mock()
    database_cm = MagicMock()
    database_cm.__enter__.return_value = database
    database_factory = Mock(return_value=database_cm)
    monkeypatch.setattr(script, "DatabaseManager", database_factory)

    embeddings_manager = Mock()
    embeddings_cm = MagicMock()
    embeddings_cm.__enter__.return_value = embeddings_manager
    embeddings_factory = Mock(return_value=embeddings_cm)
    monkeypatch.setattr(script, "EmbeddingsManager", embeddings_factory)

    compute = Mock(return_value={"statistics": {"total_papers": 10, "n_clusters": 2}})
    monkeypatch.setattr(script, "compute_clusters_with_cache", compute)

    return types.SimpleNamespace(
        database=database,
        embeddings_manager=embeddings_manager,
        embeddings_factory=embeddings_factory,
        compute=compute,
        config=config,
    )


class TestAffectedScopes:
    """Tests for the hard-coded affected scope list."""

    def test_contains_all_recovered_scopes(self, script):
        """The 31 cached scopes affected by the embedding recovery are listed."""
        scopes = list(script.AFFECTED_SCOPES)

        assert len(scopes) == 31
        assert len(set(scopes)) == 31
        for year in range(2016, 2022):
            assert ("AGU", year) in scopes
        for year in range(2013, 2020):
            assert ("ICML", year) in scopes
        for year in (*range(1987, 2004), 2019):
            assert ("NeurIPS", year) in scopes

    def test_excludes_uncached_scopes(self, script):
        """Scopes without a cache and unaffected years are not included."""
        scopes = set(script.AFFECTED_SCOPES)

        assert ("AGU", 2014) not in scopes
        assert ("AGU", 2015) not in scopes
        assert ("AGU", 2023) not in scopes
        assert ("ICLR", 2017) not in scopes
        assert ("ICLR", 2018) not in scopes
        assert ("NeurIPS", 2004) not in scopes
        assert ("NeurIPS", 2018) not in scopes


class TestDryRun:
    """Tests for ``--dry-run``."""

    def test_lists_scopes_without_connecting(self, script, monkeypatch, capsys):
        """Dry run prints every scope and does not touch the database."""
        config = Mock(collection_name="papers", embedding_model="test-model")
        monkeypatch.setattr(script, "get_config", Mock(return_value=config))
        db_factory = Mock(side_effect=AssertionError("should not connect"))
        monkeypatch.setattr(script, "DatabaseManager", db_factory)
        monkeypatch.setattr(script, "EmbeddingsManager", db_factory)

        assert script.main(["--dry-run"]) == 0

        output = capsys.readouterr().out
        assert "31" in output
        assert "AGU 2016" in output
        assert "NeurIPS 2019" in output
        db_factory.assert_not_called()


class TestRecluster:
    """Tests for the regeneration loop."""

    def test_deletes_and_recomputes_each_scope(self, script, runtime):
        """Each affected scope is invalidated and recomputed exactly once."""
        assert script.main([]) == 0

        database = runtime.database
        compute = runtime.compute
        assert database.delete_clustering_cache_by_conference_year.call_count == 31
        assert compute.call_count == 31

        deleted_scopes = [call.args for call in database.delete_clustering_cache_by_conference_year.call_args_list]
        computed_scopes = [(call.kwargs["conferences"], call.kwargs["years"]) for call in compute.call_args_list]
        expected = [([scope[0]], [scope[1]]) for scope in script.AFFECTED_SCOPES]
        assert deleted_scopes == list(script.AFFECTED_SCOPES)
        assert computed_scopes == expected

    def test_passes_llm_label_setting(self, script, runtime):
        """``--no-llm-labels`` disables LLM labels for every recomputation."""
        assert script.main(["--no-llm-labels"]) == 0

        for call in runtime.compute.call_args_list:
            assert call.kwargs["use_llm_labels"] is False
            assert call.kwargs["force"] is True

    def test_disables_sdk_retries_and_forwards_rate_limit(self, script, runtime):
        """Labeling owns retries and the request rate is configurable."""
        assert script.main(["--requests-per-minute", "42"]) == 0

        kwargs = runtime.embeddings_factory.call_args.kwargs
        assert kwargs["requests_per_minute"] == 42
        assert kwargs["collection_name"] == "papers"

    def test_continues_after_failure_and_returns_error(self, script, runtime, capsys):
        """A failing scope does not stop the remaining scopes."""

        def compute_with_failure(*args, **kwargs):
            if kwargs["conferences"] == ["AGU"] and kwargs["years"] == [2016]:
                raise RuntimeError("boom")
            return {"statistics": {"total_papers": 10, "n_clusters": 2}}

        runtime.compute.side_effect = compute_with_failure

        assert script.main([]) == 1
        assert runtime.compute.call_count == 31
        assert runtime.database.delete_clustering_cache_by_conference_year.call_count == 31

        errors = capsys.readouterr().err
        assert "AGU 2016: RuntimeError: boom" in errors

    def test_recluster_scope_returns_results(self, script, monkeypatch):
        """The helper deletes the stale row before returning new results."""
        database = Mock()
        embeddings_manager = Mock()
        results = {"statistics": {"n_clusters": 3}}
        compute = Mock(return_value=results)
        monkeypatch.setattr(script, "compute_clusters_with_cache", compute)

        got = script.recluster_scope(
            embeddings_manager,
            database,
            "test-model",
            "ICML",
            2015,
            use_llm_labels=False,
        )

        assert got is results
        database.delete_clustering_cache_by_conference_year.assert_called_once_with("ICML", 2015)
        kwargs = compute.call_args.kwargs
        assert kwargs["conferences"] == ["ICML"]
        assert kwargs["years"] == [2015]
        assert kwargs["use_llm_labels"] is False

"""
Tests for the ``check-consistency`` CLI command.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from abstracts_explorer.cli import consistency as consistency_cli
from abstracts_explorer.consistency import ConsistencyIssue, ConsistencyReport


@pytest.fixture
def runtime(monkeypatch):
    config = SimpleNamespace(embedding_model="test-model", collection_name="papers")
    monkeypatch.setattr(consistency_cli, "get_config", lambda: config)

    class FakeContextManager:
        def __init__(self, value):
            self._value = value

        def __enter__(self):
            return self._value

        def __exit__(self, *exc):
            return False

    database = SimpleNamespace(create_tables=lambda: None)
    monkeypatch.setattr(consistency_cli, "DatabaseManager", lambda: FakeContextManager(database))

    embeddings = SimpleNamespace()
    monkeypatch.setattr(consistency_cli, "EmbeddingsManager", lambda: FakeContextManager(embeddings))

    report = ConsistencyReport(scopes_checked=2, papers_checked=5, embeddings_checked=5)

    def check(*a, **k):
        return report

    monkeypatch.setattr(consistency_cli, "check_consistency", check)
    monkeypatch.setattr(consistency_cli, "_resolve_conference_arg", lambda value: value)

    def fix(db, rep):
        return 3

    monkeypatch.setattr(consistency_cli, "apply_cache_fixes", fix)

    def fix_embeddings(em, rep):
        return 2

    monkeypatch.setattr(consistency_cli, "apply_embedding_fixes", fix_embeddings)

    return SimpleNamespace(report=report, config=config, database=database)


def _args(**overrides):
    base = {"conference": None, "year": None, "json": False, "fix": False}
    base.update(overrides)
    return SimpleNamespace(**base)


class TestConsistencyCommand:
    def test_clean_run_returns_zero(self, runtime, capsys):
        assert consistency_cli.consistency_check_command(_args()) == 0
        output = capsys.readouterr().out
        assert "Consistency" in output
        assert "0 error" in output.lower()

    def test_errors_return_one(self, runtime, capsys):
        runtime.report.add(ConsistencyIssue(category="embeddings", severity="error", message="boom"))

        assert consistency_cli.consistency_check_command(_args()) == 1
        assert "boom" in capsys.readouterr().out

    def test_json_output_is_machine_readable(self, runtime, capsys):
        runtime.report.add(
            ConsistencyIssue(
                category="cache-coverage",
                severity="info",
                message="missing cache",
                scope="NeurIPS 2025",
            )
        )

        assert consistency_cli.consistency_check_command(_args(json=True)) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["summary"]["scopes_checked"] == 2
        assert payload["issues"][0]["scope"] == "NeurIPS 2025"

    def test_fix_invokes_safe_fixes(self, runtime, capsys):
        runtime.report.fix_cache_ids = [1, 2, 3]

        assert consistency_cli.consistency_check_command(_args(fix=True)) == 0
        assert "3" in capsys.readouterr().out

    def test_fix_json_reports_deleted(self, runtime, capsys):
        runtime.report.fix_cache_ids = [1]

        assert consistency_cli.consistency_check_command(_args(fix=True, json=True)) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["fix_deleted"] == 3

    def test_fix_deletes_orphan_embeddings(self, runtime, capsys):
        runtime.report.fix_embedding_ids = ["orphan"]

        assert consistency_cli.consistency_check_command(_args(fix=True)) == 0
        assert "orphan embedding" in capsys.readouterr().out

    def test_fix_json_reports_deleted_embeddings(self, runtime, capsys):
        runtime.report.fix_embedding_ids = ["orphan"]

        assert consistency_cli.consistency_check_command(_args(fix=True, json=True)) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["fix_embeddings_deleted"] == 2

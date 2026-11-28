"""Tests for the web request profiling workload script."""

import argparse
import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "profile_web_requests.py"
_SPEC = importlib.util.spec_from_file_location("profile_web_requests", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
profile_web_requests = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = profile_web_requests
_SPEC.loader.exec_module(profile_web_requests)


def _namespace(**overrides):
    defaults = {
        "base_url": "http://localhost:5001",
        "runs": 1,
        "timeout": 10,
        "json": None,
        "insecure": False,
        "scenarios": None,
        "conference": "NeurIPS",
        "year": 2025,
        "skip_chat": False,
        "skip_clustering": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_percentile_interpolates():
    assert profile_web_requests._percentile([10, 20, 30], 0.5) == 20
    assert profile_web_requests._percentile([10, 20], 0.95) == 19.5
    assert profile_web_requests._percentile([], 0.5) == 0


def test_build_scenarios_includes_core_and_dynamic_scenarios():
    context = {"uids": ["paper-1", "paper-2"], "priorities": {"paper-1": {"priority": 5}}}
    scenarios = profile_web_requests.build_scenarios(_namespace(), context)
    names = {scenario.name for scenario in scenarios}

    assert "page.index" in names
    assert "search.learning" in names
    assert "paper.detail" in names
    assert "papers.batch" in names
    assert "clusters.compute" in names
    assert "chat.search_papers" in names
    assert any(scenario.name == "paper.detail" and scenario.path.endswith("paper-1") for scenario in scenarios)


def test_build_scenarios_respects_skip_and_filter_flags():
    context = {"uids": [], "priorities": {}}
    scenarios = profile_web_requests.build_scenarios(
        _namespace(skip_chat=True, skip_clustering=True, scenarios="search"),
        context,
    )
    names = [scenario.name for scenario in scenarios]

    assert names
    assert all("search" in name for name in names)
    assert not any(name.startswith("chat.") for name in names)
    assert not any(name.startswith("clusters.") for name in names)


def test_summarize_groups_by_scenario_and_sorts_by_median():
    Result = profile_web_requests.Result
    results = [
        Result("fast", 0, "GET", "/fast", 200, 10.0),
        Result("fast", 1, "GET", "/fast", 200, 20.0),
        Result("slow", 0, "GET", "/slow", 200, 100.0),
        Result("slow", 1, "GET", "/slow", None, 200.0, error="timeout"),
    ]

    summary = profile_web_requests.summarize(results)

    assert [row["scenario"] for row in summary] == ["slow", "fast"]
    slow = summary[0]
    assert slow["runs"] == 2
    assert slow["errors"] == 1
    assert slow["elapsed_ms"]["median"] == 100.0
    fast = summary[1]
    assert fast["successes"] == 2

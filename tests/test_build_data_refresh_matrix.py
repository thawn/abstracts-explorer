"""Tests for ``.github/scripts/build-data-refresh-matrix.py``."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "build-data-refresh-matrix.py"


def _load_module():
    """Load the hyphenated workflow script as an importable module."""
    spec = importlib.util.spec_from_file_location("build_data_refresh_matrix", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def matrix_module():
    """Provide a freshly imported copy of the matrix builder script."""
    return _load_module()


class FakePlugin:
    """Minimal stand-in for a downloader plugin."""

    def __init__(self, name, conference, supported_years=(2026,), parameters=None):
        self.plugin_name = name
        self.conference_name = conference
        self.supported_years = set(supported_years)
        self._parameters = parameters or {}

    def get_metadata(self):
        """Return plugin metadata in the shape expected by the matrix builder."""
        return {"parameters": self._parameters}


class FakeRegistryClient:
    """Minimal registry client fake for the matrix builder."""

    def __init__(self, tags, artifact_years=None):
        self._tags = tags
        self._artifact_years = artifact_years or {}

    def list_tags(self):
        """Return the configured tags."""
        return self._tags

    def get_artifact_info(self, tag):
        """Return a manifest annotation with the configured years."""
        years = self._artifact_years.get(tag, [])
        return {"annotations": {"com.abstracts-explorer.years": ",".join(str(y) for y in years)}}


def _run_main(
    monkeypatch,
    tmp_path,
    module,
    *,
    plugins,
    tags,
    model="model",
    target_year="2026",
    artifact_years=None,
):
    """Run ``main`` with fakes and return the parsed matrix and year."""
    output = tmp_path / "output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("TARGET_YEAR", target_year)
    monkeypatch.setenv("CONFERENCE_FILTER", "")
    monkeypatch.setattr(
        module,
        "get_config",
        lambda: SimpleNamespace(
            registry_repository="ghcr.io/thawn/abstracts-data",
            embedding_model=model,
            github_token="tok",
        ),
    )
    monkeypatch.setattr(module, "get_all_plugins", lambda: plugins)
    monkeypatch.setattr(
        module,
        "RegistryClient",
        lambda repository, token: FakeRegistryClient(tags, artifact_years),
    )

    assert module.main() == 0

    values = {}
    for line in output.read_text().splitlines():
        key, value = line.split("=", 1)
        values[key] = value
    return json.loads(values["matrix"]), int(values["year"])


def test_matching_tags_prefers_conference_level(matrix_module):
    tags = ["neurips-2024_model_0.3.0", "neurips_model_0.4.0"]
    result = matrix_module._matching_tags(tags, "neurips", "model", 2026)
    assert result == ["neurips_model_0.4.0"]


def test_matching_tags_collects_years_without_conference_tag(matrix_module):
    tags = [
        "neurips-2024_model_0.3.0",
        "neurips-2025_model_0.3.1",
        "neurips-2027_model_0.3.2",
    ]
    result = matrix_module._matching_tags(tags, "neurips", "model", 2026)
    assert result == ["neurips-2024_model_0.3.0", "neurips-2025_model_0.3.1"]


def test_matching_tags_returns_empty_for_new_conference(matrix_module):
    tags = ["iclr-2026_model_0.3.0"]
    assert matrix_module._matching_tags(tags, "neurips", "model", 2026) == []


def test_matching_tags_ignores_other_embedding_models(matrix_module):
    tags = ["neurips_model_0.4.0"]
    assert matrix_module._matching_tags(tags, "neurips", "other-model", 2026) == []


def test_main_marks_new_and_existing_conferences(matrix_module, monkeypatch, tmp_path):
    plugins = [FakePlugin("neurips", "neurips"), FakePlugin("iclr", "iclr")]
    matrix, year = _run_main(
        monkeypatch,
        tmp_path,
        matrix_module,
        plugins=plugins,
        tags=["neurips_model_0.4.0"],
    )

    entries = {entry["conference"]: entry for entry in matrix["include"]}
    assert year == 2026
    assert entries["neurips"]["is_new"] is False
    assert entries["neurips"]["registry_tags"] == ["neurips_model_0.4.0"]
    assert entries["iclr"]["is_new"] is True
    assert entries["iclr"]["registry_tags"] == []


def test_main_skips_plugins_requiring_input_file(matrix_module, monkeypatch, tmp_path):
    plugins = [
        FakePlugin("chi", "chi", parameters={"input_path": {"type": "string"}}),
        FakePlugin("iclr", "iclr"),
    ]
    matrix, _ = _run_main(monkeypatch, tmp_path, matrix_module, plugins=plugins, tags=[])

    assert [entry["plugin"] for entry in matrix["include"]] == ["iclr"]


def test_main_errors_when_no_supported_plugins(matrix_module, monkeypatch, tmp_path):
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output.txt"))
    monkeypatch.setenv("TARGET_YEAR", "2026")
    monkeypatch.setenv("CONFERENCE_FILTER", "")
    monkeypatch.setattr(
        matrix_module,
        "get_config",
        lambda: SimpleNamespace(
            registry_repository="ghcr.io/thawn/abstracts-data",
            embedding_model="model",
            github_token="tok",
        ),
    )
    monkeypatch.setattr(matrix_module, "get_all_plugins", lambda: [])
    monkeypatch.setattr(
        matrix_module,
        "RegistryClient",
        lambda repository, token: SimpleNamespace(list_tags=lambda: []),
    )

    assert matrix_module.main() == 1


def test_target_year_present_in_summary(matrix_module, monkeypatch, tmp_path):
    matrix, _ = _run_main(
        monkeypatch,
        tmp_path,
        matrix_module,
        plugins=[FakePlugin("neurips", "neurips")],
        tags=["neurips_model_0.4.0"],
        artifact_years={"neurips_model_0.4.0": [2024, 2025, 2026]},
    )
    entry = matrix["include"][0]
    assert entry["target_year_present"] is True
    assert entry["needs_update"] is False


def test_target_year_missing_marks_needs_update(matrix_module, monkeypatch, tmp_path):
    matrix, _ = _run_main(
        monkeypatch,
        tmp_path,
        matrix_module,
        plugins=[FakePlugin("neurips", "neurips")],
        tags=["neurips_model_0.4.0"],
        artifact_years={"neurips_model_0.4.0": [2024, 2025]},
    )
    entry = matrix["include"][0]
    assert entry["is_new"] is False
    assert entry["target_year_present"] is False
    assert entry["needs_update"] is True


def test_target_year_present_via_year_tag(matrix_module, monkeypatch, tmp_path):
    matrix, _ = _run_main(
        monkeypatch,
        tmp_path,
        matrix_module,
        plugins=[FakePlugin("neurips", "neurips", supported_years=(2025,))],
        tags=["neurips-2024_model_0.4.0", "neurips-2025_model_0.4.0"],
        target_year="2025",
    )
    entry = matrix["include"][0]
    assert entry["target_year_present"] is True
    assert entry["needs_update"] is False


def test_new_conference_needs_update(matrix_module, monkeypatch, tmp_path):
    matrix, _ = _run_main(
        monkeypatch,
        tmp_path,
        matrix_module,
        plugins=[FakePlugin("iclr", "iclr")],
        tags=[],
    )
    entry = matrix["include"][0]
    assert entry["is_new"] is True
    assert entry["target_year_present"] is False
    assert entry["needs_update"] is True

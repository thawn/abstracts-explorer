"""Tests for the ChromaDB vacuum helper script."""

from __future__ import annotations

import subprocess
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "vacuum-chromadb.sh"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT_PATH), *args],
        capture_output=True,
        text=True,
        check=True,
    )


def test_shell_syntax() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT_PATH)], check=True)


def test_help_does_not_require_podman() -> None:
    result = _run("--help")

    assert "Usage: vacuum-chromadb.sh" in result.stdout
    assert "--dry-run" in result.stdout


def test_dry_run_prints_stop_vacuum_and_restart_commands() -> None:
    result = _run(
        "--dry-run",
        "--yes",
        "--volume",
        "test-volume",
        "--path",
        "/data/chroma",
    )

    assert "systemctl --user stop abstracts-explorer abstracts-chromadb" in result.stdout
    assert "podman run --rm --entrypoint chroma" in result.stdout
    assert "test-volume:/data" in result.stdout
    assert "vacuum --path /data/chroma --force" in result.stdout
    assert "systemctl --user start abstracts-explorer abstracts-chromadb" in result.stdout

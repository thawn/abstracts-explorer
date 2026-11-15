"""Tests for the ChromaDB vacuum helper script."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "vacuum-chromadb.sh"


def _bash_path(path: Path) -> str:
    """Return a path that bash can use on the current platform."""
    if os.name == "nt":
        try:
            return subprocess.check_output(["cygpath", "-u", str(path)], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return path.as_posix()


def _script_arg() -> str:
    """Return a bash-friendly path, including on Windows runners."""
    return _bash_path(SCRIPT_PATH)


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", _script_arg(), *args],
        capture_output=True,
        text=True,
        check=True,
    )


def _fake_runtime(tmp_path: Path, vacuum_output: str) -> tuple[subprocess.CompletedProcess[str], str, str]:
    podman_log = tmp_path / "podman.log"
    systemctl_log = tmp_path / "systemctl.log"
    fake_env = tmp_path / "bash_env.sh"
    fake_env.write_text(
        f"""podman() {{
    printf '%s\\n' "$*" >> "{_bash_path(podman_log)}"
    case "$*" in
        "volume inspect"*) return 0 ;;
        *"--entrypoint sh"*) return 0 ;;
        *" vacuum "*) printf '%s\\n' "{vacuum_output}"; return 0 ;;
        *) return 0 ;;
    esac
}}

systemctl() {{
    printf '%s\\n' "$*" >> "{_bash_path(systemctl_log)}"
    return 0
}}
""",
        encoding="utf-8",
        newline="\n",
    )

    env = os.environ.copy()
    env["BASH_ENV"] = _bash_path(fake_env)
    result = subprocess.run(
        ["bash", _script_arg(), "--yes", "--volume", "test-volume", "--path", "/data"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    return result, podman_log.read_text(encoding="utf-8"), systemctl_log.read_text(encoding="utf-8")


def test_shell_syntax() -> None:
    subprocess.run(["bash", "-n", _script_arg()], check=True)


def test_help_does_not_require_podman() -> None:
    result = _run("--help")

    assert "Usage: vacuum-chromadb.sh" in result.stdout
    assert "--dry-run" in result.stdout
    assert "(default: /data)" in result.stdout


def test_dry_run_prints_stop_vacuum_and_restart_commands() -> None:
    result = _run(
        "--dry-run",
        "--yes",
        "--volume",
        "test-volume",
        "--path",
        "/data",
    )

    assert "systemctl --user stop abstracts-explorer abstracts-chromadb" in result.stdout
    assert "podman run --rm --entrypoint chroma" in result.stdout
    assert "test-volume:/data" in result.stdout
    assert "vacuum --path /data --force" in result.stdout
    assert "systemctl --user start abstracts-explorer abstracts-chromadb" in result.stdout


def test_success_requires_vacuum_complete_marker(tmp_path: Path) -> None:
    result, podman_log, systemctl_log = _fake_runtime(tmp_path, "Vacuum complete!")

    assert result.returncode == 0
    assert "ChromaDB vacuum complete" in result.stdout
    assert "vacuum --path /data --force" in podman_log
    assert "start abstracts-explorer abstracts-chromadb" in systemctl_log


def test_cli_failure_is_not_reported_as_success(tmp_path: Path) -> None:
    result, podman_log, systemctl_log = _fake_runtime(tmp_path, "Failed to vacuum Chroma")

    assert result.returncode == 1
    assert "ChromaDB vacuum failed" in result.stderr
    assert "ChromaDB vacuum complete" not in result.stdout
    assert "vacuum --path /data --force" in podman_log
    assert "start abstracts-explorer abstracts-chromadb" in systemctl_log

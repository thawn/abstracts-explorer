"""
Download persistence helpers.

Shared helpers for per-year JSON caches, failure sidecars, and resumable
checkpoints used by conference downloaders.
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def get_failed_downloads_path(output_path: str) -> str:
    """
    Return the sidecar path used to record failed download URLs.

    Parameters
    ----------
    output_path : str
        Path to the main lightweight JSON cache.

    Returns
    -------
    str
        Sibling path ending in ``.failed.json``.
    """
    return str(Path(output_path).with_suffix(".failed.json"))


class FailureSidecar:
    """
    Read/write the ``.failed.json`` sidecar recording failed download pages.

    The sidecar stores ``{"conference": ..., "year": ..., "failed": [...]}``
    and is ignored (treated as empty) when absent, malformed, or belonging to
    a different conference/year.
    """

    def __init__(self, output_path: Optional[str], conference: str, year: int) -> None:
        """
        Initialise the sidecar for a per-year cache file.

        Parameters
        ----------
        output_path : str, optional
            Path to the main lightweight JSON cache. When ``None``, loading
            returns an empty list and saving is a no-op.
        conference : str
            Conference name the sidecar must match.
        year : int
            Year the sidecar must match.
        """
        self.output_path = output_path
        self.conference = conference
        self.year = year

    @property
    def path(self) -> Optional[Path]:
        """Return the sidecar path, or ``None`` without an output path."""
        if not self.output_path:
            return None
        return Path(get_failed_downloads_path(self.output_path))

    def load(self) -> List[Dict[str, Any]]:
        """
        Load recorded failed entries, ignoring malformed or mismatched files.

        Returns
        -------
        list of dict
            Entries with a non-empty ``url`` key.
        """
        path = self.path
        if path is None or not path.exists():
            return []
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read failed-download list %s: %s", path, exc)
            return []
        if not isinstance(data, dict):
            logger.warning("Ignoring malformed failed-download list: %s", path)
            return []
        if data.get("conference") != self.conference or data.get("year") != self.year:
            logger.warning("Ignoring mismatched failed-download list: %s", path)
            return []
        failures = data.get("failed", [])
        if not isinstance(failures, list):
            logger.warning("Ignoring malformed failed-download list: %s", path)
            return []
        return [
            entry
            for entry in failures
            if isinstance(entry, dict) and isinstance(entry.get("url"), str) and entry["url"]
        ]

    def save(self, failures: List[Dict[str, Any]]) -> None:
        """
        Save failed entries, or remove the sidecar when there are none.

        Parameters
        ----------
        failures : list of dict
            Failed entries, each containing a non-empty ``url``.
        """
        path = self.path
        if path is None:
            return
        if not failures:
            path.unlink(missing_ok=True)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"conference": self.conference, "year": self.year, "failed": failures}, f, indent=2)
        logger.warning(
            "Recorded %d failed page(s) in %s; they will be retried automatically on the next run. "
            "Delete this file to skip retrying them.",
            len(failures),
            path,
        )


def clear_directory(directory: Path) -> None:
    """
    Remove a checkpoint directory if it exists.

    Parameters
    ----------
    directory : Path
        Directory to remove (recursively).
    """
    shutil.rmtree(directory, ignore_errors=True)


def load_json(path: Optional[Path]) -> Any:
    """
    Load a JSON file, returning ``None`` when it is missing or unusable.

    Parameters
    ----------
    path : Path, optional
        File to load.

    Returns
    -------
    Any
        Parsed JSON value, or ``None``.
    """
    if path is None or not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def save_json(path: Path, data: Any) -> None:
    """
    Write JSON atomically so an interrupted write cannot corrupt it.

    Parameters
    ----------
    path : Path
        Destination file.
    data : Any
        JSON-serializable value.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, separators=(",", ":"))
    os.replace(temporary, path)


def load_jsonl(path: Optional[Path]) -> List[Dict[str, Any]]:
    """
    Load append-only JSONL records, ignoring a corrupt trailing line.

    Parameters
    ----------
    path : Path, optional
        JSONL file to load.

    Returns
    -------
    list of dict
        Parsed records (non-object lines are skipped).
    """
    records: List[Dict[str, Any]] = []
    if path is None or not path.exists():
        return records
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
    except OSError:
        return records
    return records


def append_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    """
    Append records to a JSONL checkpoint.

    Parameters
    ----------
    path : Path
        JSONL file to append to.
    records : list of dict
        Records to append.
    """
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")

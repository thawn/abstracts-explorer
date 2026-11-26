"""
``check-consistency`` command.

Compares the paper database, ChromaDB embeddings, and clustering cache and
optionally removes duplicate or stale clustering cache rows.
"""

import argparse
import json
import logging
from typing import Optional

from tqdm import tqdm

from abstracts_explorer.cli._common import _print_header, _resolve_conference_arg
from abstracts_explorer.config import get_config
from abstracts_explorer.consistency import apply_cache_fixes, check_consistency
from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.embeddings import EmbeddingsManager

logger = logging.getLogger(__name__)

_SEVERITY_ICONS = {"error": "❌", "warning": "⚠️ ", "info": "ℹ️ "}


class _ProgressBar:
    """
    Render a single tqdm bar that advances through consecutive check stages.

    Attributes
    ----------
    _stage : str or None
        Current stage label, used to refresh the bar description.
    _bar : tqdm or None
        Active progress bar, if any.
    """

    def __init__(self) -> None:
        self._stage: Optional[str] = None
        self._bar: Optional[tqdm] = None

    def update(self, stage: str, completed: int, total: int) -> None:
        """
        Advance the progress bar, switching stages when *stage* changes.

        Parameters
        ----------
        stage : str
            Human-readable label for the current stage.
        completed : int
            Number of completed units in the stage.
        total : int
            Total number of units in the stage.
        """
        if total <= 0:
            return
        if stage != self._stage or self._bar is None:
            self._close()
            self._stage = stage
            self._bar = tqdm(total=total, desc=stage, unit="item", leave=False)
        self._bar.total = total
        self._bar.n = min(completed, total)
        self._bar.refresh()

    def _close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def close(self) -> None:
        """Close the active progress bar, if any."""
        self._close()
        self._stage = None


def _print_issue(severity: str, scope: Optional[str], message: str, fix: Optional[str]) -> None:
    """Print one finding in a human-readable form."""
    icon = _SEVERITY_ICONS.get(severity, "•")
    prefix = f"[{scope}] " if scope else ""
    print(f"{icon} {prefix}{message}")
    if fix:
        print(f"   → {fix}")


def consistency_check_command(args: argparse.Namespace) -> int:
    """
    Run the cross-database consistency check.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments with ``conference``, ``year``, ``json``, and ``fix``.

    Returns
    -------
    int
        ``0`` when no errors remain, ``1`` when errors were found.
    """
    config = get_config()
    conference = _resolve_conference_arg(getattr(args, "conference", None))
    year = getattr(args, "year", None)
    as_json = bool(getattr(args, "json", False))
    should_fix = bool(getattr(args, "fix", False))

    progress = None if as_json else _ProgressBar()
    if not as_json:
        print("🔍 Scanning paper database, embeddings, and clustering cache...")

    try:
        with DatabaseManager() as db, EmbeddingsManager() as em:
            db.create_tables()
            report = check_consistency(
                db,
                em,
                embedding_model=config.embedding_model,
                collection_name=config.collection_name,
                conference=conference,
                year=year,
                progress=progress.update if progress is not None else None,
            )

            deleted = 0
            if should_fix and report.fix_cache_ids:
                if not as_json:
                    print(f"🧹 Removing {len(report.fix_cache_ids)} stale/duplicate cache entry/entries...")
                deleted = apply_cache_fixes(db, report)
    finally:
        if progress is not None:
            progress.close()

    if as_json:
        payload = report.to_dict()
        if should_fix:
            payload["fix_deleted"] = deleted
        print(json.dumps(payload, indent=2))
        return 1 if report.errors else 0

    _print_header(
        "Check Consistency",
        f"Embedding model: {config.embedding_model}",
        f"Collection: {config.collection_name}",
        f"Conference: {conference}" if conference else None,
        f"Year: {year}" if year is not None else None,
    )

    print(
        f"Scanned {report.papers_checked} paper(s) and {report.embeddings_checked} "
        f"embedding(s) across {report.scopes_checked} scope(s)."
    )

    if not report.issues:
        print("✅ No inconsistencies found.")
    for issue in report.errors + report.warnings + report.infos:
        _print_issue(issue.severity, issue.scope, issue.message, issue.fix)

    if should_fix:
        if deleted:
            print(f"\n🧹 Deleted {deleted} duplicate/stale clustering cache entry/entries.")
        else:
            print("\n🧹 No clustering cache entries needed fixing.")

    print(
        f"\n{len(report.errors)} error(s), {len(report.warnings)} warning(s), "
        f"{len(report.infos)} info item(s) across {report.scopes_checked} scope(s)."
    )
    return 1 if report.errors else 0

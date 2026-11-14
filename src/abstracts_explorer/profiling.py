"""
SQL profiling utilities
=======================

Lightweight helpers for measuring database access patterns.  The profiler
attaches SQLAlchemy event listeners at the :class:`~sqlalchemy.engine.Engine`
class level, so it captures statements from every ``DatabaseManager`` instance
in the process (including managers created lazily inside MCP tools).

The main entry point is :class:`SqlProfiler`::

    from abstracts_explorer.profiling import SqlProfiler

    with SqlProfiler() as profiler:
        with profiler.case("search"):
            db.search_papers(keyword="transformers")

    for case in profiler.aggregated():
        print(case.label, case.wall_ms_mean, case.sql_ms_mean)

Profiling is read-only and does not modify the database.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import math
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sqlalchemy import event
from sqlalchemy.engine import Engine

__all__ = [
    "AggregatedCase",
    "CaseStats",
    "QueryRecord",
    "SqlProfiler",
    "aggregate_cases",
    "explain_statement",
    "normalize_sql",
    "suppress_profiling",
]


_SUPPRESSED: contextvars.ContextVar[bool] = contextvars.ContextVar("profiling_suppressed", default=False)

_COMMENT_RE = re.compile(r"/\*.*?\*/|--[^\n]*", re.DOTALL)
_STRING_RE = re.compile(r"'(?:''|[^'])*'")
_PYFORMAT_RE = re.compile(r"%\([A-Za-z_][A-Za-z0-9_]*\)s")
_FORMAT_RE = re.compile(r"(?<!%)%s")
_NAMED_RE = re.compile(r"(?<!:):[A-Za-z_][A-Za-z0-9_]*")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_IN_LIST_RE = re.compile(r"\(\s*\?(?:\s*,\s*\?)*\s*\)")
_WS_RE = re.compile(r"\s+")


def normalize_sql(sql: str) -> str:
    """
    Normalize a SQL statement into a stable fingerprint.

    String literals, numeric literals, and bind parameters are replaced with
    ``?`` so that structurally identical statements aggregate together.  The
    number of elements in ``IN (...)`` lists is also collapsed.

    Parameters
    ----------
    sql : str
        SQL statement as emitted by the database driver.

    Returns
    -------
    str
        Normalized SQL fingerprint.

    Examples
    --------
    >>> normalize_sql("SELECT * FROM papers WHERE title LIKE '%x%' AND year = 2025")
    'SELECT * FROM papers WHERE title LIKE ? AND year = ?'
    >>> normalize_sql("SELECT * FROM papers WHERE uid IN (?, ?, ?)")
    'SELECT * FROM papers WHERE uid IN (?)'
    """
    normalized = _COMMENT_RE.sub(" ", sql)
    normalized = _STRING_RE.sub("?", normalized)
    normalized = _PYFORMAT_RE.sub("?", normalized)
    normalized = _FORMAT_RE.sub("?", normalized)
    normalized = _NAMED_RE.sub("?", normalized)
    normalized = _NUMBER_RE.sub("?", normalized)
    normalized = _IN_LIST_RE.sub("(?)", normalized)
    normalized = _WS_RE.sub(" ", normalized)
    return normalized.strip()


def _percentile(values: Iterable[float], fraction: float) -> float:
    """Return a linearly interpolated percentile of *values*."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * fraction
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[int(rank)]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _count_parameters(parameters: Any) -> int:
    """Return the number of bound parameters in *parameters*."""
    if parameters is None:
        return 0
    if isinstance(parameters, (dict, list, tuple)):
        return len(parameters)
    return 1


def _is_select(statement: str) -> bool:
    """Return ``True`` when *statement* is a read-only SELECT/WITH query."""
    stripped = statement.lstrip().upper()
    return stripped.startswith("SELECT") or stripped.startswith("WITH") or stripped.startswith("EXPLAIN")


@dataclass
class QueryRecord:
    """
    A single profiled SQL statement.

    Attributes
    ----------
    label : str
        Case label (innermost active ``case()``) that triggered the statement.
    fingerprint : str
        Normalized SQL text.
    duration_ms : float
        Statement execution time in milliseconds.
    rowcount : int
        Rows reported by the driver (``-1`` when unavailable).
    is_select : bool
        Whether the statement is read-only.
    param_count : int
        Number of bound parameters.
    executemany : bool
        Whether the driver used ``executemany``.
    sql : str or None
        Raw SQL text when captured.
    parameters : Any
        Bound parameters when ``capture_params`` is enabled.
    """

    label: str
    fingerprint: str
    duration_ms: float
    rowcount: int
    is_select: bool
    param_count: int
    executemany: bool
    sql: Optional[str] = None
    parameters: Any = field(default=None, repr=False)
    thread_id: int = field(default=-1, repr=False)


@dataclass
class CaseStats:
    """
    Aggregated statistics for one execution of a profiling case.

    Attributes
    ----------
    label : str
        Full nested case label.
    wall_ms : float
        Wall-clock time spent inside the case.
    sql_ms : float
        Sum of SQL statement durations inside the case.
    non_sql_ms : float
        Wall time not accounted for by SQL (e.g. ChromaDB, LLM, CPU).
    statement_count : int
        Number of SQL statements executed.
    rows : int
        Sum of positive row counts.
    fingerprint_totals : dict
        Total milliseconds per normalized SQL fingerprint.
    fingerprint_counts : dict
        Number of executions per normalized SQL fingerprint.
    fingerprint_select : dict
        Whether each fingerprint is read-only.
    slowest : list of QueryRecord
        Slowest statements observed in this case.
    """

    label: str
    wall_ms: float
    sql_ms: float
    non_sql_ms: float
    statement_count: int
    rows: int
    fingerprint_totals: Dict[str, float]
    fingerprint_counts: Dict[str, int]
    fingerprint_select: Dict[str, bool]
    slowest: List[QueryRecord]

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        fingerprints = [
            {
                "sql": sql,
                "count": self.fingerprint_counts[sql],
                "total_ms": round(self.fingerprint_totals[sql], 3),
                "avg_ms": round(self.fingerprint_totals[sql] / max(self.fingerprint_counts[sql], 1), 3),
                "is_select": self.fingerprint_select.get(sql, False),
            }
            for sql in sorted(self.fingerprint_totals, key=self.fingerprint_totals.get, reverse=True)  # type: ignore[arg-type]
        ]
        return {
            "label": self.label,
            "wall_ms": round(self.wall_ms, 3),
            "sql_ms": round(self.sql_ms, 3),
            "non_sql_ms": round(self.non_sql_ms, 3),
            "statement_count": self.statement_count,
            "rows": self.rows,
            "fingerprints": fingerprints,
            "slowest": [
                {
                    "label": record.label,
                    "sql": record.fingerprint,
                    "duration_ms": round(record.duration_ms, 3),
                    "rowcount": record.rowcount,
                    "is_select": record.is_select,
                }
                for record in self.slowest
            ],
        }


@dataclass
class AggregatedCase:
    """
    Statistics aggregated across repeated executions of a case label.

    Attributes
    ----------
    label : str
        Full nested case label.
    runs : int
        Number of executions that were aggregated.
    wall_ms_min, wall_ms_median, wall_ms_p95, wall_ms_mean : float
        Wall-clock distribution in milliseconds.
    sql_ms_mean : float
        Mean SQL time per run.
    non_sql_ms_mean : float
        Mean non-SQL time per run.
    statement_count_mean : float
        Mean number of statements per run.
    rows_mean : float
        Mean number of rows per run.
    fingerprints : list of dict
        Per-fingerprint totals, counts, and average duration.
    """

    label: str
    runs: int
    wall_ms_min: float
    wall_ms_median: float
    wall_ms_p95: float
    wall_ms_mean: float
    sql_ms_mean: float
    non_sql_ms_mean: float
    statement_count_mean: float
    rows_mean: float
    fingerprints: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "label": self.label,
            "runs": self.runs,
            "wall_ms": {
                "min": round(self.wall_ms_min, 3),
                "median": round(self.wall_ms_median, 3),
                "p95": round(self.wall_ms_p95, 3),
                "mean": round(self.wall_ms_mean, 3),
            },
            "sql_ms_mean": round(self.sql_ms_mean, 3),
            "non_sql_ms_mean": round(self.non_sql_ms_mean, 3),
            "statement_count_mean": round(self.statement_count_mean, 3),
            "rows_mean": round(self.rows_mean, 3),
            "fingerprints": self.fingerprints,
        }


def aggregate_cases(cases: List[CaseStats], label: Optional[str] = None) -> AggregatedCase:
    """
    Aggregate a list of :class:`CaseStats` into an :class:`AggregatedCase`.

    Parameters
    ----------
    cases : list of CaseStats
        Case executions to aggregate.  Must not be empty.
    label : str, optional
        Label to report.  Defaults to the label of the first case.

    Returns
    -------
    AggregatedCase
        Aggregated statistics.

    Raises
    ------
    ValueError
        If *cases* is empty.
    """
    if not cases:
        raise ValueError("Cannot aggregate an empty list of cases.")

    wall_values = [case.wall_ms for case in cases]
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    select: Dict[str, bool] = {}
    for case in cases:
        for sql, total in case.fingerprint_totals.items():
            totals[sql] = totals.get(sql, 0.0) + total
        for sql, count in case.fingerprint_counts.items():
            counts[sql] = counts.get(sql, 0) + count
        select.update(case.fingerprint_select)

    fingerprints = [
        {
            "sql": sql,
            "count": counts[sql],
            "total_ms": round(totals[sql], 3),
            "avg_ms": round(totals[sql] / max(counts[sql], 1), 3),
            "is_select": select.get(sql, False),
        }
        for sql in sorted(totals, key=totals.get, reverse=True)  # type: ignore[arg-type]
    ]

    return AggregatedCase(
        label=label or cases[0].label,
        runs=len(cases),
        wall_ms_min=min(wall_values),
        wall_ms_median=_percentile(wall_values, 0.5),
        wall_ms_p95=_percentile(wall_values, 0.95),
        wall_ms_mean=sum(wall_values) / len(wall_values),
        sql_ms_mean=sum(case.sql_ms for case in cases) / len(cases),
        non_sql_ms_mean=sum(case.non_sql_ms for case in cases) / len(cases),
        statement_count_mean=sum(case.statement_count for case in cases) / len(cases),
        rows_mean=sum(case.rows for case in cases) / len(cases),
        fingerprints=fingerprints,
    )


@contextlib.contextmanager
def suppress_profiling():
    """
    Temporarily suppress the active profiler.

    Used by :func:`explain_statement` so diagnostic ``EXPLAIN`` queries do not
    pollute the measurements.

    Yields
    ------
    None
    """
    token = _SUPPRESSED.set(True)
    try:
        yield
    finally:
        _SUPPRESSED.reset(token)


def explain_statement(
    engine: Engine,
    sql: str,
    parameters: Any = None,
    analyze: bool = True,
) -> Any:
    """
    Run ``EXPLAIN`` for a captured PostgreSQL statement.

    Parameters
    ----------
    engine : Engine
        SQLAlchemy engine connected to PostgreSQL.
    sql : str
        Raw SQL statement as emitted by the driver.
    parameters : Any, optional
        Bound parameters for the statement.
    analyze : bool, optional
        Whether to use ``EXPLAIN ANALYZE`` (executes the statement).  Only
        pass ``True`` for read-only SELECT statements.

    Returns
    -------
    Any
        The ``EXPLAIN`` result as parsed JSON.

    Raises
    ------
    ValueError
        If the engine is not PostgreSQL or *parameters* represents an
        ``executemany`` batch.
    """
    if engine.dialect.name != "postgresql":
        raise ValueError("EXPLAIN is only supported for PostgreSQL engines.")
    if isinstance(parameters, list):
        raise ValueError("Cannot EXPLAIN executemany parameter batches.")

    prefix = "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " if analyze else "EXPLAIN (FORMAT JSON) "
    with suppress_profiling():
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            result = conn.exec_driver_sql(prefix + sql, parameters or {})
            return result.scalar()


class SqlProfiler:
    """
    Profile SQL statements executed anywhere in the process.

    The profiler installs listeners on the SQLAlchemy :class:`Engine` class,
    so it also captures statements from ``DatabaseManager`` instances created
    internally by MCP tools.  Use :meth:`case` to attribute statements to a
    logical operation and :meth:`measure` to repeat a callable.

    Parameters
    ----------
    capture_params : bool, optional
        Store bound parameters for every statement (needed for
        :func:`explain_statement`).  Disabled by default to limit memory use.
    max_slowest : int, optional
        Maximum number of slowest statements retained per case.

    Examples
    --------
    >>> with SqlProfiler() as profiler:
    ...     with profiler.case("lookup"):
    ...         db.get_paper_by_uid("abc123")
    >>> stats = profiler.aggregated()[0]
    >>> stats.statement_count_mean
    1.0
    """

    def __init__(self, capture_params: bool = False, max_slowest: int = 10):
        self.capture_params = capture_params
        self.max_slowest = max_slowest
        self._records: List[QueryRecord] = []
        self._cases: List[CaseStats] = []
        self._lock = threading.Lock()
        self._labels: contextvars.ContextVar[Tuple[str, ...]] = contextvars.ContextVar(
            f"sql_profiler_labels_{id(self)}", default=()
        )
        self._pending: Dict[int, float] = {}
        self._attached = False
        self._before_cb = self._handle_before
        self._after_cb = self._handle_after

    def attach(self) -> None:
        """
        Install class-level SQLAlchemy event listeners.

        Calling this method twice is a no-op.
        """
        if self._attached:
            return
        event.listen(Engine, "before_cursor_execute", self._before_cb)
        event.listen(Engine, "after_cursor_execute", self._after_cb)
        self._attached = True

    def detach(self) -> None:
        """
        Remove the SQLAlchemy event listeners.

        Calling this method when not attached is a no-op.
        """
        if not self._attached:
            return
        event.remove(Engine, "before_cursor_execute", self._before_cb)
        event.remove(Engine, "after_cursor_execute", self._after_cb)
        self._attached = False

    def __enter__(self) -> "SqlProfiler":
        """Attach the profiler and return it."""
        self.attach()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Detach the profiler."""
        self.detach()

    def reset(self) -> None:
        """Clear all recorded statements and cases."""
        with self._lock:
            self._records.clear()
            self._cases.clear()

    @contextlib.contextmanager
    def case(self, label: str):
        """
        Attribute all SQL executed inside the block to *label*.

        Nested cases are supported; the innermost label is attached to each
        statement, while parent cases still include statements from their
        children in their totals.

        Parameters
        ----------
        label : str
            Human-readable case label.

        Yields
        ------
        None
        """
        labels = self._labels.get()
        token = self._labels.set(labels + (label,))
        thread_id = threading.get_ident()
        start_time = time.perf_counter()
        with self._lock:
            start_index = len(self._records)

        try:
            yield
        finally:
            wall_ms = (time.perf_counter() - start_time) * 1000.0
            self._labels.reset(token)
            with self._lock:
                candidates = list(self._records[start_index:])
            own = [record for record in candidates if record.thread_id == thread_id]
            self._append_case(" > ".join(labels + (label,)), wall_ms, own)

    def measure(self, label: str, func: Any, runs: int = 3, warmup: int = 0) -> AggregatedCase:
        """
        Execute *func* repeatedly inside a case and aggregate the results.

        Parameters
        ----------
        label : str
            Case label.
        func : callable
            Zero-argument callable to execute.
        runs : int, optional
            Number of measured executions (default: 3).
        warmup : int, optional
            Number of unmeasured warmup executions (default: 0).

        Returns
        -------
        AggregatedCase
            Aggregated statistics for the measured runs.

        Raises
        ------
        ValueError
            If *runs* is less than 1.
        """
        if runs < 1:
            raise ValueError("runs must be at least 1")
        for _ in range(max(warmup, 0)):
            func()
        start = len(self._cases)
        for _ in range(runs):
            with self.case(label):
                func()
        with self._lock:
            cases = [case for case in self._cases[start:] if case.label == label]
        return aggregate_cases(cases, label=label)

    def results(self) -> List[CaseStats]:
        """Return a copy of all recorded case statistics."""
        with self._lock:
            return list(self._cases)

    def records(self) -> List[QueryRecord]:
        """Return a copy of all recorded SQL statements."""
        with self._lock:
            return list(self._records)

    def aggregated(self, label: Optional[str] = None) -> List[AggregatedCase]:
        """
        Return statistics aggregated by case label.

        Parameters
        ----------
        label : str, optional
            Return only this label when provided.

        Returns
        -------
        list of AggregatedCase
            Aggregated cases sorted by mean SQL time (descending).
        """
        with self._lock:
            groups: Dict[str, List[CaseStats]] = {}
            for case in self._cases:
                groups.setdefault(case.label, []).append(case)

        if label is not None:
            groups = {label: groups[label]} if label in groups else {}

        aggregated = [aggregate_cases(cases, label=name) for name, cases in groups.items()]
        aggregated.sort(key=lambda item: (item.sql_ms_mean, item.wall_ms_mean), reverse=True)
        return aggregated

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable report of all measurements."""
        with self._lock:
            cases = list(self._cases)
        settings = {
            "capture_params": self.capture_params,
            "max_slowest": self.max_slowest,
        }
        return {
            "settings": settings,
            "cases": [case.to_dict() for case in cases],
            "aggregated": [case.to_dict() for case in self.aggregated()],
        }

    def to_json(self, path: Optional[str] = None) -> str:
        """
        Serialize the report to JSON.

        Parameters
        ----------
        path : str, optional
            When provided, write the JSON to this file.

        Returns
        -------
        str
            The serialized JSON document.
        """
        document = json.dumps(self.to_dict(), indent=2, default=str)
        if path:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(document)
        return document

    def _append_case(self, label: str, wall_ms: float, records: List[QueryRecord]) -> None:
        """Build and store a :class:`CaseStats` from raw records."""
        totals: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        select: Dict[str, bool] = {}
        for record in records:
            totals[record.fingerprint] = totals.get(record.fingerprint, 0.0) + record.duration_ms
            counts[record.fingerprint] = counts.get(record.fingerprint, 0) + 1
            select[record.fingerprint] = record.is_select

        sql_ms = sum(record.duration_ms for record in records)
        rows = sum(record.rowcount for record in records if record.rowcount > 0)
        slowest = sorted(records, key=lambda record: record.duration_ms, reverse=True)[: self.max_slowest]
        stats = CaseStats(
            label=label,
            wall_ms=wall_ms,
            sql_ms=sql_ms,
            non_sql_ms=max(0.0, wall_ms - sql_ms),
            statement_count=len(records),
            rows=rows,
            fingerprint_totals=totals,
            fingerprint_counts=counts,
            fingerprint_select=select,
            slowest=slowest,
        )
        with self._lock:
            self._cases.append(stats)

    def _handle_before(
        self,
        conn: Any,
        cursor: Any,
        statement: str,
        parameters: Any,
        context: Any,
        executemany: bool,
    ) -> None:
        """Record the statement start time."""
        if _SUPPRESSED.get():
            return
        start = time.perf_counter()
        thread_id = threading.get_ident()
        if context is not None:
            try:
                setattr(context, "_abstracts_profiler_start", start)
                return
            except Exception:
                pass
        self._pending[thread_id] = start

    def _handle_after(
        self,
        conn: Any,
        cursor: Any,
        statement: str,
        parameters: Any,
        context: Any,
        executemany: bool,
    ) -> None:
        """Record a completed statement."""
        if _SUPPRESSED.get():
            return

        thread_id = threading.get_ident()
        start: Optional[float] = None
        if context is not None:
            start = getattr(context, "_abstracts_profiler_start", None)
            if start is not None:
                try:
                    delattr(context, "_abstracts_profiler_start")
                except Exception:
                    pass
        if start is None:
            start = self._pending.pop(thread_id, None)
        if start is None:
            return

        duration_ms = (time.perf_counter() - start) * 1000.0
        labels = self._labels.get()
        label = " > ".join(labels)
        rowcount = getattr(cursor, "rowcount", -1)
        try:
            rowcount = int(rowcount)
        except (TypeError, ValueError):
            rowcount = -1

        record = QueryRecord(
            label=label,
            fingerprint=normalize_sql(statement),
            duration_ms=duration_ms,
            rowcount=rowcount,
            is_select=_is_select(statement),
            param_count=_count_parameters(parameters),
            executemany=bool(executemany),
            sql=statement if self.capture_params else None,
            parameters=parameters if self.capture_params else None,
            thread_id=thread_id,
        )
        with self._lock:
            self._records.append(record)

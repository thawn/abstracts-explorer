"""
Request, span, and SQL profiling for the web UI.

The session is inactive unless :func:`set_active_session` is called.  When
inactive, :func:`profile_span` and :func:`async_profile_span` are near no-ops,
so production code can call them unconditionally.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

import httpx2
from sqlalchemy import event
from sqlalchemy.engine import Engine

from abstracts_explorer.profiling.sql import normalize_sql

logger = logging.getLogger(__name__)

LEAF_CATEGORIES = ("sql", "chroma", "embedding", "llm", "guard")
_TOP_SPANS = 25
_TOP_FINGERPRINTS = 10

_CURRENT_PROFILE: contextvars.ContextVar[Optional["RequestProfile"]] = contextvars.ContextVar(
    "abstracts_explorer_current_profile", default=None
)
_ACTIVE_SESSION: Optional["ProfilingSession"] = None
_FALLBACK_PROFILE: Optional["RequestProfile"] = None
_FALLBACK_LOCK = threading.Lock()


@dataclass
class ProfileSpan:
    """A single timed span inside a request."""

    name: str
    category: str
    duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SqlProfileRecord:
    """A profiled SQL statement inside a request."""

    fingerprint: str
    duration_ms: float
    rowcount: int
    is_select: bool
    param_count: int


def _safe_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Return metadata values that are safe to serialize to JSON."""
    result: Dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[str(key)] = value
        else:
            result[str(key)] = str(value)
    return result


def _is_select_statement(statement: str) -> bool:
    """Return whether *statement* is read-only."""
    stripped = statement.lstrip().upper()
    return stripped.startswith(("SELECT", "WITH", "EXPLAIN"))


def _count_parameters(parameters: Any) -> int:
    """Return the number of bound parameters."""
    if parameters is None:
        return 0
    if isinstance(parameters, (dict, list, tuple)):
        return len(parameters)
    return 1


def _current_profile() -> Optional["RequestProfile"]:
    """Return the active request profile for the current context or thread."""
    profile = _CURRENT_PROFILE.get()
    if profile is not None:
        return profile
    return _FALLBACK_PROFILE


def profiling_enabled() -> bool:
    """Return whether a profiling session is active."""
    return _ACTIVE_SESSION is not None


def set_active_session(session: Optional["ProfilingSession"]) -> None:
    """Set or clear the process-wide profiling session."""
    global _ACTIVE_SESSION
    _ACTIVE_SESSION = session


@dataclass
class RequestProfile:
    """Timing state for one HTTP request."""

    request_id: str
    method: str
    path: str
    query: str = ""
    user_agent: str = ""
    response_bytes: int = 0
    spans: List[ProfileSpan] = field(default_factory=list)
    sql: List[SqlProfileRecord] = field(default_factory=list)
    started: float = field(default_factory=time.perf_counter)
    _token: Any = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_span(self, name: str, category: str, duration_ms: float, metadata: Dict[str, Any]) -> None:
        """Record a span."""
        with self._lock:
            self.spans.append(
                ProfileSpan(
                    name=name,
                    category=category,
                    duration_ms=max(0.0, duration_ms),
                    metadata=_safe_metadata(metadata),
                )
            )

    def add_sql(
        self,
        statement: str,
        duration_ms: float,
        rowcount: int,
        parameters: Any,
    ) -> None:
        """Record a SQL statement."""
        with self._lock:
            self.sql.append(
                SqlProfileRecord(
                    fingerprint=normalize_sql(statement),
                    duration_ms=max(0.0, duration_ms),
                    rowcount=int(rowcount) if rowcount is not None else -1,
                    is_select=_is_select_statement(statement),
                    param_count=_count_parameters(parameters),
                )
            )

    def to_record(self, status: int, wall_ms: float) -> Dict[str, Any]:
        """Build a JSON-serializable request report."""
        with self._lock:
            spans = list(self.spans)
            sql_records = list(self.sql)

        span_totals: Dict[str, float] = {}
        for span in spans:
            span_totals[span.category] = span_totals.get(span.category, 0.0) + span.duration_ms

        sql_total = sum(record.duration_ms for record in sql_records)
        categories = {
            "sql": sql_total,
            "chroma": span_totals.get("chroma", 0.0),
            "embedding": span_totals.get("embedding", 0.0),
            "llm": span_totals.get("llm", 0.0),
            "guard": span_totals.get("guard", 0.0),
            # Overlapping/informational categories:
            "tool": span_totals.get("tool", 0.0),
            "rag": span_totals.get("rag", 0.0),
        }
        categories["other"] = max(0.0, wall_ms - sum(categories[name] for name in LEAF_CATEGORIES))

        fingerprint_totals: Dict[str, Dict[str, Any]] = {}
        for record in sql_records:
            item = fingerprint_totals.setdefault(
                record.fingerprint,
                {"count": 0, "total_ms": 0.0, "is_select": record.is_select},
            )
            item["count"] += 1
            item["total_ms"] += record.duration_ms

        top_fingerprints = [
            {
                "sql": fingerprint,
                "count": item["count"],
                "total_ms": round(item["total_ms"], 3),
                "avg_ms": round(item["total_ms"] / max(item["count"], 1), 3),
                "is_select": item["is_select"],
            }
            for fingerprint, item in sorted(
                fingerprint_totals.items(),
                key=lambda pair: pair[1]["total_ms"],
                reverse=True,
            )[:_TOP_FINGERPRINTS]
        ]

        top_spans = [
            {
                "name": span.name,
                "category": span.category,
                "duration_ms": round(span.duration_ms, 3),
                "metadata": span.metadata,
            }
            for span in sorted(spans, key=lambda item: item.duration_ms, reverse=True)[:_TOP_SPANS]
        ]

        return {
            "request_id": self.request_id,
            "method": self.method,
            "path": self.path,
            "query": self.query[:500],
            "user_agent": self.user_agent[:200],
            "status": status,
            "wall_ms": round(wall_ms, 3),
            "response_bytes": self.response_bytes,
            "categories_ms": {name: round(value, 3) for name, value in categories.items()},
            "sql": {
                "count": len(sql_records),
                "total_ms": round(sql_total, 3),
                "top": top_fingerprints,
            },
            "spans": top_spans,
        }


class ProfilingSession:
    """Collect request profiles and SQL statements for a web process."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._pending: Dict[int, float] = {}
        self._attached = False

    def attach(self) -> None:
        """Attach SQLAlchemy event listeners."""
        if self._attached:
            return
        event.listen(Engine, "before_cursor_execute", self._before_cursor_execute)
        event.listen(Engine, "after_cursor_execute", self._after_cursor_execute)
        self._attached = True
        set_active_session(self)

    def detach(self) -> None:
        """Detach SQLAlchemy event listeners."""
        if not self._attached:
            return
        event.remove(Engine, "before_cursor_execute", self._before_cursor_execute)
        event.remove(Engine, "after_cursor_execute", self._after_cursor_execute)
        self._attached = False
        set_active_session(None)

    def start_request(self, method: str, path: str, query: str = "", user_agent: str = "") -> RequestProfile:
        """Start profiling an HTTP request."""
        global _FALLBACK_PROFILE
        profile = RequestProfile(
            request_id=uuid.uuid4().hex,
            method=method,
            path=path,
            query=query,
            user_agent=user_agent,
        )
        profile._token = _CURRENT_PROFILE.set(profile)
        with _FALLBACK_LOCK:
            _FALLBACK_PROFILE = profile
        return profile

    def finish_request(self, profile: RequestProfile, status: int) -> None:
        """Finish and persist a request profile."""
        global _FALLBACK_PROFILE
        wall_ms = (time.perf_counter() - profile.started) * 1000.0
        try:
            record = profile.to_record(status, wall_ms)
            line = json.dumps(record, default=str, separators=(",", ":"))
            with self._lock:
                with self.output_path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            logger.info(
                "WEB_PROFILE %s %s %s status=%s wall=%.0fms sql=%.0fms chroma=%.0fms "
                "embedding=%.0fms llm=%.0fms tool=%.0fms other=%.0fms",
                record["method"],
                record["path"],
                record["request_id"][:8],
                status,
                wall_ms,
                record["categories_ms"]["sql"],
                record["categories_ms"]["chroma"],
                record["categories_ms"]["embedding"],
                record["categories_ms"]["llm"],
                record["categories_ms"]["tool"],
                record["categories_ms"]["other"],
            )
        except Exception:
            logger.warning("Failed to write web profile record", exc_info=True)
        finally:
            try:
                _CURRENT_PROFILE.reset(profile._token)
            except (ValueError, LookupError):
                pass
            with _FALLBACK_LOCK:
                if _FALLBACK_PROFILE is profile:
                    _FALLBACK_PROFILE = None

    def _before_cursor_execute(
        self,
        conn: Any,
        cursor: Any,
        statement: str,
        parameters: Any,
        context: Any,
        executemany: bool,
    ) -> None:
        if _current_profile() is None:
            return
        start = time.perf_counter()
        if context is not None:
            try:
                setattr(context, "_abstracts_profile_start", start)
                return
            except Exception:
                pass
        self._pending[threading.get_ident()] = start

    def _after_cursor_execute(
        self,
        conn: Any,
        cursor: Any,
        statement: str,
        parameters: Any,
        context: Any,
        executemany: bool,
    ) -> None:
        profile = _current_profile()
        if profile is None:
            return

        start: Optional[float] = None
        if context is not None:
            start = getattr(context, "_abstracts_profile_start", None)
            if start is not None:
                try:
                    delattr(context, "_abstracts_profile_start")
                except Exception:
                    pass
        if start is None:
            start = self._pending.pop(threading.get_ident(), None)
        if start is None:
            return

        rowcount = getattr(cursor, "rowcount", -1)
        profile.add_sql(statement, (time.perf_counter() - start) * 1000.0, rowcount, parameters)


@contextlib.contextmanager
def profile_span(name: str, category: str = "other", **metadata: Any) -> Iterator[None]:
    """Time a synchronous operation when profiling is active."""
    profile = _current_profile()
    if profile is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        profile.add_span(name, category, (time.perf_counter() - start) * 1000.0, metadata)


@contextlib.asynccontextmanager
async def async_profile_span(name: str, category: str = "other", **metadata: Any) -> AsyncIterator[None]:
    """Time an asynchronous operation when profiling is active."""
    profile = _current_profile()
    if profile is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        profile.add_span(name, category, (time.perf_counter() - start) * 1000.0, metadata)


class ProfilingAsyncTransport(httpx2.AsyncBaseTransport):
    """Async HTTP transport that records LLM request spans."""

    def __init__(self, transport: httpx2.AsyncBaseTransport) -> None:
        self._transport = transport

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        """Forward the request while recording its duration."""
        async with async_profile_span("llm.http", "llm", host=request.url.host, method=request.method):
            return await self._transport.handle_async_request(request)

    async def aclose(self) -> None:
        """Close the wrapped transport."""
        await self._transport.aclose()


def instrument_async_transport(transport: httpx2.AsyncBaseTransport) -> httpx2.AsyncBaseTransport:
    """Wrap *transport* when profiling is active, otherwise return it unchanged."""
    if not profiling_enabled():
        return transport
    return ProfilingAsyncTransport(transport)

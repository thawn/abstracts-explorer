"""
Profiling utilities for SQL, request spans, and Chroma/LLM operations.

The SQL profiler is available as :class:`SqlProfiler`.  Web request profiling
is opt-in through :class:`ProfilingSession`; the :func:`profile_span` helpers
are no-ops when no session is active.
"""

from abstracts_explorer.profiling.spans import (
    ProfilingAsyncTransport,
    ProfilingSession,
    RequestProfile,
    async_profile_span,
    instrument_async_transport,
    profile_span,
    profiling_enabled,
    set_active_session,
)
from abstracts_explorer.profiling.sql import (
    AggregatedCase,
    CaseStats,
    QueryRecord,
    SqlProfiler,
    aggregate_cases,
    explain_statement,
    normalize_sql,
    suppress_profiling,
)

__all__ = [
    "AggregatedCase",
    "CaseStats",
    "ProfilingAsyncTransport",
    "ProfilingSession",
    "QueryRecord",
    "RequestProfile",
    "SqlProfiler",
    "aggregate_cases",
    "async_profile_span",
    "explain_statement",
    "instrument_async_transport",
    "normalize_sql",
    "profile_span",
    "profiling_enabled",
    "set_active_session",
    "suppress_profiling",
]

#!/usr/bin/env python3
"""
Profile common database access patterns, including MCP tools.

The script measures representative ``DatabaseManager`` calls and every MCP
tool, splitting wall-clock time into SQL time (captured through SQLAlchemy
event listeners) and non-SQL time (ChromaDB, LLM, in-memory work).  It also
records nested ``db:`` calls so N+1 patterns are visible.

Usage::

    uv run python scripts/profile_db_access.py
    uv run python scripts/profile_db_access.py --paper-db data/abstracts.db
    uv run python scripts/profile_db_access.py --paper-db "$POSTGRES_TEST_URL" \\
        --conference NeurIPS --year 2025 --json baseline.json
    uv run python scripts/profile_db_access.py --tools db-only --include-writes

The target database is only read.  Write benchmarks run against a temporary
SQLite database and require ``--include-writes``.
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from abstracts_explorer.config import get_config
from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.embeddings import EmbeddingsManager
from abstracts_explorer.mcp_tools import execute_mcp_tool
from abstracts_explorer.plugin import LightweightPaper
from abstracts_explorer.profiling import (
    AggregatedCase,
    SqlProfiler,
    explain_statement,
    suppress_profiling,
)

try:  # openai is a transitive dependency, but keep the script importable anyway
    from openai import OpenAI
except ImportError:  # pragma: no cover - defensive
    OpenAI = None  # type: ignore[assignment]


DEFAULT_TOPIC = "transformers"
CLUSTERING_PARAMS = {"linkage": "ward", "distance_threshold": 150.0}


@dataclass
class Scope:
    """
    Representative data selected from the target database.

    Attributes
    ----------
    conference : str or None
        Conference used by the benchmark cases.
    year : int or None
        Year used by the benchmark cases.
    topic : str
        Topic text used for keyword and semantic searches.
    paper_uids : list of str
        Sample paper UIDs.
    priorities : dict
        Sample interesting-paper payload for the summary/recommendation tools.
    author_token : str
        Author token used for author field-filter cases.
    """

    conference: Optional[str]
    year: Optional[int]
    topic: str
    paper_uids: List[str] = field(default_factory=list)
    priorities: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    author_token: str = "Smith"


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Profile DatabaseManager access patterns and MCP tools, reporting SQL vs non-SQL time "
            "and N+1 call patterns."
        )
    )
    parser.add_argument(
        "--paper-db",
        help="Override PAPER_DB for this command only. Accepts a SQLite path or full database URL.",
    )
    parser.add_argument("--conference", help="Conference to profile (default: busiest in the database).")
    parser.add_argument("--year", type=int, help="Year to profile (default: most recent for the conference).")
    parser.add_argument(
        "--topic", default=DEFAULT_TOPIC, help=f"Topic text for search cases (default: {DEFAULT_TOPIC})."
    )
    parser.add_argument("--runs", type=int, default=3, help="Measured runs for raw DB cases (default: 3).")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs for raw DB cases (default: 1).")
    parser.add_argument("--mcp-runs", type=int, default=1, help="Measured runs per MCP tool (default: 1).")
    parser.add_argument(
        "--tools",
        default="all",
        help="MCP tool selection: 'all', 'db-only', or a comma-separated list of tool names.",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip MCP tools that need ChromaDB or the embedding backend.",
    )
    parser.add_argument(
        "--force-embeddings",
        action="store_true",
        help="Attempt embedding-backed tools even when the availability probe fails.",
    )
    parser.add_argument(
        "--include-writes",
        action="store_true",
        help="Benchmark add_papers/delete against a temporary SQLite database.",
    )
    parser.add_argument("--write-count", type=int, default=1000, help="Rows per write benchmark (default: 1000).")
    parser.add_argument(
        "--include-full-scan",
        action="store_true",
        help="Include the full-table read used by embed_from_database (can be slow).",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Run EXPLAIN (ANALYZE, BUFFERS) for top PostgreSQL SELECTs (requires PostgreSQL).",
    )
    parser.add_argument("--json", help="Write the full report to this JSON file.")
    parser.add_argument("--top", type=int, default=40, help="Maximum aggregated rows to print (default: 40).")
    parser.add_argument("--verbose", action="store_true", help="Enable INFO logging.")
    return parser.parse_args()


def _set_paper_db(value: Optional[str]) -> Optional[str]:
    """Set PAPER_DB and reload config; return the previous value."""
    if not value:
        return os.environ.get("PAPER_DB")
    previous = os.environ.get("PAPER_DB")
    os.environ["PAPER_DB"] = value
    get_config(reload=True)
    return previous


def _restore_paper_db(previous: Optional[str]) -> None:
    """Restore a previously saved PAPER_DB value."""
    if previous is None:
        os.environ.pop("PAPER_DB", None)
    else:
        os.environ["PAPER_DB"] = previous
    get_config(reload=True)


def _busiest_conference(db: DatabaseManager) -> Tuple[Optional[str], Optional[int]]:
    """Return the conference with the most papers and its most recent year."""
    mapping = db.get_conference_years_from_db()
    if not mapping:
        return None, None
    conference = max(mapping, key=lambda name: db.get_paper_count({"conference": name}))
    years = mapping[conference]
    return conference, max(years) if years else None


def select_scope(db: DatabaseManager, args: argparse.Namespace) -> Scope:
    """
    Select representative papers and filters from the database.

    Parameters
    ----------
    db : DatabaseManager
        Connected database manager.
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    Scope
        Selected benchmark scope.
    """
    default_conference, default_year = _busiest_conference(db)
    conference = args.conference or default_conference
    if args.year is not None:
        year = args.year
    elif conference == default_conference:
        year = default_year
    elif conference is not None:
        years = db.get_years(conference=conference)
        year = max(years) if years else None
    else:
        year = None

    filters: Dict[str, Any] = {}
    if conference:
        filters["conference"] = conference
    if year is not None:
        filters["year"] = year

    rows = db.get_paper(filters, limit=50) if filters else db.get_paper(limit=50)
    if not rows and filters:
        rows = db.get_paper({"conference": conference} if conference else {}, limit=50)
    if not rows:
        rows = db.get_paper(limit=50)

    uids = [str(row["uid"]) for row in rows if row.get("uid")]
    priorities = {uid: {"priority": 5 - (index % 5), "searchTerm": args.topic} for index, uid in enumerate(uids[:10])}

    author_token = "Smith"
    for row in rows:
        authors = row.get("authors") or []
        if isinstance(authors, str):
            authors = [name.strip() for name in authors.split(";") if name.strip()]
        if authors:
            author_token = str(authors[0]).split()[-1]
            break

    return Scope(
        conference=conference,
        year=year,
        topic=args.topic,
        paper_uids=uids,
        priorities=priorities,
        author_token=author_token,
    )


def _llm_reachable(em: EmbeddingsManager, timeout: float = 3.0) -> bool:
    """Return whether the configured OpenAI-compatible backend responds."""
    if OpenAI is None:
        return False
    try:
        client = OpenAI(
            base_url=f"{em.lm_studio_url}/v1",
            api_key=em.llm_backend_auth_token or "lm-studio-local",
            timeout=timeout,
        )
        client.models.list()
        return True
    except Exception:
        return False


def _chroma_collection_available(em: EmbeddingsManager) -> bool:
    """Return whether the configured ChromaDB collection exists."""
    try:
        collections = em.client.list_collections()
    except Exception:
        return False
    names: List[str] = []
    for collection in collections:
        name = getattr(collection, "name", None)
        if name is None and isinstance(collection, dict):
            name = collection.get("name")
        names.append(str(name))
    return em.collection_name in names


def detect_backends(skip_embeddings: bool) -> Dict[str, bool]:
    """
    Probe optional backends used by MCP tools.

    Parameters
    ----------
    skip_embeddings : bool
        When ``True``, skip probing entirely.

    Returns
    -------
    dict
        Mapping with ``llm`` and ``chroma`` availability flags.
    """
    result = {"llm": False, "chroma": False}
    if skip_embeddings:
        return result

    em: Optional[EmbeddingsManager] = None
    try:
        em = EmbeddingsManager()
        result["llm"] = _llm_reachable(em)
        result["chroma"] = _chroma_collection_available(em)
    except Exception:
        pass
    finally:
        if em is not None:
            em.close()
    return result


def _has_clustering_cache(db: DatabaseManager, conference: Optional[str], year: Optional[int]) -> bool:
    """Return whether a clustering cache entry exists for the scope."""
    if not conference:
        return False
    config = get_config()
    kwargs = {
        "embedding_model": config.embedding_model,
        "reduction_method": "tsne",
        "n_components": 2,
        "clustering_method": "agglomerative",
        "n_clusters": None,
        "clustering_params": CLUSTERING_PARAMS,
        "conference": conference,
    }
    if db.get_clustering_cache(year=year, **kwargs):
        return True
    return bool(db.get_clustering_cache(year=None, **kwargs))


def instrument_database(profiler: SqlProfiler) -> Callable[[], None]:
    """
    Wrap DatabaseManager methods with nested profiling cases.

    Parameters
    ----------
    profiler : SqlProfiler
        Active profiler.

    Returns
    -------
    callable
        Restores the original methods when called.
    """
    method_names = [
        "add_papers",
        "get_clustering_cache",
        "get_conference_years_from_db",
        "get_conferences",
        "get_paper",
        "get_paper_by_original_id_or_uid",
        "get_paper_by_uid",
        "get_paper_count",
        "get_papers_by_uids",
        "get_sessions",
        "get_stats",
        "get_years",
        "search_papers",
        "search_papers_keyword",
        "search_papers_semantic",
    ]
    originals: Dict[str, Callable[..., Any]] = {}

    def make_wrapper(name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(original)
        def wrapper(self: DatabaseManager, *args: Any, **kwargs: Any) -> Any:
            with profiler.case(f"db:{name}"):
                return original(self, *args, **kwargs)

        return wrapper

    for name in method_names:
        original = getattr(DatabaseManager, name, None)
        if original is None:
            continue
        originals[name] = original
        setattr(DatabaseManager, name, make_wrapper(name, original))

    def restore() -> None:
        for name, original in originals.items():
            setattr(DatabaseManager, name, original)

    return restore


def raw_cases(db: DatabaseManager, scope: Scope, args: argparse.Namespace) -> List[Tuple[str, Callable[[], Any]]]:
    """
    Build raw DatabaseManager benchmark cases.

    Parameters
    ----------
    db : DatabaseManager
        Connected database manager.
    scope : Scope
        Representative data.
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    list of (str, callable)
        Label and zero-argument callable pairs.
    """
    conference = scope.conference
    year = scope.year
    topic = scope.topic
    uid = scope.paper_uids[0] if scope.paper_uids else ""
    many_uids = (scope.paper_uids * 10)[:100]

    cases: List[Tuple[str, Callable[[], Any]]] = []

    def add(label: str, func: Callable[[], Any]) -> None:
        cases.append((label, func))

    add("raw:search.keyword.unfiltered", lambda: db.search_papers(keyword=topic, limit=50))
    if conference:
        add(
            "raw:search.keyword.conference",
            lambda: db.search_papers(keyword=topic, conferences=[conference], limit=50),
        )
        add(
            "raw:search.keyword.conference_year",
            lambda: db.search_papers(
                keyword=topic, conferences=[conference], years=[year] if year else None, limit=50
            ),
        )
        add(
            "raw:search.field.authors",
            lambda: db.search_papers(field_filters={"authors": scope.author_token}, limit=50),
        )
        add(
            "raw:search.field.title",
            lambda: db.search_papers(field_filters={"title": topic}, limit=50),
        )
        add(
            "raw:list.conference_year",
            lambda: db.get_paper({"conference": conference, "year": year}, limit=100) if year is not None else [],
        )
        add(
            "raw:get_stats.conference_year",
            lambda: db.get_stats(year=year, conference=conference),
        )
    add("raw:get_stats.all", db.get_stats)
    if year is not None:
        add("raw:get_stats.year", lambda: db.get_stats(year=year))
    if uid:
        add("raw:get_paper_by_uid", lambda: db.get_paper_by_uid(uid))
        add("raw:get_paper_by_original_id_or_uid", lambda: db.get_paper_by_original_id_or_uid(uid))
        add("raw:get_papers_by_uids.1", lambda: db.get_papers_by_uids([uid]))
        add("raw:get_papers_by_uids.10", lambda: db.get_papers_by_uids(scope.paper_uids[:10]))
        add("raw:get_papers_by_uids.100", lambda: db.get_papers_by_uids(many_uids))
    add("raw:get_sessions.all", db.get_sessions)
    if conference:
        add("raw:get_sessions.conference_year", lambda: db.get_sessions(conference=conference, year=year))
        add("raw:get_years.conference", lambda: db.get_years(conference=conference))
    add("raw:get_conferences", db.get_conferences)
    add("raw:get_conference_years_from_db", db.get_conference_years_from_db)

    if conference:
        config = get_config()

        def clustering_cache() -> Any:
            return db.get_clustering_cache(
                embedding_model=config.embedding_model,
                reduction_method="tsne",
                n_components=2,
                clustering_method="agglomerative",
                n_clusters=None,
                clustering_params=CLUSTERING_PARAMS,
                conference=conference,
                year=year,
            )

        add("raw:get_clustering_cache", clustering_cache)

    if args.include_full_scan:
        add("raw:get_paper.full_scan", lambda: db.get_paper(limit=0))

    return cases


def mcp_cases(scope: Scope) -> List[Tuple[str, str, Dict[str, Any], Set[str]]]:
    """
    Build MCP tool benchmark cases.

    Parameters
    ----------
    scope : Scope
        Representative data.

    Returns
    -------
    list of (label, tool_name, arguments, requirements)
        Each requirement is one of ``llm``, ``chroma``, or ``cache``.
    """
    conference = scope.conference
    year = scope.year
    uid = scope.paper_uids[0] if scope.paper_uids else ""
    years = [year] if year is not None else None
    cases: List[Tuple[str, str, Dict[str, Any], Set[str]]] = []

    if uid:
        cases.append(("mcp:get_paper_details.uid", "get_paper_details", {"paper_id": uid}, set()))
    cases.append(
        (
            "mcp:get_paper_details.title",
            "get_paper_details",
            {
                "title": scope.topic,
                "conference": conference,
                "year": year,
                "limit": 5,
            },
            set(),
        )
    )
    cases.append(
        (
            "mcp:summarize_interesting_papers",
            "summarize_interesting_papers",
            {
                "paper_ids": scope.paper_uids[:5],
                "priorities": scope.priorities,
                "conference": conference,
                "years": years,
            },
            set(),
        )
    )
    cases.append(
        (
            "mcp:recommend_related_papers",
            "recommend_related_papers",
            {
                "paper_ids": scope.paper_uids[:5],
                "priorities": scope.priorities,
                "conferences": [conference] if conference else None,
                "years": years,
                "n_results": 10,
            },
            {"chroma"},
        )
    )
    cases.append(
        (
            "mcp:search_papers",
            "search_papers",
            {
                "topic_keywords": scope.topic,
                "conference": conference,
                "years": years,
                "n_results": 10,
            },
            {"llm", "chroma"},
        )
    )
    cases.append(
        (
            "mcp:get_topic_evolution",
            "get_topic_evolution",
            {
                "topic_keywords": scope.topic,
                "conferences": [conference] if conference else None,
                "start_year": year,
                "end_year": year,
            },
            {"llm", "chroma"},
        )
    )
    cases.append(
        (
            "mcp:analyze_topic_relevance",
            "analyze_topic_relevance",
            {
                "topic": scope.topic,
                "conferences": [conference] if conference else None,
                "years": years,
            },
            {"llm", "chroma"},
        )
    )
    cases.append(
        (
            "mcp:get_conference_topics",
            "get_conference_topics",
            {"conferences": [conference] if conference else None, "years": years},
            {"chroma", "cache"},
        )
    )
    cases.append(
        (
            "mcp:get_cluster_visualization",
            "get_cluster_visualization",
            {"conferences": [conference] if conference else None, "years": years},
            {"cache"},
        )
    )
    return cases


def _selected_tools(args: argparse.Namespace, cases: Sequence[Tuple[str, str, Dict[str, Any], Set[str]]]) -> Set[str]:
    """Resolve the --tools selection into a set of tool names."""
    if args.tools == "all":
        return {tool for _, tool, _, _ in cases}
    if args.tools == "db-only":
        return {tool for _, tool, _, requirements in cases if not (requirements - {"cache"})}
    return {name.strip() for name in args.tools.split(",") if name.strip()}


def _invoke_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
    """Execute an MCP tool and raise when it returns an error payload."""
    raw = execute_mcp_tool(tool_name, arguments)
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return raw
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(str(data["error"]))
    return data


def benchmark_writes(profiler: SqlProfiler, count: int) -> List[AggregatedCase]:
    """
    Benchmark add_papers and delete_papers_by_conference_year on temp SQLite DBs.

    Parameters
    ----------
    profiler : SqlProfiler
        Active profiler.
    count : int
        Number of papers to insert per run.

    Returns
    -------
    list of AggregatedCase
        Write benchmark results.
    """
    papers = [
        LightweightPaper(
            title=f"Profiler benchmark paper {index}",
            authors=["Benchmark Author"],
            abstract="Benchmark abstract used to measure bulk insert performance.",
            session="Benchmark",
            poster_position=f"P{index}",
            year=2025,
            conference="Benchmark",
        )
        for index in range(count)
    ]
    results: List[AggregatedCase] = []
    original_paper_db = os.environ.get("PAPER_DB")

    def open_temp_db(name: str) -> Tuple[tempfile.TemporaryDirectory, DatabaseManager]:
        temp_dir = tempfile.TemporaryDirectory()
        os.environ["PAPER_DB"] = str(Path(temp_dir.name) / name)
        get_config(reload=True)
        db = DatabaseManager()
        db.create_tables()
        return temp_dir, db

    try:
        temp_dir, db = open_temp_db("add.db")
        try:
            results.append(profiler.measure("raw:add_papers", lambda: db.add_papers(papers), runs=1))
        finally:
            db.close()
            temp_dir.cleanup()

        temp_dir, db = open_temp_db("delete.db")
        try:
            with suppress_profiling():
                db.add_papers(papers)
            with profiler.case("raw:delete_papers_conference_year"):
                db.delete_papers_by_conference_year("Benchmark", 2025)
            delete_results = profiler.aggregated("raw:delete_papers_conference_year")
            if delete_results:
                results.append(delete_results[0])
        finally:
            db.close()
            temp_dir.cleanup()
    finally:
        _restore_paper_db(original_paper_db)
    return results


def _truncate(text: str, width: int) -> str:
    """Truncate *text* to *width* characters with an ellipsis."""
    if len(text) <= width:
        return text
    return text[: max(0, width - 1)] + "…"


def _print_table(title: str, rows: Sequence[AggregatedCase], limit: int) -> None:
    """Print aggregated cases as a table."""
    print(f"\n{title}")
    print("-" * len(title))
    header = f"{'case':<56} {'runs':>4} {'wall_ms':>10} {'sql_ms':>9} " f"{'nonsql_ms':>10} {'stmts':>6}  top SQL"
    print(header)
    for case in list(rows)[:limit]:
        top = case.fingerprints[0] if case.fingerprints else None
        top_text = "-"
        if top:
            top_text = f"{top['avg_ms']:.3f}ms x{top['count']} {_truncate(str(top['sql']), 70)}"
        print(
            f"{_truncate(case.label, 56):<56} {case.runs:>4} {case.wall_ms_mean:>10.2f} "
            f"{case.sql_ms_mean:>9.2f} {case.non_sql_ms_mean:>10.2f} "
            f"{case.statement_count_mean:>6.1f}  {top_text}"
        )


def _maybe_explain(db: DatabaseManager, profiler: SqlProfiler, args: argparse.Namespace) -> None:
    """Print EXPLAIN output for the top PostgreSQL SELECTs."""
    if not args.explain:
        return
    if db.engine is None:
        return
    if db.engine.dialect.name != "postgresql":
        print("\nEXPLAIN skipped: the target database is not PostgreSQL.")
        return

    print("\nEXPLAIN (ANALYZE, BUFFERS) for top SELECT statements")
    print("-----------------------------------------------------")
    seen: Set[str] = set()
    shown = 0
    for record in profiler.records():
        if shown >= args.top:
            break
        if not record.is_select or record.executemany or not record.sql or record.fingerprint in seen:
            continue
        seen.add(record.fingerprint)
        try:
            plan = explain_statement(db.engine, record.sql, record.parameters, analyze=True)
        except Exception as exc:
            print(f"- {_truncate(record.fingerprint, 100)}: EXPLAIN failed ({exc})")
            continue
        print(f"\n{_truncate(record.fingerprint, 120)}")
        print(json.dumps(plan, indent=2, default=str)[:4000])
        shown += 1


def main() -> int:
    """
    Run the profiler.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    """
    args = parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    previous_paper_db = _set_paper_db(args.paper_db)
    db = DatabaseManager()
    skipped: List[Tuple[str, str]] = []
    try:
        if db.get_paper_count() == 0:
            print("The target database contains no papers; nothing to profile.", file=sys.stderr)
            return 1

        scope = select_scope(db, args)
        backends = detect_backends(args.skip_embeddings)
        if args.force_embeddings:
            backends = {key: True for key in backends}
        if scope.conference:
            backends["cache"] = _has_clustering_cache(db, scope.conference, scope.year)

        print("Profiling scope")
        print("---------------")
        print(f"database      : {db._mask_url(db.database_url)}")
        print(f"conference    : {scope.conference}")
        print(f"year          : {scope.year}")
        print(f"topic         : {scope.topic}")
        print(f"sample papers : {len(scope.paper_uids)}")
        print(f"backends      : {json.dumps(backends, sort_keys=True)}")
        if not scope.paper_uids:
            print("warning: no sample papers found; uid-based cases will be skipped.")

        with SqlProfiler(capture_params=args.explain) as profiler:
            restore_database = instrument_database(profiler)
            try:
                raw_results: List[AggregatedCase] = []
                for label, func in raw_cases(db, scope, args):
                    try:
                        raw_results.append(profiler.measure(label, func, runs=args.runs, warmup=args.warmup))
                    except Exception as exc:
                        skipped.append((label, str(exc)))

                if args.include_writes:
                    raw_results.extend(benchmark_writes(profiler, max(1, args.write_count)))

                selected = _selected_tools(args, mcp_cases(scope))
                mcp_results: List[AggregatedCase] = []
                for label, tool_name, arguments, requirements in mcp_cases(scope):
                    if tool_name not in selected:
                        continue
                    missing = {name for name in requirements if not backends.get(name, False)}
                    if missing:
                        skipped.append((label, f"requires {', '.join(sorted(missing))}"))
                        continue
                    try:
                        mcp_results.append(
                            profiler.measure(
                                label,
                                functools.partial(_invoke_mcp_tool, tool_name, arguments),
                                runs=max(1, args.mcp_runs),
                            )
                        )
                    except Exception as exc:
                        skipped.append((label, str(exc)))

                nested = [
                    case for case in profiler.aggregated() if case.label.startswith("mcp:") and " > db:" in case.label
                ]
                _print_table("Raw DatabaseManager patterns", raw_results, args.top)
                _print_table("MCP tools", mcp_results, args.top)
                _print_table("Nested DB calls inside MCP tools", nested, args.top)

                if skipped:
                    print("\nSkipped")
                    print("-------")
                    for label, reason in skipped:
                        print(f"{_truncate(label, 56):<56} {_truncate(reason, 100)}")

                _maybe_explain(db, profiler, args)

                if args.json:
                    report = profiler.to_dict()
                    report["scope"] = {
                        "database": db._mask_url(db.database_url),
                        "conference": scope.conference,
                        "year": scope.year,
                        "topic": scope.topic,
                        "sample_papers": len(scope.paper_uids),
                    }
                    report["backends"] = backends
                    report["skipped"] = [{"case": label, "reason": reason} for label, reason in skipped]
                    Path(args.json).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
                    print(f"\nJSON report written to {args.json}")
            finally:
                restore_database()
        return 0
    finally:
        db.close()
        _restore_paper_db(previous_paper_db)


if __name__ == "__main__":
    raise SystemExit(main())

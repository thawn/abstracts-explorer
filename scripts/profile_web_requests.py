#!/usr/bin/env python3
"""
Replay representative web UI requests for profiling.

This is the repeatable workload used for local and staging comparisons.  It is
read-only: write endpoints (donations, exports) are intentionally excluded.

Usage::

    uv run python scripts/profile_web_requests.py \
        --base-url http://127.0.0.1:5001 --runs 3 --json local.json

    uv run python scripts/profile_web_requests.py \
        --base-url https://abstracts-staging.fz-rossendorf.de \
        --runs 3 --json staging.json --insecure
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import requests

DEFAULT_CONFERENCE = "NeurIPS"
DEFAULT_YEAR = 2025
DEFAULT_LIMIT = 10
DEFAULT_DISTANCE = 1.2

SEARCH_QUERIES: List[tuple[str, str]] = [
    ("search.learning", "learning"),
    ("search.large_language_models", "large language models"),
    ("search.author_name", "LeCun"),
    ("search.author_field", 'author:"LeCun"'),
    ("search.author_topic", 'authors:"LeCun" world model'),
    ("search.no_results", "xyzqwertyuiopasdfghjkl9999"),
]

CHAT_PROMPTS: List[tuple[str, str, bool]] = [
    ("chat.conference_topics", "What are the main topics at NeurIPS 2025?", False),
    ("chat.topic_evolution", "How has research on transformers evolved at NeurIPS over the years?", False),
    ("chat.search_papers", "Find papers about reinforcement learning at NeurIPS 2025.", False),
    ("chat.paper_details", "Who are the authors of 'Attention is All You Need'?", False),
    ("chat.topic_relevance", "How relevant is uncertainty quantification at NeurIPS 2025?", False),
    ("chat.cluster_visualization", "Show me a visual overview of NeurIPS 2025 clusters.", False),
    ("chat.summarize_interesting", "Summarize the papers I marked as interesting.", True),
    ("chat.recommend_related", "Based on my interesting papers, suggest other papers that may interest me.", True),
    ("chat.suggest_authors", "Who should I meet based on my interesting papers?", True),
    (
        "chat.orcid_search_terms",
        "Suggest search terms based on ORCID record 0000-0002-2315-9247.",
        False,
    ),
]


@dataclass
class Scenario:
    """A single HTTP scenario to replay."""

    name: str
    method: str
    path: str
    json_body: Optional[Dict[str, Any]] = None
    params: Optional[Dict[str, Any]] = None
    chat: bool = False
    note: str = ""


@dataclass
class Result:
    """One measured HTTP request."""

    scenario: str
    run: int
    method: str
    path: str
    status: Optional[int]
    elapsed_ms: float
    response_bytes: int = 0
    error: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="Deployment base URL.")
    parser.add_argument("--runs", type=int, default=3, help="Measured runs per scenario (default: 3).")
    parser.add_argument("--timeout", type=float, default=180.0, help="Per-request timeout in seconds (default: 180).")
    parser.add_argument("--json", help="Write the full report to this JSON file.")
    parser.add_argument("--insecure", action="store_true", help="Disable TLS certificate verification.")
    parser.add_argument(
        "--scenarios",
        help="Comma-separated substrings; only matching scenario names run (default: all).",
    )
    parser.add_argument(
        "--conference",
        default=DEFAULT_CONFERENCE,
        help=f"Conference used by scenarios (default: {DEFAULT_CONFERENCE}).",
    )
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR, help=f"Year (default: {DEFAULT_YEAR}).")
    parser.add_argument("--skip-chat", action="store_true", help="Skip RAG chat scenarios.")
    parser.add_argument("--skip-clustering", action="store_true", help="Skip clustering scenarios.")
    return parser.parse_args(argv)


def _percentile(values: Iterable[float], fraction: float) -> float:
    """Return a linearly interpolated percentile."""
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


def _request(
    session: requests.Session,
    base_url: str,
    scenario: Scenario,
    timeout: float,
    verify: bool,
    run: int,
) -> Result:
    """Execute one scenario and measure it."""
    url = f"{base_url}{scenario.path}"
    started = time.perf_counter()
    try:
        response = session.request(
            scenario.method,
            url,
            json=scenario.json_body,
            params=scenario.params,
            timeout=timeout,
            verify=verify,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        body = response.content
        details: Dict[str, Any] = {}
        if response.headers.get("content-type", "").startswith("application/json"):
            try:
                payload = response.json()
                if isinstance(payload, dict):
                    if "count" in payload:
                        details["count"] = payload["count"]
                    if "total_similar" in payload:
                        details["total_similar"] = payload["total_similar"]
                    metadata = payload.get("metadata")
                    if isinstance(metadata, dict):
                        details["tools_executed"] = metadata.get("tools_executed", [])
                        details["n_papers"] = metadata.get("n_papers")
            except ValueError:
                pass
        return Result(
            scenario=scenario.name,
            run=run,
            method=scenario.method,
            path=scenario.path,
            status=response.status_code,
            elapsed_ms=elapsed_ms,
            response_bytes=len(body),
            details=details,
        )
    except requests.exceptions.RequestException as exc:
        return Result(
            scenario=scenario.name,
            run=run,
            method=scenario.method,
            path=scenario.path,
            status=None,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            error=str(exc),
        )


def _search(
    session: requests.Session,
    base_url: str,
    body: Dict[str, Any],
    timeout: float,
    verify: bool,
) -> Optional[Dict[str, Any]]:
    """Run one search and return its JSON payload when available."""
    try:
        response = session.post(f"{base_url}/api/search", json=body, timeout=timeout, verify=verify)
        if response.status_code != 200:
            return None
        payload = response.json()
        return payload if isinstance(payload, dict) else None
    except requests.exceptions.RequestException:
        return None


def _discover_context(
    session: requests.Session,
    args: argparse.Namespace,
    verify: bool,
) -> Dict[str, Any]:
    """Discover paper UIDs and priorities for detail/batch/interesting-paper scenarios."""
    payload = _search(
        session,
        args.base_url,
        {
            "query": "learning",
            "use_embeddings": True,
            "limit": DEFAULT_LIMIT,
            "years": [args.year],
            "conferences": [args.conference],
            "distance_threshold": DEFAULT_DISTANCE,
        },
        args.timeout,
        verify,
    )
    uids: List[str] = []
    if payload:
        for paper in payload.get("papers") or []:
            uid = paper.get("uid") if isinstance(paper, dict) else None
            if uid:
                uids.append(str(uid))
    priorities = {
        uid: {"priority": 5 if index < 2 else 3, "searchTerm": "learning"} for index, uid in enumerate(uids[:5])
    }
    return {"uids": uids, "priorities": priorities}


def build_scenarios(args: argparse.Namespace, context: Dict[str, Any]) -> List[Scenario]:
    """Build the scenario list for the selected deployment."""
    conference = args.conference
    year = args.year
    scenarios: List[Scenario] = [
        Scenario("page.index", "GET", "/"),
        Scenario("filters.available", "GET", "/api/available-filters"),
        Scenario(
            "filters.conference_year",
            "GET",
            "/api/filters",
            params={"conference": conference, "year": year},
        ),
        Scenario(
            "stats.conference_year",
            "GET",
            "/api/stats",
            params={"conference": conference, "year": year},
        ),
        Scenario("embedding_model_check", "GET", "/api/embedding-model-check"),
        Scenario(
            "papers_per_year",
            "GET",
            "/api/papers-per-year",
            params={"conference": conference},
        ),
    ]

    for name, query in SEARCH_QUERIES:
        scenarios.append(
            Scenario(
                name,
                "POST",
                "/api/search",
                json_body={
                    "query": query,
                    "use_embeddings": True,
                    "limit": DEFAULT_LIMIT,
                    "years": [year],
                    "conferences": [conference],
                    "distance_threshold": DEFAULT_DISTANCE,
                },
            )
        )

    uids = context.get("uids") or []
    if uids:
        scenarios.append(Scenario("paper.detail", "GET", f"/api/paper/{uids[0]}"))
        scenarios.append(
            Scenario(
                "papers.batch",
                "POST",
                "/api/papers/batch",
                json_body={"paper_ids": uids[:10]},
            )
        )

    if not args.skip_clustering:
        scenarios.extend(
            [
                Scenario(
                    "clusters.compute",
                    "POST",
                    "/api/clusters/compute",
                    json_body={"conferences": [conference], "years": [year]},
                ),
                Scenario(
                    "clusters.search",
                    "POST",
                    "/api/clusters/search",
                    json_body={
                        "query": "reinforcement learning",
                        "distance": DEFAULT_DISTANCE,
                        "conferences": [conference],
                        "years": [year],
                    },
                ),
                Scenario(
                    "topic_evolution",
                    "POST",
                    "/api/topic-evolution",
                    json_body={"topic_keywords": "transformers", "conferences": [conference]},
                ),
            ]
        )

    if not args.skip_chat:
        priorities = context.get("priorities") or {}
        for name, prompt, use_priorities in CHAT_PROMPTS:
            body: Dict[str, Any] = {
                "message": prompt,
                "n_papers": 5,
                "years": [year],
                "conferences": [conference],
            }
            if use_priorities and priorities:
                body["paper_priorities"] = priorities
            scenarios.append(Scenario(name, "POST", "/api/chat", json_body=body, chat=True))

    selected = args.scenarios
    if selected:
        needles = [item.strip() for item in selected.split(",") if item.strip()]
        scenarios = [scenario for scenario in scenarios if any(needle in scenario.name for needle in needles)]
    return scenarios


def _reset_chat(session: requests.Session, base_url: str, timeout: float, verify: bool) -> None:
    """Reset server-side chat state between chat runs."""
    try:
        session.post(f"{base_url}/api/chat/reset", timeout=min(timeout, 30.0), verify=verify)
    except requests.exceptions.RequestException:
        pass


def summarize(results: List[Result]) -> List[Dict[str, Any]]:
    """Aggregate measured results by scenario."""
    grouped: Dict[str, List[Result]] = {}
    for result in results:
        grouped.setdefault(result.scenario, []).append(result)

    summary: List[Dict[str, Any]] = []
    for scenario, items in grouped.items():
        elapsed = [item.elapsed_ms for item in items if item.status is not None]
        successes = [item for item in items if item.status is not None and 200 <= item.status < 400]
        summary.append(
            {
                "scenario": scenario,
                "runs": len(items),
                "successes": len(successes),
                "errors": sum(1 for item in items if item.status is None or item.status >= 400),
                "statuses": sorted({item.status for item in items if item.status is not None}),
                "elapsed_ms": {
                    "min": min(elapsed) if elapsed else 0.0,
                    "median": statistics.median(elapsed) if elapsed else 0.0,
                    "p95": _percentile(elapsed, 0.95) if elapsed else 0.0,
                    "mean": statistics.fmean(elapsed) if elapsed else 0.0,
                },
                "response_bytes_mean": (
                    statistics.fmean([item.response_bytes for item in successes]) if successes else 0.0
                ),
            }
        )
    summary.sort(key=lambda item: item["elapsed_ms"]["median"], reverse=True)
    return summary


def print_summary(summary: List[Dict[str, Any]]) -> None:
    """Print a compact table sorted by median wall time."""
    print(f"\n{'scenario':<28} {'runs':>4} {'ok':>4} {'err':>4} {'median ms':>10} {'p95 ms':>10} {'bytes':>10}")
    print("-" * 78)
    for row in summary:
        elapsed = row["elapsed_ms"]
        print(
            f"{row['scenario']:<28} {row['runs']:>4} {row['successes']:>4} {row['errors']:>4} "
            f"{elapsed['median']:>10.0f} {elapsed['p95']:>10.0f} {row['response_bytes_mean']:>10.0f}"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the profiling workload."""
    args = parse_args(argv)
    base_url = args.base_url.rstrip("/")
    verify = not args.insecure
    session = requests.Session()
    session.headers["User-Agent"] = "abstracts-explorer-profiler"

    context = _discover_context(session, args, verify)
    scenarios = build_scenarios(args, context)

    print(
        f"Profiling {base_url}\n"
        f"conference={args.conference} year={args.year} runs={args.runs} "
        f"scenarios={len(scenarios)} discovered_uids={len(context.get('uids') or [])}"
    )

    results: List[Result] = []
    for scenario in scenarios:
        for run in range(max(1, args.runs)):
            if scenario.chat:
                _reset_chat(session, base_url, args.timeout, verify)
            result = _request(session, base_url, scenario, args.timeout, verify, run)
            results.append(result)
            status = result.status if result.status is not None else result.error
            print(f"  {scenario.name:<28} run {run + 1}/{args.runs}: {result.elapsed_ms:8.0f} ms  {status}")

    summary = summarize(results)
    print_summary(summary)

    report = {
        "base_url": base_url,
        "conference": args.conference,
        "year": args.year,
        "runs": args.runs,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "summary": summary,
        "results": [
            {
                "scenario": result.scenario,
                "run": result.run,
                "method": result.method,
                "path": result.path,
                "status": result.status,
                "elapsed_ms": round(result.elapsed_ms, 3),
                "response_bytes": result.response_bytes,
                "error": result.error,
                "details": result.details,
            }
            for result in results
        ],
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nReport written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

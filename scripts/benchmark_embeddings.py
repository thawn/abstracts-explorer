#!/usr/bin/env python3
"""
Benchmark embedding batch sizes against the configured backend.

The script reads real abstracts from the configured paper database, sends them
to the configured OpenAI-compatible embedding backend in single requests of
varying size, and reports latency, throughput, and failures.  The configured
rate limit is respected by default.  Each batch is sent directly to the backend
so a rejected or timed-out request is reported instead of triggering the
production per-text fallback, which would turn one failure into thousands of
throttled single requests.  Client retries are disabled as well, so each batch
size is attempted exactly once per repetition and failures are only logged.
HTTP 429 responses are not retried either, but the script honours the
``Retry-After`` header (or the configured ``LLM_RATE_LIMIT_DELAY``) and backs
off before sending the next batch.

Usage::

    uv run python scripts/benchmark_embeddings.py
    uv run python scripts/benchmark_embeddings.py --batch-sizes 64 256 1024 4096
    uv run python scripts/benchmark_embeddings.py --conference NeurIPS --year 2025
    uv run python scripts/benchmark_embeddings.py --json embedding-benchmark.json

The embedding vectors are discarded; nothing is written to ChromaDB.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from abstracts_explorer.config import get_config
from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.embeddings import EmbeddingsManager
from abstracts_explorer.embeddings.tokens import count_tokens

DEFAULT_BATCH_SIZES = [16, 32, 64, 128, 256, 512, 1024]


@dataclass
class BatchResult:
    """
    Outcome of benchmarking one batch size.

    Attributes
    ----------
    batch_size : int
        Number of texts sent in the request.
    status : str
        ``"ok"`` when every repetition succeeded, otherwise ``"error"``.
    seconds : float or None
        Median wall-clock request duration in seconds.
    vectors : int
        Number of embedding vectors returned by the last successful request.
    texts_per_second : float or None
        Median throughput in embedded texts per second.
    tokens_per_second : float or None
        Median throughput in input tokens per second, when a tokenizer exists.
    total_chars : int
        Number of input characters in the batch.
    total_tokens : int or None
        Number of input tokens in the batch, when a tokenizer exists.
    error : str or None
        Human-readable error description when the request failed.
    backoff_seconds : float or None
        Cooldown to apply before the next batch after a 429 response.
    """

    batch_size: int
    status: str
    seconds: Optional[float]
    vectors: int
    texts_per_second: Optional[float]
    tokens_per_second: Optional[float]
    total_chars: int
    total_tokens: Optional[int]
    error: Optional[str] = None
    backoff_seconds: Optional[float] = None


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark embedding batch sizes against the configured backend.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_BATCH_SIZES,
        help="Batch sizes to benchmark.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Timed requests per batch size.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="Abstracts to load; 0 uses the largest batch size.",
    )
    parser.add_argument("--conference", help="Only use papers from this conference.")
    parser.add_argument("--year", type=int, help="Only use papers from this year.")
    parser.add_argument(
        "--paper-db",
        help="Override PAPER_DB for this command only. Accepts a SQLite path or full database URL.",
    )
    parser.add_argument("--backend-url", help="Override LLM_BACKEND_URL.")
    parser.add_argument("--model", help="Override EMBEDDING_MODEL.")
    parser.add_argument(
        "--requests-per-minute",
        type=int,
        default=None,
        help="Override REQUESTS_PER_MINUTE; 0 disables rate limiting. Defaults to the configured value.",
    )
    parser.add_argument("--json", help="Write the full report to this JSON file.")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip the backend connection test.")
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


def _load_texts(
    limit: int,
    conference: Optional[str],
    year: Optional[int],
) -> List[str]:
    """
    Load embedding texts from the configured paper database.

    Parameters
    ----------
    limit : int
        Maximum number of papers to load.
    conference : str or None
        Optional conference filter.
    year : int or None
        Optional year filter.

    Returns
    -------
    list of str
        Title and abstract text for each usable paper.
    """
    where_filters: Dict[str, Any] = {}
    if conference:
        where_filters["conference"] = conference
    if year is not None:
        where_filters["year"] = year

    with DatabaseManager() as db:
        rows = db.get_paper(where_filters or None, limit=limit)

    texts: List[str] = []
    for paper in rows:
        try:
            texts.append(EmbeddingsManager.embedding_text_from_paper(paper))
        except ValueError:
            continue
    return texts


def _cycle(texts: Sequence[str], size: int) -> List[str]:
    """
    Return exactly *size* texts, cycling through *texts* when needed.

    Parameters
    ----------
    texts : sequence of str
        Source texts.
    size : int
        Desired number of texts.

    Returns
    -------
    list of str
        Texts of length *size*.

    Raises
    ------
    ValueError
        If *texts* is empty.
    """
    if not texts:
        raise ValueError("No embedding texts available")
    return [texts[index % len(texts)] for index in range(size)]


def _retry_after_seconds(exc: Exception) -> Optional[float]:
    """
    Extract a ``Retry-After`` delay from an API exception.

    Parameters
    ----------
    exc : Exception
        Exception raised by the embedding client.

    Returns
    -------
    float or None
        Delay in seconds, or None when the header is missing or unparseable.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after") or headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        pass
    try:
        retry_at = parsedate_to_datetime(str(value))
    except (TypeError, ValueError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


def _error_detail(exc: Exception) -> str:
    """
    Return the most specific message available from an API exception.

    Parameters
    ----------
    exc : Exception
        Exception raised by the embedding client.

    Returns
    -------
    str
        Backend ``error.message`` when present, otherwise the exception text.
    """
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message:
                return message
        message = body.get("message")
        if isinstance(message, str) and message:
            return message
    if isinstance(body, str) and body:
        return body
    return str(exc)


def _describe_error(exc: Exception) -> Tuple[str, Optional[float]]:
    """
    Build a correct, concise error message and any required backoff.

    Parameters
    ----------
    exc : Exception
        Exception raised by the embedding client.

    Returns
    -------
    tuple of (str, float or None)
        Error message and the cooldown to apply before the next batch. The
        cooldown is only set for HTTP 429 responses.
    """
    status = getattr(exc, "status_code", None)
    detail = _error_detail(exc)

    if status == 429:
        retry_after = _retry_after_seconds(exc)
        delay = retry_after if retry_after is not None else get_config().llm_rate_limit_delay
        message = "HTTP 429 Too Many Requests"
        if retry_after is not None:
            message += f" (Retry-After {retry_after:.1f}s)"
        if detail:
            message += f": {detail}"
        return message, delay

    if "Timeout" in type(exc).__name__:
        return f"request timed out (configured limit {get_config().llm_request_timeout}s)", None

    if isinstance(status, int):
        response = getattr(exc, "response", None)
        reason = getattr(response, "reason_phrase", "") or ""
        message = f"HTTP {status}" + (f" {reason}" if reason else "")
        if detail:
            message += f": {detail}"
        return message, None

    if "Connection" in type(exc).__name__:
        return f"connection error: {detail}" if detail else "connection error", None

    name = type(exc).__name__
    return f"{name}: {detail}" if detail else name, None


def _benchmark_size(
    em: EmbeddingsManager,
    texts: Sequence[str],
    batch_size: int,
    repetitions: int,
) -> BatchResult:
    """
    Send one batch size to the backend and measure it.

    Parameters
    ----------
    em : EmbeddingsManager
        Manager providing the configured backend client and token budget.
    texts : sequence of str
        Source texts used to fill the batch.
    batch_size : int
        Number of texts per request.
    repetitions : int
        Number of timed requests.

    Returns
    -------
    BatchResult
        Measured latency, throughput, and failure information.
    """
    prepared = [em._prepare_embedding_input(text) for text in _cycle(texts, batch_size)]
    total_chars = sum(len(text) for text in prepared)
    token_counts = [count_tokens(text) for text in prepared]
    total_tokens = None if any(count is None for count in token_counts) else sum(token_counts)  # type: ignore[arg-type]

    durations: List[float] = []
    vectors = 0
    error: Optional[str] = None
    backoff_seconds: Optional[float] = None

    for _ in range(max(1, repetitions)):
        started = time.perf_counter()
        try:
            response = em.openai_client.embeddings.create(model=em.model_name, input=prepared)
        except Exception as exc:
            error, backoff_seconds = _describe_error(exc)
            break
        durations.append(time.perf_counter() - started)
        vectors = len(response.data)
        indices = sorted(item.index for item in response.data)
        if indices != list(range(len(prepared))):
            error = f"invalid embedding indices ({len(indices)} returned for {len(prepared)} inputs)"
            break

    if error is not None or not durations:
        return BatchResult(
            batch_size=batch_size,
            status="error",
            seconds=min(durations) if durations else None,
            vectors=vectors,
            texts_per_second=None,
            tokens_per_second=None,
            total_chars=total_chars,
            total_tokens=total_tokens,
            error=error or "request failed",
            backoff_seconds=backoff_seconds,
        )

    seconds = statistics.median(durations)
    texts_per_second = batch_size / seconds if seconds > 0 else None
    tokens_per_second = total_tokens / seconds if total_tokens and seconds > 0 else None

    return BatchResult(
        batch_size=batch_size,
        status="ok",
        seconds=seconds,
        vectors=vectors,
        texts_per_second=texts_per_second,
        tokens_per_second=tokens_per_second,
        total_chars=total_chars,
        total_tokens=total_tokens,
    )


def _print_report(
    results: Sequence[BatchResult],
    backend_url: str,
    model_name: str,
    requests_per_minute: int,
    sample_count: int,
    has_tokenizer: bool,
) -> None:
    """Print the benchmark header and result table."""
    rate_limit = f"{requests_per_minute} req/min" if requests_per_minute > 0 else "disabled"
    print("Embedding batch-size benchmark")
    print(f"  Backend:     {backend_url}")
    print(f"  Model:       {model_name}")
    print(f"  Rate limit:  {rate_limit}")
    print(f"  Abstracts:   {sample_count} loaded")
    print(f"  Tokenizer:   {'tiktoken' if has_tokenizer else 'unavailable (tokens/s omitted)'}")
    print()
    header = (
        f"{'batch':>8}  {'status':>6}  {'seconds':>9}  {'vectors':>8}  "
        f"{'texts/s':>9}  {'tokens/s':>9}  {'chars':>12}  error"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        seconds = f"{result.seconds:.3f}" if result.seconds is not None else "-"
        texts_per_second = f"{result.texts_per_second:.2f}" if result.texts_per_second is not None else "-"
        tokens_per_second = f"{result.tokens_per_second:.0f}" if result.tokens_per_second is not None else "-"
        print(
            f"{result.batch_size:>8}  {result.status:>6}  {seconds:>9}  {result.vectors:>8}  "
            f"{texts_per_second:>9}  {tokens_per_second:>9}  {result.total_chars:>12}  "
            f"{result.error or ''}"
        )


def main() -> int:
    """
    Run the embedding batch-size benchmark.

    Returns
    -------
    int
        Exit code (0 on success, 1 when no results could be produced).
    """
    args = parse_args()
    if any(size <= 0 for size in args.batch_sizes):
        print("Batch sizes must be positive integers.", flush=True)
        return 1

    previous_paper_db = _set_paper_db(args.paper_db)
    try:
        config = get_config()
        batch_sizes = sorted(dict.fromkeys(args.batch_sizes))
        sample_limit = args.samples if args.samples > 0 else max(batch_sizes)
        requests_per_minute = (
            args.requests_per_minute if args.requests_per_minute is not None else config.requests_per_minute
        )

        em = EmbeddingsManager(
            backend_url=args.backend_url,
            model_name=args.model,
            requests_per_minute=requests_per_minute,
        )
        em.openai_client.max_retries = 0

        if not args.skip_preflight and not em.test_api_connection(verbose=True):
            em.close()
            return 1

        texts = _load_texts(sample_limit, args.conference, args.year)
        if not texts:
            print("No usable abstracts found in the configured database.", flush=True)
            em.close()
            return 1

        has_tokenizer = count_tokens(texts[0]) is not None
        results: List[BatchResult] = []
        for batch_size in batch_sizes:
            result = _benchmark_size(em, texts, batch_size, args.repetitions)
            results.append(result)
            if result.status == "ok":
                tokens_per_second = (
                    f", {result.tokens_per_second:.0f} tokens/s" if result.tokens_per_second is not None else ""
                )
                print(
                    f"batch {result.batch_size}: ok in {result.seconds:.3f}s, "
                    f"{result.texts_per_second:.2f} embeddings/s{tokens_per_second}",
                    flush=True,
                )
            else:
                seconds = f" after {result.seconds:.3f}s" if result.seconds is not None else ""
                print(f"batch {result.batch_size}: failed{seconds}: {result.error}", flush=True)
                if result.backoff_seconds is not None:
                    print(
                        f"  backing off {result.backoff_seconds:.1f}s before the next batch",
                        flush=True,
                    )
                    time.sleep(result.backoff_seconds)

        _print_report(
            results,
            em.backend_url,
            em.model_name,
            requests_per_minute,
            len(texts),
            has_tokenizer,
        )

        if args.json:
            payload = {
                "backend_url": em.backend_url,
                "model": em.model_name,
                "requests_per_minute": requests_per_minute,
                "repetitions": args.repetitions,
                "sample_count": len(texts),
                "has_tokenizer": has_tokenizer,
                "results": [asdict(result) for result in results],
            }
            Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"\nReport written to {args.json}")

        em.close()
        return 0 if any(result.status == "ok" for result in results) else 1
    finally:
        _restore_paper_db(previous_paper_db)


if __name__ == "__main__":
    raise SystemExit(main())

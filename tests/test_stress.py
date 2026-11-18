"""
Stress tests for a live Abstracts Explorer deployment.

The test discovers the maximum number of parallel RAG chat requests a server
can handle.  It sends waves of ``/api/chat`` requests with increasing
concurrency (1, 2, 4, 8, ...) until at least one request in a wave fails, then
reports the last fully successful wave.

Usage
-----
Run against a deployment::

    uv run pytest tests/test_stress.py -m stress --server-url http://localhost:5000 -s

Alternatively, set the ``SERVER_URL`` environment variable::

    SERVER_URL=https://staging.example.com uv run pytest tests/test_stress.py -m stress -s

Notes
-----
* The test is marked with ``@pytest.mark.stress`` and is excluded from the
  default test run.
* A wave counts as successful only when **every** request returns HTTP 200.
* The concurrency doubles up to the safety cap ``_MAX_PARALLEL`` so that a
  server that never fails cannot make the test run indefinitely.
"""

import concurrent.futures
import time
from collections.abc import Callable
from dataclasses import dataclass

import pytest
import requests

_START_PARALLEL = 1
_MAX_PARALLEL = 256
_REQUEST_TIMEOUT = 180
_WAVE_TIMEOUT = _REQUEST_TIMEOUT + 60

_RAG_QUERIES = [
    "Find papers about reinforcement learning at NeurIPS 2025.",
    "Find papers about large language models at NeurIPS 2025.",
    "Find papers about computer vision at NeurIPS 2025.",
    "Find papers about generative adversarial networks at NeurIPS 2025.",
    "Find papers about transformers at NeurIPS 2025.",
    "Find papers about graph neural networks at NeurIPS 2025.",
    "Find papers about federated learning at NeurIPS 2025.",
    "Find papers about causal inference at NeurIPS 2025.",
    "Find papers about robust machine learning at NeurIPS 2025.",
    "Find papers about unsupervised learning at NeurIPS 2025.",
]

pytestmark = [
    pytest.mark.stress,
    pytest.mark.skipif(
        "not (config.getoption('--server-url') or os.environ.get('SERVER_URL'))",
        reason="Server URL not provided. Use --server-url or set SERVER_URL env var.",
    ),
]


@dataclass
class WaveResult:
    """
    Result of one wave of parallel chat requests.

    Parameters
    ----------
    concurrency : int
        Number of parallel requests in the wave.
    results : list[tuple[int, str | None]]
        One ``(status_code, error)`` tuple per request.  ``status_code`` is
        ``-1`` when the request raised an exception or timed out.
    elapsed : float
        Wall-clock duration of the wave in seconds.
    """

    concurrency: int
    results: list[tuple[int, str | None]]
    elapsed: float

    @property
    def successes(self) -> int:
        """Return the number of requests that completed with HTTP 200."""
        return sum(1 for status, _ in self.results if status == 200)

    @property
    def failures(self) -> int:
        """Return the number of requests that did not complete with HTTP 200."""
        return self.concurrency - self.successes

    @property
    def succeeded(self) -> bool:
        """Return whether every request in the wave completed with HTTP 200."""
        return self.failures == 0


def _check_health(server_url: str) -> None:
    """
    Skip the test when the target deployment is not reachable or unhealthy.

    Parameters
    ----------
    server_url : str
        Base URL of the deployment under test.
    """
    try:
        response = requests.get(f"{server_url}/health", timeout=10, verify=False)
    except requests.RequestException as exc:
        pytest.skip(f"Server at {server_url} is not reachable: {exc}")

    if response.status_code != 200:
        pytest.skip(f"Server at {server_url} is unhealthy: HTTP {response.status_code}")


def _chat_request(server_url: str, query: str) -> tuple[int, str | None]:
    """
    Send a single RAG chat request.

    Parameters
    ----------
    server_url : str
        Base URL of the deployment under test.
    query : str
        Chat message to send.

    Returns
    -------
    tuple[int, str | None]
        HTTP status code and an error description for non-200 responses.
        The status is ``-1`` when the request raised an exception.
    """
    try:
        response = requests.post(
            f"{server_url}/api/chat",
            json={"message": query, "reset": True},
            timeout=_REQUEST_TIMEOUT,
            verify=False,
        )
    except requests.RequestException as exc:
        return -1, str(exc)

    if response.status_code == 200:
        return 200, None
    return response.status_code, response.text[:500]


def _run_wave(server_url: str, concurrency: int) -> WaveResult:
    """
    Send one wave of parallel chat requests.

    Parameters
    ----------
    server_url : str
        Base URL of the deployment under test.
    concurrency : int
        Number of parallel requests to send.

    Returns
    -------
    WaveResult
        Status codes, errors, and elapsed time for the wave.
    """
    queries = [f"{_RAG_QUERIES[index % len(_RAG_QUERIES)]} Request {index}." for index in range(concurrency)]
    results: list[tuple[int, str | None]] = []
    start = time.monotonic()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_chat_request, server_url, query) for query in queries]
        try:
            for future in concurrent.futures.as_completed(futures, timeout=_WAVE_TIMEOUT):
                results.append(future.result())
        except concurrent.futures.TimeoutError:
            for future in futures:
                if not future.done():
                    future.cancel()
            results.extend((-1, "wave timeout") for _ in range(concurrency - len(results)))

    return WaveResult(concurrency=concurrency, results=results, elapsed=time.monotonic() - start)


def test_parallel_chat_capacity(server_url: str, record_property: Callable[[str, object], None]) -> None:
    """
    Find the maximum number of parallel RAG chat requests the server handles.

    Parameters
    ----------
    server_url : str
        Base URL of the deployment under test, provided by the ``server_url``
        fixture.
    record_property : Callable[[str, object], None]
        Pytest fixture for attaching the measured limit to the test report.
    """
    _check_health(server_url)

    try:
        requests.post(f"{server_url}/api/chat/reset", timeout=10, verify=False)
    except requests.RequestException:
        pass

    print(f"\nStress test target: {server_url}")
    print(f"{'concurrency':>11} | {'successes':>9} | {'failures':>8} | {'elapsed (s)':>11}")
    print("-" * 50)

    last_success = 0
    first_failure: int | None = None
    concurrency = _START_PARALLEL

    while concurrency <= _MAX_PARALLEL:
        result = _run_wave(server_url, concurrency)
        print(f"{result.concurrency:>11} | {result.successes:>9} | {result.failures:>8} | {result.elapsed:>11.2f}")
        for status, error in result.results:
            if status != 200:
                print(f"  [ERR] status={status}, error={error}")

        if not result.succeeded:
            first_failure = concurrency
            break

        last_success = concurrency
        concurrency *= 2

    record_property("stress_max_parallel_requests", last_success)
    record_property("stress_first_failure", first_failure or "")

    if last_success == 0:
        pytest.fail("Server could not complete a single /api/chat request.")

    if first_failure is None:
        print(f"\nSTRESS RESULT: server handled at least {last_success} parallel chat requests (cap reached)")
    else:
        print(f"\nSTRESS RESULT: server handled {last_success} parallel chat requests")
        print(f"First failure at {first_failure} parallel chat requests")

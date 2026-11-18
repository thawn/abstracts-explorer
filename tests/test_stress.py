"""
Stress tests for a live Abstracts Explorer deployment.

The HTTP test discovers the maximum number of parallel RAG chat requests a
server can handle.  The browser test does the same through the full web UI,
using one Playwright browser context per simulated user.  Both tests send waves
of requests with increasing concurrency (1, 2, 4, 8, ...) until at least one
request in a wave fails, then report the last fully successful wave.

Usage
-----
Run against a deployment::

    uv run pytest tests/test_stress.py -m stress --server-url http://localhost:5000 -s

Alternatively, set the ``SERVER_URL`` environment variable::

    SERVER_URL=https://staging.example.com uv run pytest tests/test_stress.py -m stress -s

Run only the browser UI stress test::

    uv run pytest tests/test_stress.py -m stress --server-url http://localhost:5000 -k browser -s

Notes
-----
* The tests are marked with ``@pytest.mark.stress`` and are excluded from the
  default test run.
* A wave counts as successful only when **every** request or browser session in
  the wave succeeds.
* The HTTP concurrency doubles up to ``_MAX_PARALLEL`` and the browser
  concurrency doubles up to ``_MAX_BROWSER_PARALLEL`` so that a server that
  never fails cannot make the tests run indefinitely.
* The browser test requires Chromium to be installed with
  ``uv run playwright install chromium``.
"""

import asyncio
import concurrent.futures
import time
from collections.abc import Callable
from dataclasses import dataclass

import pytest
import requests
from playwright.async_api import (
    Browser,
    Error as PlaywrightError,
    Page,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)

_START_PARALLEL = 1
_MAX_PARALLEL = 256
_MAX_BROWSER_PARALLEL = 32
_REQUEST_TIMEOUT = 180
_WAVE_TIMEOUT = _REQUEST_TIMEOUT + 60
_BROWSER_NAV_TIMEOUT = 30_000
_BROWSER_CHAT_TIMEOUT = 120_000
_BROWSER_SESSION_TIMEOUT = 180
_DEFAULT_CONFERENCE = "NeurIPS"
_DEFAULT_YEAR = "2025"

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


@dataclass
class BrowserWaveResult:
    """
    Result of one wave of parallel browser chat sessions.

    Parameters
    ----------
    concurrency : int
        Number of parallel browser sessions in the wave.
    results : list[tuple[bool, str | None]]
        One ``(succeeded, error)`` tuple per session.
    elapsed : float
        Wall-clock duration of the wave in seconds.
    """

    concurrency: int
    results: list[tuple[bool, str | None]]
    elapsed: float

    @property
    def successes(self) -> int:
        """Return the number of sessions that completed the UI flow."""
        return sum(1 for succeeded, _ in self.results if succeeded)

    @property
    def failures(self) -> int:
        """Return the number of sessions that did not complete the UI flow."""
        return self.concurrency - self.successes

    @property
    def succeeded(self) -> bool:
        """Return whether every session in the wave completed the UI flow."""
        return self.failures == 0


async def _select_conference_and_year_async(
    page: Page,
    conference: str = _DEFAULT_CONFERENCE,
    year: str = _DEFAULT_YEAR,
    timeout: int = 15_000,
) -> None:
    """
    Select *conference* and *year* from the header dropdowns.

    Parameters
    ----------
    page : Page
        The Playwright page.
    conference : str, optional
        Conference name to select (default: ``"NeurIPS"``).
    year : str, optional
        Year to select as a string (default: ``"2025"``).
    timeout : int, optional
        Maximum milliseconds to wait for the dropdowns (default: 15000).
    """
    await page.wait_for_function(
        "sel => [...document.querySelectorAll('#conference-selector option')].some(o => o.value === sel)",
        arg=conference,
        timeout=timeout,
    )
    await page.locator("#conference-selector").select_option(value=conference)
    await page.evaluate("handleConferenceChange()")

    await page.wait_for_function(
        "sel => [...document.querySelectorAll('#year-selector option')].some(o => o.value === sel)",
        arg=year,
        timeout=timeout,
    )
    await page.locator("#year-selector").select_option(value=year)
    await page.evaluate("handleYearChange()")


async def _browser_chat_session(browser: Browser, server_url: str, query: str) -> tuple[bool, str | None]:
    """
    Exercise the complete browser chat flow for one simulated user.

    Parameters
    ----------
    browser : Browser
        Shared Chromium browser instance.
    server_url : str
        Base URL of the deployment under test.
    query : str
        Chat message to send.

    Returns
    -------
    tuple[bool, str | None]
        Whether the session completed successfully and an error description
        for failed sessions.
    """
    context = await browser.new_context(ignore_https_errors=True)
    try:
        page = await context.new_page()
        page.set_default_navigation_timeout(_BROWSER_NAV_TIMEOUT)
        await page.goto(server_url, wait_until="domcontentloaded", timeout=_BROWSER_NAV_TIMEOUT)
        await _select_conference_and_year_async(page)

        await page.locator("#tab-chat").click()
        await page.wait_for_timeout(500)

        chat_input = page.locator("#chat-input")
        await chat_input.fill(query)
        async with page.expect_response(
            lambda response: response.request.method == "POST" and response.url.endswith("/api/chat"),
            timeout=_BROWSER_CHAT_TIMEOUT,
        ) as response_info:
            await chat_input.press("Enter")

        response = await response_info.value
        if response.status != 200:
            return False, f"chat API returned HTTP {response.status}"

        try:
            payload = await response.json()
        except Exception:
            payload = None
        if isinstance(payload, dict) and payload.get("error"):
            return False, str(payload["error"])[:200]

        await page.wait_for_function(
            """() => {
                const msgs = document.querySelectorAll(".chat-message[data-role='assistant']");
                if (!msgs.length) {
                    return false;
                }
                return msgs[msgs.length - 1].querySelectorAll(".spinner").length === 0;
            }""",
            timeout=_BROWSER_CHAT_TIMEOUT,
        )

        text = (await page.locator(".chat-message[data-role='assistant']").last.inner_text()).strip()
        if not text:
            return False, "assistant response was empty"
        if text.startswith("Sorry, I couldn't complete that request."):
            return False, text[:200]
        return True, None
    except PlaywrightTimeoutError as exc:
        return False, f"timeout: {exc}"
    except PlaywrightError as exc:
        return False, str(exc)
    finally:
        await context.close()


async def _run_browser_worker(browser: Browser, server_url: str, query: str) -> tuple[bool, str | None]:
    """
    Run one browser chat session with an overall session timeout.

    Parameters
    ----------
    browser : Browser
        Shared Chromium browser instance.
    server_url : str
        Base URL of the deployment under test.
    query : str
        Chat message to send.

    Returns
    -------
    tuple[bool, str | None]
        Whether the session completed successfully and an error description
        for failed sessions.
    """
    try:
        return await asyncio.wait_for(
            _browser_chat_session(browser, server_url, query),
            timeout=_BROWSER_SESSION_TIMEOUT,
        )
    except asyncio.TimeoutError:
        return False, "browser session timeout"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


async def _run_browser_wave(browser: Browser, server_url: str, concurrency: int) -> BrowserWaveResult:
    """
    Run one wave of parallel browser chat sessions.

    Parameters
    ----------
    browser : Browser
        Shared Chromium browser instance.
    server_url : str
        Base URL of the deployment under test.
    concurrency : int
        Number of parallel browser sessions to run.

    Returns
    -------
    BrowserWaveResult
        Session outcomes and elapsed time for the wave.
    """
    queries = [f"{_RAG_QUERIES[index % len(_RAG_QUERIES)]} Browser request {index}." for index in range(concurrency)]
    start = time.monotonic()
    results = await asyncio.gather(*(_run_browser_worker(browser, server_url, query) for query in queries))
    return BrowserWaveResult(
        concurrency=concurrency,
        results=list(results),
        elapsed=time.monotonic() - start,
    )


async def _run_browser_stress(server_url: str, headed: bool) -> tuple[int, int | None]:
    """
    Launch Chromium and probe increasing parallel browser chat sessions.

    Parameters
    ----------
    server_url : str
        Base URL of the deployment under test.
    headed : bool
        Whether to show the browser window.

    Returns
    -------
    tuple[int, int | None]
        The last fully successful concurrency and the first failing
        concurrency, or ``None`` when the cap was reached without failure.
    """
    async with async_playwright() as playwright:
        try:
            browser = await playwright.chromium.launch(headless=not headed)
        except PlaywrightError as exc:
            message = str(exc)
            if "Executable doesn't exist" in message or "playwright install" in message:
                pytest.skip("Chromium is not installed. Run 'uv run playwright install chromium'.")
            raise

        try:
            print(f"\nBrowser stress test target: {server_url}")
            print(f"{'concurrency':>11} | {'successes':>9} | {'failures':>8} | {'elapsed (s)':>11}")
            print("-" * 50)

            last_success = 0
            first_failure: int | None = None
            concurrency = _START_PARALLEL

            while concurrency <= _MAX_BROWSER_PARALLEL:
                result = await _run_browser_wave(browser, server_url, concurrency)
                print(
                    f"{result.concurrency:>11} | {result.successes:>9} | "
                    f"{result.failures:>8} | {result.elapsed:>11.2f}"
                )
                for succeeded, error in result.results:
                    if not succeeded:
                        print(f"  [ERR] error={error}")

                if not result.succeeded:
                    first_failure = concurrency
                    break

                last_success = concurrency
                concurrency *= 2

            return last_success, first_failure
        finally:
            await browser.close()


def test_parallel_browser_chat_capacity(
    pytestconfig: pytest.Config,
    server_url: str,
    record_property: Callable[[str, object], None],
) -> None:
    """
    Find the maximum number of parallel browser chat sessions the server handles.

    Parameters
    ----------
    pytestconfig : pytest.Config
        Pytest configuration object, used to honour the ``--headed`` option.
    server_url : str
        Base URL of the deployment under test, provided by the ``server_url``
        fixture.
    record_property : Callable[[str, object], None]
        Pytest fixture for attaching the measured limit to the test report.
    """
    _check_health(server_url)
    last_success, first_failure = asyncio.run(
        _run_browser_stress(server_url, headed=bool(pytestconfig.getoption("--headed")))
    )

    record_property("stress_browser_max_parallel_sessions", last_success)
    record_property("stress_browser_first_failure", first_failure or "")

    if last_success == 0:
        pytest.fail("Server could not complete a single browser chat session.")

    if first_failure is None:
        print(f"\nSTRESS RESULT: browser UI handled at least {last_success} parallel chat sessions (cap reached)")
    else:
        print(f"\nSTRESS RESULT: browser UI handled {last_success} parallel chat sessions")
        print(f"First failure at {first_failure} parallel browser sessions")

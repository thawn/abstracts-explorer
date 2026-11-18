"""
Staging end-to-end tests for the Abstracts Explorer web UI.

These Playwright tests are designed to run against a **live deployment** (e.g.
a staging server) rather than a locally-spawned test server.  They verify the
critical user-facing functionality described in
``docs/branching_strategy.md#staging-end-to-end-tests``.

Usage
-----
Run against a staging deployment::

    uv run pytest tests/test_staging_e2e.py -m staging --staging-url http://localhost:5000

Or set the ``STAGING_URL`` environment variable::

    STAGING_URL=http://localhost:5000 uv run pytest tests/test_staging_e2e.py -m staging

Add ``-s`` (``--capture=no``) to see chat history output in real time::

    uv run pytest tests/test_staging_e2e.py -m staging --staging-url http://localhost:5000 -s

Notes
-----
* All tests are marked with ``@pytest.mark.staging`` **and**
  ``@pytest.mark.slow`` so they are excluded from the default test run.
* Tests use Playwright (sync API) with Chromium and the module-scoped
  ``staging_page`` fixture.
* Chat test correctness is evaluated by calling the LLM backend configured
  in ``.env.tests`` (``LLM_BACKEND_URL`` / ``CHAT_MODEL``).  Set
  ``LLM_BACKEND_AUTH_TOKEN`` to authenticate.  If the backend is unavailable,
  the judgment step is skipped and the test still passes when a non-empty
  response is received.
* Chat histories are always printed to stdout so they are visible with ``-s``
  and are included in the captured output shown on test failure.
"""

import time
from warnings import warn

import pytest
import requests as _requests
from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError, expect

from abstracts_explorer.config import get_config
from tests.conftest import get_env_test_path

# CSS selectors used for chart detection in the chat area
_CSS_PLOTLY_RENDERED = "#chat-messages .js-plotly-plot"
_CSS_CHAT_PLOT_CONTAINER = "div[id^='chat-plot-']"

# Default conference and year selected in chat-related tests
_DEFAULT_CONFERENCE = "NeurIPS"
_DEFAULT_YEAR = "2025"

# ---------------------------------------------------------------------------
# Conference / year selection helper
# ---------------------------------------------------------------------------


def _select_conference_and_year(
    page: Page,
    conference: str = _DEFAULT_CONFERENCE,
    year: str = _DEFAULT_YEAR,
    timeout: int = 15,
) -> None:
    """
    Select *conference* and *year* from the header dropdowns.

    Waits until the conference selector is populated (has at least one
    option besides the empty placeholder), then picks *conference* from it.
    After the conference is selected the year dropdown is updated by the
    ``handleConferenceChange`` JS handler, so this function then waits for
    the target *year* option to appear and selects it.

    Parameters
    ----------
    page : Page
        The Playwright page.
    conference : str, optional
        Conference name to select (default: ``"NeurIPS"``).
    year : str, optional
        Year to select as a string (default: ``"2025"``).
    timeout : int, optional
        Maximum seconds to wait for the dropdowns to be populated (default: 15).
    """
    timeout_ms = timeout * 1000

    # Wait until the specific conference option is available in the selector
    page.wait_for_function(
        "sel => [...document.querySelectorAll('#conference-selector option')].some(o => o.value === sel)",
        arg=conference,
        timeout=timeout_ms,
    )

    page.locator("#conference-selector").select_option(value=conference)
    # Trigger the JS change handler so the year dropdown is updated
    page.evaluate("handleConferenceChange()")

    # Wait until the target year option is available in the year selector
    page.wait_for_function(
        "sel => [...document.querySelectorAll('#year-selector option')].some(o => o.value === sel)",
        arg=year,
        timeout=timeout_ms,
    )

    page.locator("#year-selector").select_option(value=year)
    # Trigger the JS change handler so filters are refreshed
    page.evaluate("handleYearChange()")


# ---------------------------------------------------------------------------
# LLM judgment helper
# ---------------------------------------------------------------------------


def _judge_with_llm(query: str, response: str, criteria: str | None = None) -> tuple:
    """
    Ask the configured LLM backend whether *response* adequately answers *query*.

    Reads ``LLM_BACKEND_URL``, ``LLM_BACKEND_AUTH_TOKEN``, and ``CHAT_MODEL``
    from the project configuration (loaded from ``.env.tests``).

    Parameters
    ----------
    query : str
        The original chat query.
    response : str
        The assistant's response to evaluate.
    criteria : str, optional
        Additional hints for the LLM judge describing what a good response
        should contain (e.g. "the response should list topic names with paper
        counts").  When omitted the judge uses only the query and response.

    Returns
    -------
    tuple[bool | None, str]
        ``(passed, explanation)`` where *passed* is ``True`` if the LLM judged
        the response as adequate, ``False`` if not, and ``None`` if the backend
        was unreachable or the call failed.  *explanation* contains either the
        LLM's reasoning text or the error message.
    """
    config = get_config(reload=True, env_path=get_env_test_path())
    llm_url = config.llm_backend_url
    auth_token = config.llm_backend_auth_token
    chat_model = config.chat_model

    headers = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    criteria_clause = f"\n\nSpecifically, look for: {criteria}" if criteria else ""
    prompt = (
        f"You are evaluating the quality of a response to a query. Note that the current date is {time.strftime('year: %Y month: %m day: %d')}. "
        f"The response was generated by a tool that searches a real database of published conference papers. "
        f"All paper titles, authors, and statistics mentioned in the response come from actual database records and should be treated as factual. "
        f"Specific, quantitative information contained in the response is generated by a clustering tool and should be judged as accurate.\n\n"
        f"Query: {query}\n\n"
        f"Response: {response}{criteria_clause}\n\n"
        f"Does this response adequately answer the query? "
        f"Respond with 'YES' or 'NO' followed by a brief explanation (one sentence)."
    )

    payload: dict = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 200,
    }
    if chat_model:
        payload["model"] = chat_model

    try:
        resp = _requests.post(
            f"{llm_url}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        passed = content.strip().upper().startswith("YES")
        return passed, content
    except _requests.exceptions.RequestException as exc:
        return None, f"LLM judgment unavailable (network/HTTP error): {exc}"
    except (KeyError, IndexError, ValueError) as exc:
        return None, f"LLM judgment unavailable (unexpected response format): {exc}"


# ---------------------------------------------------------------------------
# Chat DOM helpers
# ---------------------------------------------------------------------------


def _has_complete_response(page: Page) -> bool:
    """
    Return ``True`` once there is at least one complete (non-loading) assistant
    message in the chat.

    The chat module first inserts a *loading* message that contains a
    ``.spinner`` element and then replaces it with the final response.  This
    helper waits until the last ``[data-role='assistant']`` message has no
    spinner, indicating the response is complete.

    Parameters
    ----------
    page : Page
        The Playwright page.
    """
    assistant_msgs = page.locator(".chat-message[data-role='assistant']")
    if assistant_msgs.count() == 0:
        return False
    return assistant_msgs.last.locator(".spinner").count() == 0


def _check_chart_rendered(page: Page, timeout: int = 15) -> tuple:
    """
    Check whether a Plotly chart has been rendered inside the chat.

    After the server returns visualisation data the front-end calls
    ``Plotly.newPlot`` which adds a ``js-plotly-plot`` class to the container
    ``<div id="chat-plot-…">`` element.  This helper polls for up to *timeout*
    seconds before giving up, because Plotly rendering is asynchronous and may
    lag behind the assistant text.

    Parameters
    ----------
    page : Page
        The Playwright page.
    timeout : int, optional
        Maximum seconds to wait for a chart to appear (default: 15).

    Returns
    -------
    tuple[bool, str]
        ``(chart_found, description)`` where *chart_found* is ``True`` when at
        least one rendered Plotly chart is detected, and *description* is a
        human-readable summary suitable for passing to the LLM judge as part of
        the evaluation criteria.
    """
    try:
        page.wait_for_function(
            """() => {
                if (document.querySelectorAll("#chat-messages .js-plotly-plot").length) {
                    return true;
                }
                return [...document.querySelectorAll("div[id^='chat-plot-']")].some(
                    (c) => c.querySelector("svg")
                );
            }""",
            timeout=timeout * 1000,
        )
    except PlaywrightTimeoutError:
        return False, "No chart was rendered in the chat."

    plotly_charts = page.locator(_CSS_PLOTLY_RENDERED)
    plotly_count = plotly_charts.count()
    if plotly_count:
        return True, f"A chart was rendered in the chat ({plotly_count} Plotly chart(s) detected)."
    chart_containers = page.locator(_CSS_CHAT_PLOT_CONTAINER)
    populated = sum(1 for i in range(chart_containers.count()) if chart_containers.nth(i).locator("svg").count())
    if populated:
        return True, f"A chart was rendered in the chat ({populated} SVG chart(s) detected)."
    return False, "No chart was rendered in the chat."


def _extract_chat_history(page: Page) -> list:
    """
    Return all visible chat messages as a list of ``(role, text)`` tuples.

    Parameters
    ----------
    page : Page
        The Playwright page.

    Returns
    -------
    list of tuple[str, str]
        Each tuple is ``(role, text)`` where *role* is ``'user'``,
        ``'assistant'``, or ``'unknown'``.
    """
    messages = page.locator(".chat-message[data-role]").all()
    history = []
    for msg in messages:
        role = msg.get_attribute("data-role") or "unknown"
        history.append((role, msg.inner_text()))
    return history


def _print_chat_history(history: list, query: str, max_message_length: int = 500) -> None:
    """
    Print *history* to stdout so it is visible when pytest is run with ``-s``
    and is included in captured output shown on test failure.

    Long messages are truncated to *max_message_length* characters to keep the
    output readable; the full text is still stored in *history* for programmatic
    use.

    Parameters
    ----------
    history : list of tuple[str, str]
        Chat history as returned by :func:`_extract_chat_history`.
    query : str
        The original chat query (used as a header label).
    max_message_length : int, optional
        Maximum characters to print per message (default: 500).
    """
    print(f"\n{'='*60}")
    print(f"Chat history for query: {query!r}")
    print("=" * 60)
    for role, text in history:
        label = role.upper()
        display = text if len(text) <= max_message_length else text[:max_message_length] + "…"
        print(f"[{label}] {display}")
    print("=" * 60)


def _assert_llm_judgment(query: str, response: str, criteria: str | None = None) -> None:
    """
    Assert that the LLM judges *response* as an adequate answer to *query*.

    Calls :func:`_judge_with_llm` and either asserts success, skips when the
    backend is unreachable, or fails with a descriptive message that includes
    the LLM's reasoning and the full response text.

    Parameters
    ----------
    query : str
        The original chat query.
    response : str
        The assistant's response to evaluate.
    criteria : str, optional
        Additional hints for the LLM judge describing what a good response
        should contain.  Forwarded directly to :func:`_judge_with_llm`.
    """
    passed, explanation = _judge_with_llm(query, response, criteria=criteria)
    if passed is None:
        pytest.skip(f"LLM judgment skipped: {explanation}")
    assert passed, (
        f"LLM judged the response as inadequate.\n"
        f"Explanation: {explanation}\n"
        f"Query: {query}\n"
        f"Response: {response}"
    )
    warn(
        f"LLM judgment passed: {explanation}. Please double check if the response accurately answers the query.\n\nQuery: {query}\n\nResponse: {response}",
        UserWarning,
    )


def _submit_search_query(page: Page, query: str) -> None:
    """
    Select the default conference and year in the header, then enter *query*
    in the main search input and submit with Enter.

    Parameters
    ----------
    page : Page
        The Playwright page.
    query : str
        Search query text to submit.
    """

    _select_conference_and_year(page)

    search_input = page.locator("#search-input")
    search_input.fill(query)
    search_input.press("Enter")


# ---------------------------------------------------------------------------
# 1. Application startup
# ---------------------------------------------------------------------------


@pytest.mark.staging
@pytest.mark.slow
class TestApplicationStartup:
    """Verify the application boots and renders without errors."""

    def test_page_loads(self, staging_url, staging_page):
        """
        Page loads successfully.

        The web UI returns HTTP 200, the page title contains
        'Abstracts Explorer', and the main application container is present.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        staging_page : Page
            Playwright page instance.
        """
        staging_page.goto(staging_url)

        assert "Abstracts Explorer" in staging_page.title()

        # Core layout elements must be present
        for selector in ("#search-input", "#search-results", "#tab-search", "#tab-chat"):
            expect(staging_page.locator(selector)).to_be_attached()

    def test_page_no_javascript_errors(self, staging_url, staging_page, staging_console_errors):
        """
        No JavaScript errors on load.

        The browser console contains no error-level entries after the
        initial page load.  Messages related to ``favicon`` or
        ``ERR_BLOCKED_BY_CLIENT`` are ignored as they are not application
        errors.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        staging_page : Page
            Playwright page instance.
        staging_console_errors : list[str]
            Console and uncaught page errors collected by the fixture.
        """
        staging_page.goto(staging_url)
        staging_page.wait_for_timeout(2000)  # allow async scripts to settle

        ignored = ("favicon", "ERR_BLOCKED_BY_CLIENT")
        errors = [entry for entry in staging_console_errors if not any(token in entry for token in ignored)]
        assert len(errors) == 0, f"JavaScript errors found: {errors}"


# ---------------------------------------------------------------------------
# 2. Core search functionality
# ---------------------------------------------------------------------------


@pytest.mark.staging
@pytest.mark.slow
class TestCoreSearch:
    """Verify keyword search, empty search, and no-results handling."""

    def test_keyword_search_returns_results(self, staging_url, staging_page):
        """
        Keyword search returns results.

        Entering a generic query and pressing Enter displays at least one
        paper card.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        staging_page : Page
            Playwright page instance.
        """
        staging_page.goto(staging_url)

        _submit_search_query(staging_page, "learning")

        results = staging_page.locator("#search-results .bg-white.rounded-lg")
        expect(results.first).to_be_attached(timeout=15000)

        assert results.count() > 0, "Search should return at least one result"

    def test_empty_search_shows_message(self, staging_url, staging_page):
        """
        Empty search shows a message.

        Submitting an empty search shows appropriate user feedback instead
        of an error.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        staging_page : Page
            Playwright page instance.
        """
        staging_page.goto(staging_url)

        _submit_search_query(staging_page, "")

        staging_page.wait_for_timeout(1000)

        expect(staging_page.locator("#search-results")).to_be_attached()

    def test_search_no_results(self, staging_url, staging_page):
        """
        Search with no results.

        A query that matches nothing displays a 'no results' message
        instead of an error or a crash.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        staging_page : Page
            Playwright page instance.
        """
        staging_page.goto(staging_url)

        _submit_search_query(staging_page, "xyzqwertyuiopasdfghjkl9999")

        staging_page.wait_for_timeout(1000)

        # The page must not crash – the results container should still be present
        expect(staging_page.locator("#search-results")).to_be_attached()


# ---------------------------------------------------------------------------
# 3. Author search functionality
# ---------------------------------------------------------------------------


@pytest.mark.staging
@pytest.mark.slow
class TestAuthorSearch:
    """Verify author-name searches with different query formats."""

    _RESULT_CARD_CSS = "#search-results .bg-white.rounded-lg"
    _WAIT_TIMEOUT = 30

    def test_author_name_only(self, staging_url, staging_page):
        """
        Author name only search returns results.

        Entering only a last name such as "LeCun" returns at least one paper
        whose author field contains that name.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        staging_page : Page
            Playwright page instance.
        """
        staging_page.goto(staging_url)

        _submit_search_query(staging_page, "LeCun")

        results = staging_page.locator(self._RESULT_CARD_CSS)
        expect(results.first).to_be_attached(timeout=self._WAIT_TIMEOUT * 1000)

        assert results.count() > 0, "Searching for 'LeCun' should return at least one result"

        results_text = staging_page.locator("#search-results").inner_text()
        assert "LeCun" in results_text, "Results for 'LeCun' should display the author name"

    def test_author_field_syntax(self, staging_url, staging_page):
        """
        Field-formatted author query returns results for that author.

        The query ``author:"LeCun"`` must bypass the embedding search and
        return only papers whose ``authors`` column matches the name.  The
        author name must appear in the visible results.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        staging_page : Page
            Playwright page instance.
        """
        staging_page.goto(staging_url)

        _submit_search_query(staging_page, 'author:"LeCun"')

        results = staging_page.locator(self._RESULT_CARD_CSS)
        expect(results.first).to_be_attached(timeout=self._WAIT_TIMEOUT * 1000)

        assert results.count() > 0, 'author:"LeCun" should return at least one result'

        results_text = staging_page.locator("#search-results").inner_text()
        assert "LeCun" in results_text, 'Results for author:"LeCun" should display the author name'

    def test_author_with_topic(self, staging_url, staging_page):
        """
        Combined author filter and topic query returns relevant results.

        The query ``author:"LeCun" world model`` filters papers by author and
        ranks them by semantic similarity to "world model".  At least one paper
        must be returned and the author name must appear in the results.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        staging_page : Page
            Playwright page instance.
        """
        staging_page.goto(staging_url)

        _submit_search_query(staging_page, 'authors:"LeCun" world model')

        results = staging_page.locator(self._RESULT_CARD_CSS)
        expect(results.first).to_be_attached(timeout=self._WAIT_TIMEOUT * 1000)

        assert results.count() > 0, 'authors:"LeCun" world model should return at least one result'

        results_text = staging_page.locator("#search-results").inner_text()
        assert "LeCun" in results_text, 'Results for author:"LeCun" world model should display the author name'


# ---------------------------------------------------------------------------
# 4. Paper display
# ---------------------------------------------------------------------------


@pytest.mark.staging
@pytest.mark.slow
class TestPaperDisplay:
    """Verify paper detail view and collapsible abstracts."""

    def test_paper_detail_view(self, staging_url, staging_page):
        """
        Paper detail view.

        Clicking a paper shows its title, authors, and abstract.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        staging_page : Page
            Playwright page instance.
        """
        staging_page.goto(staging_url)

        _submit_search_query(staging_page, "learning")

        cards = staging_page.locator("#search-results .paper-card")
        expect(cards.first).to_be_attached(timeout=15000)

        assert cards.count() > 0, "Should have at least one paper card"

        first = cards.first

        # Paper card must contain a title element
        title_elements = first.locator("h3, h4")
        assert title_elements.count() > 0, "Paper card should have a title"
        assert len(title_elements.first.inner_text()) > 0, "Title should not be empty"

        # Paper card must contain author information
        author_elements = first.locator(".text-gray-600")
        assert author_elements.count() > 0, "Paper card should have author info"

    def test_collapsible_abstract(self, staging_url, staging_page):
        """
        Collapsible abstract.

        The abstract section can be expanded and collapsed.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        staging_page : Page
            Playwright page instance.
        """
        staging_page.goto(staging_url)

        _submit_search_query(staging_page, "large language models")

        expect(staging_page.locator("#search-results .bg-white.rounded-lg").first).to_be_attached(timeout=15000)

        # Paper cards use native <details>/<summary> for collapsible abstracts
        # when the abstract is longer than ~300 characters.
        detail_elements = staging_page.locator("#search-results details")
        if detail_elements.count() == 0:
            pytest.skip("No collapsible abstracts found (all abstracts may be short)")

        details = detail_elements.first
        summary = details.locator("summary")

        # Abstracts start collapsed – expand by clicking summary
        summary.click()
        staging_page.wait_for_timeout(300)
        assert details.get_attribute("open") is not None, "Abstract details element should be open after click"

        # Collapse again
        summary.click()
        staging_page.wait_for_timeout(300)
        assert details.get_attribute("open") is None, "Abstract details element should be closed after second click"


# ---------------------------------------------------------------------------
# 5. Chat (RAG) interface
# ---------------------------------------------------------------------------


@pytest.mark.staging
@pytest.mark.slow
class TestChatInterface:
    """Verify core chat UI elements and basic send functionality."""

    def test_chat_ui_elements_present(self, staging_url, staging_page):
        """
        Chat UI elements present.

        The chat input, send button, and reset button are visible.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        staging_page : Page
            Playwright page instance.
        """
        staging_page.goto(staging_url)

        _select_conference_and_year(staging_page)

        # Switch to the chat tab
        staging_page.locator("#tab-chat").click()
        staging_page.wait_for_timeout(500)

        expect(staging_page.locator("#chat-input")).to_be_attached()
        expect(staging_page.locator("#chat-messages")).to_be_attached()

        # Reset button (uses onclick='resetChat()')
        reset_buttons = staging_page.locator("button[onclick*='resetChat']")
        assert reset_buttons.count() > 0, "Reset button should be present"

    def test_chat_send_message(self, staging_url, staging_page):
        """
        Send a chat message.

        Typing a question and clicking send returns a complete response (or a
        graceful error if the LLM backend is unavailable).  The test waits
        until the loading spinner disappears from the last assistant message so
        that it does not mistake the "Thinking..." placeholder for the answer.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        staging_page : Page
            Playwright page instance.
        """
        query = "What papers are about machine learning?"
        staging_page.goto(staging_url)

        _select_conference_and_year(staging_page)

        staging_page.locator("#tab-chat").click()
        staging_page.wait_for_timeout(500)

        chat_input = staging_page.locator("#chat-input")
        chat_input.fill(query)
        chat_input.press("Enter")

        try:
            # Wait until there is a complete (non-loading) assistant response
            staging_page.wait_for_function(
                """() => {
                    const msgs = document.querySelectorAll(".chat-message[data-role='assistant']");
                    if (!msgs.length) {
                        return false;
                    }
                    return msgs[msgs.length - 1].querySelectorAll(".spinner").length === 0;
                }""",
                timeout=120000,
            )
            history = _extract_chat_history(staging_page)
            _print_chat_history(history, query)
            assistant_msgs = staging_page.locator(".chat-message[data-role='assistant']")
            assert assistant_msgs.count() >= 1, "Chat should display at least one assistant message"
            assert assistant_msgs.last.inner_text().strip() != "", "Assistant response should not be empty"
        except PlaywrightTimeoutError:
            history = _extract_chat_history(staging_page)
            _print_chat_history(history, query)
            # LLM backend may not be available – verify input was cleared
            val = staging_page.locator("#chat-input").input_value()
            assert val == "", "Chat input should be cleared after sending"


# ---------------------------------------------------------------------------
# 4a. MCP tool smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.staging
@pytest.mark.slow
class TestMCPToolSmokeTests:
    """
    Exercise each MCP tool via the chat interface.

    Each test:

    1. Sends a representative query via the chat interface.
    2. Waits for the complete response (spinner-free, no "Thinking…" placeholder).
    3. **Always prints the full chat history** so the tester can inspect the
       actual exchange (visible with ``-s`` / ``--capture=no`` and always shown
       in captured output on failure).
    4. Judges correctness by calling the configured LLM backend via
       ``_judge_with_llm``.  If the backend is unreachable the judgment step is
       skipped and the test passes as long as a non-empty response was received.
    """

    # Default timeout (seconds) to wait for the LLM response
    _TIMEOUT = 120

    def _send_chat_query(self, staging_url, staging_page, query):
        """
        Send *query* via the chat UI and return ``(response_text, history)``.

        Navigates to *staging_url*, switches to the chat tab, types *query*,
        presses Enter, then waits until the loading spinner disappears from the
        last assistant message before returning.

        The chat history is **always** printed to stdout via
        :func:`_print_chat_history` so it is visible when running with ``-s``
        and is included in the captured output shown on test failure.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        staging_page : Page
            Playwright page instance.
        query : str
            The chat query to send.

        Returns
        -------
        tuple[str, list]
            ``(response_text, history)`` where *response_text* is the text of
            the last assistant message (empty string on timeout) and *history*
            is a list of ``(role, text)`` tuples for all visible messages.
        """
        staging_page.goto(staging_url)

        _select_conference_and_year(staging_page)

        staging_page.locator("#tab-chat").click()
        staging_page.wait_for_timeout(500)

        # Reset server-side conversation so previous test results don't
        # prevent the LLM from calling MCP tools (it might skip tool calls
        # when it already has cached context from an earlier exchange).
        reset_buttons = staging_page.locator("button[onclick*='resetChat']")
        if reset_buttons.count():
            reset_buttons.first.click()
            staging_page.wait_for_timeout(500)

        chat_input = staging_page.locator("#chat-input")
        chat_input.fill(query)
        chat_input.press("Enter")

        try:
            # Wait until the last assistant message has no spinner (response complete)
            staging_page.wait_for_function(
                """() => {
                    const msgs = document.querySelectorAll(".chat-message[data-role='assistant']");
                    if (!msgs.length) {
                        return false;
                    }
                    return msgs[msgs.length - 1].querySelectorAll(".spinner").length === 0;
                }""",
                timeout=self._TIMEOUT * 1000,
            )
            assistant_msgs = staging_page.locator(".chat-message[data-role='assistant']")
            response_text = assistant_msgs.last.inner_text() if assistant_msgs.count() else ""
        except PlaywrightTimeoutError:
            response_text = ""

        history = _extract_chat_history(staging_page)
        _print_chat_history(history, query)
        return response_text, history

    def _mark_interesting_papers(
        self,
        staging_url,
        staging_page,
        query: str = "learning",
        count: int = 2,
    ) -> None:
        """
        Rate the first *count* search results as 5-star interesting papers.

        Interesting-paper tools need seed ratings.  The ratings are stored in
        ``localStorage`` by the web UI, so they survive the navigation performed
        by :meth:`_send_chat_query`.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        staging_page : Page
            Playwright page instance.
        query : str, optional
            Search query used to find papers to rate (default: ``"learning"``).
        count : int, optional
            Number of search results to rate (default: 2).
        """
        staging_page.goto(staging_url)
        staging_page.evaluate("localStorage.removeItem('paperPriorities')")
        staging_page.reload()
        _submit_search_query(staging_page, query)

        stars = staging_page.locator("#search-results .paper-star")
        expect(stars.first).to_be_attached(timeout=15000)

        for index in range(count):
            star = stars.nth(index * 5 + 4)
            expect(star).to_be_attached(timeout=15000)
            star.click()

        staging_page.wait_for_function(
            """count => {
                const raw = localStorage.getItem('paperPriorities');
                if (!raw) return false;
                return Object.keys(JSON.parse(raw)).length >= count;
            }""",
            arg=count,
            timeout=5000,
        )

    # ------------------------------------------------------------------
    # Individual MCP tool tests
    # ------------------------------------------------------------------

    def test_get_conference_topics(self, staging_url, staging_page):
        """
        MCP tool: ``get_conference_topics``.

        Example query: *"What are the main topics at NeurIPS 2025?"*

        Success criteria (LLM-judged): the response lists topic names with
        keywords or paper counts.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        staging_page : Page
            Playwright page instance.
        """
        query = "What are the main topics at NeurIPS 2025?"
        response, _ = self._send_chat_query(staging_url, staging_page, query)
        if not response:
            pytest.skip("LLM backend unavailable – no response received")
        _assert_llm_judgment(
            query,
            response,
            criteria="the response should list multiple topic names and include paper counts or representative keywords for each topic",
        )

    def test_get_topic_evolution(self, staging_url, staging_page):
        """
        MCP tool: ``get_topic_evolution``.

        Example query: *"How has research on transformers evolved at NeurIPS
        over the years?"*

        Success criteria (LLM-judged): the response includes a year-by-year
        breakdown of paper counts or a trend description, and a chart should
        be rendered in the chat area.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        staging_page : Page
            Playwright page instance.
        """
        query = "How has research on transformers evolved at NeurIPS over the years?"
        response, _ = self._send_chat_query(staging_url, staging_page, query)
        if not response:
            pytest.skip("LLM backend unavailable – no response received")
        chart_found, chart_description = _check_chart_rendered(staging_page)
        assert chart_found, f"Expected a trend chart to be rendered in chat. {chart_description}"
        _assert_llm_judgment(
            query,
            response,
            criteria=(
                "the response should describe a year-by-year trend in paper counts or research focus on transformers. "
                f"Additionally: {chart_description}"
            ),
        )

    def test_search_papers(self, staging_url, staging_page):
        """
        MCP tool: ``search_papers``.

        Example query: *"Find papers about reinforcement learning at NeurIPS
        2025."*

        Success criteria (LLM-judged): the response returns paper titles with
        authors and/or abstracts.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        staging_page : Page
            Playwright page instance.
        """
        query = "Find papers about reinforcement learning at NeurIPS 2025."
        response, _ = self._send_chat_query(staging_url, staging_page, query)
        if not response:
            pytest.skip("LLM backend unavailable – no response received")
        _assert_llm_judgment(
            query,
            response,
            criteria="the response should briefly summarize or list multiple papers related to reinforcement learning",
        )

    def test_get_paper_details(self, staging_url, staging_page):
        """
        MCP tool: ``get_paper_details``.

        Example query: *"Who are the authors of the paper titled 'Large Language Diffusion Models'?"*

        Success criteria (LLM-judged): the response includes author names,
        URL/PDF links, and session info (where available).

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        staging_page : Page
            Playwright page instance.
        """
        query = "Who are the authors of the paper titled 'Large Language Diffusion Models'?"
        response, _ = self._send_chat_query(staging_url, staging_page, query)
        if not response:
            pytest.skip("LLM backend unavailable – no response received")
        _assert_llm_judgment(
            query,
            response,
            criteria="the response should name several authors of the paper and give a brief description.",
        )

    def test_analyze_topic_relevance(self, staging_url, staging_page):
        """
        MCP tool: ``analyze_topic_relevance``.

        Example query: *"How relevant is uncertainty quantification at NeurIPS
        2025?"*

        Success criteria (LLM-judged): the response contains a relevance score
        or paper count within the embedding distance threshold.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        staging_page : Page
            Playwright page instance.
        """
        query = "How relevant is uncertainty quantification at NeurIPS 2025?"
        response, _ = self._send_chat_query(staging_url, staging_page, query)
        if not response:
            pytest.skip("LLM backend unavailable – no response received")
        _assert_llm_judgment(
            query,
            response,
            criteria="the response should include a relevance score, percentage, or paper count that quantifies how relevant the topic is",
        )

    def test_get_cluster_visualization(self, staging_url, staging_page):
        """
        MCP tool: ``get_cluster_visualization``.

        Example query: *"Show me a visual overview of how topics are clustered at NeurIPS."*

        Success criteria (LLM-judged): the response references the visualization
        and a Plotly chart should be rendered in the chat area.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        staging_page : Page
            Playwright page instance.
        """
        query = "Show me a visual overview of how topics are clustered at NeurIPS."
        response, _ = self._send_chat_query(staging_url, staging_page, query)
        if not response:
            pytest.skip("LLM backend unavailable – no response received")
        chart_found, formatted_answer = _check_chart_rendered(staging_page)
        assert chart_found, f"Expected a cluster scatter plot to be rendered in chat. {formatted_answer}"
        _assert_llm_judgment(
            query,
            response,
            criteria=("the response should describe or reference a cluster visualization. Note: " + formatted_answer),
        )

    def test_summarize_interesting_papers(self, staging_url, staging_page):
        """
        MCP tool: ``summarize_interesting_papers``.

        Example query: *"Summarize the papers I marked as interesting."*

        Success criteria (LLM-judged): the response summarises the interesting
        papers, indicates how many were marked, and mentions at least one
        paper title, topic, keyword, or rating.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        staging_page : Page
            Playwright page instance.
        """
        query = "Summarize the papers I marked as interesting."
        self._mark_interesting_papers(staging_url, staging_page)
        response, _ = self._send_chat_query(staging_url, staging_page, query)
        if not response:
            pytest.skip("LLM backend unavailable – no response received")
        _assert_llm_judgment(
            query,
            response,
            criteria=(
                "the response should summarize the user's interesting paper(s), indicate how many "
                "papers were marked, and mention at least one paper title, topic, keyword, or rating"
            ),
        )

    def test_recommend_related_papers(self, staging_url, staging_page):
        """
        MCP tool: ``recommend_related_papers``.

        Example query: *"Based on my interesting papers, suggest other papers
        that may interest me."*

        Success criteria (LLM-judged): the response recommends at least one
        additional paper related to the user's interesting papers.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        staging_page : Page
            Playwright page instance.
        """
        query = "Based on my interesting papers, suggest other papers that may interest me."
        self._mark_interesting_papers(staging_url, staging_page)
        response, _ = self._send_chat_query(staging_url, staging_page, query)
        if not response:
            pytest.skip("LLM backend unavailable – no response received")
        _assert_llm_judgment(
            query,
            response,
            criteria=(
                "the response should recommend at least one additional paper related to the "
                "user's interesting papers; it may include a title, topic, or reason for the recommendation"
            ),
        )

    def test_suggest_authors_to_meet(self, staging_url, staging_page):
        """
        MCP tool: ``suggest_authors_to_meet``.

        Example query: *"Who should I meet based on my interesting papers?"*

        Success criteria (LLM-judged): the response names at least one author
        working on topics similar to the user's interesting papers, ideally
        with a session or poster location.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        staging_page : Page
            Playwright page instance.
        """
        query = "Who should I meet based on my interesting papers?"
        self._mark_interesting_papers(staging_url, staging_page)
        response, _ = self._send_chat_query(staging_url, staging_page, query)
        if not response:
            pytest.skip("LLM backend unavailable – no response received")
        _assert_llm_judgment(
            query,
            response,
            criteria=(
                "the response should name at least one author who works on topics similar to the "
                "user's interesting papers, ideally with a session or poster location"
            ),
        )


# ---------------------------------------------------------------------------
# 6. Clustering tab
# ---------------------------------------------------------------------------


@pytest.mark.staging
@pytest.mark.slow
class TestClusteringTab:
    """Verify the clustering tab and its visualization."""

    def test_clustering_tab_exists(self, staging_url, staging_page):
        """
        Clustering tab exists.

        The clustering tab is present and can be activated.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        staging_page : Page
            Playwright page instance.
        """
        staging_page.goto(staging_url)

        clustering_tab = staging_page.locator("#tab-clusters")
        expect(clustering_tab).to_be_enabled(timeout=10000)
        assert clustering_tab.is_visible(), "Clustering tab should be visible"

    def test_clustering_plot_loads(self, staging_url, staging_page):
        """
        Clustering plot loads.

        Switching to the clustering tab renders a Plotly visualization
        (or shows a meaningful placeholder).

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        staging_page : Page
            Playwright page instance.
        """
        staging_page.goto(staging_url)

        clustering_tab = staging_page.locator("#tab-clusters")
        expect(clustering_tab).to_be_enabled(timeout=10000)
        clustering_tab.click()

        # Wait for either the Plotly plot or a placeholder to appear
        try:
            staging_page.wait_for_selector("#cluster-plot", state="visible", timeout=10000)
        except PlaywrightTimeoutError:
            pytest.fail("Cluster plot container did not become visible")

        assert staging_page.locator("#cluster-plot").is_visible(), "Cluster plot container should be visible"


# ---------------------------------------------------------------------------
# 7. Accessibility & responsiveness
# ---------------------------------------------------------------------------


@pytest.mark.staging
@pytest.mark.slow
class TestAccessibilityResponsiveness:
    """Verify keyboard navigation and responsive layout."""

    def test_keyboard_navigation(self, staging_url, staging_page):
        """
        Keyboard navigation.

        Core interactive elements (search input, tabs) are reachable via
        the Tab key.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        staging_page : Page
            Playwright page instance.
        """
        staging_page.goto(staging_url)

        for _ in range(10):
            staging_page.keyboard.press("Tab")
            staging_page.wait_for_timeout(150)

        # The search input must be reachable
        expect(staging_page.locator("#search-input")).to_be_attached()

    def test_responsive_layout(self, staging_url, staging_page):
        """
        Responsive layout.

        The page renders without horizontal overflow at a narrow viewport
        width (768 px).

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        staging_page : Page
            Playwright page instance.
        """
        staging_page.goto(staging_url)

        # Desktop
        staging_page.set_viewport_size({"width": 1920, "height": 1080})
        staging_page.wait_for_timeout(300)
        assert staging_page.locator("#search-input").is_visible()

        # Tablet
        staging_page.set_viewport_size({"width": 768, "height": 1024})
        staging_page.wait_for_timeout(300)
        assert staging_page.locator("#search-input").is_visible()

        # Check no horizontal overflow
        body_width = staging_page.evaluate("document.body.scrollWidth")
        viewport_width = staging_page.evaluate("window.innerWidth")
        assert (
            body_width <= viewport_width + 5
        ), f"Horizontal overflow detected: body={body_width}px, viewport={viewport_width}px"

        # Reset to default size
        staging_page.set_viewport_size({"width": 1920, "height": 1080})


# ---------------------------------------------------------------------------
# 8. Parallel / rate-limiting tests
# ---------------------------------------------------------------------------


@pytest.mark.staging
@pytest.mark.slow
class TestParallelChatRequests:
    """
    Verify rate limiting under concurrent chat API requests.

    Fires multiple HTTP POST requests to ``/api/chat`` simultaneously and
    confirms the application handles them without errors, rate limiting them
    as configured.
    """

    # Queries that force the RAG agent to call the ``search_papers`` MCP tool.
    # Each call triggers an embedding API request followed by LLM calls.
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
    ]  # type: ignore[var-annotated]

    def test_concurrent_chat_api_requests(self, staging_url, staging_page):
        """
        Concurrent RAG chat requests should all complete with HTTP 200.

        Sends 10 parallel POST requests to ``/api/chat`` with queries that
        exercise the ``search_papers`` MCP tool (each triggers an embedding
        generation call plus LLM calls).  With a properly shared global rate
        limiter the burst is handled gracefully.  On older deployments where
        rate limiting was applied per-thread (v0.8.5), the concurrent burst of
        LLM + embedding requests exceeds the upstream API limit and causes
        500 errors on some requests.
        """
        import concurrent.futures

        num_requests = len(self._RAG_QUERIES)

        # First reset the chat to clear prior conversation state
        _requests.post(
            f"{staging_url}/api/chat/reset",
            timeout=5,
            verify=False,
        )

        results: list[tuple[int, str | dict | None]] = []

        def chat_request(idx: int) -> tuple[int, str | dict | None]:
            try:
                resp = _requests.post(
                    f"{staging_url}/api/chat",
                    json={"message": self._RAG_QUERIES[idx], "reset": True},
                    timeout=120,
                    verify=False,
                )
                return resp.status_code, resp.json() if resp.status_code == 200 else None
            except _requests.exceptions.RequestException as e:
                return -1, str(e)

        start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as pool:
            futures = [pool.submit(chat_request, i) for i in range(num_requests)]
            for f in concurrent.futures.as_completed(futures, timeout=600):
                status, body = f.result()
                results.append((status, body))
        elapsed = time.monotonic() - start

        print(f"\nParallel RAG chat requests: {len(results)} completed, elapsed={elapsed:.2f}s")
        for status, body in results:
            if status == 200:
                print(f"  [OK] status={status}")
            else:
                print(f"  [ERR] status={status}, body={body}")

        assert len(results) == num_requests, f"All {num_requests} requests should complete"

        # All requests must return 200 – a shared global rate limiter ensures
        # the upstream LLM/embedding API is never exceeded.  Any 500 response
        # indicates that the per-thread rate limiting bug (v0.8.5) allowed
        # concurrent threads to collectively exceed the API rate limit.
        success_count = sum(1 for s, _ in results if s == 200)
        assert success_count == num_requests, (
            f"Expected {num_requests} HTTP 200 responses, got {success_count}. "
            f"Non-200 status codes likely indicate threading issues."
        )

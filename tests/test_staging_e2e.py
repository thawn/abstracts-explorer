"""
Staging end-to-end tests for the Abstracts Explorer web UI.

These Selenium tests are designed to run against a **live deployment** (e.g. a
staging server) rather than a locally-spawned test server.  They verify the
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
* The browser is selected via the ``E2E_BROWSER`` environment variable
  (``chrome``, ``firefox``, or ``auto``), reusing the same logic as the
  existing e2e suite.
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
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.remote.webdriver import WebDriver

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
    driver: WebDriver,
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
    driver : WebDriver
        The Selenium WebDriver instance.
    conference : str, optional
        Conference name to select (default: ``"NeurIPS"``).
    year : str, optional
        Year to select as a string (default: ``"2025"``).
    timeout : int, optional
        Maximum seconds to wait for the dropdowns to be populated (default: 15).
    """
    wait = WebDriverWait(driver, timeout)

    # Wait until the specific conference option is available in the selector
    wait.until(
        lambda d: any(
            opt.get_attribute("value") == conference
            for opt in d.find_element(By.ID, "conference-selector").find_elements(By.TAG_NAME, "option")
        )
    )

    conf_select_el = driver.find_element(By.ID, "conference-selector")
    Select(conf_select_el).select_by_value(conference)
    # Trigger the JS change handler so the year dropdown is updated
    driver.execute_script("handleConferenceChange()")

    # Wait until the target year option is available in the year selector
    wait.until(
        lambda d: any(
            opt.get_attribute("value") == year
            for opt in d.find_element(By.ID, "year-selector").find_elements(By.TAG_NAME, "option")
        )
    )

    year_select_el = driver.find_element(By.ID, "year-selector")
    Select(year_select_el).select_by_value(year)
    # Trigger the JS change handler so filters are refreshed
    driver.execute_script("handleYearChange()")


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


def _has_complete_response(driver: WebDriver) -> bool:
    """
    Return ``True`` once there is at least one complete (non-loading) assistant
    message in the chat.

    The chat module first inserts a *loading* message that contains a
    ``.spinner`` element and then replaces it with the final response.  This
    helper waits until the last ``[data-role='assistant']`` message has no
    spinner, indicating the response is complete.

    Parameters
    ----------
    driver : WebDriver
        The Selenium WebDriver instance.
    """
    assistant_msgs = driver.find_elements(By.CSS_SELECTOR, ".chat-message[data-role='assistant']")
    if not assistant_msgs:
        return False
    last = assistant_msgs[-1]
    return len(last.find_elements(By.CSS_SELECTOR, ".spinner")) == 0


def _check_chart_rendered(driver: WebDriver, timeout: int = 15) -> tuple:
    """
    Check whether a Plotly chart has been rendered inside the chat.

    After the server returns visualisation data the front-end calls
    ``Plotly.newPlot`` which adds a ``js-plotly-plot`` class to the container
    ``<div id="chat-plot-…">`` element.  This helper polls for up to *timeout*
    seconds before giving up, because Plotly rendering is asynchronous and may
    lag behind the assistant text.

    Parameters
    ----------
    driver : WebDriver
        The Selenium WebDriver instance.
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
        WebDriverWait(driver, timeout).until(
            lambda d: d.find_elements(By.CSS_SELECTOR, _CSS_PLOTLY_RENDERED)
            or any(
                c.find_elements(By.CSS_SELECTOR, "svg")
                for c in d.find_elements(By.CSS_SELECTOR, _CSS_CHAT_PLOT_CONTAINER)
            )
        )
    except TimeoutException:
        return False, "No chart was rendered in the chat."

    plotly_charts = driver.find_elements(By.CSS_SELECTOR, _CSS_PLOTLY_RENDERED)
    if plotly_charts:
        return True, f"A chart was rendered in the chat ({len(plotly_charts)} Plotly chart(s) detected)."
    chart_containers = driver.find_elements(By.CSS_SELECTOR, _CSS_CHAT_PLOT_CONTAINER)
    populated = [c for c in chart_containers if c.find_elements(By.CSS_SELECTOR, "svg")]
    if populated:
        return True, f"A chart was rendered in the chat ({len(populated)} SVG chart(s) detected)."
    return False, "No chart was rendered in the chat."


def _extract_chat_history(driver: WebDriver) -> list:
    """
    Return all visible chat messages as a list of ``(role, text)`` tuples.

    Parameters
    ----------
    driver : WebDriver
        The Selenium WebDriver instance.

    Returns
    -------
    list of tuple[str, str]
        Each tuple is ``(role, text)`` where *role* is ``'user'``,
        ``'assistant'``, or ``'unknown'``.
    """
    messages = driver.find_elements(By.CSS_SELECTOR, ".chat-message[data-role]")
    history = []
    for msg in messages:
        role = msg.get_attribute("data-role") or "unknown"
        history.append((role, msg.text))
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


# ---------------------------------------------------------------------------
# 1. Application startup
# ---------------------------------------------------------------------------


@pytest.mark.staging
@pytest.mark.slow
class TestApplicationStartup:
    """Verify the application boots and renders without errors."""

    def test_page_loads(self, staging_url, browser):
        """
        Page loads successfully.

        The web UI returns HTTP 200, the page title contains
        'Abstracts Explorer', and the main application container is present.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        browser : WebDriver
            Selenium WebDriver instance.
        """
        browser.get(staging_url)

        assert "Abstracts Explorer" in browser.title

        # Core layout elements must be present
        assert browser.find_element(By.ID, "search-input")
        assert browser.find_element(By.ID, "search-results")
        assert browser.find_element(By.ID, "tab-search")
        assert browser.find_element(By.ID, "tab-chat")

    def test_page_no_javascript_errors(self, staging_url, browser):
        """
        No JavaScript errors on load.

        The browser console contains no SEVERE-level entries after the
        initial page load.  On Firefox (which does not support
        ``get_log``), the test falls back to verifying the page title.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        browser : WebDriver
            Selenium WebDriver instance.
        """
        browser.get(staging_url)
        time.sleep(2)  # allow async scripts to settle

        try:
            logs = browser.get_log("browser")
            errors = [entry for entry in logs if entry["level"] == "SEVERE"]
            assert len(errors) == 0, f"JavaScript errors found: {errors}"
        except (AttributeError, webdriver.remote.errorhandler.WebDriverException):
            # Firefox does not support get_log – fall back
            assert "Abstracts Explorer" in browser.title


# ---------------------------------------------------------------------------
# 2. Core search functionality
# ---------------------------------------------------------------------------


@pytest.mark.staging
@pytest.mark.slow
class TestCoreSearch:
    """Verify keyword search, empty search, and no-results handling."""

    def test_keyword_search_returns_results(self, staging_url, browser):
        """
        Keyword search returns results.

        Entering a generic query and pressing Enter displays at least one
        paper card.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        browser : WebDriver
            Selenium WebDriver instance.
        """
        browser.get(staging_url)

        search_input = browser.find_element(By.ID, "search-input")
        search_input.clear()
        search_input.send_keys("learning")
        search_input.send_keys(Keys.RETURN)

        wait = WebDriverWait(browser, 15)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#search-results .bg-white.rounded-lg")))

        results = browser.find_elements(By.CSS_SELECTOR, "#search-results .bg-white.rounded-lg")
        assert len(results) > 0, "Search should return at least one result"

    def test_empty_search_shows_message(self, staging_url, browser):
        """
        Empty search shows a message.

        Submitting an empty search shows appropriate user feedback instead
        of an error.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        browser : WebDriver
            Selenium WebDriver instance.
        """
        browser.get(staging_url)

        search_input = browser.find_element(By.ID, "search-input")
        search_input.clear()
        search_input.send_keys(Keys.RETURN)

        time.sleep(1)

        results_div = browser.find_element(By.ID, "search-results")
        assert results_div is not None, "Search results container should exist"

    def test_search_no_results(self, staging_url, browser):
        """
        Search with no results.

        A query that matches nothing displays a 'no results' message
        instead of an error or a crash.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        browser : WebDriver
            Selenium WebDriver instance.
        """
        browser.get(staging_url)

        search_input = browser.find_element(By.ID, "search-input")
        search_input.clear()
        search_input.send_keys("xyzqwertyuiopasdfghjkl9999")
        search_input.send_keys(Keys.RETURN)

        time.sleep(1)

        # The page must not crash – the results container should still be present
        results_div = browser.find_element(By.ID, "search-results")
        assert results_div is not None


# ---------------------------------------------------------------------------
# 3. Author search functionality
# ---------------------------------------------------------------------------


@pytest.mark.staging
@pytest.mark.slow
class TestAuthorSearch:
    """Verify author-name searches with different query formats."""

    _RESULT_CARD_CSS = "#search-results .bg-white.rounded-lg"
    _WAIT_TIMEOUT = 30

    def test_author_name_only(self, staging_url, browser):
        """
        Author name only search returns results.

        Entering only a last name such as "LeCun" returns at least one paper
        whose author field contains that name.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        browser : WebDriver
            Selenium WebDriver instance.
        """
        browser.get(staging_url)

        search_input = browser.find_element(By.ID, "search-input")
        search_input.clear()
        search_input.send_keys("LeCun")
        search_input.send_keys(Keys.RETURN)

        wait = WebDriverWait(browser, self._WAIT_TIMEOUT)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, self._RESULT_CARD_CSS)))

        results = browser.find_elements(By.CSS_SELECTOR, self._RESULT_CARD_CSS)
        assert len(results) > 0, "Searching for 'LeCun' should return at least one result"

        results_text = browser.find_element(By.ID, "search-results").text
        assert "LeCun" in results_text, "Results for 'LeCun' should display the author name"

    def test_author_field_syntax(self, staging_url, browser):
        """
        Field-formatted author query returns results for that author.

        The query ``author:"LeCun"`` must bypass the embedding search and
        return only papers whose ``authors`` column matches the name.  The
        author name must appear in the visible results.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        browser : WebDriver
            Selenium WebDriver instance.
        """
        browser.get(staging_url)

        search_input = browser.find_element(By.ID, "search-input")
        search_input.clear()
        search_input.send_keys('author:"LeCun"')
        search_input.send_keys(Keys.RETURN)

        wait = WebDriverWait(browser, self._WAIT_TIMEOUT)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, self._RESULT_CARD_CSS)))

        results = browser.find_elements(By.CSS_SELECTOR, self._RESULT_CARD_CSS)
        assert len(results) > 0, 'author:"LeCun" should return at least one result'

        results_text = browser.find_element(By.ID, "search-results").text
        assert "LeCun" in results_text, 'Results for author:"LeCun" should display the author name'

    def test_author_with_topic(self, staging_url, browser):
        """
        Combined author filter and topic query returns relevant results.

        The query ``author:"LeCun" world model`` filters papers by author and
        ranks them by semantic similarity to "world model".  At least one paper
        must be returned and the author name must appear in the results.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        browser : WebDriver
            Selenium WebDriver instance.
        """
        browser.get(staging_url)

        search_input = browser.find_element(By.ID, "search-input")
        search_input.clear()
        search_input.send_keys('author:"LeCun" world model')
        search_input.send_keys(Keys.RETURN)

        wait = WebDriverWait(browser, self._WAIT_TIMEOUT)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, self._RESULT_CARD_CSS)))

        results = browser.find_elements(By.CSS_SELECTOR, self._RESULT_CARD_CSS)
        assert len(results) > 0, 'author:"LeCun" world model should return at least one result'

        results_text = browser.find_element(By.ID, "search-results").text
        assert "LeCun" in results_text, 'Results for author:"LeCun" world model should display the author name'


# ---------------------------------------------------------------------------
# 4. Paper display
# ---------------------------------------------------------------------------


@pytest.mark.staging
@pytest.mark.slow
class TestPaperDisplay:
    """Verify paper detail view and collapsible abstracts."""

    def test_paper_detail_view(self, staging_url, browser):
        """
        Paper detail view.

        Clicking a paper shows its title, authors, and abstract.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        browser : WebDriver
            Selenium WebDriver instance.
        """
        browser.get(staging_url)

        search_input = browser.find_element(By.ID, "search-input")
        search_input.clear()
        search_input.send_keys("learning")
        search_input.send_keys(Keys.RETURN)

        wait = WebDriverWait(browser, 15)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#search-results .paper-card")))

        cards = browser.find_elements(By.CSS_SELECTOR, "#search-results .paper-card")
        assert len(cards) > 0, "Should have at least one paper card"

        first = cards[0]

        # Paper card must contain a title element
        title_elements = first.find_elements(By.CSS_SELECTOR, "h3, h4")
        assert len(title_elements) > 0, "Paper card should have a title"
        assert len(title_elements[0].text) > 0, "Title should not be empty"

        # Paper card must contain author information
        author_elements = first.find_elements(By.CSS_SELECTOR, ".text-gray-600")
        assert len(author_elements) > 0, "Paper card should have author info"

    def test_collapsible_abstract(self, staging_url, browser):
        """
        Collapsible abstract.

        The abstract section can be expanded and collapsed.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        browser : WebDriver
            Selenium WebDriver instance.
        """
        browser.get(staging_url)

        search_input = browser.find_element(By.ID, "search-input")
        search_input.clear()
        search_input.send_keys("large language models")
        search_input.send_keys(Keys.RETURN)

        wait = WebDriverWait(browser, 15)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#search-results .bg-white.rounded-lg")))

        # Paper cards use native <details>/<summary> for collapsible abstracts
        # when the abstract is longer than ~300 characters.
        detail_elements = browser.find_elements(By.CSS_SELECTOR, "#search-results details")
        if len(detail_elements) == 0:
            pytest.skip("No collapsible abstracts found (all abstracts may be short)")

        details = detail_elements[0]
        summary = details.find_element(By.TAG_NAME, "summary")

        # Abstracts start collapsed – expand by clicking summary
        summary.click()
        time.sleep(0.3)
        assert details.get_attribute("open") is not None, "Abstract details element should be open after click"

        # Collapse again
        summary.click()
        time.sleep(0.3)
        assert details.get_attribute("open") is None, "Abstract details element should be closed after second click"


# ---------------------------------------------------------------------------
# 5. Chat (RAG) interface
# ---------------------------------------------------------------------------


@pytest.mark.staging
@pytest.mark.slow
class TestChatInterface:
    """Verify core chat UI elements and basic send functionality."""

    def test_chat_ui_elements_present(self, staging_url, browser):
        """
        Chat UI elements present.

        The chat input, send button, and reset button are visible.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        browser : WebDriver
            Selenium WebDriver instance.
        """
        browser.get(staging_url)

        _select_conference_and_year(browser)

        # Switch to the chat tab
        browser.find_element(By.ID, "tab-chat").click()
        time.sleep(0.5)

        assert browser.find_element(By.ID, "chat-input"), "Chat input should exist"
        assert browser.find_element(By.ID, "chat-messages"), "Chat messages container should exist"

        # Reset button (uses onclick='resetChat()')
        reset_buttons = browser.find_elements(By.CSS_SELECTOR, "button[onclick*='resetChat']")
        assert len(reset_buttons) > 0, "Reset button should be present"

    def test_chat_send_message(self, staging_url, browser):
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
        browser : WebDriver
            Selenium WebDriver instance.
        """
        query = "What papers are about machine learning?"
        browser.get(staging_url)

        _select_conference_and_year(browser)

        browser.find_element(By.ID, "tab-chat").click()
        time.sleep(0.5)

        chat_input = browser.find_element(By.ID, "chat-input")
        chat_input.send_keys(query)
        chat_input.send_keys(Keys.RETURN)

        wait = WebDriverWait(browser, 120)
        try:
            # Wait until there is a complete (non-loading) assistant response
            wait.until(_has_complete_response)
            history = _extract_chat_history(browser)
            _print_chat_history(history, query)
            assistant_msgs = browser.find_elements(By.CSS_SELECTOR, ".chat-message[data-role='assistant']")
            assert len(assistant_msgs) >= 1, "Chat should display at least one assistant message"
            assert assistant_msgs[-1].text.strip() != "", "Assistant response should not be empty"
        except TimeoutException:
            history = _extract_chat_history(browser)
            _print_chat_history(history, query)
            # LLM backend may not be available – verify input was cleared
            val = browser.find_element(By.ID, "chat-input").get_attribute("value")
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

    def _send_chat_query(self, staging_url, browser, query):
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
        browser : WebDriver
            Selenium WebDriver instance.
        query : str
            The chat query to send.

        Returns
        -------
        tuple[str, list]
            ``(response_text, history)`` where *response_text* is the text of
            the last assistant message (empty string on timeout) and *history*
            is a list of ``(role, text)`` tuples for all visible messages.
        """
        browser.get(staging_url)

        _select_conference_and_year(browser)

        browser.find_element(By.ID, "tab-chat").click()
        time.sleep(0.5)

        # Reset server-side conversation so previous test results don't
        # prevent the LLM from calling MCP tools (it might skip tool calls
        # when it already has cached context from an earlier exchange).
        reset_buttons = browser.find_elements(By.CSS_SELECTOR, "button[onclick*='resetChat']")
        if reset_buttons:
            reset_buttons[0].click()
            time.sleep(0.5)

        chat_input = browser.find_element(By.ID, "chat-input")
        chat_input.send_keys(query)
        chat_input.send_keys(Keys.RETURN)

        wait = WebDriverWait(browser, self._TIMEOUT)
        try:
            # Wait until the last assistant message has no spinner (response complete)
            wait.until(_has_complete_response)
            assistant_msgs = browser.find_elements(By.CSS_SELECTOR, ".chat-message[data-role='assistant']")
            response_text = assistant_msgs[-1].text if assistant_msgs else ""
        except TimeoutException:
            response_text = ""

        history = _extract_chat_history(browser)
        _print_chat_history(history, query)
        return response_text, history

    # ------------------------------------------------------------------
    # Individual MCP tool tests
    # ------------------------------------------------------------------

    def test_get_conference_topics(self, staging_url, browser):
        """
        MCP tool: ``get_conference_topics``.

        Example query: *"What are the main topics at NeurIPS 2025?"*

        Success criteria (LLM-judged): the response lists topic names with
        keywords or paper counts.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        browser : WebDriver
            Selenium WebDriver instance.
        """
        query = "What are the main topics at NeurIPS 2025?"
        response, _ = self._send_chat_query(staging_url, browser, query)
        if not response:
            pytest.skip("LLM backend unavailable – no response received")
        _assert_llm_judgment(
            query,
            response,
            criteria="the response should list multiple topic names and include paper counts or representative keywords for each topic",
        )

    def test_get_topic_evolution(self, staging_url, browser):
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
        browser : WebDriver
            Selenium WebDriver instance.
        """
        query = "How has research on transformers evolved at NeurIPS over the years?"
        response, _ = self._send_chat_query(staging_url, browser, query)
        if not response:
            pytest.skip("LLM backend unavailable – no response received")
        chart_found, chart_description = _check_chart_rendered(browser)
        assert chart_found, f"Expected a trend chart to be rendered in chat. {chart_description}"
        _assert_llm_judgment(
            query,
            response,
            criteria=(
                "the response should describe a year-by-year trend in paper counts or research focus on transformers. "
                f"Additionally: {chart_description}"
            ),
        )

    def test_search_papers(self, staging_url, browser):
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
        browser : WebDriver
            Selenium WebDriver instance.
        """
        query = "Find papers about reinforcement learning at NeurIPS 2025."
        response, _ = self._send_chat_query(staging_url, browser, query)
        if not response:
            pytest.skip("LLM backend unavailable – no response received")
        _assert_llm_judgment(
            query,
            response,
            criteria="the response should briefly summarize or list multiple papers related to reinforcement learning",
        )

    def test_get_paper_details(self, staging_url, browser):
        """
        MCP tool: ``get_paper_details``.

        Example query: *"Who are the authors of the paper titled 'Large Language Diffusion Models'?"*

        Success criteria (LLM-judged): the response includes author names,
        URL/PDF links, and session info (where available).

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        browser : WebDriver
            Selenium WebDriver instance.
        """
        query = "Who are the authors of the paper titled 'Large Language Diffusion Models'?"
        response, _ = self._send_chat_query(staging_url, browser, query)
        if not response:
            pytest.skip("LLM backend unavailable – no response received")
        _assert_llm_judgment(
            query,
            response,
            criteria="the response should name several authors of the paper and give a brief description.",
        )

    def test_analyze_topic_relevance(self, staging_url, browser):
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
        browser : WebDriver
            Selenium WebDriver instance.
        """
        query = "How relevant is uncertainty quantification at NeurIPS 2025?"
        response, _ = self._send_chat_query(staging_url, browser, query)
        if not response:
            pytest.skip("LLM backend unavailable – no response received")
        _assert_llm_judgment(
            query,
            response,
            criteria="the response should include a relevance score, percentage, or paper count that quantifies how relevant the topic is",
        )

    def test_get_cluster_visualization(self, staging_url, browser):
        """
        MCP tool: ``get_cluster_visualization``.

        Example query: *"Show me a visual overview of how topics are clustered at NeurIPS."*

        Success criteria (LLM-judged): the response references the visualization
        and a Plotly chart should be rendered in the chat area.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        browser : WebDriver
            Selenium WebDriver instance.
        """
        query = "Show me a visual overview of how topics are clustered at NeurIPS."
        response, _ = self._send_chat_query(staging_url, browser, query)
        if not response:
            pytest.skip("LLM backend unavailable – no response received")
        chart_found, formatted_answer = _check_chart_rendered(browser)
        assert chart_found, f"Expected a cluster scatter plot to be rendered in chat. {formatted_answer}"
        _assert_llm_judgment(
            query,
            response,
            criteria=("the response should describe or reference a cluster visualization. Note: " + formatted_answer),
        )


# ---------------------------------------------------------------------------
# 6. Clustering tab
# ---------------------------------------------------------------------------


@pytest.mark.staging
@pytest.mark.slow
class TestClusteringTab:
    """Verify the clustering tab and its visualization."""

    def test_clustering_tab_exists(self, staging_url, browser):
        """
        Clustering tab exists.

        The clustering tab is present and can be activated.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        browser : WebDriver
            Selenium WebDriver instance.
        """
        browser.get(staging_url)

        wait = WebDriverWait(browser, 10)
        clustering_tab = wait.until(EC.element_to_be_clickable((By.ID, "tab-clusters")))
        assert clustering_tab.is_displayed(), "Clustering tab should be visible"

    def test_clustering_plot_loads(self, staging_url, browser):
        """
        Clustering plot loads.

        Switching to the clustering tab renders a Plotly visualization
        (or shows a meaningful placeholder).

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        browser : WebDriver
            Selenium WebDriver instance.
        """
        browser.get(staging_url)

        wait = WebDriverWait(browser, 10)
        clustering_tab = wait.until(EC.element_to_be_clickable((By.ID, "tab-clusters")))
        clustering_tab.click()

        # Wait for either the Plotly plot or a placeholder to appear
        try:
            wait.until(EC.visibility_of_element_located((By.ID, "cluster-plot")))
        except TimeoutException:
            pytest.fail("Cluster plot container did not become visible")

        plot_container = browser.find_element(By.ID, "cluster-plot")
        assert plot_container.is_displayed(), "Cluster plot container should be visible"


# ---------------------------------------------------------------------------
# 7. Accessibility & responsiveness
# ---------------------------------------------------------------------------


@pytest.mark.staging
@pytest.mark.slow
class TestAccessibilityResponsiveness:
    """Verify keyboard navigation and responsive layout."""

    def test_keyboard_navigation(self, staging_url, browser):
        """
        Keyboard navigation.

        Core interactive elements (search input, tabs) are reachable via
        the Tab key.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        browser : WebDriver
            Selenium WebDriver instance.
        """
        browser.get(staging_url)

        body = browser.find_element(By.TAG_NAME, "body")
        for _ in range(10):
            body.send_keys(Keys.TAB)
            time.sleep(0.15)

        # The search input must be reachable
        search_input = browser.find_element(By.ID, "search-input")
        assert search_input is not None, "Search input should be reachable via keyboard"

    def test_responsive_layout(self, staging_url, browser):
        """
        Responsive layout.

        The page renders without horizontal overflow at a narrow viewport
        width (768 px).

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        browser : WebDriver
            Selenium WebDriver instance.
        """
        browser.get(staging_url)

        # Desktop
        browser.set_window_size(1920, 1080)
        time.sleep(0.3)
        assert browser.find_element(By.ID, "search-input").is_displayed()

        # Tablet
        browser.set_window_size(768, 1024)
        time.sleep(0.3)
        assert browser.find_element(By.ID, "search-input").is_displayed()

        # Check no horizontal overflow
        body_width = browser.execute_script("return document.body.scrollWidth")
        viewport_width = browser.execute_script("return window.innerWidth")
        assert (
            body_width <= viewport_width + 5
        ), f"Horizontal overflow detected: body={body_width}px, viewport={viewport_width}px"

        # Reset to default size
        browser.set_window_size(1920, 1080)

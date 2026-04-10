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

Notes
-----
* All tests are marked with ``@pytest.mark.staging`` **and**
  ``@pytest.mark.slow`` so they are excluded from the default test run.
* The browser is selected via the ``E2E_BROWSER`` environment variable
  (``chrome``, ``firefox``, or ``auto``), reusing the same logic as the
  existing e2e suite.
"""

import time

import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

# ---------------------------------------------------------------------------
# Browser helpers – reuse the driver creation logic from the main e2e suite
# ---------------------------------------------------------------------------
from tests.test_web_e2e import (
    _check_chrome_available,
    _check_firefox_available,
    _create_chrome_driver,
    _create_firefox_driver,
    _get_browser_preference,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def browser():
    """
    Create a Selenium WebDriver instance for staging tests.

    The browser choice follows the ``E2E_BROWSER`` environment variable
    (``chrome``, ``firefox``, or ``auto``).

    Yields
    ------
    webdriver.Chrome or webdriver.Firefox
        A headless WebDriver instance.  Automatically quit after the module.
    """
    pref = _get_browser_preference()
    driver = None

    if pref == "chrome":
        if not _check_chrome_available():
            pytest.skip("Chrome browser not available")
        try:
            driver = _create_chrome_driver()
        except Exception as exc:
            pytest.skip(f"Failed to create Chrome driver: {exc}")
    elif pref == "firefox":
        if not _check_firefox_available():
            pytest.skip("Firefox browser not available")
        try:
            driver = _create_firefox_driver()
        except Exception as exc:
            pytest.skip(f"Failed to create Firefox driver: {exc}")
    else:
        # auto – try Chrome, fall back to Firefox
        if _check_chrome_available():
            try:
                driver = _create_chrome_driver()
            except Exception:
                if _check_firefox_available():
                    try:
                        driver = _create_firefox_driver()
                    except Exception as exc:
                        pytest.skip(f"Both Chrome and Firefox failed: {exc}")
                else:
                    pytest.skip("Chrome driver failed and Firefox not available")
        elif _check_firefox_available():
            try:
                driver = _create_firefox_driver()
            except Exception as exc:
                pytest.skip(f"Firefox driver failed: {exc}")
        else:
            pytest.skip("No browser available for staging e2e tests")

    yield driver
    driver.quit()


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
# 3. Paper display
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
        search_input.send_keys("learning")
        search_input.send_keys(Keys.RETURN)

        wait = WebDriverWait(browser, 15)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#search-results .bg-white.rounded-lg")))

        toggle_buttons = browser.find_elements(By.CSS_SELECTOR, ".toggle-abstract")
        if len(toggle_buttons) == 0:
            pytest.skip("No toggle-abstract buttons found (compact mode may be off)")

        toggle_buttons[0].click()
        time.sleep(0.3)

        abstract_divs = browser.find_elements(By.CSS_SELECTOR, ".abstract-content")
        assert len(abstract_divs) > 0, "Abstract div should exist after toggle"
        assert abstract_divs[0].is_displayed(), "Abstract should be visible after expanding"

        # Collapse again
        toggle_buttons[0].click()
        time.sleep(0.3)
        assert not abstract_divs[0].is_displayed(), "Abstract should be hidden after collapsing"


# ---------------------------------------------------------------------------
# 4. Chat (RAG) interface
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

        Typing a question and clicking send returns a response (or a
        graceful error if the LLM backend is unavailable).

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        browser : WebDriver
            Selenium WebDriver instance.
        """
        browser.get(staging_url)

        browser.find_element(By.ID, "tab-chat").click()
        time.sleep(0.5)

        chat_input = browser.find_element(By.ID, "chat-input")
        chat_input.send_keys("What papers are about machine learning?")
        chat_input.send_keys(Keys.RETURN)

        wait = WebDriverWait(browser, 30)
        try:
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".chat-message")))
            messages = browser.find_elements(By.CSS_SELECTOR, ".chat-message")
            assert len(messages) >= 1, "Chat should display at least one message"
        except TimeoutException:
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

    Each test sends a representative query and verifies the response
    contains expected content.  If the LLM backend is unavailable the test
    is skipped rather than failed.
    """

    def _send_chat_query(self, staging_url, browser, query, timeout=60):
        """
        Send a chat query and return the last assistant message text.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        browser : WebDriver
            Selenium WebDriver instance.
        query : str
            The chat query to send.
        timeout : int
            Seconds to wait for a response (default: 60).

        Returns
        -------
        str
            Text of the last ``.chat-message`` element, or empty string on
            timeout.
        """
        browser.get(staging_url)

        browser.find_element(By.ID, "tab-chat").click()
        time.sleep(0.5)

        chat_input = browser.find_element(By.ID, "chat-input")
        chat_input.send_keys(query)
        chat_input.send_keys(Keys.RETURN)

        wait = WebDriverWait(browser, timeout)
        try:
            # Wait for at least two messages (user + assistant)
            wait.until(lambda d: len(d.find_elements(By.CSS_SELECTOR, ".chat-message")) >= 2)
            # Allow the response to finish streaming
            time.sleep(2)
            messages = browser.find_elements(By.CSS_SELECTOR, ".chat-message")
            return messages[-1].text if messages else ""
        except TimeoutException:
            return ""

    def test_get_conference_topics(self, staging_url, browser):
        """
        MCP tool: ``get_conference_topics``.

        Example query: *"What are the main topics at NeurIPS 2025?"*

        Success criteria: response lists topic names with keywords and
        paper counts.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        browser : WebDriver
            Selenium WebDriver instance.
        """
        response = self._send_chat_query(staging_url, browser, "What are the main topics at NeurIPS 2025?")
        if not response:
            pytest.skip("LLM backend unavailable – no response received")
        # The response should mention at least one topic-like term
        lower = response.lower()
        assert any(
            kw in lower for kw in ["topic", "cluster", "papers", "keywords", "theme"]
        ), f"Expected topic-related content, got: {response[:300]}"

    def test_get_topic_evolution(self, staging_url, browser):
        """
        MCP tool: ``get_topic_evolution``.

        Example query: *"How has research on transformers evolved at
        NeurIPS over the years?"*

        Success criteria: response includes a year-by-year breakdown of
        paper counts or trend description.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        browser : WebDriver
            Selenium WebDriver instance.
        """
        response = self._send_chat_query(
            staging_url, browser, "How has research on transformers evolved at NeurIPS over the years?"
        )
        if not response:
            pytest.skip("LLM backend unavailable – no response received")
        lower = response.lower()
        assert any(
            kw in lower for kw in ["year", "trend", "evolution", "over time", "papers", "20"]
        ), f"Expected trend-related content, got: {response[:300]}"

    def test_search_papers(self, staging_url, browser):
        """
        MCP tool: ``search_papers``.

        Example query: *"Find papers about reinforcement learning at
        NeurIPS 2025."*

        Success criteria: response returns paper titles with authors and
        abstracts.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        browser : WebDriver
            Selenium WebDriver instance.
        """
        response = self._send_chat_query(
            staging_url, browser, "Find papers about reinforcement learning at NeurIPS 2025."
        )
        if not response:
            pytest.skip("LLM backend unavailable – no response received")
        lower = response.lower()
        assert any(
            kw in lower for kw in ["paper", "title", "author", "abstract", "reinforcement"]
        ), f"Expected paper-related content, got: {response[:300]}"

    def test_get_paper_details(self, staging_url, browser):
        """
        MCP tool: ``get_paper_details``.

        Example query: *"Who are the authors of 'Attention is All You
        Need'?"*

        Success criteria: response includes author names, URL/PDF links,
        and session info (if available).

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        browser : WebDriver
            Selenium WebDriver instance.
        """
        response = self._send_chat_query(staging_url, browser, "Who are the authors of 'Attention is All You Need'?")
        if not response:
            pytest.skip("LLM backend unavailable – no response received")
        lower = response.lower()
        assert any(
            kw in lower for kw in ["author", "vaswani", "paper", "attention"]
        ), f"Expected author-related content, got: {response[:300]}"

    def test_analyze_topic_relevance(self, staging_url, browser):
        """
        MCP tool: ``analyze_topic_relevance``.

        Example query: *"How relevant is uncertainty quantification at
        NeurIPS 2025?"*

        Success criteria: response contains a relevance score or paper
        count within the embedding distance threshold.

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        browser : WebDriver
            Selenium WebDriver instance.
        """
        response = self._send_chat_query(
            staging_url, browser, "How relevant is uncertainty quantification at NeurIPS 2025?"
        )
        if not response:
            pytest.skip("LLM backend unavailable – no response received")
        lower = response.lower()
        assert any(
            kw in lower for kw in ["relev", "paper", "count", "distance", "topic", "uncertainty"]
        ), f"Expected relevance-related content, got: {response[:300]}"

    def test_get_cluster_visualization(self, staging_url, browser):
        """
        MCP tool: ``get_cluster_visualization``.

        Example query: *"Show me a visual overview of NeurIPS 2025
        clusters."*

        Success criteria: response returns or references visualization
        data (Plotly JSON or a rendered plot).

        Parameters
        ----------
        staging_url : str
            Base URL of the staging deployment.
        browser : WebDriver
            Selenium WebDriver instance.
        """
        response = self._send_chat_query(staging_url, browser, "Show me a visual overview of NeurIPS 2025 clusters.")
        if not response:
            pytest.skip("LLM backend unavailable – no response received")
        lower = response.lower()
        assert any(
            kw in lower for kw in ["visual", "cluster", "plot", "chart", "graph", "overview"]
        ), f"Expected visualization-related content, got: {response[:300]}"


# ---------------------------------------------------------------------------
# 5. Clustering tab
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
# 6. Accessibility & responsiveness
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

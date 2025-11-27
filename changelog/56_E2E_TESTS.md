# End-to-End Test Implementation

## Summary

Added comprehensive end-to-end (E2E) tests for the web UI using Selenium WebDriver for browser automation. These tests verify the actual user interface functionality through real browser interactions.

## Changes Made

### 1. Dependencies Added

Updated `pyproject.toml` to include:
- `selenium>=4.0.0` - Browser automation framework
- `webdriver-manager>=4.0.0` - Automatic WebDriver management

### 2. Test Configuration

Added new pytest marker in `pyproject.toml`:
```toml
markers = [
    ...
    "e2e: marks tests as end-to-end tests (require browser automation)",
]
```

### 3. Test File Created

**File**: `tests/test_web_e2e.py`

Contains two test classes with comprehensive test coverage:

#### `TestWebUIE2E`
Main E2E test class with the following tests:

1. **Page Load Tests**
   - `test_page_loads` - Verifies page loads correctly with expected elements
   - `test_search_tab_visible_by_default` - Checks default tab visibility
   - `test_switch_to_chat_tab` - Tests tab switching functionality

2. **Search Functionality Tests**
   - `test_keyword_search_interaction` - Full search user flow
   - `test_search_limit_filter` - Limit dropdown functionality
   - `test_search_with_topic_filter` - Topic filter application
   - `test_search_with_eventtype_filter` - Event type filter application
   - `test_empty_search_shows_message` - Empty search handling
   - `test_search_no_results` - No results scenario
   - `test_search_special_characters` - Special character handling
   - `test_multiple_searches_in_sequence` - Sequential searches

3. **UI Interaction Tests**
   - `test_collapsible_abstract` - Abstract expand/collapse
   - `test_select_deselect_all_filters` - Filter selection controls
   - `test_stats_display` - Statistics display verification
   - `test_paper_detail_view` - Paper detail viewing

4. **Chat Interface Tests**
   - `test_chat_interface_elements` - Chat UI elements presence
   - `test_chat_filters_exist` - Chat filter elements

5. **Layout and Navigation Tests**
   - `test_responsive_layout` - Responsive design at different viewport sizes
   - `test_browser_back_forward` - Browser navigation
   - `test_page_no_javascript_errors` - JavaScript error checking

#### `TestWebUIAccessibility`
Accessibility-focused tests:

1. `test_keyboard_navigation` - Keyboard navigation support
2. `test_form_labels` - Form accessibility
3. `test_semantic_html` - Semantic HTML structure

## Test Features

### Browser Automation

- Supports both Chrome and Firefox browsers
- Uses headless mode by default for CI/CD compatibility
- Automatic WebDriver management via `webdriver-manager`
- Graceful handling of missing browsers (skips tests)
- Environment variable control for browser selection

### Test Database
- Creates isolated test database with realistic sample data
- 5 test papers covering different topics and event types
- 9 authors with realistic affiliations
- Proper paper-author relationships

### Web Server Fixture
- Starts Flask app in background thread
- Uses dynamic port allocation to avoid conflicts
- Patches configuration to use test database
- Automatic cleanup after tests

### Robust Test Design
- Explicit waits for dynamic content
- Multiple viewport sizes tested
- Console log checking for JavaScript errors
- Timeout handling for various scenarios

## Running the Tests

### Run All E2E Tests
```bash
pytest tests/test_web_e2e.py -v -m e2e
```

### Run Specific Test
```bash
pytest tests/test_web_e2e.py::TestWebUIE2E::test_page_loads -v -m e2e
```

### Run Without Slow Tests Filter
```bash
pytest tests/test_web_e2e.py -v -m e2e -m slow
```

### Run with Chrome Visible (Non-Headless)
Modify the `browser` fixture to remove the `--headless` argument.

### Run with Firefox Instead of Chrome
```bash
E2E_BROWSER=firefox pytest tests/test_web_e2e.py -v -m e2e
```

### Run with Specific Browser Only
```bash
# Force Chrome only
E2E_BROWSER=chrome pytest tests/test_web_e2e.py -v -m e2e

# Force Firefox only
E2E_BROWSER=firefox pytest tests/test_web_e2e.py -v -m e2e
```

## Prerequisites

### Browser Installation
Tests support both Google Chrome and Mozilla Firefox browsers.

**Chrome Installation:**

macOS:
```bash
brew install --cask google-chrome
```

Linux (Debian/Ubuntu):
```bash
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo dpkg -i google-chrome-stable_current_amd64.deb
```

**Firefox Installation:**

macOS:
```bash
brew install --cask firefox
```

Linux (Debian/Ubuntu):
```bash
sudo apt-get update
sudo apt-get install firefox
```

**CI/CD Environments:**
Most CI platforms have browsers pre-installed:
- GitHub Actions: Chrome and Firefox pre-installed
- GitLab CI: Use `selenium/standalone-chrome` or `selenium/standalone-firefox` images
- CircleCI: Chrome and Firefox pre-installed

### Dependencies Installation
```bash
pip install selenium webdriver-manager
```

Or install all dev dependencies:
```bash
pip install -e ".[dev]"
```

## Test Markers

Tests are marked with both:
- `@pytest.mark.e2e` - Identifies as end-to-end test
- `@pytest.mark.slow` - Marks as slow (browser automation takes time)

By default, slow tests are excluded from normal test runs.

## Architecture

### Fixtures

1. **`test_database`** (module scope)
   - Creates SQLite database with test data
   - Includes papers, authors, and relationships
   - Persists for entire test module

2. **`web_server`** (module scope)
   - Starts Flask app in background thread
   - Patches configuration for test database
   - Waits for server to be ready
   - Cleans up after test module

3. **`browser`** (function scope)
   - Creates Chrome WebDriver instance
   - Configures headless mode
   - Handles Chrome unavailability gracefully
   - Automatically quits after each test

### Test Data

Papers cover major ML topics:
- Attention mechanisms (Transformers)
- Natural Language Processing (BERT, NMT)
- Computer Vision (ResNet)
- Generative Models (GANs)

Multiple event types:
- Oral presentations
- Poster presentations
- Spotlight presentations

## Best Practices Followed

1. **Isolation**: Each test is independent
2. **Cleanup**: Automatic resource cleanup
3. **Waits**: Explicit waits for dynamic content
4. **Error Handling**: Graceful failure when Chrome unavailable
5. **Documentation**: Comprehensive docstrings
6. **Reusability**: Shared fixtures across tests
7. **Realistic Data**: Test data mirrors production structure

## Future Enhancements

Potential additions:
1. Safari support for additional cross-browser testing
2. Visual regression testing (screenshot comparison)
3. Performance metrics collection
4. Chat interaction E2E tests (requires LLM mock)
5. Form validation error message verification
6. File download testing
7. Mobile touch gesture simulation

## Troubleshooting

### Browser Not Found
**Error**: `cannot find Chrome binary` or `cannot find Firefox binary`

**Solutions**: 
1. Install the browser (see Browser Installation section above)
2. Use the other browser: `E2E_BROWSER=firefox pytest tests/test_web_e2e.py -m e2e`
3. Set browser binary path environment variable:
```bash
export CHROME_BIN=/path/to/chrome
# or
export FIREFOX_BIN=/path/to/firefox
```

### Port Already in Use
**Error**: `Address already in use`

**Solution**: Tests use `find_free_port()` to avoid conflicts. If issues persist, check for zombie Flask processes:
```bash
ps aux | grep flask
kill <pid>
```

### Slow Test Execution
E2E tests are inherently slower. To speed up:
1. Run specific tests instead of full suite
2. Use headless mode (default)
3. Reduce sleep/wait times (may cause flakiness)
4. Run in parallel: `pytest -n auto` (requires pytest-xdist)

### Flaky Tests
If tests occasionally fail:
1. Increase wait timeouts
2. Check network connectivity
3. Ensure sufficient system resources
4. Review browser console logs

## Coverage

The E2E tests complement existing integration tests by:
- Testing actual browser rendering
- Verifying JavaScript functionality
- Checking CSS/layout issues
- Simulating real user interactions
- Validating end-to-end workflows

While integration tests verify API responses, E2E tests ensure the complete user experience works correctly.

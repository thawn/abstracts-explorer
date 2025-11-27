# E2E Test Driver Installation Optimization

**Date**: November 27, 2025  
**Status**: ✅ Completed

## Summary

Optimized the E2E test suite to install browser drivers (ChromeDriver/GeckoDriver) only once per test session and only when tests are actually executed (not skipped).

## Problem

Previously, the E2E test suite had several inefficiencies:

1. **Multiple Installations**: Browser drivers were installed multiple times during a single test session:
   - Once during browser availability check in `_check_chrome_available()`
   - Again when creating the actual driver in `_create_chrome_driver()`
   - Repeated for each test that used the `browser` fixture

2. **Installation on Skip**: Driver installation occurred even when checking browser availability, which happened before determining if tests would be skipped

3. **No Caching**: No mechanism to reuse driver paths across multiple test instances

## Solution

Implemented a module-level caching mechanism that:

1. **Browser Availability Checks**: Modified `_check_chrome_available()` and `_check_firefox_available()` to only check if browser binaries exist, without installing drivers

   ```python
   # Before: Installed driver to test availability
   service = ChromeService(ChromeDriverManager().install())
   
   # After: Only checks if browser binary exists
   chrome_binary = shutil.which("google-chrome") or shutil.which("chromium")
   ```

2. **Single Driver Installation**: Modified `_create_chrome_driver()` and `_create_firefox_driver()` to install drivers only once per session using cache

   ```python
   # Install driver only once per session
   if _driver_cache["chrome"] is None:
       _driver_cache["chrome"] = ChromeDriverManager().install()
   service = ChromeService(_driver_cache["chrome"])
   ```

3. **Module-Level Cache**: Added `_driver_cache` dictionary to store:
   - Driver installation paths
   - Browser availability status
   - Results persist across all tests in the session

## Changes Made

### Modified Functions

1. **`_check_chrome_available()`**
   - Now uses `shutil.which()` to check for Chrome binary
   - Caches result to avoid repeated checks
   - No driver installation during check

2. **`_check_firefox_available()`**
   - Now uses `shutil.which()` to check for Firefox binary
   - Caches result to avoid repeated checks
   - No driver installation during check

3. **`_create_chrome_driver()`**
   - Installs driver only once and caches the path
   - Reuses cached path for subsequent driver instances

4. **`_create_firefox_driver()`**
   - Installs driver only once and caches the path
   - Reuses cached path for subsequent driver instances

### Cache Structure

```python
_driver_cache = {
    "chrome": None,              # Path to installed ChromeDriver
    "firefox": None,             # Path to installed GeckoDriver
    "chrome_available": None,    # Boolean: Is Chrome browser available?
    "firefox_available": None,   # Boolean: Is Firefox browser available?
}
```

## Benefits

1. **Faster Test Execution**: Driver installation happens only once instead of 22+ times
2. **Reduced Network Usage**: Single download per driver type per session
3. **No Installation on Skip**: Driver installation only occurs when tests actually run
4. **Better CI/CD Performance**: Significant speedup in continuous integration environments

## Testing

All 22 E2E tests pass successfully:

```bash
pytest tests/test_web_e2e.py -v -m e2e
# ✅ 22 passed in ~160 seconds
```

Verified behavior:

- ✅ Driver installed only once per session
- ✅ No installation when running non-E2E tests
- ✅ Browser availability checks don't trigger installation
- ✅ Multiple tests reuse the same driver installation

## Example Usage

```bash
# Run all E2E tests - driver installed once
pytest tests/test_web_e2e.py -m e2e

# Run specific test - driver installed only if needed
pytest tests/test_web_e2e.py::TestWebUIE2E::test_page_loads -m e2e

# Run with specific browser - only that driver installed
E2E_BROWSER=firefox pytest tests/test_web_e2e.py -m e2e
```

## Technical Details

### Installation Flow

1. Test collection phase:
   - No driver installation
   - No browser checks

2. First test execution:
   - `browser` fixture requested
   - Browser availability checked (binary lookup only)
   - Driver installed and cached
   - Browser instance created

3. Subsequent tests:
   - Reuse cached driver path
   - Create new browser instances
   - No re-installation

### Cache Lifetime

- **Scope**: Module-level, persists for entire pytest session
- **Reset**: Cleared when pytest process exits
- **Sharing**: Shared across all tests in `test_web_e2e.py`

## Related Files

- `tests/test_web_e2e.py`: E2E test suite with optimization
- `changelog/56_E2E_TESTS.md`: Original E2E implementation
- `changelog/57_FIREFOX_E2E_SUPPORT.md`: Firefox support addition

## Future Considerations

- Consider adding explicit cache clearing mechanism for testing
- Monitor webdriver-manager updates for built-in caching improvements
- Potentially extend caching to other test files if E2E tests expand

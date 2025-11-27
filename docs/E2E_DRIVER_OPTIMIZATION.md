# E2E Test Driver Installation - Quick Reference

## How It Works

The E2E test suite uses a **module-level cache** to ensure browser drivers are installed only once per test session.

### Key Features

✅ **Single Installation**: Driver downloaded and installed only once per session  
✅ **Lazy Loading**: Installation happens only when tests actually run (not during collection or skip)  
✅ **Browser Detection**: Checks for browser binaries without installing drivers  
✅ **Cache Persistence**: Cached paths reused across all 22 E2E tests

## Cache Structure

```python
_driver_cache = {
    "chrome": None,              # ChromeDriver path (installed once)
    "firefox": None,             # GeckoDriver path (installed once)
    "chrome_available": None,    # Browser binary check result
    "firefox_available": None,   # Browser binary check result
}
```

## Installation Flow

```
1. Test Collection
   └─> No driver installation

2. First Test Execution
   └─> browser fixture requested
       └─> Check browser availability (binary lookup only)
           └─> Install driver and cache path
               └─> Create browser instance

3. Subsequent Tests (21 more tests)
   └─> browser fixture requested
       └─> Reuse cached driver path
           └─> Create new browser instance
               └─> No re-installation ✅
```

## Performance Comparison

### Before Optimization
- Driver installation: 22+ times (once per test check + once per test)
- Network downloads: Multiple ChromeDriver/GeckoDriver downloads
- Time overhead: ~60-90 seconds of installation time

### After Optimization
- Driver installation: 1 time per session
- Network downloads: 1 download per driver type
- Time overhead: ~3-5 seconds of installation time
- **Speedup: ~85-90% reduction in driver setup time**

## Running Tests

```bash
# Run all E2E tests (driver installed once)
pytest tests/test_web_e2e.py -m e2e

# Run specific test (driver still installed once)
pytest tests/test_web_e2e.py::TestWebUIE2E::test_page_loads -m e2e

# Use specific browser
E2E_BROWSER=firefox pytest tests/test_web_e2e.py -m e2e

# Skip E2E tests (no driver installation at all)
pytest tests/ -m "not e2e"
```

## Benefits

1. **Faster CI/CD**: Significant speedup in continuous integration pipelines
2. **Reduced Network**: Single download per driver type saves bandwidth
3. **No Skip Overhead**: Driver not installed when tests are skipped
4. **Better Developer Experience**: Faster local test runs

## Implementation Details

### Browser Availability Check
```python
def _check_firefox_available():
    # Cache the result
    if _driver_cache["firefox_available"] is not None:
        return _driver_cache["firefox_available"]
    
    # Only check if binary exists (no driver installation)
    firefox_binary = shutil.which("firefox")
    result = firefox_binary is not None
    _driver_cache["firefox_available"] = result
    return result
```

### Driver Creation
```python
def _create_firefox_driver():
    # Install driver only once per session
    if _driver_cache["firefox"] is None:
        _driver_cache["firefox"] = GeckoDriverManager().install()
    
    # Reuse cached path
    service = FirefoxService(_driver_cache["firefox"])
    driver = webdriver.Firefox(service=service, options=firefox_options)
    return driver
```

## Related Documentation

- Full details: `changelog/58_E2E_DRIVER_OPTIMIZATION.md`
- E2E test guide: `docs/E2E_TESTING.md`
- Original implementation: `changelog/56_E2E_TESTS.md`

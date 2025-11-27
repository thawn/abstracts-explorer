# E2E Testing Quick Reference

## Overview

End-to-end (E2E) tests verify the web UI through automated browser interactions using Selenium. These tests support both Chrome and Firefox browsers.

## Quick Start

### 1. Check Browser Availability

```bash
# Run the browser check script
python examples/check_e2e_browsers.py
```

### 2. Run Tests

```bash
# Run with default settings (tries Chrome, falls back to Firefox)
pytest tests/test_web_e2e.py -m e2e -v

# Use Firefox specifically
E2E_BROWSER=firefox pytest tests/test_web_e2e.py -m e2e -v

# Use Chrome specifically
E2E_BROWSER=chrome pytest tests/test_web_e2e.py -m e2e -v
```

## Browser Selection

### Environment Variable: `E2E_BROWSER`

Controls which browser to use for testing:

| Value            | Behavior                                 |
| ---------------- | ---------------------------------------- |
| `auto` (default) | Tries Chrome first, then Firefox         |
| `chrome`         | Uses Chrome only (skips if unavailable)  |
| `firefox`        | Uses Firefox only (skips if unavailable) |

### Examples

```bash
# Auto mode (default)
pytest tests/test_web_e2e.py -m e2e

# Explicit Chrome
E2E_BROWSER=chrome pytest tests/test_web_e2e.py -m e2e

# Explicit Firefox
E2E_BROWSER=firefox pytest tests/test_web_e2e.py -m e2e
```

## Installation

### Chrome

**macOS:**
```bash
brew install --cask google-chrome
```

**Linux (Ubuntu/Debian):**
```bash
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo dpkg -i google-chrome-stable_current_amd64.deb
```

### Firefox

**macOS:**
```bash
brew install --cask firefox
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install firefox
```

## Test Organization

### Test Classes

**`TestWebUIE2E`** - Main functionality tests (19 tests)
- Page loading and structure
- Search functionality
- Filters (topic, session, event type)
- Tab switching
- UI interactions
- Browser navigation

**`TestWebUIAccessibility`** - Accessibility tests (3 tests)
- Keyboard navigation
- Form labels
- Semantic HTML

### Running Specific Tests

```bash
# Run single test
pytest tests/test_web_e2e.py::TestWebUIE2E::test_page_loads -m e2e -v

# Run test class
pytest tests/test_web_e2e.py::TestWebUIE2E -m e2e -v

# Run accessibility tests only
pytest tests/test_web_e2e.py::TestWebUIAccessibility -m e2e -v

# Run with pattern matching
pytest tests/test_web_e2e.py -k "search" -m e2e -v
```

## Common Use Cases

### CI/CD Integration

```yaml
# GitHub Actions example
- name: Run E2E Tests
  run: |
    pytest tests/test_web_e2e.py -m e2e -v
  env:
    E2E_BROWSER: firefox  # Firefox is more stable in CI
```

### Local Development

```bash
# Quick test during development
pytest tests/test_web_e2e.py::TestWebUIE2E::test_page_loads -m e2e -v

# Full test suite
pytest tests/test_web_e2e.py -m e2e -v
```

### Cross-Browser Testing

```bash
# Test with Chrome
E2E_BROWSER=chrome pytest tests/test_web_e2e.py -m e2e -v

# Test with Firefox
E2E_BROWSER=firefox pytest tests/test_web_e2e.py -m e2e -v

# Compare results
```

## Troubleshooting

### Browser Not Found

**Problem:** Test skips with "browser not available"

**Solutions:**
1. Install the browser (see Installation section)
2. Try the other browser: `E2E_BROWSER=firefox` or `E2E_BROWSER=chrome`
3. Run browser check: `python examples/check_e2e_browsers.py`

### Tests Are Slow

**Expected:** E2E tests take 2-5 seconds per test (browser startup overhead)

**Tips:**
- Run specific tests: `pytest tests/test_web_e2e.py::TestWebUIE2E::test_page_loads -m e2e`
- Use `-v` for progress visibility
- Tests are marked as `slow` by default

### Headless Mode Issues

**Problem:** Tests fail in headless but pass in normal mode

**Debug:**
1. Modify test file to remove `--headless` argument
2. Watch browser automation in real-time
3. Check for timing issues (add more waits)

### Port Conflicts

**Problem:** "Address already in use"

**Solution:**
```bash
# Find and kill Flask processes
ps aux | grep flask
kill <pid>
```

## Test Coverage

E2E tests cover:
- ✓ Page loading and rendering
- ✓ Tab navigation (Search/Chat)
- ✓ Keyword search
- ✓ Filter interactions (topic, session, event type)
- ✓ Results display and formatting
- ✓ Collapsible abstracts
- ✓ Statistics display
- ✓ Responsive layout (multiple viewport sizes)
- ✓ Keyboard navigation
- ✓ Form accessibility
- ✓ JavaScript error detection
- ✓ Browser back/forward navigation

## Performance

Typical execution times:
- Single test: 2-5 seconds
- Test class (19 tests): 45-90 seconds
- Full suite (22 tests): 60-120 seconds

Times vary based on:
- Browser (Firefox slightly faster)
- System resources
- Network conditions (for driver downloads)

## Best Practices

1. **Run specific tests during development** - Faster feedback
2. **Use auto mode locally** - Convenient browser selection
3. **Pin browser in CI/CD** - Consistent environment
4. **Check browser availability first** - Avoid surprises
5. **Monitor test times** - Detect performance issues
6. **Review browser console logs** - Catch JavaScript errors

## Related Files

- `tests/test_web_e2e.py` - E2E test implementation
- `examples/check_e2e_browsers.py` - Browser availability checker
- `changelog/56_E2E_TESTS.md` - Detailed implementation docs
- `changelog/57_FIREFOX_E2E_SUPPORT.md` - Firefox support details
- `README.md` - Project overview and testing section

## Support

For issues or questions:
- Check troubleshooting section above
- Review test output with `-v` flag
- Run browser check script
- Check browser installation
- Review changelog files for details

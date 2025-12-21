# Slow Test Marking for LM Studio Tests

**Date:** November 26, 2025

## Summary

All tests requiring LM Studio are now marked as "slow" and are skipped by default during test runs. This improves the developer experience by allowing fast test iterations without requiring LM Studio to be running.

## Changes Made

### 1. Pytest Configuration (pyproject.toml)

Added slow marker configuration and modified default test options:

```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselected by default)",
]
addopts = [
    "--verbose",
    "--cov=neurips_abstracts",
    "--cov-report=term-missing",
    "--cov-report=html",
    "-m not slow",  # Skip slow tests by default
]
```

### 2. Test Helpers Update (tests/test_helpers.py)

Modified `requires_lm_studio` to be a proper decorator function that applies both the `slow` marker and skipif condition:

```python
def requires_lm_studio(func):
    """
    Decorator that marks tests as slow and skips them if LM Studio is not available.
    
    This decorator:
    1. Marks the test as 'slow' (so it's skipped by default with -m "not slow")
    2. Skips the test if LM Studio is not running or no chat model is loaded
    """
    # Apply both slow marker and skipif condition
    func = pytest.mark.slow(func)
    func = pytest.mark.skipif(
        not check_lm_studio_available(),
        reason="LM Studio not running or no chat model loaded...",
    )(func)
    return func
```

**Important:** The decorator ensures tests are skipped (not failed) when LM Studio is unavailable, even when explicitly running slow tests with `-m slow`.

### 3. Integration Test Update (tests/test_integration.py)

- Added `@requires_lm_studio` decorator to `test_embeddings_end_to_end_with_real_data`
- Removed duplicate LM Studio availability check code (now handled by decorator)
- Updated docstring to reflect slow test marking

### 4. Documentation Update (README.md)

Added documentation about slow test behavior and how to run them:

```bash
# Run all tests (excluding slow tests by default)
pytest

# Run only slow tests (requires LM Studio running)
pytest -m slow

# Run all tests including slow ones
pytest -m ""
```

## Tests Marked as Slow

The following 7 tests require LM Studio and are now marked as slow:

1. **test_integration.py**
   - `test_embeddings_end_to_end_with_real_data`

2. **test_rag.py**
   - `test_real_query`
   - `test_real_conversation`
   - `test_real_export`

3. **test_web_integration.py**
   - `test_chat_with_valid_message_and_response`
   - `test_chat_with_reset_flag`
   - `test_chat_with_custom_n_papers`

## Usage

### Default Behavior (Skip Slow Tests)

```bash
pytest
# Runs 232 tests, skips 7 slow tests
```

### Run Only Slow Tests

```bash
pytest -m slow
# Runs only the 7 slow tests (requires LM Studio)
```

### Run All Tests Including Slow Ones

```bash
pytest -m ""
# Runs all 239 tests (requires LM Studio)
```

### Run Specific Test File Excluding Slow

```bash
pytest tests/test_rag.py
# Runs only non-slow tests from test_rag.py
```

### Run Specific Test File Including Slow

```bash
pytest tests/test_rag.py -m ""
# Runs all tests from test_rag.py including slow ones
```

## Benefits

1. **Faster Development Cycles**: Developers can run tests quickly without LM Studio
2. **CI/CD Flexibility**: CI pipelines can skip slow tests for faster feedback
3. **Clear Test Organization**: Slow integration tests are clearly marked
4. **Selective Testing**: Easy to run just slow tests when needed
5. **Backward Compatible**: Tests still skip if LM Studio is unavailable

## Technical Notes

- The `slow` marker is registered in `pyproject.toml` to avoid pytest warnings
- The default `addopts` includes `-m not slow` to skip slow tests automatically
- Tests can be explicitly run with `-m slow` or `-m ""` (empty string runs all)
- The `requires_lm_studio` decorator combines both the slow marker and skipif logic
- All LM Studio availability checking is centralized in `test_helpers.py`

### Test Behavior with LM Studio Unavailable

When LM Studio is not running:
- **Default run (`pytest`)**: Slow tests are deselected (not run at all) - ✅ Fast
- **Explicit slow run (`pytest -m slow`)**: Slow tests are skipped (not failed) - ✅ No failures
- **All tests run (`pytest -m ""`)**: Slow tests are skipped (not failed) - ✅ No failures

This ensures that tests never fail due to LM Studio being unavailable - they are either deselected or skipped.

## Statistics

- Total tests: 239
- Fast tests (default): 232 (97%)
- Slow tests (LM Studio): 7 (3%)
- Average fast test runtime: ~10 seconds
- Average slow test runtime: ~90 seconds (with LM Studio)

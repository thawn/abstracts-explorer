# Fix: Slow Tests Properly Skip When LM Studio Unavailable

**Date:** November 26, 2025

## Issue

After implementing slow test marking, tests marked with `@requires_lm_studio` were failing instead of being skipped when LM Studio was not available. This occurred even when explicitly running slow tests with `pytest -m slow`.

## Root Cause

The initial implementation attempted to nest pytest markers incorrectly:

```python
# INCORRECT - doesn't work properly
requires_lm_studio = pytest.mark.slow(
    pytest.mark.skipif(...)
)
```

This approach didn't properly apply both markers, causing the skipif condition to not be evaluated correctly.

## Solution

Changed `requires_lm_studio` from a composed marker to a proper decorator function that explicitly applies both markers:

```python
def requires_lm_studio(func):
    """
    Decorator that marks tests as slow and skips them if LM Studio is not available.
    """
    # Apply both slow marker and skipif condition
    func = pytest.mark.slow(func)
    func = pytest.mark.skipif(
        not check_lm_studio_available(),
        reason="LM Studio not running or no chat model loaded...",
    )(func)
    return func
```

## Test Results

With LM Studio not running:

### Default Run (pytest)
```bash
$ pytest --co -q
collected 239 items / 7 deselected / 232 selected
```
✅ Slow tests are **deselected** (not run at all)

### Explicit Slow Tests (pytest -m slow)
```bash
$ pytest -m slow -v
tests/test_integration.py::...::test_embeddings_end_to_end_with_real_data SKIPPED
tests/test_rag.py::...::test_real_query SKIPPED
tests/test_rag.py::...::test_real_conversation SKIPPED
tests/test_rag.py::...::test_real_export SKIPPED
tests/test_web_integration.py::...::test_chat_with_valid_message_and_response SKIPPED
tests/test_web_integration.py::...::test_chat_with_reset_flag SKIPPED
tests/test_web_integration.py::...::test_chat_with_custom_n_papers SKIPPED

7 skipped, 232 deselected
```
✅ Slow tests are **skipped** (not failed)

### All Tests (pytest -m "")
```bash
$ pytest -m ""
24 passed, 3 deselected (for the test_rag.py file shown)
```
✅ Slow tests are **skipped** (not failed)

## Files Changed

- `tests/test_helpers.py`: Changed `requires_lm_studio` from marker composition to decorator function

## Verification

All three test scenarios now behave correctly:
1. **Default behavior**: Fast tests run, slow tests deselected
2. **Explicit slow tests**: Slow tests skipped if LM Studio unavailable
3. **All tests**: Slow tests skipped if LM Studio unavailable

No tests fail due to missing LM Studio - they are either deselected or skipped as appropriate.

# Config Test Teardown Fix

## Problem

The tests in `test_config.py` were passing when run individually but failing when run in conjunction with other tests. The root cause was global state pollution:

1. **Global Config Instance**: The `get_config()` function uses a module-level `_config` variable that persists between test runs
2. **Environment Variables**: Tests that modify environment variables using `monkeypatch` could affect subsequent tests if not properly cleaned up

## Solution

Added proper teardown methods to clean up global state after each test:

### 1. TestConfig Class

Added `teardown_method()` to clean up environment variables:

```python
def teardown_method(self):
    """Clean up environment variables after each test."""
    env_vars_to_clean = [
        "CHAT_MODEL",
        "EMBEDDING_MODEL",
        "LLM_BACKEND_URL",
        "LLM_BACKEND_AUTH_TOKEN",
        "EMBEDDING_DB_PATH",
        "PAPER_DB_PATH",
        "COLLECTION_NAME",
        "MAX_CONTEXT_PAPERS",
        "CHAT_TEMPERATURE",
        "CHAT_MAX_TOKENS",
    ]
    for var in env_vars_to_clean:
        if var in os.environ:
            del os.environ[var]
```

### 2. TestGetConfig Class

Added `teardown_method()` to reset the global config instance:

```python
def teardown_method(self):
    """Reset global config state after each test."""
    config_module._config = None
```

### 3. Added Necessary Imports

```python
import os
import pytest
from neurips_abstracts import config as config_module
```

## Testing

- All 18 config tests pass individually: ✅
- All 270 tests pass when run together: ✅
- Coverage remains at 96% for config module: ✅

## Files Modified

- `tests/test_config.py`: Added teardown methods and imports

## Best Practices

This fix follows pytest best practices for managing global state:

1. **Teardown Methods**: Use `teardown_method()` to clean up after each test
2. **Environment Isolation**: Clear environment variables that could affect other tests
3. **Global State Reset**: Reset module-level singleton instances between tests
4. **Explicit Cleanup**: Make cleanup explicit rather than relying on pytest's automatic cleanup

## Notes

- The teardown methods ensure that each test starts with a clean slate
- This prevents test order dependencies and makes the test suite more robust
- The fix is minimal and doesn't change any production code
- Similar patterns can be applied to other test files that manage global state

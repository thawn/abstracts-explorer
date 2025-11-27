# Multiprocessing Path Resolution Fix

**Date:** November 27, 2025  
**Issue:** Web integration tests failing with 500 errors on Ubuntu (Linux) CI

## Problem

The web integration tests were passing on macOS but failing on Ubuntu with 500 Internal Server Error responses. The failures occurred in tests that interact with the database through the web API:

- `test_api_stats_endpoint`
- `test_api_search_keyword`
- `test_api_paper_detail`
- `test_concurrent_requests`
- `test_search_with_different_limits`
- `test_search_special_characters`
- `test_keyword_search_end_to_end`
- `test_search_comparison_keyword_vs_semantic`
- `test_empty_search_results`

## Root Cause

The issue was related to how `multiprocessing.Process` works differently on different platforms:

1. **Linux (fork)**: Uses `fork()` by default, which can cause the child process to inherit stale configuration from the parent process
2. **macOS (spawn)**: Uses `spawn()` by default, which creates a fresh Python interpreter

When the web server subprocess was started on Linux:

- The `Config` singleton might have already been initialized in the parent process with default values
- The child process inherited this stale config even after setting the environment variable
- Path resolution wasn't explicitly converting to absolute paths, causing issues when the working directory differed between processes

## Solution

Made three key changes to ensure robust path handling across platforms:

### 1. Improved Path Resolution (`config.py`)

Updated `Config._resolve_path()` to always return absolute paths:

```python
def _resolve_path(self, path: str) -> str:
    path_obj = Path(path)
    if path_obj.is_absolute():
        # Return absolute path as string (expanduser to handle ~)
        return str(path_obj.expanduser().absolute())

    # Resolve relative to data_dir and make absolute
    return str((Path(self.data_dir) / path).absolute())
```

**Benefits:**

- Ensures paths work correctly regardless of current working directory
- Handles `~` in paths with `expanduser()`
- Makes paths platform-independent

### 2. Force Config Reload in Subprocess (`test_web_integration.py`)

Added config reload in `start_web_server()`:

```python
def start_web_server(db_path, port):
    import os
    
    # Set environment variable for database path
    os.environ["PAPER_DB_PATH"] = str(db_path)
    
    # Import after setting env var
    from neurips_abstracts.web_ui import run_server
    from neurips_abstracts.config import get_config
    
    # Force reload config to pick up the environment variable
    # This is important on Linux where multiprocessing uses fork()
    get_config(reload=True)
    
    # Run server
    run_server(host="127.0.0.1", port=port, debug=False)
```

**Benefits:**

- Ensures fresh config in subprocess with correct environment variables
- Handles fork() vs spawn() differences automatically

### 3. Ensure Absolute Paths in Test Fixture (`test_web_integration.py`)

Modified `web_server` fixture to use absolute paths:

```python
# Ensure the database path is absolute
abs_db_path = str(Path(test_database).absolute())
os.environ["PAPER_DB_PATH"] = abs_db_path

# Pass absolute path to subprocess
server_process = Process(target=start_web_server, args=(abs_db_path, port), daemon=True)
```

**Benefits:**

- Eliminates any ambiguity about relative vs absolute paths
- Works correctly even if working directory changes

### 4. Added Error Logging (`web_ui/app.py`)

Added detailed error logging to help debug future issues:

```python
except Exception as e:
    logger.error(f"Error in stats endpoint: {e}", exc_info=True)
    return jsonify({"error": str(e)}), 500
```

**Benefits:**

- Makes debugging easier in CI environments
- Provides stack traces for troubleshooting

## Testing

All tests now pass on both platforms:

- ✅ macOS (spawn-based multiprocessing)
- ✅ Ubuntu/Linux (fork-based multiprocessing)

## Files Modified

1. `src/neurips_abstracts/config.py` - Enhanced path resolution
2. `src/neurips_abstracts/web_ui/app.py` - Added error logging
3. `tests/test_web_integration.py` - Fixed test setup and added config reload

## Related Issues

This fix addresses a common pattern when using `multiprocessing` with configuration that needs to be loaded from environment variables. The key insight is that:

1. On Linux with fork(), the child process inherits the parent's memory space, including any already-initialized singleton objects
2. Config must be explicitly reloaded in the child process after setting environment variables
3. Absolute paths should always be used to avoid working directory issues

## Verification

To verify the fix works on Ubuntu, run:

```bash
pytest tests/test_web_integration.py -xvs
```

All 31 tests should pass without any 500 errors.

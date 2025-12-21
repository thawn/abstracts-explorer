# Changelog Entry 33: Web UI Critical Bug Fixes

**Date**: 2025-11-26  
**Type**: Bug Fix  
**Component**: Web UI  
**Status**: Completed

## Summary

Fixed critical bugs in the web UI that prevented it from functioning properly in production. The fixes address configuration initialization, year statistics removal, and thread-safety issues with database connections.

## Issues Fixed

### 1. Configuration Loading at Import Time

**Problem**: The web UI app was calling `config = get_config()` at module import time (line 26). This meant that if the `PAPER_DB_PATH` environment variable was set after the module was imported, the config would still use the old value.

**Impact**: Integration tests couldn't override the database path because config was already initialized.

**Solution**: Changed all config usage to be lazy - now `get_config()` is called within each function that needs config, not at module import time.

**Files Changed**:
- `src/neurips_abstracts/web_ui/app.py`:
  - Removed global `config = get_config()` statement
  - Added `config = get_config()` calls in: `get_database()`, `get_embeddings_manager()`, `get_rag_chat()`, chat endpoint, and `run_server()`

### 2. Year Statistics Still Querying Removed Column

**Problem**: The `/api/stats` endpoint was still querying the `year` column from the database, even though we removed year filtering from the frontend. The database schema doesn't have a `year` column, causing SQL errors.

**Error**: `Query failed: no such column: year`

**Impact**: Stats endpoint returned 500 errors, breaking the frontend statistics display.

**Solution**: Simplified the stats endpoint to only return `total_papers`, removing all year-related queries.

**Files Changed**:
- `src/neurips_abstracts/web_ui/app.py`:
  - Removed year statistics collection from `stats()` endpoint
  - Removed fields: `years`, `year_counts`, `min_year`, `max_year`
  - Now only returns `total_papers`

### 3. SQLite Thread-Safety Issue

**Problem**: Flask spawns multiple threads to handle concurrent requests. The web UI was using a global database connection cached at the module level. SQLite connections can only be used in the thread that created them, causing errors when Flask handled concurrent requests from different threads.

**Error**: `SQLite objects created in a thread can only be used in that same thread. The object was created in thread id 6110490624 and this is thread id 6127316992.`

**Impact**: All database operations failed after the first request in multi-threaded scenarios. Concurrent requests would crash with 500 errors.

### 4. EmbeddingsManager Initialization Parameter Error

**Problem**: The web UI was passing incorrect parameter `db_path` to `EmbeddingsManager.__init__()`, which doesn't accept that parameter. The correct parameters are `lm_studio_url`, `model_name`, `chroma_path`, and `collection_name`.

**Error**: `EmbeddingsManager.__init__() got an unexpected keyword argument 'db_path'`

**Impact**: Semantic search functionality was completely broken. Any attempt to use embeddings would crash with a TypeError.

**Solution**: Fixed initialization to use correct parameters from the EmbeddingsManager API, and added explicit `connect()` call.

**Files Changed**:
- `src/neurips_abstracts/web_ui/app.py`:
  - Changed `get_embeddings_manager()` to pass correct parameters: `lm_studio_url`, `model_name`, `chroma_path`, `collection_name`
  - Added `embeddings_manager.connect()` call after initialization
  - Added `embeddings_manager.create_collection()` to load/create the ChromaDB collection
  - Fixed search endpoint to use `search_similar()` instead of non-existent `search()` method
  - Added code to transform ChromaDB results (nested dict format) into paper list format
  - Now fetches full paper details from database using paper IDs from embeddings results
  - Adds similarity score to each paper in results

**Solution**: Changed database connection management to use Flask's `g` object, which provides thread-local storage. Each request thread now gets its own database connection.

**Files Changed**:
- `src/neurips_abstracts/web_ui/app.py`:
  - Removed global `db = None` variable
  - Changed `get_database()` to use `g.db` instead of global `db`
  - Added `@app.teardown_appcontext` decorator with `teardown_db()` function to close connections after each request
  - Added `from flask import g` import

## Testing

### Integration Tests Created

Created comprehensive integration test suite in `tests/test_web_integration.py`:
- 17 tests covering:
  - Server startup and page rendering
  - Static file serving (JS, CSS)
  - API endpoints (stats, search, paper detail, chat)
  - Input validation
  - Error handling (404, 500)
  - Concurrent requests (thread-safety)
  - Content-type and CORS headers
  - CLI command existence and module import

### Test Results

Before fixes:
- 10 passing, 7 failing
- Failures: Database not connected, SQL errors, thread-safety issues

After fixes:
- **All 17 integration tests passing**
- Full test suite: **182 passed, 3 skipped**

### Bug Discovery Process

1. Integration tests revealed "Not connected to database" errors
2. Fixed by adding `db.connect()` call in `get_database()`
3. Revealed "no such column: year" SQL error
4. Fixed by removing year statistics from stats endpoint
5. Revealed SQLite thread-safety errors in concurrent requests
6. Fixed by using Flask's `g` object for thread-local storage

## Technical Details

### Flask `g` Object

Flask's `g` object is a namespace object that can store data during an application context. It's perfect for storing database connections because:
- Each request gets its own `g` namespace
- Data stored in `g` is isolated per thread
- Automatically cleaned up after request
- Works with `@app.teardown_appcontext` for cleanup

### Database Connection Lifecycle

```python
# Before (BROKEN - shared across threads):
db = None  # Global variable

def get_database():
    global db
    if db is None:
        db = DatabaseManager(path)
        db.connect()
    return db

# After (FIXED - thread-local):
def get_database():
    if 'db' not in g:
        g.db = DatabaseManager(path)
        g.db.connect()
    return g.db

@app.teardown_appcontext
def teardown_db(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()
```

## Impact

- ✅ Web UI now works correctly with concurrent requests
- ✅ Database path can be configured via environment variable
- ✅ Stats endpoint returns correct data
- ✅ No more SQL errors or thread-safety issues
- ✅ Comprehensive integration test coverage ensures stability

## Lessons Learned

1. **Import-time initialization is problematic** - Configuration should be loaded lazily to allow environment overrides
2. **Integration tests catch production issues** - Unit tests with mocks wouldn't have caught the thread-safety issue
3. **SQLite threading restrictions** - Always use thread-local storage for SQLite connections in multi-threaded environments
4. **Schema changes require full review** - Removing the year column from DB required updating all API endpoints

## Related Changes

- Changelog 31: Web search bug fix (year filter in UI)
- Changelog 32: Web UI integration as package module
- This change (33): Critical bug fixes for production use

## Verification

To verify the fixes:

```bash
# Run integration tests
venv/bin/python -m pytest tests/test_web_integration.py -v

# Start web server and test manually
neurips-abstracts web-ui --host 127.0.0.1 --port 5000

# Test concurrent requests
curl http://localhost:5000/api/stats &
curl http://localhost:5000/api/stats &
curl http://localhost:5000/api/stats &
wait
```

## Files Modified

1. `src/neurips_abstracts/web_ui/app.py`:
   - Changed config loading to be lazy
   - Simplified stats endpoint (removed year statistics)
   - Changed database connection to use Flask `g` (thread-local)
   - Added teardown handler for database cleanup

2. `tests/test_web_integration.py`:
   - Cleaned up debug print statements
   - Simplified server startup function
   - Removed config reload workarounds (no longer needed)

## Next Steps

- Consider adding request-level connection pooling for better performance
- Add monitoring/logging for database connection lifecycle
- Consider caching stats results to reduce database queries
- Document thread-safety requirements for future contributors

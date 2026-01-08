# Fix: Web Integration Test Timeouts with OpenAI API Mocking

## Summary

Fixed timeout issues in `test_web_integration.py` where semantic search tests were failing with `ReadTimeout` errors (10 seconds) when attempting to generate embeddings via the OpenAI API. Also improved cross-platform compatibility by using werkzeug's `make_server` instead of Flask's `app.run()`.

## Problem

The web integration tests were failing with timeout errors on all platforms, with additional issues on Windows:

**All Platforms:**
```
requests.exceptions.ReadTimeout: HTTPConnectionPool(host='127.0.0.1', port=...): Read timed out. (read timeout=10)
```

**Windows-Specific:**
Flask's development server (`app.run()`) has reliability issues when run in daemon threads on Windows, causing tests to timeout even with proper mocking.

The root cause was:
1. The `web_server` fixture started Flask in a **separate process** (multiprocessing)
2. Semantic search tests triggered API calls to generate embeddings
3. Without a running LM Studio/blablador backend, these calls timed out
4. Mocks don't work across process boundaries, so mocking in the parent process had no effect

## Changes

### Phase 1: Switch to Threading with Mocking

**tests/test_web_integration.py**
- Changed `web_server` fixture from multiprocessing to threading (similar to `test_web_e2e.py`)
- Added OpenAI client mocking to prevent actual API calls during tests
- Set environment variables before importing Flask app to ensure config is loaded correctly
- Added proper ChromaDB setup with mock embeddings
- Pre-populated embeddings collection with test data
- Fixed test assertions to check for correct field names (`uid`, `title` instead of `id`, `name`)
- Removed `start_web_server` helper function (no longer needed with threading)
- Added `MOCK_EMBEDDING_DIMENSION = 4096` constant for consistency

### Phase 2: Improve Cross-Platform Compatibility

**tests/test_web_integration.py & tests/test_web_e2e.py**
- Replaced Flask's `app.run()` with werkzeug's `make_server()` for better threading support
- Added proper server shutdown in cleanup using `server.shutdown()`
- Enabled `threaded=True` option for concurrent request handling
- Improved error handling in server startup wait loop

## Solution Details

### Before (multiprocessing approach - didn't work)
```python
# Started server in separate process
server_process = Process(target=start_web_server, args=(abs_db_path, port), daemon=True)
server_process.start()
# Mocks in parent process don't affect child process
```

### After Phase 1 (threading with mocking)
```python
# Mock OpenAI API before importing Flask app
mock_openai_patcher = patch("neurips_abstracts.embeddings.OpenAI")
mock_openai_class = mock_openai_patcher.start()

# Configure mock to return fake embeddings
mock_client.embeddings.create.return_value = mock_embedding_response

# Start server in thread (same process, mocks work)
server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()
```

### After Phase 2 (werkzeug for reliability)
```python
# Use werkzeug's make_server for better cross-platform compatibility
from werkzeug.serving import make_server

server = make_server(host, port, flask_app, threaded=True)

def run_server():
    server.serve_forever()

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

# Later in cleanup
server.shutdown()
```

## Impact

**Fixed Tests** (all 5 previously failing tests now pass):
- `test_semantic_search_embeddings_manager_init`
- `test_semantic_search_end_to_end`
- `test_search_comparison_keyword_vs_semantic`
- `test_semantic_search_with_multiple_results`
- `test_search_semantic_exception`

**Result**: 
- All 31 tests in `test_web_integration.py` now pass (5.5-7 seconds runtime)
- No timeout errors on Linux
- Should fix Windows timeout issues (werkzeug's `make_server` is more reliable than Flask's `app.run()` in threads)
- Tests are now backend-agnostic (don't require running LM Studio or blablador)

## Testing

```bash
# Run the fixed tests
uv run pytest tests/test_web_integration.py -v

# Result: 31 passed, 3 deselected (6-7 seconds)
```

## Technical Notes

**Why werkzeug?**
- `make_server()` is designed for testing scenarios
- Better thread safety than Flask's development server
- More reliable across Windows, macOS, and Linux
- Proper shutdown support with `server.shutdown()`

**Why threading instead of multiprocessing?**
- Mocks work in same process (shared memory)
- Faster test startup (no process fork overhead)
- Easier cleanup (no orphan processes)
- Better compatibility with test frameworks

## Related Issues

- Addresses comment feedback about ReadTimeout errors in test_web_integration.py on Windows
- Makes tests backend-agnostic (no longer requires running LM Studio or blablador)
- Aligns with the approach used in `test_web_e2e.py` for consistency
- Improves test reliability across all platforms

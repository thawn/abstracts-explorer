# Fix: Web Integration Test Timeouts with OpenAI API Mocking

## Summary

Fixed timeout issues in `test_web_integration.py` where semantic search tests were failing with `ReadTimeout` errors (10 seconds) when attempting to generate embeddings via the OpenAI API.

## Problem

The web integration tests were failing with timeout errors:
```
requests.exceptions.ReadTimeout: HTTPConnectionPool(host='127.0.0.1', port=...): Read timed out. (read timeout=10)
```

The root cause was:
1. The `web_server` fixture started Flask in a **separate process** (multiprocessing)
2. Semantic search tests triggered API calls to generate embeddings
3. Without a running LM Studio/blablador backend, these calls timed out
4. Mocks don't work across process boundaries, so mocking in the parent process had no effect

## Changes

**tests/test_web_integration.py**
- Changed `web_server` fixture from multiprocessing to threading (similar to `test_web_e2e.py`)
- Added OpenAI client mocking to prevent actual API calls during tests
- Set environment variables before importing Flask app to ensure config is loaded correctly
- Added proper ChromaDB setup with mock embeddings
- Pre-populated embeddings collection with test data
- Fixed test assertions to check for correct field names (`uid`, `title` instead of `id`, `name`)
- Removed `start_web_server` helper function (no longer needed with threading)
- Added `MOCK_EMBEDDING_DIMENSION = 4096` constant for consistency

## Solution Details

### Before
```python
# Started server in separate process
server_process = Process(target=start_web_server, args=(abs_db_path, port), daemon=True)
server_process.start()
# Mocks in parent process don't affect child process
```

### After
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

## Impact

**Fixed Tests** (all 5 previously failing tests now pass):
- `test_semantic_search_embeddings_manager_init`
- `test_semantic_search_end_to_end`
- `test_search_comparison_keyword_vs_semantic`
- `test_semantic_search_with_multiple_results`
- `test_search_semantic_exception`

**Result**: All 31 tests in `test_web_integration.py` now pass without timeouts or external API dependencies.

## Testing

```bash
# Run the fixed tests
uv run pytest tests/test_web_integration.py -v

# Result: 31 passed, 3 deselected
```

## Related Issues

- Addresses comment feedback about ReadTimeout errors in test_web_integration.py
- Makes tests backend-agnostic (no longer requires running LM Studio or blablador)
- Aligns with the approach used in `test_web_e2e.py` for consistency

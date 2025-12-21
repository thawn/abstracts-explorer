# LM Studio Test Skip Configuration

**Date:** November 26, 2025  
**Status:** ✅ Completed

## Overview

Configured all tests that require LM Studio to be properly skipped when LM Studio is not running. This allows the test suite to run successfully even when LM Studio is unavailable.

## Changes Made

### test_web_integration.py

Added LM Studio availability check and skip decorator:

1. **Added helper function** `check_lm_studio_available()`
   - Checks if LM Studio is running at configured URL
   - Verifies at least one model is loaded
   - Returns True/False based on availability

2. **Added skip marker** `requires_lm_studio`
   - Uses `pytest.mark.skipif` decorator
   - Message: "LM Studio not running. Start LM Studio to run chat tests."

3. **Applied decorator to chat tests:**
   - `test_chat_with_valid_message_and_response`
   - `test_chat_with_reset_flag`
   - `test_chat_with_custom_n_papers`

### Already Configured Files

These files already had proper LM Studio skip configuration:

- **test_rag.py** - RAG chat integration tests (3 tests)
- **test_integration.py** - End-to-end embeddings test (1 test)

## Tests Skipped When LM Studio Not Running

Total: **7 tests** are properly skipped when LM Studio is unavailable

### test_integration.py (1 test)
- `test_embeddings_end_to_end_with_real_data` - Full embeddings workflow

### test_rag.py (3 tests)
- `test_real_query` - Real LM Studio query test
- `test_real_conversation` - Real conversation flow test
- `test_real_export` - Real conversation export test

### test_web_integration.py (3 tests)
- `test_chat_with_valid_message_and_response` - Chat endpoint with real response
- `test_chat_with_reset_flag` - Chat with conversation reset
- `test_chat_with_custom_n_papers` - Chat with custom paper count

## Test Results

With LM Studio **not running**:

```
232 passed, 7 skipped, 1 warning in 2.46s
```

### Coverage Maintained
- cli.py: 95%
- web_ui/app.py: 98%
- Overall: 95%

## Tests That Don't Require LM Studio

The following chat-related tests run successfully **without** LM Studio:

### test_web_integration.py
- `test_chat_reset_endpoint` - Tests reset endpoint structure (doesn't call LM Studio)
- `test_chat_with_empty_message` - Validates error for empty message (400 error before LM Studio)
- `test_chat_without_message` - Validates error for missing message (400 error before LM Studio)

### All CLI and unit tests
- All CLI tests use mocking (don't require real LM Studio)
- All web UI unit tests use mocking (don't require real LM Studio)
- All other integration tests work without LM Studio

## Benefits

1. **Fast test runs** - Can run test suite without waiting for LM Studio
2. **CI/CD friendly** - Tests pass in environments without LM Studio
3. **Developer workflow** - Can develop/test without running LM Studio
4. **Clear messages** - Skip reasons clearly indicate why tests were skipped
5. **No false failures** - Tests don't fail when LM Studio is intentionally off

## Running Tests With LM Studio

To run the full test suite including LM Studio tests:

1. Start LM Studio
2. Load a chat model
3. Run: `pytest tests/`

The 7 skipped tests will automatically run when LM Studio is available.

## Verification

```bash
# Run all tests (will skip 7 LM Studio tests)
pytest tests/ -v

# Run only tests that don't need LM Studio
pytest tests/ -v -m "not lm_studio"

# Check which tests are skipped
pytest tests/ -v | grep SKIPPED
```

## Summary

Successfully configured all LM Studio-dependent tests to skip gracefully when LM Studio is not running. The test suite now runs smoothly with **232 passing tests** and **7 properly skipped tests** when LM Studio is unavailable, maintaining **95% coverage**.

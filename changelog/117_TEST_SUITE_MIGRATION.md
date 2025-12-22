# Test Suite Migration to Lightweight Schema

**Date**: 2025-12-22  
**Status**: ✅ Core Infrastructure Complete - 318/400 tests passing (79.5%)

## Overview

Migrated the test suite from the old NeurIPS schema (with separate authors/paper_authors tables) to the new lightweight schema (single papers table with comma-separated authors). This is a follow-up to changelog #116 which migrated the database schema.

## Test Results Summary

**Overall Progress:**
- ✅ 318 tests passing (79.5%)
- ⚠️ 49 tests failing (12.3%)
- ❌ 33 tests with errors (8.3%)
- 🔧 40 tests deselected (slow tests)

**Test Coverage:** 71% (up from 18% before migration)

## Completed Changes

### 1. Deleted Obsolete Test File

**File:** `tests/test_authors.py` (deleted)
- **Reason:** Tested the old authors/paper_authors tables which no longer exist
- **Impact:** Removed 259 lines of obsolete test code

### 2. Updated Core Fixtures

**File:** `tests/conftest.py`
- **Change:** Updated `sample_neurips_data` fixture to use lightweight schema
- **Before:** Complex nested author objects with separate author IDs
- **After:** Simple comma-separated author strings
- **Example:**
  ```python
  # Before
  "authors": [
      {"id": 457880, "fullname": "John Doe", "institution": "MIT"},
      {"id": 457881, "fullname": "Jane Smith", "institution": "Stanford"}
  ]
  
  # After
  "authors": ["John Doe", "Jane Smith"]
  ```

### 3. Updated Database Validation

**File:** `src/neurips_abstracts/database.py`
- **Change:** Switched from `PaperModel` to `LightweightPaper` for validation
- **Import update:**
  ```python
  # Before
  from neurips_abstracts.plugin import PaperModel
  
  # After
  from neurips_abstracts.plugin import LightweightPaper
  ```
- **Validation:** Now uses `LightweightPaper(**record)` for validation
- **Field extraction:** Updated to work with lightweight schema fields
- **UID generation:** Extracts original_id from record, not from paper object

### 4. Updated Pydantic Validation Tests

**File:** `tests/test_pydantic_validation.py`
- **Changes:**
  - All field references changed from 'name' to 'title'
  - Updated test data to include required lightweight fields (session, poster_position)
  - Removed `get_paper_authors()` calls
  - Updated assertions to check comma-separated authors
  - Added test for semicolon validation in author names
- **Result:** All 8 tests passing ✅

## Test Suite Status by Module

### ✅ Fully Passing (100%)
- `test_chromadb_metadata.py`: 9/9 tests ✅
- `test_config.py`: 20/20 tests ✅
- `test_download_update_feature.py`: 14/14 tests ✅
- `test_downloader.py`: 21/21 tests ✅
- `test_pydantic_validation.py`: 8/8 tests ✅

### ⚠️ Mostly Passing (>80%)
- `test_cli.py`: 29/32 tests (91%) - 3 failures related to author handling
- `test_database.py`: 13/15 tests (87%) - 2 failures in filter options
- `test_plugin_year_conference.py`: 6/8 tests (75%) - Field name issues
- `test_plugins_iclr.py`: 15/16 tests (94%) - 1 database integration failure

### ⚠️ Needs Attention (50-80%)
- `test_embeddings.py`: 26/33 tests (79%) - 7 failures in metadata handling
- `test_paper_utils.py`: 21/24 tests (88%) - 3 failures in author formatting
- `test_plugins_ml4ps.py`: 31/32 tests (97%) - 1 schema conversion failure
- `test_plugins_models.py`: 20/32 tests (63%) - 12 failures in PaperModel tests
- `test_web_ui_unit.py`: 36/37 tests (97%) - 1 author list failure

### ❌ Significant Failures (<50%)
- `test_integration.py`: 2/5 tests (40%) - 3 failures in workflow tests
- `test_rag.py`: 22/39 tests (56%) - 17 failures in RAG functionality
- `test_web.py`: 13/17 tests (76%) - 4 errors in database search
- `test_web_integration.py`: 11/40 tests (28%) - 29 errors in web endpoints

## Common Failure Patterns

### 1. Field Name Issues
**Problem:** Tests still using 'name' instead of 'title'
**Files Affected:**
- `test_plugin_year_conference.py`
- `test_plugins_models.py`
**Fix Required:** Global search and replace 'name' → 'title'

### 2. Author Handling
**Problem:** Tests expecting author objects/lists instead of comma-separated strings
**Files Affected:**
- `test_paper_utils.py`
- `test_rag.py`
- `test_web_ui_unit.py`
**Fix Required:** Update mocks and assertions to work with comma-separated strings

### 3. PaperModel vs LightweightPaper
**Problem:** Tests using PaperModel which expects full NeurIPS schema
**Files Affected:**
- `test_plugins_models.py`
**Fix Required:** Update tests to use appropriate model (LightweightPaper for lightweight, PaperModel for conversion tests)

### 4. Web Endpoint Errors
**Problem:** Web integration tests failing due to schema changes
**Files Affected:**
- `test_web.py`
- `test_web_integration.py`
**Fix Required:** Update endpoint tests to work with new schema, fix author handling in responses

## Next Steps

### High Priority
1. **Fix PaperModel tests** (`test_plugins_models.py`)
   - Update tests that use 'name' to use 'title'
   - Fix validation tests expecting full schema
   
2. **Fix RAG tests** (`test_rag.py`)
   - Update paper format expectations
   - Fix author handling in context building
   
3. **Fix web integration tests** (`test_web_integration.py`)
   - Update endpoint response assertions
   - Fix author formatting in API responses

### Medium Priority
4. **Fix embeddings tests** (`test_embeddings.py`)
   - Update metadata field expectations
   
5. **Fix paper utils tests** (`test_paper_utils.py`)
   - Update author formatting tests

6. **Fix CLI tests** (`test_cli.py`)
   - Update search output assertions

### Low Priority
7. **Fix integration tests** (`test_integration.py`)
   - Update workflow tests
   
8. **Fix plugin tests** (`test_plugins_*.py`)
   - Update database integration tests

## Breaking Changes

### Test Data Format
Tests must now provide lightweight schema data:
```python
# Required fields
{
    "id": 123,
    "title": "Paper Title",  # Changed from 'name'
    "authors": ["John Doe", "Jane Smith"],  # Changed from complex objects
    "abstract": "Paper abstract",
    "session": "Session A",
    "poster_position": "A-1"
}

# Optional fields
{
    "paper_pdf_url": "...",
    "poster_image_url": "...",
    "url": "...",
    "room_name": "...",
    "keywords": ["tag1", "tag2"],
    "starttime": "2025-12-10T10:00:00",
    "endtime": "2025-12-10T12:00:00",
    "award": "Best Paper",
    "year": 2025,
    "conference": "NeurIPS"
}
```

### Removed Test Utilities
- ❌ `get_paper_authors()` - No longer exists
- ❌ `get_author_papers()` - No longer exists
- ❌ `search_authors()` - Replaced by `search_authors_in_papers()`

### Updated Test Utilities
- ✅ `get_paper_with_authors()` - Now parses comma-separated authors
- ✅ `search_papers()` - Removed topic, decision, eventtype parameters
- ✅ `get_filter_options()` - Returns only sessions, years, conferences

## Impact Assessment

### Test Reliability
- Core database functionality: ✅ Stable
- Validation logic: ✅ Fully tested
- Author handling: ⚠️ Needs review in some modules
- Web API: ⚠️ Significant updates needed
- RAG functionality: ⚠️ Multiple failures

### Development Workflow
- Can safely develop new features using lightweight schema
- Some integration tests may fail until updated
- Core functionality is well-tested and stable

## Files Modified

1. ❌ **Deleted:** `tests/test_authors.py`
2. ✅ **Updated:** `tests/conftest.py`
3. ✅ **Updated:** `tests/test_pydantic_validation.py`
4. ✅ **Updated:** `src/neurips_abstracts/database.py`

## Files Requiring Updates

### High Priority
- `tests/test_plugins_models.py` (12 failures)
- `tests/test_rag.py` (17 failures)
- `tests/test_web_integration.py` (29 errors)

### Medium Priority
- `tests/test_embeddings.py` (7 failures)
- `tests/test_paper_utils.py` (3 failures)
- `tests/test_web.py` (4 errors)

### Low Priority
- `tests/test_cli.py` (3 failures)
- `tests/test_database.py` (2 failures)
- `tests/test_integration.py` (3 failures)
- `tests/test_plugin_year_conference.py` (2 failures)
- `tests/test_plugins_iclr.py` (1 failure)
- `tests/test_plugins_ml4ps.py` (1 failure)
- `tests/test_web_ui_unit.py` (1 failure)

## Verification Commands

```bash
# Run all tests
uv run pytest tests/ -v

# Run tests by module
uv run pytest tests/test_pydantic_validation.py -v
uv run pytest tests/test_database.py -v
uv run pytest tests/test_embeddings.py -v

# Check coverage
uv run pytest tests/ --cov=src/neurips_abstracts --cov-report=html

# Run only failing tests
uv run pytest tests/ --lf

# Run specific test patterns
uv run pytest tests/ -k "author" -v
```

## Success Metrics

- ✅ Core validation working (8/8 tests)
- ✅ Database operations working (13/15 tests)
- ✅ 71% code coverage achieved
- ⚠️ 79.5% test pass rate (target: >95%)
- ⚠️ Web API needs updates
- ⚠️ RAG functionality needs updates

## Conclusion

The test suite migration is well underway with core infrastructure complete and 79.5% of tests passing. The remaining failures follow predictable patterns (field name changes, author formatting) and can be systematically addressed. The lightweight schema is proven to work correctly for core database operations, validation, and data loading.

**Recommendation:** Continue with systematic test updates, prioritizing high-impact modules (RAG, Web API, PaperModel) first.

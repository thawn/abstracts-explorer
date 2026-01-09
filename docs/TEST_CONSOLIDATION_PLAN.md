# Test Consolidation Plan

## Goal
Consolidate test files to follow the principle: **one test file per source module**.

## Current Status
✅ Documentation updated in:
- `.github/instructions/project.instructions.md`
- `docs/contributing.md`

## Files to Consolidate

### 1. Plugin Tests → test_plugin.py

Consolidate these files into `tests/test_plugin.py`:

- **test_plugin_helpers.py** (561 lines)
  - Tests for `sanitize_author_names()`
  - Tests for `convert_neurips_to_lightweight_schema()`
  - Tests for `validate_lightweight_paper()`
  - Tests for `prepare_chroma_db_paper_data()`

- **test_plugin_year_conference.py** (372 lines)
  - Tests for year/conference field handling in plugins
  - Tests for NeurIPS, ICML, ML4PS plugins

- **test_plugins_models.py** (263 lines)
  - Tests for Pydantic models in plugin system
  - Tests for `LightweightPaper` model

- **test_pydantic_validation.py** (160 lines)
  - Additional Pydantic validation tests for `LightweightPaper`

**Total:** ~1,356 lines to consolidate

**Organization in test_plugin.py:**
```python
"""
Tests for plugin.py module.

This module tests all plugin-related functionality including:
- Plugin helper functions (sanitize_author_names, convert_neurips_to_lightweight_schema)
- Plugin data models (LightweightPaper, validation)
- Plugin base classes and interfaces
- Year/conference field handling
"""

# Test classes organized by functionality:
class TestPluginHelpers:
    """Tests for plugin helper functions."""
    class TestSanitizeAuthorNames: ...
    class TestConvertNeuripsToLightweightSchema: ...
    class TestValidateLightweightPaper: ...
    class TestPrepareChromaDbPaperData: ...

class TestPluginModels:
    """Tests for plugin Pydantic models."""
    class TestLightweightPaperValidation: ...
    class TestLightweightPaperFields: ...

class TestPluginYearConference:
    """Tests for year/conference field handling in plugins."""
    class TestNeurIPSPluginYearConference: ...
    class TestICMLPluginYearConference: ...
    class TestML4PSPluginYearConference: ...
```

### 2. Web UI Tests → test_web_ui.py

Consolidate these files into `tests/test_web_ui.py`:

- **test_web.py** (340 lines)
  - Basic web UI unit tests
  - Tests for Flask app initialization
  - Tests for basic endpoints

- **test_web_ui_unit.py** (571 lines)
  - Detailed unit tests for web_ui/app.py
  - Tests for semantic search functionality
  - Tests for chat endpoints
  - Tests for error handling

**Total:** ~911 lines to consolidate

**Keep separate:**
- `test_web_integration.py` (975 lines) - Integration tests with real server
- `test_web_e2e.py` (1,313 lines) - End-to-end browser tests

**Organization in test_web_ui.py:**
```python
"""
Tests for web_ui/app.py module.

This module contains unit tests for the web UI application including:
- Flask app configuration
- API endpoints (search, chat, stats)
- Semantic search functionality
- Error handling
- Response formatting
"""

# Test classes organized by functionality:
class TestWebUIApp:
    """Tests for Flask app initialization and configuration."""

class TestWebUIEndpoints:
    """Tests for API endpoints."""
    class TestSearchEndpoint: ...
    class TestChatEndpoint: ...
    class TestStatsEndpoint: ...
    class TestPaperDetailEndpoint: ...

class TestWebUISemanticSearch:
    """Tests for semantic search functionality."""

class TestWebUIErrorHandling:
    """Tests for error handling and edge cases."""
```

### 3. ChromaDB Tests → test_embeddings.py

Move `test_chromadb_metadata.py` (455 lines) into `test_embeddings.py`:

**Rationale:** ChromaDB functionality is part of the embeddings module. The metadata tests verify that embeddings are stored correctly with metadata.

**Organization:**
Add new test classes to existing `test_embeddings.py`:
```python
class TestChromaDBMetadata:
    """Tests for ChromaDB metadata functionality."""
    # Tests for metadata filtering
    # Tests for metadata storage
    # Tests for semantic search with metadata filters
```

## Files to Keep Separate

### Plugin Implementation Tests
- `test_plugins_iclr.py` - Tests for `plugins/iclr_downloader.py`
- `test_plugins_ml4ps.py` - Tests for `plugins/ml4ps_downloader.py`

**Rationale:** These test specific plugin implementations in the plugins/ subdirectory, not the base plugin.py module.

### Integration and E2E Tests
- `test_integration.py` - Cross-module integration tests
- `test_web_integration.py` - Web UI integration tests with real server
- `test_web_e2e.py` - End-to-end browser automation tests

**Rationale:** These are different types of tests that test multiple modules together or require special setup (browsers, servers).

## Implementation Steps

1. **Create test_plugin.py**
   - Copy test classes from source files
   - Organize into logical sections with clear headers
   - Ensure all imports are included
   - Run tests to verify: `uv run pytest tests/test_plugin.py -v`

2. **Update test_web_ui.py**
   - Merge test_web.py and test_web_ui_unit.py
   - Remove duplicates (if any)
   - Organize by functionality
   - Run tests to verify: `uv run pytest tests/test_web_ui.py -v`

3. **Extend test_embeddings.py**
   - Add ChromaDB metadata tests
   - Keep existing structure
   - Run tests to verify: `uv run pytest tests/test_embeddings.py -v`

4. **Remove old files**
   - Delete consolidated source files
   - Update any references in documentation
   - Run full test suite: `uv run pytest tests/ -v`

5. **Update conftest.py**
   - Check if any fixtures need to be moved/updated
   - Ensure no broken fixture references

## Testing Strategy

After each consolidation:
1. Run the new consolidated test file
2. Verify all tests pass
3. Check coverage hasn't decreased
4. Run full test suite to catch any issues

## Rollback Plan

If issues arise:
- Each file is consolidated separately with its own commit
- Can revert individual commits if needed
- Git history preserves all test code

## Notes

- Total lines to consolidate: ~2,700 lines across 7 files
- This is a large refactoring but follows clear principles
- No test logic changes, only file organization
- All tests should continue to pass

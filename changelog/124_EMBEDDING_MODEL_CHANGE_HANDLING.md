# Changelog Entry 124: Embedding Model Change Handling

**Date**: 2026-01-05

## Summary

Implemented comprehensive handling of embedding model changes to prevent compatibility issues when users switch between different embedding models. The system now tracks which model was used to create embeddings, warns users about incompatibilities, and provides clear guidance for resolving model mismatches.

## Changes Made

### 1. Database Schema Updates

**File**: `src/neurips_abstracts/database.py`

- **Added `embeddings_metadata` table**: Created a new table to store embedding model information
  - Fields: `id` (auto-increment primary key), `embedding_model` (model name), `created_at`, `updated_at`
  - Tracks which embedding model was used to generate the current embeddings

- **Added `get_embedding_model()` method**: Retrieves the stored embedding model from the database
  - Returns `None` if no model has been stored yet (first-time setup)
  - Returns the most recent embedding model name

- **Added `set_embedding_model()` method**: Stores or updates the embedding model in the database
  - Updates existing record if one exists
  - Creates new record on first call
  - Updates timestamp automatically

### 2. EmbeddingsManager Enhancements

**File**: `src/neurips_abstracts/embeddings.py`

- **Added `check_model_compatibility()` method**: Checks if current model matches stored model
  - Returns tuple: `(compatible, stored_model, current_model)`
  - Considers compatible if: database doesn't exist, no model stored, or models match
  - Returns `False` if models differ

- **Updated `embed_from_database()` method**:
  - Now automatically stores the embedding model when creating embeddings
  - Added `force_recreate` parameter to skip existence checks and recreate all embeddings
  - Uses `DatabaseManager` for all database operations (removed direct SQLite usage)

### 3. CLI Command Improvements

**File**: `src/neurips_abstracts/cli.py`

- **Updated `create-embeddings` command**:
  - Checks for model compatibility before generating embeddings
  - Warns user if stored model differs from configured model
  - Provides clear error messages with stored vs. current model names
  - Interactive prompt asks user for confirmation when mismatch detected
  - `--force` flag bypasses prompt and automatically recreates embeddings
  - Updates help text to explain force flag behavior

### 4. Web UI Additions

**Files**: 
- `src/neurips_abstracts/web_ui/app.py`
- `src/neurips_abstracts/web_ui/templates/index.html`
- `src/neurips_abstracts/web_ui/static/app.js`

- **Added `/api/embedding-model-check` endpoint**:
  - Returns compatibility status and model information
  - Provides detailed warning message when models don't match
  - Includes instructions for recreating embeddings

- **Added warning banner in UI**:
  - Displays at top of page when model mismatch detected
  - Shows both current and stored model names
  - Dismissible by user
  - Prominently styled with yellow warning colors

- **Added JavaScript compatibility check**:
  - Automatically checks compatibility on page load
  - Fetches status from API endpoint
  - Shows/hides warning banner based on results

### 5. Test Coverage

**Files**: `tests/test_database.py`, `tests/test_embeddings.py`

- **Added `TestEmbeddingModelMetadata` test class**:
  - Tests getting model when none is set
  - Tests setting and retrieving model
  - Tests updating model
  - Tests model persistence across connections

- **Added embedding compatibility tests**:
  - Tests compatibility check with non-existent database
  - Tests compatibility check with no stored model
  - Tests compatibility check with matching models
  - Tests compatibility check with mismatched models
  - Tests that `embed_from_database()` stores the model

- **Fixed `test_database` fixture**:
  - Updated to use `DatabaseManager` instead of raw SQLite
  - Ensures proper schema including `embeddings_metadata` table
  - Uses `LightweightPaper` models for test data

## API Changes

### DatabaseManager

```python
# New methods
def get_embedding_model() -> Optional[str]:
    """Get the stored embedding model name."""
    
def set_embedding_model(model_name: str) -> None:
    """Store the embedding model name."""
```

### EmbeddingsManager

```python
# New method
def check_model_compatibility(db_path: Union[str, Path]) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Check if current model matches stored model.
    Returns: (compatible, stored_model, current_model)
    """

# Updated method signature
def embed_from_database(
    db_path: Union[str, Path],
    where_clause: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    force_recreate: bool = False,  # New parameter
) -> int:
    """Now supports force_recreate to skip caching."""
```

### Web API

```
GET /api/embedding-model-check
```

Returns:
```json
{
  "compatible": false,
  "stored_model": "text-embedding-qwen3-embedding-4b",
  "current_model": "text-embedding-new-model",
  "warning": "Detailed warning message with instructions..."
}
```

## User Impact

### Breaking Changes

None. The changes are backward compatible. Existing databases without the `embeddings_metadata` table will have it created automatically on the next table creation.

### New Behavior

1. **First-time embedding creation**: Model is automatically stored in database
2. **Model change detection**: Users are warned when changing embedding models
3. **CLI interaction**: Users must confirm or use `--force` when model changes
4. **Web UI warning**: Prominent banner shows when embeddings need recreation

### Migration Path

For existing users:

1. **Existing embeddings without stored model**: Next embedding creation will store the current model
2. **Changing models**: 
   - CLI: Run `neurips-abstracts create-embeddings --force` to recreate
   - Web UI: Follow instructions in warning banner

## Benefits

1. **Prevents silent failures**: Users are warned before using incompatible embeddings
2. **Clear guidance**: Specific instructions provided for resolving mismatches
3. **Automated tracking**: Model is automatically stored, no manual intervention needed
4. **Multiple interfaces**: Warnings in both CLI and Web UI
5. **Safe by default**: Requires explicit confirmation to overwrite embeddings

## Testing

All tests pass:
- 4 database metadata tests
- 5 embedding compatibility tests
- Test fixture updated to ensure proper schema

Coverage improved for:
- `database.py`: 48% (up from 32%)
- `embeddings.py`: 49% (up from 15%)

## Future Enhancements

Potential future improvements:
1. Store embedding dimensions in metadata
2. Track creation timestamp for embeddings
3. Support gradual migration between models
4. Add CLI command to check compatibility without embedding
5. Add telemetry for model usage patterns

## Related Files

- `src/neurips_abstracts/database.py`
- `src/neurips_abstracts/embeddings.py`
- `src/neurips_abstracts/cli.py`
- `src/neurips_abstracts/web_ui/app.py`
- `src/neurips_abstracts/web_ui/templates/index.html`
- `src/neurips_abstracts/web_ui/static/app.js`
- `tests/test_database.py`
- `tests/test_embeddings.py`

## References

- Issue: "gracefully handle embedding model change"
- Requirements: Store model, CLI validation, frontend warning

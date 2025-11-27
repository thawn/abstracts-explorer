# Data Directory Configuration

**Date**: November 27, 2025

## Summary

Moved database files (`neurips_2025.db` and `chroma_db/`) into a configurable `data/` directory to better organize project files and make data storage location configurable.

## Changes Made

### 1. Configuration Updates

- Added new `DATA_DIR` configuration option (default: `data`)
- Updated `EMBEDDING_DB_PATH` and `PAPER_DB_PATH` to resolve relative to `DATA_DIR`
- Absolute paths continue to work unchanged
- Added `_resolve_path()` method to handle path resolution logic

### 2. File Organization

- Moved `neurips_2025.db` → `data/neurips_2025.db`
- Moved `chroma_db/` → `data/chroma_db/`
- Updated `.gitignore` to reflect new structure

### 3. Documentation Updates

- Updated `README.md` to document new `DATA_DIR` setting
- Updated `docs/configuration.md` with detailed path resolution information
- Updated `src/neurips_abstracts/web_ui/README.md`
- Updated `.env.example` with `DATA_DIR` configuration

### 4. Test Updates

- Updated all configuration tests to account for path resolution
- Added tests for `DATA_DIR` configuration
- Added tests for absolute path handling
- All 20 config tests pass

## Configuration Example

```bash
# .env file
DATA_DIR=data  # Base directory for all data files

# These paths are resolved relative to DATA_DIR
EMBEDDING_DB_PATH=chroma_db      # Resolves to: data/chroma_db
PAPER_DB_PATH=neurips_2025.db    # Resolves to: data/neurips_2025.db

# Absolute paths work too:
# PAPER_DB_PATH=/absolute/path/to/neurips_2025.db
```

## Benefits

1. **Better Organization**: All data files in one place
2. **Configurable**: Easy to change data directory location
3. **Backward Compatible**: Existing absolute paths continue to work
4. **Cleaner Root**: Keeps project root directory cleaner
5. **Flexible**: Can use relative or absolute paths

## Migration

For existing users:

1. Move your database files to the `data/` directory:
   ```bash
   mkdir -p data
   mv neurips_2025.db data/
   mv chroma_db data/
   ```

2. Configuration will automatically work with defaults, or you can customize:
   ```bash
   # In .env
   DATA_DIR=data  # This is the default
   ```

## Testing

All tests pass:
- 20 configuration tests
- Tests verify default path resolution
- Tests verify absolute path handling
- Tests verify custom DATA_DIR configuration

## Files Modified

- `src/neurips_abstracts/config.py` - Added DATA_DIR and path resolution
- `tests/test_config.py` - Updated tests for new behavior
- `.env.example` - Added DATA_DIR setting
- `.gitignore` - Cleaned up redundant entries
- `README.md` - Updated documentation
- `docs/configuration.md` - Updated documentation
- `src/neurips_abstracts/web_ui/README.md` - Updated documentation

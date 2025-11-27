# Data Directory Migration Summary

**Date**: November 27, 2025  
**Status**: ✅ Complete

## What Changed

The `neurips_2025.db` database and `chroma_db/` directory have been moved into a configurable `data/` directory for better organization.

### Before

```
neurips-abstracts/
├── neurips_2025.db      # In project root
├── chroma_db/           # In project root
├── src/
├── tests/
└── ...
```

### After

```
neurips-abstracts/
├── data/
│   ├── neurips_2025.db   # Moved here
│   ├── chroma_db/        # Moved here
│   └── README.md         # Documentation
├── src/
├── tests/
└── ...
```

## Configuration

New `DATA_DIR` environment variable controls the base directory for data files:

```bash
# .env
DATA_DIR=data  # Default value
```

Paths are automatically resolved:
- `EMBEDDING_DB_PATH=chroma_db` → `data/chroma_db`
- `PAPER_DB_PATH=neurips_2025.db` → `data/neurips_2025.db`

Absolute paths continue to work unchanged:
- `PAPER_DB_PATH=/absolute/path/to/db.db` → `/absolute/path/to/db.db`

## Files Updated

### Core
- ✅ `src/neurips_abstracts/config.py` - Added DATA_DIR and path resolution
- ✅ `tests/test_config.py` - Updated tests (20/20 passing)

### Documentation
- ✅ `.env.example` - Added DATA_DIR setting
- ✅ `README.md` - Updated with DATA_DIR documentation
- ✅ `docs/configuration.md` - Detailed path resolution docs
- ✅ `src/neurips_abstracts/web_ui/README.md` - Updated web UI config
- ✅ `data/README.md` - New data directory documentation

### Examples
- ✅ `demo_search_with_authors.py` - Now uses config
- ✅ `examples/rag_chat_demo.py` - Now uses config
- ✅ `examples/embeddings_demo.py` - Now uses config
- ✅ `examples/add_missing_metadata.py` - Now uses config

### Git
- ✅ `.gitignore` - Cleaned up redundant entries
- ✅ `changelog/62_DATA_DIR_CONFIGURATION.md` - Detailed changelog

## Migration Steps (Already Done)

The following has been completed:

```bash
# 1. Moved files
mv neurips_2025.db data/
mv chroma_db data/

# 2. Verified configuration
python -c "from neurips_abstracts.config import get_config; c = get_config(); print(c.paper_db_path)"
# Output: data/neurips_2025.db

# 3. Ran tests
pytest tests/test_config.py -v
# Result: 20/20 tests passing

# 4. Verified demo script
python demo_search_with_authors.py
# Result: Works correctly with new paths
```

## Benefits

1. **Better Organization** - All data files in one place
2. **Configurable Location** - Easy to change where data is stored
3. **Cleaner Root** - Project root directory is less cluttered
4. **Backward Compatible** - Absolute paths still work
5. **Flexible** - Can use relative or absolute paths

## No Action Required

For existing users, everything should work automatically with the default settings. The configuration will resolve paths relative to the `data/` directory by default.

To customize:
```bash
# Change data directory location
DATA_DIR=/custom/location

# Or use absolute paths
PAPER_DB_PATH=/absolute/path/to/neurips_2025.db
EMBEDDING_DB_PATH=/absolute/path/to/chroma_db
```

## Testing

All configuration tests pass:
- ✅ 20/20 config tests passing
- ✅ Demo script works with new paths
- ✅ Path resolution works for relative and absolute paths
- ✅ Custom DATA_DIR configuration tested

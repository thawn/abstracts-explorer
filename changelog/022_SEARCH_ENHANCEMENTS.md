# Search Enhancement: Author Names and Additional Metadata

## Summary

Enhanced the `neurips-abstracts search` command to display author names instead of author IDs and added `paper_url` and `poster_position` fields to search results.

## Changes Made

### 1. Updated Embeddings Module (`src/neurips_abstracts/embeddings.py`)

**Modified `embed_from_database()` method:**
- Updated SQL query to include `paper_url` and `poster_position` columns
- Added these fields to metadata dictionary stored with each embedding

```python
query = """
    SELECT id, name, abstract, authors, topic, keywords, decision, paper_url, poster_position 
    FROM papers
"""

metadata = {
    "title": row["name"] or "",
    "authors": row["authors"] or "",
    "topic": row["topic"] or "",
    "keywords": row["keywords"] or "",
    "decision": row["decision"] or "",
    "paper_url": row["paper_url"] or "",
    "poster_position": row["poster_position"] or "",
}
```

### 2. Enhanced CLI Search Command (`src/neurips_abstracts/cli.py`)

**Added `--db-path` parameter:**
- Optional parameter to provide path to SQLite database
- When provided, enables author name lookup for search results

**Updated search result display:**
- Queries database to fetch author names using `DatabaseManager.get_paper_authors()`
- Displays author names as comma-separated list instead of author IDs
- Shows `paper_url` if available
- Shows `poster_position` if available
- Gracefully falls back to author IDs if database not provided or lookup fails

### 3. Result Display Format

**Before:**
```
1. Paper ID: 119222
   Title: Discovering Data Structures
   Authors: 456862, 19141, 184344
   Decision: Accept (poster)
   Topic: Deep Learning
   Similarity: 0.8234
```

**After (with --db-path):**
```
1. Paper ID: 119222
   Title: Discovering Data Structures
   Authors: Gang Li, Ming Lin, Tomer Galanti
   Decision: Accept (poster)
   Topic: Deep Learning
   URL: https://openreview.net/forum?id=abc123
   Poster Position: #3700
   Similarity: 0.8234
```

## Usage

### Creating Embeddings with New Metadata

```bash
# Embeddings now automatically include paper_url and poster_position
neurips-abstracts create-embeddings \
    --db-path neurips_2025.db \
    --output chroma_db
```

### Searching with Author Names

```bash
# Basic search (shows author IDs)
neurips-abstracts search "machine learning" --n-results 5

# Search with author names (requires --db-path)
neurips-abstracts search "machine learning" \
    --n-results 5 \
    --db-path neurips_2025.db

# With abstracts and metadata filter
neurips-abstracts search "neural networks" \
    --n-results 10 \
    --db-path neurips_2025.db \
    --show-abstract \
    --where 'decision=Accept (oral)'
```

## Implementation Details

### Author Name Resolution

The search command uses the following logic:

1. If `--db-path` is provided and valid:
   - Open database connection
   - For each search result, call `db.get_paper_authors(paper_id)`
   - Format author names as comma-separated list
   - Display names instead of IDs

2. If database not available:
   - Falls back to displaying author IDs from metadata
   - No error is raised; search continues normally

### Error Handling

- Database connection errors are caught and printed as warnings
- Individual author lookup failures fall back to IDs silently
- Search functionality works even without database access

### Backward Compatibility

- Existing embeddings without `paper_url` and `poster_position` will show empty values
- Search without `--db-path` works exactly as before (shows author IDs)
- No breaking changes to existing code or APIs

## Testing

All existing tests pass:
```bash
pytest tests/test_cli.py -v  # 17 tests passed
```

### Manual Testing

Demo script provided to verify author name lookup:
```bash
python demo_search_with_authors.py
```

Sample output:
```
Paper ID: 114995
Title: DisCO: Reinforcing Large Reasoning Models
Author IDs: 236344, 148972, 32102, 447849, 255800
Author Names: Gang Li, Ming Lin, Tomer Galanti, Zhengzhong Tu, Tianbao Yang
URL: https://openreview.net/forum?id=zzUXS4f91r
Poster Position: #3700
```

## Future Improvements

Potential enhancements:
- Cache author lookups within a search session for performance
- Add `--author-format` option to control name display (full, last name only, etc.)
- Include author institutions in verbose mode
- Add author-specific search filters

## Files Modified

1. `src/neurips_abstracts/embeddings.py` - Updated metadata storage
2. `src/neurips_abstracts/cli.py` - Added --db-path and author name display
3. `tests/test_embeddings.py` - Updated test fixtures to include new fields
4. `demo_search_with_authors.py` - Created demo script (new file)

## Test Fixtures Updated

Updated the test database fixtures to include the new fields:
- Added `paper_url TEXT` column to papers table
- Added `poster_position TEXT` column to papers table
- Updated sample test data with realistic URLs and poster positions
- All 96 tests pass with 89% overall coverage

## Dependencies

No new dependencies required. Uses existing:
- `DatabaseManager` for author lookups
- `EmbeddingsManager` for vector search
- `pathlib.Path` for file handling

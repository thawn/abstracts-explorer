# Database Schema Migration: NeurIPS to Lightweight ML4PS Schema

## Summary

Successfully migrated the neurips-abstracts database from a complex multi-table schema to a lightweight single-table schema inspired by ML4PS. The new schema simplifies the data model while maintaining all essential functionality.

## Key Changes

### 1. Database Schema (database.py)

#### Removed Tables
- **authors table**: Separate table for author information
- **paper_authors junction table**: Many-to-many relationship between papers and authors

#### Modified papers Table
- **Renamed column**: `name` → `title` (paper titles)
- **Added column**: `original_id` (stores original UID from scraped data)
- **Modified column**: `authors` (now stores comma-separated list of author names instead of IDs)
- **Modified column**: `uid` (now generated as SHA-256 hash from `title + original_id`)
- **Added indices**: On `title` and `original_id` for better query performance

### 2. Data Loading (database.py - load_json_data)

- Authors are now extracted and stored as comma-separated names directly in the papers table
- UID generation: `SHA256(title:original_id)[:16]`
- Removed all author table and junction table insertions
- Supports both `name` and `title` fields in input data for backward compatibility

### 3. Removed/Modified Methods

**Removed:**
- `get_paper_authors()` - No longer needed with comma-separated authors
- `get_author_papers()` - Cannot query by author ID anymore
- `search_authors()` - Authors table no longer exists

**Added:**
- `search_authors_in_papers()` - Search for authors within the comma-separated field
- `get_author_count()` - Estimates unique authors by parsing all papers' author fields

### 4. Updated Modules

#### paper_utils.py
- `get_paper_with_authors()`: Parses comma-separated authors string
- Ensures both `title` and `name` fields are available for backward compatibility
- `format_search_results()`: Uses updated `get_paper_with_authors()`

#### embeddings.py
- `embed_from_database()`: Supports both `title` and `name` column names
- Dynamically detects which column exists in the database

#### web_ui/app.py
- Search endpoint: Parses comma-separated authors instead of querying authors table
- Ensures both `title` and `name` are available in API responses

#### plugin.py
- `PaperModel`: Added optional `title` field alongside `name` for compatibility
- Updated documentation to reflect authors as comma-separated string

### 5. Migration Script

Created `scripts/migrate_to_lightweight_schema.py` that:
1. Creates automatic backup of original database
2. Creates new lightweight schema table
3. Migrates all data, converting authors from separate table to comma-separated list
4. Generates new UIDs based on title+original_id hash
5. Drops old tables and renames new table
6. Creates all necessary indices
7. Supports `--dry-run` mode for safe testing

Usage:
```bash
python scripts/migrate_to_lightweight_schema.py data/neurips_2025.db
python scripts/migrate_to_lightweight_schema.py data/neurips_2025.db --dry-run  # Test first
```

## Benefits of New Schema

1. **Simplicity**: Single table instead of three tables with complex joins
2. **Performance**: No joins needed to get authors with papers
3. **Portability**: Easier to export and share data
4. **Compatibility**: Aligns with ML4PS lightweight schema design
5. **Maintainability**: Less code to maintain and test

## Backward Compatibility

- Migration script preserves all original data in `original_id` column
- Automatic backup created before migration
- **Note**: The old `name` field is replaced with `title` - no backward compatibility maintained

## Tests Requiring Updates

The following test files need updates to work with the new schema:

### High Priority (Directly test old schema)
- `tests/test_authors.py` - Entire file tests old author tables
- `tests/test_pydantic_validation.py` - Uses `get_paper_authors()`
- `tests/test_paper_utils.py` - Mocks `get_paper_authors()`

### Medium Priority (Use author tables in fixtures)
- `tests/test_web_e2e.py` - Creates `paper_authors` table in fixtures
- `tests/test_cli.py` - Creates `paper_authors` table
- `tests/test_rag.py` - Creates `paper_authors` table and mocks `get_paper_authors()`
- `tests/test_integration.py` - Calls `get_paper_authors()`
- `tests/test_web.py` - References authors table
- `tests/test_plugins_iclr.py` - Queries paper_authors table

### Low Priority (May work with minimal changes)
- `tests/test_plugin_year_conference.py` - Uses `paper["name"]`
- `tests/test_plugins_models.py` - Tests PaperModel with `name` field
- `tests/test_database.py` - May need schema updates

## Test Update Strategy

For each test file:
1. Remove tests for `authors` and `paper_authors` tables
2. Update fixtures to use comma-separated authors string
3. Replace `get_paper_authors()` calls with string parsing
4. Update assertions to check comma-separated format
5. Replace `paper["name"]` with `paper["title"]` where appropriate

## Next Steps

1. **Update test files** (see list above)
2. **Run migration on existing databases**:
   ```bash
   python scripts/migrate_to_lightweight_schema.py data/neurips_2025.db
   ```
3. **Run full test suite** to verify all changes:
   ```bash
   pytest tests/ -v
   ```
4. **Update documentation** (README.md, API docs) to reflect new schema
5. **Create changelog entry** documenting this major change

## Example: Before and After

### Before (Complex Schema)
```sql
-- Three tables with joins
SELECT p.*, a.fullname 
FROM papers p
JOIN paper_authors pa ON p.id = pa.paper_id
JOIN authors a ON pa.author_id = a.id
WHERE p.id = 123;
```

### After (Lightweight Schema)
```sql
-- Single table, simple query
SELECT * FROM papers WHERE id = 123;
-- authors field contains: "John Doe, Jane Smith, Bob Johnson"
```

### Python Usage Before
```python
paper = db.query("SELECT * FROM papers WHERE id = ?", (paper_id,))[0]
authors = db.get_paper_authors(paper_id)
author_names = [a["fullname"] for a in authors]
title = paper["name"]
```

### Python Usage After
```python
paper = db.query("SELECT * FROM papers WHERE id = ?", (paper_id,))[0]
author_names = [a.strip() for a in paper["authors"].split(",")]
title = paper["title"]
```

## Migration Rollback

If issues arise, rollback is simple:
1. The migration script creates a timestamped backup automatically
2. To rollback: `mv neurips_2025.backup_TIMESTAMP.db neurips_2025.db`

## Files Modified

1. `src/neurips_abstracts/database.py` - Schema and data loading
2. `src/neurips_abstracts/plugin.py` - PaperModel
3. `src/neurips_abstracts/paper_utils.py` - Paper formatting
4. `src/neurips_abstracts/embeddings.py` - Database reading
5. `src/neurips_abstracts/web_ui/app.py` - Web API
6. `scripts/migrate_to_lightweight_schema.py` - Migration script (NEW)

## Compatibility Notes

- Input data should use `title` field (not `name`)
- Authors should be provided as list of strings or list of dicts
- Original UID is preserved in `original_id` column
- All existing functionality is maintained except direct author table queries

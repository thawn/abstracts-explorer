# Schema Update: INTEGER Paper IDs and Author ID References

## Summary of Changes

Updated the database schema to use proper data types and relationships:

### 1. Papers Table: `id` field changed from TEXT to INTEGER

**Before:**
```sql
CREATE TABLE papers (
    id TEXT PRIMARY KEY,
    ...
)
```

**After:**
```sql
CREATE TABLE papers (
    id INTEGER PRIMARY KEY,
    ...
)
```

**Reason:** Paper IDs in NeurIPS data are integers (e.g., 123456). Using INTEGER is more appropriate and efficient than TEXT.

### 2. Papers Table: `authors` field now contains author IDs instead of names

**Before:**
```
authors: "Miaomiao Huang, John Smith"
```

**After:**
```
authors: "457880, 457881"
```

**Reason:** This creates a proper reference to the authors table, enabling:
- Consistent author identification across papers
- Easy lookups using author IDs
- No duplication of author names
- Proper relational database design

### 3. Paper-Authors Junction Table: `paper_id` changed from TEXT to INTEGER

**Before:**
```sql
CREATE TABLE paper_authors (
    paper_id TEXT NOT NULL,
    ...
)
```

**After:**
```sql
CREATE TABLE paper_authors (
    paper_id INTEGER NOT NULL,
    ...
)
```

**Reason:** Must match the data type of papers.id for foreign key integrity.

## Benefits

1. **Type Safety**: INTEGER primary keys are more efficient and prevent type confusion
2. **Referential Integrity**: Author IDs in papers.authors can be validated against authors table
3. **Query Performance**: INTEGER joins and comparisons are faster than TEXT
4. **Data Consistency**: Author IDs are unique and immutable, while names might have variations
5. **Storage Efficiency**: Storing "457880, 457881" is more compact than "Miaomiao Huang, John Smith"

## Code Changes

### In `database.py`

1. **Table Definition** (line ~152):
   ```python
   id INTEGER PRIMARY KEY,  # Changed from TEXT
   ```

2. **Data Loading** (line ~327):
   ```python
   paper_id = int(paper_id)  # Changed from str(paper_id)
   ```

3. **Author Processing** (lines ~332-357):
   ```python
   # Store comma-separated author IDs instead of names
   author_ids = []
   for author_dict in authors_data:
       author_id = author_dict.get('id')
       if author_id:
           author_ids.append(str(author_id))
   authors_str = ", ".join(author_ids)
   ```

4. **Junction Table** (line ~242):
   ```python
   paper_id INTEGER NOT NULL,  # Changed from TEXT
   ```

## Verification

All tests pass with the new schema:

```bash
$ pytest tests/test_authors.py -v
======================= 12 passed in 0.17s =======================
```

### Database Inspection

```
Papers table schema:
  id: INTEGER ✓
  authors: TEXT ✓

Papers data:
  Paper ID: 123456 (type: int) ✓
  Authors field: 457880, 457881 ✓
```

## Usage Examples

### Get Author Names from Papers Table

Since the `authors` field now contains IDs, you should use the junction table for author details:

```python
# Get paper with author details
with DatabaseManager("neurips.db") as db:
    # Get paper
    papers = db.query("SELECT * FROM papers WHERE id = ?", (123456,))
    paper = papers[0]
    
    # Get full author details
    authors = db.get_paper_authors(paper['id'])
    for author in authors:
        print(f"{author['fullname']} ({author['institution']})")
```

### Search Papers by Author

```python
with DatabaseManager("neurips.db") as db:
    # Find author
    authors = db.search_authors(name="Huang")
    author_id = authors[0]['id']
    
    # Get all their papers
    papers = db.get_author_papers(author_id)
    for paper in papers:
        print(paper['name'])
```

## Migration Notes

**Important:** This is a breaking change for existing databases. If you have an existing database with TEXT paper IDs, you will need to:

1. Export your data
2. Drop the old database
3. Create a new database with the updated schema
4. Re-import your data

Or use a migration script to convert the schema.

## Backward Compatibility

The `papers.authors` field still exists but now serves as a quick reference to author IDs rather than display names. For displaying author names, use the `get_paper_authors()` method which joins with the authors table.

## Files Updated

1. ✅ `src/neurips_abstracts/database.py` - Schema and data loading logic
2. ✅ `README.md` - Documentation updated to reflect INTEGER type
3. ✅ `AUTHORS_TABLE_IMPLEMENTATION.md` - Implementation docs updated
4. ✅ All tests passing with new schema

## Summary

The schema is now more normalized and follows database best practices:
- INTEGER primary keys for better performance
- Author IDs as references instead of denormalized names
- Proper foreign key relationships with matching data types
- Efficient storage and querying

**All functionality works correctly with the updated schema!** ✅

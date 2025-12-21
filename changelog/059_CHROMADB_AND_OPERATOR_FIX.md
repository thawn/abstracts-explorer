# ChromaDB Filter Fix - $and Operator

**Date**: November 27, 2025

## Issue

When selecting multiple filters in the RAG chat (e.g., sessions + topics), the following error occurred:

```
Error: Query failed: Failed to search: Expected where to have exactly one operator, 
got {'session': {'$in': [...]}, 'topic': {'$in': [...]}}
```

## Root Cause

ChromaDB's `where` clause requires that when multiple filter conditions are present, they must be wrapped in a single top-level operator like `$and`. The code was creating a flat dictionary with multiple keys, which ChromaDB interprets as having multiple operators.

### Incorrect Structure (Before)
```python
metadata_filter = {
    "session": {"$in": sessions},
    "topic": {"$in": topics},
    "eventtype": {"$in": eventtypes}
}
```

### Correct Structure (After)
```python
metadata_filter = {
    "$and": [
        {"session": {"$in": sessions}},
        {"topic": {"$in": topics}},
        {"eventtype": {"$in": eventtypes}}
    ]
}
```

## Solution

Updated both `/api/chat` and `/api/search` endpoints to properly construct the metadata filter:

1. **Build filter conditions as a list**:
   ```python
   filter_conditions = []
   if sessions:
       filter_conditions.append({"session": {"$in": sessions}})
   if topics:
       filter_conditions.append({"topic": {"$in": topics}})
   if eventtypes:
       filter_conditions.append({"eventtype": {"$in": eventtypes}})
   ```

2. **Use $and operator when multiple conditions exist**:
   ```python
   metadata_filter = None
   if len(filter_conditions) > 1:
       metadata_filter = {"$and": filter_conditions}
   elif len(filter_conditions) == 1:
       metadata_filter = filter_conditions[0]
   ```

## Additional Improvements

### Search Endpoint
- Moved filtering from post-processing to the embeddings search level
- Now passes `where` parameter to `search_similar()` method
- More efficient: filters at database level instead of retrieving excess results

**Before**:
```python
results = em.search_similar(query, n_results=limit * 3)  # Get 3x results
# ... then filter in Python
filtered_papers = [p for p in papers if matches_filters(p)]
```

**After**:
```python
results = em.search_similar(query, n_results=limit * 3, where=where_filter)
papers = papers[:limit]  # Already filtered at DB level
```

## Files Modified

- `src/neurips_abstracts/web_ui/app.py`:
  - `/api/chat` endpoint: Fixed filter construction
  - `/api/search` endpoint: Fixed filter construction + moved filtering to DB level

## Testing

Test cases to verify:
1. ✅ Single filter (session only) - should work
2. ✅ Single filter (topic only) - should work  
3. ✅ Single filter (eventtype only) - should work
4. ✅ Two filters (session + topic) - now works with $and
5. ✅ Three filters (session + topic + eventtype) - now works with $and
6. ✅ No filters - should work (metadata_filter = None)

## ChromaDB Query Operators

For reference, ChromaDB supports these operators:
- `$and`: All conditions must match
- `$or`: At least one condition must match
- `$in`: Value must be in array
- `$nin`: Value must not be in array
- `$eq`: Value equals
- `$ne`: Value not equals
- `$gt`, `$gte`, `$lt`, `$lte`: Comparison operators

Our implementation uses `$and` with `$in` operators for array-based filtering.

## Performance Benefits

Moving filters to the database level provides:
- **Reduced data transfer**: Only matching papers retrieved
- **Lower memory usage**: No need to load excess results
- **Faster response**: Database-level filtering is more efficient
- **Better scaling**: Works well with large datasets

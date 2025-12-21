# ChromaDB Paper ID Conversion Fix

**Date:** November 26, 2025  
**Status:** ✅ Complete

## Problem

The RAG chat feature was failing with the error:
```
"Error: Failed to format papers: No valid papers could be formatted from 5 search results. 
Check database connectivity and paper IDs."
```

Users were unable to use the chat functionality in the web UI.

## Root Cause

ChromaDB stores paper IDs as **strings** (e.g., `'119941'`), but the `get_paper_with_authors()` function requires paper IDs to be **integers**. When `format_search_results()` tried to fetch papers from the database using the string IDs directly, the validation check failed.

```python
# In get_paper_with_authors()
if not isinstance(paper_id, int) or paper_id <= 0:
    raise PaperFormattingError(f"Invalid paper_id: {paper_id}. Must be positive integer.")
```

## Solution

Added explicit type conversion in `format_search_results()` to convert ChromaDB's string IDs to integers before calling `get_paper_with_authors()`.

### Code Changes

**File:** `src/neurips_abstracts/paper_utils.py`

**Before:**
```python
papers = []
for i in range(result_count):
    paper_id = search_results["ids"][0][i]

    try:
        # Get complete paper from database (this validates paper exists)
        paper = get_paper_with_authors(database, paper_id)
```

**After:**
```python
papers = []
for i in range(result_count):
    paper_id = search_results["ids"][0][i]

    try:
        # Convert paper_id to integer (ChromaDB stores as string)
        try:
            paper_id_int = int(paper_id)
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid paper_id format: {paper_id} ({type(paper_id)})")
            continue
        
        # Get complete paper from database (this validates paper exists)
        paper = get_paper_with_authors(database, paper_id_int)
```

## Testing

Verified that:
1. ChromaDB returns IDs as strings: `['119941', '117491', '120165']`
2. SQLite accepts both string and integer IDs in queries
3. Our validation requires integer IDs
4. The conversion works correctly

Test output:
```python
>>> results['ids'][0][0]
'119941'  # String type
>>> int(results['ids'][0][0])
119941  # Integer type
```

## Verification

1. **Direct function test**: ✅
   ```
   Successfully formatted 3 papers
     - 119941: From Softmax to Score: Transformers Can Effectivel
     - 117491: Two Heads are Better than One: Simulating Large Tr
     - 120165: Characterizing the Expressivity of Fixed-Precision
   ```

2. **RAG query test**: ✅
   ```
   Success! Got response with 5 papers
   Response preview: Transformers are a class of deep learning models...
   ```

3. **Web UI chat endpoint**: ✅
   ```
   {
       "response": {
           "metadata": {"model": "diffbot-small-xl-2508", "n_papers": 5},
           "papers": [...]
       }
   }
   ```

4. **All paper_utils tests pass**: ✅ 27/27 tests passing

## Impact

- **Fixed**: RAG chat functionality in web UI
- **Fixed**: All embedding-based search features that use `format_search_results()`
- **No breaking changes**: Regular database queries still work
- **Graceful handling**: Invalid IDs are logged and skipped rather than crashing

## Related Issues

This complements:
- #49: Strict Authors Type Checking in Frontend
- #50: Search Error Handling Fix

Together these changes ensure:
1. Backend provides clean, validated data
2. Frontend enforces strict type checking
3. Errors are handled gracefully with clear messages

## Files Modified

- `src/neurips_abstracts/paper_utils.py` - Added string-to-int conversion for ChromaDB IDs

## Test Results

```
27 passed in 0.23s
Coverage: 99% for paper_utils.py
```

All functionality working correctly.

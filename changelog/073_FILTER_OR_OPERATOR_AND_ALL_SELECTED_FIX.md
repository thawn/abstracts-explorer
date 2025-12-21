# Filter Logic Updates - $or Operator and All-Selected Fix

**Date**: November 27, 2025

## Changes Made

### 1. Changed from `$and` to `$or` Operator

**Issue**: Using `$and` for multiple filter types was too restrictive. A paper can't simultaneously be in multiple sessions, so `$and` logic didn't make semantic sense.

**Solution**: Changed to `$or` operator, which allows papers matching ANY of the filter conditions.

#### Before (AND logic)
```python
# Paper must match session AND topic AND eventtype
metadata_filter = {"$and": [
    {"session": {"$in": ["Session 1", "Session 2"]}},
    {"topic": {"$in": ["Deep Learning"]}},
    {"eventtype": {"$in": ["Poster"]}}
]}
# Result: Only papers that are somehow in both sessions (impossible)
```

#### After (OR logic)
```python
# Paper must match session OR topic OR eventtype
metadata_filter = {"$or": [
    {"session": {"$in": ["Session 1", "Session 2"]}},
    {"topic": {"$in": ["Deep Learning"]}},
    {"eventtype": {"$in": ["Poster"]}}
]}
# Result: Papers in Session 1 OR Session 2 OR with Deep Learning topic OR Poster eventtype
```

### 2. Fixed "All Selected" Behavior

**Issue**: When all filter options were selected (the default), the code was sending ALL values to the backend, creating massive `$in` arrays that were too restrictive and caused no results.

**Solution**: Check if all options are selected, and if so, don't send that filter at all (which means "no filtering").

#### JavaScript Changes

**Search function**:
```javascript
// Before: Always sent selected values
if (sessions.length > 0) requestBody.sessions = sessions;

// After: Only send if NOT all selected
if (sessions.length > 0 && sessions.length < sessionSelect.options.length) {
    requestBody.sessions = sessions;
}
```

**Chat function**: Same logic applied to chat filters.

## Semantic Meaning

The filter logic now works as expected:

### Within a single filter type (uses `$in`)
- "Session 1 OR Session 2" - papers in either session
- "Topic A OR Topic B OR Topic C" - papers with any of these topics

### Across filter types (uses `$or`)
- Papers matching ANY filter condition from ANY filter type
- More inclusive and user-friendly

### Examples

**Example 1**: Select "Session 1" + "Deep Learning" topic
- Returns: Papers in Session 1 OR papers about Deep Learning
- Broad, inclusive results

**Example 2**: Select only "Poster" event type
- Returns: All poster presentations
- No other filters applied

**Example 3**: Select nothing (all options selected)
- Returns: All papers matching the search query
- No filters applied

## Files Modified

- `src/neurips_abstracts/web_ui/app.py`:
  - `/api/search` endpoint: Changed `$and` to `$or`
  - `/api/chat` endpoint: Changed `$and` to `$or`
  
- `src/neurips_abstracts/web_ui/static/app.js`:
  - `searchPapers()`: Added check for all-selected filters
  - `sendChatMessage()`: Added check for all-selected filters

## Benefits

1. **More intuitive**: OR logic matches user expectations for filters
2. **Better results**: Broader search results when multiple filters selected
3. **Fixed defaults**: "All selected" now correctly means "no filter"
4. **Performance**: Not sending massive arrays when all options selected

## Testing

Test scenarios:
- ✅ All filters selected (default) - returns all results
- ✅ One session selected - returns papers from that session
- ✅ Multiple sessions selected - returns papers from any of those sessions
- ✅ Sessions + Topics selected - returns papers matching any condition
- ✅ Deselect all in one filter - that filter type is ignored

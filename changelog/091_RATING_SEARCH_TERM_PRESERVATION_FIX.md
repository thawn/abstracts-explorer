# Rating Search Term Preservation Fix

**Date:** 2025-12-14

## Problem

When changing the rating of a paper in the "Interesting Papers" tab, the search term associated with that paper would be replaced with the last used search term from any tab (search or chat). This happened because the search term was being tracked globally and would get overwritten whenever a new search was performed.

### Bug Behavior

1. User searches for "neural networks" and rates paper X with 3 stars
2. User searches for "transformers" and rates paper Y with 4 stars
3. User goes to "Interesting Papers" tab and changes paper X's rating from 3 to 5 stars
4. **BUG:** Paper X's search term changes from "neural networks" to "transformers"

## Root Cause

In the `setPaperPriority()` function in `app.js`, when updating an existing paper's rating, the code always used the global `currentSearchTerm` variable:

```javascript
paperPriorities[paperId] = {
    priority: priority,
    searchTerm: currentSearchTerm || 'Unknown'  // ❌ Always overwrites existing search term
};
```

The `currentSearchTerm` variable gets updated whenever:

- A search is performed in the Search tab
- A message is sent in the Chat tab
- Query rewriting occurs

## Solution

Modified the `setPaperPriority()` function to preserve the existing search term when updating a rating:

```javascript
const existingSearchTerm = paperPriorities[paperId]?.searchTerm;

paperPriorities[paperId] = {
    priority: priority,
    // Preserve the existing search term when updating, only use currentSearchTerm for new ratings
    searchTerm: existingSearchTerm || currentSearchTerm || 'Unknown'
};
```

### Logic Flow

1. Check if the paper already has a stored search term
2. If it does, keep it when updating the rating
3. If it doesn't (new rating), use the current search term
4. Fall back to 'Unknown' if neither is available

## Changes Made

### Modified Files

- `src/neurips_abstracts/web_ui/static/app.js`
  - Updated `setPaperPriority()` function to preserve existing search terms

## Testing

### Manual Testing Steps

1. Search for a term (e.g., "neural networks") and rate a paper
2. Search for a different term (e.g., "transformers")
3. Go to Interesting Papers tab
4. Change the rating of the first paper
5. Verify the search term remains "neural networks"

### Expected Behavior

- When rating a paper for the first time, it should capture the current search term
- When changing an existing rating, it should preserve the original search term
- Search terms should remain stable regardless of subsequent searches

## Impact

- **Fixed:** Search terms now remain associated with the original search that led to rating the paper
- **Improved:** "Interesting Papers" tab organization by search term is now reliable
- **User Experience:** Users can confidently adjust ratings without losing track of which search found each paper

## Related Features

- Star Rating System (changelog/74_STAR_RATING_FEATURE.md)
- Interesting Papers Tab (changelog/75_INTERESTING_PAPERS_TAB.md)
- Search Term Grouping (changelog/77_SEARCH_TERM_GROUPING.md)

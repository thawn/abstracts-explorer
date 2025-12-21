# Interesting Papers Tab

**Date:** November 29, 2025

## Overview

Added a new "Interesting Papers" tab that displays all papers that users have rated with stars. Papers are organized by session, priority (highest first), and poster number, making it easy to review and export your selections.

## Features

### New Tab
- **"Interesting Papers" tab** in main navigation
- Shows count of rated papers in the tab label
- Automatically updates when papers are rated or unrated

### Paper Organization
Papers are displayed grouped and sorted by:
1. **Session** - Papers grouped by their session
2. **Priority** - Within each session, sorted by star rating (5 stars first, then 4, 3, 2, 1)
3. **Poster Number** - Papers with same priority sorted by poster position

### Empty State
When no papers are rated:
- Friendly message: "No papers rated yet"
- Clear instructions to rate papers using stars

### Export to Markdown
- **"Save as Markdown" button** at the top of the tab
- Downloads a formatted markdown file with:
  - Document header with generation timestamp
  - Search context (if papers were rated from a search)
  - Total paper count
  - Papers grouped by session
  - For each paper:
    - Title
    - Star rating (⭐ symbols)
    - Authors
    - Poster position
    - Paper URL
    - Full abstract
  - Separator lines between papers

### Markdown Format
```markdown
# Interesting Papers from NeurIPS 2025

Generated: [timestamp]

## Search Context

**Search Query:** [if available]

**Total Papers:** [count]

---

## [Session Name]

### [Paper Title]

**Rating:** ⭐⭐⭐⭐⭐ (5/5)

**Authors:** [Author list]

**Poster:** [Position]

**Paper URL:** [OpenReview URL]

**Source URL:** [NeurIPS API URL]

**Abstract:**

[Full abstract text]

---
```

## Changes Made

### Frontend (index.html)

1. **Tab Navigation**
   - Added "Interesting Papers" tab button
   - Added counter showing number of rated papers

2. **Tab Content**
   - New tab section with header and save button
   - Empty state for when no papers are rated
   - Container for displaying papers

### Frontend (app.js)

1. **State Management**
   - `updateInterestingPapersCount()` - Updates count in tab label
   - Modified `loadPriorities()` to update count on load
   - Modified `setPaperPriority()` to refresh tab when changed

2. **Tab Loading**
   - Modified `switchTab()` to load papers when switching to interesting tab
   - `loadInterestingPapers()` - Fetches and displays rated papers
   - `displayInterestingPapers()` - Renders papers grouped by session

3. **Export Functionality**
   - `saveInterestingPapersAsMarkdown()` - Generates and downloads markdown
   - `generateInterestingPapersMarkdown()` - Creates formatted markdown content
   - Includes search context if available
   - Downloads with timestamped filename

### Backend (app.py)

1. **New API Endpoint**
   - `POST /api/papers/batch` - Fetch multiple papers by IDs
   - Accepts list of paper IDs in request body
   - Returns array of paper objects with full details
   - Handles missing papers gracefully

## Implementation Details

### Sorting Algorithm
```javascript
papers.sort((a, b) => {
    // First by session
    const sessionCompare = (a.session || '').localeCompare(b.session || '');
    if (sessionCompare !== 0) return sessionCompare;
    
    // Then by priority (descending - higher priority first)
    if (a.priority !== b.priority) return b.priority - a.priority;
    
    // Finally by poster position
    const aPoster = a.poster_position || '';
    const bPoster = b.poster_position || '';
    return aPoster.localeCompare(bPoster);
});
```

### API Request/Response
```javascript
// Request
POST /api/papers/batch
{
    "paper_ids": [123, 456, 789]
}

// Response
{
    "papers": [
        {
            "id": 123,
            "name": "Paper Title",
            "authors": ["Author 1", "Author 2"],
            "session": "Session A",
            "poster_position": "42",
            "abstract": "...",
            ...
        }
    ]
}
```

## User Experience

### Rating Papers
1. User rates papers with stars in search or chat tabs
2. Counter in "Interesting Papers" tab updates automatically
3. Switching to tab shows all rated papers organized by session

### Reviewing Papers
1. Papers grouped by session for easy navigation
2. Within each session, highest priority papers shown first
3. Full paper details including abstract visible

### Exporting Papers
1. Click "Save as Markdown" button
2. Markdown file downloads with timestamp in filename
3. File includes all paper details in readable format
4. Can be used for conference planning, sharing, or documentation

## Testing

Manual testing performed:
- ✅ Tab appears in navigation with correct count
- ✅ Count updates when papers are rated/unrated
- ✅ Papers load when switching to tab
- ✅ Papers correctly sorted by session, priority, poster
- ✅ Empty state shows when no papers rated
- ✅ Save button downloads markdown file
- ✅ Markdown file has correct format
- ✅ Markdown includes all paper details
- ✅ Search context included in markdown when available
- ✅ Backend API returns multiple papers correctly

## Files Modified

- `src/neurips_abstracts/web_ui/templates/index.html` - Added tab navigation and content
- `src/neurips_abstracts/web_ui/static/app.js` - Added interesting papers functionality
- `src/neurips_abstracts/web_ui/app.py` - Added batch papers API endpoint

## Future Enhancements

Planned for next iteration:
- Download PDF files for prioritized papers
- Download poster images where available
- Link PDFs and posters in markdown
- Filter interesting papers by priority level
- Sort options (by priority, by poster, by title)
- Print-friendly view

## Related Issues

Addresses TODO item:
> prioritized entries should appear in a new `interesting papers` tab. The entries should be ordered by session, priority and poster number the markdown should start with a section that summarizes search phrase and filter settings the `interesting papers` tab should have a save button that allows the user to save its content as markdown file.

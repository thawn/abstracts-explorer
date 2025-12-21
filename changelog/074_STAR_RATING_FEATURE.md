# Priority Star Rating Feature

**Date:** November 29, 2025

## Overview

Added a visual star rating system (1-5 stars) next to each paper entry in both the search results and chat relevant papers list. This allows users to mark papers of interest for later review with an intuitive, interactive interface.

## Changes Made

### Frontend (app.js)

1. **State Management**
   - Added `paperPriorities` object to track paper priorities
   - Priorities stored as `{ paperId: priority }` where priority is 0-5

2. **LocalStorage Persistence**
   - `loadPriorities()` - Loads priorities from localStorage on app initialization
   - `savePriorities()` - Saves priorities to localStorage when changed
   - Data persists across browser sessions

3. **Priority Functions**
   - `setPaperPriority(paperId, priority)` - Sets or removes paper priority
   - `updatePriorityDropdown(paperId)` - Updates dropdown appearance based on priority
   - Priority 0 removes the priority entry

4. **UI Updates**
   - Added interactive star rating to `formatPaperCard()` function
   - Stars appear next to paper title in both compact and full views
   - Visual feedback: empty stars (☆) vs filled stars (★)
   - Stars change color on hover and scale on interaction
   - Stops event propagation to prevent opening paper details when clicking stars

## Features

### Star Rating System
- **5 clickable stars** next to each paper title
- Click a star to rate from 1-5
- Click the same star again to remove the rating
- Visual feedback:
  - Unrated: Gray outlined stars
  - Rated: Yellow filled stars
  - Hover: Stars scale up and change color

### Behavior
- Clicking stars does NOT open paper details
- Priority changes are immediately saved to localStorage
- Priorities persist across page refreshes and browser sessions
- Works in both search results and chat relevant papers panels
- Toggle behavior: clicking an already-selected star removes the rating

### Styling
- Smooth transitions and animations
- Stars scale up on hover (1.15x)
- Stars scale down on click (0.95x) for tactile feedback
- Yellow color (#FBBF24) for filled stars
- Gray color for empty stars
- Consistent sizing in both compact and full views

## Usage

1. **Set Priority**: Click the dropdown next to any paper title and select 1-5
2. **Remove Priority**: Select the ☆ option
3. **View Priorities**: Prioritized papers show ★ with the number in purple/bold

## Technical Details

### LocalStorage Schema
```javascript
{
  "paperPriorities": {
    "123456": 5,
    "789012": 3,
    "345678": 1
  }
}
```

### Event Handling
- `onclick="event.stopPropagation()"` on select element prevents card click
- `onchange` handler calls `setPaperPriority()` to update state

## Future Enhancements

This feature lays the groundwork for:
- "Interesting Papers" tab showing all prioritized papers
- Sorting by priority
- Export prioritized papers to markdown
- Download PDFs for prioritized papers

## Testing

Manual testing performed:
- ✅ Star rating appears in search results
- ✅ Star rating appears in chat relevant papers
- ✅ Clicking stars sets priority in localStorage
- ✅ Priority persists across page refresh
- ✅ Stars update visually when clicked
- ✅ Clicking stars doesn't open paper details
- ✅ Clicking the same star removes the rating
- ✅ Hover effects work smoothly
- ✅ Stars scale on hover and click for tactile feedback

## Files Modified

- `src/neurips_abstracts/web_ui/static/app.js` - Added star rating functionality and UI
- `src/neurips_abstracts/web_ui/static/style.css` - Added star animation styles

## Related Issues

Addresses the first part of TODO item:
> add a dropdown next to each entry in the search results (and in the chat relevant paper list) allowing the user to prioritize each entry from 1 to 5.

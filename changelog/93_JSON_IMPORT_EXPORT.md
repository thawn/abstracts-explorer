# JSON Import/Export Feature for Interesting Papers

## Summary

Added functionality to save and load the "interesting papers" data (paper ratings stored in browser local storage) as JSON files. This allows users to:

- Back up their paper ratings
- Share their ratings with colleagues
- Transfer ratings between browsers/devices
- Restore ratings after clearing browser data

## Changes Made

### HTML Template (`src/neurips_abstracts/web_ui/templates/index.html`)

Added two new buttons to the Interesting Papers tab header:

- **Save JSON** button: Downloads current paper ratings as a JSON file
- **Load JSON** button: Uploads and merges paper ratings from a JSON file
- Added hidden file input element for file selection

### JavaScript Functions (`src/neurips_abstracts/web_ui/static/app.js`)

#### `saveInterestingPapersAsJSON()`

Exports the current paper ratings to a downloadable JSON file with:

- Version information
- Export timestamp
- Current sort order preference
- All paper priorities with metadata (priority level and search term)
- Total paper count

#### `loadInterestingPapersFromJSON()`

Triggers the file input dialog to select a JSON file to import.

#### `handleJSONFileLoad(event)`

Processes the uploaded JSON file:

- Validates file format (.json)
- Parses and validates JSON structure
- Merges imported ratings with existing ones (existing ratings take precedence)
- Shows confirmation dialog if there are conflicts
- Updates UI and localStorage
- Provides detailed success/error messages

## JSON File Format

```json
{
  "version": "1.0",
  "exportDate": "2025-12-14T10:00:00.000Z",
  "sortOrder": "search-rating-poster",
  "paperPriorities": {
    "1": {
      "priority": 3,
      "searchTerm": "machine learning"
    },
    "2": {
      "priority": 2,
      "searchTerm": "neural networks"
    }
  },
  "paperCount": 2
}
```

## Features

### Save Functionality

- ✅ Validates that papers have been rated before saving
- ✅ Creates properly formatted JSON with metadata
- ✅ Automatically names file with current date
- ✅ Error handling with user-friendly messages

### Load Functionality

- ✅ Validates file type (.json only)
- ✅ Validates JSON structure and required fields
- ✅ Smart merging: preserves existing ratings on conflict
- ✅ Confirmation dialog when merging with existing data
- ✅ Detailed success messages showing merge statistics
- ✅ Error handling for invalid files
- ✅ File input reset after processing

### User Experience

- Clear button labels with Font Awesome icons
- Hover tooltips explaining button functionality
- Color-coded buttons (blue for save, green for load)
- Comprehensive alert messages for all scenarios
- Non-destructive merge by default

## Use Cases

1. **Backup**: Save your ratings before clearing browser data
2. **Collaboration**: Share your rated papers list with colleagues
3. **Device Transfer**: Move ratings from desktop to laptop
4. **Archival**: Keep historical records of conference papers of interest
5. **Recovery**: Restore ratings after accidental browser data loss

## Technical Notes

- Data is stored in browser's localStorage under key `paperPriorities`
- Export files are human-readable JSON (pretty-printed)
- Import is non-destructive (won't overwrite existing ratings unless user confirms)
- Sort order preference can be imported if not already set locally
- File names include ISO date format for easy chronological sorting

## Testing

Unit tests added in `src/neurips_abstracts/web_ui/tests/app.test.js`:

- Test for empty ratings validation
- Test for JSON file creation and structure
- Test for file type validation
- Test for JSON parsing and merging logic
- Test for conflict resolution
- Test for error handling

## Future Enhancements

Potential future improvements:

- Option to force overwrite existing ratings during import
- Merge strategies (keep newest, keep highest priority, etc.)
- Export selected papers only (filtered by rating, session, etc.)
- CSV export format option
- Direct integration with backend for server-side storage

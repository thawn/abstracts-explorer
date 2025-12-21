# PDF and Poster Image Download Option

**Date:** 2025-11-29  
**Status:** ✅ Completed

## Overview

Added a checkbox option next to the "Save as Markdown" button in the Interesting Papers tab that allows users to optionally download PDFs and poster images along with the markdown file. When checked, the export generates a ZIP file containing the markdown and all downloaded assets.

## Changes Made

### Frontend (`src/neurips_abstracts/web_ui/templates/index.html`)

1. **Added Checkbox UI**
   - Added a checkbox labeled "Include PDFs & Images" next to the Save button
   - Used Tailwind CSS for consistent styling
   - Wrapped button and checkbox in a flex container for proper alignment

### Frontend (`src/neurips_abstracts/web_ui/static/app.js`)

1. **Updated `saveInterestingPapersAsMarkdown()` Function**
   - Reads the checkbox state from `#download-assets-checkbox`
   - Passes `download_assets` parameter to the backend API based on checkbox state
   - Determines file extension dynamically (`.zip` when assets included, `.md` otherwise)
   - Updates filename accordingly

## User Experience

### Without Checkbox (Unchecked - Default)
- Downloads: `interesting-papers-2025-11-29.md`
- Contains: Markdown file with paper details and URLs
- Fast download, small file size

### With Checkbox (Checked)
- Downloads: `interesting-papers-2025-11-29.zip`
- Contains: 
  - `interesting-papers.md` with embedded asset links
  - `assets/` folder with PDFs and poster images
- Longer download time, larger file size
- Useful for offline access

## Technical Details

### Checkbox HTML
```html
<label class="flex items-center cursor-pointer">
    <input type="checkbox" id="download-assets-checkbox" 
           class="mr-2 w-4 h-4 text-purple-600 border-gray-300 rounded focus:ring-purple-500">
    <span class="text-sm text-gray-700">Include PDFs & Images</span>
</label>
```

### JavaScript Logic
```javascript
// Check if user wants to download assets
const downloadAssetsCheckbox = document.getElementById('download-assets-checkbox');
const downloadAssets = downloadAssetsCheckbox ? downloadAssetsCheckbox.checked : false;

// Determine file extension based on assets option
const fileExtension = downloadAssets ? 'zip' : 'md';
a.download = `interesting-papers-${new Date().toISOString().split('T')[0]}.${fileExtension}`;
```

## Backend Integration

The backend (`app.py`) already has full support for asset downloads:
- PDF downloads from OpenReview URLs
- Poster image extraction from eventmedia JSON
- ZIP file creation with assets folder
- Conditional logic based on `download_assets` parameter

## Benefits

1. **User Choice**: Users can decide whether they need offline assets
2. **Flexibility**: Fast markdown-only export for quick saves, full export when needed
3. **Bandwidth Friendly**: Default behavior doesn't download large files unnecessarily
4. **Conference Preparation**: Checkbox enabled provides complete offline access for conference attendance

## UI Layout

```
┌─────────────────────────────────────────────────────────────┐
│ ⭐ Interesting Papers    [ ] Include PDFs & Images  [Save]  │
└─────────────────────────────────────────────────────────────┘
```

## Testing Recommendations

1. **Unchecked (Default)**
   - Click "Save as Markdown" without checking the box
   - Verify `.md` file downloads
   - Verify markdown contains URLs but no embedded assets

2. **Checked**
   - Check "Include PDFs & Images"
   - Click "Save as Markdown"
   - Verify `.zip` file downloads
   - Extract and verify `assets/` folder contains PDFs
   - Verify markdown has relative links to assets

3. **Toggle Behavior**
   - Try multiple exports with different checkbox states
   - Verify correct file type each time

## Files Modified

- `src/neurips_abstracts/web_ui/templates/index.html` - Added checkbox UI
- `src/neurips_abstracts/web_ui/static/app.js` - Updated export function logic

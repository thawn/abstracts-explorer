# PDF and Poster Image Download Feature

**Date:** November 29, 2025

## Overview

Enhanced the "Interesting Papers" export feature to automatically download PDF files and poster images when saving. The export now creates a ZIP file containing a markdown file with all papers plus a folder with downloaded assets.

## Features

### Automatic Asset Downloads
- **PDF Files**: Downloads paper PDFs from `paper_pdf_url` field
- **Poster Images**: Extracts and downloads poster images from `eventmedia` field
- **Organized Structure**: Creates `assets/` folder with all downloaded files
- **Markdown Links**: Automatically links to local files in the markdown

### ZIP File Export
- Single ZIP file contains everything
- `interesting-papers.md` - Main markdown file
- `assets/` - Folder with all PDFs and images
- Easy to share and archive

### Progress Feedback
- Button shows "Preparing download..." during processing
- Success message after completion
- Error handling with informative messages

### Markdown Format with Assets

```markdown
### Paper Title

**Rating:** ⭐⭐⭐⭐⭐ (5/5)

**Authors:** John Doe, Jane Smith

**Poster:** #42

**PDF:** [Download](assets/paper_12345.pdf)

**Poster Image:** ![Poster](assets/poster_12345.jpg)

**Paper URL:** https://openreview.net/...

**Source URL:** https://neurips.cc/...

**Abstract:**

[Full abstract text...]
```

## Implementation Details

### Backend (app.py)

1. **New API Endpoint**
   - `POST /api/export/interesting-papers`
   - Accepts paper IDs, priorities, search query, and download_assets flag
   - Returns ZIP file with markdown and assets

2. **File Download Functions**
   - `download_file(url, target_dir, filename)` - Downloads any file from URL
   - `download_poster_image(eventmedia, target_dir, base_filename)` - Extracts and downloads images from eventmedia JSON

3. **Markdown Generation**
   - `generate_markdown_with_assets(papers, search_query, assets_dir)` - Generates markdown with local file links
   - Downloads assets during generation
   - Handles missing files gracefully (links to online URLs if download fails)
   - Constructs PDF URLs from OpenReview paper URLs when `paper_pdf_url` is not available

4. **PDF URL Construction**
   - If `paper_pdf_url` is not in database, constructs it from `paper_url`
   - Converts: `https://openreview.net/forum?id=ABC123` 
   - To: `https://openreview.net/pdf?id=ABC123`
   - Ensures maximum PDF availability

4. **Temporary File Management**
   - Creates temporary directory for downloads
   - Packages everything into ZIP
   - Cleans up temporary files after sending

### Frontend (app.js)

1. **Updated Export Function**
   - `saveInterestingPapersAsMarkdown()` - Now calls new backend API
   - Shows progress indicator while processing
   - Downloads ZIP file instead of plain markdown
   - Better error handling and user feedback

2. **Button State Management**
   - Disables button during processing
   - Shows spinner icon
   - Updates text to "Preparing download..."
   - Shows success message after completion

### UI Updates (index.html)

1. **Button Text**
   - Changed from "Save as Markdown" to "Export with PDFs & Images"
   - More descriptive of what it does

2. **Help Text**
   - Added explanation about export contents
   - Clarifies that PDFs and images are included

## Dependencies Added

- `requests` - For downloading files from URLs
- Python's built-in modules:
  - `tempfile` - Temporary directory creation
  - `shutil` - File operations and cleanup
  - `zipfile` - Creating ZIP archives
  - `io.BytesIO` - In-memory file buffer

## Error Handling

### Download Failures
- If PDF download fails, markdown includes online URL instead
- If poster image not available, skips gracefully
- Continues processing even if some downloads fail
- Logs warnings for failed downloads

### Backend Errors
- Returns appropriate HTTP status codes
- Provides descriptive error messages
- Cleans up temporary files on error

### Frontend Errors
- Shows alert with error message
- Resets button state
- Logs errors to console for debugging

## File Naming Convention

- PDFs: `paper_[paper_id].pdf`
- Posters: `poster_[paper_id].[ext]` (extension from image URL)
- ZIP: `interesting-papers-[year].zip`

## Asset Discovery

### PDF URLs
- Reads from `paper_pdf_url` field in database first
- If not available, constructs PDF URL from `paper_url` by replacing `/forum?id=` with `/pdf?id=`
- Direct download from OpenReview

### Poster Images
- Parses `eventmedia` JSON field
- Looks for URLs ending in common image extensions
- Supports: .jpg, .jpeg, .png, .gif, .webp

## Testing

Manual testing scenarios:
- ✅ Export with papers that have PDFs
- ✅ Export with papers that have poster images
- ✅ Export with papers missing assets
- ✅ ZIP file structure is correct
- ✅ Markdown links work correctly
- ✅ Progress indicator shows during processing
- ✅ Error handling works properly
- ✅ Temporary files cleaned up
- ✅ Large downloads complete successfully

## Example ZIP Structure

```
interesting-papers-2025.zip
├── interesting-papers.md
└── assets/
    ├── paper_12345.pdf
    ├── paper_12346.pdf
    ├── poster_12345.jpg
    └── poster_12347.png
```

## Performance Considerations

### Download Timeout
- Set to 30 seconds per file
- Prevents hanging on slow connections

### Streaming
- Uses streaming for large PDF downloads
- Writes in chunks to handle large files

### Parallel Processing
- Downloads happen sequentially (simpler, more reliable)
- Could be parallelized in future for faster performance

## Security Considerations

### URL Validation
- Only downloads from provided URLs
- No arbitrary file access

### Temporary Files
- Created in system temp directory
- Cleaned up after use
- Unique directory per request

### File Size
- No explicit limit (relies on timeout)
- Could add size checks in future

## Future Enhancements

Potential improvements:
- Progress bar showing download status
- Option to skip PDF/image downloads
- Batch download optimization
- Caching of previously downloaded files
- Size estimates before download
- Parallel downloads for faster export

## Files Modified

- `src/neurips_abstracts/web_ui/app.py` - Added export endpoint and download functions
- `src/neurips_abstracts/web_ui/static/app.js` - Updated export function
- `src/neurips_abstracts/web_ui/templates/index.html` - Updated button text and help text

## Related Issues

Addresses TODO item:
> upon saving the markdown, download PDF files and poster images where available and link them in the markdown.

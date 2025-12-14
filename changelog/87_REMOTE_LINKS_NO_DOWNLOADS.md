# Link to Remote Resources Instead of Downloading

**Date:** December 13, 2025

## Summary

Modified the "Interesting Papers" export feature to link directly to remote resources (PDFs on OpenReview and poster images on neurips.cc) instead of downloading them. This results in instant exports, zero bandwidth usage, and significantly smaller export files while maintaining full access to all resources.

## Motivation

Previously, the export feature downloaded poster images locally. While this provided offline access, it had several drawbacks:

1. **Speed**: Downloads take time, even with parallelization (5-10 seconds for 50 papers)
2. **Bandwidth**: Each poster is ~300 KB, totaling ~15 MB for 50 papers
3. **Storage**: Export files are large due to embedded images
4. **Maintenance**: Images may become outdated if updated on the source
5. **Reliability**: Download failures can occur due to network issues

By linking to remote resources:

- ✅ **Instant exports** - no download time
- ✅ **Zero bandwidth** - no files transferred
- ✅ **Tiny export files** - just markdown text
- ✅ **Always current** - links point to latest versions
- ✅ **No failures** - no download errors to handle

## Changes

### New Function

**`get_poster_url(eventmedia, paper_id)`** in `src/neurips_abstracts/web_ui/app.py`:

- Extracts poster image URLs from eventmedia JSON
- Handles both `file` and `url` fields
- Skips thumbnail versions in favor of full-size images
- Falls back to constructing URL from paper ID
- Returns the poster URL without downloading

### Modified Functions

**`generate_markdown_with_assets(papers, search_query, assets_dir)`**:

- Removed all download logic (no longer calls `download_assets_parallel`)
- Changed `assets_dir` parameter to ignored (kept for backward compatibility)
- Uses `get_poster_url()` to get direct URLs to poster images
- Generates markdown with remote image links instead of local file references

### Removed Logic

- No longer downloads poster images in parallel
- No longer creates assets directory
- No longer saves files locally

## Impact

### Performance Comparison

For an export of 50 papers:

**Before (poster downloads):**

- Export time: ~5-10 seconds
- Bandwidth used: ~15 MB (50 × 0.3 MB)
- Export file size: ~15 MB + markdown (~15 MB total)

**After (remote links):**

- Export time: <1 second (instant)
- Bandwidth used: 0 bytes
- Export file size: ~50 KB (markdown only)

**Improvements:**

- ⚡ **~10-50x faster** - instant vs 5-10 seconds
- 💾 **100% bandwidth reduction** - 0 MB vs 15 MB
- 📦 **~300x smaller files** - 50 KB vs 15 MB

### User Experience

**Markdown Output:**

```markdown
**PDF:** [View on OpenReview](https://openreview.net/pdf?id=zytITzY4IW)
**Poster Image:** ![Poster](https://neurips.cc/media/PosterPDFs/NeurIPS%202025/114996.png)
```

**Benefits:**

- Instant markdown generation and download
- Works with any markdown viewer that supports remote images
- Images load on-demand when viewing
- Always shows the latest version from neurips.cc
- No storage requirements on user's device

**Requirements:**

- Internet connection required to view poster images
- Poster images must be publicly accessible on neurips.cc

## Testing

### Unit Tests

Added comprehensive tests in `tests/test_web_ui_unit.py`:

**New test class `TestGetPosterUrl`:**

- `test_get_poster_url_from_file_path`: Tests URL extraction from `file` field
- `test_get_poster_url_skips_thumbnails`: Tests thumbnail skipping logic
- `test_get_poster_url_fallback`: Tests fallback URL construction
- `test_get_poster_url_from_url_field`: Tests URL extraction from `url` field

**Updated test:**

- `test_generate_markdown_with_remote_links`: Verifies markdown contains remote URLs

All 35 web UI unit tests pass.

### Manual Testing

Verified with real NeurIPS 2025 papers:

- ✅ No files created during export
- ✅ PDFs link to OpenReview correctly
- ✅ Poster images link to neurips.cc correctly
- ✅ Markdown format is correct: `![Poster](https://neurips.cc/...)`
- ✅ Export completes instantly
- ✅ Generated markdown file is ~50 KB (vs ~15 MB previously)

## Files Changed

- `src/neurips_abstracts/web_ui/app.py`:
  - Added `get_poster_url()` function to extract poster URLs
  - Modified `generate_markdown_with_assets()` to use remote links
  - Removed download logic from markdown generation
- `tests/test_web_ui_unit.py`:
  - Added `TestGetPosterUrl` class with 4 test methods
  - Updated markdown generation test to verify remote links
  - All tests pass

## Backward Compatibility

**Breaking Changes:**

- Export no longer includes local copies of poster images
- `assets_dir` parameter in `generate_markdown_with_assets()` is now ignored
- Export is now just a markdown file (no ZIP with assets folder)

**Maintained:**

- Same markdown structure and formatting
- Same function signatures
- PDFs still link to OpenReview (unchanged)
- Poster images still embedded in markdown

## Migration

For users who previously exported papers:

**Old format (ZIP with assets):**

```text
export.zip
├── interesting_papers.md
└── assets/
    ├── poster_114996.png
    ├── poster_114997.png
    └── ...
```

**New format (markdown only):**

```text
interesting_papers.md
```

Images now load from remote URLs when the markdown is viewed.

## Future Considerations

Potential future enhancements:

1. **Optional downloads**: Add a checkbox to optionally download images locally
2. **Hybrid mode**: Link to remote by default, but allow selective downloads
3. **Caching**: Browser/viewer cache handles repeated access efficiently
4. **Offline mode**: Pre-download for offline viewing when needed
5. **PDF previews**: Add preview images for PDFs (could also be remote links)

## Notes

- Poster images on neurips.cc are publicly accessible
- No authentication required to view poster images
- Images load on-demand in markdown viewers
- Modern markdown viewers handle remote images well
- If neurips.cc is down, images won't display (but links remain valid)

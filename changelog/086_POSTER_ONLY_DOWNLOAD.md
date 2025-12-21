# Download Posters Only, Link to PDFs on OpenReview

**Date:** December 13, 2025

## Summary

Modified the "Interesting Papers" export feature to download only poster images and link to PDFs on OpenReview instead of downloading them. This reduces download time, bandwidth usage, and storage requirements while still providing full access to all paper resources.

## Motivation

Previously, the export feature downloaded both PDFs and poster images. However:

1. **Storage**: PDF files are large (typically 1-10 MB each), while poster images are smaller (200-500 KB)
2. **Bandwidth**: Downloading many PDFs consumes significant bandwidth
3. **Speed**: Even with parallel downloads, PDFs take longer to download
4. **Accessibility**: PDFs on OpenReview are permanently accessible and include the full paper with references
5. **Up-to-date**: OpenReview may have updated versions with corrections

By linking to PDFs instead of downloading them, users get:

- Faster exports (only download poster images)
- Smaller export file sizes
- Direct access to the canonical source on OpenReview
- Access to any updates or corrections made after export

## Changes

### Modified Functions

**`download_assets_parallel(papers, assets_dir, max_workers=10)`** in `src/neurips_abstracts/web_ui/app.py`:

- Removed PDF download tasks
- Now only downloads poster images in parallel
- Returns single dict (poster_results) instead of tuple (pdf_results, poster_results)
- Updated docstring to clarify posters-only behavior

**`generate_markdown_with_assets(papers, search_query, assets_dir)`** in `src/neurips_abstracts/web_ui/app.py`:

- Removed `pdf_results` dictionary
- Changed to call `download_assets_parallel()` for posters only
- Always generates PDF links to OpenReview (not local files)
- Uses "View on OpenReview" as link text instead of "Download"

### Updated Tests

Modified tests in `tests/test_web_ui_unit.py`:

- `test_download_assets_parallel_success`: Now tests poster-only downloads
- `test_download_assets_parallel_handles_failures`: Updated for poster-only behavior
- `test_generate_markdown_with_parallel_downloads`: Verifies PDF links and poster downloads

All 31 tests pass.

## Impact

### Performance Improvements

For an export of 50 papers:

**Before:**

- Download time: ~10-30 seconds (parallel)
- Average file sizes: PDFs ~5 MB, Posters ~300 KB
- Total download: ~265 MB (50 × 5.3 MB)
- Export file size: ~265 MB + markdown

**After:**

- Download time: ~5-10 seconds (parallel, posters only)
- Average file sizes: Posters ~300 KB only
- Total download: ~15 MB (50 × 0.3 MB)
- Export file size: ~15 MB + markdown

**Improvements:**

- ~2-3x faster downloads
- ~95% reduction in bandwidth usage
- ~95% reduction in export file size

### User Experience

**Markdown Output Changes:**

Before:

```markdown
**PDF:** [Download](assets/paper_114996.pdf)
```

After:

```markdown
**PDF:** [View on OpenReview](https://openreview.net/pdf?id=zytITzY4IW)
```

Poster images are still embedded:

```markdown
**Poster Image:** ![Poster](assets/poster_114996.png)
```

### Backward Compatibility

- Export format remains consistent (still creates ZIP with markdown and assets folder)
- Markdown structure unchanged
- Poster images still downloaded and embedded
- Only change: PDFs are now external links instead of local files

## Testing

### Unit Tests

All existing tests updated and passing:

- PDF download task functions still exist (may be used elsewhere)
- Poster download tasks work correctly
- Parallel execution validated
- Markdown generation produces correct links

### Manual Testing

Verified with real NeurIPS 2025 papers:

- ✅ Poster images downloaded successfully
- ✅ PDF links point to OpenReview
- ✅ No PDF files created in assets directory
- ✅ Markdown contains "View on OpenReview" links
- ✅ Export file size significantly reduced

## Files Changed

- `src/neurips_abstracts/web_ui/app.py`:
  - Modified `download_assets_parallel()` to download only posters
  - Modified `generate_markdown_with_assets()` to link to PDFs on OpenReview
- `tests/test_web_ui_unit.py`:
  - Updated 3 test methods to reflect poster-only downloads
  - All tests pass

## Future Considerations

Potential future enhancements:

1. **Optional PDF downloads**: Add a toggle in the UI to optionally download PDFs
2. **Cached downloads**: Download PDFs only if not already cached locally
3. **Selective downloads**: Let users choose which papers to download PDFs for
4. **PDF previews**: Show PDF preview thumbnails in the web interface

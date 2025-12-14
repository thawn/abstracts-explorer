# Parallel Download of Papers and Poster Images

**Date:** December 13, 2025

## Summary

Implemented parallel downloading of PDFs and poster images in the web UI's "Interesting Papers" export feature. Downloads now execute concurrently using a thread pool, significantly reducing export time when downloading multiple papers.

## Problem

The previous implementation downloaded PDFs and poster images sequentially, one at a time. For exports with many papers, this resulted in long wait times since each download had to complete before the next one started. With typical download times of 1-3 seconds per file, exporting 50 papers could take several minutes.

## Solution

### Architecture Changes

Refactored the download process to use `concurrent.futures.ThreadPoolExecutor`:

1. **Separated download logic from markdown generation**: Downloads now happen in a batch operation before generating the markdown
2. **Parallel execution**: Multiple downloads run concurrently with configurable worker threads (default: 10)
3. **Result mapping**: Download results are collected in dictionaries keyed by paper ID for efficient lookup during markdown generation

### New Functions

Added to `src/neurips_abstracts/web_ui/app.py`:

1. **`download_paper_pdf_task(paper, assets_dir)`**
   - Wrapper function for downloading a single PDF
   - Returns tuple of (paper_id, filename)
   - Handles URL construction from paper_url if needed

2. **`download_poster_image_task(paper, assets_dir)`**
   - Wrapper function for downloading a single poster image
   - Returns tuple of (paper_id, filename)
   - Uses existing `download_poster_image` function

3. **`download_assets_parallel(papers, assets_dir, max_workers=10)`**
   - Main parallel download coordinator
   - Submits all PDF and poster download tasks to thread pool
   - Collects results as they complete using `as_completed()`
   - Returns two dictionaries: pdf_results and poster_results
   - Logs summary of successful downloads

### Modified Functions

**`generate_markdown_with_assets()`**:

- Now calls `download_assets_parallel()` first if assets_dir is provided
- Uses result dictionaries to look up downloaded filenames
- Falls back to showing URLs if downloads failed
- Maintains same markdown output format

## Performance Impact

### Time Complexity

- **Before**: O(n) sequential downloads, ~1-3 seconds each
- **After**: O(n/workers) parallel downloads with 10 workers

### Expected Speedup

For 50 papers with mixed PDFs and posters:

- **Sequential**: ~100-300 seconds (50 × 2 files × 1-3 sec)
- **Parallel**: ~10-30 seconds (50 × 2 / 10 workers × 1-3 sec)
- **Speedup**: ~10x faster

### Resource Usage

- Memory: Minimal increase (thread overhead only)
- Network: Up to 10 concurrent connections (configurable)
- CPU: Negligible (IO-bound operation)

## Configuration

The number of parallel workers can be adjusted in the `download_assets_parallel()` function:

```python
pdf_results, poster_results = download_assets_parallel(papers, assets_dir, max_workers=10)
```

Default of 10 workers provides good balance between speed and server load.

## Error Handling

- Individual download failures don't stop the entire process
- Failed downloads are logged with warnings
- Markdown generation falls back to showing URLs for failed downloads
- Graceful degradation ensures export always completes

## Testing

Added comprehensive unit tests in `tests/test_web_ui_unit.py`:

- `test_download_paper_pdf_task_success`: Tests PDF download task
- `test_download_paper_pdf_task_constructs_url`: Tests URL construction
- `test_download_paper_pdf_task_no_url`: Tests handling of missing URLs
- `test_download_poster_image_task_success`: Tests poster download task
- `test_download_assets_parallel_success`: Tests parallel execution
- `test_download_assets_parallel_handles_failures`: Tests error handling
- `test_generate_markdown_with_parallel_downloads`: Tests markdown integration

All 31 web UI unit tests pass (7 new tests added).

## Files Changed

- `src/neurips_abstracts/web_ui/app.py`:
  - Added `concurrent.futures` import
  - Added `download_paper_pdf_task()` function
  - Added `download_poster_image_task()` function
  - Added `download_assets_parallel()` function
  - Modified `generate_markdown_with_assets()` to use parallel downloads
- `tests/test_web_ui_unit.py`:
  - Added `TestParallelDownload` test class with 7 test methods

## Backward Compatibility

- No breaking changes
- Same API for `generate_markdown_with_assets()`
- Same markdown output format
- Existing download_file() and download_poster_image() functions unchanged

## Future Improvements

Potential enhancements:

1. Make max_workers configurable via web UI settings
2. Add progress reporting during downloads
3. Implement download retry logic with exponential backoff
4. Add caching to avoid re-downloading same files
5. Support resumable downloads for large files

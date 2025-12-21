# Poster Image Download Fix

**Date:** December 13, 2025

## Summary

Fixed the poster image download functionality in the web UI's "Interesting Papers" export feature. The system now correctly downloads poster images from NeurIPS 2025 by parsing the `file` field in the eventmedia JSON and constructing proper URLs. Added fallback mechanism to attempt downloading posters directly using the paper ID when they're not listed in eventmedia.

## Problem

The poster image download feature was not working because:

1. The `download_poster_image` function only looked for `url` keys in the eventmedia JSON
2. NeurIPS 2025 poster images are stored with a `file` key containing paths like `/media/PosterPDFs/NeurIPS%202025/114996.png`
3. These paths need to be prefixed with `https://neurips.cc` to create valid download URLs
4. About 46% of poster papers don't have poster images listed in their eventmedia field

## Solution

### Updated `download_poster_image` Function

Modified `/src/neurips_abstracts/web_ui/app.py`:

1. **Added paper_id parameter**: Function now accepts an optional `paper_id` parameter for fallback URL construction

2. **Parse `file` field**: Added logic to check for `file` key in eventmedia JSON entries:

   ```python
   file_path = media_item.get("file")
   if file_path and any(ext in file_path.lower() for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]):
       full_url = f"https://neurips.cc{file_path}"
   ```

3. **Skip thumbnails**: Added logic to skip thumbnail versions (`-thumb.png`) in favor of full-size images

4. **Fallback mechanism**: If no poster is found in eventmedia, construct URL directly from paper ID:

   ```python
   if paper_id:
       poster_url = f"https://neurips.cc/media/PosterPDFs/NeurIPS%202025/{paper_id}.png"
       return download_file(poster_url, target_dir, filename)
   ```

### Updated Function Call

Modified the call site to pass the paper ID:

```python
poster_filename = download_poster_image(
    paper.get("eventmedia"), 
    assets_dir, 
    f"poster_{paper['id']}", 
    paper['id']  # Added paper ID for fallback
)
```

## Testing

Added comprehensive unit tests in `tests/test_web_ui_unit.py`:

- `test_download_poster_from_file_path`: Verifies downloading from `file` field in eventmedia
- `test_download_poster_skips_thumbnails`: Ensures thumbnails are skipped for full-size images
- `test_download_poster_fallback_to_paper_id`: Tests fallback URL construction from paper ID
- `test_download_poster_no_eventmedia_uses_fallback`: Tests fallback when eventmedia is None
- `test_download_poster_returns_none_on_error`: Verifies graceful error handling
- `test_download_poster_from_url_field`: Tests backward compatibility with `url` field

All tests pass (6/6 in new test class, 24/24 in full test suite).

## Impact

- **Coverage**: About 3,159 of 5,846 poster papers (54%) have poster images explicitly listed in eventmedia
- **Fallback benefit**: The remaining papers can now attempt download using the constructed URL pattern
- **URL format**: `https://neurips.cc/media/PosterPDFs/NeurIPS%202025/{paper_id}.png`

## Files Changed

- `src/neurips_abstracts/web_ui/app.py`: Updated `download_poster_image` function and call site
- `tests/test_web_ui_unit.py`: Added 6 new unit tests for poster download functionality

## Verification

Example poster URLs that now work correctly:

- <https://neurips.cc/media/PosterPDFs/NeurIPS%202025/114996.png>
- <https://neurips.cc/media/PosterPDFs/NeurIPS%202025/114997.png>
- <https://neurips.cc/media/PosterPDFs/NeurIPS%202025/115951.png> (from user's example)

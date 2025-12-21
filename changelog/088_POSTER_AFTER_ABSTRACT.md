# Poster Images Placement After Abstracts

**Date:** December 13, 2025

## Summary

Modified the markdown generation to place poster images below abstracts instead of before them. This improves the reading flow by ensuring users read the abstract first before seeing the visual representation.

## Motivation

Previously, poster images appeared between the paper URL and the abstract:

```markdown
**Paper URL:** https://openreview.net/forum?id=test123

**Poster Image:** ![Poster](...)

**Abstract:**

The paper abstract text...
```

This interrupted the natural flow of information. The abstract is the primary textual description of the paper and should be read before viewing the poster image.

## Changes

### Modified Function

**`generate_markdown_with_assets()`** in `src/neurips_abstracts/web_ui/app.py`:

- Moved the poster image section from before the abstract to after it
- Poster images now appear as the last element before the separator

## Impact

### New Markdown Structure

Papers now have this order:

1. Title (heading)
2. Rating (stars)
3. Authors
4. Poster position (if available)
5. PDF link (to OpenReview)
6. Paper URL (to OpenReview forum)
7. Source URL (if available)
8. **Abstract** ← textual content
9. **Poster Image** ← visual content
10. Separator (---)

### Example Output

```markdown
#### Example Paper Title

**Rating:** ⭐⭐⭐ (3/5)

**Authors:** Author One, Author Two

**Poster:** #1234

**PDF:** [View on OpenReview](https://openreview.net/pdf?id=test123)

**Paper URL:** https://openreview.net/forum?id=test123

**Abstract:**

This is the abstract text. It contains important information about the paper.

**Poster Image:** ![Poster](https://neurips.cc/media/PosterPDFs/NeurIPS%202025/114996.png)

---
```

## Benefits

- ✅ **Better reading flow**: Abstract comes before visual content
- ✅ **Logical order**: Text description followed by visual representation
- ✅ **Standard practice**: Most academic papers show abstracts before figures
- ✅ **Improved accessibility**: Screen readers encounter text before images

## Testing

### Manual Verification

Tested with sample papers:

- ✅ Abstract appears before poster image
- ✅ Poster image appears as last element before separator
- ✅ All other fields remain in correct order

### Unit Tests

All 35 existing web UI unit tests pass with no changes needed.

## Files Changed

- `src/neurips_abstracts/web_ui/app.py`:
  - Moved poster image generation after abstract in `generate_markdown_with_assets()`
  - Added comment indicating placement is intentional

## Backward Compatibility

**Non-breaking change:**

- Same markdown elements, just reordered
- No API changes
- All tests pass without modification
- Export format remains the same

# Export File Structure Based on Sort Order

**Date:** 2025-12-14  
**Type:** Feature Enhancement

## Summary

Enhanced the "Export as Zip" functionality to organize papers in the exported markdown files based on the selected sort order. The folder structure now respects the first sort priority, making it easier to navigate exported papers according to user preference.

## Changes Made

### 1. Modified Export File Structure (`app.py`)

Updated `generate_folder_structure_export()` to create different file organizations based on sort order:

#### Sort Order: "Search Term → Rating → Poster #" (`search-rating-poster`)
- **Structure:** Separate file per search term
- **Files:** `{search_term}.md` (e.g., `reinforcement_learning.md`, `transformers.md`)
- **Organization:** Papers grouped by search term (first sort priority)

#### Sort Order: "Rating → Poster # → Search Term" (`rating-poster-search`)
- **Structure:** Separate file per rating level
- **Files:** `{priority}_stars.md` (e.g., `5_stars.md`, `4_stars.md`, `3_stars.md`)
- **Organization:** Papers grouped by rating (first sort priority)
- **Metadata:** Each paper includes its search term for reference

#### Sort Order: "Poster # → Search Term → Rating" (`poster-search-rating`)
- **Structure:** Single file with all papers
- **Files:** `all_papers.md`
- **Organization:** All papers in one file (poster # is first priority)
- **Metadata:** Each paper includes its search term and rating

### 2. Added New Markdown Generation Function (`app.py`)

Created `generate_all_papers_markdown()` to generate comprehensive markdown files with:
- Session grouping
- Full paper metadata (title, authors, rating, search term, poster #)
- Links to PDFs on OpenReview
- Links to poster images
- Full abstracts

### 3. Updated README.md Table of Contents (`app.py`)

Modified `generate_main_readme()` to generate context-aware links:

#### For Search Term Mode
- Table: "Papers by Search Term"
- Links directly to `{search_term}.md` files
- Shows paper count, sessions, and average rating per search term

#### For Rating Mode
- Table: "Papers by Rating"
- Links directly to `{priority}_stars.md` files
- Shows paper count and which search terms are included at each rating level
- Includes additional "Search Terms Summary" table for reference

#### For Poster Mode
- Single link to `all_papers.md`
- Shows "Search Terms Summary" table (without links) for reference
- Notes that all papers are in one file

### 4. Updated Function Documentation

Updated docstrings for:
- `generate_folder_structure_export()`: Documented new file organization behavior
- `generate_all_papers_markdown()`: New function with NumPy-style docstring

## Technical Details

### File Organization Logic

```python
if sort_order == "poster-search-rating":
    # All papers in one file
    zipf.writestr("all_papers.md", markdown)
elif sort_order == "rating-poster-search":
    # Separate files per rating
    zipf.writestr(f"{priority}_stars.md", markdown)
else:  # search-rating-poster
    # Separate files per search term
    zipf.writestr(f"{search_term}.md", markdown)
```

### Link Generation Logic

The main README now includes conditional logic to generate appropriate tables and links based on `sort_order` parameter, ensuring all links point to files that actually exist in the zip archive.

## Benefits

1. **Consistency:** Export structure matches the sort order displayed in the UI
2. **Usability:** Users can quickly find papers organized by their preferred primary attribute
3. **Flexibility:** Different workflows supported (rating-first vs search-term-first vs poster-first)
4. **Navigation:** README.md provides accurate links to all exported files

## Files Modified

- `src/neurips_abstracts/web_ui/app.py`:
  - Modified `generate_folder_structure_export()` (lines ~638-703)
  - Added `generate_all_papers_markdown()` (lines ~538-614)
  - Modified `generate_main_readme()` (lines ~786-844)

## Testing

Manual testing confirmed:
- ✅ Export with "Search Term → Rating → Poster #" creates separate files per search term
- ✅ Export with "Rating → Poster # → Search Term" creates separate files per rating
- ✅ Export with "Poster # → Search Term → Rating" creates single file with all papers
- ✅ All links in README.md work correctly for each mode
- ✅ Papers are sorted correctly within each file according to sort order

## Related Changes

This builds upon:
- Issue #75: Interesting Papers Tab implementation
- Issue #88: Sort order dropdown functionality
- Issue #89: Checkbox removal and simplified interesting papers workflow

---
applyTo: "changelog/**/*.md"
---

# Changelog Instructions

Create markdown files in the `changelog/` directory to document changes made to the NeurIPS Abstracts project. Each file should follow the naming convention `<unique_increasing_number>_<short_description>.md` and include concise information about the changes, including features added, bugs fixed, and any relevant notes for users or developers.

## Finding the Right Number

To determine the next unique increasing number for your changelog file, check the existing files in the `changelog/` directory and identify the highest number used. Increment that number by one for your new file. For example, if the highest existing file is `95_JAVASCRIPT_UNIT_TESTS_FIXED.md`, your new file should start with `96_`.

## Changelog Format

Each changelog entry should include:

1. **Title**: Clear, descriptive heading
2. **Summary**: Brief overview of the change
3. **Details**: Specific changes made
4. **Impact**: How this affects users or developers
5. **Related Issues**: Link to any related issues or PRs (if applicable)

## Example Changelog Entry

```markdown
# Feature: Add New Search Algorithm

## Summary

Implemented a new semantic search algorithm that improves result relevance by 30%.

## Changes

- Added `semantic_search()` function to `embeddings.py`
- Updated test suite with comprehensive search tests
- Added documentation for the new search feature

## Impact

Users will see more relevant search results when querying the paper database.

## Testing

- Added 15 new test cases covering edge cases
- All tests pass with 95% coverage
- Integration tests verify end-to-end functionality
```

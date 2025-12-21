# Editable Search Terms in Interesting Papers Tab

**Date:** December 14, 2025

## Summary

Added the ability to edit search terms for papers in the "Interesting Papers" tab, both for individual papers and for entire groups of papers sharing the same search term.

## Changes Made

### 1. JavaScript Functions (`src/neurips_abstracts/web_ui/static/app.js`)

#### Added `editSearchTerm` Function
- Allows editing search terms for all papers in a group
- Prompts user with current search term
- Updates all papers with the matching search term
- Saves changes to localStorage
- Reloads the interesting papers view

#### Added `editPaperSearchTerm` Function
- Allows editing search terms for individual papers
- Prompts user with current search term for the specific paper
- Updates only the selected paper's search term
- Saves changes to localStorage
- Reloads the interesting papers view

### 2. UI Enhancements

#### Group-Level Edit Button
- Added edit icon button next to search term group headers
- Only visible when sorting by "Search Term → Rating → Poster #"
- Clicking opens a prompt to edit the search term for all papers in that group

#### Individual Paper Edit Button
- Added search term badge to paper cards in the interesting papers tab
- Badge includes paper's search term with an inline edit button
- Edit button allows changing the search term for individual papers
- Badge visibility controlled by sort order:
  - Hidden when sorting by search terms (to avoid redundancy)
  - Shown for other sort orders

### 3. Modified Functions

#### `formatPaperCard`
- Added `showSearchTerm` option to display search term badge
- Search term badge includes:
  - Purple badge with search icon
  - Paper's search term text
  - Inline edit button for individual editing
  - Proper event handling to prevent card click

#### `displayInterestingPapers`
- Added edit button to group headers when sorting by search terms
- Updated `formatPaperCard` calls to show search term badges appropriately
- Different behavior based on sort order

## Features

### Group Editing
1. Click the edit icon next to a search term group header
2. Enter new search term in the prompt
3. All papers in that group (across all sessions) are updated

### Individual Editing
1. Find the search term badge on a paper card
2. Click the edit icon within the badge
3. Enter new search term in the prompt
4. Only that specific paper is updated

### User Experience
- Changes persist in localStorage
- Immediate visual feedback with page reload
- Cancel support (ESC or empty input)
- Trimming of whitespace from input
- Validation to prevent duplicate updates

## Technical Details

### Data Structure
- Search terms stored in `paperPriorities` object
- Each paper has: `{ priority: number, searchTerm: string }`
- Changes synced to localStorage automatically

### Event Handling
- `event.stopPropagation()` prevents card click when editing
- Inline buttons in badges handle their own click events
- Group header buttons prevent propagation to tabs

## Testing Recommendations

1. **Group Editing:**
   - Edit a search term group
   - Verify all papers with that term are updated
   - Check that grouping updates correctly

2. **Individual Editing:**
   - Edit a single paper's search term
   - Verify only that paper is updated
   - Verify it moves to the appropriate group

3. **Sort Order Testing:**
   - Test all three sort orders
   - Verify edit buttons appear/disappear correctly
   - Verify search term badges show appropriately

4. **Edge Cases:**
   - Cancel editing (ESC or empty input)
   - Edit to same value (no-op)
   - Whitespace handling
   - Special characters in search terms

## Benefits

1. **Flexibility:** Users can correct or refine search terms after rating papers
2. **Organization:** Better grouping by correcting mislabeled papers
3. **Workflow:** Supports iterative refinement of paper organization
4. **Bulk Operations:** Group editing saves time when reorganizing
5. **Granular Control:** Individual editing for fine-tuning

## Future Enhancements

Possible improvements:
- Autocomplete for existing search terms
- Merge functionality to combine groups
- Bulk selection for moving papers between groups
- Search term history or suggestions
- Export with edited search terms

# Search Term Grouping Feature

**Date:** 2025-11-29  
**Status:** ✅ Completed

## Overview

Enhanced the paper rating system to store the search term or chat query used when rating each paper. The "Interesting Papers" tab and markdown export now group papers by session first, then by search term within each session, making it easier to organize and review papers from different research areas while maintaining the conference structure.

## Changes Made

### Frontend (`src/neurips_abstracts/web_ui/static/app.js`)

1. **Enhanced State Management**
   - Changed `paperPriorities` structure from `{ paperId: priority }` to `{ paperId: { priority: number, searchTerm: string } }`
   - Added `currentSearchTerm` variable to track the active search/chat query

2. **Search Term Capture**
   - Updated `searchPapers()` to set `currentSearchTerm` when performing a search
   - Updated `sendChatMessage()` to set `currentSearchTerm` when sending a chat message
   - Modified `setPaperPriority()` to store both priority and search term

3. **Display Updates**
   - Refactored `displayInterestingPapers()` to group by session, then by search term
   - Updated sorting to prioritize: session → search term → priority → poster position
   - Enhanced HTML generation with two-level grouping:
     - Top level: Session (white card with green calendar icon)
     - Second level: Search term within each session (blue gradient cards with search icon)

4. **Backward Compatibility**
   - Updated `updateStarDisplay()` to handle new object structure
   - Used optional chaining (`?.`) to safely access priority from new format

### Backend (`src/neurips_abstracts/web_ui/app.py`)

1. **Data Handling**
   - Updated `/api/export/interesting-papers` endpoint to handle new priorities format
   - Added backward compatibility for old integer-only priority format
   - Extract both `priority` and `searchTerm` from the priorities dictionary

2. **Sorting Enhancement**
   - Updated paper sorting to include search term as secondary sort key
   - Sort order: `session → searchTerm → -priority → poster_position`

3. **Markdown Generation**
   - Refactored `generate_markdown_with_assets()` to group papers by session first
   - Changed grouping structure from single-level (sessions) to two-level (sessions → search terms)
   - Updated markdown headings:
     - H2 (`##`) for sessions
     - H3 (`###`) for search terms within each session
     - H4 (`####`) for paper titles

## User Experience Improvements

### Before
- Papers were grouped only by session
- No indication of which search/query led to rating a paper
- Difficult to track papers from different research areas

### After
- Papers clearly organized by session, maintaining conference structure
- Within each session, papers grouped by the search term or chat query used
- Easy to see which searches yielded the most interesting papers per session
- Better organization for reviewing papers across multiple topics within sessions
- Visual hierarchy in the UI:
  - Sessions in white cards with green calendar icons
  - Search terms in blue gradient cards with search icons (within each session)
  - Papers with star ratings within each search term

## Example Structure

```text
Interesting Papers Tab:
├── � Poster Session 1
│   ├── �🔍 "neural architecture search"
│   │   ├── ⭐⭐⭐⭐⭐ Paper A
│   │   └── ⭐⭐⭐⭐ Paper B
│   └── � "reinforcement learning"
│       └── ⭐⭐⭐⭐⭐ Paper D
├── � Poster Session 2
│   └── � "neural architecture search"
│       └── ⭐⭐⭐ Paper C
└── 📅 Oral Session 3
    └── 🔍 "reinforcement learning"
        └── ⭐⭐⭐⭐ Paper E
```

## Markdown Export Format

```markdown
# Interesting Papers from NeurIPS 2025

Generated: 2025-11-29 14:30:00
**Total Papers:** 5

---

## Poster Session 1

### neural architecture search

#### Paper A Title
**Rating:** ⭐⭐⭐⭐⭐ (5/5)
...

#### Paper B Title
**Rating:** ⭐⭐⭐⭐ (4/5)
...

### reinforcement learning

#### Paper D Title
**Rating:** ⭐⭐⭐⭐⭐ (5/5)
...

## Poster Session 2

### neural architecture search

#### Paper C Title
**Rating:** ⭐⭐⭐ (3/5)
...
```

## Technical Details

### Data Migration
- Old localStorage data (integer priorities) automatically handled
- When old format detected: `paper['searchTerm'] = search_query or 'Unknown'`
- New ratings store: `{ priority: number, searchTerm: string }`

### Sorting Logic

```javascript
// Frontend
papers.sort((a, b) => {
    const sessionCompare = (a.session || '').localeCompare(b.session || '');
    if (sessionCompare !== 0) return sessionCompare;
    
    const searchTermCompare = (a.searchTerm || '').localeCompare(b.searchTerm || '');
    if (searchTermCompare !== 0) return searchTermCompare;
    
    if (a.priority !== b.priority) return b.priority - a.priority;
    
    return (a.poster_position || '').localeCompare(b.poster_position || '');
});
```

```python
# Backend
papers.sort(key=lambda p: (
    p.get('session') or '',
    p.get('searchTerm') or '',
    -p.get('priority', 0),
    p.get('poster_position') or ''
))
```

## Testing Recommendations

1. **New Ratings**
   - Search for a topic (e.g., "transformers")
   - Rate some papers with stars
   - Switch to Interesting Papers tab
   - Verify papers grouped under "transformers" search term

2. **Multiple Search Terms**
   - Search for different topics and rate papers from each
   - Verify clear separation between search term groups
   - Check proper sorting within each group

3. **Chat Integration**
   - Ask questions in chat
   - Rate papers from relevant papers list
   - Verify papers grouped by the chat question

4. **Export**
   - Export interesting papers as markdown
   - Verify H1 headings for search terms
   - Verify H2 headings for sessions
   - Verify proper hierarchical structure

5. **Backward Compatibility**
   - If you have old ratings, verify they still work
   - Old ratings should show "Unknown" as search term

## Benefits

1. **Better Organization**: Papers organized by session (conference structure) with search term sub-grouping
2. **Context Preservation**: Remember why you rated a paper within each session
3. **Multi-Topic Research**: Easy to track papers from different searches within the same session
4. **Conference Structure**: Maintains session organization for planning conference attendance
5. **Export Clarity**: Markdown file shows session structure with search context
6. **Review Efficiency**: Navigate by session, then see which searches led to interesting papers

## Files Modified

- `src/neurips_abstracts/web_ui/static/app.js` - Frontend logic and display
- `src/neurips_abstracts/web_ui/app.py` - Backend API and markdown generation

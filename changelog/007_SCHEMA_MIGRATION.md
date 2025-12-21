# Database Schema Migration Guide

## Schema Changes

The database schema has been updated to match the actual NeurIPS 2025 JSON structure. Here are the key changes:

### Field Name Mappings

| Old Field Name      | New Field Name | Notes                              |
| ------------------- | -------------- | ---------------------------------- |
| `paper_id`          | `id`           | Primary key (TEXT)                 |
| `title`             | `name`         | Paper title                        |
| `track`             | `eventtype`    | Type of event (Poster, Oral, etc.) |
| `type`              | `eventtype`    | Unified as eventtype               |
| `presentation_type` | `eventtype`    | Unified as eventtype               |

### New Fields Added

The schema now includes all fields from the actual NeurIPS JSON:

- `uid` - Unique identifier hash
- `topic` - Research topic (e.g., "General Machine Learning->Representation Learning")
- `decision` - Acceptance decision (e.g., "Accept (poster)")
- `event_type` - Event type template
- `room_name` - Physical location
- `virtualsite_url` - Virtual conference URL
- `paper_url` - OpenReview URL
- `paper_pdf_url` - PDF URL
- `starttime` / `endtime` - Event scheduling
- `poster_position` - Physical poster location
- `eventmedia` - Media files (JSON array)
- `related_events` - Related sessions (JSON array)
- And many more...

### Complete Schema

```sql
CREATE TABLE IF NOT EXISTS papers (
    id TEXT PRIMARY KEY,              -- Paper ID (string or numeric)
    uid TEXT,                          -- Unique hash identifier  
    name TEXT NOT NULL,                -- Paper title
    authors TEXT,                      -- Comma-separated authors
    abstract TEXT,                     -- Paper abstract
    topic TEXT,                        -- Research topic
    keywords TEXT,                     -- Comma-separated keywords
    decision TEXT,                     -- Acceptance decision
    session TEXT,                      -- Session name
    eventtype TEXT,                    -- Event type (Poster/Oral)
    event_type TEXT,                   -- Event type template
    room_name TEXT,                    -- Physical location
    virtualsite_url TEXT,              -- Virtual URL
    url TEXT,                          -- General URL
    sourceid INTEGER,                  -- Source ID
    sourceurl TEXT,                    -- Source URL
    starttime TEXT,                    -- Event start time (ISO format)
    endtime TEXT,                      -- Event end time (ISO format)
    starttime2 TEXT,                   -- Secondary start time
    endtime2 TEXT,                     -- Secondary end time
    diversity_event TEXT,              -- Diversity event flag
    paper_url TEXT,                    -- Paper URL (OpenReview)
    paper_pdf_url TEXT,                -- PDF URL
    children_url TEXT,                 -- Children URL
    children TEXT,                     -- Children data (JSON)
    children_ids TEXT,                 -- Children IDs (JSON)
    parent1 TEXT,                      -- Parent event 1
    parent2 TEXT,                      -- Parent event 2
    parent2_id TEXT,                   -- Parent 2 ID
    eventmedia TEXT,                   -- Event media (JSON)
    show_in_schedule_overview INTEGER, -- Show in schedule (0/1)
    visible INTEGER,                   -- Visibility (0/1)
    poster_position TEXT,              -- Poster position
    schedule_html TEXT,                -- Schedule HTML
    latitude REAL,                     -- Location latitude
    longitude REAL,                    -- Location longitude
    related_events TEXT,               -- Related events (JSON)
    related_events_ids TEXT,           -- Related event IDs (JSON)
    raw_data TEXT,                     -- Complete raw JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

### Indexes

```sql
CREATE INDEX IF NOT EXISTS idx_uid ON papers(uid);
CREATE INDEX IF NOT EXISTS idx_id ON papers(id);
CREATE INDEX IF NOT EXISTS idx_decision ON papers(decision);
CREATE INDEX IF NOT EXISTS idx_topic ON papers(topic);
CREATE INDEX IF NOT EXISTS idx_eventtype ON papers(eventtype);
CREATE INDEX IF NOT EXISTS idx_session ON papers(session);
```

## Backward Compatibility

The `load_json_data()` function maintains backward compatibility by:

1. Accepting both `title` and `name` fields (maps `title` → `name`)
2. Accepting both `type` and `eventtype` fields (maps `type` → `eventtype`)
3. Converting numeric IDs to strings
4. Handling both list and string formats for authors/keywords

The `search_papers()` function maintains backward compatibility by:

1. Supporting the deprecated `track` parameter (maps to `eventtype`)
2. Searching across both old and new field names

## Example Queries

### Old Style (Still Works)
```python
# Using deprecated 'track' parameter
results = db.search_papers(track="poster")
```

### New Style (Recommended)
```python
# Using new parameter names
results = db.search_papers(eventtype="Poster")
results = db.search_papers(decision="Accept (oral)")
results = db.search_papers(topic="Machine Learning")
```

### Accessing Fields
```python
for paper in results:
    print(paper['name'])          # Use 'name' instead of 'title'
    print(paper['eventtype'])     # Use 'eventtype' instead of 'track'
    print(paper['decision'])      # New field
    print(paper['topic'])          # New field
    print(paper['paper_url'])      # New field
```

## Migration for Existing Code

If you have existing code using the old schema:

1. **Replace `title` with `name`:**
   ```python
   # Old
   print(paper['title'])
   
   # New
   print(paper['name'])
   ```

2. **Replace `track` with `eventtype`:**
   ```python
   # Old
   results = db.search_papers(track="oral")
   print(paper['track'])
   
   # New  
   results = db.search_papers(eventtype="Oral")
   print(paper['eventtype'])
   ```

3. **Replace `paper_id` with `id`:**
   ```python
   # Old
   db.query("SELECT * FROM papers WHERE paper_id = ?", ("123",))
   
   # New
   db.query("SELECT * FROM papers WHERE id = ?", ("123",))
   ```

## Real NeurIPS Data Example

```python
from neurips_abstracts import download_neurips_data, DatabaseManager

# Download actual NeurIPS 2025 data
data = download_neurips_data(2025)

# Load into database
with DatabaseManager("neurips.db") as db:
    db.create_tables()
    count = db.load_json_data(data)
    print(f"Loaded {count} papers")
    
    # Search by decision type
    oral_papers = db.search_papers(decision="Accept (oral)")
    print(f"Oral presentations: {len(oral_papers)}")
    
    # Search by topic
    ml_papers = db.search_papers(topic="Machine Learning")
    
    # Access new fields
    for paper in oral_papers[:5]:
        print(f"Title: {paper['name']}")
        print(f"Decision: {paper['decision']}")
        print(f"Topic: {paper['topic']}")
        print(f"Session: {paper['session']}")
        print(f"Room: {paper['room_name']}")
        print(f"Time: {paper['starttime']} - {paper['endtime']}")
        print(f"URL: {paper['paper_url']}")
        print()
```

## Summary

The schema now fully supports the actual NeurIPS conference data structure while maintaining backward compatibility for test code and simple use cases. New features include:

- ✅ Full NeurIPS JSON schema support
- ✅ Event scheduling information
- ✅ Location data (room, position)
- ✅ Decision and topic filtering
- ✅ URL references (paper, PDF, virtual site)
- ✅ Backward compatibility with old field names
- ✅ Flexible ID handling (string or numeric)

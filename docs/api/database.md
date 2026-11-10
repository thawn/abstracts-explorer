# Database Module

The database module provides an interface for storing and retrieving conference papers using SQLAlchemy, supporting both SQLite and PostgreSQL backends.

## Overview

The `DatabaseManager` class handles all database operations including:

- Creating and managing the database schema
- Adding papers and paper metadata
- Searching and filtering papers
- Storing embeddings metadata, clustering caches, validation data, and evaluation results

## Class Reference

```{eval-rst}
.. automodule:: abstracts_explorer.database
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: datetime, timezone, Path, Any, Dict, List, Optional, Union, create_engine, select, delete, func, or_, and_, text, sessionmaker, Session, Engine, OperationalError, ProgrammingError, IntegrityError, logger
```

## Usage Examples

### Basic Operations

`DatabaseManager` reads its connection settings from the application configuration
(`PAPER_DB`). Papers are added as validated `LightweightPaper` objects.

```python
from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.plugin import LightweightPaper

db = DatabaseManager()
with db:
    db.create_tables()

    paper = LightweightPaper(
        title="Example Paper",
        authors=["John Doe", "Jane Smith"],
        abstract="This is an example abstract.",
        session="Poster Session 1",
        poster_position="P1",
        year=2025,
        conference="NeurIPS",
        paper_pdf_url="https://example.com/paper.pdf",
    )
    paper_uid = db.add_paper(paper)
```

### Searching Papers

```python
# Keyword search across title, abstract, and keywords
results = db.search_papers(keyword="transformer")

# Field-specific substring search
results = db.search_papers(field_filters={"title": "attention"})
results = db.search_papers(field_filters={"authors": "John Doe"})

# Filter by year and/or conference
papers_2025 = db.search_papers(years=[2025])
neurips_2025 = db.search_papers(conferences=["NeurIPS"], years=[2025])

# Structured filters with get_paper, the single paper-search authority
results = db.get_paper(
    {
        "$and": [
            {"year": 2025},
            {"conference": "NeurIPS"},
        ]
    }
)
```

### Retrieving Data

```python
# Get all papers (limit=0 disables the default limit)
all_papers = db.get_paper(limit=0)

# Get a specific paper by UID
paper = db.get_paper_by_uid(paper_uid)

# Get multiple papers by UID
papers = db.get_papers_by_uids([paper_uid, another_uid])

# Authors are stored on the paper record as a semicolon-separated string
authors = paper["authors"] if paper else None
```

## Database Schema

### papers Table

| Column           | Type      | Description                                  |
| ---------------- | --------- | -------------------------------------------- |
| uid              | TEXT      | Primary key (hash-based unique identifier)   |
| original_id      | TEXT      | Original ID from the source                  |
| title            | TEXT      | Paper title                                  |
| authors          | TEXT      | Semicolon-separated author names             |
| abstract         | TEXT      | Paper abstract                               |
| session          | TEXT      | Conference session name                      |
| poster_position  | TEXT      | Poster position identifier                   |
| paper_pdf_url    | TEXT      | URL to paper PDF                             |
| poster_image_url | TEXT      | URL to poster image                          |
| url              | TEXT      | General URL for the paper                    |
| room_name        | TEXT      | Room name for presentation                   |
| keywords         | TEXT      | Comma-separated keywords                     |
| starttime        | TEXT      | Presentation start time                      |
| endtime          | TEXT      | Presentation end time                        |
| award            | TEXT      | Award received (e.g. "Best Paper")           |
| year             | INTEGER   | Publication year                             |
| conference       | TEXT      | Conference name (e.g. "NeurIPS", "ML4PS")    |
| created_at       | TIMESTAMP | Creation timestamp                           |

Additional tables store embeddings metadata, clustering caches, validation data,
chat donations, and evaluation data. See `abstracts_explorer.db_models` for the
full model definitions.

## Error Handling

Database operations raise `DatabaseError`:

```python
from abstracts_explorer.database import DatabaseError

try:
    paper_uid = db.add_paper(paper)
except DatabaseError as exc:
    print(f"Database error: {exc}")
```

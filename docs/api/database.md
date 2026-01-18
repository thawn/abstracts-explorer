# Database Module

The database module provides an interface for storing and retrieving conference papers in SQLite.

## Overview

The `DatabaseManager` class handles all database operations including:

- Creating and managing database schema
- Adding papers and authors
- Searching and filtering papers
- Managing paper-author relationships

## Class Reference

```{eval-rst}
.. automodule:: abstracts_explorer.database
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
```

## Usage Examples

### Basic Operations

```python
from abstracts_explorer.database import DatabaseManager

# Initialize database
db = DatabaseManager()

# Add a paper
paper_id = db.add_paper({
    'openreview_id': 'abc123',
    'title': 'Example Paper',
    'abstract': 'This is an example abstract.',
    'year': 2025,
    'pdf_url': 'https://example.com/paper.pdf'
})

# Add authors
db.add_author(paper_id, 'John Doe', 0)
db.add_author(paper_id, 'Jane Smith', 1)
```

### Searching Papers

```python
# Search by title
results = db.search_papers(title="transformer")

# Search by abstract content
results = db.search_papers(abstract="attention mechanism")

# Filter by year
results = db.get_papers_by_year(2025)

# Get papers by author
results = db.get_papers_by_author("John Doe")
```

### Retrieving Data

```python
# Get all papers
all_papers = db.get_all_papers()

# Get specific paper
paper = db.get_paper_by_id(paper_id)

# Get authors for a paper
authors = db.get_authors_for_paper(paper_id)
```

## Database Schema

### papers Table

| Column        | Type      | Description                  |
| ------------- | --------- | ---------------------------- |
| id            | INTEGER   | Primary key (auto-increment) |
| openreview_id | TEXT      | Unique OpenReview ID         |
| title         | TEXT      | Paper title                  |
| abstract      | TEXT      | Paper abstract               |
| year          | INTEGER   | Conference year              |
| pdf_url       | TEXT      | URL to PDF                   |
| created_at    | TIMESTAMP | Creation timestamp           |

### authors Table

| Column   | Type    | Description                  |
| -------- | ------- | ---------------------------- |
| id       | INTEGER | Primary key (auto-increment) |
| paper_id | INTEGER | Foreign key to papers        |
| name     | TEXT    | Author name                  |
| position | INTEGER | Author position in list      |

## Error Handling

The database module raises standard SQLite exceptions. Wrap operations in try-except blocks:

```python
try:
    db.add_paper(paper_data)
except sqlite3.IntegrityError:
    print("Paper already exists")
except Exception as e:
    print(f"Database error: {e}")
```

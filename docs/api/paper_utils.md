# Paper Utilities Module

The paper utilities module provides shared helper functions for formatting papers
from various sources (database, search results, ChromaDB) with consistent
structure and error handling.

## Quick Start

```python
from abstracts_explorer.paper_utils import (
    get_paper_with_authors,
    format_search_results,
    build_context_from_papers,
)
from abstracts_explorer.database import DatabaseManager

db = DatabaseManager()

# Get a paper with its authors
paper = get_paper_with_authors(db, paper_uid="abc123")

# Format search results for display
formatted = format_search_results(search_results, db)

# Build a context string for RAG
context = build_context_from_papers(papers)
```

## API Reference

```{eval-rst}
.. automodule:: abstracts_explorer.paper_utils
   :members:
   :undoc-members:
   :show-inheritance:
```

# Paper Utilities Module

The paper utilities module provides shared helper functions for formatting papers
from various sources (database, search results, ChromaDB) with consistent
structure and error handling.

## Quick Start

```python
from abstracts_explorer.paper_utils import (
    format_search_results,
    build_context_from_papers,
)
from abstracts_explorer.database import DatabaseManager

db = DatabaseManager()

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

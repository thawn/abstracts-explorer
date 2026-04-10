# Export Utilities Module

The export utilities module provides functions for generating markdown documents
and ZIP archives from paper collections.

## Features

- Natural sorting of strings with embedded numbers
- Markdown generation for paper lists and search results
- ZIP archive creation with organized folder structures
- Conference information fetching

## Quick Start

```python
from abstracts_explorer.export_utils import (
    export_papers_to_zip,
    generate_all_papers_markdown,
)

# Generate a markdown document listing all papers
markdown = generate_all_papers_markdown(papers, title="Conference Papers")

# Export papers and search results to a ZIP archive
zip_buffer = export_papers_to_zip(
    papers=papers,
    search_results=search_results,
    title="Conference Papers"
)

# Save the ZIP file
with open("papers.zip", "wb") as f:
    f.write(zip_buffer.getvalue())
```

## API Reference

```{eval-rst}
.. automodule:: abstracts_explorer.export_utils
   :members:
   :undoc-members:
   :show-inheritance:
```

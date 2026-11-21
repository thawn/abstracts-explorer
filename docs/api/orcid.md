# ORCID Module

The ORCID module fetches a researcher's public ORCID record and turns its works
and keywords into concise paper search terms. The MCP tool
`suggest_search_terms_from_orcid` is a thin wrapper around this module.

## Features

- Normalize ORCID iDs and profile URLs
- Fetch public works, author keywords, and display name
- Generate search terms with the configured chat model
- Fall back to ORCID keywords and TF-IDF title phrases when the LLM is unavailable
- Cache public records in memory for the configured metadata-cache TTL

## Quick Start

```python
from abstracts_explorer.orcid import suggest_search_terms_from_orcid

result = suggest_search_terms_from_orcid("0000-0002-1825-0097", max_terms=10)
for item in result["suggested_terms"]:
    print(item["term"], "-", item["reason"])
```

## API Reference

```{eval-rst}
.. automodule:: abstracts_explorer.orcid
   :members:
   :undoc-members:
   :show-inheritance:
```

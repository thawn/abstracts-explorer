# MCP Tools Module

The MCP tools module provides integration between MCP clustering tools and the
RAG chat system. It normalizes tool arguments to absorb common LLM output
quirks and formats tool execution results for LLM consumption.

## Overview

This module enables the LLM to automatically decide when to use clustering
analysis tools to answer questions about conference topics, trends, and
developments during RAG chat sessions. It also exposes
`summarize_interesting_papers`, `recommend_related_papers`, and
`suggest_authors_to_meet`, which use the paper ratings forwarded from the web
UI, and `suggest_search_terms_from_orcid`, which reads a public ORCID record.

## Quick Start

```python
from abstracts_explorer.mcp_tools import (
    execute_mcp_tool,
    format_tool_result_for_llm,
)

# Execute a tool
result = execute_mcp_tool(
    "get_conference_topics",
    {"conferences": ["NeurIPS"], "years": [2025]},
)

# Format the result for LLM consumption
formatted = format_tool_result_for_llm("get_conference_topics", result)
```

## Conference topics

`get_conference_topics` reads the cached BERTopic hierarchy and uses fixed
level 6 (clamped to the available levels). It returns the level’s parent
labels, aggregated keywords, paper counts, and sample titles without loading
embeddings. The web UI independently defaults to level 4.

## API Reference

```{eval-rst}
.. automodule:: abstracts_explorer.mcp_tools
   :members:
   :undoc-members:
   :show-inheritance:
```

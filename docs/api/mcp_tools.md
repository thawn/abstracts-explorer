# MCP Tools Module

The MCP tools module provides integration between MCP clustering tools and the
RAG chat system. It converts MCP tool definitions to OpenAI function calling
format and handles tool execution.

## Overview

This module enables the LLM to automatically decide when to use clustering
analysis tools to answer questions about conference topics, trends, and
developments during RAG chat sessions.

## Quick Start

```python
from abstracts_explorer.mcp_tools import (
    get_mcp_tools_schema,
    execute_mcp_tool,
    format_tool_result_for_llm,
)

# Get tool schemas for OpenAI function calling
schemas = get_mcp_tools_schema()

# Execute a tool
result = execute_mcp_tool(
    "get_conference_topics",
    {"n_clusters": 8}
)

# Format the result for LLM consumption
formatted = format_tool_result_for_llm("get_conference_topics", result)
```

## API Reference

```{eval-rst}
.. automodule:: abstracts_explorer.mcp_tools
   :members:
   :undoc-members:
   :show-inheritance:
```

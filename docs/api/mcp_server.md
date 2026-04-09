# MCP Server Module

The MCP server module provides a Model Context Protocol (MCP) server that exposes
tools for analyzing clustered conference paper embeddings. It enables LLM-based
assistants to answer questions about conference paper topics, trends, and developments.

## Features

- Analyze frequently mentioned topics from clustered papers
- Track topic evolution across conference years
- Search for papers related to specific topics
- Generate cluster visualization data
- Analyze topic relevance across conferences

## Quick Start

```python
from abstracts_explorer.mcp_server import (
    get_conference_topics,
    get_topic_evolution,
    search_papers,
)

# Get main topics across conferences
topics = get_conference_topics(n_clusters=8)

# Track evolution of a topic
evolution = get_topic_evolution(
    topic_keywords="transformer attention",
    start_year=2020,
    end_year=2025,
)

# Search for papers on a topic
results = search_papers(
    topic_keywords="reinforcement learning",
    n_results=10,
)
```

See the [MCP Server Guide](../mcp_server.md) for usage with LLM assistants.

## API Reference

```{eval-rst}
.. automodule:: abstracts_explorer.mcp_server
   :members:
   :undoc-members:
   :show-inheritance:
```

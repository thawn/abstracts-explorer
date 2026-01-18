# Abstracts Explorer Documentation

Welcome to the documentation for Abstracts Explorer! This package provides tools for downloading, storing, and analyzing NeurIPS conference paper abstracts.

## Features

- **Download NeurIPS abstracts** from the official OpenReview API
- **Plugin system** for downloading from workshops and other conferences
- **Store abstracts** in a SQLite database with full metadata
- **Create vector embeddings** for semantic search
- **Cluster and visualize** paper embeddings with multiple algorithms
- **MCP server** for LLM-based cluster analysis and topic exploration
- **RAG (Retrieval-Augmented Generation)** chat interface for querying papers
- **Web interface** for browsing and searching papers
- **Command-line interface** for easy interaction
- **Configuration system** with .env file support

## Quick Start

Install the package:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the package with all dependencies
uv sync --all-extras
```

Download abstracts for NeurIPS 2025:

```bash
uv run abstracts-explorer download --year 2025
```

Or download from a workshop using plugins:

```bash
uv run abstracts-explorer download --plugin ml4ps --year 2025
```

Create embeddings for semantic search:

```bash
uv run abstracts-explorer create-embeddings
```

Search papers:

```bash
uv run abstracts-explorer search "machine learning"
```

Chat with papers using RAG:

```bash
uv run abstracts-explorer chat
```

## Documentation Contents

```{toctree}
:maxdepth: 2
:caption: User Guide

installation
docker
configuration
usage
plugins
cli_reference
mcp_server
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/modules
api/database
api/embeddings
api/rag
api/config
```

```{toctree}
:maxdepth: 1
:caption: Development

contributing
```

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`

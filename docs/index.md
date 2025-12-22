# NeurIPS Abstracts Documentation

Welcome to the documentation for the NeurIPS Abstracts package! This package provides tools for downloading, storing, and analyzing NeurIPS conference paper abstracts.

## Features

- **Download NeurIPS abstracts** from the official OpenReview API
- **Plugin system** for downloading from workshops and other conferences
- **Store abstracts** in a SQLite database with full metadata
- **Create vector embeddings** for semantic search
- **RAG (Retrieval-Augmented Generation)** chat interface for querying papers
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
uv run neurips-abstracts download --year 2025 --db-path data/neurips_2025.db
```

Or download from a workshop using plugins:

```bash
uv run neurips-abstracts download --plugin ml4ps --year 2025 --db-path data/ml4ps.db
```

Create embeddings for semantic search:

```bash
uv run neurips-abstracts create-embeddings --db-path data/neurips_2025.db
```

Search papers:

```bash
uv run neurips-abstracts search "machine learning" --db-path data/neurips_2025.db
```

Chat with papers using RAG:

```bash
uv run neurips-abstracts chat --db-path data/neurips_2025.db
```

## Documentation Contents

```{toctree}
:maxdepth: 2
:caption: User Guide

installation
configuration
usage
plugins
cli_reference
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/modules
api/database
api/downloader
api/embeddings
api/rag
api/config
```

```{toctree}
:maxdepth: 1
:caption: Development

changelog
contributing
```

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`

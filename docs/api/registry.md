# Registry Module

The registry module provides functionality to upload and download Abstracts Explorer
data artifacts to and from OCI-compatible container registries such as GitHub Container
Registry (`ghcr.io`).

## Overview

Data artifacts include:

- **Paper database** — conference papers with metadata
- **Embeddings** — ChromaDB vector embeddings
- **Clustering cache** — pre-computed clustering results

Artifacts are tagged by conference, year, and embedding model (e.g.,
`neurips-2024_text-embedding-qwen3-embedding-4b`).

## Quick Start

```python
from abstracts_explorer.registry import RegistryClient

# Initialize client
client = RegistryClient(
    repository="ghcr.io/thawn/abstracts-data",
    token="ghp_xxxxxxxxxxxxxxxxxxxx"
)

# List available tags
tags = client.list_tags()
print(tags)

# Download data for a specific conference and year
client.download(conference="neurips", year=2024)

# Upload local data
client.upload(conference="neurips", year=2024)
```

See the [Registry Guide](../registry.md) for CLI usage and full documentation.

## API Reference

```{eval-rst}
.. automodule:: abstracts_explorer.registry
   :members:
   :undoc-members:
   :show-inheritance:
```

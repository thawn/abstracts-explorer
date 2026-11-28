# Usage Guide

This guide covers common usage patterns for Abstracts Explorer.

## Basic Workflow

### 1. Download Papers

Download papers for a specific conference and year:

```bash
# Download from the default conference (NeurIPS)
uv run abstracts-explorer data download --year 2025

# Download from a specific conference using a plugin
uv run abstracts-explorer data download --conference iclr --year 2025
```

Alternatively, download pre-built data from a registry (no API access or LLM backend required):

```bash
uv run abstracts-explorer registry pull \
  -r ghcr.io/thawn/abstracts-data \
  --conference neurips --year 2025
```

Options:
- `--conference`: Conference to download (default: all conferences)
- `--year`: Conference year (e.g., 2025; default: all available years)
- `--force`: Force re-download even if papers already exist
- `--list-plugins`: List available downloader plugins and exit

#### Generated keywords

Each paper has a `generated_keywords` property containing TF-IDF keywords
derived from its title and abstract. Keywords are generated per
`(conference, year)` corpus and stored in the database. Downloads generate
keywords automatically for each downloaded unit.

Existing databases can be backfilled once with the script-only helper:

```bash
# Preview how many papers would be updated
uv run python scripts/backfill_generated_keywords.py --dry-run --json

# Fill missing keywords for all conferences and years
uv run python scripts/backfill_generated_keywords.py

# Limit the backfill to a single scope, or regenerate existing values
uv run python scripts/backfill_generated_keywords.py --conference NeurIPS --year 2025
uv run python scripts/backfill_generated_keywords.py --force
```

On staging (Podman), production images do not include `scripts/`. Stream the
script from the matching repository ref and run it inside the container:

```bash
podman exec -i abstracts-explorer sh -c '
  curl -fsSL https://raw.githubusercontent.com/thawn/abstracts-explorer/<tag-or-branch>/scripts/backfill_generated_keywords.py \
  | python - --dry-run
'
```

See [Run repository helper scripts](docker.md#run-repository-helper-scripts) for
details. The script adds the `generated_keywords` column when it is missing and
is safe to re-run.

The source `keywords` field supplied by downloaders is still stored, but it is
**not** used for topic extraction. Runtime related-topic aggregation uses
`generated_keywords`; TF-IDF extraction remains a fallback for inputs that do
not carry generated keywords (for example ORCID works).

### 2. Create Embeddings

Generate vector embeddings for semantic search:

```bash
uv run abstracts-explorer embeddings create
```

Options:
- `--conference`: Only embed papers from this conference
- `--year`: Only embed papers from this year
- `--collection`: Collection name in ChromaDB (default: `papers`)
- `--force`: Reset collection if it already exists
- `--ignore-embedding-model-mismatch`: Recreate embeddings after an
  embedding model mismatch without prompting

### 3. Search Papers

Search papers by keyword or semantic similarity:

```bash
# Simple search
uv run abstracts-explorer papers search "transformer architecture"

# Limit results
uv run abstracts-explorer papers search "reinforcement learning" --limit 10

# Filter by year
uv run abstracts-explorer papers search "neural networks" --year 2025
```

### 4. Chat with Papers (RAG)

Interactive chat interface powered by RAG:

```bash
uv run abstracts-explorer papers chat
```

In the chat interface:
- Ask questions about papers
- Get AI-generated responses with paper references
- Type `exit` or `quit` to leave

[here is an example chat](chat_example.md)

To enable the optional offline input guard, run Ollaya locally and set
`GUARD_ENABLED=true`; see the
[local Ollaya guard setup](contributing.md#5-optional-local-ollaya-guard) and
[configuration](configuration.md#chat-input-guard-ollaya).

## Python API

### Database Operations

```python
from abstracts_explorer.database import DatabaseManager

# Open database (connection settings come from PAPER_DB)
db = DatabaseManager()

# Get all papers (limit=0 disables the default limit)
papers = db.get_paper(limit=0)

# Keyword search across title, abstract, and keywords
results = db.keyword_search(keyword="transformer")

# Field-specific substring search
title_matches = db.keyword_search(field_filters={"title": "attention"})

# Filter by year
papers_2025 = db.keyword_search(years=[2025])

# Authors are stored on the paper record as a semicolon-separated string
authors = papers[0]["authors"] if papers else None
```

### Downloading Papers

Use the plugin system to download papers from different conferences:

```python
from abstracts_explorer.plugins import get_plugin
from abstracts_explorer import DatabaseManager
from abstracts_explorer.plugin import LightweightPaper

# Get a conference plugin (e.g. neurips, iclr, icml, ml4ps)
plugin = get_plugin('neurips')

# Download papers for a specific year
papers_data = plugin.download(year=2025, output_path='papers_2025.json')

# Convert to LightweightPaper objects
papers = [LightweightPaper(**paper) for paper in papers_data]

# Save to database
db = DatabaseManager()
db.create_tables()
db.add_papers(papers)
```

### Embeddings

```python
from abstracts_explorer.embeddings import EmbeddingsManager

# Initialize embeddings manager
em = EmbeddingsManager(
    collection_name="papers"
)

# Create embeddings from database
em.embed_from_database()

# Search by semantic similarity
from abstracts_explorer.embeddings import QueryEmbedding

query = QueryEmbedding("transformer attention mechanism", em)
results = em.search_similar(query, limit=5)

# Search with metadata filter
from abstracts_explorer.database import DatabaseManager

db = DatabaseManager()
uids = db.resolve_paper_uids(years=[2025])
results = em.search_similar(
    QueryEmbedding("deep learning", em),
    limit=10,
    ids=uids,
)
```

### RAG Chat

```python
from abstracts_explorer.embeddings import EmbeddingsManager
from abstracts_explorer.rag import RAGChat

# Initialize
em = EmbeddingsManager()

chat = RAGChat(
    em,
    backend_url="http://localhost:1234",
    model="diffbot-small-xl-2508"
)

# Ask a question
response = chat.query("What are the latest developments in transformers?")
print(response)

# Continue conversation
response = chat.query("Tell me more about the first paper")
print(response)

# Export conversation
chat.export_conversation("conversation.json")

# Reset conversation
chat.reset_conversation()
```

## Advanced Usage

### Batch Processing

Process multiple years:

```bash
#!/bin/bash
for year in 2023 2024 2025; do
    uv run abstracts-explorer data download --year $year
    uv run abstracts-explorer embeddings create
done
```

### Custom Configuration

Use a custom configuration file:

```python
import os
os.environ['PAPER_DB'] = 'custom_papers.db'
os.environ['EMBEDDING_DB'] = 'custom_embeddings'

from abstracts_explorer.config import get_config
config = get_config()
```

### Programmatic Search

```python
from abstracts_explorer.database import DatabaseManager

db = DatabaseManager()

# Complex search with multiple filters
papers = db.keyword_search(
    title="learning",
    abstract="neural network",
    year=2025,
    limit=20
)

for paper in papers:
    print(f"{paper['title']} - {paper['year']}")
```

# Usage Guide

This guide covers common usage patterns for Abstracts Explorer.

## Basic Workflow

### 1. Download Papers

Download papers for a specific year:

```bash
uv run abstracts-explorer download --year 2025
```

Options:
- `--year`: Conference year (e.g., 2025)
- `--db-path`: Path to SQLite database (will be created if doesn't exist)
- `--force`: Force re-download even if papers already exist

### 2. Create Embeddings

Generate vector embeddings for semantic search:

```bash
uv run abstracts-explorer create-embeddings
```

Options:
- `--db-path`: Path to SQLite database with papers
- `--embedding-db-path`: Path to ChromaDB database (default: from config)
- `--collection-name`: Collection name in ChromaDB (default: from config)
- `--force`: Recreate embeddings even if they exist

### 3. Search Papers

Search papers by keyword or semantic similarity:

```bash
# Simple search
uv run abstracts-explorer search "transformer architecture"

# Limit results
uv run abstracts-explorer search "reinforcement learning" --limit 10

# Filter by year
uv run abstracts-explorer search "neural networks" --year 2025
```

### 4. Chat with Papers (RAG)

Interactive chat interface powered by RAG:

```bash
uv run abstracts-explorer chat
```

In the chat interface:
- Ask questions about papers
- Get AI-generated responses with paper references
- Type `exit` or `quit` to leave

## Python API

### Database Operations

```python
from abstracts_explorer.database import DatabaseManager

# Open database
db = DatabaseManager()

# Get all papers
papers = db.get_all_papers()

# Search by title
results = db.search_papers(title="transformer")

# Get papers by year
papers_2025 = db.get_papers_by_year(2025)

# Get authors for a paper
authors = db.get_authors_for_paper(paper_id)
```

### Downloading Papers

Use the plugin system to download papers from different conferences:

```python
from abstracts_explorer.plugins import get_plugin
from abstracts_explorer import DatabaseManager
from abstracts_explorer.plugin import LightweightPaper

# Get the NeurIPS plugin
neurips_plugin = get_plugin('neurips')

# Download papers for 2025
papers_data = neurips_plugin.download(year=2025, output_path='neurips_2025.json')

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
em.create_embeddings_from_db()

# Search by semantic similarity
results = em.search("transformer attention mechanism", n_results=5)

# Search with metadata filter
results = em.search(
    "deep learning",
    n_results=10,
    where={"year": 2025}
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
    lm_studio_url="http://localhost:1234",
    model="gemma-3-4b-it-qat"
)

# Ask a question
response = chat.query("What are the latest developments in transformers?")
print(response)

# Continue conversation
response = chat.chat("Tell me more about the first paper")
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
    uv run abstracts-explorer download --year $year
    uv run abstracts-explorer create-embeddings
done
```

### Custom Configuration

Use a custom configuration file:

```python
import os
os.environ['PAPER_DB'] = 'custom_papers.db'
os.environ['EMBEDDING_DB_PATH'] = 'custom_embeddings'

from abstracts_explorer.config import get_config
config = get_config()
```

### Programmatic Search

```python
from abstracts_explorer.database import DatabaseManager

db = DatabaseManager()

# Complex search with multiple filters
papers = db.search_papers(
    title="learning",
    abstract="neural network",
    year=2025,
    limit=20
)

for paper in papers:
    print(f"{paper['title']} - {paper['year']}")
```

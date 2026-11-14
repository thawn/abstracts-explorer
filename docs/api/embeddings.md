# Embeddings Module

The embeddings module provides vector embeddings functionality for semantic search using ChromaDB.

## Overview

The `EmbeddingsManager` class handles:

- Creating vector embeddings from paper abstracts
- Storing embeddings in ChromaDB
- Semantic similarity search
- Integration with LM Studio embedding models

## Class Reference

```{eval-rst}
.. automodule:: abstracts_explorer.embeddings
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
```

## Usage Examples

### Basic Setup

```python
from abstracts_explorer.embeddings import EmbeddingsManager

# Initialize with database paths
em = EmbeddingsManager(
    collection_name="papers"
)
```

### Creating Embeddings

```python
# Create embeddings from all papers in database
em.embed_from_database()

# Add a single paper
em.add_paper({
    "uid": "abc123",
    "title": "Example Paper",
    "abstract": "This is the abstract...",
})
```

### Semantic Search

```python
# Search by semantic similarity
results = em.search_similar("transformer architecture", n_results=10)

for paper_id, distance in zip(results["ids"][0], results["distances"][0]):
    print(f"{paper_id}: similarity={1 - distance:.3f}")
```

### Filtered Search

ChromaDB stores only IDs, vectors, and content hashes. Resolve metadata filters
to paper UIDs in the SQL database first:

```python
from abstracts_explorer.database import DatabaseManager

db = DatabaseManager()
uids = db.resolve_paper_uids(conferences=["NeurIPS"], years=[2025])

results = em.search_similar(
    query="deep learning",
    n_results=5,
    ids=uids,
)
```

## Embedding Models

The module supports any embedding model available through LM Studio:

### Popular Models

- `text-embedding-qwen3-embedding-4b` (default)
- `text-embedding-nomic-embed-text-v1.5`
- `all-MiniLM-L6-v2`

### Configuring Model

```python
# Via configuration
from abstracts_explorer.config import get_config
config = get_config()
# Set EMBEDDING_MODEL in .env file

# Or directly
em = EmbeddingsManager(
    model="text-embedding-nomic-embed-text-v1.5"
)
```

## Long Abstracts

Embedding models have a finite context window (for example, 8192 tokens).
`EmbeddingsManager` truncates the text sent to the embedding API when it exceeds
`EMBEDDING_MAX_INPUT_TOKENS` (default: `8192`). The full title and abstract are
still stored as the ChromaDB document, so search results and exports are not
truncated.

```python
# Use a smaller budget for a model with a limited context window
em = EmbeddingsManager(max_input_tokens=2048)

# Disable proactive truncation; context-length errors are still retried
em = EmbeddingsManager(max_input_tokens=0)
```

## ChromaDB Integration

The module uses ChromaDB for vector storage:

### Collection Structure

- **Documents**: Paper abstracts
- **Metadata**: Paper ID, title, year, etc.
- **Embeddings**: Vector representations
- **IDs**: Unique identifiers (paper_id)

### Collection Management

```python
# Get collection info
collection = em.collection
print(f"Total papers: {collection.count()}")

# Clear collection
em.clear_collection()

# Check if paper exists
exists = em.paper_exists(paper_id=123)
```

## Search Results Format

Search results are returned as a list of dictionaries:

```python
[
    {
        'id': 'paper_123',
        'title': 'Paper Title',
        'abstract': 'Abstract text...',
        'year': 2025,
        'distance': 0.234,  # Lower is more similar
    },
    # ... more results
]
```

## Performance Considerations

### Batch Processing

Process papers in batches for better performance:

```python
# Default batch size: 100
em.create_embeddings_from_db(batch_size=100)

# Larger batches for more memory
em.create_embeddings_from_db(batch_size=500)
```

### Caching

ChromaDB caches embeddings on disk:

- **Location**: Specified by `embedding_db_path`
- **Persistence**: Embeddings persist across sessions
- **Updates**: Only new papers are embedded

### Memory Usage

Embedding models can be memory-intensive:

- Smaller models: ~1-2 GB RAM
- Larger models: 4-8 GB RAM
- Batch size affects peak memory usage

## Error Handling

```python
try:
    em.create_embeddings_from_db()
except requests.RequestException:
    print("LM Studio connection failed")
except Exception as e:
    print(f"Embedding error: {e}")
```

## Best Practices

1. **Create embeddings once** - They're cached and reused
2. **Use appropriate batch sizes** - Balance speed and memory
3. **Filter searches** - Resolve SQL filters to UIDs, then pass `ids=` to ChromaDB
4. **Choose good models** - Larger models are more accurate but slower
5. **Monitor LM Studio** - Ensure it's running and model is loaded

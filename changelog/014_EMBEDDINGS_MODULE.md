# Embeddings Module

## Overview

The embeddings module provides functionality to generate text embeddings for paper abstracts and store them in a vector database for semantic similarity search. It uses the `text-embedding-qwen3-embedding-4b` model via a local LM Studio API and ChromaDB for vector storage.

## Features

- 🤖 **LM Studio Integration**: Connect to local LM Studio API for embedding generation
- 🗄️ **Vector Database**: Store embeddings in ChromaDB with persistent storage
- 🔍 **Semantic Search**: Find similar papers using cosine similarity
- 📦 **Batch Processing**: Efficiently process multiple papers at once
- 🔗 **Database Integration**: Directly embed papers from SQLite database
- 🏷️ **Metadata Storage**: Store paper metadata alongside embeddings

## Prerequisites

### 1. Install Dependencies

```bash
pip install chromadb>=0.4.0
```

Or install with the embeddings optional dependency:

```bash
pip install -e ".[embeddings]"
```

### 2. Set Up LM Studio

1. Download and install [LM Studio](https://lmstudio.ai/)
2. Start LM Studio and go to the "Local Server" tab
3. Load the `text-embedding-qwen3-embedding-4b` model
4. Start the server (default: http://localhost:1234)

## Quick Start

### Basic Usage

```python
from neurips_abstracts import EmbeddingsManager

# Initialize the manager
em = EmbeddingsManager(
    lm_studio_url="http://localhost:1234",
    chroma_path="./chroma_db",
    collection_name="neurips_papers"
)

# Connect and create collection
em.connect()
em.create_collection()

# Add a paper
em.add_paper(
    paper_id=1,
    abstract="This paper presents a novel deep learning approach...",
    metadata={
        "title": "Deep Learning Paper",
        "authors": "John Doe, Jane Smith",
        "topic": "Machine Learning"
    }
)

# Search for similar papers
results = em.search_similar("deep learning neural networks", n_results=5)

# Print results
for i, paper_id in enumerate(results['ids'][0], 1):
    print(f"{i}. Paper {paper_id}: {results['metadatas'][0][i-1]['title']}")

em.close()
```

### Using Context Manager

```python
from neurips_abstracts import EmbeddingsManager

with EmbeddingsManager() as em:
    em.create_collection()
    em.add_paper(1, "Abstract text...", {"title": "Paper Title"})
    results = em.search_similar("query text")
```

### Embedding from Database

```python
from neurips_abstracts import EmbeddingsManager

em = EmbeddingsManager()
em.connect()
em.create_collection()

# Embed all papers from database
count = em.embed_from_database("neurips.db")
print(f"Embedded {count} papers")

# Or with a filter (only accepted papers)
count = em.embed_from_database(
    "neurips.db",
    where_clause="decision = 'Accept'"
)

em.close()
```

### Batch Processing

```python
from neurips_abstracts import EmbeddingsManager

papers = [
    (1, "Abstract 1...", {"title": "Paper 1", "authors": "Author A"}),
    (2, "Abstract 2...", {"title": "Paper 2", "authors": "Author B"}),
    (3, "Abstract 3...", {"title": "Paper 3", "authors": "Author C"}),
]

with EmbeddingsManager() as em:
    em.create_collection()
    em.add_papers_batch(papers, batch_size=100)
```

## API Reference

### EmbeddingsManager

#### Constructor

```python
EmbeddingsManager(
    lm_studio_url: str = "http://localhost:1234",
    model_name: str = "text-embedding-qwen3-embedding-4b",
    chroma_path: Union[str, Path] = "./chroma_db",
    collection_name: str = "neurips_papers"
)
```

**Parameters:**
- `lm_studio_url`: URL of the LM Studio API endpoint
- `model_name`: Name of the embedding model
- `chroma_path`: Path to ChromaDB persistent storage
- `collection_name`: Name of the ChromaDB collection

#### Methods

##### connect()

Connect to ChromaDB and initialize the client.

```python
em.connect()
```

##### close()

Close the ChromaDB connection.

```python
em.close()
```

##### test_lm_studio_connection()

Test if LM Studio API is accessible.

```python
is_connected = em.test_lm_studio_connection()
```

**Returns:** `bool` - True if connection is successful

##### generate_embedding(text)

Generate embedding for a given text.

```python
embedding = em.generate_embedding("Text to embed")
```

**Parameters:**
- `text`: Text to generate embedding for

**Returns:** `List[float]` - Embedding vector (typically 4096 dimensions)

##### create_collection(reset=False)

Create or retrieve a ChromaDB collection.

```python
em.create_collection()
em.create_collection(reset=True)  # Delete and recreate
```

**Parameters:**
- `reset`: If True, delete existing collection and create new one

##### add_paper(paper_id, abstract, metadata=None, embedding=None)

Add a single paper to the vector database.

```python
em.add_paper(
    paper_id=1,
    abstract="Paper abstract...",
    metadata={"title": "Paper Title", "authors": "Authors"},
    embedding=None  # Auto-generated if not provided
)
```

**Parameters:**
- `paper_id`: Unique identifier for the paper
- `abstract`: Paper abstract text
- `metadata`: Additional metadata dictionary
- `embedding`: Pre-computed embedding (optional)

##### add_papers_batch(papers, batch_size=100)

Add multiple papers in batches.

```python
papers = [
    (id1, abstract1, metadata1),
    (id2, abstract2, metadata2),
    ...
]
em.add_papers_batch(papers, batch_size=100)
```

**Parameters:**
- `papers`: List of tuples (paper_id, abstract, metadata)
- `batch_size`: Number of papers to process in each batch

##### search_similar(query, n_results=10, where=None)

Search for similar papers using semantic similarity.

```python
results = em.search_similar(
    query="deep learning transformers",
    n_results=10,
    where={"decision": "Accept"}  # Optional metadata filter
)
```

**Parameters:**
- `query`: Query text to search for
- `n_results`: Number of results to return
- `where`: Optional metadata filter dictionary

**Returns:** `dict` with keys:
- `ids`: List of paper IDs
- `distances`: List of distance scores
- `documents`: List of abstract texts
- `metadatas`: List of metadata dictionaries

##### get_collection_stats()

Get statistics about the collection.

```python
stats = em.get_collection_stats()
print(f"Collection has {stats['count']} papers")
```

**Returns:** `dict` with keys:
- `name`: Collection name
- `count`: Number of papers
- `metadata`: Collection metadata

##### embed_from_database(db_path, batch_size=100, where_clause=None)

Embed papers from a SQLite database.

```python
count = em.embed_from_database(
    db_path="neurips.db",
    batch_size=100,
    where_clause="decision = 'Accept'"
)
```

**Parameters:**
- `db_path`: Path to SQLite database file
- `batch_size`: Number of papers to process in each batch
- `where_clause`: SQL WHERE clause to filter papers

**Returns:** `int` - Number of papers successfully embedded

## Advanced Usage

### Custom Embedding Dimensions

The `text-embedding-qwen3-embedding-4b` model produces 4096-dimensional embeddings. If you need different dimensions, you can:

1. Use a different model in LM Studio
2. Update the `model_name` parameter when initializing `EmbeddingsManager`

### Metadata Filtering

You can filter search results by metadata:

```python
# Only search in Machine Learning papers
results = em.search_similar(
    "neural networks",
    n_results=10,
    where={"topic": "Machine Learning"}
)

# Search accepted papers only
results = em.search_similar(
    "transformers",
    where={"decision": "Accept"}
)
```

### Persistent Storage

ChromaDB stores embeddings persistently on disk. The collection persists between sessions:

```python
# Session 1: Create and populate
em = EmbeddingsManager(chroma_path="./my_embeddings")
em.connect()
em.create_collection()
em.add_paper(1, "Abstract...", {"title": "Paper 1"})
em.close()

# Session 2: Use existing collection
em = EmbeddingsManager(chroma_path="./my_embeddings")
em.connect()
em.create_collection()  # Gets existing collection
results = em.search_similar("query")
em.close()
```

### Error Handling

```python
from neurips_abstracts import EmbeddingsManager, EmbeddingsError

try:
    em = EmbeddingsManager()
    em.connect()
    em.create_collection()
    em.add_paper(1, "Abstract...", {"title": "Paper"})
except EmbeddingsError as e:
    print(f"Error: {e}")
finally:
    em.close()
```

## Performance Considerations

### Embedding Generation

- Embedding generation requires LM Studio to be running
- Each embedding request takes ~100-500ms depending on text length
- Use batch processing for multiple papers
- Consider caching embeddings to avoid regeneration

### Batch Size

- Default batch size: 100 papers
- Larger batches: Faster processing, more memory usage
- Smaller batches: Slower processing, less memory usage
- Recommended: 50-200 papers per batch

### Database Integration

When embedding from a database:

```python
# Process only papers with abstracts
count = em.embed_from_database(
    "neurips.db",
    where_clause="abstract IS NOT NULL AND abstract != ''"
)

# Process specific year
count = em.embed_from_database(
    "neurips.db",
    where_clause="CAST(id AS TEXT) LIKE '2025%'"
)
```

## Troubleshooting

### LM Studio Connection Issues

If `test_lm_studio_connection()` returns False:

1. Check if LM Studio is running
2. Verify the server is on http://localhost:1234
3. Ensure the embedding model is loaded
4. Check firewall settings

### ChromaDB Issues

If you encounter ChromaDB errors:

1. Delete the ChromaDB directory and recreate:
   ```python
   em.create_collection(reset=True)
   ```
2. Check disk space for storage
3. Verify write permissions

### Memory Issues

For large datasets:

1. Reduce batch size: `batch_size=50`
2. Process papers in chunks
3. Monitor memory usage

## Examples

See the complete example in `examples/embeddings_demo.py` for a full demonstration of all features.

## Integration with Other Modules

### With DatabaseManager

```python
from neurips_abstracts import DatabaseManager, EmbeddingsManager

# Create database
db = DatabaseManager("neurips.db")
db.connect()
db.create_tables()
db.load_json_data("neurips_2025.json")
db.close()

# Embed papers from database
em = EmbeddingsManager()
em.connect()
em.create_collection()
count = em.embed_from_database("neurips.db")
print(f"Embedded {count} papers")
em.close()
```

### With Downloader

```python
from neurips_abstracts import download_neurips_data, EmbeddingsManager

# Download data
data = download_neurips_data(year=2025)

# Extract and embed papers
em = EmbeddingsManager()
em.connect()
em.create_collection()

papers = []
for paper in data:
    if paper.get('abstract'):
        papers.append((
            paper['id'],
            paper['abstract'],
            {
                'title': paper.get('name', ''),
                'authors': paper.get('authors', ''),
                'topic': paper.get('topic', '')
            }
        ))

em.add_papers_batch(papers)
em.close()
```

## References

- [LM Studio](https://lmstudio.ai/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Qwen Embedding Models](https://huggingface.co/Qwen)

# Embeddings Quick Reference

## Setup

```bash
# Install dependencies
pip install chromadb

# Start LM Studio with text-embedding-qwen3-embedding-4b model
# Server should run on http://localhost:1234
```

## Import

```python
from neurips_abstracts import EmbeddingsManager
```

## Basic Pattern

```python
# Initialize
em = EmbeddingsManager()
em.connect()
em.create_collection()

# Use the embeddings manager
# ... your code here ...

# Clean up
em.close()
```

## Context Manager (Recommended)

```python
with EmbeddingsManager() as em:
    em.create_collection()
    # Your code here
```

## Common Operations

### Test Connection

```python
em = EmbeddingsManager()
if em.test_lm_studio_connection():
    print("LM Studio is ready!")
```

### Embed from Database

```python
with EmbeddingsManager() as em:
    em.create_collection()
    count = em.embed_from_database("neurips.db")
    print(f"Embedded {count} papers")
```

### Embed Specific Papers

```python
with EmbeddingsManager() as em:
    em.create_collection()
    # Only accepted papers
    count = em.embed_from_database(
        "neurips.db",
        where_clause="decision = 'Accept'"
    )
```

### Add Single Paper

```python
with EmbeddingsManager() as em:
    em.create_collection()
    em.add_paper(
        paper_id=1,
        abstract="Your paper abstract here...",
        metadata={
            "title": "Paper Title",
            "authors": "John Doe",
            "year": "2025"
        }
    )
```

### Add Multiple Papers

```python
papers = [
    (1, "Abstract 1", {"title": "Paper 1"}),
    (2, "Abstract 2", {"title": "Paper 2"}),
    (3, "Abstract 3", {"title": "Paper 3"}),
]

with EmbeddingsManager() as em:
    em.create_collection()
    em.add_papers_batch(papers, batch_size=100)
```

### Search Similar Papers

```python
with EmbeddingsManager() as em:
    em.create_collection()
    results = em.search_similar(
        "machine learning neural networks",
        n_results=10
    )
    
    # Access results
    paper_ids = results['ids'][0]
    distances = results['distances'][0]
    abstracts = results['documents'][0]
    metadata = results['metadatas'][0]
```

### Search with Metadata Filter

```python
with EmbeddingsManager() as em:
    em.create_collection()
    results = em.search_similar(
        "deep learning",
        n_results=5,
        where={"decision": "Accept"}  # Only accepted papers
    )
```

### Get Collection Stats

```python
with EmbeddingsManager() as em:
    em.create_collection()
    stats = em.get_collection_stats()
    print(f"Collection: {stats['name']}")
    print(f"Papers: {stats['count']}")
```

### Reset Collection

```python
with EmbeddingsManager() as em:
    em.create_collection(reset=True)  # Deletes and recreates
```

## Configuration

```python
em = EmbeddingsManager(
    lm_studio_url="http://localhost:1234",  # LM Studio API URL
    model_name="text-embedding-qwen3-embedding-4b",  # Model name
    chroma_path="./chroma_db",  # ChromaDB storage path
    collection_name="neurips_papers"  # Collection name
)
```

## Error Handling

```python
from neurips_abstracts import EmbeddingsManager, EmbeddingsError

try:
    with EmbeddingsManager() as em:
        em.create_collection()
        em.add_paper(1, "Abstract", {"title": "Paper"})
except EmbeddingsError as e:
    print(f"Error: {e}")
```

## Complete Example

```python
from neurips_abstracts import DatabaseManager, EmbeddingsManager

# Step 1: Create database (if not exists)
with DatabaseManager("neurips.db") as db:
    db.create_tables()
    db.load_json_file("data/neurips_2025.json")

# Step 2: Generate embeddings
with EmbeddingsManager() as em:
    em.create_collection()
    count = em.embed_from_database("neurips.db")
    print(f"Embedded {count} papers")

# Step 3: Search
with EmbeddingsManager() as em:
    em.create_collection()
    results = em.search_similar(
        "transformer architectures for vision",
        n_results=5
    )
    
    for i, paper_id in enumerate(results['ids'][0], 1):
        meta = results['metadatas'][0][i-1]
        similarity = 1 - results['distances'][0][i-1]
        print(f"{i}. {meta['title']} (similarity: {similarity:.4f})")
```

## Tips

1. **First Time**: Test LM Studio connection before embedding
2. **Large Datasets**: Use batch processing with appropriate batch size (50-200)
3. **Filters**: Use `where_clause` to embed only relevant papers
4. **Persistence**: Embeddings persist in ChromaDB between sessions
5. **Performance**: Each embedding takes ~100-500ms depending on text length
6. **Memory**: Reduce batch size if you encounter memory issues
7. **Errors**: Wrap operations in try-except to handle API/network issues

## Performance

| Operation           | Time (approx) |
| ------------------- | ------------- |
| Single embedding    | 100-500ms     |
| Batch (100 papers)  | 1-5 minutes   |
| Search query        | < 1 second    |
| Collection creation | < 1 second    |

## Common Issues

### LM Studio not connecting
```python
em = EmbeddingsManager()
if not em.test_lm_studio_connection():
    print("Check that LM Studio is running on http://localhost:1234")
```

### Empty results
```python
# Make sure collection has data
stats = em.get_collection_stats()
print(f"Papers in collection: {stats['count']}")
```

### Reset everything
```python
with EmbeddingsManager() as em:
    em.create_collection(reset=True)
    em.embed_from_database("neurips.db")
```

## See Also

- `EMBEDDINGS_MODULE.md` - Full documentation
- `examples/embeddings_demo.py` - Complete working example
- `tests/test_embeddings.py` - Test cases showing usage patterns

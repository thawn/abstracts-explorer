# CLI Command Reference

The `neurips-abstracts` package provides a command-line interface for working with NeurIPS conference data and generating embeddings.

## Installation

After installing the package, the `neurips-abstracts` command will be available:

```bash
pip install -e .
```

## Commands

### `download`

Download NeurIPS conference data and create a SQLite database.

**Usage:**
```bash
neurips-abstracts download [OPTIONS]
```

**Options:**
- `--year YEAR`: Year of NeurIPS conference (default: 2025)
- `--output PATH`: Output database file path (default: neurips.db)
- `--force`: Force re-download even if file exists

**Examples:**

```bash
# Download NeurIPS 2025 data
neurips-abstracts download --year 2025 --output neurips_2025.db

# Force re-download
neurips-abstracts download --year 2025 --output neurips_2025.db --force
```

### `search`

Search the vector database for papers similar to a query using semantic similarity.

**Usage:**
```bash
neurips-abstracts search QUERY [OPTIONS]
```

**Required Arguments:**
- `QUERY`: Search query text (positional argument)

**Optional Settings:**
- `--embeddings-path PATH`: Path to ChromaDB vector database (default: chroma_db)
- `--collection NAME`: Name of the ChromaDB collection (default: neurips_papers)
- `--n-results N`: Number of results to return (default: 5)
- `--where FILTER`: Metadata filter as key=value pairs, comma-separated
- `--show-abstract`: Show paper abstracts in results
- `--lm-studio-url URL`: URL for LM Studio API (default: http://localhost:1234)
- `--model NAME`: Name of the embedding model (default: text-embedding-qwen3-embedding-4b)

**Prerequisites:**
- Embeddings database must exist (created with `create-embeddings` command)
- LM Studio must be running

**Examples:**

```bash
# Basic search
neurips-abstracts search "graph neural networks"

# Search with more results
neurips-abstracts search "deep learning for vision" --n-results 10

# Show abstracts in results
neurips-abstracts search "reinforcement learning" --show-abstract

# Search with metadata filter (only spotlight papers)
neurips-abstracts search "transformer models" --where "decision=Accept (spotlight)"

# Search in custom embeddings database
neurips-abstracts search "molecular generation" \
  --embeddings-path my_embeddings/ \
  --collection my_collection
```

**Output:**
```
NeurIPS Semantic Search
======================================================================
Query: graph neural networks
Embeddings: chroma_db
Collection: neurips_papers
Results: 5
======================================================================

📊 Searching 5,989 papers in collection 'neurips_papers'

🔍 Searching for: 'graph neural networks'...

✅ Found 5 similar paper(s):

1. Paper ID: 119825
   Title: DMol: A Highly Efficient and Chemical Motif-Preserving Molecule Generation Platform
   Authors: 457120, 406892, 458097, 253414, 253411, 10483
   Decision: Accept (poster)
   Topic: Applications->Chemistry and Drug Discovery
   Similarity: 0.5243

...
```

### `create-embeddings`

Generate embeddings for NeurIPS abstracts and store them in a ChromaDB vector database.

**Usage:**
```bash
neurips-abstracts create-embeddings --db-path DB_PATH [OPTIONS]
```

**Required Options:**
- `--db-path PATH`: Path to the SQLite database with papers (required)

**Optional Settings:**
- `--output PATH`: Output directory for ChromaDB vector database (default: chroma_db)
- `--collection NAME`: Name for the ChromaDB collection (default: neurips_papers)
- `--batch-size N`: Number of papers to process at once (default: 100)
- `--lm-studio-url URL`: URL for LM Studio API (default: http://localhost:1234)
- `--model NAME`: Name of the embedding model (default: text-embedding-qwen3-embedding-4b)
- `--force`: Reset collection if it already exists
- `--where CLAUSE`: SQL WHERE clause to filter papers

**Prerequisites:**
- LM Studio must be running at the specified URL (default: http://localhost:1234)
- The embedding model must be loaded in LM Studio

**Progress Bar:**
The command displays a real-time progress bar showing:
- Number of papers processed
- Percentage complete
- Processing speed (papers/second)
- Estimated time remaining

**Examples:**

```bash
# Generate embeddings for all papers (with progress bar)
neurips-abstracts create-embeddings --db-path neurips_2025.db

# Use custom output directory and collection name
neurips-abstracts create-embeddings \
  --db-path neurips_2025.db \
  --output embeddings/ \
  --collection neurips_2025

# Process with smaller batch size
neurips-abstracts create-embeddings \
  --db-path neurips_2025.db \
  --batch-size 50

# Generate embeddings only for accepted papers
neurips-abstracts create-embeddings \
  --db-path neurips_2025.db \
  --where "decision LIKE '%Accept%'"

# Use custom LM Studio URL and model
neurips-abstracts create-embeddings \
  --db-path neurips_2025.db \
  --lm-studio-url http://localhost:5000 \
  --model custom-embedding-model

# Reset existing collection and regenerate embeddings
neurips-abstracts create-embeddings \
  --db-path neurips_2025.db \
  --force
```

## Complete Workflow Example

Here's a complete workflow from downloading data to generating embeddings:

```bash
# Step 1: Download NeurIPS 2025 data
neurips-abstracts download --year 2025 --output neurips_2025.db

# Step 2: Start LM Studio and load the embedding model
# (Do this manually in the LM Studio application)

# Step 3: Generate embeddings for all papers
neurips-abstracts create-embeddings \
  --db-path neurips_2025.db \
  --output neurips_2025_embeddings \
  --collection neurips_2025_papers

# Step 4: Use the embeddings in your Python code
```

## Filtering Examples

The `--where` option accepts any valid SQLite WHERE clause:

```bash
# Only poster papers
neurips-abstracts create-embeddings \
  --db-path neurips_2025.db \
  --where "decision LIKE '%poster%'"

# Papers from specific track
neurips-abstracts create-embeddings \
  --db-path neurips_2025.db \
  --where "track = 'Main Conference Track'"

# Papers with specific keywords
neurips-abstracts create-embeddings \
  --db-path neurips_2025.db \
  --where "keywords LIKE '%machine learning%'"

# Combine multiple conditions
neurips-abstracts create-embeddings \
  --db-path neurips_2025.db \
  --where "decision LIKE '%Accept%' AND abstract IS NOT NULL"

# Specific paper IDs
neurips-abstracts create-embeddings \
  --db-path neurips_2025.db \
  --where "id IN (119718, 119663, 114995)"
```

## Output

The `create-embeddings` command provides detailed progress information:

```
NeurIPS Embeddings Generator
======================================================================
Database: neurips_2025.db
Output:   chroma_db
Collection: neurips_papers
Model:    text-embedding-qwen3-embedding-4b
LM Studio: http://localhost:1234
======================================================================

📊 Found 5,990 papers in database

🔧 Initializing embeddings manager...
🔌 Testing LM Studio connection...
✅ Successfully connected to LM Studio

📁 Creating collection 'neurips_papers'...

🚀 Generating embeddings (batch size: 100)...
This may take a while for large databases...

✅ Successfully generated embeddings for 5,990 papers

📊 Collection Statistics:
   Name:  neurips_papers
   Count: 5,990 documents

💾 Vector database saved to: chroma_db

You can now use the search_similar() method to find relevant papers!
```

## Error Handling

The CLI provides helpful error messages:

**Database not found:**
```
❌ Error: Database file not found: neurips_2025.db

You can create a database using:
  neurips-abstracts download --output neurips_2025.db
```

**LM Studio not available:**
```
❌ Failed to connect to LM Studio!

Please ensure:
  - LM Studio is running at http://localhost:1234
  - The text-embedding-qwen3-embedding-4b model is loaded
```

## Exit Codes

- `0`: Success
- `1`: Error (database not found, LM Studio unavailable, etc.)

## Performance Tips

1. **Batch Size**: Adjust `--batch-size` based on your system:
   - Smaller batches (50-100): More stable, slower
   - Larger batches (200-500): Faster, more memory usage

2. **Filtering**: Use `--where` to process only relevant papers and reduce processing time

3. **Incremental Updates**: Use `--where` to process only new papers:
   ```bash
   neurips-abstracts create-embeddings \
     --db-path neurips_2025.db \
     --where "id > 120000"
   ```

4. **Reset**: Use `--force` to completely reset and regenerate embeddings

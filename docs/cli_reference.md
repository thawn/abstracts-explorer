# Command-Line Interface Reference

Complete reference for all CLI commands.

**Note:** All commands shown below should be prefixed with `uv run` when running from the project directory, or you can activate the virtual environment first with `source .venv/bin/activate`.

## Global Options

All commands support these global options:

- `--help`: Show help message and exit
- `--version`: Show version information

## Commands

### download

Download NeurIPS papers from OpenReview API.

**Usage:**

```bash
abstracts-explorer download [OPTIONS]
```

**Options:**

- `--year INTEGER`: Conference year to download (required)
- `--db-path TEXT`: Path to SQLite database file (required)
- `--force`: Force re-download even if papers exist
- `--cache/--no-cache`: Enable/disable caching (default: enabled)

**Examples:**

```bash
# Download 2025 papers
uv run abstracts-explorer download --year 2025 --db-path data/neurips_2025.db

# Force re-download
uv run abstracts-explorer download --year 2025 --db-path data/neurips_2025.db --force

# Disable caching
uv run abstracts-explorer download --year 2025 --db-path data/neurips_2025.db --no-cache
```

### create-embeddings

Create vector embeddings for semantic search.

**Usage:**

```bash
abstracts-explorer create-embeddings [OPTIONS]
```

**Options:**

- `--db-path TEXT`: Path to SQLite database with papers (required)
- `--embedding-db-path TEXT`: Path to ChromaDB database (default: from config)
- `--collection-name TEXT`: Collection name in ChromaDB (default: from config)
- `--model TEXT`: Embedding model to use (default: from config)
- `--force`: Recreate embeddings even if they exist

**Examples:**

```bash
# Create embeddings with defaults
uv run abstracts-explorer create-embeddings --db-path data/neurips_2025.db

# Use custom paths
uv run abstracts-explorer create-embeddings \
    --db-path data/neurips_2025.db \
    --embedding-db-path custom_embeddings \
    --collection-name my_papers

# Force recreation
uv run abstracts-explorer create-embeddings --db-path data/neurips_2025.db --force
```

### search

Search papers by keywords or semantic similarity.

**Usage:**

```bash
abstracts-explorer search QUERY [OPTIONS]
```

**Arguments:**

- `QUERY`: Search query string (required)

**Options:**

- `--db-path TEXT`: Path to SQLite database (required)
- `--limit INTEGER`: Maximum number of results (default: 10)
- `--year INTEGER`: Filter by conference year
- `--use-embeddings`: Use semantic search (requires embeddings)
- `--title-only`: Search only in paper titles
- `--abstract-only`: Search only in abstracts

**Examples:**

```bash
# Basic search
uv run abstracts-explorer search "transformer" --db-path data/neurips_2025.db

# Limit results
uv run abstracts-explorer search "deep learning" --db-path data/neurips_2025.db --limit 20

# Filter by year
uv run abstracts-explorer search "neural network" --db-path data/neurips_2025.db --year 2025

# Semantic search using embeddings
uv run abstracts-explorer search "attention mechanism" --db-path data/neurips_2025.db --use-embeddings

# Search only titles
uv run abstracts-explorer search "BERT" --db-path data/neurips_2025.db --title-only
```

### chat

Interactive RAG-powered chat interface.

**Usage:**

```bash
abstracts-explorer chat [OPTIONS]
```

**Options:**

- `--db-path TEXT`: Path to SQLite database (required)
- `--embedding-db-path TEXT`: Path to ChromaDB database (default: from config)
- `--model TEXT`: LLM model to use (default: from config)
- `--temperature FLOAT`: Temperature for responses (default: from config)
- `--max-tokens INTEGER`: Maximum tokens in response (default: from config)
- `--n-papers INTEGER`: Number of papers for context (default: from config)

**Interactive Commands:**

While in the chat session:

- Type your question and press Enter to get a response
- `exit` or `quit`: Exit the chat session
- `reset`: Reset the conversation history
- `export [filename]`: Export conversation to JSON file

**Examples:**

```bash
# Start chat with defaults
uv run abstracts-explorer chat --db-path data/neurips_2025.db

# Use custom model
uv run abstracts-explorer chat --db-path data/neurips_2025.db --model llama-3.2-3b-instruct

# Adjust response parameters
uv run abstracts-explorer chat \
    --db-path data/neurips_2025.db \
    --temperature 0.9 \
    --max-tokens 2000 \
    --n-papers 10
```

### info

Show database information and statistics.

**Usage:**

```bash
abstracts-explorer info [OPTIONS]
```

**Options:**

- `--db-path TEXT`: Path to SQLite database (required)
- `--show-embeddings`: Also show embedding statistics

**Examples:**

```bash
# Basic info
uv run abstracts-explorer info --db-path data/neurips_2025.db

# Include embedding info
uv run abstracts-explorer info --db-path data/neurips_2025.db --show-embeddings
```

## Environment Variables

All CLI commands respect configuration from environment variables and `.env` files. See the [Configuration](configuration.md) page for details.

## Exit Codes

- `0`: Success
- `1`: General error
- `2`: Invalid arguments or options

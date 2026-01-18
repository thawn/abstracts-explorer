# MCP Server for Cluster Analysis

The Abstracts Explorer includes a Model Context Protocol (MCP) server that provides tools for analyzing clustered paper embeddings. This enables LLM-based assistants to answer sophisticated questions about research topics, trends, and developments.

## What is MCP?

Model Context Protocol (MCP) is a protocol that allows tools and servers to provide context and capabilities to LLM-based applications. The MCP server exposes tools that can be called by LLM assistants to perform specific tasks.

## Features

The MCP server provides four main tools:

### 1. `get_cluster_topics`

Analyzes clustered embeddings to identify the most frequently mentioned topics in each cluster.

**Parameters:**
- `n_clusters` (int): Number of clusters to create (default: 8)
- `reduction_method` (str): Dimensionality reduction method - 'pca' or 'tsne' (default: 'pca')
- `clustering_method` (str): Clustering algorithm - 'kmeans', 'dbscan', or 'agglomerative' (default: 'kmeans')
- `embeddings_path` (str, optional): Path to ChromaDB embeddings database
- `collection_name` (str, optional): Name of ChromaDB collection
- `db_path` (str, optional): Path to SQLite database

**Returns:** JSON with cluster statistics and topics for each cluster, including:
- Keywords and their frequencies
- Common sessions
- Year distribution
- Sample paper titles

**Example use case:** "What are the most frequently mentioned topics in the conference papers?"

### 2. `get_topic_evolution`

Analyzes how specific topics have evolved over the years.

**Parameters:**
- `topic_keywords` (str): Keywords describing the topic (e.g., "transformers attention")
- `conference` (str, optional): Filter by conference name
- `start_year` (int, optional): Start year for analysis
- `end_year` (int, optional): End year for analysis
- `embeddings_path` (str, optional): Path to ChromaDB embeddings database
- `collection_name` (str, optional): Name of ChromaDB collection
- `db_path` (str, optional): Path to SQLite database

**Returns:** JSON with topic evolution data including:
- Year-by-year paper counts
- Sample papers from each year
- Relevance scores

**Example use case:** "How have topics related to 'transformer architectures' evolved over the years at NeurIPS?"

### 3. `get_recent_developments`

Finds the most important recent developments in a specific topic.

**Parameters:**
- `topic_keywords` (str): Keywords describing the topic
- `n_years` (int): Number of recent years to consider (default: 2)
- `n_results` (int): Number of papers to return (default: 10)
- `conference` (str, optional): Filter by conference name
- `embeddings_path` (str, optional): Path to ChromaDB embeddings database
- `collection_name` (str, optional): Name of ChromaDB collection
- `db_path` (str, optional): Path to SQLite database

**Returns:** JSON with recent papers including:
- Paper titles and abstracts
- Years and conferences
- Relevance scores

**Example use case:** "What are the most important recent developments in large language models?"

### 4. `get_cluster_visualization`

Generates visualization data for clustered embeddings.

**Parameters:**
- `n_clusters` (int): Number of clusters (default: 8)
- `reduction_method` (str): Reduction method - 'pca' or 'tsne' (default: 'tsne')
- `clustering_method` (str): Clustering method (default: 'kmeans')
- `n_components` (int): Number of dimensions - 2 or 3 (default: 2)
- `output_path` (str, optional): Path to save visualization JSON
- `embeddings_path` (str, optional): Path to ChromaDB embeddings database
- `collection_name` (str, optional): Name of ChromaDB collection
- `db_path` (str, optional): Path to SQLite database

**Returns:** JSON with visualization data including:
- Point coordinates (x, y, optionally z)
- Cluster assignments
- Paper metadata
- Statistics

**Example use case:** "Display a graphical representation of the paper clusters."

## Starting the MCP Server

### Basic Usage

Start the server with default settings:

```bash
abstracts-explorer mcp-server
```

This starts the server on `http://127.0.0.1:8000` with SSE transport.

### Custom Host and Port

Start on a custom host and port:

```bash
abstracts-explorer mcp-server --host 0.0.0.0 --port 8080
```

### STDIO Transport

For local CLI integration, use stdio transport:

```bash
abstracts-explorer mcp-server --transport stdio
```

## Configuration

The MCP server uses the same configuration as the rest of Abstracts Explorer. Configure via `.env` file:

```bash
# Database Configuration
PAPER_DB=abstracts.db
EMBEDDING_DB_PATH=chroma_db
COLLECTION_NAME=papers

# LLM Backend (for embeddings in tools)
LLM_BACKEND_URL=http://localhost:1234
EMBEDDING_MODEL=text-embedding-qwen3-embedding-4b
```

## Integration with LLM Assistants

The MCP server can be integrated with any MCP-compatible LLM assistant or client. Here's how tools would be called:

### Example: Claude Desktop Integration

Add to Claude Desktop MCP configuration:

```json
{
  "mcpServers": {
    "abstracts-explorer": {
      "command": "abstracts-explorer",
      "args": ["mcp-server", "--transport", "stdio"]
    }
  }
}
```

### Example Tool Call

When an LLM assistant needs to analyze topics, it can call:

```json
{
  "tool": "get_cluster_topics",
  "arguments": {
    "n_clusters": 8,
    "reduction_method": "tsne",
    "clustering_method": "kmeans"
  }
}
```

The server will:
1. Load embeddings from ChromaDB
2. Perform clustering
3. Analyze topics in each cluster
4. Return structured JSON results

## API Reference

### Tool Response Format

All tools return JSON strings with the following general structure:

```json
{
  "statistics": {
    "n_clusters": 8,
    "total_papers": 1000
  },
  "clusters": [
    {
      "cluster_id": 0,
      "paper_count": 150,
      "keywords": [
        {"keyword": "transformer", "count": 45},
        {"keyword": "attention", "count": 38}
      ],
      "sessions": [
        {"session": "Deep Learning", "count": 100}
      ],
      "years": {"2023": 80, "2024": 70},
      "sample_titles": ["Paper 1", "Paper 2", "Paper 3"]
    }
  ]
}
```

### Error Handling

If an error occurs, tools return JSON with an error field:

```json
{
  "error": "Failed to load clustering data: Database not found"
}
```

## Requirements

Before using the MCP server, ensure:

1. **Embeddings are created**: Run `abstracts-explorer create-embeddings` first
2. **Database exists**: Download papers with `abstracts-explorer download`
3. **MCP package installed**: `uv sync` or `pip install mcp>=1.0.0`

## Troubleshooting

### "No embeddings found"

Make sure to create embeddings first:

```bash
abstracts-explorer create-embeddings --db-path data/abstracts.db
```

### "Failed to connect to database"

Check that the database paths in `.env` are correct:

```bash
# .env
PAPER_DB=data/abstracts.db
EMBEDDING_DB_PATH=chroma_db
```

### Port already in use

Change the port:

```bash
abstracts-explorer mcp-server --port 8001
```

## Advanced Usage

### Custom Clustering Parameters

Each tool accepts clustering parameters. For DBSCAN:

```python
# Tool arguments
{
  "clustering_method": "dbscan",
  "eps": 0.5,
  "min_samples": 5
}
```

### Filtering by Conference

Analyze specific conferences:

```python
# get_topic_evolution arguments
{
  "topic_keywords": "neural networks",
  "conference": "neurips",
  "start_year": 2020,
  "end_year": 2024
}
```

## See Also

- [CLI Reference](cli_reference.md) - Command-line interface documentation
- [Clustering Guide](clustering.md) - Clustering and visualization guide
- [RAG Chat](rag.md) - Using RAG chat with MCP tools
- [Configuration](configuration.md) - Environment configuration options

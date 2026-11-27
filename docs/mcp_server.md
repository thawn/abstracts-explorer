# MCP Server for Cluster Analysis

The Abstracts Explorer includes a Model Context Protocol (MCP) server that provides tools for analyzing clustered paper embeddings. This enables LLM-based assistants to answer sophisticated questions about research topics, trends, and developments.

**NEW**: MCP tools are automatically integrated into the RAG chat system, allowing the LLM to decide when to use clustering analysis versus paper retrieval.

## What is MCP?

Model Context Protocol (MCP) is a protocol that allows tools and servers to provide context and capabilities to LLM-based applications. The MCP server exposes tools that can be called by LLM assistants to perform specific tasks.

## Features

The MCP server provides ten tools for conference analysis, including three
personalized tools that work with the papers a user marked as interesting and
an ORCID-based search-term suggestion tool.

### 1. `get_conference_topics`

Analyzes clustered embeddings to identify the most frequently mentioned topics in each cluster.

**Parameters:**
- `conferences` (list of str): Conference names to analyze (e.g., `["NeurIPS"]`). Required.
- `years` (list of int, optional): Filter by specific years
- `collection_name` (str, optional): Name of ChromaDB collection

**Returns:** JSON with cluster statistics and topics for each cluster. Each
topic entry uses the descriptive sentence as its primary `topic` value (falling
back to the short legend label when no description is cached) and keeps the
short label in `label`. Entries also include:
- Keywords and their frequencies
- Common sessions
- Year distribution
- Sample paper titles

**Example use case:** "What are the most frequently mentioned topics in the conference papers?"

### 2. `get_topic_evolution`

Analyzes how specific topics have evolved over the years.

**Parameters:**
- `topic_keywords` (str): Keywords describing the topic (e.g., "transformers attention")
- `conferences` (list of str, optional): Conference names to analyze (e.g., `["NeurIPS", "ICLR"]`)
- `start_year` (int, optional): Start year for analysis (inclusive)
- `end_year` (int, optional): End year for analysis (inclusive)
- `distance_threshold` (float, optional): Maximum embedding distance for papers to count (default: 1.1)
- `collection_name` (str, optional): Name of ChromaDB collection

**Returns:** JSON with topic evolution data including:
- Year-by-year paper counts
- Sample papers from each year
- Relevance scores

**Example use case:** "How have topics related to 'transformer architectures' evolved over the years at NeurIPS?"

### 3. `search_papers`

Finds the most important recent developments in a specific topic.

**Parameters:**
- `topic_keywords` (str): Keywords describing the topic
- `conference` (str, optional): Filter by conference name (required in practice; pass a conference)
- `years` (list, optional): Years to consider (default: all years)
- `sessions` (list, optional): Paper sessions to consider (default: all sessions)
- `limit` (int): Number of papers to return (default: 10)
- `where_filters` (dict, optional): SQL-style paper filters (e.g. `{"session": {"$in": ["Oral"]}}`)
- `collection_name` (str, optional): Name of ChromaDB collection

**Returns:** JSON with recent papers including:
- Paper titles and abstracts
- Years and conferences
- Relevance scores

**Example use case:** "What are the most important recent developments in large language models?"

### 4. `get_cluster_visualization`

Generates visualization data for clustered embeddings.

**Parameters:**
- `conferences` (list of str): Conference names to retrieve visualization for (e.g., `["NeurIPS"]`). Required.
- `years` (list of int, optional): Filter by specific years
- `output_path` (str, optional): Path to save visualization JSON
- `collection_name` (str, optional): Name of ChromaDB collection

**Returns:** JSON with visualization data including:
- Point coordinates (x, y, optionally z)
- Cluster assignments
- Paper metadata
- Statistics

**Example use case:** "Display a graphical representation of the paper clusters."

### 5. `summarize_interesting_papers`

Summarizes the papers a user marked, starred, liked, or rated as interesting.

Ratings are stored in the browser (`localStorage.paperPriorities`) and are
forwarded automatically by the web UI chat. The tool receives them as
parameters; no ratings are persisted server-side.

**Parameters:**
- `paper_ids` (list of str, optional): Paper UIDs. When omitted, the keys of `priorities` are used.
- `priorities` (dict, optional): Mapping of paper UID to a dict with `priority` and optional `searchTerm`.
- `conference` (str, optional): Conference filter (e.g. `"ICLR"` or `"ICLR 2026"`).
- `years` (list of int, optional): Year filter (e.g. `[2026]`).

**Returns:** JSON with:
- Total interesting papers and rating distribution/average
- Conferences, years, sessions, search terms, and top keywords
- Highest-rated papers and compact paper records

**Example use case:** "Give me a summary about the papers I marked as interesting for ICLR'26!"

### 6. `recommend_related_papers`

Recommends further papers based on a rating-weighted embedding centroid of the
user's interesting papers. Papers the user already rated are excluded.

**Parameters:**
- `paper_ids` (list of str, optional): Paper UIDs. When omitted, the keys of `priorities` are used.
- `priorities` (dict, optional): Mapping of paper UID to a dict with `priority` and optional `searchTerm`.
- `conferences` (list of str, optional): Restrict recommendations to these conferences. Defaults to the dominant conference of the interesting papers.
- `years` (list of int, optional): Restrict recommendations to these years. Defaults to the years of the interesting papers.
- `limit` (int, optional): Number of recommendations (default: 10).
- `exclude_ids` (list of str, optional): Additional UIDs to exclude.

**Returns:** JSON with recommended papers, each including a relevance score and
reasons (shared keywords, authors, session, or semantic closeness).

**Example use case:** "Based on my interesting papers, suggest other papers that may interest me."

### 7. `suggest_authors_to_meet`

Suggests authors to meet at the conference based on the user's interesting
papers. The tool reuses the related-paper semantic search, aggregates authors
across similar papers, and ranks them by similarity-weighted frequency. Authors
of papers the user starred are included and marked.

**Parameters:**
- `paper_ids` (list of str, optional): Paper UIDs. When omitted, the keys of `priorities` are used.
- `priorities` (dict, optional): Mapping of paper UID to a dict with `priority` and optional `searchTerm`.
- `conferences` (list of str, optional): Restrict similar papers to these conferences. Defaults to the dominant conference of the interesting papers.
- `years` (list of int, optional): Restrict similar papers to these years. Defaults to the years of the interesting papers.
- `limit` (int, optional): Number of authors to return (default: 10).
- `paper_limit` (int, optional): Number of similar papers to consider (default: 50).
- `exclude_ids` (list of str, optional): Additional UIDs to exclude.

**Returns:** JSON with ranked authors, each including a similarity-weighted
score, similar/starred paper counts, a star marker for authors of starred
papers, and the sessions, rooms, and poster positions where they can be found.

**Example use case:** "Who should I meet at ICLR'26 based on my interesting papers?"

### 8. `get_paper_details`

Fetches detailed metadata for one or more papers — for follow-up questions
after a semantic search, not for topic searches.

**Parameters:**
- `title` (str, optional): Title or partial title to search for (case-insensitive)
- `paper_id` (str, optional): Paper UID or original conference/OpenReview ID
- `conference` (str, optional): Filter by conference name (e.g., `"NeurIPS"`)
- `year` (int, optional): Filter by publication year
- `limit` (int, optional): Maximum number of papers to return when searching by title (default: 5)

**Returns:** JSON with full paper metadata:
- Authors, URLs, PDF and poster links
- Session and room information
- Keywords, awards, and other stored fields

**Example use case:** "Who wrote that paper? Where is the poster?"

### 9. `analyze_topic_relevance`

Counts papers within a distance threshold of a topic's embedding to gauge
how prominent a research area is.

**Parameters:**
- `topic` (str): The topic or research question to analyze (e.g., "Uncertainty quantification")
- `conferences` (list of str): Conference names to analyze (e.g., `["NeurIPS"]`). Required.
- `years` (list of int, optional): Filter by specific years (e.g., `[2024, 2025]`)
- `distance_threshold` (float, optional): Maximum embedding distance for a paper to count (default: 1.1)
- `collection_name` (str, optional): Name of ChromaDB collection

**Returns:** JSON with relevance statistics:
- Number of matching papers per year and overall
- Share relative to the total number of papers

**Example use case:** "How prominent is uncertainty quantification at this year's conference?"

### 10. `suggest_search_terms_from_orcid`

Suggests paper search terms based on a researcher's public ORCID record. The
tool reads the public works, author keywords, and display name, then asks the
configured chat model for concise semantic-search terms. If the LLM is
unavailable or returns unusable output, it falls back to ORCID keywords and
phrases extracted from work titles.

The ORCID record must be public; private works are not returned by the API.

**Parameters:**
- `orcid` (str): ORCID iD or profile URL (e.g., `"0000-0002-2315-9247"` or `"https://orcid.org/0000-0002-2315-9247"`)
- `max_terms` (int, optional): Maximum number of search terms (default: 15, capped at 50)
- `min_year` (int, optional): Only consider works published in or after this year
- `max_works` (int, optional): Maximum works sent to the LLM (default: 100, capped at 500)

**Returns:** JSON with:
- Normalized ORCID iD and display name
- Total public works and works considered after filtering
- ORCID author keywords
- Suggested terms with a reason and source (`llm` or `fallback`)
- Up to ten recent work titles
- A warning when no terms could be derived

**Example use case:** "Find papers that are interesting to me based on my ORCID record 0000-0002-2315-9247."

## Starting the MCP Server

### Basic Usage

Start the server with default settings:

```bash
abstracts-explorer serve mcp
```

This starts the server on `http://127.0.0.1:8000/mcp` with streamable HTTP transport.

### Custom Host and Port

Start on a custom host and port:

```bash
abstracts-explorer serve mcp --host 0.0.0.0 --port 8080
```

### STDIO Transport

For local CLI integration, use stdio transport:

```bash
abstracts-explorer serve mcp --transport stdio
```

## Configuration

The MCP server uses the same configuration as the rest of Abstracts Explorer. Configure via `.env` file:

```bash
# Database Configuration
PAPER_DB=abstracts.db
EMBEDDING_DB=chroma_db
COLLECTION_NAME=papers

# LLM Backend (for embeddings and ORCID search-term suggestions)
LLM_BACKEND_URL=http://localhost:1234
EMBEDDING_MODEL=text-embedding-qwen3-embedding-4b
CHAT_MODEL=diffbot-small-xl-2508
```

## Integration with LLM Assistants

The MCP server can be integrated with any MCP-compatible LLM assistant or client. **It's now automatically integrated into the RAG chat system.**

### RAG Chat Integration (Recommended)

The easiest way to use MCP tools is through the RAG chat system, which automatically calls the appropriate tools:

```python
from abstracts_explorer.rag import RAGChat
from abstracts_explorer.embeddings import EmbeddingsManager
from abstracts_explorer.database import DatabaseManager

# Initialize components
em = EmbeddingsManager()
db = DatabaseManager()

# Create RAG chat (MCP tools enabled by default)
chat = RAGChat(em, db, enable_mcp_tools=True)

# The LLM automatically uses clustering tools when appropriate
response = chat.query("What are the main topics at NeurIPS?")
# Internally calls get_conference_topics() and uses results

response = chat.query("How have transformers evolved over time?")
# Internally calls get_topic_evolution(topic_keywords="transformers")
```

See the [RAG API documentation](api/rag.md#mcp-clustering-tools-integration) for more details.

### Claude Desktop Integration

Add to Claude Desktop MCP configuration:

```json
{
  "mcpServers": {
    "abstracts-explorer": {
      "command": "abstracts-explorer",
      "args": ["serve", "mcp", "--transport", "stdio"]
    }
  }
}
```

### Example Tool Call

When an LLM assistant needs to analyze topics, it can call:

```json
{
  "tool": "get_conference_topics",
  "arguments": {
    "conferences": ["NeurIPS"],
    "years": [2024, 2025]
  }
}
```

The server will:
1. Load pre-generated clustering results from the database cache
2. Analyze topics in each cluster
3. Return structured JSON results

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

1. **Embeddings are created**: Run `abstracts-explorer embeddings create` first
2. **Database exists**: Download papers with `abstracts-explorer data download`
3. **MCP package installed**: `uv sync` or `pip install "mcp>=2.0.0,<3"`

## Troubleshooting

### "No embeddings found"

Make sure to create embeddings first:

```bash
abstracts-explorer embeddings create
```

### "Failed to connect to database"

Check that the database paths in `.env` are correct:

```bash
# .env
PAPER_DB=data/abstracts.db
EMBEDDING_DB=chroma_db
```

### Port already in use

Change the port:

```bash
abstracts-explorer serve mcp --port 8001
```

## Advanced Usage

### Clustering Configuration

The MCP tools read pre-generated clustering results from the database cache. The
pipeline is fixed to BERTopic with Agglomerative clustering (ward linkage,
distance threshold 0.5) on 5-D UMAP embeddings and 2-D UMAP visualization.
`get_conference_topics` uses hierarchy level 6 (clamped to the available
levels); the web UI defaults to level 4. Generate the cache with:

```bash
abstracts-explorer clustering cache populate --conference NeurIPS
```

### Filtering by Conference

Analyze specific conferences:

```python
# get_topic_evolution arguments
{
  "topic_keywords": "neural networks",
  "conferences": ["NeurIPS"],
  "start_year": 2020,
  "end_year": 2024
}
```

## See Also

- [CLI Reference](cli_reference.md) - Command-line interface documentation
- [Clustering Guide](api/clustering.md) - Clustering and visualization guide
- [RAG Chat](api/rag.md) - Using RAG chat with MCP tools
- [Configuration](configuration.md) - Environment configuration options

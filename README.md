# Abstracts Explorer

A package to download conference data and search it with LLM-based semantic search including document retrieval and question answering.

## Features

- 📥 Download conference data from various sources (NeurIPS, ICLR, ICML, ML4PS)
- 💾 Store data in SQL database (SQLite or PostgreSQL) with efficient indexing
- 🔍 Search papers by keywords, track, and other attributes
- 🤖 Generate text embeddings for semantic search
- 🔎 Find similar papers using AI-powered semantic similarity
- 💬 Interactive RAG chat to ask questions about papers
- 🎨 **NEW: Cluster and visualize paper embeddings with interactive plots**
- 🌐 Web interface for browsing and searching papers
- 🔌 **NEW: MCP server for LLM-based cluster analysis**
- 🗄️ **NEW: Multi-database backend support (SQLite and PostgreSQL)**
- ⚙️ Environment-based configuration with `.env` file support

## Installation

**Requirements:** Python 3.11+, [uv](https://docs.astral.sh/uv/) package manager, Node.js 14+ (for web UI)

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/thawn/neurips-abstracts.git
cd abstracts-explorer

# Install dependencies
uv sync --all-extras

# Install Node.js dependencies for web UI
npm install
npm run install:vendor
```

📖 **[Full Installation Guide](docs/installation.md)**

## Configuration

Create a `.env` file to customize settings:

```bash
cp .env.example .env
# Edit .env with your preferred settings
```

📖 **[Configuration Guide](docs/configuration.md)** - Complete list of settings and options

### Database Backend

Abstracts Explorer supports both SQLite and PostgreSQL backends:

```bash
# Option 1: SQLite (default, no additional setup required)
PAPER_DB_PATH=data/abstracts.db

# Option 2: PostgreSQL (requires PostgreSQL server)
DATABASE_URL=postgresql://user:password@localhost/abstracts
```

**PostgreSQL Setup:**

```bash
# Install PostgreSQL support
uv sync --extra postgres

# Create database
createdb abstracts

# Configure in .env
DATABASE_URL=postgresql://user:password@localhost/abstracts
```

📖 See [Configuration Guide](docs/configuration.md) for more database options

## Quick Start

### Download Conference Data

```bash
# Download NeurIPS 2025 papers
abstracts-explorer download --year 2025 --output data/abstracts.db
```

### Generate Embeddings for Semantic Search

```bash
# Requires LM Studio running with embedding model loaded
abstracts-explorer create-embeddings --db-path data/abstracts.db
```

### Cluster and Visualize Embeddings

```bash
# Cluster embeddings using K-Means (PCA reduction)
abstracts-explorer cluster-embeddings --n-clusters 8 --output clusters.json

# Cluster using t-SNE and DBSCAN
abstracts-explorer cluster-embeddings \
  --reduction-method tsne \
  --clustering-method dbscan \
  --eps 0.5 \
  --min-samples 5 \
  --output clusters.json

# The web UI includes an interactive cluster visualization tab!
```

### Start MCP Server for Cluster Analysis

```bash
# Start MCP server for LLM-based cluster analysis
abstracts-explorer mcp-server

# The MCP server provides tools to analyze clustered papers:
# - Get most frequently mentioned topics
# - Analyze topic evolution over years
# - Find recent developments in topics
# - Generate cluster visualizations
```

### Start Web Interface

```bash
abstracts-explorer web-ui
# Open http://127.0.0.1:5000 in your browser
```

📖 **[Usage Guide](docs/usage.md)** - Detailed examples and workflows  
📖 **[CLI Reference](docs/cli_reference.md)** - Complete command-line documentation  
📖 **[API Reference](docs/api/modules.md)** - Python API documentation

## Web Interface

The web UI provides an intuitive interface for browsing and searching papers:

- 🔍 **Search**: Keyword and AI-powered semantic search  
- 💬 **Chat**: Interactive RAG chat with query rewriting
- ⭐ **Ratings**: Save and organize interesting papers
- 📊 **Filters**: Filter by track, decision, event type, and more
- 🎨 **Clusters**: Interactive visualization of paper embeddings (NEW!)

```bash
abstracts-explorer web-ui
# Open http://127.0.0.1:5000
```

![Web UI Screenshot](docs/images/screenshot.png)
*The web interface provides an intuitive way to search and explore conference papers*

## Python API Examples

### Download and Search Papers

```python
from abstracts_explorer.plugins import get_plugin
from abstracts_explorer import DatabaseManager

# Download papers
neurips_plugin = get_plugin('neurips')
papers_data = neurips_plugin.download(year=2025)

# Load into database and search
with DatabaseManager("data/abstracts.db") as db:
    db.create_tables()
    db.add_papers(papers_data)
    
    # Search papers
    papers = db.search_papers(keyword="deep learning", limit=5)
    for paper in papers:
        print(f"{paper['title']} by {paper['authors']}")
```

### Semantic Search with Embeddings

```python
from abstracts_explorer import EmbeddingsManager

with EmbeddingsManager() as em:
    em.create_collection()
    em.embed_from_database("data/abstracts.db")
    
    # Find similar papers
    results = em.search_similar(
        "transformers for natural language processing",
        n_results=5
    )
```

### Cluster and Visualize Embeddings

```python
from abstracts_explorer.clustering import perform_clustering

# Perform complete clustering pipeline
results = perform_clustering(
    embeddings_path="chroma_db",
    reduction_method="tsne",      # or "pca"
    n_components=2,
    clustering_method="kmeans",    # or "dbscan", "agglomerative"
    n_clusters=8,
    output_path="clusters.json"
)

# Access clustering results
print(f"Found {results['statistics']['n_clusters']} clusters")
for point in results['points']:
    print(f"Paper: {point['title']} -> Cluster {point['cluster']}")
```

📖 **[Complete Usage Guide](docs/usage.md)** - More examples and workflows

## Documentation

📚 **[Full Documentation](docs/index.md)** - Complete documentation built with Sphinx

### Quick Links

- **[Installation Guide](docs/installation.md)** - Detailed installation instructions
- **[Usage Guide](docs/usage.md)** - Examples and workflows  
- **[Configuration Guide](docs/configuration.md)** - Environment variables and settings
- **[CLI Reference](docs/cli_reference.md)** - Command-line interface documentation
- **[Plugins Guide](docs/plugins.md)** - Plugin system and conference downloaders
- **[API Reference](docs/api/modules.md)** - Python API documentation
- **[Contributing Guide](docs/contributing.md)** - Development setup and guidelines

## Development

```bash
# Install with development dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Run linters
ruff check src/ tests/
mypy src/ --ignore-missing-imports
```

📖 **[Contributing Guide](docs/contributing.md)** - Complete development documentation

## Contributing

Contributions are welcome! Please read our [Contributing Guide](docs/contributing.md) for details on:

- Development setup
- Running tests and linters  
- Code style and conventions
- Submitting pull requests

## License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

## Support

For issues, questions, or contributions:
- 🐛 [Report issues](https://github.com/thawn/neurips-abstracts/issues)
- 💬 [Discussions](https://github.com/thawn/neurips-abstracts/discussions)
- 📧 Contact the maintainers

# Abstracts Explorer

A package to download conference data and search it with LLM-based semantic search including document retrieval and question answering.

## Features

- 📥 Download conference data from various sources (NeurIPS, ICLR, ICML, ML4PS)
- 💾 Store data in SQLite database with efficient indexing
- 🔍 Search papers by keywords, track, and other attributes
- 🤖 Generate text embeddings for semantic search
- 🔎 Find similar papers using AI-powered semantic similarity
- 💬 Interactive RAG chat to ask questions about papers
- 🌐 Web interface for browsing and searching papers
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

```bash
abstracts-explorer web-ui
# Open http://127.0.0.1:5000
```

![Web UI Screenshot](https://github.com/user-attachments/assets/25b88d66-cd12-4564-bad6-9312861d51d6)
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

---
applyTo: "**"
---

# AI Coding Instructions

This file contains instructions and conventions for AI assistants working on this project.

## Project Overview

**abstracts-explorer** is a Python package to download conference data and search it with LLM-based semantic search, including document retrieval, question answering, clustering, and visualization. It supports multiple database backends (SQLite, PostgreSQL) and includes a web interface and MCP server for LLM integration.

## Package Manager: uv

**IMPORTANT**: This project uses **uv** for package management, NOT pip/venv.

### Quick Reference

- **Install dependencies**: `uv sync` or `uv sync --all-extras`
- **Run commands**: `uv run pytest`, `uv run abstracts-explorer`, etc.
- **Add dependencies**: `uv add package-name`
- **Virtual environment**: Automatically created in `.venv/`

### Why uv?

- **10-100x faster** than pip for installation and dependency resolution
- **Automatic** virtual environment management
- **Reliable** dependency resolution with lock files
- **Modern** tooling with better error messages

### When Writing Code or Instructions

Always use uv commands:
- ✅ `uv sync --extra dev` (correct)
- ❌ `pip install -e ".[dev]"` (outdated)
- ✅ `uv run pytest` (correct)
- ❌ `pytest` (may not work without activation)

See the "Package Management with uv" section below for complete documentation.

## Code Style & Conventions

### Python Style

- **PEP 8 compliance**: Follow Python style guidelines
- **Line length**: 88 characters (Black default)
- **Type hints**: Use type hints for all function signatures
- **Docstrings**: NumPy-style docstrings for all public functions/classes/methods

### Docstring Format

All public functions, classes, and methods must have NumPy-style docstrings:

```python
def example_function(param1: str, param2: int = 10) -> bool:
    """
    Brief one-line description.

    More detailed description if needed. Can span multiple lines
    and include additional context about the function's purpose.

    Parameters
    ----------
    param1 : str
        Description of param1
    param2 : int, optional
        Description of param2 (default: 10)

    Returns
    -------
    bool
        Description of what the function returns

    Raises
    ------
    ValueError
        When param1 is empty
    RuntimeError
        When operation fails

    Examples
    --------
    >>> example_function("test")
    True
    >>> example_function("test", param2=20)
    True

    Notes
    -----
    Additional notes about implementation, performance, or usage.

    See Also
    --------
    related_function : Related functionality
    AnotherClass : Related class
    """
    pass
```

### Code Organization

```
src/abstracts_explorer/    # Main package
├── __init__.py           # Package initialization
├── _version.py           # Auto-generated version (hatch-vcs)
├── database.py           # Database manager (SQLAlchemy-based)
├── db_models.py          # SQLAlchemy ORM models (Paper, EmbeddingsMetadata, ClusteringCache)
├── embeddings.py         # ChromaDB embeddings management
├── clustering.py         # Clustering & visualization (5 algorithms, auto-labeling)
├── rag.py                # RAG chat interface with MCP integration
├── mcp_server.py         # MCP server for LLM cluster analysis
├── mcp_tools.py          # MCP tools for topic/trend analysis
├── export_utils.py       # Paper export utilities (ZIP, JSON, Markdown)
├── config.py             # Configuration management
├── cli.py                # Command-line interface
├── plugin.py             # Plugin system and helpers
├── paper_utils.py        # Paper formatting utilities
├── plugins/              # Plugin implementations
│   ├── __init__.py
│   ├── json_conference_downloader.py  # Base class for JSON downloaders
│   ├── neurips_downloader.py
│   ├── iclr_downloader.py
│   ├── icml_downloader.py
│   └── ml4ps_downloader.py
└── web_ui/               # Flask web interface
    ├── __init__.py
    ├── app.py            # Flask app (25+ API routes)
    ├── static/           # CSS, JS (modular ES6), vendor files
    └── templates/        # HTML templates

tests/                     # Test suite (19 test files)
├── conftest.py           # Shared fixtures
├── helpers.py            # Shared helper functions
├── test_cli.py           # Tests for cli.py
├── test_config.py        # Tests for config.py
├── test_database.py      # Tests for database.py
├── test_embeddings.py    # Tests for embeddings.py (includes ChromaDB)
├── test_clustering.py    # Tests for clustering.py
├── test_chromadb_metadata.py  # Embedding metadata tracking tests
├── test_multi_database.py     # PostgreSQL support tests
├── test_export_utils.py  # Tests for export_utils.py
├── test_mcp_server.py    # Tests for mcp_server.py
├── test_mcp_tools.py     # Tests for mcp_tools.py
├── test_paper_utils.py   # Tests for paper_utils.py
├── test_plugin.py        # Tests for plugin.py (plugin system)
├── test_plugins_iclr.py  # Tests for plugins/iclr_downloader.py
├── test_plugins_ml4ps.py # Tests for plugins/ml4ps_downloader.py
├── test_rag.py           # Tests for rag.py
├── test_web_ui.py        # Tests for web_ui/app.py (unit tests)
├── test_web_integration.py # Web UI integration tests
├── test_web_e2e.py       # Web UI end-to-end tests
└── test_integration.py   # Integration tests across modules

docs/                      # Sphinx documentation
├── conf.py               # Sphinx configuration
├── index.md              # Documentation homepage
├── api/                  # API reference (auto-generated)
├── docker.md             # Docker/Podman deployment guide
├── mcp_server.md         # MCP server documentation
├── plugins.md            # Plugin system guide
└── *.md                  # User guide pages
```

## Testing Requirements

### Test Framework

- **Framework**: pytest
- **Coverage**: Use pytest-cov, aim for >90% coverage
- **Mocking**: Use pytest-mock for external dependencies

### Test Organization Principle

**One test file per source module**: Each source module should have exactly one corresponding test file with the same base name. This keeps tests organized and easy to find.

Examples:
- `src/abstracts_explorer/database.py` → `tests/test_database.py`
- `src/abstracts_explorer/plugin.py` → `tests/test_plugin.py` (includes all plugin-related tests)
- `src/abstracts_explorer/web_ui/app.py` → `tests/test_web_ui.py` (unit tests)

**Exception for test types**: Different types of tests (unit, integration, e2e) may have separate files:
- `test_integration.py` - Cross-module integration tests
- `test_web_integration.py` - Web UI integration tests  
- `test_web_e2e.py` - End-to-end browser tests

### Test Organization Structure

1. **Unit tests**: Test individual functions/methods in isolation
2. **Integration tests**: Test component interactions
3. **Fixtures**: Use fixtures for common setup (databases, temp files)
4. **Mocking**: Mock external APIs, LLM backends, file I/O

### Writing Tests

```python
import pytest
from abstracts_explorer.database import DatabaseManager

@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary test database."""
    db_path = tmp_path / "test.db"
    db = DatabaseManager(str(db_path))
    yield db
    db.close()

def test_add_paper(temp_db):
    """Test adding a paper to the database."""
    paper_data = {
        'openreview_id': 'test123',
        'title': 'Test Paper',
        'abstract': 'Test abstract',
        'year': 2025,
    }
    
    paper_id = temp_db.add_paper(paper_data)
    assert paper_id is not None
    
    # Verify paper was added
    paper = temp_db.get_paper_by_id(paper_id)
    assert paper['title'] == 'Test Paper'
```

### Test Coverage Guidelines

- All new functions must have tests
- Aim for >80% code coverage
- Test both success and error cases
- Mock external dependencies (API calls, LLM backends)
- Use `pytest.mark.skipif` for conditional tests (e.g., LM Studio)

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/abstracts_explorer --cov-report=html

# Run specific test file
uv run pytest tests/test_database.py

# Run specific test
uv run pytest tests/test_database.py::test_add_paper

# Verbose output
uv run pytest -v

# Show print statements
uv run pytest -s
```

## Package Management with uv

This project uses **uv** for fast, reliable Python package management.

### Why uv?

- **Fast**: 10-100x faster than pip
- **Reliable**: Consistent dependency resolution
- **Compatible**: Works with existing pip/PyPI packages
- **Modern**: Built-in virtual environment management

### Basic Commands

```bash
# Create virtual environment and install all dependencies
uv sync

# Install with specific extra dependencies
uv sync --extra dev
uv sync --extra web
uv sync --extra docs
uv sync --all-extras  # Install all optional dependencies

# Add a new dependency
uv add requests

# Add a development dependency
uv add --dev pytest

# Run commands in the virtual environment
uv run python script.py
uv run pytest
uv run abstracts-explorer --help

# Activate the virtual environment manually (optional)
source .venv/bin/activate  # Unix/macOS
.venv\Scripts\activate     # Windows
```

### Installation

```bash
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

## Configuration System

### Environment Variables

The package uses a flexible configuration system:

1. **Built-in defaults** (in `config.py`)
2. **`.env` file** (in project root)
3. **Environment variables**
4. **CLI arguments** (when applicable)

### Configuration File (.env)

```bash
# Chat/LLM Settings
CHAT_MODEL=gemma-3-4b-it-qat
CHAT_TEMPERATURE=0.7
CHAT_MAX_TOKENS=1000

# Embedding Settings
EMBEDDING_MODEL=text-embedding-qwen3-embedding-4b

# Backend Settings
LLM_BACKEND_URL=http://localhost:1234
LLM_BACKEND_AUTH_TOKEN=

# Database Settings
PAPER_DB=data/abstracts.db  # SQLite (default)
# PAPER_DB=postgresql://user:password@localhost/abstracts  # PostgreSQL option
EMBEDDING_DB_PATH=chroma_db

# RAG Settings
COLLECTION_NAME=papers
MAX_CONTEXT_PAPERS=5
```

### Using Configuration

```python
from abstracts_explorer.config import get_config

config = get_config()
print(config.chat_model)
print(config.llm_backend_url)
```

## Dependencies

### Core Dependencies

- **requests**: API calls to OpenReview
- **chromadb**: Vector embeddings storage
- **pydantic**: Data validation
- **beautifulsoup4**: HTML parsing
- **openai**: OpenAI-compatible API client
- **scikit-learn**: Machine learning utilities
- **numpy**: Numerical computations
- **mcp**: Model Context Protocol for LLM integration
- **sqlalchemy**: ORM for database abstraction (SQLite/PostgreSQL)
- **psycopg2-binary**: PostgreSQL database driver
- **numba**: JIT compilation for clustering performance
- **llvmlite**: LLVM backend for Numba
- **umap-learn**: UMAP dimensionality reduction
- **scikit-fuzzy**: Fuzzy C-Means clustering

### Development Dependencies

Install with `uv sync --extra dev`:

- **pytest**: Testing framework
- **pytest-cov**: Coverage reporting
- **pytest-mock**: Mocking support
- **selenium**: Browser automation testing
- **webdriver-manager**: Browser driver management

### Web Dependencies

Install with `uv sync --extra web`:

- **flask**: Web framework
- **flask-cors**: CORS support
- **waitress**: Production WSGI server

### Documentation Dependencies

Install with `uv sync --extra docs`:

- **sphinx**: Documentation generator
- **sphinx-rtd-theme**: Read the Docs theme
- **myst-parser**: Markdown support
- **sphinx-autodoc-typehints**: Type hints in docs
- **linkify-it-py**: URL auto-linking

## Documentation

### Sphinx Documentation

Build documentation:

```bash
cd docs
uv run make html
open _build/html/index.html
```

### Documentation Structure

- **User Guide**: Installation, configuration, usage
- **API Reference**: Auto-generated from docstrings
- **CLI Reference**: Command-line interface documentation
- **Contributing**: Development guidelines
- **Changelog**: Links to detailed change logs
- **Docker Guide**: Container deployment with Docker/Podman
- **MCP Server**: Model Context Protocol integration
- **Plugins Guide**: Plugin system and conference downloaders

### Updating Documentation

1. **User guide**: Edit Markdown files in `docs/`
2. **API docs**: Update docstrings in source code
3. **Rebuild**: Run `make html` in `docs/` directory

## Web UI Architecture

### Framework & Structure

- **Backend**: Flask 3.0+ with CORS support
- **Frontend**: Modular ES6 JavaScript (15 modules)
- **Styling**: TailwindCSS with vendor libraries (Chart.js, Plotly.js, Marked.js)
- **Production Server**: Waitress WSGI server

### API Routes (25+ endpoints)

**Paper Operations:**
- `GET /api/papers` - List all papers
- `GET /api/papers/<id>` - Get paper details
- `POST /api/search` - Keyword/semantic search
- `POST /api/papers/<id>/rating` - Rate paper

**Semantic Search:**
- `POST /api/semantic-search` - Vector similarity search
- `GET /api/embedding-model` - Check embedding model

**RAG Chat:**
- `POST /api/chat` - Chat with conversation history
- `POST /api/reset-conversation` - Clear history

**Clustering:**
- `POST /api/clusters/compute` - Generate clusters
- `GET /api/clusters/cached` - Get cached results

**Filters & Stats:**
- `GET /api/stats` - Database statistics
- `GET /api/filters` - Available filter values
- `GET /api/available-filters` - Filter metadata

**Export:**
- `POST /api/export/interesting-papers` - Export papers (ZIP/JSON/Markdown)

### Frontend Architecture (Modular ES6)

**Feature Modules (5):**
1. `search.js` - Search functionality
2. `chat.js` - RAG chat interface
3. `clustering.js` - Clustering visualization
4. `filters.js` - Filter panel
5. `interesting-papers.js` - Saved papers management

**UI Component Modules (2):**
1. `tabs.js` - Tab navigation
2. `paper-card.js` - Paper display components

**Utility Modules (8):**
1. `state.js` - Centralized state management (200+ lines)
2. `api-utils.js` - API client
3. `cluster-utils.js` - Clustering helpers
4. `constants.js` - Configuration constants
5. `dom-utils.js` - DOM manipulation helpers
6. `markdown-utils.js` - Markdown rendering
7. `sort-utils.js` - Sorting utilities
8. `ui-utils.js` - UI helpers

**Architecture**: Refactored from 2700-line monolithic app.js to 15 focused modules

### UI Tabs

1. **Search**: Keyword and semantic search with filters
2. **Chat**: RAG interface with conversation history
3. **Interesting Papers**: Rated/saved papers with export
4. **Clusters**: Interactive visualization with Plotly.js

## Git Workflow

### Branch Strategy

- **main**: Production-ready code
- **feature/***: New features
- **bugfix/***: Bug fixes
- **docs/***: Documentation updates

### Commit Messages

Use clear, descriptive commit messages:

```
Add feature: RAG chat interface with LM Studio

- Implement RAGChat class with query/chat methods
- Add conversation history management
- Integrate with embeddings for context retrieval
- Add comprehensive tests with 97% coverage
```

### Files to Ignore

See `.gitignore`:
- `__pycache__/`, `*.pyc`: Python cache
- `.venv/`, `venv/`, `env/`: Virtual environments (uv uses `.venv` by default)
- `.env`: Configuration secrets
- `*.db`: Database files
- `chroma_db/`: Embeddings database
- `_build/`: Documentation build
- `.pytest_cache/`: Test cache
- `htmlcov/`: Coverage reports
- `uv.lock`: Lock file (should be committed for applications, optional for libraries)

## External Integrations

### LM Studio or Blablador

- **Purpose**: LLM backend for embeddings, chat, and clustering
- **URL**: http://localhost:1234 (LM Studio default) or https://blablador.jsc.fz-juelich.de/v1 (Blablador)
- **API**: OpenAI-compatible endpoints
- **Models**: Configurable via CHAT_MODEL and EMBEDDING_MODEL
- **Authentication**: Optional token via LLM_BACKEND_AUTH_TOKEN

### OpenReview API

- **Purpose**: Download conference paper data (NeurIPS, ICLR, ICML)
- **Base URL**: https://api.openreview.net
- **Rate Limits**: Respect API rate limits
- **Caching**: Use caching to reduce API calls

### ChromaDB

- **Purpose**: Vector embeddings storage for semantic search
- **Path**: Configurable via EMBEDDING_DB_PATH (default: chroma_db)
- **Collections**: Papers stored in collections
- **Persistence**: Automatic disk persistence
- **Metadata**: Tracks embedding model and parameters

### SQLAlchemy Database

- **Purpose**: Paper metadata storage with ORM
- **Backends**: SQLite (default) or PostgreSQL
- **Configuration**: Set via PAPER_DB environment variable
  - SQLite: `PAPER_DB=data/abstracts.db`
  - PostgreSQL: `PAPER_DB=postgresql://user:password@localhost/abstracts`
- **Models**: Paper, EmbeddingsMetadata, ClusteringCache

## Best Practices

### Code Quality

1. **Type hints**: Use for all function signatures
2. **Docstrings**: NumPy-style for all public APIs
3. **Error handling**: Use try-except with specific exceptions
4. **Logging**: Use informative log messages (not implemented yet)
5. **Constants**: Define at module level, use uppercase

### Performance

1. **Batch operations**: Process in batches where possible
2. **Caching**: Cache API responses and embeddings
3. **Database**: Use indexes for frequent queries
4. **Memory**: Be mindful of large datasets

### Security

1. **No secrets in code**: Use environment variables
2. **No .env in git**: Keep configuration private
3. **Input validation**: Validate user inputs
4. **SQL injection**: Use parameterized queries

### Testing

1. **Write tests first**: TDD when possible
2. **Mock externals**: Don't depend on external services
3. **Isolated tests**: Each test should be independent
4. **Clear assertions**: Make test intent obvious

## Common Tasks

### Adding a New Feature

1. Create feature branch
2. Implement feature with type hints and NumPy-style docstrings
3. Write comprehensive tests (unit + integration)
4. Update documentation (docstrings + user guide)
5. Run tests: `uv run pytest --cov=src/abstracts_explorer`
6. Build docs: `cd docs && uv run make html`
7. Commit and create pull request

### Fixing a Bug

1. Create bugfix branch
2. Write failing test that reproduces bug
3. Fix the bug
4. Verify test passes
5. Add regression test if needed
6. Commit and create pull request

### Updating Documentation

1. Edit relevant Markdown files in `docs/`
2. Or update docstrings in source code
3. Rebuild: `cd docs && uv run make html`
4. Review in browser
5. Commit changes

### Adding a New Conference Plugin

1. Create plugin in `src/abstracts_explorer/plugins/`
2. Inherit from `DownloaderPlugin` or use `LightweightDownloaderPlugin`
3. Implement `download()` method
4. Register plugin in `plugins/__init__.py`
5. Add tests in `tests/test_plugins_<name>.py`
6. Update `docs/plugins.md` with usage examples

### Common CLI Commands

```bash
# Download papers
abstracts-explorer download --conference neurips --year 2025

# Create embeddings
abstracts-explorer create-embeddings

# Cluster embeddings
abstracts-explorer cluster-embeddings --n-clusters 8 --output clusters.json

# Start web UI
abstracts-explorer web-ui

# Start MCP server
abstracts-explorer mcp-server

# Export papers
abstracts-explorer export --format zip --output papers.zip

# List available plugins
abstracts-explorer list-plugins

# Get help
abstracts-explorer --help
abstracts-explorer download --help
```

## Troubleshooting

### Tests Failing

- Check virtual environment is activated (or use `uv run`)
- Verify all dependencies installed: `uv sync --all-extras`
- Check configuration in `.env`
- Review test output for specific errors

### LM Studio Integration Tests Skipping

- Ensure LM Studio or Blablador is running
- Load the configured chat model (CHAT_MODEL)
- Load the configured embedding model (EMBEDDING_MODEL)
- Verify URL in .env matches backend (default: http://localhost:1234)
- Set LLM_BACKEND_AUTH_TOKEN if using Blablador

### Documentation Build Errors

- Install doc dependencies: `uv sync --extra docs`
- Check for syntax errors in Markdown files
- Clean build: `make clean && uv run make html`
- Review error messages for specific issues

### Import Errors

- Use `uv run` to run commands in the virtual environment
- Or activate virtual environment: `source .venv/bin/activate`
- Reinstall package: `uv sync`
- Check Python path
- Verify module structure

### Database Errors

- **SQLite**: Check PAPER_DB path exists and is writable
- **PostgreSQL**: Verify connection string, ensure server is running
- **Migration**: Run `abstracts-explorer` commands to auto-create tables
- **Permissions**: Ensure write access to database directory

### Clustering Performance

- **Slow clustering**: Enable caching with ClusteringCache
- **Memory issues**: Reduce n_components or use PCA instead of t-SNE/UMAP
- **Numba errors**: Ensure numba and llvmlite are installed correctly

## Package Information

- **Name**: abstracts-explorer
- **Version**: Dynamic (managed by hatch-vcs)
- **Python**: >=3.11
- **License**: Apache-2.0
- **CLI Command**: `abstracts-explorer`

## Key Features

### Paper Management
- Download conference data from NeurIPS, ICLR, ICML, ML4PS
- Store in SQLite (default) or PostgreSQL database
- Search by keywords, track, decision status
- Export papers in ZIP, JSON, or Markdown formats

### Semantic Search & RAG
- Generate embeddings with ChromaDB
- Find similar papers using vector similarity
- RAG chat with conversation history
- Automatic MCP tool integration for clustering questions

### Clustering & Visualization
- 5 clustering algorithms: K-Means, DBSCAN, Agglomerative, Spectral, Fuzzy C-Means
- 3 dimensionality reduction methods: PCA, t-SNE, UMAP
- LLM-powered automatic cluster labeling
- Interactive visualization in web UI
- Database-backed caching for performance

### MCP Server
- FastMCP-based server for LLM integration
- Tools for topic analysis, trend detection, recent developments
- Automatic integration with RAG chat
- Standalone mode for external LLM clients

### Web Interface
- Flask-based with 25+ API routes
- 4 main tabs: Search, Chat, Interesting Papers, Clusters
- Modular ES6 frontend (15 modules)
- Real-time clustering visualization
- Paper rating and export functionality

## Key Design Decisions

### Database Architecture

- **ORM**: SQLAlchemy 2.0+ with declarative base
- **Multi-backend**: SQLite (default) and PostgreSQL support
- **Models**: 
  - `Paper` - Main paper table with indexed fields (title, year, openreview_id)
  - `EmbeddingsMetadata` - Tracks embedding model and parameters
  - `ClusteringCache` - Caches clustering results for performance
- **Migration**: Uses SQLAlchemy's metadata.create_all() for schema creation
- **Indexes**: On year, openreview_id, title for fast queries

### Configuration Priority

1. CLI arguments (highest priority)
2. Environment variables
3. .env file
4. Built-in defaults (lowest priority)

### Testing Strategy

- **Unit tests**: Mock all external dependencies (APIs, LLM backends, databases)
- **Integration tests**: Conditional on LM Studio/Blablador availability
- **E2E tests**: Browser automation with Selenium for web UI
- **Skip conditions**: Tests skip gracefully when dependencies unavailable
- **Coverage target**: >80% code coverage
- **Test markers**: `@pytest.mark.slow`, `@pytest.mark.integration`, `@pytest.mark.e2e`

### Documentation Approach

- **Markdown first**: Use Markdown for all user-facing docs
- **Auto-generate API**: Extract from NumPy-style docstrings
- **Examples included**: Every feature has usage examples
- **Multiple formats**: Support HTML, PDF, ePub via Sphinx

### Clustering Architecture

- **Reduction methods**: PCA, t-SNE, UMAP (configurable)
- **Clustering algorithms**: K-Means, DBSCAN, Agglomerative, Spectral, Fuzzy C-Means
- **Auto-labeling**: LLM-powered cluster naming via TF-IDF keywords
- **Caching**: Database-backed caching to avoid recomputation
- **Visualization**: Interactive Plotly charts in web UI

### MCP Integration

- **Server**: FastMCP-based server for LLM cluster analysis
- **Auto-integration**: RAG chat automatically uses MCP tools when appropriate
- **Tools**: Topic frequency, trend analysis, recent developments, visualization

## Contact & Resources

- **Documentation**: `docs/_build/html/index.html`
- **Tests**: `tests/`
- **Configuration**: `.env` (create from `.env.example`)

## Docker/Podman Deployment

### Quick Start

```bash
# 1. Create .env file with your configuration
echo "LLM_BACKEND_AUTH_TOKEN=your_token_here" > .env

# 2. Download docker-compose.yml
curl -o docker-compose.yml https://raw.githubusercontent.com/thawn/abstracts-explorer/refs/heads/main/docker-compose.yml

# 3. Start services
docker compose up -d    # or: podman-compose up -d

# 4. Access at http://localhost:5000
```

### Service Architecture

Three services on internal network (`abstracts-network`):
- **abstracts-explorer** (port 5000) - Web UI, accessible from host
- **postgres** (port 5432) - PostgreSQL database, internal only
- **chromadb** (port 8000) - Vector database, internal only

**Security**: Database ports NOT exposed to host. Inter-service communication via Docker network.

### Key Environment Variables (.env.docker)

```bash
DATABASE_URL=postgresql://abstracts:abstracts_password@postgres:5432/abstracts
EMBEDDING_DB_URL=http://chromadb:8000
LLM_BACKEND_URL=http://host.docker.internal:1234
CHAT_MODEL=gemma-3-4b-it-qat
EMBEDDING_MODEL=text-embedding-qwen3-embedding-4b
```

### Testing PR Changes with Docker

When testing PR changes, use PR-specific image tags:

```bash
# 1. Update docker-compose.yml (line 7)
image: ghcr.io/thawn/abstracts-explorer:pr-<number>

# 2. Pull and start
docker compose pull
docker compose up -d

# 3. Check health
docker compose ps
curl http://localhost:5000/health

# 4. View logs
docker compose logs abstracts-explorer

# 5. Clean up
docker compose down -v
```

**PR Image Tags**: Format `ghcr.io/thawn/abstracts-explorer:pr-<number>`, built automatically on push.

### Common Commands

```bash
# Execute CLI inside container
docker compose exec abstracts-explorer abstracts-explorer --help

# Interactive shell
docker compose exec -it abstracts-explorer /bin/bash

# Restart service
docker compose restart abstracts-explorer

# Real-time logs
docker compose logs -f abstracts-explorer
```

### Troubleshooting

**ChromaDB health check fails**: Check logs with `docker compose logs chromadb`

**Cannot access PostgreSQL**: Port not exposed intentionally. Use `docker compose exec abstracts-explorer psql ...`

**Web UI unavailable**: Wait ~10 seconds for health checks, then check logs

---

**Last Updated**: February 3, 2026

This file should be kept up-to-date as the project evolves. When making significant changes to conventions or structure, update this file accordingly.

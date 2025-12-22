---
applyTo: "**"
---

# AI Coding Instructions

This file contains instructions and conventions for AI assistants working on this project.

## Project Overview

**neurips-abstracts** is a Python package for downloading, storing, and analyzing NeurIPS conference paper abstracts with vector embeddings and RAG (Retrieval-Augmented Generation) capabilities.

## Package Manager: uv

**IMPORTANT**: This project uses **uv** for package management, NOT pip/venv.

### Quick Reference

- **Install dependencies**: `uv sync` or `uv sync --all-extras`
- **Run commands**: `uv run pytest`, `uv run neurips-abstracts`, etc.
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
src/neurips_abstracts/    # Main package
├── __init__.py           # Package initialization
├── database.py           # SQLite database operations
├── downloader.py         # OpenReview API integration
├── embeddings.py         # ChromaDB embeddings management
├── rag.py                # RAG chat interface
├── config.py             # Configuration management
└── cli.py                # Command-line interface

tests/                     # Test suite
├── test_database.py      # Database tests
├── test_downloader.py    # Downloader tests
├── test_embeddings.py    # Embeddings tests
├── test_rag.py           # RAG tests
├── test_config.py        # Configuration tests
└── test_integration.py   # Integration tests

docs/                      # Sphinx documentation
├── conf.py               # Sphinx configuration
├── index.md              # Documentation homepage
├── api/                  # API reference
└── *.md                  # User guide pages

changelog/                 # Detailed change documentation
├── 01_*.md               # Chronologically numbered
├── 02_*.md
└── ...
```

## Testing Requirements

### Test Framework

- **Framework**: pytest
- **Coverage**: Use pytest-cov, aim for >90% coverage
- **Mocking**: Use pytest-mock for external dependencies

### Test Organization

1. **Unit tests**: Test individual functions/methods in isolation
2. **Integration tests**: Test component interactions
3. **Fixtures**: Use fixtures for common setup (databases, temp files)
4. **Mocking**: Mock external APIs, LLM backends, file I/O

### Writing Tests

```python
import pytest
from neurips_abstracts.database import NeurIPSDatabase

@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary test database."""
    db_path = tmp_path / "test.db"
    db = NeurIPSDatabase(str(db_path))
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
uv run pytest --cov=src/neurips_abstracts --cov-report=html

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
uv run neurips-abstracts --help

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

# Database Paths
EMBEDDING_DB_PATH=chroma_db
PAPER_DB_PATH=neurips_2025.db

# RAG Settings
COLLECTION_NAME=neurips_papers
MAX_CONTEXT_PAPERS=5
```

### Using Configuration

```python
from neurips_abstracts.config import get_config

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
- **sqlite3**: Built-in, for paper database

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

### Updating Documentation

1. **User guide**: Edit Markdown files in `docs/`
2. **API docs**: Update docstrings in source code
3. **Rebuild**: Run `make html` in `docs/` directory

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

### LM Studio

- **Purpose**: LLM backend for embeddings and chat
- **URL**: http://localhost:1234 (default)
- **API**: OpenAI-compatible endpoints
- **Models**: Configurable via CHAT_MODEL and EMBEDDING_MODEL

### OpenReview API

- **Purpose**: Download NeurIPS paper data
- **Base URL**: https://api.openreview.net
- **Rate Limits**: Respect API rate limits
- **Caching**: Use caching to reduce API calls

### ChromaDB

- **Purpose**: Vector embeddings storage
- **Path**: Configurable via EMBEDDING_DB_PATH
- **Collections**: Papers stored in collections
- **Persistence**: Automatic disk persistence

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
2. Implement feature with type hints and docstrings
3. Write comprehensive tests (unit + integration)
4. Update documentation (docstrings + user guide)
5. Run tests: `uv run pytest --cov=src/neurips_abstracts`
6. Build docs: `cd docs && uv run make html`
7. Create changelog entry in `changelog/`
8. Commit and create pull request

### Fixing a Bug

1. Create bugfix branch
2. Write failing test that reproduces bug
3. Fix the bug
4. Verify test passes
5. Add regression test if needed
6. Update changelog if significant
7. Commit and create pull request

### Updating Documentation

1. Edit relevant Markdown files in `docs/`
2. Or update docstrings in source code
3. Rebuild: `cd docs && uv run make html`
4. Review in browser
5. Commit changes

## Troubleshooting

### Tests Failing

- Check virtual environment is activated (or use `uv run`)
- Verify all dependencies installed: `uv sync --all-extras`
- Check configuration in `.env`
- Review test output for specific errors

### LM Studio Integration Tests Skipping

- Ensure LM Studio is running
- Load the configured chat model (CHAT_MODEL)
- Load the configured embedding model (EMBEDDING_MODEL)
- Verify URL in .env matches LM Studio (default: http://localhost:1234)

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

## Package Information

- **Name**: neurips-abstracts
- **Version**: 0.1.0
- **Python**: >=3.11
- **License**: Apache-2.0
- **CLI Command**: `neurips-abstracts`

## Key Design Decisions

### Database Schema

- **Integer IDs**: Papers use integer primary keys
- **Authors table**: Separate table with position tracking
- **Timestamps**: Track creation time for papers
- **Indexes**: On year, openreview_id for fast queries

### Configuration Priority

1. CLI arguments (highest priority)
2. Environment variables
3. .env file
4. Built-in defaults (lowest priority)

### Testing Strategy

- **Unit tests**: Mock all external dependencies
- **Integration tests**: Conditional on LM Studio availability
- **Skip conditions**: Tests skip gracefully when dependencies unavailable
- **Coverage target**: >80% code coverage

### Documentation Approach

- **Markdown first**: Use Markdown for all user-facing docs
- **Auto-generate API**: Extract from docstrings
- **Examples included**: Every feature has usage examples
- **Multiple formats**: Support HTML, PDF, ePub

## Contact & Resources

- **Documentation**: `docs/_build/html/index.html`
- **Tests**: `tests/`
- **Changelog**: `changelog/`
- **Configuration**: `.env` (create from `.env.example`)

## Version History

- See `changelog/` directory for detailed version history
- When creating changelog entries, check the existing files and make sure that the new file continues the numbering sequence

---

**Last Updated**: December 21, 2025

This file should be kept up-to-date as the project evolves. When making significant changes to conventions or structure, update this file accordingly.

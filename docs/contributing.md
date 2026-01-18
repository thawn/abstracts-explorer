# Contributing

Thank you for your interest in contributing to Abstracts Explorer!

## Development Setup

### 1. Clone Repository

```bash
git clone <repository-url>
cd abstracts
```

### 2. Install uv

If you don't have uv installed yet:

```bash
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

### 3. Install Dependencies

```bash
# Install all dependencies including dev, web, and docs
uv sync --all-extras

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

## Docker Development Setup

For containerized development with Podman or Docker:

**Note:** The Docker image uses pre-built static vendor files (CSS/JS libraries) that are committed to the repository. Node.js is **not required** for production containers - it's only needed for local development if you want to rebuild these vendor files.

### Quick Start

```bash
# Build and start with Podman
podman-compose up -d

# Or with Docker
docker compose up -d

# Access at http://localhost:5000
```

### Running Commands in Container

```bash
# Execute CLI commands
podman-compose exec abstracts-explorer abstracts-explorer download --year 2025

# Run tests
podman-compose exec abstracts-explorer pytest

# Interactive shell
podman-compose exec -it abstracts-explorer /bin/bash
```

### Development Workflow

1. Make code changes locally
2. Rebuild container: `podman-compose build`
3. Restart services: `podman-compose up -d`
4. Test changes in container

See the [Docker Guide](docker.md) for more details.

## Code Style

### Python Style

- Follow PEP 8 style guide
- Use NumPy-style docstrings
- Maximum line length: 88 characters (Black default)
- Use type hints where appropriate

### Example Function

```python
def search_papers(
    query: str,
    limit: int = 10,
    year: int | None = None
) -> list[dict]:
    """
    Search for papers matching the query.

    Parameters
    ----------
    query : str
        Search query string
    limit : int, optional
        Maximum number of results (default: 10)
    year : int or None, optional
        Filter by conference year (default: None)

    Returns
    -------
    list of dict
        List of paper dictionaries matching the query

    Examples
    --------
    >>> results = search_papers("transformer", limit=5)
    >>> print(len(results))
    5
    """
    # Implementation here
    pass
```

## Testing

### Running Tests

```bash
# Run all tests (excludes slow tests by default)
uv run pytest

# Run with coverage
uv run pytest --cov=src/abstracts_explorer

# Run specific test file
uv run pytest tests/test_database.py

# Run specific test
uv run pytest tests/test_database.py::test_add_paper

# Verbose output
uv run pytest -v

# Show print statements
uv run pytest -s

# Run only slow tests (requires LM Studio)
uv run pytest -m slow

# Run all tests including slow ones
uv run pytest -m ""
```

**Note about slow tests:** Tests requiring LM Studio are marked as `slow` and skipped by default. This allows for faster development cycles. To run slow tests, you need:
- LM Studio running at the configured URL (default: http://localhost:1234)
- A chat model loaded in LM Studio
- Use `pytest -m slow` to run only slow tests

### Test Organization

**One test file per module**: Each source module should have exactly one corresponding test file. This makes tests easy to find and maintain.

Examples:
- `src/abstracts_explorer/database.py` → `tests/test_database.py`
- `src/abstracts_explorer/plugin.py` → `tests/test_plugin.py`
- `src/abstracts_explorer/web_ui/app.py` → `tests/test_web_ui.py`

**Shared test code**:
- `tests/conftest.py` - Shared pytest fixtures
- `tests/helpers.py` - Shared helper functions

**Exception for test types**: Different test types may have separate files:
- `test_integration.py` - Cross-module integration tests
- `test_web_integration.py` - Web UI integration tests
- `test_web_e2e.py` - End-to-end browser tests

### Writing Tests

- Use pytest framework
- Follow the one test file per module principle
- Create unit tests for all new functions
- Use fixtures for common setup (defined in `conftest.py`)
- Mock external dependencies (API calls, LLM backends)
- Aim for >80% code coverage

### Example Test

```python
import pytest
from abstracts_explorer.database import DatabaseManager

@pytest.fixture
def db(tmp_path):
    """Create a temporary test database."""
    db_path = tmp_path / "test.db"
    return DatabaseManager(str(db_path))

def test_add_paper(db):
    """Test adding a paper to the database."""
    paper_data = {
        'openreview_id': 'test123',
        'title': 'Test Paper',
        'abstract': 'Test abstract',
        'year': 2025,
    }
    
    paper_id = db.add_paper(paper_data)
    assert paper_id is not None
    
    # Verify paper was added
    paper = db.get_paper_by_id(paper_id)
    assert paper['title'] == 'Test Paper'
```

## Documentation

### Docstrings

All public functions, classes, and methods must have docstrings:

```python
def my_function(param1: str, param2: int = 10) -> bool:
    """
    Brief description of the function.

    More detailed description if needed.

    Parameters
    ----------
    param1 : str
        Description of param1
    param2 : int, optional
        Description of param2 (default: 10)

    Returns
    -------
    bool
        Description of return value

    Raises
    ------
    ValueError
        When param1 is empty
    RuntimeError
        When operation fails

    Examples
    --------
    >>> my_function("test")
    True

    Notes
    -----
    Additional notes about the function.

    See Also
    --------
    related_function : Related functionality
    """
    pass
```

### Building Documentation

```bash
# Build HTML documentation
cd docs
uv run make html

# View documentation
open _build/html/index.html

# Clean build
make clean
```

### Updating Documentation

1. Update docstrings in source code
2. Update Markdown files in `docs/`
3. Rebuild documentation
4. Review changes in browser

## Pull Request Process

### 1. Create Branch

```bash
git checkout -b feature/my-new-feature
```

### 2. Make Changes

- Write code following style guidelines
- Add tests for new functionality
- Update documentation
- Ensure all tests pass

### 3. Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "Add feature: description of changes"
```

### 4. Push Branch

```bash
git push origin feature/my-new-feature
```

### 5. Create Pull Request

- Provide clear description of changes
- Reference related issues
- Include test results

## Code Review

### What We Look For

- Correct functionality
- Adequate test coverage
- Clear documentation
- Code style compliance
- Performance considerations
- Error handling

### Review Process

1. Automated tests must pass
2. Code review by maintainer
3. Address feedback
4. Final approval and merge

## Database Backend Support

Abstracts Explorer supports both SQLite and PostgreSQL backends through SQLAlchemy.

### Architecture

**Core Components:**
- `db_models.py` - SQLAlchemy ORM models (Paper, EmbeddingsMetadata)
- `database.py` - DatabaseManager with SQLAlchemy session management
- `config.py` - Database URL configuration

**Configuration:**
```bash
# SQLite (default)
PAPER_DB=data/abstracts.db

# PostgreSQL
PAPER_DB=postgresql://user:pass@localhost/abstracts
```

### Working with Databases

**Using DatabaseManager:**
```python
from abstracts_explorer.database import DatabaseManager

# SQLite (legacy)
db = DatabaseManager(db_path="abstracts.db")

# Any backend via URL
db = DatabaseManager(database_url="postgresql://...")

with db:
    db.create_tables()
    # Database operations...
```

**Adding Database Fields:**
1. Update ORM model in `db_models.py`
2. Create migration if needed (manual for now)
3. Update `DatabaseManager` methods if necessary
4. Add tests for new fields

**Testing Different Backends:**
```bash
# SQLite tests (always run)
uv run pytest tests/test_database.py

# PostgreSQL tests (requires server)
export POSTGRES_TEST_URL=postgresql://localhost/test_db
uv run pytest tests/test_multi_database.py
```

**Best Practices:**
- Use SQLAlchemy ORM for new queries
- Maintain backward compatibility with existing API
- Test with both SQLite and PostgreSQL when possible
- Use timezone-aware datetime (Python 3.12+)

## Development Guidelines

### Adding New Features

1. **Discuss first** - Open an issue to discuss major changes
2. **Write tests first** - TDD when possible
3. **Document thoroughly** - Code and user documentation
4. **Consider backward compatibility** - Avoid breaking changes

### Fixing Bugs

1. **Add failing test** - Reproduce the bug
2. **Fix the bug** - Make the test pass
3. **Add regression test** - Prevent future recurrence
4. **Document the fix** - Update relevant docs

### Refactoring

1. **Ensure tests pass** - Before starting
2. **Make small changes** - Incremental improvements
3. **Run tests frequently** - Catch issues early
4. **Update documentation** - If interfaces change

## Performance

### Benchmarking

```python
import time

def benchmark():
    start = time.time()
    # Code to benchmark
    end = time.time()
    print(f"Execution time: {end - start:.2f}s")
```

### Profiling

```bash
# Profile code
python -m cProfile -o profile.stats script.py

# View results
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"
```

## Security Considerations

### Debug Mode and Production Server

The web UI uses a production-ready WSGI server (Waitress) by default to avoid security risks associated with Flask's development server.

**Key Points:**
- **Production mode (default)**: Uses Waitress WSGI server (secure)
- **Development mode**: Use `--dev` flag to enable Flask development server for easier debugging
- **Debug mode**: Use `-vv` (double verbose) to enable Flask debug mode, which works with either server
- Debug mode should only be used during development, never in production

**Example:**
```bash
# Production (secure, default)
abstracts-explorer web-ui

# Development with debug mode  
abstracts-explorer web-ui --dev -vv

# Production with debug logging (still secure)
abstracts-explorer web-ui -vv
```

**Security Note:** Flask's debug mode includes an interactive debugger that could allow arbitrary code execution if exposed. The production server (Waitress) is always recommended for deployed applications, even when debugging is needed.

## Questions?

- Open an issue for questions
- Check existing documentation
- Review test files for examples

Thank you for contributing!

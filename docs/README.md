# Documentation

This directory contains the Sphinx documentation for Abstracts Explorer.

## Building the Documentation

### Prerequisites

Install the documentation dependencies:

```bash
uv sync --extra docs
```

Or install them manually if not using uv:

```bash
pip install sphinx sphinx-rtd-theme myst-parser sphinx-autodoc-typehints linkify-it-py
```

### Build HTML Documentation

```bash
cd docs
uv run make html
```

The generated HTML documentation will be in `_build/html/`. Open `_build/html/index.html` in your browser.

### Build Other Formats

```bash
# PDF (requires LaTeX)
make latexpdf

# Plain text
make text

# ePub
make epub

# Clean build
make clean
```

### View Documentation

After building, you can view the documentation by opening:

```bash
# macOS
open _build/html/index.html

# Linux
xdg-open _build/html/index.html

# Windows
start _build/html/index.html
```

## Documentation Structure

- **index.md**: Main documentation homepage
- **installation.md**: Installation instructions
- **configuration.md**: Configuration guide
- **usage.md**: Usage examples and guide
- **cli_reference.md**: Command-line interface reference
- **contributing.md**: Contribution guidelines
- **api/**: API reference documentation
  - Automatically generated from source code docstrings
  - Manual documentation with examples

## Writing Documentation

### Markdown Files

Documentation uses Markdown (via MyST Parser) for most pages. Features:

- Standard Markdown syntax
- Code blocks with syntax highlighting
- Links and references
- Tables
- Admonitions (notes, warnings, etc.)

### Docstrings

API documentation is auto-generated from NumPy-style docstrings in the source code:

```python
def my_function(param1: str, param2: int = 10) -> bool:
    """
    Brief description.

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

    Examples
    --------
    >>> my_function("test")
    True
    """
    pass
```

### Adding New Pages

1. Create a new Markdown file in `docs/`
2. Add it to the `toctree` in `index.md`
3. Rebuild documentation

### Configuration

Documentation configuration is in `conf.py`:

- Project metadata
- Sphinx extensions
- Theme options
- Build settings

## Deployment

### Read the Docs

The documentation can be hosted on Read the Docs:

1. Connect your GitHub repository
2. Configure build settings (Python 3.8+)
3. Specify requirements: `docs/requirements.txt` or `pyproject.toml`
4. Documentation will auto-build on commits

### GitHub Pages

Build and deploy to GitHub Pages:

```bash
# Build documentation
make html

# Copy to gh-pages branch
# (manual or using sphinx-gh-pages)
```

## Troubleshooting

### Build Warnings

Warnings about duplicate IDs are expected when combining manual and auto-generated docs.

### Missing Dependencies

If you see import errors, ensure all dependencies are installed:

```bash
uv sync --extra docs
```

### Clean Build

If you encounter issues, try a clean rebuild:

```bash
make clean
uv run make html
```

## Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [MyST Parser](https://myst-parser.readthedocs.io/)
- [Read the Docs](https://readthedocs.org/)
- [NumPy Docstring Guide](https://numpydoc.readthedocs.io/)

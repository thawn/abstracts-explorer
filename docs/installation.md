# Installation

## Requirements

- Python 3.8 or higher
- [uv](https://docs.astral.sh/uv/) - Fast Python package installer and resolver
- Node.js 14 or higher (for web UI)
- npm package manager

## Install uv

If you don't have uv installed yet:

```bash
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip (if you already have Python)
pip install uv
```

See the [official uv documentation](https://docs.astral.sh/uv/getting-started/installation/) for more details.

## Install from Source

Clone the repository and install in development mode:

```bash
git clone <repository-url>
cd neurips-abstracts

# Create virtual environment and install dependencies with uv
uv sync

# Or install with all optional dependencies
uv sync --all-extras

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Node.js dependencies
npm install

# Install vendor files for web UI
npm run install:vendor
```

## Python Dependencies

The package will automatically install the following dependencies when you run `uv sync`:

- **requests**: For API calls to OpenReview
- **chromadb**: For vector embeddings storage
- **pydantic**: For data validation
- **beautifulsoup4**: For HTML parsing

### Development Dependencies

Install development dependencies with:

```bash
uv sync --extra dev
```

This includes:
- **pytest**: For running tests
- **pytest-cov**: For test coverage reports
- **pytest-mock**: For test mocking
- **selenium**: For browser automation testing
- **webdriver-manager**: For managing browser drivers

### Web Interface Dependencies

Install web dependencies with:

```bash
uv sync --extra web
```

This includes:
- **flask**: Web framework
- **flask-cors**: CORS support for API endpoints

### Documentation Dependencies

Install documentation dependencies with:

```bash
uv sync --extra docs
```

This includes:
- **sphinx**: Documentation generator
- **sphinx-rtd-theme**: Read the Docs theme
- **myst-parser**: Markdown support
- **sphinx-autodoc-typehints**: Type hints in docs
- **linkify-it-py**: URL auto-linking

## Node.js Dependencies

For the web interface, the following dependencies are installed via npm:

- **@fortawesome/fontawesome-free**: Icon library for the UI
- **marked**: Markdown parser for rendering chat responses
- **tailwindcss**: CSS framework (reference)
- **jest**: JavaScript testing framework
- **@testing-library/dom**: Testing utilities for DOM manipulation

### Vendor Files

The web UI uses local copies of external libraries instead of CDN links for:

- **Tailwind CSS** (standalone build): Modern utility-first CSS framework
- **Font Awesome**: Icon fonts and CSS
- **Marked.js**: Markdown parsing and rendering

These files are installed automatically when you run:

```bash
npm run install:vendor
```

This command copies the necessary files from `node_modules` to `src/neurips_abstracts/web_ui/static/vendor/` and `src/neurips_abstracts/web_ui/static/webfonts/`.

**Note**: The vendor files are excluded from git (via `.gitignore`) and must be generated locally or during deployment.

## Optional Dependencies

You can install specific groups of optional dependencies:

```bash
# Development tools only
uv sync --extra dev

# Web interface only
uv sync --extra web

# Documentation tools only
uv sync --extra docs

# All optional dependencies
uv sync --all-extras
```

## Verify Installation

Check that the package is installed correctly:

```bash
neurips-abstracts --help
```

You should see the available commands listed.

Test the web UI:

```bash
neurips-abstracts web-ui
```

The web interface should start at <http://127.0.0.1:5000>.

## Troubleshooting

### Missing Vendor Files

If the web UI loads but styling/icons are broken, regenerate vendor files:

```bash
npm run install:vendor
```

### Node.js Not Found

If you don't have Node.js installed:

- **macOS**: `brew install node`
- **Ubuntu/Debian**: `sudo apt install nodejs npm`
- **Windows**: Download from <https://nodejs.org/>

### Python Version Issues

Ensure you're using Python 3.8 or higher:

```bash
python --version
```

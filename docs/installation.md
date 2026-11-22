# Installation

## Requirements

- Python 3.12 or higher
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
cd abstracts-explorer

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

To run the optional chat input guard during development, see
[Local Ollaya Guard](contributing.md#5-optional-local-ollaya-guard) and the
[chat input guard configuration](configuration.md#chat-input-guard-ollaya).

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
- **pytest-playwright**: For Playwright browser automation testing
- **black**: For code formatting
- **ruff**: For linting
- **mypy**: For type checking (required by the pre-commit hook)
- **types-requests**: Type stubs for `requests`

Install the Chromium browser used by the end-to-end tests:

```bash
uv run playwright install chromium
```

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

This command copies the necessary files from `node_modules` to `src/abstracts_explorer/web_ui/static/vendor/` and `src/abstracts_explorer/web_ui/static/webfonts/`.

**Note**: The vendor files are excluded from git (via `.gitignore`) and must be generated locally or during deployment.

## Optional Dependencies

The extras below install Python packages. The optional Ollaya chat guard runs
as a separate Podman container and does not add a Python extra.

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

### Ollaya Chat Input Guard (Optional, external)

Ollaya is not a Python extra. It is a separate CPU-only container used only
when `GUARD_ENABLED=true`. It provides the optional guard that classifies chat
input before the RAG agent runs. The guard is fail-open: if Ollaya is not
running, chat still works.

Requirements: Podman.

Pull the guard model once:

```bash
podman volume create abstracts-ollaya

podman run --rm \
  -v abstracts-ollaya:/home/ollaya/.ollaya \
  ghcr.io/ollaya-dev/ollaya pull laya:multilingual
```

Start the server on loopback for a host-run development setup:

```bash
podman run -d --name abstracts-ollaya \
  -p 127.0.0.1:11435:11435 \
  -v abstracts-ollaya:/home/ollaya/.ollaya \
  -e OLLAYA_HOST=0.0.0.0:11435 \
  -e OLLAYA_PRELOAD=laya:multilingual \
  -e OLLAYA_KEEP_ALIVE=-1 \
  -e OLLAYA_MAX_LOADED_MODELS=1 \
  -e OLLAYA_THREADS=4 \
  -e OLLAYA_API_KEY=local-dev-key \
  ghcr.io/ollaya-dev/ollaya
```

Enable the guard in `.env`:

```bash
GUARD_ENABLED=true
GUARD_BACKEND_URL=http://127.0.0.1:11435
GUARD_API_KEY=local-dev-key
GUARD_MODEL=laya:multilingual
GUARD_TIMEOUT=2.0
GUARD_TOPIC_THRESHOLD=0.8
GUARD_JAILBREAK_THRESHOLD=0.8
```

`OLLAYA_API_KEY` and `GUARD_API_KEY` must match. Keep `GUARD_ENABLED=false`
if you do not want the guard.

For the full host-development walkthrough, see
[Local Ollaya Guard](contributing.md#5-optional-local-ollaya-guard). For the
containerized app stack, see
[Docker and Podman Setup](docker.md#optional-chat-input-guard-ollaya).

#### macOS

Ollaya’s image is a Linux image (`linux/amd64` and `linux/arm64`). On macOS,
run it inside Podman’s Linux VM and give the VM at least 4 GB of memory.

1. Configure the Podman VM memory:

   ```bash
   podman machine stop
   podman machine set --memory 4096
   podman machine start
   podman machine ssh 'free -m'
   ```

2. On Apple Silicon, confirm the VM architecture is arm64:

   ```bash
   podman info --format '{{.Host.Arch}}'
   ```

3. Continue with the pull and start commands above. Ollaya may print
   `onnxruntime cpuid_info warning: Unknown CPU vendor. cpuinfo_vendor value: 0`;
   this is expected and does not affect inference.

4. To run without a VM, install Ollaya natively on macOS and point
   `GUARD_BACKEND_URL` at the native server:

   ```bash
   curl -fsSL https://ollaya.dev/install.sh | sh
   ollaya serve
   ```

   ```bash
   GUARD_BACKEND_URL=http://127.0.0.1:11435
   ```

## Verify Installation

Check that the package is installed correctly:

```bash
abstracts-explorer --help
```

You should see the available commands listed.

Test the web UI:

```bash
abstracts-explorer web-ui
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

Ensure you're using Python 3.12 or higher:

```bash
python --version
```

## Quick Start: Download Pre-Built Data from a Registry

Instead of downloading conference data from source APIs and generating embeddings locally, you can download pre-built paper databases, embeddings, and clustering caches from an OCI-compatible container registry. This is the fastest way to get started with a fully populated instance.

```bash
# Download all available conference data (e.g. NeurIPS, ICLR, ICML)
uv run abstracts-explorer registry download \
  -r ghcr.io/thawn/abstracts-data \
  --conference all \
  --embedding-model text-embedding-qwen3-embedding-4b

# Or download a specific conference and year
uv run abstracts-explorer registry download \
  -r ghcr.io/thawn/abstracts-data \
  --conference neurips --year 2025

# List available data in the registry
uv run abstracts-explorer registry list \
  -r ghcr.io/thawn/abstracts-data
```

Authentication may be required depending on the registry. Set your token via the `.env` file or environment variable:

```bash
export GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
```

See the [Registry documentation](registry.md) for full details on downloading and uploading data.

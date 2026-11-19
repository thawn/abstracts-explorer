# Web UI Integration

The web UI has been successfully integrated as an optional module in the `abstracts_explorer` package.

## Installation

The web UI requires Flask and Flask-CORS. Install them using:

```bash
# Install with web UI dependencies
uv sync --extra web

# Or install Flask manually if not using uv
pip install flask flask-cors
```

## Usage

### Command Line

Start the web interface using the new CLI command:

```bash
# Start with defaults (127.0.0.1:5000)
uv run abstracts-explorer web-ui

# Specify host and port
uv run abstracts-explorer web-ui --host 0.0.0.0 --port 8080

# Enable debug mode
uv run abstracts-explorer web-ui --debug

# Get help
uv run abstracts-explorer web-ui --help
```

### Programmatic Usage

You can also start the server from Python code:

```python
from abstracts_explorer.web_ui import run_server

# Start with defaults
run_server()

# Custom configuration
run_server(host='0.0.0.0', port=8080, debug=True)
```

## Features

The web interface provides:

- **Paper Search**: Keyword and semantic search capabilities
- **AI Chat**: RAG-powered Q&A about papers
- **Interesting Papers**: Rate and organize papers with star ratings
  - **JSON Import/Export**: Save and load your paper ratings
  - Export ratings to backup or share with colleagues
  - Import ratings with smart merging (preserves existing ratings)
- **Statistics**: Database statistics and insights
- **Modern UI**: Clean, responsive design using Tailwind CSS
- **REST API**: JSON API for all functionality

## Module Structure

```
src/abstracts_explorer/web_ui/
├── __init__.py          # Module exports
├── app.py               # Flask application (routes + server startup)
├── dependencies.py      # Request-scoped DB/embeddings/RAG helpers and caches
├── api_utils.py         # Shared API helpers
├── static/
│   ├── app.ts           # Frontend entry point (bundled by esbuild)
│   ├── modules/         # TypeScript modules (chat, search, clustering, state, ...)
│   │   └── utils/       # Shared TypeScript utilities
│   ├── templates/       # HTML templates
│   ├── style.css        # Custom styles
│   ├── dist/            # esbuild output (generated, gitignored)
│   └── vendor/          # CSS/font assets (JS libs are bundled from npm)
└── tests/               # Jest test suite
```

## Configuration

The web UI uses the same configuration system as the rest of the package. Set environment variables or use a `.env` file:

```bash
# Data directory (default: data)
DATA_DIR=data

# Database paths (relative to DATA_DIR unless absolute)
PAPER_DB=abstracts.db
EMBEDDING_DB=chroma_db

# OpenAI-compatible API (such as LM Studio or blablador) settings
LLM_BACKEND_URL=http://localhost:1234
CHAT_MODEL=your-model-name
EMBEDDING_MODEL=your-embedding-model

# Collection name
COLLECTION_NAME=papers
```

## API Endpoints

The web UI exposes the following REST API endpoints:

- `GET /` - Main web interface
- `GET /<conference_name>` - Deep link to a conference view
- `GET /api/stats` - Database statistics
- `GET /api/filters` - Filters for the current selection
- `GET /api/available-filters` - All available filters
- `POST /api/search` - Search papers
- `GET /api/paper/<paper_uid>` - Get paper details
- `POST /api/papers/batch` - Batch-fetch paper details
- `POST /api/chat` - Chat with RAG (forwards interesting-paper ratings)
- `POST /api/chat/reset` - Reset conversation
- `POST /api/clusters/compute` - Serve (cached) cluster visualization
- `POST /api/clusters/search` - Search within a distance of a custom query
- `POST /api/topic-evolution` - Topic evolution data for plotting
- `GET /api/papers-per-year` - Papers-per-year counts
- `GET /api/embedding-model-check` - Report stored vs configured embedding model
- `POST /api/export/interesting-papers` - Export interesting papers as ZIP
- `POST /api/donate-data` - Donate interesting-paper ratings
- `POST /api/donate-chat` - Donate chat transcripts
- `GET /health` - Health check (used by container healthchecks)

## Testing

Run the web UI tests:

```bash
uv run pytest tests/test_web_ui.py -v          # Python (Flask) tests
npm run typecheck && npm test                     # Type-check and test the TypeScript modules
```

## Development

The web UI is fully integrated into the package:

1. **Module Location**: `src/abstracts_explorer/web_ui/`
2. **CLI Command**: `uv run abstracts-explorer web-ui`
3. **Optional Dependency**: Install with `uv sync --extra web`
4. **Test Coverage**: 18 comprehensive tests

## Migration from Standalone

If you were using the standalone `web/` directory before:

- **Old**: `cd web && python app.py`
- **New**: `uv run abstracts-explorer web-ui`

The functionality is identical, but now it's properly integrated into the package structure.

## Error Handling

If Flask is not installed, you'll see a helpful error message:

```
❌ Web UI dependencies not installed!

The web UI requires Waitress and Flask. Install them with:
  uv sync --extra web

Or install Flask manually if not using uv:
  pip install flask flask-cors
```

## Notes

- The web UI requires a database to be created first (`abstracts-explorer download`)
- For semantic search, embeddings must be generated (`abstracts-explorer create-embeddings`)
- For AI chat, OpenAI-compatible API (such as LM Studio or blablador) must be running with a compatible model loaded

# Web UI Integration

The web UI has been successfully integrated as an optional module in the `neurips_abstracts` package.

## Installation

The web UI requires Flask and Flask-CORS. Install them using:

```bash
# Install with web UI dependencies
pip install neurips-abstracts[web]

# Or install Flask manually
pip install flask flask-cors
```

## Usage

### Command Line

Start the web interface using the new CLI command:

```bash
# Start with defaults (127.0.0.1:5000)
neurips-abstracts web-ui

# Specify host and port
neurips-abstracts web-ui --host 0.0.0.0 --port 8080

# Enable debug mode
neurips-abstracts web-ui --debug

# Get help
neurips-abstracts web-ui --help
```

### Programmatic Usage

You can also start the server from Python code:

```python
from neurips_abstracts.web_ui import run_server

# Start with defaults
run_server()

# Custom configuration
run_server(host='0.0.0.0', port=8080, debug=True)
```

## Features

The web interface provides:

- **Paper Search**: Keyword and semantic search capabilities
- **AI Chat**: RAG-powered Q&A about papers
- **Statistics**: Database statistics and insights
- **Modern UI**: Clean, responsive design using Tailwind CSS
- **REST API**: JSON API for all functionality

## Module Structure

```
src/neurips_abstracts/web_ui/
├── __init__.py          # Module exports
├── app.py               # Flask application
├── static/              # Static assets
│   ├── app.js          # Frontend JavaScript
│   └── style.css       # Custom styles
└── templates/           # HTML templates
    └── index.html      # Main page
```

## Configuration

The web UI uses the same configuration system as the rest of the package. Set environment variables or use a `.env` file:

```bash
# Data directory (default: data)
DATA_DIR=data

# Database paths (relative to DATA_DIR unless absolute)
PAPER_DB_PATH=neurips_2025.db
EMBEDDING_DB_PATH=chroma_db

# LM Studio settings
LLM_BACKEND_URL=http://localhost:1234
CHAT_MODEL=your-model-name
EMBEDDING_MODEL=your-embedding-model

# Collection name
COLLECTION_NAME=neurips_papers
```

## API Endpoints

The web UI exposes the following REST API endpoints:

- `GET /` - Main web interface
- `GET /api/stats` - Database statistics
- `POST /api/search` - Search papers
- `GET /api/paper/<id>` - Get paper details
- `POST /api/chat` - Chat with RAG
- `POST /api/chat/reset` - Reset conversation
- `GET /api/years` - Get available years

## Testing

Run the web UI tests:

```bash
pytest tests/test_web.py -v
```

## Development

The web UI is fully integrated into the package:

1. **Module Location**: `src/neurips_abstracts/web_ui/`
2. **CLI Command**: `neurips-abstracts web-ui`
3. **Optional Dependency**: Install with `pip install neurips-abstracts[web]`
4. **Test Coverage**: 18 comprehensive tests

## Migration from Standalone

If you were using the standalone `web/` directory before:

- **Old**: `cd web && python app.py`
- **New**: `neurips-abstracts web-ui`

The functionality is identical, but now it's properly integrated into the package structure.

## Error Handling

If Flask is not installed, you'll see a helpful error message:

```
❌ Web UI dependencies not installed!

The web UI requires Flask. Install it with:
  pip install neurips-abstracts[web]

Or install Flask manually:
  pip install flask flask-cors
```

## Notes

- The web UI requires a database to be created first (`neurips-abstracts download`)
- For semantic search, embeddings must be generated (`neurips-abstracts create-embeddings`)
- For AI chat, LM Studio must be running with a compatible model loaded

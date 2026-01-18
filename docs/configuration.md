# Configuration

Abstracts Explorer uses a flexible configuration system that supports environment variables, `.env` files, and command-line arguments.

## Configuration Priority

Settings are loaded in the following priority order (later overrides earlier):

1. Built-in defaults
2. `.env` file in the current directory
3. Environment variables
4. Command-line arguments (when applicable)

## Configuration File

Create a `.env` file in your project directory:

```bash
# Copy the example file
cp .env.example .env

# Edit with your preferences
nano .env
```

## Available Settings

### Chat/Language Model Settings

- **CHAT_MODEL**: The LLM model to use for RAG chat (default: `gemma-3-4b-it-qat`)
- **CHAT_TEMPERATURE**: Temperature for LLM responses, 0.0-2.0 (default: `0.7`)
- **CHAT_MAX_TOKENS**: Maximum tokens in LLM responses (default: `1000`)

### Embedding Model Settings

- **EMBEDDING_MODEL**: The embedding model to use (default: `text-embedding-qwen3-embedding-4b`)

### LLM Backend Configuration

- **LLM_BACKEND_URL**: URL of the LLM backend server (default: `http://localhost:1234`)
- **LLM_BACKEND_AUTH_TOKEN**: Optional authentication token for LLM backend (default: empty)

### Data Directory

- **DATA_DIR**: Base directory for data files (default: `data`)

### Database Configuration

#### Paper Database

- **PAPER_DB**: Database connection for papers. Can be either:
  - **PostgreSQL URL**: `postgresql://user:password@host:port/database`
  - **SQLite file path**: `abstracts.db` (relative to DATA_DIR) or `/absolute/path/to/abstracts.db`
  - Default: `abstracts.db`

The configuration automatically detects the database type based on the format:
- URLs starting with `postgresql://`, `sqlite://`, or other database schemes are treated as database URLs
- Other values are treated as SQLite file paths (relative to DATA_DIR unless absolute)

#### Embedding Database

- **EMBEDDING_DB_PATH**: Path to local ChromaDB database (default: `chroma_db`). If relative, resolved relative to DATA_DIR.
- **EMBEDDING_DB_URL**: URL for remote ChromaDB HTTP service (e.g., `http://chromadb:8000`). Use this for Docker deployments.

**Note**: `EMBEDDING_DB_URL` and `EMBEDDING_DB_PATH` are mutually exclusive. Only set one of them.

### RAG Settings

- **COLLECTION_NAME**: ChromaDB collection name (default: `papers`)
- **MAX_CONTEXT_PAPERS**: Number of papers to include in RAG context (default: `5`)

## Example Configurations

### Local Development (SQLite)

```bash
# .env file for local development

# Base directory for data files
DATA_DIR=data

CHAT_MODEL=diffbot-small-xl-2508
CHAT_TEMPERATURE=0.7
CHAT_MAX_TOKENS=1000

EMBEDDING_MODEL=text-embedding-qwen3-embedding-4b

LLM_BACKEND_URL=http://localhost:1234
LLM_BACKEND_AUTH_TOKEN=

# SQLite database (relative to DATA_DIR - will resolve to data/abstracts.db)
PAPER_DB=abstracts.db

# Local ChromaDB (relative to DATA_DIR - will resolve to data/chroma_db)
EMBEDDING_DB_PATH=chroma_db

COLLECTION_NAME=papers
MAX_CONTEXT_PAPERS=5
```

### Production/Docker (PostgreSQL)

```bash
# .env file for production with PostgreSQL

DATA_DIR=data

CHAT_MODEL=diffbot-small-xl-2508
CHAT_TEMPERATURE=0.7
CHAT_MAX_TOKENS=1000

EMBEDDING_MODEL=text-embedding-qwen3-embedding-4b

LLM_BACKEND_URL=http://localhost:1234
LLM_BACKEND_AUTH_TOKEN=

# PostgreSQL database URL
PAPER_DB=postgresql://abstracts:password@postgres:5432/abstracts

# Remote ChromaDB HTTP service
EMBEDDING_DB_URL=http://chromadb:8000

COLLECTION_NAME=papers
MAX_CONTEXT_PAPERS=5
```

### Alternative: Absolute Paths

```bash
# Using absolute paths for both databases
PAPER_DB=/var/data/abstracts.db
EMBEDDING_DB_PATH=/var/data/chroma_db
```

## Using Configuration in Code

```python
from abstracts_explorer.config import get_config

# Get the singleton configuration instance
config = get_config()

# Access configuration values
print(f"Chat model: {config.chat_model}")
print(f"Backend URL: {config.llm_backend_url}")
print(f"Database URL: {config.database_url}")  # SQLAlchemy-compatible URL

# Check which embedding database mode is active
if config.embedding_db_url:
    print(f"Using remote ChromaDB: {config.embedding_db_url}")
else:
    print(f"Using local ChromaDB: {config.embedding_db_path}")
```

## Environment Variables

You can also set configuration via environment variables:

```bash
export CHAT_MODEL=llama-3.2-3b-instruct
export LLM_BACKEND_URL=http://localhost:8080
abstracts-explorer chat
```

## Security Best Practices

- Never commit `.env` files to version control
- Use `.env.example` as a template without sensitive data
- Keep authentication tokens secure
- Use environment variables in production environments

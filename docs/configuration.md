# Configuration

The NeurIPS Abstracts package uses a flexible configuration system that supports environment variables, `.env` files, and command-line arguments.

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

### Database Paths

- **EMBEDDING_DB_PATH**: Path to ChromaDB database (default: `chroma_db`). If relative, resolved relative to DATA_DIR.
- **PAPER_DB_PATH**: Path to SQLite database with papers (default: `data/neurips_2025.db`). If relative, resolved relative to DATA_DIR.

### RAG Settings

- **COLLECTION_NAME**: ChromaDB collection name (default: `neurips_papers`)
- **MAX_CONTEXT_PAPERS**: Number of papers to include in RAG context (default: `5`)

## Example Configuration

```bash
# .env file example

# Base directory for data files
DATA_DIR=data

CHAT_MODEL=gemma-3-4b-it-qat
CHAT_TEMPERATURE=0.7
CHAT_MAX_TOKENS=1000

EMBEDDING_MODEL=text-embedding-qwen3-embedding-4b

LLM_BACKEND_URL=http://localhost:1234
LLM_BACKEND_AUTH_TOKEN=

# Paths relative to DATA_DIR (will resolve to data/chroma_db and data/neurips_2025.db)
EMBEDDING_DB_PATH=chroma_db
PAPER_DB_PATH=neurips_2025.db

# Or use absolute paths:
# EMBEDDING_DB_PATH=/absolute/path/to/chroma_db
# PAPER_DB_PATH=/absolute/path/to/neurips_2025.db

COLLECTION_NAME=neurips_papers
MAX_CONTEXT_PAPERS=5
```

## Using Configuration in Code

```python
from neurips_abstracts.config import get_config

# Get the singleton configuration instance
config = get_config()

# Access configuration values
print(f"Chat model: {config.chat_model}")
print(f"Backend URL: {config.llm_backend_url}")
print(f"Database path: {config.paper_db_path}")
```

## Environment Variables

You can also set configuration via environment variables:

```bash
export CHAT_MODEL=llama-3.2-3b-instruct
export LLM_BACKEND_URL=http://localhost:8080
neurips-abstracts chat
```

## Security Best Practices

- Never commit `.env` files to version control
- Use `.env.example` as a template without sensitive data
- Keep authentication tokens secure
- Use environment variables in production environments

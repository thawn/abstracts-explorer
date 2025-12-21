# Configuration Guide

The `neurips-abstracts` package supports environment-based configuration through `.env` files. This allows you to customize default settings without modifying command-line arguments.

## Quick Start

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your preferred settings:
   ```bash
   nano .env  # or vim, emacs, code, etc.
   ```

3. The configuration is automatically loaded when you run any command.

## Configuration Options

### Chat Model Settings

**`CHAT_MODEL`**
- Description: Language model for RAG chat feature
- Default: `diffbot-small-xl-2508`
- Example: `CHAT_MODEL=gpt-4`
- Used by: `neurips-abstracts chat` command and `RAGChat` class

**`CHAT_TEMPERATURE`**
- Description: Sampling temperature for chat responses (0.0 = deterministic, 1.0 = creative)
- Default: `0.7`
- Example: `CHAT_TEMPERATURE=0.5`
- Used by: `neurips-abstracts chat` command

**`CHAT_MAX_TOKENS`**
- Description: Maximum tokens in generated responses
- Default: `1000`
- Example: `CHAT_MAX_TOKENS=2000`
- Used by: RAG chat generation

### Embedding Model Settings

**`EMBEDDING_MODEL`**
- Description: Text embedding model name
- Default: `text-embedding-qwen3-embedding-4b`
- Example: `EMBEDDING_MODEL=text-embedding-ada-002`
- Used by: `neurips-abstracts create-embeddings` and `search` commands

### LLM Backend Configuration

**`LLM_BACKEND_URL`**
- Description: Base URL for LM Studio or compatible API
- Default: `http://localhost:1234`
- Example: `LLM_BACKEND_URL=https://api.openai.com/v1`
- Used by: All embedding and chat operations

**`LLM_BACKEND_AUTH_TOKEN`**
- Description: Authentication token for LLM backend (if required)
- Default: `` (empty)
- Example: `LLM_BACKEND_AUTH_TOKEN=sk-proj-...`
- Used by: API authentication headers
- **Security Note**: Never commit `.env` file with tokens to version control!

### Database Paths

**`EMBEDDING_DB_PATH`**
- Description: Path to ChromaDB vector database directory
- Default: `chroma_db`
- Example: `EMBEDDING_DB_PATH=./embeddings/neurips_2025`
- Used by: All embedding and search operations

**`PAPER_DB_PATH`**
- Description: Path to SQLite database with paper metadata
- Default: `neurips_2025.db`
- Example: `PAPER_DB_PATH=./data/papers.db`
- Used by: Database operations and author resolution

### Collection Settings

**`COLLECTION_NAME`**
- Description: Name of the ChromaDB collection
- Default: `neurips_papers`
- Example: `COLLECTION_NAME=neurips_2025_accepted`
- Used by: Embedding storage and search

**`MAX_CONTEXT_PAPERS`**
- Description: Default number of papers for RAG context
- Default: `5`
- Example: `MAX_CONTEXT_PAPERS=10`
- Used by: Search and chat commands

## Configuration Priority

Settings are loaded in the following order (later overrides earlier):

1. **Built-in defaults** (defined in `config.py`)
2. **`.env` file** (if present)
3. **Environment variables** (system-level)
4. **Command-line arguments** (highest priority)

### Example Priority

If you have:
- `.env` file: `CHAT_MODEL=diffbot-small-xl-2508`
- Environment variable: `export CHAT_MODEL=gpt-4`
- Command-line: `--model llama2`

The result will be: `llama2` (command-line wins)

## Python API Usage

You can access configuration in Python code:

```python
from neurips_abstracts import get_config

# Get configuration instance
config = get_config()

# Access settings
print(f"Chat model: {config.chat_model}")
print(f"Embedding model: {config.embedding_model}")
print(f"LLM backend: {config.llm_backend_url}")

# View all settings
print(config.to_dict())
```

### Create Custom Configuration

```python
from neurips_abstracts.config import Config
from pathlib import Path

# Load from specific .env file
config = Config(env_path=Path("production.env"))

# Access values
print(config.chat_model)
```

### Force Reload Configuration

```python
from neurips_abstracts import get_config

# Reload from environment
config = get_config(reload=True)
```

## Example `.env` Files

### Development Environment

```bash
# .env.development
CHAT_MODEL=diffbot-small-xl-2508
EMBEDDING_MODEL=text-embedding-qwen3-embedding-4b
LLM_BACKEND_URL=http://localhost:1234
LLM_BACKEND_AUTH_TOKEN=
EMBEDDING_DB_PATH=dev_chroma_db
PAPER_DB_PATH=dev_neurips_2025.db
COLLECTION_NAME=dev_papers
MAX_CONTEXT_PAPERS=3
CHAT_TEMPERATURE=0.7
CHAT_MAX_TOKENS=500
```

### Production Environment

```bash
# .env.production
CHAT_MODEL=gpt-4
EMBEDDING_MODEL=text-embedding-ada-002
LLM_BACKEND_URL=https://api.openai.com/v1
LLM_BACKEND_AUTH_TOKEN=sk-proj-your-token-here
EMBEDDING_DB_PATH=/var/lib/neurips/embeddings
PAPER_DB_PATH=/var/lib/neurips/papers.db
COLLECTION_NAME=neurips_2025_production
MAX_CONTEXT_PAPERS=10
CHAT_TEMPERATURE=0.5
CHAT_MAX_TOKENS=2000
```

### Testing Environment

```bash
# .env.test
CHAT_MODEL=mock-model
EMBEDDING_MODEL=mock-embedding
LLM_BACKEND_URL=http://localhost:9999
LLM_BACKEND_AUTH_TOKEN=test-token
EMBEDDING_DB_PATH=test_chroma_db
PAPER_DB_PATH=test.db
COLLECTION_NAME=test_collection
MAX_CONTEXT_PAPERS=2
CHAT_TEMPERATURE=0.0
CHAT_MAX_TOKENS=100
```

## Security Best Practices

### 1. Never Commit Secrets

The `.env` file is already in `.gitignore`. Always verify before committing:

```bash
git status  # Should NOT show .env
```

### 2. Use Environment-Specific Files

Keep separate configuration files:
- `.env.example` - Template (commit this)
- `.env` - Local development (do NOT commit)
- `.env.production` - Production secrets (secure storage only)

### 3. Restrict File Permissions

```bash
chmod 600 .env  # Only owner can read/write
```

### 4. Use Secret Management

For production, consider:
- AWS Secrets Manager
- Azure Key Vault
- HashiCorp Vault
- Kubernetes Secrets

## Troubleshooting

### Configuration Not Loading

**Problem**: Changes to `.env` are not reflected

**Solutions**:
1. Ensure `.env` is in the working directory
2. Check file permissions: `ls -la .env`
3. Verify no typos in variable names
4. Force reload: `get_config(reload=True)`

### Override Not Working

**Problem**: Command-line argument doesn't override `.env`

**Solution**: Ensure you're using the correct argument name:
```bash
# Correct
neurips-abstracts chat --model gpt-4

# Incorrect
neurips-abstracts chat --chat-model gpt-4
```

### Finding Configuration File

The loader searches for `.env` in:
1. Current working directory
2. Up to 5 parent directories

Run from your project root to ensure `.env` is found.

### Invalid Values

**Problem**: Configuration value is wrong type (e.g., string instead of number)

**Effect**: Falls back to default value

**Solution**: Check `.env` syntax:
```bash
# Correct
MAX_CONTEXT_PAPERS=5

# Incorrect (parsed as string "5")
MAX_CONTEXT_PAPERS="5"
```

### View Active Configuration

```bash
python -c "from neurips_abstracts import get_config; print(get_config().to_dict())"
```

## Command-Line Examples

### Using Default Configuration

```bash
# Uses values from .env (if present) or built-in defaults
neurips-abstracts create-embeddings --db-path neurips_2025.db
neurips-abstracts search "deep learning"
neurips-abstracts chat
```

### Overriding Configuration

```bash
# Override embedding model for this command only
neurips-abstracts create-embeddings \
  --db-path neurips_2025.db \
  --model custom-embedding-model \
  --lm-studio-url http://remote-server:8080

# Override chat settings
neurips-abstracts chat \
  --model llama2 \
  --temperature 0.9 \
  --max-context 10
```

### Mixed Configuration

```bash
# Some from .env, some from CLI
neurips-abstracts search "transformers" \
  --n-results 10 \
  --show-abstract
# Uses EMBEDDING_DB_PATH and COLLECTION_NAME from .env
# Uses n-results=10 from CLI
```

## Migration from Hardcoded Defaults

If you're upgrading from a version without configuration support:

### Old Way
```bash
neurips-abstracts create-embeddings \
  --db-path neurips_2025.db \
  --output chroma_db \
  --model text-embedding-qwen3-embedding-4b \
  --lm-studio-url http://localhost:1234
```

### New Way (Equivalent)

Create `.env`:
```bash
EMBEDDING_DB_PATH=chroma_db
EMBEDDING_MODEL=text-embedding-qwen3-embedding-4b
LLM_BACKEND_URL=http://localhost:1234
```

Run command:
```bash
neurips-abstracts create-embeddings --db-path neurips_2025.db
# All other settings loaded from .env automatically
```

## Advanced Usage

### Multiple Configurations

Switch between configurations using environment variables:

```bash
# Use development config
export NEURIPS_ENV=development
neurips-abstracts chat  # Uses defaults for dev

# Use production config  
export NEURIPS_ENV=production
neurips-abstracts chat  # Uses defaults for prod
```

You can implement this by creating `load_env_file()` logic that checks `NEURIPS_ENV` variable.

### Configuration in Scripts

```python
#!/usr/bin/env python3
"""Automated embedding generation script."""

import os
from pathlib import Path
from neurips_abstracts import EmbeddingsManager, get_config

# Set environment variables before importing
os.environ["EMBEDDING_MODEL"] = "custom-model"
os.environ["MAX_CONTEXT_PAPERS"] = "10"

# Force reload to pick up changes
config = get_config(reload=True)

# Use configuration
em = EmbeddingsManager()  # Uses config automatically
print(f"Using model: {em.model_name}")
print(f"Backend: {em.lm_studio_url}")
```

## Related Documentation

- [`.env.example`](.env.example) - Configuration template
- [`RAG_CHAT_DOCUMENTATION.md`](RAG_CHAT_DOCUMENTATION.md) - RAG chat feature
- [`README.md`](README.md) - Main package documentation

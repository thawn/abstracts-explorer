# Config Module

The config module provides configuration management for Abstracts Explorer.

## Overview

The configuration system supports:

- Environment variables
- `.env` file loading
- Type conversion (string, int, float)
- Singleton pattern for global configuration
- Priority-based configuration loading

## Class Reference

```{eval-rst}
.. automodule:: abstracts_explorer.config
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
```

## Usage Examples

### Getting Configuration

```python
from abstracts_explorer.config import get_config

# Get singleton instance
config = get_config()

# Access configuration values
print(f"Chat model: {config.chat_model}")
print(f"Backend URL: {config.llm_backend_url}")
print(f"Database URL: {config.database_url}")
```

### Configuration Values

```python
# Chat/LLM settings
config.chat_model                    # str
config.chat_temperature              # float
config.chat_max_tokens              # int
config.llm_backend_url              # str
config.llm_backend_auth_token       # str

# Embedding settings
config.embedding_model              # str
config.embedding_db_path            # str (for local ChromaDB)
config.embedding_db_url             # str (for remote ChromaDB)
config.collection_name              # str

# Database settings
config.database_url                 # str (SQLAlchemy-compatible URL)

# RAG settings
config.max_context_papers           # int
```

### Custom .env File

```python
from abstracts_explorer.config import load_env_file

# Load from specific file
env_vars = load_env_file("/path/to/custom.env")

# Use with os.environ
import os
for key, value in env_vars.items():
    os.environ[key] = value
```

## Configuration Priority

Settings are loaded in order (later overrides earlier):

1. **Built-in defaults** - Hardcoded in Config class
2. **`.env` file** - In current directory
3. **Environment variables** - System environment
4. **CLI arguments** - Command-line overrides (when applicable)

### Example Priority

```bash
# 1. Default in code
chat_model = "gemma-3-4b-it-qat"

# 2. .env file (overrides default)
CHAT_MODEL=llama-3.2-3b-instruct

# 3. Environment variable (overrides .env)
export CHAT_MODEL=diffbot-small-xl-2508

# 4. CLI argument (overrides all)
abstracts-explorer chat --model custom-model
```

## .env File Format

The `.env` file uses simple `KEY=VALUE` format:

```bash
# Comments start with #
CHAT_MODEL=gemma-3-4b-it-qat

# Quotes are optional
LLM_BACKEND_URL=http://localhost:1234

# Empty values allowed
LLM_BACKEND_AUTH_TOKEN=

# No spaces around =
CHAT_TEMPERATURE=0.7
```

### Supported Features

- Comments (`#`)
- Empty lines (ignored)
- Quoted values (`"value"` or `'value'`)
- Unquoted values
- Empty values

### Not Supported

- Variable expansion (`$VAR`)
- Multi-line values
- Export statements (`export VAR=value`)
- Inline comments (`VAR=value # comment`)

## Type Conversion

The Config class automatically converts types:

```python
# String values
config.chat_model          # str: "gemma-3-4b-it-qat"
config.llm_backend_url     # str: "http://localhost:1234"

# Integer values
config.chat_max_tokens     # int: 1000
config.max_context_papers  # int: 5

# Float values
config.chat_temperature    # float: 0.7
```

## Default Values

Default values when not configured:

```python
DATA_DIR = "data"

CHAT_MODEL = "diffbot-small-xl-2508"
CHAT_TEMPERATURE = 0.7
CHAT_MAX_TOKENS = 1000

EMBEDDING_MODEL = "text-embedding-qwen3-embedding-4b"
EMBEDDING_DB_PATH = "chroma_db"
EMBEDDING_DB_URL = ""

LLM_BACKEND_URL = "http://localhost:1234"
LLM_BACKEND_AUTH_TOKEN = ""

PAPER_DB = "abstracts.db"  # Converted to database_url internally
COLLECTION_NAME = "papers"
MAX_CONTEXT_PAPERS = 5
```

## Configuration in Tests

Tests use configuration from environment:

```python
import pytest
from abstracts_explorer.config import get_config

def test_with_config():
    config = get_config()
    
    # Tests use configured values
    assert config.chat_model is not None
    assert config.llm_backend_url is not None
```

### Overriding in Tests

```python
import os
import pytest

@pytest.fixture
def custom_config():
    # Save original
    original = os.environ.get('CHAT_MODEL')
    
    # Override
    os.environ['CHAT_MODEL'] = 'test-model'
    
    yield
    
    # Restore
    if original:
        os.environ['CHAT_MODEL'] = original
    else:
        del os.environ['CHAT_MODEL']

def test_with_custom_config(custom_config):
    config = get_config()
    assert config.chat_model == 'test-model'
```

## Security Best Practices

### Do Not Commit Secrets

```bash
# .gitignore
.env
*.env
!.env.example
```

### Use Environment Variables in Production

```bash
# Production environment
export LLM_BACKEND_AUTH_TOKEN="secret-token"
export LLM_BACKEND_URL="https://production-api.example.com"
```

### Provide Template

```bash
# .env.example (commit this)
CHAT_MODEL=gemma-3-4b-it-qat
LLM_BACKEND_URL=http://localhost:1234
LLM_BACKEND_AUTH_TOKEN=

# Users copy and customize
cp .env.example .env
```

## Best Practices

1. **Use .env for development** - Easy local configuration
2. **Use environment variables in production** - Secure and flexible
3. **Document all settings** - Keep .env.example up to date
4. **Validate configuration** - Check required settings exist
5. **Use defaults wisely** - Provide sensible defaults
6. **Don't commit secrets** - Use .gitignore properly

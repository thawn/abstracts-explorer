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

- **CHAT_MODEL**: The LLM model to use for RAG chat (default: `diffbot-small-xl-2508`)
- **CHAT_TEMPERATURE**: Temperature for LLM responses, 0.0-2.0 (default: `0.7`)
- **CHAT_MAX_TOKENS**: Maximum tokens in LLM responses (default: `2000`)
- **CLUSTER_LABEL_MODEL**: Chat model used to auto-generate cluster labels during clustering (default: `alias-fast`)

### Embedding Model Settings

- **EMBEDDING_MODEL**: The embedding model to use (default: `text-embedding-qwen3-embedding-4b`)
- **EMBEDDING_MAX_INPUT_TOKENS**: Maximum number of tokens sent to the embedding model in one request (default: `8192`). Longer titles/abstracts are truncated before the API call; the full text is still stored in ChromaDB. Set to `0` to disable proactive truncation.
- **REQUESTS_PER_MINUTE**: Maximum embedding API requests per minute, `0` disables rate limiting (default: `60`). Can be overridden per command with `--requests-per-minute`.
- **WEB_RATE_LIMIT_PER_MINUTE**: Maximum expensive web API requests per client per minute for chat, semantic search, cluster search, and topic evolution; `0` disables client rate limiting (default: `30`). The limiter is per-process; multi-worker deployments need a shared store for global enforcement.

### LLM Backend Configuration

- **LLM_BACKEND_URL**: URL of the LLM backend server (default: `http://localhost:1234`)
- **LLM_BACKEND_AUTH_TOKEN**: Optional authentication token for LLM backend (default: empty)
- **LLM_CONNECT_TIMEOUT**: TCP/TLS connect (and connection-pool) timeout in seconds for LLM and embedding HTTP requests (default: `10`)
- **LLM_REQUEST_TIMEOUT**: Timeout in seconds for one HTTP request to the LLM backend, applied to RAG, embeddings, clustering labels, and evaluation calls (default: `60`). It is clamped to `RAG_QUERY_TIMEOUT`, so it can never exceed the overall RAG query limit.

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

PostgreSQL connections use the `psycopg` v3 driver, which is installed with the
package. Plain `postgresql://` URLs select this driver automatically; no
`postgresql+psycopg://` prefix is required.

##### PostgreSQL search indexes

Keyword search uses `ILIKE '%term%'` predicates on `title`, `abstract`,
`keywords`, and `authors`. These predicates cannot use ordinary B-tree indexes,
so PostgreSQL may scan the `papers` table. For larger databases, create the
managed trigram and composite indexes:

```bash
uv run python scripts/migrate_pg_search_indexes.py --paper-db "$PAPER_DB" --dry-run
uv run python scripts/migrate_pg_search_indexes.py --paper-db "$PAPER_DB"
```

The migration:

- installs the `pg_trgm` extension if needed,
- creates GIN trigram indexes for title, abstract, keywords, and authors,
- creates composite indexes on `(conference, year)` and
  `(conference, year, session)`,
- runs `ANALYZE papers`.

It is manual and idempotent. There is no automatic migration at application
startup. Rerunning the script skips valid indexes and recreates invalid ones.

Indexes preserve existing substring search semantics. They increase disk usage
and add write/import overhead; this is usually acceptable for a read-heavy
paper database. Use `--no-concurrently` only for small or development databases,
because blocking index builds lock writes.

Rollback:

```bash
uv run python scripts/migrate_pg_search_indexes.py --paper-db "$PAPER_DB" --drop
```

For container deployments, see [Docker: Create PostgreSQL search indexes](docker.md#create-postgresql-search-indexes).

##### Metadata cache

- **METADATA_CACHE_TTL**: Time-to-live in seconds for cached PostgreSQL metadata
  queries such as conference lists, year lists, session lists, and the
  conference/year mapping (default: `86400`).
  - Set to `0` to disable caching.
  - Applies only to PostgreSQL; SQLite always queries directly.
  - The cache is process-local. Restart web-ui after external imports so it
    reloads metadata immediately; otherwise entries expire after the TTL.

#### Embedding Database

- **EMBEDDING_DB**: ChromaDB location. Can be either:
  - **HTTP URL**: `http://chromadb:8000` for remote ChromaDB service (Docker deployments)
  - **File path**: `chroma_db` (relative to DATA_DIR) or `/absolute/path/to/chroma_db` for local ChromaDB
  - Default: `chroma_db`

The configuration automatically detects the type based on the format:
- URLs starting with `http://` or `https://` are treated as remote ChromaDB services
- Other values are treated as local file paths (relative to DATA_DIR unless absolute)

### RAG Settings

- **COLLECTION_NAME**: ChromaDB collection name (default: `papers`)
- **MAX_CONTEXT_PAPERS**: Number of papers to include in RAG context (default: `5`)
- **RAG_MAX_PARALLEL_TOOLS**: Maximum number of tool calls executed concurrently by the RAG agent; excess calls are queued (default: `8`)
- **RAG_REQUEST_LIMIT**: Maximum number of model requests per RAG query (default: `50`)
- **RAG_QUERY_TIMEOUT**: Wall-clock timeout in seconds for one RAG query (default: `120`). This must not exceed the reverse proxy's `proxy_read_timeout` (nginx default used in this project: `120s`); otherwise the proxy may close the connection before the RAG query finishes. `LLM_REQUEST_TIMEOUT` is clamped to this value.
- **ENABLE_QUERY_REWRITING**: Rewrite follow-up questions into self-contained search queries before retrieval (default: `true`)
- **QUERY_SIMILARITY_THRESHOLD**: Maximum embedding distance for a rewritten-query answer to be accepted as a hit (default: `0.7`)

### Web UI Defaults

- **DEFAULT_CONFERENCE**: Pre-selected conference in the web UI (default: `ML4PS`)
- **DEFAULT_YEAR**: Pre-selected year in the web UI; `0` disables the default (default: `0`)
- **IMPRINT_LINK**: Optional URL of a legal notice/imprint shown in the web UI footer (default: empty = not shown)

### Reverse Proxy Header Handling

These settings configure Werkzeug's `ProxyFix` middleware, i.e. the number of
trusted reverse-proxy layers for each `X-Forwarded-*` header. The defaults are
`0` (trust no forwarded headers), so deployments behind nginx should set the
hop counts explicitly. The bundled `docker-compose.yml` and systemd unit already
set them to `1`.

- **PROXY_X_FOR**: Trusted `X-Forwarded-For` proxy hops (default: `0`)
- **PROXY_X_PROTO**: Trusted `X-Forwarded-Proto` proxy hops (default: `0`)
- **PROXY_X_HOST**: Trusted `X-Forwarded-Host` proxy hops (default: `0`)
- **PROXY_X_PREFIX**: Trusted `X-Forwarded-Prefix` proxy hops (default: `0`)

### Web UI Security

- **FLASK_SECRET_KEY**: Flask session signing key. When unset, a stable key is generated once in `<DATA_DIR>/.flask_secret` (mode `0600`).
- **MAX_CONTENT_LENGTH**: Maximum accepted request body size in bytes (default: `2097152`, 2 MiB).
- **CORS_ORIGINS**: Comma-separated allowed browser origins. CORS is disabled when unset; use `*` only for trusted deployments.

### Logging

- **LOG_LEVEL**: Optional root log level override (`DEBUG`, `INFO`, `WARNING`, ...). Empty uses the default level.

### Registry Settings

These settings are used by the `abstracts-explorer registry` commands for sharing data via OCI-compatible container registries (e.g. GitHub Container Registry).

- **GITHUB_TOKEN**: Personal Access Token for authenticating with the container registry. Requires `read:packages` scope for downloads and `write:packages` scope for uploads.
- **REGISTRY_REPOSITORY**: Default OCI repository for registry operations (e.g. `ghcr.io/thawn/abstracts-data`). Can be overridden per command with `-r`.

```bash
# .env file — registry settings
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
REGISTRY_REPOSITORY=ghcr.io/thawn/abstracts-data
```

See [Registry documentation](registry.md) for complete usage instructions.

## Example Configurations

### Local Development (SQLite)

```bash
# .env file for local development

# Base directory for data files
DATA_DIR=data

CHAT_MODEL=diffbot-small-xl-2508
CHAT_TEMPERATURE=0.7
CHAT_MAX_TOKENS=2000

EMBEDDING_MODEL=text-embedding-qwen3-embedding-4b
EMBEDDING_MAX_INPUT_TOKENS=8192

LLM_BACKEND_URL=http://localhost:1234
LLM_BACKEND_AUTH_TOKEN=

# SQLite database (relative to DATA_DIR - will resolve to data/abstracts.db)
PAPER_DB=abstracts.db

# Local ChromaDB (relative to DATA_DIR - will resolve to data/chroma_db)
EMBEDDING_DB=chroma_db

COLLECTION_NAME=papers
MAX_CONTEXT_PAPERS=5
```

### Production/Docker (PostgreSQL)

```bash
# .env file for production with PostgreSQL

DATA_DIR=data

CHAT_MODEL=diffbot-small-xl-2508
CHAT_TEMPERATURE=0.7
CHAT_MAX_TOKENS=2000

EMBEDDING_MODEL=text-embedding-qwen3-embedding-4b
EMBEDDING_MAX_INPUT_TOKENS=8192

LLM_BACKEND_URL=http://localhost:1234
LLM_BACKEND_AUTH_TOKEN=

# PostgreSQL database URL
PAPER_DB=postgresql://abstracts:password@postgres:5432/abstracts

# Remote ChromaDB HTTP service
EMBEDDING_DB=http://chromadb:8000

COLLECTION_NAME=papers
MAX_CONTEXT_PAPERS=5
```

### Alternative: Absolute Paths

```bash
# Using absolute paths for both databases
PAPER_DB=/var/data/abstracts.db
EMBEDDING_DB=/var/data/chroma_db
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

# ChromaDB location (automatically detected as URL or path)
print(f"ChromaDB: {config.embedding_db}")
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

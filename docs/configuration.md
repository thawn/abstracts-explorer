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
- **CHAT_MAX_TOKENS**: Maximum tokens in LLM responses (default: `4000`)
- **CLUSTER_LABEL_MODEL**: Chat model used to auto-generate cluster labels during clustering (default: `alias-fast`)
- **CLUSTER_LABEL_MAX_WORKERS**: Maximum number of cluster-label LLM requests submitted in parallel per labeling batch (default: `8`). Set to `1` to label sequentially. Throughput remains bounded by `REQUESTS_PER_MINUTE`, so parallelism only hides per-request latency.
- **LLM_MAX_ATTEMPTS**: Total HTTP attempts for any LLM request, including cluster labels, embeddings, evaluation, ORCID, and RAG chat (default: `6`). Set to `1` to disable retries. This maps to the OpenAI SDK's `max_retries = LLM_MAX_ATTEMPTS - 1`; the SDK owns retryable status classification, `Retry-After` handling, and exponential backoff.
- **LLM_RETRY_BASE_DELAY**: Base delay in seconds for retrying cluster-label responses that contain no usable content (default: `1.0`). HTTP-level backoff is handled by the OpenAI SDK.
- **LLM_RETRY_MAX_DELAY**: Maximum delay in seconds for empty-content cluster-label retries (default: `10.0`)
- **LLM_RATE_LIMIT_DELAY**: Cooldown applied after an HTTP `429` when the backend does not send a usable `Retry-After` header (default: `60.0`). The cooldown is shared process-wide so parallel sync and async LLM requests pause together.
- **ADAPTIVE_RATE_LIMIT_ENABLED**: Whether the shared rate limiter lowers its effective request rate after sustained `429` responses and recovers it after successful requests (default: `true`). Disable to keep the rate fixed at `REQUESTS_PER_MINUTE`.
- **ADAPTIVE_MIN_RPM**: Lower bound for the adaptively throttled request rate in requests per minute (default: `6`). The limiter halves the effective rate on each `429` down to this floor and restores it as requests succeed again.

### Embedding Model Settings

- **EMBEDDING_MODEL**: The embedding model to use (default: `text-embedding-qwen3-embedding-4b`)
- **EMBEDDING_MAX_INPUT_TOKENS**: Maximum number of tokens sent to the embedding model in one request (default: `8192`). Longer titles/abstracts are truncated before the API call; the full text is still stored in ChromaDB. Set to `0` to disable proactive truncation.
- **REQUESTS_PER_MINUTE**: Maximum OpenAI-compatible API requests per minute. This covers embedding and LLM (chat-completion) calls because both share the same rate-limited HTTP client and global token bucket. Requests are paced smoothly with a one-request burst so an initial burst cannot exhaust a provider’s per-minute quota. Set this at or below the server’s global per-minute quota; with adaptive throttling enabled the effective rate is automatically reduced on sustained `429` responses and recovers afterwards. Requests waiting for a token re-check the shared cooldown, so parallel workers do not fire during a pause. `0` disables rate limiting (default: `60`). Can be overridden per command with `--requests-per-minute`.
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

- **MIN_FREE_DISK_MB**: Minimum free disk space in MiB required before writing
  embeddings to a local ChromaDB store (default: `4096`). If the filesystem has
  less space available, `create-embeddings` fails fast instead of risking a
  corrupted vector index. Set to `0` to disable the check. Ignored for HTTP
  `EMBEDDING_DB` values.

!!! warning "Local ChromaDB is single-process"
    A local ChromaDB (`EMBEDDING_DB` file path) is locked for the lifetime of
    the process using it. Do **not** run multiple `create-embeddings`,
    clustering, web UI, or MCP processes against the same local store
    simultaneously: each process persists its own copy of the HNSW index, and
    concurrent writers can corrupt the store. A second process now fails fast
    with an "Another process is using the local ChromaDB" error. For concurrent
    access, use an HTTP ChromaDB server (`EMBEDDING_DB=http://...`).

    If a local store was corrupted by concurrent writers, stop all processes,
    move the store aside, and rebuild it with `create-embeddings` (the paper
    database is unaffected).

### RAG Settings

- **COLLECTION_NAME**: ChromaDB collection name (default: `papers`)
- **MAX_CONTEXT_PAPERS**: Number of papers to include in RAG context (default: `5`)
- **RAG_MAX_PARALLEL_TOOLS**: Maximum number of tool calls executed concurrently by the RAG agent; excess calls are queued (default: `8`)
- **RAG_REQUEST_LIMIT**: Maximum number of model requests per RAG query (default: `50`)
- **RAG_QUERY_TIMEOUT**: Wall-clock timeout in seconds for one RAG query (default: `120`). This must not exceed the reverse proxy's `proxy_read_timeout` (nginx default used in this project: `120s`); otherwise the proxy may close the connection before the RAG query finishes. `LLM_REQUEST_TIMEOUT` is clamped to this value.
- **CHAT_HISTORY_TTL**: Idle time in seconds after which a web chat session's in-memory history is expired (default: `86400`, i.e. 24 hours). Set to `0` to disable expiry. An hourly background sweep removes idle sessions, and stale history is also discarded on next access.
- **ENABLE_QUERY_REWRITING**: Rewrite follow-up questions into self-contained search queries before retrieval (default: `true`)
- **QUERY_SIMILARITY_THRESHOLD**: Maximum embedding distance for a rewritten-query answer to be accepted as a hit (default: `0.7`)

### Chat Input Guard (Ollaya)

The optional chat input guard classifies each user message with a local
[Ollaya](https://ollaya.dev/) decision model before the RAG agent runs. It
refuses off-topic, coding, general-purpose LLM, and jailbreak requests with a
polite fixed message. The guard is **fail-open**: if Ollaya is disabled,
unreachable, slow, or returns an unexpected response, chat proceeds normally.

- **GUARD_ENABLED**: Enable the guard (default: `false`). Container deployments set `true`.
- **GUARD_BACKEND_URL**: Ollaya base URL (default: `http://127.0.0.1:11435`). The bundled containers use `http://abstracts-ollaya:11435`.
- **GUARD_API_KEY**: Bearer token for Ollaya (default: empty). Inject it via the `ollaya-api-key` Podman secret in container deployments.
- **GUARD_MODEL**: Ollaya model name (default: `laya:multilingual`).
- **GUARD_TIMEOUT**: Per-check HTTP timeout in seconds (default: `2.0`).
- **GUARD_TOPIC_THRESHOLD**: Minimum probability for the off-topic/abuse scope choice to block a message (default: `0.8`).
- **GUARD_JAILBREAK_THRESHOLD**: Minimum probability for the jailbreak yes/no question to block a message (default: `0.8`).
- **GUARD_MAX_CHARS**: Maximum number of characters sent to Ollaya (default: `1500`).
- **GUARD_REFUSAL_MESSAGE**: Optional custom refusal text. Empty uses the built-in message.

For a host-run development server (`uv run abstracts-explorer ...`), publish
Ollaya on loopback and use the matching `GUARD_API_KEY`:

```bash
podman volume create abstracts-ollaya

podman run --rm \
  -v abstracts-ollaya:/home/ollaya/.ollaya \
  ghcr.io/ollaya-dev/ollaya pull laya:multilingual

podman run -d --name abstracts-ollaya \
  -p 127.0.0.1:11435:11435 \
  -v abstracts-ollaya:/home/ollaya/.ollaya \
  -e OLLAYA_PRELOAD=laya:multilingual \
  -e OLLAYA_KEEP_ALIVE=-1 \
  -e OLLAYA_THREADS=4 \
  -e OLLAYA_API_KEY="$GUARD_API_KEY" \
  ghcr.io/ollaya-dev/ollaya
```

For the containerized stack, Ollaya is reached over the internal network at
`http://abstracts-ollaya:11435` and no host port is needed. See
[Docker and Podman Setup](docker.md#optional-chat-input-guard-ollaya) and the
[local development walkthrough](contributing.md#5-optional-local-ollaya-guard).

### Web UI Defaults

- **DEFAULT_CONFERENCE**: Pre-selected conference in the web UI (default: `ML4PS`)
- **DEFAULT_YEAR**: Pre-selected year in the web UI; `0` disables the default (default: `0`)
- **IMPRINT_LINK**: Optional URL of a legal notice/imprint shown in the web UI footer (default: empty = not shown)
- **WEB_UI_THREADS**: Number of Waitress worker threads for the web UI (default: `6`). Values below `1` are clamped to `1`. Ignored when the Flask development server is used (`--dev`). The `web-ui --threads` option overrides this setting.

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

### OpenReview Authentication (ICLR)

The ICLR downloader uses OpenReview for years 2017–2019. The OpenReview API
can be protected by a bot challenge; logging in with an account may avoid it.
When an unauthenticated request is rejected and no credentials are configured,
the downloader prompts for an OpenReview email and password in an interactive
terminal. Leave the email empty to skip the prompt and use the Semantic Scholar
fallback instead. In non-interactive environments, configure the variables
below or pass the matching `--openreview-*` command-line options.

- **OPENREVIEW_EMAIL**: OpenReview account email.
- **OPENREVIEW_PASSWORD**: OpenReview account password.
- **OPENREVIEW_TOKEN**: Existing OpenReview bearer token. Takes precedence over
  email/password.

The same values can be passed per command with `--openreview-email`,
`--openreview-password`, and `--openreview-token`. Prefer environment variables
or `.env` for the password so it does not appear in shell history.

```bash
# .env file — OpenReview authentication
OPENREVIEW_EMAIL=you@example.org
OPENREVIEW_PASSWORD=your_password
# OPENREVIEW_TOKEN=alternative_existing_token
```

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

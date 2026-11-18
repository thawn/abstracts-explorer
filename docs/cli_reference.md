# Command-Line Interface Reference

Complete reference for all `abstracts-explorer` CLI commands, generated from
the actual `--help` output of the tool.

**Note:** When running from the project directory, prefix commands with
`uv run`, or activate the virtual environment first with
`source .venv/bin/activate`.

## Commands overview

| Command | Purpose |
| ------- | ------- |
| `download` | Download conference data and create database |
| `create-embeddings` | Generate embeddings for abstracts |
| `search` | Search the vector database for similar papers |
| `chat` | Interactive RAG chat with papers |
| `web-ui` | Start the web interface |
| `clustering` | Clustering commands (`run`, `clear-cache`, `pre-generate`) |
| `mcp-server` | Start MCP server for cluster analysis |
| `eval` | Automatic evaluation of the RAG system (`generate`, `verify`, `run`, `results`, `clear`) |
| `registry` | Upload/download data to/from OCI container registries (`upload`, `download`, `list`, `delete`) |
| `feedback` | Explore user feedback (`stats`, `list`, `browse`, `clear`) |
| `delete-data` | Delete all data for a conference/year from all databases |
| `pre-process` | Run the full pipeline (download → embeddings → clustering) |

## Global options

```text
-v, --verbose   Increase verbosity (can be repeated: -v for INFO, -vv for DEBUG)
-h, --help      Show help message and exit
--version       Show version information
```

## download

Download papers from various sources using plugins. Without arguments,
downloads ALL conferences for ALL available years (plugins that require
manual input are skipped).

```text
usage: abstracts-explorer download [-h] [--output OUTPUT] [--conference CONFERENCE]
                                   [--year YEAR] [--force] [--list-plugins]
                                   [--max-workers MAX_WORKERS]
                                   [--request-delay REQUEST_DELAY] [--no-resume]
                                   [--input-file INPUT_FILE]
```

**Options:**

| Option | Description |
| ------ | ----------- |
| `--output OUTPUT` | Output path for intermediate JSON file (default: `data/abstracts.json`) |
| `--conference CONFERENCE` | Conference to use (default: all conferences). Case-insensitive. |
| `--year YEAR` | Year of conference/workshop (default: all available years). |
| `--force` | Force re-download even if file exists |
| `--list-plugins` | List available downloader plugins and exit |
| `--max-workers MAX_WORKERS` | Maximum parallel workers for fetching data (default: 20) |
| `--request-delay REQUEST_DELAY` | Delay before each HTTP request (AGU plugin) |
| `--no-resume` | Discard an interrupted download checkpoint and start fresh (AGU) |
| `--input-file INPUT_FILE` | Path to a pre-downloaded conference JSON file. Required for the `chi` plugin. |

The plugin list is printed dynamically by `--list-plugins` (includes aaai,
aistats, agu, chi, haicon, iclr, icml, igarss, ieeevis, ml4ps, and neurips).

**Examples:**

```bash
# Download ALL conference data for ALL available years
abstracts-explorer download

# Download all years for a specific conference
abstracts-explorer download --conference neurips

# Download a specific conference and year
abstracts-explorer download --conference neurips --year 2025

# Load CHI 2024 papers from a pre-downloaded JSON
abstracts-explorer download --conference chi --year 2024 --input-file chi_2024_program.json

# List available plugins
abstracts-explorer download --list-plugins
```

## create-embeddings

Generate embeddings for paper abstracts using an OpenAI-compatible API.
Without additional arguments, embeds all conference data for all available
years; with `--conference` and/or `--year`, only the matching papers are
embedded.

```text
usage: abstracts-explorer create-embeddings [-h] [--conference CONFERENCE]
                                            [--year YEAR] [--collection COLLECTION]
                                            [--backend-url BACKEND_URL] [--model MODEL]
                                            [--force] [--ignore-embedding-model-mismatch]
                                            [--requests-per-minute REQUESTS_PER_MINUTE]
```

**Options:**

| Option | Description |
| ------ | ----------- |
| `--conference CONFERENCE` | Conference to use (default: all). Case-insensitive. |
| `--year YEAR` | Year of conference/workshop (default: all available). |
| `--collection COLLECTION` | Name for the ChromaDB collection (default: `papers`) |
| `--backend-url BACKEND_URL` | URL for OpenAI-compatible API |
| `--model MODEL` | Name of the embedding model |
| `--force` | Reset collection if it already exists |
| `--ignore-embedding-model-mismatch` | Recreate embeddings after a model mismatch without prompting |
| `--requests-per-minute REQUESTS_PER_MINUTE` | Max API requests per minute, 0 to disable rate limiting (default: 60) |

**Examples:**

```bash
# Embed all papers (default)
abstracts-explorer create-embeddings

# Embed only NeurIPS papers
abstracts-explorer create-embeddings --conference NeurIPS

# Embed only NeurIPS 2024 papers
abstracts-explorer create-embeddings --conference NeurIPS --year 2024

# Embed only papers from 2025 (all conferences)
abstracts-explorer create-embeddings --year 2025
```

## search

Search the vector database for semantically similar papers.

```text
usage: abstracts-explorer search [-h] [--collection COLLECTION]
                                 [--limit N_RESULTS] [--where WHERE]
                                 [--show-abstract] [--backend-url BACKEND_URL]
                                 [--model MODEL] [--distance-threshold DISTANCE_THRESHOLD]
                                 [--conference CONFERENCE] [--year YEAR]
                                 query
```

**Options:**

| Option | Description |
| ------ | ----------- |
| `query` (positional) | Search query text |
| `--collection COLLECTION` | Name of the ChromaDB collection (default: `papers`) |
| `--limit N_RESULTS` | Number of results to return (default: 5) |
| `--where WHERE` | Metadata filter as key=value pairs, comma-separated (e.g., `"decision=Accept (poster)"`) |
| `--show-abstract` | Show paper abstracts in results |
| `--backend-url BACKEND_URL` | URL for OpenAI-compatible API |
| `--model MODEL` | Name of the embedding model |
| `--distance-threshold DISTANCE_THRESHOLD` | Maximum L2 distance for a result to be included (default: 1.2) |
| `--conference CONFERENCE` | Conference to use (default: all). Case-insensitive. |
| `--year YEAR` | Year of conference/workshop (default: all available). |

**Examples:**

```bash
abstracts-explorer search "graph neural networks for molecular generation"

abstracts-explorer search "deep learning transformers" \
  --limit 10 --show-abstract

abstracts-explorer search "reinforcement learning" \
  --where "decision=Accept (poster)"
```

## chat

Start an interactive chat session using RAG to answer questions about papers.

```text
usage: abstracts-explorer chat [-h] [--collection COLLECTION] [--backend-url BACKEND_URL]
                               [--model MODEL] [--embedding-model EMBEDDING_MODEL]
                               [--max-context MAX_CONTEXT] [--temperature TEMPERATURE]
                               [--show-sources] [--export EXPORT]
                               [--conference CONFERENCE] [--year YEAR]
```

**Options:**

| Option | Description |
| ------ | ----------- |
| `--collection COLLECTION` | Name of the ChromaDB collection (default: `papers`) |
| `--backend-url BACKEND_URL` | URL for OpenAI-compatible API |
| `--model MODEL` | Name of the language model for chat |
| `--embedding-model EMBEDDING_MODEL` | Name of the embedding model |
| `--max-context MAX_CONTEXT` | Maximum number of papers to use as context (default: 5) |
| `--temperature TEMPERATURE` | Sampling temperature for generation (default: 0.7) |
| `--show-sources` | Show source papers for each response |
| `--export EXPORT` | Export conversation to JSON file |
| `--conference CONFERENCE` | Conference to use (default: all). Case-insensitive. |
| `--year YEAR` | Year of conference/workshop (default: all available). |

## web-ui

Start a Flask web server with a modern UI for exploring papers.

```text
usage: abstracts-explorer web-ui [-h] [--host HOST] [--port PORT] [--dev]
                                 [--threads THREADS]
```

**Options:**

| Option | Description |
| ------ | ----------- |
| `--host HOST` | Host to bind to (default: `127.0.0.1`) |
| `--port PORT` | Port to bind to (default: 5000) |
| `--dev` | Use the Flask development server instead of Waitress. Note: use `-vv` to enable debug mode with any server. |
| `--threads THREADS` | Number of Waitress worker threads (default: 6). Must be >= 1. Ignored when `--dev` is set. |

## clustering

Clustering commands. Uses agglomerative clustering (ward linkage, distance
threshold 150) with t-SNE dimensionality reduction.

```text
usage: abstracts-explorer clustering [-h] {run,clear-cache,pre-generate} ...
```

### clustering run

Cluster embeddings and optionally export results.

```text
usage: abstracts-explorer clustering run [-h] [--collection COLLECTION]
                                         [--output OUTPUT] [--limit LIMIT]
```

| Option | Description |
| ------ | ----------- |
| `--collection COLLECTION` | Name of the ChromaDB collection (default: `papers`) |
| `--output OUTPUT` | Path to export clustering results as JSON (optional) |
| `--limit LIMIT` | Maximum number of embeddings to process (optional) |

### clustering clear-cache

Clear clustering cache from the database (useful when embeddings change or
the cache becomes stale).

```text
usage: abstracts-explorer clustering clear-cache [-h] [--embedding-model EMBEDDING_MODEL]
```

| Option | Description |
| ------ | ----------- |
| `--embedding-model EMBEDDING_MODEL` | Only clear cache for this embedding model (optional) |

**Examples:**

```bash
# Clear all clustering cache entries
abstracts-explorer clustering clear-cache

# Clear cache for a specific embedding model
abstracts-explorer clustering clear-cache --embedding-model text-embedding-3-large
```

### clustering pre-generate

Pre-generate clustering results and persist them to the database cache so
the web UI serves them instantly.

- Without arguments: generates for every conference combined with each
  individual year.
- With `--conference` only: generates for that conference for each
  individual year.
- With `--conference` and `--year`: generates for that specific
  conference/year only.

```text
usage: abstracts-explorer clustering pre-generate [-h] [--collection COLLECTION]
                                                  [--conference CONFERENCE] [--year YEAR]
                                                  [--force]
                                                  [--requests-per-minute REQUESTS_PER_MINUTE]
```

| Option | Description |
| ------ | ----------- |
| `--collection COLLECTION` | Name of the ChromaDB collection (default: `papers`) |
| `--conference CONFERENCE` / `--year YEAR` | Scope the pre-generation. |
| `--force` | Force recompute even if cache already exists |
| `--requests-per-minute REQUESTS_PER_MINUTE` | Max API requests per minute (default: 60) |

**Examples:**

```bash
# Pre-generate for all conference/year combinations (default)
abstracts-explorer clustering pre-generate

# Pre-generate for a specific conference (each year)
abstracts-explorer clustering pre-generate --conference NeurIPS

# Pre-generate for a specific conference and year
abstracts-explorer clustering pre-generate --conference NeurIPS --year 2024

# Force recompute
abstracts-explorer clustering pre-generate --force
```

## mcp-server

Start a Model Context Protocol (MCP) server for cluster analysis.

```text
usage: abstracts-explorer mcp-server [-h] [--host HOST] [--port PORT]
                                     [--transport {streamable-http,stdio}]
```

| Option | Description |
| ------ | ----------- |
| `--host HOST` | Host to bind to (default: `127.0.0.1`) |
| `--port PORT` | Port to listen on (default: 8000) |
| `--transport {streamable-http,stdio}` | Transport method (default: `streamable-http`) |

**Examples:**

```bash
abstracts-explorer mcp-server
abstracts-explorer mcp-server --host 0.0.0.0 --port 8080
abstracts-explorer mcp-server --transport stdio
```

See the [MCP server guide](mcp_server.md) for tool descriptions.

## delete-data

Delete all data for a specific conference and year from every database
(papers, embeddings, clustering cache). Shows a summary and requires the
user to type `yes` to confirm.

```text
usage: abstracts-explorer delete-data [-h] --conference CONFERENCE --year YEAR [--yes]
```

| Option | Description |
| ------ | ----------- |
| `--conference, -c CONFERENCE` | Conference to delete (required). Case-insensitive. |
| `--year YEAR` | Year of conference/workshop to delete (required). |
| `--yes` | Skip the interactive confirmation prompt. |

**Examples:**

```bash
abstracts-explorer delete-data --conference neurips --year 2024
abstracts-explorer delete-data --conference iclr --year 2023 --yes
```

## pre-process

Run the full pre-processing pipeline:

1. `download` — download conference data
2. `create-embeddings` — generate embeddings for abstracts
3. `clustering pre-generate` — pre-generate clustering results for the web UI

By default the pipeline is parallel: each conference/year unit is embedded
and clustered as soon as its download finishes, while the remaining downloads
keep running. Embeddings and clustering also overlap. Use
`--no-overlap-post-process` to run embeddings and clustering in a single
worker.

```text
usage: abstracts-explorer pre-process [-h] [--conference CONFERENCE] [--year YEAR]
                                      [--force] [--no-overlap-post-process]
                                      [--collection COLLECTION] [--backend-url BACKEND_URL]
                                      [--model MODEL] [--requests-per-minute REQUESTS_PER_MINUTE]
                                      [--ignore-embedding-model-mismatch]
```

| Option | Description |
| ------ | ----------- |
| `--conference CONFERENCE` / `--year YEAR` | Filters forwarded to all steps. |
| `--force` | Force re-download and reset embeddings/clustering |
| `--no-overlap-post-process` | Run embeddings and clustering in a single worker |
| `--collection COLLECTION` | Name for the ChromaDB collection (default: `papers`) |
| `--backend-url BACKEND_URL` | URL for OpenAI-compatible API |
| `--model MODEL` | Name of the embedding model |
| `--requests-per-minute REQUESTS_PER_MINUTE` | Max API requests per minute (default: 60) |
| `--ignore-embedding-model-mismatch` | Recreate embeddings after a model mismatch without prompting |

**Examples:**

```bash
# Process all conferences and years (parallel pipeline)
abstracts-explorer pre-process

# Process only NeurIPS (all years)
abstracts-explorer pre-process --conference neurips

# Re-process, overwriting existing data
abstracts-explorer pre-process --conference neurips --force

# Recreate embeddings after switching embedding models
abstracts-explorer pre-process --ignore-embedding-model-mismatch --model my-new-model

# Do not overlap embeddings and clustering
abstracts-explorer pre-process --no-overlap-post-process
```

## eval

Automatic evaluation of the RAG system.

```text
usage: abstracts-explorer eval [-h] {generate,verify,run,results,clear} ...
```

### eval generate

Generate evaluation Q/A pairs using an LLM.

```text
usage: abstracts-explorer eval generate [-h] [--n-pairs N_PAIRS]
                                        [--tools TOOLS [TOOLS ...]] [--no-followups]
                                        [--n-followups N_FOLLOWUPS] [--model MODEL]
                                        [--backend-url BACKEND_URL]
```

| Option | Description |
| ------ | ----------- |
| `--n-pairs N_PAIRS` | Number of Q/A pairs per tool (default: 2) |
| `--tools TOOLS [TOOLS ...]` | Specific MCP tool(s) (default: all) |
| `--no-followups` | Disable follow-up question generation |
| `--n-followups N_FOLLOWUPS` | Follow-up turns per initial pair (default: 1) |
| `--model MODEL` / `--backend-url BACKEND_URL` | LLM used for generation |

```bash
abstracts-explorer eval generate
abstracts-explorer eval generate --n-pairs 5 --n-followups 2
abstracts-explorer eval generate --tools search_papers
```

### eval verify

Interactively verify, edit, or delete Q/A pairs.

```text
usage: abstracts-explorer eval verify [-h] [--sample SAMPLE] [--all]
```

| Option | Description |
| ------ | ----------- |
| `--sample SAMPLE` | Number of pairs to randomly sample for review |
| `--all` | Review all pairs (including already verified) |

```bash
abstracts-explorer eval verify --sample 10
```

### eval run

Run evaluation on stored Q/A pairs.

```text
usage: abstracts-explorer eval run [-h] [--collection COLLECTION] [--model MODEL]
                                   [--embedding-model EMBEDDING_MODEL]
                                   [--backend-url BACKEND_URL] [--limit LIMIT]
                                   [--include-unverified]
```

| Option | Description |
| ------ | ----------- |
| `--collection COLLECTION` | ChromaDB collection (default: `papers`) |
| `--model MODEL` / `--embedding-model EMBEDDING_MODEL` / `--backend-url BACKEND_URL` | Models used for the run |
| `--limit LIMIT` | Maximum number of pairs to evaluate |
| `--include-unverified` | Include unverified Q/A pairs |

```bash
abstracts-explorer eval run
```

### eval results

Browse and display evaluation results.

```text
usage: abstracts-explorer eval results [-h] [--run-id RUN_ID] [--detail]
                                       [--sample SAMPLE] [--clear] [--yes]
```

| Option | Description |
| ------ | ----------- |
| `--run-id RUN_ID` | Specific run ID to display (default: latest) |
| `--detail` | Show detailed per-pair results |
| `--sample SAMPLE` | Randomly sample N results for browsing |
| `--clear` | Delete all stored results for the selected run |
| `--yes, -y` | Skip confirmation prompt when using `--clear` |

```bash
abstracts-explorer eval results --detail
abstracts-explorer eval results --detail --sample 5
```

### eval clear

Delete all accepted (verified) Q/A pairs.

```text
usage: abstracts-explorer eval clear [-h] [--yes, -y]
```

```bash
abstracts-explorer eval clear
abstracts-explorer eval clear --yes
```

## registry

Upload/download paper databases, embeddings, and clustering cache to/from
OCI-compatible container registries (e.g. GitHub Container Registry). Data
is always transferred as a complete unit per conference and year. Use
`--conference all` to process all conferences at once.

```text
usage: abstracts-explorer registry [-h] {upload,download,list,delete} ...
```

### registry upload

```text
usage: abstracts-explorer registry upload [-h] [--repository REPOSITORY]
                                          [--token TOKEN] [--conference CONFERENCE]
                                          [--year YEAR] [--tag TAG]
```

| Option | Description |
| ------ | ----------- |
| `--repository, -r REPOSITORY` | OCI repository (or `REGISTRY_REPOSITORY` env var) |
| `--token TOKEN` | Authentication token, e.g. GitHub PAT (or `GITHUB_TOKEN` env var) |
| `--conference, -c CONFERENCE` | Conference name or `all` (default: all) |
| `--year YEAR` | Conference year (default: all available years) |
| `--tag TAG` | Custom tag (default: derived from conference, year and embedding model) |

```bash
abstracts-explorer registry upload --conference neurips --year 2024 \
  --repository ghcr.io/thawn/abstracts-data
```

### registry download

```text
usage: abstracts-explorer registry download [-h] [--repository REPOSITORY]
                                            [--token TOKEN] [--conference CONFERENCE]
                                            [--year YEAR] [--tag TAG]
                                            [--embedding-model EMBEDDING_MODEL] [--yes]
                                            [--ignore-embedding-model-mismatch]
```

| Option | Description |
| ------ | ----------- |
| `--repository, -r REPOSITORY` / `--token TOKEN` | Registry location and auth (see upload) |
| `--conference, -c CONFERENCE` | Conference name or `all` (default: all) |
| `--year YEAR` | Conference year (default: all years in the artifact) |
| `--tag TAG` | Custom tag (default: derived from conference, year and embedding model) |
| `--embedding-model EMBEDDING_MODEL` | Embedding model for tag derivation (defaults to local DB metadata / `EMBEDDING_MODEL` env var) |
| `--yes` | Skip confirmation prompt |
| `--ignore-embedding-model-mismatch` | Proceed despite an embedding-model mismatch (updates local model metadata afterwards) |

```bash
abstracts-explorer registry download --conference neurips --year 2024 \
  --repository ghcr.io/thawn/abstracts-data
```

### registry list

```text
usage: abstracts-explorer registry list [-h] [--repository REPOSITORY]
                                        [--token TOKEN] [--tag TAG]
                                        [--conference CONFERENCE] [--year YEAR]
```

| Option | Description |
| ------ | ----------- |
| `--repository, -r REPOSITORY` / `--token TOKEN` | Registry location and auth |
| `--tag TAG` | Inspect a specific tag (shows details instead of listing) |

```bash
abstracts-explorer registry list --repository ghcr.io/thawn/abstracts-data
abstracts-explorer registry list --repository ghcr.io/thawn/abstracts-data --tag neurips-2024
```

### registry delete

Delete registry package versions whose OCI tag version is strictly below a
given version. Untagged (dangling) versions are left untouched.

```text
usage: abstracts-explorer registry delete [-h] [--repository REPOSITORY]
                                          [--token TOKEN] --below-version VERSION
                                          [--conference CONFERENCE] [--dry-run] [--yes]
```

| Option | Description |
| ------ | ----------- |
| `--repository, -r REPOSITORY` / `--token TOKEN` | Registry location and auth |
| `--below-version VERSION` | Delete all versions strictly below this version (PEP 440, e.g. `0.4.0`) |
| `--conference, -c CONFERENCE` | Only delete versions whose tags belong to this conference (optional) |
| `--dry-run` | Preview without deleting |
| `--yes` | Skip confirmation prompt |

```bash
abstracts-explorer registry delete --below-version 0.4.0 \
  --repository ghcr.io/thawn/abstracts-data
abstracts-explorer registry delete --below-version 0.4.0 --dry-run \
  --repository ghcr.io/thawn/abstracts-data
```

## feedback

Explore user feedback collected by the web UI. Two types are tracked:
**chat donations** (anonymized transcripts with thumbs up/down) and
**data donations** (anonymized interesting-paper ratings, 1–5).

```text
usage: abstracts-explorer feedback [-h] {stats,list,browse,clear} ...
```

### feedback stats

```text
usage: abstracts-explorer feedback stats [-h]
```

### feedback list

```text
usage: abstracts-explorer feedback list [-h] [--type {chat,data,all}]
                                        [--rating {up,down}] [--limit LIMIT]
                                        [--offset OFFSET]
```

| Option | Description |
| ------ | ----------- |
| `--type {chat,data,all}` | Type of feedback to list (default: all) |
| `--rating {up,down}` | Filter chat donations by rating (only with `--type chat` or `all`) |
| `--limit LIMIT` | Maximum entries per type (default: 20) |
| `--offset OFFSET` | Entries to skip for pagination (default: 0) |

```bash
abstracts-explorer feedback list --type chat --rating down
abstracts-explorer feedback list --type data --limit 50
```

### feedback browse

```text
usage: abstracts-explorer feedback browse [-h] [--type {chat,data}]
                                          [--rating {up,down}] [--sample SAMPLE]
```

| Option | Description |
| ------ | ----------- |
| `--type {chat,data}` | Type of feedback to browse (default: chat) |
| `--rating {up,down}` | Filter chat donations by rating (only with `--type chat`) |
| `--sample SAMPLE` | Randomly sample N entries to browse |

### feedback clear

```text
usage: abstracts-explorer feedback clear [-h] [--type {chat,data,all}] [--yes, -y]
```

| Option | Description |
| ------ | ----------- |
| `--type {chat,data,all}` | Type of feedback to delete (default: all) |
| `--yes, -y` | Skip the confirmation prompt |

```bash
abstracts-explorer feedback clear
abstracts-explorer feedback clear --type chat --yes
```

## Environment variables

Commands read their defaults from the environment / `.env` file; see the
[configuration guide](configuration.md) for the full list (`PAPER_DB`,
`EMBEDDING_DB`, `COLLECTION_NAME`, `EMBEDDING_MODEL`, `CHAT_MODEL`,
`LLM_BACKEND_URL`, `LLM_BACKEND_AUTH_TOKEN`, `REGISTRY_REPOSITORY`,
`GITHUB_TOKEN`, and more).

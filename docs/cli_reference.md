# Command-Line Interface Reference

Complete reference for all `abstracts-explorer` CLI commands, grouped by
resource. Every top-level command is a noun group; subcommands are verbs.

**Note:** When running from the project directory, prefix commands with
`uv run`, or activate the virtual environment first with
`source .venv/bin/activate`.

## Commands overview

| Command | Purpose |
| ------- | ------- |
| `data download` | Download conference data and create the database |
| `data delete` | Delete all data for a conference/year from all databases |
| `data check` | Check papers, embeddings, and clustering cache for inconsistencies |
| `embeddings create` | Generate embeddings for abstracts |
| `clustering run` | Cluster embeddings for visualization |
| `clustering cache populate` | Compute clustering results and store them in the cache |
| `clustering cache clear` | Clear the clustering cache |
| `papers search` | Search the vector database for similar papers |
| `papers chat` | Interactive RAG chat with papers |
| `pipeline run` | Run the full pipeline (download → embeddings → clustering cache) |
| `serve web` | Start the web interface |
| `serve mcp` | Start the MCP server for cluster analysis |
| `registry push/pull/list/update-conference-image/delete` | Push/pull data to/from OCI-compatible container registries |
| `eval generate/verify/run/list/clear-pairs/clear-results` | Automatic evaluation of the RAG system |
| `feedback stats/list/browse/clear` | Explore user feedback |

## Global options

```text
-v, --verbose   Increase verbosity (can be repeated: -v for INFO, -vv for DEBUG)
--read-only     Refuse database and embedding writes (overrides READ_ONLY)
-h, --help      Show help message and exit
--version       Show version information
```

See [Read-Only Mode](configuration.md#read-only-mode) for details.

## data

### data download

Download papers from various sources using plugins. Without arguments,
downloads ALL conferences for ALL available years (plugins that require
manual input are skipped).

```text
usage: abstracts-explorer data download [-h] [--output OUTPUT]
                                        [--conference CONFERENCE] [--year YEAR]
                                        [--force] [--list-plugins]
                                        [--max-workers MAX_WORKERS]
                                        [--request-delay REQUEST_DELAY]
                                        [--no-resume] [--input-file INPUT_FILE]
```

**Options:**

| Option | Description |
| ------ | ----------- |
| `--output OUTPUT` | Output path for intermediate JSON file (default: `<DATA_DIR>/abstracts.json`) |
| `--conference, -c CONFERENCE` | Conference to use (default: all conferences). Case-insensitive. |
| `--year YEAR` | Year of conference/workshop (default: all available years). |
| `--force` | Force re-download even if file exists (matching checkpoints are still resumed) |
| `--list-plugins` | List available downloader plugins and exit |
| `--max-workers MAX_WORKERS` | Maximum parallel workers for fetching data (default: 20) |
| `--request-delay REQUEST_DELAY` | Delay before each HTTP request (plugins that support it) |
| `--no-resume` | Discard an interrupted download checkpoint and start fresh (AGU) |
| `--input-file INPUT_FILE` | Path to a pre-downloaded conference JSON file. Required for the `chi` plugin. |

The plugin list is printed dynamically by `--list-plugins` (includes aaai,
aistats, agu, chi, haicon, iclr, icml, igarss, ieeevis, ml4ps, and neurips).

**Examples:**

```bash
# Download ALL conference data for ALL available years
abstracts-explorer data download

# Download all years for a specific conference
abstracts-explorer data download --conference neurips

# Download a specific conference and year
abstracts-explorer data download --conference neurips --year 2025

# Load CHI 2024 papers from a pre-downloaded JSON
abstracts-explorer data download --conference chi --year 2024 --input-file chi_2024_program.json

# List available plugins
abstracts-explorer data download --list-plugins
```

Each downloaded `(conference, year)` unit also generates TF-IDF
`generated_keywords` for the new papers. To backfill an existing database
(including staging), use the script-only helper; it is safe to re-run and adds
the column when missing:

```bash
# Preview changes
uv run python scripts/backfill_generated_keywords.py --dry-run --json

# Fill missing keywords, or regenerate with --force for a single scope
uv run python scripts/backfill_generated_keywords.py
uv run python scripts/backfill_generated_keywords.py --conference NeurIPS --year 2025 --force
```

### data delete

Delete all data for a specific conference and year from every database
(papers, embeddings, clustering cache). Shows a summary and requires the
user to type `yes` to confirm.

```text
usage: abstracts-explorer data delete [-h] --conference CONFERENCE --year YEAR [--yes]
```

| Option | Description |
| ------ | ----------- |
| `--conference, -c CONFERENCE` | Conference to delete (required). Case-insensitive. |
| `--year YEAR` | Year of conference/workshop to delete (required). |
| `--yes, -y` | Skip the interactive confirmation prompt. |

**Examples:**

```bash
abstracts-explorer data delete --conference neurips --year 2024
abstracts-explorer data delete --conference iclr --year 2023 --yes
```

### data check

Compare the paper database, ChromaDB embeddings, and clustering cache.

The command reports:

- papers without embeddings, orphaned embeddings, and embeddings whose content
  hash no longer matches the paper text
- embedding-model mismatches between the configured model, the database
  metadata, and the clustering cache
- paper scopes without a clustering cache entry
- duplicate clustering cache rows, cache payloads whose points or totals do not
  match the paper database, points from other scopes, and cache rows built from
  a different collection
- ChromaDB collections other than the active `COLLECTION_NAME`

Use `--fix` to delete duplicate clustering cache rows (keeping the newest row
per scope), cache rows for scopes without papers, and orphan embeddings without
a matching paper row. Missing embeddings, stale embeddings, and missing caches
are never modified automatically; each finding includes a suggested command.

```text
usage: abstracts-explorer data check [-h] [--conference CONFERENCE]
                                    [--year YEAR] [--json] [--fix]
```

| Option | Description |
| ------ | ----------- |
| `--conference, -c CONFERENCE` | Only check one conference. Case-insensitive. |
| `--year YEAR` | Only check one year. |
| `--json` | Print the report as JSON instead of human-readable text. |
| `--fix` | Delete duplicate/stale clustering cache rows and orphan embeddings. |

Exit code is `0` when no errors remain and `1` when errors were found.

**Examples:**

```bash
abstracts-explorer data check
abstracts-explorer data check --conference NeurIPS --year 2025 --json
abstracts-explorer data check --fix
```

## embeddings

### embeddings create

Generate embeddings for paper abstracts using an OpenAI-compatible API.
Without additional arguments, embeds all conference data for all available
years; with `--conference` and/or `--year`, only the matching papers are
embedded.

```text
usage: abstracts-explorer embeddings create [-h] [--conference CONFERENCE]
                                            [--year YEAR] [--collection COLLECTION]
                                            [--embedding-backend-url EMBEDDING_BACKEND_URL]
                                            [--embedding-model EMBEDDING_MODEL]
                                            [--force] [--ignore-embedding-model-mismatch]
                                            [--requests-per-minute REQUESTS_PER_MINUTE]
```

**Options:**

| Option | Description |
| ------ | ----------- |
| `--conference, -c CONFERENCE` | Conference to use (default: all). Case-insensitive. |
| `--year YEAR` | Year of conference/workshop (default: all available). |
| `--collection COLLECTION` | Name for the ChromaDB collection (default: `papers`) |
| `--embedding-backend-url EMBEDDING_BACKEND_URL` | URL for OpenAI-compatible embedding API |
| `--embedding-model EMBEDDING_MODEL` | Name of the embedding model |
| `--force` | Reset collection if it already exists |
| `--ignore-embedding-model-mismatch` | Recreate embeddings after a model mismatch without prompting |
| `--requests-per-minute REQUESTS_PER_MINUTE` | Max API requests per minute, 0 to disable rate limiting (default: 60) |

**Examples:**

```bash
# Embed all papers (default)
abstracts-explorer embeddings create

# Embed only NeurIPS papers
abstracts-explorer embeddings create --conference NeurIPS

# Embed only NeurIPS 2024 papers
abstracts-explorer embeddings create --conference NeurIPS --year 2024

# Embed only papers from 2025 (all conferences)
abstracts-explorer embeddings create --year 2025
```

## clustering

Clustering commands. Uses BERTopic with Agglomerative clustering (ward
linkage, distance threshold 0.5) on 5-D UMAP embeddings and 2-D UMAP
visualization.

```text
usage: abstracts-explorer clustering [-h] {run,cache} ...
```

### clustering run

Cluster embeddings and optionally export results.

```text
usage: abstracts-explorer clustering run [-h] [--collection COLLECTION]
                                        [--output OUTPUT] [--limit LIMIT]
                                        [--no-llm-labels]
                                        [--requests-per-minute REQUESTS_PER_MINUTE]
                                        [--conference CONFERENCE] [--year YEAR]
```

| Option | Description |
| ------ | ----------- |
| `--collection COLLECTION` | Name of the ChromaDB collection (default: `papers`) |
| `--output OUTPUT` | Path to export clustering results as JSON (optional) |
| `--limit LIMIT` | Maximum number of embeddings to process (optional) |
| `--no-llm-labels` | Use c-TF-IDF keywords instead of LLM-generated labels |
| `--requests-per-minute REQUESTS_PER_MINUTE` | Max API requests per minute, 0 to disable rate limiting (default: 60) |
| `--conference, -c CONFERENCE` / `--year YEAR` | Scope the embeddings to cluster. |

LLM labels are generated for flat topics and merged hierarchy levels. Each
cluster receives a short 3-6 word legend label plus a one-sentence description
shown in the web UI hover/tooltip and used by the MCP topic tools. Flat labeling
shows a `Labeling clusters (LLM)` progress bar; hierarchy labeling uses
BERTopic's per-merge progress bar. Failed or empty LLM responses are retried,
then fall back to top c-TF-IDF keywords.

### clustering cache populate

Compute clustering results and persist them to the database cache so the web
UI serves them instantly.

- Without arguments: populates the cache for every conference combined with
  each individual year.
- With `--conference` only: populates the cache for that conference for each
  individual year.
- With `--conference` and `--year`: populates the cache for that specific
  conference/year only.

The clustering cache stores the complete BERTopic result atomically, including
the full hierarchy tree. A cache hit avoids recomputing clusters and reuses
successful labels. `--force` recomputes clustering and labels; there is no
separate resumable label cache.

Hierarchy level settings (`nr_levels`, `level_scale`) are presentation
settings, not clustering parameters. They are not part of the cache key; the
web API and MCP tools derive levels from the cached tree on demand.

A cache hit also repairs labels whose LLM attempt failed. A failed attempt is
identified by a missing descriptive sentence, so only flat clusters and merged
hierarchy nodes without a description are retried. Existing successful labels
are reused through the same BERTopic representation pipeline, without
re-clustering and without LLM calls. A fully labeled cache is a no-op, and
`--no-llm-labels` (c-TF-IDF) cache entries are never modified. Repaired results
are written back to the existing cache row. If the paper metadata needed to
rebuild representative documents is unavailable, the command reports an error
instead of falling back to truncated cached titles.

Clustering caches must contain `hierarchy.tree`. Outdated caches without a tree
are rejected by the web API and MCP tools and must be regenerated with
`clustering cache populate --force`.

```text
usage: abstracts-explorer clustering cache populate [-h] [--collection COLLECTION]
                                                    [--conference CONFERENCE] [--year YEAR]
                                                    [--force] [--no-llm-labels]
                                                    [--requests-per-minute REQUESTS_PER_MINUTE]
```

| Option | Description |
| ------ | ----------- |
| `--collection COLLECTION` | Name of the ChromaDB collection (default: `papers`) |
| `--conference, -c CONFERENCE` / `--year YEAR` | Scope the cache population. |
| `--force` | Force clustering recomputation |
| `--no-llm-labels` | Use c-TF-IDF keywords instead of LLM-generated labels |
| `--requests-per-minute REQUESTS_PER_MINUTE` | Max API requests per minute (default: 60) |

**Examples:**

```bash
# Populate for all conference/year combinations (default)
abstracts-explorer clustering cache populate

# Populate for a specific conference (each year)
abstracts-explorer clustering cache populate --conference NeurIPS

# Populate for a specific conference and year
abstracts-explorer clustering cache populate --conference NeurIPS --year 2024

# Force recompute
abstracts-explorer clustering cache populate --force

# Use c-TF-IDF keywords instead of LLM labels
abstracts-explorer clustering cache populate --no-llm-labels
```

### clustering cache clear

Clear clustering cache from the database (useful when embeddings change or
the cache becomes stale).

```text
usage: abstracts-explorer clustering cache clear [-h] [--embedding-model EMBEDDING_MODEL]
```

| Option | Description |
| ------ | ----------- |
| `--embedding-model EMBEDDING_MODEL` | Only clear cache for this embedding model (optional) |

**Examples:**

```bash
# Clear all clustering cache entries
abstracts-explorer clustering cache clear

# Clear cache for a specific embedding model
abstracts-explorer clustering cache clear --embedding-model text-embedding-3-large
```

## papers

### papers search

Search the vector database for semantically similar papers.

```text
usage: abstracts-explorer papers search [-h] [--collection COLLECTION]
                                        [--limit N_RESULTS] [--where WHERE]
                                        [--show-abstract]
                                        [--embedding-backend-url EMBEDDING_BACKEND_URL]
                                        [--embedding-model EMBEDDING_MODEL]
                                        [--distance-threshold DISTANCE_THRESHOLD]
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
| `--embedding-backend-url EMBEDDING_BACKEND_URL` | URL for OpenAI-compatible embedding API |
| `--embedding-model EMBEDDING_MODEL` | Name of the embedding model |
| `--distance-threshold DISTANCE_THRESHOLD` | Maximum L2 distance for a result to be included (default: 1.2) |
| `--conference, -c CONFERENCE` | Conference to use (default: all). Case-insensitive. |
| `--year YEAR` | Year of conference/workshop (default: all available). |

**Examples:**

```bash
abstracts-explorer papers search "graph neural networks for molecular generation"

abstracts-explorer papers search "deep learning transformers" \
  --limit 10 --show-abstract

abstracts-explorer papers search "reinforcement learning" \
  --where "decision=Accept (poster)"
```

### papers chat

Start an interactive chat session using RAG to answer questions about papers.

```text
usage: abstracts-explorer papers chat [-h] [--collection COLLECTION]
                                      [--llm-backend-url LLM_BACKEND_URL]
                                      [--model MODEL] [--embedding-model EMBEDDING_MODEL]
                                      [--max-context MAX_CONTEXT] [--temperature TEMPERATURE]
                                      [--show-sources] [--export EXPORT]
                                      [--conference CONFERENCE] [--year YEAR]
```

**Options:**

| Option | Description |
| ------ | ----------- |
| `--collection COLLECTION` | Name of the ChromaDB collection (default: `papers`) |
| `--llm-backend-url LLM_BACKEND_URL` | URL for OpenAI-compatible LLM API |
| `--model MODEL` | Name of the language model for chat |
| `--embedding-model EMBEDDING_MODEL` | Name of the embedding model |
| `--max-context MAX_CONTEXT` | Maximum number of papers to use as context (default: 5) |
| `--temperature TEMPERATURE` | Sampling temperature for generation (default: 0.7) |
| `--show-sources` | Show source papers for each response |
| `--export EXPORT` | Export conversation to JSON file |
| `--conference, -c CONFERENCE` | Conference to use (default: all). Case-insensitive. |
| `--year YEAR` | Year of conference/workshop (default: all available). |

## pipeline

### pipeline run

Run the full data processing pipeline:

1. `data download` — download conference data
2. `embeddings create` — generate embeddings for abstracts
3. `clustering cache populate` — compute clustering results for the web UI

By default the pipeline is parallel: each conference/year unit is embedded
and clustered as soon as its download finishes, while the remaining downloads
keep running. Embeddings and clustering also overlap. Use
`--serial-post-process` to run embeddings and clustering in a single worker.

```text
usage: abstracts-explorer pipeline run [-h] [--conference CONFERENCE] [--year YEAR]
                                      [--force] [--no-llm-labels]
                                      [--serial-post-process]
                                      [--collection COLLECTION]
                                      [--embedding-backend-url EMBEDDING_BACKEND_URL]
                                      [--embedding-model EMBEDDING_MODEL]
                                      [--requests-per-minute REQUESTS_PER_MINUTE]
                                      [--ignore-embedding-model-mismatch]
```

| Option | Description |
| ------ | ----------- |
| `--conference, -c CONFERENCE` / `--year YEAR` | Filters forwarded to all steps. |
| `--force` | Force re-download and reset embeddings/clustering |
| `--no-llm-labels` | Use c-TF-IDF keywords instead of LLM-generated labels |
| `--serial-post-process` | Run embeddings and clustering in a single worker |
| `--collection COLLECTION` | Name for the ChromaDB collection (default: `papers`) |
| `--embedding-backend-url EMBEDDING_BACKEND_URL` | URL for OpenAI-compatible embedding API |
| `--embedding-model EMBEDDING_MODEL` | Name of the embedding model |
| `--requests-per-minute REQUESTS_PER_MINUTE` | Max API requests per minute (default: 60) |
| `--ignore-embedding-model-mismatch` | Recreate embeddings after a model mismatch without prompting |

**Examples:**

```bash
# Process all conferences and years (parallel pipeline)
abstracts-explorer pipeline run

# Process only NeurIPS (all years)
abstracts-explorer pipeline run --conference neurips

# Re-process, overwriting existing data
abstracts-explorer pipeline run --conference neurips --force

# Use c-TF-IDF keywords instead of LLM labels
abstracts-explorer pipeline run --no-llm-labels

# Recreate embeddings after switching embedding models
abstracts-explorer pipeline run --ignore-embedding-model-mismatch --embedding-model my-new-model

# Do not overlap embeddings and clustering
abstracts-explorer pipeline run --serial-post-process
```

Pipeline downloads generate `generated_keywords` automatically. Run
`uv run python scripts/backfill_generated_keywords.py` once for pre-existing
databases; see `data download` above for the available backfill options.

## serve

### serve web

Start a Flask web server with a modern UI for exploring papers.

```text
usage: abstracts-explorer serve web [-h] [--host HOST] [--port PORT] [--dev]
                                    [--threads THREADS]
```

**Options:**

| Option | Description |
| ------ | ----------- |
| `--host HOST` | Host to bind to (default: `127.0.0.1`) |
| `--port PORT` | Port to bind to (default: 5001) |
| `--dev` | Use the Flask development server instead of Waitress. Note: use `-vv` to enable debug mode with any server. |
| `--threads THREADS` | Number of Waitress worker threads (default: `WEB_UI_THREADS` or 6). Must be >= 1. Ignored when `--dev` is set. |

### serve mcp

Start a Model Context Protocol (MCP) server for cluster analysis.

```text
usage: abstracts-explorer serve mcp [-h] [--host HOST] [--port PORT]
                                    [--transport {streamable-http,stdio}]
```

| Option | Description |
| ------ | ----------- |
| `--host HOST` | Host to bind to (default: `127.0.0.1`) |
| `--port PORT` | Port to listen on (default: 8000) |
| `--transport {streamable-http,stdio}` | Transport method (default: `streamable-http`) |

**Examples:**

```bash
abstracts-explorer serve mcp
abstracts-explorer serve mcp --host 0.0.0.0 --port 8080
abstracts-explorer serve mcp --transport stdio
```

See the [MCP server guide](mcp_server.md) for tool descriptions.

## eval

Automatic evaluation of the RAG system.

```text
usage: abstracts-explorer eval [-h] {generate,verify,run,list,clear-pairs,clear-results} ...
```

### eval generate

Generate evaluation Q/A pairs using an LLM.

```text
usage: abstracts-explorer eval generate [-h] [--n-pairs N_PAIRS]
                                        [--tools TOOLS [TOOLS ...]]
                                        [--no-followups | --n-followups N_FOLLOWUPS]
                                        [--model MODEL] [--llm-backend-url LLM_BACKEND_URL]
```

| Option | Description |
| ------ | ----------- |
| `--n-pairs N_PAIRS` | Number of Q/A pairs per tool (default: 2) |
| `--tools TOOLS [TOOLS ...]` | Specific MCP tool(s) (default: all) |
| `--no-followups` | Disable follow-up question generation |
| `--n-followups N_FOLLOWUPS` | Follow-up turns per initial pair (default: 1) |
| `--model MODEL` / `--llm-backend-url LLM_BACKEND_URL` | LLM used for generation |

```bash
abstracts-explorer eval generate
abstracts-explorer eval generate --n-pairs 5 --n-followups 2
abstracts-explorer eval generate --tools search_papers
```

### eval verify

Interactively verify, edit, or delete Q/A pairs.

```text
usage: abstracts-explorer eval verify [-h] [--sample SAMPLE | --all]
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
                                   [--llm-backend-url LLM_BACKEND_URL] [--limit LIMIT]
                                   [--include-unverified]
```

| Option | Description |
| ------ | ----------- |
| `--collection COLLECTION` | ChromaDB collection (default: `papers`) |
| `--model MODEL` / `--embedding-model EMBEDDING_MODEL` / `--llm-backend-url LLM_BACKEND_URL` | Models used for the run |
| `--limit LIMIT` | Maximum number of pairs to evaluate |
| `--include-unverified` | Include unverified Q/A pairs |

```bash
abstracts-explorer eval run
```

### eval list

Browse and display evaluation results.

```text
usage: abstracts-explorer eval list [-h] [--run-id RUN_ID] [--detail]
                                    [--sample SAMPLE]
```

| Option | Description |
| ------ | ----------- |
| `--run-id RUN_ID` | Specific run ID to display (default: latest) |
| `--detail` | Show detailed per-pair results |
| `--sample SAMPLE` | Randomly sample N results for browsing |

```bash
abstracts-explorer eval list --detail
abstracts-explorer eval list --detail --sample 5
```

### eval clear-pairs

Delete all accepted (verified) Q/A pairs.

```text
usage: abstracts-explorer eval clear-pairs [-h] [--yes]
```

```bash
abstracts-explorer eval clear-pairs
abstracts-explorer eval clear-pairs --yes
```

### eval clear-results

Delete stored evaluation results for a run.

```text
usage: abstracts-explorer eval clear-results [-h] [--run-id RUN_ID] [--yes]
```

| Option | Description |
| ------ | ----------- |
| `--run-id RUN_ID` | Run ID to delete (default: latest) |
| `--yes, -y` | Skip confirmation prompt |

```bash
abstracts-explorer eval clear-results --run-id 20240101-120000 --yes
```

## registry

Push/pull paper databases, embeddings, and clustering cache to/from
OCI-compatible container registries (e.g. GitHub Container Registry). Data
is always transferred as a complete unit per conference and year. Use
`--conference all` to process all conferences at once.

```text
usage: abstracts-explorer registry [-h] {push,pull,list,update-conference-image,delete} ...
```

### registry push

```text
usage: abstracts-explorer registry push [-h] [--repository REPOSITORY]
                                        [--token TOKEN] [--conference CONFERENCE]
                                        [--year YEAR] [--tag TAG] [--only-tags]
                                        [--dry-run] [--force]
```

| Option | Description |
| ------ | ----------- |
| `--repository, -r REPOSITORY` | OCI repository (or `REGISTRY_REPOSITORY` env var) |
| `--token TOKEN` | Authentication token, e.g. GitHub PAT (or `GITHUB_TOKEN` env var) |
| `--conference, -c CONFERENCE` | Conference name or `all` (default: all) |
| `--year YEAR` | Conference year (default: all available years) |
| `--tag TAG` | Custom tag (default: derived from conference, year and embedding model) |
| `--only-tags` | Do not upload data; re-tag the latest artifact for each tag base under the installed version |
| `--dry-run` | With `--only-tags`, preview re-tags without changing the registry |
| `--force` | With `--only-tags`, overwrite existing installed-version tags |

```bash
abstracts-explorer registry push --conference neurips --year 2024 \
  --repository ghcr.io/thawn/abstracts-data

# Only the version changed (data unchanged): update tags without uploading
abstracts-explorer registry push --only-tags \
  --repository ghcr.io/thawn/abstracts-data
abstracts-explorer registry push --conference neurips --only-tags --dry-run \
  --repository ghcr.io/thawn/abstracts-data
```

### registry update-conference-image

Rebuild the whole-conference image (the conference-level tag that contains
every year) from the existing per-year artifacts. Only a small config blob and
the image manifest are uploaded; existing per-year data layers are reused by
digest, so no data is re-uploaded. Use this after pushing a new year, or to
refresh the whole-conference image after other per-year changes.

```text
usage: abstracts-explorer registry update-conference-image [-h] [--repository REPOSITORY]
                                                          [--token TOKEN] --conference CONFERENCE
                                                          [--embedding-model EMBEDDING_MODEL]
```

| Option | Description |
| ------ | ----------- |
| `--repository, -r REPOSITORY` / `--token TOKEN` | Registry location and auth |
| `--conference, -c CONFERENCE` | Conference name (required) |
| `--embedding-model EMBEDDING_MODEL` | Embedding model (defaults to local DB metadata) |

```bash
abstracts-explorer registry update-conference-image --conference neurips \
  --repository ghcr.io/thawn/abstracts-data
```

### registry pull

```text
usage: abstracts-explorer registry pull [-h] [--repository REPOSITORY]
                                        [--token TOKEN] [--conference CONFERENCE]
                                        [--year YEAR] [--tag TAG]
                                        [--embedding-model EMBEDDING_MODEL] [--yes]
                                        [--ignore-embedding-model-mismatch]
```

| Option | Description |
| ------ | ----------- |
| `--repository, -r REPOSITORY` / `--token TOKEN` | Registry location and auth (see push) |
| `--conference, -c CONFERENCE` | Conference name or `all` (default: all) |
| `--year YEAR` | Conference year (default: all years in the artifact) |
| `--tag TAG` | Custom tag (default: derived from conference, year and embedding model) |
| `--embedding-model EMBEDDING_MODEL` | Embedding model for tag derivation (defaults to local DB metadata / `EMBEDDING_MODEL` env var) |
| `--yes, -y` | Skip confirmation prompt |
| `--ignore-embedding-model-mismatch` | Proceed despite an embedding-model mismatch (updates local model metadata afterwards) |

```bash
abstracts-explorer registry pull --conference neurips --year 2024 \
  --repository ghcr.io/thawn/abstracts-data
```

### registry list

```text
usage: abstracts-explorer registry list [-h] [--repository REPOSITORY]
                                        [--token TOKEN] [--tag TAG]
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
given version, and/or untagged (dangling) versions. At least one of
`--below-version` or `--untagged` is required.

Untagged versions are only removed with `--untagged`. Each candidate is
**verified immediately before deletion**: the version list is re-fetched to
confirm it is still untagged, and its manifest is checked to ensure it is an
orphaned data artifact (not an image index or an attestation/referrer). Anything
that cannot be positively identified as safe is skipped.

```text
usage: abstracts-explorer registry delete [-h] [--repository REPOSITORY]
                                          [--token TOKEN] [--below-version VERSION]
                                          [--conference CONFERENCE] [--untagged]
                                          [--dry-run] [--yes]
```

| Option | Description |
| ------ | ----------- |
| `--repository, -r REPOSITORY` / `--token TOKEN` | Registry location and auth |
| `--below-version VERSION` | Delete all versions strictly below this version (PEP 440, e.g. `0.4.0`) |
| `--conference, -c CONFERENCE` | Only delete versions whose tags belong to this conference (optional) |
| `--untagged` | Also delete verified untagged (dangling) versions; cannot be combined with `--conference` |
| `--dry-run` | Preview without deleting |
| `--yes, -y` | Skip confirmation prompt |

```bash
abstracts-explorer registry delete --below-version 0.4.0 \
  --repository ghcr.io/thawn/abstracts-data
abstracts-explorer registry delete --below-version 0.4.0 --dry-run \
  --repository ghcr.io/thawn/abstracts-data
abstracts-explorer registry delete --untagged \
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
usage: abstracts-explorer feedback clear [-h] [--type {chat,data,all}] [--yes]
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

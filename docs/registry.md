# Registry: Sharing Data via OCI Container Registries

The `registry` command group lets you exchange pre-built paper databases, embeddings, and clustering caches between different instances of Abstracts Explorer via any OCI-compatible container registry (e.g. GitHub Container Registry — `ghcr.io`).

This eliminates the need to re-download paper data, regenerate embeddings, and re-cluster papers on every new instance.

## Overview

Data is packaged and pushed as [OCI artifacts](https://github.com/opencontainers/image-spec) using the [ORAS](https://oras.land/) protocol. Each artifact tag identifies a conference, year, and embedding model, for example:

```
ghcr.io/thawn/abstracts-data:neurips-2024_text-embedding-qwen3-embedding-4b
```

Each uploaded artifact contains:
- **Paper database** (`papers-YYYY.db`) — all papers, clustering cache, hierarchical labels, and embedding metadata for the given conference and year
- **Embeddings** (`embeddings-YYYY.json`) — ChromaDB vector embeddings for all papers

Data for all years of a conference is bundled into a single tag (e.g. `neurips_model-name`) where each year is stored as its own pair of layers. The OCI registry deduplicates blobs by content hash, so no data is stored twice even when individual year tags and the all-years tag both exist.

## Authentication

Downloading from a public registry requires **no authentication**. You can run `registry download` and `registry list` without a token.

Uploading and deleting data requires a GitHub Personal Access Token (PAT) with the `write:packages` scope.

You can provide the token in three ways:

**1. `.env` file (recommended)**

```bash
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
REGISTRY_REPOSITORY=ghcr.io/thawn/abstracts-data
```

**2. Environment variable**

```bash
export GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
```

**3. CLI flag**

```bash
abstracts-explorer registry upload --token $GITHUB_TOKEN -r ghcr.io/thawn/abstracts-data ...
```

## Commands

### `registry upload`

Upload paper database and embeddings for a conference/year to the registry.

**Usage:**

```bash
abstracts-explorer registry upload [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `-r, --repository TEXT` | OCI repository (e.g. `ghcr.io/thawn/abstracts-data`). Falls back to `REGISTRY_REPOSITORY` env var. |
| `--token TEXT` | Registry authentication token. Falls back to `GITHUB_TOKEN` env var. |
| `-c, --conference TEXT` | Conference name to upload (e.g. `neurips`). Use `all` to upload every conference. Case-insensitive. |
| `-y, --year INTEGER` | Year to upload. When omitted, all available years are uploaded. |
| `--yes` | Skip confirmation prompts (for CI/non-interactive use). |

**Upload validation:**

- Raises an error if no papers exist for the specified conference/year
- Raises an error if no embeddings exist for the specified conference/year
- Raises an error if no embedding model is recorded in the local database (run `create-embeddings` first)

**Examples:**

```bash
# Upload NeurIPS 2024 (tag derived automatically from embedding model)
abstracts-explorer registry upload \
  -r ghcr.io/thawn/abstracts-data \
  --token $GITHUB_TOKEN \
  --conference neurips \
  --year 2024

# Upload all available NeurIPS years (pushes individual year tags then an all-years tag)
abstracts-explorer registry upload \
  -r ghcr.io/thawn/abstracts-data \
  --token $GITHUB_TOKEN \
  --conference neurips

# Upload all conferences (non-interactive, for CI)
abstracts-explorer registry upload \
  -r ghcr.io/thawn/abstracts-data \
  --token $GITHUB_TOKEN \
  --conference all \
  --yes
```

### `registry download`

Download paper database and embeddings for a conference/year from the registry.

**Usage:**

```bash
abstracts-explorer registry download [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `-r, --repository TEXT` | OCI repository. Falls back to `REGISTRY_REPOSITORY` env var. |
| `--token TEXT` | Registry authentication token. Falls back to `GITHUB_TOKEN` env var. |
| `-c, --conference TEXT` | Conference name to download (e.g. `neurips`). Use `all` to download every available conference. Case-insensitive. |
| `-y, --year INTEGER` | Year to download. When omitted, all available years are downloaded. |
| `--embedding-model TEXT` | Embedding model name, used to derive the correct tag when no local data exists. Falls back to local DB metadata or `EMBEDDING_MODEL` env var. |
| `--yes` | Skip confirmation prompts (for CI/non-interactive use). |

**Download behavior:**

- Existing local data for the specified conference/year is **replaced** (no merge)
- Both paper database and embeddings are always downloaded together to ensure consistency
- If the downloaded data was created with a different embedding model than the local database, the download is rejected to prevent data corruption

**Embedding model mismatch recovery:**

If your local database uses a different embedding model from the one stored in the registry artifact, but the **configured `EMBEDDING_MODEL`** matches the remote model, the CLI offers to:
1. Clear all local embeddings, clustering cache, and embedding metadata
2. Retry the download with the new model

```
⚠️  Embedding model mismatch detected:
  Local database:  'model-a'
  Downloaded data: 'model-b'

The configured model ('model-b') matches the downloaded data.
To proceed, all local embeddings, clustering cache, and embedding metadata
must be cleared so the new model's data can be imported.
⚠️  This will delete ALL local embeddings and clustering cache!
Clear all local embedding data and retry download? [y/N]:
```

With `--yes`, this clear-and-retry happens automatically without prompting.

**Examples:**

```bash
# Download NeurIPS 2024
abstracts-explorer registry download \
  -r ghcr.io/thawn/abstracts-data \
  --token $GITHUB_TOKEN \
  --conference neurips \
  --year 2024

# Download when no local data exists (specify embedding model explicitly)
abstracts-explorer registry download \
  -r ghcr.io/thawn/abstracts-data \
  --token $GITHUB_TOKEN \
  --conference neurips \
  --year 2024 \
  --embedding-model text-embedding-qwen3-embedding-4b

# Download all available conferences (non-interactive, for CI)
abstracts-explorer registry download \
  -r ghcr.io/thawn/abstracts-data \
  --token $GITHUB_TOKEN \
  --conference all \
  --yes
```

### `registry list`

List available tags in the registry, or inspect a specific tag's metadata.

**Usage:**

```bash
abstracts-explorer registry list [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `-r, --repository TEXT` | OCI repository. Falls back to `REGISTRY_REPOSITORY` env var. |
| `--token TEXT` | Registry authentication token. Falls back to `GITHUB_TOKEN` env var. |
| `--tag TEXT` | Inspect a specific tag and display its metadata (conference, year, embedding model, etc.). |

**Examples:**

```bash
# List all available tags
abstracts-explorer registry list -r ghcr.io/thawn/abstracts-data

# Inspect a specific tag
abstracts-explorer registry list \
  -r ghcr.io/thawn/abstracts-data \
  --tag neurips-2024_text-embedding-qwen3-embedding-4b
```

## Tag Format

Tags are automatically derived from the conference, year, and embedding model:

| Data scope | Tag format | Example |
|-----------|-----------|---------|
| Single year | `{conference}-{year}_{model}` | `neurips-2024_text-embedding-qwen3-embedding-4b` |
| All years | `{conference}_{model}` | `neurips_text-embedding-qwen3-embedding-4b` |

The embedding model name is sanitized for OCI tag compatibility (lowercased, special characters replaced with hyphens, consecutive hyphens collapsed).

## Typical Workflow

### Uploading data from one instance

```bash
# 1. Download papers
abstracts-explorer download --conference neurips --year 2025

# 2. Generate embeddings
abstracts-explorer create-embeddings

# 3. (Optional) cluster embeddings to pre-populate the cache
abstracts-explorer cluster-embeddings

# 4. Upload to registry
abstracts-explorer registry upload \
  -r ghcr.io/thawn/abstracts-data \
  --token $GITHUB_TOKEN \
  --conference neurips \
  --year 2025
```

### Downloading data on another instance

```bash
# Download and import NeurIPS 2025 (no download/embedding generation needed!)
abstracts-explorer registry download \
  -r ghcr.io/thawn/abstracts-data \
  --token $GITHUB_TOKEN \
  --conference neurips \
  --year 2025
```

## Data Integrity

The registry feature is designed for safe, conflict-free imports:

- **No merges**: Downloads always replace existing data for the specified conference+year. This prevents ID conflicts between instances.
- **Atomic import**: If either the paper DB or embedding import fails, any partial changes are rolled back.
- **Pre-validation**: Both the paper DB and embeddings files must be present before any import begins.
- **Embedding model consistency**: The local database rejects imported data created with a different embedding model, preventing silent data corruption.
- **Scoped cache invalidation**: Only clustering cache entries for the imported conference+year are replaced; other conferences and years are unaffected.

## Docker / Container Usage

When running Abstracts Explorer in Docker, you can pre-populate the databases by running a one-off download command inside the container.

See [Docker Setup](docker.md#get-pre-computed-data-from-the-registry) for details.

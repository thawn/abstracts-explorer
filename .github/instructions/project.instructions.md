---
applyTo: "**"
---

# AI Coding Instructions

Conventions for AI assistants working on **abstracts-explorer**. This file
summarizes what agents need; authoritative details live in `docs/`.

## Project Overview

Python package (`>=3.12`, Apache-2.0) that downloads conference data, generates
embeddings, and provides semantic search, RAG chat, clustering, visualization,
an MCP server, and a Flask web UI. Supports SQLite (default) and PostgreSQL.

## Package Management: uv

This project uses **uv**, never pip/venv directly.

- Install: `uv sync` or `uv sync --all-extras` (extras: `dev`, `web`, `docs`)
- Run: `uv run pytest`, `uv run abstracts-explorer ...`
- Add: `uv add <pkg>` or `uv add --dev <pkg>`
- venv: `.venv/`, created automatically

Do not use `pip install -e ...` or bare `pytest`. `uv.lock` and
`package-lock.json` are committed.

## Code Style

- PEP 8, Black line length **118**, `ruff` and `mypy` clean.
- Type hints on all function signatures.
- NumPy-style docstrings on all public functions, classes, and methods.
- No comments unless requested.
- Module-level constants in uppercase.

```python
def example(param: str, count: int = 10) -> bool:
    """
    Brief description.

    Parameters
    ----------
    param : str
        Description.
    count : int, optional
        Description (default: 10).

    Returns
    -------
    bool
        Description.
    """
    return bool(param)
```

## Repository Layout

```
src/abstracts_explorer/
├── config.py            # Configuration (get_config)
├── database.py          # Database manager (SQLAlchemy)
├── db_models.py         # Paper, EmbeddingsMetadata, ClusteringCache
├── embeddings.py        # ChromaDB embeddings
├── clustering.py        # Clustering, reduction, labeling
├── rag.py               # RAG chat (pydantic-ai)
├── mcp_server.py        # MCP server (mcp MCPServer)
├── mcp_tools.py         # MCP tool schemas and execution
├── interesting_papers.py # Summaries and recommendations
├── evaluation.py        # Automatic RAG evaluation
├── registry.py          # OCI registry upload/download (oras)
├── rate_limiter.py      # Token-bucket rate limiting
├── export_utils.py      # ZIP/JSON/Markdown export
├── paper_utils.py       # Paper formatting helpers
├── plugin.py            # Plugin system
├── cli.py               # argparse CLI
├── plugins/             # Conference downloaders
└── web_ui/              # Flask app, ES6 modules, templates, Jest tests

tests/                   # pytest suite (one file per source module)
docs/                    # Sphinx Markdown docs
```

Plugins: `aaai`, `aistats`, `chi`, `haicon`, `iclr`, `icml`, `ieeevis`,
`igarss`, `ml4ps`, `neurips` (plus `json_conference_downloader.py` base).
See `docs/plugins.md` for adding one.

## Testing

- Python: pytest + pytest-mock; target **>90%** coverage.
- JavaScript: Jest + jsdom; target **>80%** coverage.
- One test file per source module (`database.py` → `tests/test_database.py`).
  Test-type exceptions: `test_integration.py`, `test_web_integration.py`,
  `test_web_e2e.py`, `test_staging_e2e.py`.
- Reuse the central test infrastructure instead of duplicating setup:
  `tests/conftest.py` fixtures (`db_manager`, `connected_db`, `test_database`,
  `mock_lm_studio`, `embeddings_manager`, `web_test_papers`, ...) and
  `tests/helpers.py` (`requires_lm_studio`, `create_test_db_with_paper`,
  `find_free_port`, `create_webdriver`, ...).
- Mock external services (APIs, LLM backends, databases). Tests must skip
  gracefully when optional services are unavailable.
- Markers: `slow` (deselected by default), `integration`, `e2e`, `staging`.

```bash
uv run pytest                                  # fast suite
uv run pytest -m slow                          # slow/LM Studio tests
uv run pytest -m ""                            # all markers
uv run pytest --cov=src/abstracts_explorer
uv run pytest tests/test_database.py::test_add_paper

npm install        # also installs vendor assets and git hooks
npm test
npm run test:coverage
```

## Configuration

Priority: **CLI args > environment variables > `.env` > built-in defaults**.
Access via `get_config()` from `abstracts_explorer.config`; copy `.env.example`
to `.env`. Full reference: `docs/configuration.md`.

| Variable | Purpose |
| --- | --- |
| `DATA_DIR` | Base directory for relative data paths (default `data`) |
| `PAPER_DB` | SQLite path (default `abstracts.db`) or PostgreSQL URL |
| `EMBEDDING_DB` | ChromaDB path (default `chroma_db`) or HTTP URL |
| `COLLECTION_NAME` | ChromaDB collection (default `papers`) |
| `CHAT_MODEL`, `CHAT_TEMPERATURE`, `CHAT_MAX_TOKENS` | Chat LLM |
| `CLUSTER_LABEL_MODEL` | Short cluster-label model (default `alias-fast`) |
| `EMBEDDING_MODEL` | Embedding model |
| `LLM_BACKEND_URL`, `LLM_BACKEND_AUTH_TOKEN` | OpenAI-compatible backend |
| `MAX_CONTEXT_PAPERS` | RAG context size (default `5`) |
| `ENABLE_QUERY_REWRITING`, `QUERY_SIMILARITY_THRESHOLD` | RAG query behavior |
| `DEFAULT_CONFERENCE`, `DEFAULT_YEAR` | Web UI initial filters |
| `LOG_LEVEL`, `PROXY_X_*`, `IMPRINT_LINK` | Web/deployment settings |
| `REQUESTS_PER_MINUTE` | API rate limit |
| `GITHUB_TOKEN`, `REGISTRY_REPOSITORY` | OCI registry access |

PostgreSQL also accepts `POSTGRES_HOST`, `POSTGRES_USER`, `POSTGRES_PORT`,
`POSTGRES_DB`, and `ABSTRACTS_DB_PASSWORD`.

## Web API

Routes are defined in `src/abstracts_explorer/web_ui/app.py`; the frontend calls
them from `static/modules/`.

- **Pages/health**: `GET /`, `GET /<conference>`, `GET /health`
- **Stats/filters**: `GET /api/stats`, `/api/filters`, `/api/available-filters`,
  `/api/years`, `/api/papers-per-year`, `/api/embedding-model-check`
- **Papers/search**: `POST /api/search`, `GET /api/paper/<uid>`,
  `POST /api/papers/batch`
- **RAG chat**: `POST /api/chat`, `POST /api/chat/reset`
- **Clustering**: `POST /api/clusters/compute`, `GET /api/clusters/cached`,
  `GET /api/clusters/default-count`, `POST /api/clusters/search`,
  `POST /api/topic-evolution`
- **Export/donation**: `POST /api/export/interesting-papers`,
  `POST /api/donate-data`, `POST /api/donate-chat`

## Architecture

- **Database**: SQLAlchemy 2.0+, SQLite default / PostgreSQL supported. Models:
  `Paper` (indexed title/year/openreview id), `EmbeddingsMetadata`,
  `ClusteringCache`. Schema via `metadata.create_all()`.
- **Embeddings**: ChromaDB; metadata records embedding model and parameters.
- **Clustering**: CLI/web/MCP pipeline is fixed to Agglomerative (ward, distance
  threshold 150) + t-SNE. `clustering.py` still exposes K-Means, DBSCAN,
  Agglomerative, Spectral, Fuzzy C-Means and PCA/t-SNE; UMAP was removed.
  LLM auto-labeling uses TF-IDF keywords; results are cached in the database.
- **RAG**: `pydantic-ai` agent against an OpenAI-compatible backend; it can
  auto-invoke MCP tools.
- **MCP**: `MCPServer` from the `mcp` package. Tools include topic relevance,
  conference topics, topic evolution, paper search, cluster visualization,
  paper details, interesting-paper summaries, and recommendations.
- **Web UI**: Flask 3.0+, 20+ API routes, 15 modular ES6 modules (8 top-level +
  `utils/`), Plotly/Marked/KaTeX/DOMPurify vendor assets, Waitress in production.
- **Registry/evaluation**: OCI artifacts via `oras`; automatic RAG evaluation.

## Common Tasks

- **Feature**: branch → implement (types + NumPy docstrings) → unit and
  integration tests → update docs → run tests → PR.
- **Bugfix**: branch → failing regression test → fix → verify → PR.
- **New plugin**: implement in `plugins/`, inherit `DownloaderPlugin` or
  `LightweightDownloaderPlugin`, register in `plugins/__init__.py`, add
  `tests/test_plugins_<name>.py`, update `docs/plugins.md`.
- **Docs**: edit `docs/*.md` and docstrings, then `cd docs && uv run make html`.

CLI groups: `download`, `create-embeddings`, `pre-process`, `search`, `chat`,
`web-ui`, `clustering {run,clear-cache,pre-generate}`, `mcp-server`,
`eval`, `registry`, `feedback`, `delete-data`. See `docs/cli_reference.md`.

## Git Workflow

Follow `docs/branching_strategy.md`:

- `main`: production-ready, tagged releases.
- `develop`: integration branch for the next release.
- Feature branches: `author/brief-description`, from and back into `develop`.
- Hotfix branches: `hotfix-*`, from `main` into `main` and `develop`.
- Use `--no-ff` merges; write clear descriptive commit messages.

Never commit `.env`, secrets, `data/`, `*.db`, `chroma_db/`, `_build/`,
`node_modules/`, `htmlcov/`, or caches.

## Troubleshooting

- **Tests**: use `uv run`, install with `uv sync --all-extras`, check `.env`.
- **LLM tests skipping**: start LM Studio/Blablador, load configured models,
  verify `LLM_BACKEND_URL`, set `LLM_BACKEND_AUTH_TOKEN` if required.
- **Docs**: `uv sync --extra docs`, fix Markdown errors, `make clean`.
- **Database**: verify `PAPER_DB` path/URL and write permissions; tables are
  auto-created.
- **Clustering**: pre-generate cache with `abstracts-explorer clustering
  pre-generate`; reduce `n_components` or use PCA if memory-bound.

## Resources

- CLI: `docs/cli_reference.md`
- Configuration: `docs/configuration.md`
- Docker/Podman: `docs/docker.md`
- Plugins: `docs/plugins.md`
- Branching: `docs/branching_strategy.md`
- Web UI: `docs/web_ui.md`
- MCP: `docs/mcp_server.md`
- API: `docs/api/`

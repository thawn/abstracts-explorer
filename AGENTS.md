# AGENTS.md

Guidance for AI coding agents working on **abstracts-explorer** (Python package for downloading conference data, semantic search, RAG, clustering, and visualization). Source of truth: `.github/instructions/project.instructions.md`.

## Package Management: uv (not pip/venv)

- Install deps: `uv sync` / `uv sync --all-extras`
- Run tools: `uv run pytest`, `uv run abstracts-explorer ...`
- Add deps: `uv add <pkg>` (or `uv add --dev <pkg>`)
- Extras: `dev`, `web`, `docs`. venv lives in `.venv/`.
- Never use `pip install -e ...` or bare `pytest`.

## Code Style

- PEP 8, 118-char lines (Black), Python >= 3.12.
- Type hints on all function signatures.
- NumPy-style docstrings on all public functions/classes/methods.
- No comments unless asked.

## Layout

- `src/abstracts_explorer/`: `database.py` (SQLAlchemy), `db_models.py`, `embeddings.py` (ChromaDB), `clustering.py`, `rag.py`, `mcp_server.py`, `mcp_tools.py`, `export_utils.py`, `config.py`, `cli.py`, `plugin.py`, `plugins/`, `web_ui/` (Flask + modular ES6).
- `tests/`: pytest suite.
- `docs/`: Sphinx (Markdown).
- `src/abstracts_explorer/web_ui/tests/`: Jest tests.

## Testing

- One test file per source module, same base name (e.g. `database.py` -> `tests/test_database.py`).
- Separate files allowed by test type: `test_integration.py`, `test_web_integration.py`, `test_web_e2e.py`.
- Python: pytest + pytest-mock. Coverage target >90%.
- JavaScript: Jest + jsdom. Coverage target >80%.
- Mock external services (APIs, LLM backends, DBs). Tests must skip gracefully when optional services are unavailable.
- Commands:
  - `uv run pytest`
  - `uv run pytest --cov=src/abstracts_explorer`
  - `uv run pytest tests/test_database.py::test_add_paper`
  - `npm test` / `npm run test:coverage`

## Configuration

- Priority: CLI args > env vars > `.env` > built-in defaults.
- Key vars: `CHAT_MODEL`, `EMBEDDING_MODEL`, `LLM_BACKEND_URL`, `LLM_BACKEND_AUTH_TOKEN`, `PAPER_DB` (SQLite path or PostgreSQL URL), `EMBEDDING_DB_PATH`, `COLLECTION_NAME`, `MAX_CONTEXT_PAPERS`.
- Access via `get_config()` from `abstracts_explorer.config`.
- Never commit `.env` or secrets.

## Architecture Notes

- DB: SQLAlchemy 2.0+, SQLite default / PostgreSQL supported. Models: `Paper`, `EmbeddingsMetadata`, `ClusteringCache`.
- Clustering: CLI/web/MCP pipeline is fixed to Agglomerative (ward, distance threshold 150) + t-SNE. `clustering.py` still exposes 5 algorithms (K-Means, DBSCAN, Agglomerative, Spectral, Fuzzy C-Means) and PCA/t-SNE; UMAP was removed. LLM auto-labeling + DB-backed cache.
- MCP: FastMCP server; RAG chat auto-invokes MCP tools.
- Web UI: Flask 3.0+, 25+ API routes, 15 ES6 modules, Waitress in production.

## Common Tasks

- Feature: branch -> implement (types + NumPy docstrings) -> unit + integration tests -> update docs -> run tests -> PR.
- Bugfix: branch -> failing regression test -> fix -> verify -> PR.
- New plugin: implement in `plugins/`, inherit `DownloaderPlugin`/`LightweightDownloaderPlugin`, register in `plugins/__init__.py`, add `tests/test_plugins_<name>.py`, update `docs/plugins.md`.
- Build docs: `cd docs && uv run make html`.

## Git

- Branches: `main` for releases,`develop` for development, `feat/*`, `fix/*`, `docs/*`.
- Clear descriptive commit messages; only commit when explicitly asked, otherwise suggest a commit message.
- Do not commit `*.db`, `chroma_db/`, `_build/`, `.env`, `__pycache__/`.

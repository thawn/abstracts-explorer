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

## Test Driven Development

- Write a failing test first, then implement the feature/bugfix, then run tests to verify.

## Lint and Format Before Tests

Run the same checks as the git hooks (`.githooks/pre-commit`) before running
tests; fix all failures first.

- `uv run black src/ tests/` (formatter; hook uses `black --check`)
- `uv run ruff check src/ tests/ --fix` (linter/autofix)
- `uv run mypy src/ --ignore-missing-imports` (type check)
- Then run tests: `uv run pytest` / `npm run typecheck && npm test`
- Build the web UI bundle with `npm run build:js` (esbuild -> `static/dist/`).

Hooks are installed by `npm install` (or `npm run setup:hooks`); do not bypass
them with `--no-verify` unless explicitly asked.

## Layout

- `src/abstracts_explorer/`: mostly packages with thin facades for stable import paths:
  - `cli/`: `main.py` (dispatch), `parsers.py` (argparse wiring), `_common.py`, one module per command group.
  - `db/`: `DatabaseManager` mixins (`manager.py`); `database.py` is a re-export facade.
  - `embeddings/`: ChromaDB store, generation, queries, ingestion, artifacts.
  - `clustering/`: BERTopic + UMAP, hierarchy tree and derived levels, LLM labeling, results, cache/pipeline.
  - `plugin/`: plugin framework, models, registry, HTTP, persistence; `plugin.py` is no longer present.
  - `plugins/`: conference downloaders.
  - `mcp/`: MCP server, resources, cache, tools; `mcp_server.py` is a facade. `mcp_tools.py` is the tool registry/normalizer/formatter.
  - `rag/`: agent, tool wrappers, conversation, extraction.
  - `evaluation/`: prompts, formatting, generator, runner.
  - `interesting_papers/`: loading, summary, recommendations.
  - `export/`: markdown/ZIP export (`export_utils.py` is a facade).
  - `utils/`: shared time/json/url/conference/paper-presentation helpers.
  - Top-level: `config.py`, `db_models.py`, `paper_utils.py`, `registry.py`, `rate_limiter.py`, `metadata_cache.py`, `llm_utils.py`, `web_ui/` (Flask + `request_services.py` + TypeScript modules bundled with esbuild).
- `tools/profiling.py`: dev-only SQL profiler (not shipped in the wheel).
- `scripts/archive/`: applied one-off migrations kept for reference.
- `tests/`: pytest suite.
- `docs/`: Sphinx (Markdown).
- `src/abstracts_explorer/web_ui/static/`: TypeScript sources (`.ts`) bundled to `static/dist/` by esbuild.
- `src/abstracts_explorer/web_ui/types/`: ambient declarations (`globals.d.ts`).
- `src/abstracts_explorer/web_ui/tests/`: Jest tests (run against `.ts` sources).

## Testing

- One test file per source module, same base name (e.g. `database.py` -> `tests/test_database.py`). The CLI is argparse (not click); test patch targets must point at the owning submodule (e.g. `abstracts_explorer.cli.search_chat.EmbeddingsManager`).
- Separate files allowed by test type: `test_integration.py`, `test_web_integration.py`, `test_web_e2e.py`.
- Reuse shared fixtures/helpers in `tests/conftest.py` and `tests/helpers.py` rather than duplicating setup.
- Python: pytest + pytest-mock. Coverage target >90%.
- Frontend: Jest + jsdom via `@swc/jest` for TypeScript sources. Coverage target >80%.
- Mock external services (APIs, LLM backends, DBs). Tests must skip gracefully when optional services are unavailable.
- Commands:
  - `uv run pytest`
  - `uv run pytest --cov=src/abstracts_explorer`
  - `uv run pytest tests/test_database.py::test_add_paper`
  - `npm run typecheck` (tsc --noEmit)
  - `npm run build:js` (esbuild bundle)
  - `npm test` / `npm run test:coverage`

## Configuration

- Priority: CLI args > env vars > `.env` > built-in defaults.
- Key vars: `CHAT_MODEL`, `EMBEDDING_MODEL`, `LLM_BACKEND_URL`, `LLM_BACKEND_AUTH_TOKEN`, `PAPER_DB` (SQLite path or PostgreSQL URL), `EMBEDDING_DB`, `COLLECTION_NAME`, `MAX_CONTEXT_PAPERS`, `DATA_DIR`, `WEB_UI_THREADS`.
- Access via `get_config()` from `abstracts_explorer.config`.
- Never commit `.env` or secrets.

## Architecture Notes

- DB: SQLAlchemy 2.0+, SQLite default / PostgreSQL supported. SQLite >= 3.32 is required (32766 bind parameters; enforced at connect). Models: `Paper`, `EmbeddingsMetadata`, `ClusteringCache`.
- Clustering: the pipeline is fixed to BERTopic with Agglomerative clustering (ward linkage, distance threshold 0.5) on 5-D UMAP embeddings; a separate 2-D UMAP supplies visualization coordinates. Shared constants live in `clustering/constants.py` (`DEFAULT_CLUSTERING_*`). LLM labels are used for flat and merged hierarchy topics, with retry and keyword fallback. The DB caches the complete JSON payload; Plotly runs only in the browser.
- MCP: `MCPServer` from the `mcp` package; RAG chat auto-invokes MCP tools.
- Web UI: Flask 3.0+, 20+ API routes in `web_ui/app.py` (frontend calls from `web_ui/static/modules/`), TypeScript modules bundled by esbuild (basic Plotly build lazy-loaded on first chart), Waitress in production.

## Suggest commit message

**Whenever code is changed, suggest a commit message** after the changes are done. Use the format:

```text
commit message:

[type]: Short description of reason for change (max 50 chars)

[body with detailed explanation of what was changed and why, wrapped at 72 chars.]
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

## Common Tasks

- Feature: branch -> implement (types + NumPy docstrings) -> unit + integration tests -> update docs -> run tests -> PR.
- Bugfix: branch -> failing regression test -> fix -> verify -> PR.
- New plugin: implement in `plugins/`, inherit `DownloaderPlugin`/`HTTPDownloaderPlugin`, register in `plugins/__init__.py`, add `tests/test_plugins_<name>.py`, update `docs/plugins.md`.
- Build docs: `cd docs && uv run make html`.

## Git

- Branches: `main` (releases), `develop` (integration), feature branches `author/brief-description` from `develop`, `hotfix-*` from `main` (see `docs/branching_strategy.md`).
- Clear descriptive commit messages; only commit when explicitly asked, otherwise suggest a commit message.
- Do not commit `*.db`, `chroma_db/`, `_build/`, `.env`, `__pycache__/`. `uv.lock` and `package-lock.json` are tracked.

## Extra instructions

- Never pipe slow commands through `tail`; it masks the exit code and breaks error handling/timeouts. Run the command on its own (`<cmd>`), or redirect and inspect separately: `<cmd> >> <log_file>; tail -20 <log_file>`. Piping fast/verbose commands is fine.
- Never write to a database file in the source tree; use a temp path or `:memory:` for tests.

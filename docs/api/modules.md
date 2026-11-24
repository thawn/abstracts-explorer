# API Reference

Complete API documentation for Abstracts Explorer.

## Core Modules

- [Database Module](database.md) — paper storage and retrieval (SQLite/PostgreSQL)
- [Embeddings Module](embeddings.md) — vector embeddings and semantic search
- [Clustering Module](clustering.md) — clustering algorithms and visualization
- [RAG Module](rag.md) — Retrieval-Augmented Generation chat
- [Config Module](config.md) — configuration management

## Plugin & Registry

- [Plugin Module](plugin.md) — plugin framework for conference downloaders
- [Registry Module](registry.md) — OCI registry upload/download of data artifacts

## MCP Integration

- [MCP Server Module](mcp_server.md) — Model Context Protocol server for cluster analysis
- [MCP Tools Module](mcp_tools.md) — MCP tools integration for RAG chat

## Utilities

- [Export Utilities](export_utils.md) — paper export (ZIP, Markdown)
- [Paper Utilities](paper_utils.md) — paper formatting helpers
- [ORCID Module](orcid.md) — public ORCID record lookup and search-term suggestions
- [Database Models](db_models.md) — SQLAlchemy ORM models
- [Evaluation Module](evaluation.md) — automatic RAG evaluation

## Package Layout

Most implementation lives in focused sub-packages with thin top-level facades
for stable imports:

- `abstracts_explorer.db` — `DatabaseManager` mixins (`database.py` facade)
- `abstracts_explorer.embeddings` — ChromaDB store, generation, queries, artifacts
- `abstracts_explorer.clustering` — BERTopic + UMAP, hierarchy tree and derived levels, labeling, cache
- `abstracts_explorer.plugin` — plugin framework, models, registry, HTTP, persistence
- `abstracts_explorer.mcp` — MCP server/resources/cache/tools (`mcp_server.py` facade)
- `abstracts_explorer.rag` — agent, tool wrappers, conversation, extraction
- `abstracts_explorer.evaluation` — prompts, formatting, generation, running
- `abstracts_explorer.interesting_papers` — loading, summary, recommendations, author suggestions
- `abstracts_explorer.orcid` — public ORCID record lookup and search-term suggestions
- `abstracts_explorer.export` — Markdown/ZIP export (`export_utils.py` facade)
- `abstracts_explorer.cli` — `main.py` dispatch, `parsers.py` wiring, per-command modules
- `abstracts_explorer.utils` — shared time/json/url/conference/presentation helpers

The development-only SQL profiler lives in `tools/profiling.py` and is not
shipped in the wheel.

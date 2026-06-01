# Abstracts Explorer

> Hundreds of abstracts. Days to conference. Which talks matter to *you*?

Abstracts Explorer is an **agentic AI** tool that helps researchers navigate conference literature — download abstracts, search semantically, chat with an AI that cites its sources, and visualise the topic landscape.

Open source · Apache-2.0 · runs locally or on your institute's server.

## How it works

| Layer          | What it does                                                                               |
| -------------- | ------------------------------------------------------------------------------------------ |
| **Data**       | Downloads & stores abstracts from 8 conferences (NeurIPS · ICLR · ICML · CHI · HAICON · …) |
| **Embeddings** | Converts abstracts to vectors via an LLM embedding model (Blablador / LM Studio)           |
| **Retrieval**  | Semantic (cosine similarity) search                                                        |
| **Reasoning**  | RAG chat: the LLM reads relevant abstracts, then answers with citations                    |
| **Agency**     | 6 MCP tools the LLM calls autonomously to analyse topics, trends, and clusters             |
| **Interface**  | Flask web UI — four tabs, no coding required                                               |

## Web UI

| Tab                       | What you can do                                                              |
| ------------------------- | ---------------------------------------------------------------------------- |
| 🔍 **Search**             | Keyword and semantic search with filters (year, track, decision type)        |
| 💬 **Chat**               | Ask anything — query rewriting, abstract retrieval, cited answers, MCP tools |
| ⭐ **Interesting Papers** | Rate 1–5 stars, export as ZIP / JSON / Markdown                              |
| 📊 **Statistics**         | Topic trend chart and interactive cluster landscape                          |

![Search tab](docs/images/web-ui-search.png)

## MCP Tools

The LLM picks the right tool automatically from your plain-English question.

| Tool                        | Triggered when you ask…                                |
| --------------------------- | ------------------------------------------------------ |
| `get_conference_topics`     | "What are the main themes at NeurIPS 2025?"            |
| `analyze_topic_relevance`   | "How prominent is uncertainty quantification?"         |
| `get_topic_evolution`       | "How has diffusion model research trended since 2020?" |
| `search_papers`             | "Find papers about graph neural networks"              |
| `get_cluster_visualization` | "Show me the topic landscape"                          |
| `get_paper_details`         | "Who wrote that paper? Where is the poster?"           |

## Quick Start

### Option A — Podman Quadlets (production / server, systemd-native)

```bash
curl -fsSL https://raw.githubusercontent.com/thawn/abstracts-explorer/main/scripts/install-quadlets.sh | bash
```

### Option B — Local Python

```bash
uv sync --all-extras
uv run abstracts-explorer download --conference haicon --year 2026
uv run abstracts-explorer create-embeddings
uv run abstracts-explorer web-ui   # → http://localhost:5000
```

### Option C — Docker Compose

```bash
echo "LLM_BACKEND_AUTH_TOKEN=<your_blablador_token>" > .env
curl -o docker-compose.yml https://raw.githubusercontent.com/thawn/abstracts-explorer/refs/heads/main/docker-compose.yml
docker compose up -d   # → http://localhost:5000
```

## Documentation

<!-- user documentation -->
- 🔍 [Using the Web UI](https://thawn.github.io/abstracts-explorer/web_ui.html)
- 🐳 [Docker / Podman Guide](https://thawn.github.io/abstracts-explorer/docker.html)
- 📖 [Installation Guide](https://thawn.github.io/abstracts-explorer/installation.html)
- ⚙️ [Configuration](https://thawn.github.io/abstracts-explorer/configuration.html)
- 🔌 [Plugins Guide](https://thawn.github.io/abstracts-explorer/plugins.html)
- 🤖 [MCP Server](https://thawn.github.io/abstracts-explorer/mcp_server.html)
- 📚 [Full Docs](https://thawn.github.io/abstracts-explorer/index.html)

## Contributing

Contributions welcome — see [Contributing Guide](https://thawn.github.io/abstracts-explorer/contributing.html).  
Apache License 2.0 — see [LICENSE](LICENSE).

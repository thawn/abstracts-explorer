# 🐳 Docker Quick Start

Get Abstracts Explorer running in minutes with Docker or Podman!

## Prerequisites

- **Podman** (recommended): [Install Podman](https://podman.io/getting-started/installation)
- **Docker** (alternative): [Install Docker](https://docs.docker.com/get-docker/)

## Quick Start

### 1. Clone and Start

```bash
# Clone the repository
git clone https://github.com/thawn/abstracts-explorer.git
cd abstracts-explorer

# Start with Podman
podman-compose up -d

# OR start with Docker
docker-compose up -d
```

### 2. Download Conference Data

```bash
# Download NeurIPS 2025 papers
podman-compose exec abstracts-explorer \
  abstracts-explorer download --year 2025 --output /app/data/abstracts.db
```

### 3. Access the Web UI

Open your browser: **http://localhost:5000**

## Configuration

### Using LM Studio on Host

The default configuration connects to LM Studio running on your host machine at port 1234.

**Make sure LM Studio is running** with:
- Server started (Local Server tab)
- Chat model loaded (e.g., gemma-3-4b-it-qat)
- Embedding model loaded (e.g., text-embedding-qwen3-embedding-4b)

### Custom Configuration

Copy and edit the Docker environment file:

```bash
cp .env.docker .env
# Edit .env with your settings
nano .env
```

Then mount it in `docker-compose.yml`:

```yaml
volumes:
  - ./.env:/app/.env:ro
```

## Common Commands

### View Logs

```bash
podman-compose logs -f abstracts-explorer
```

### Generate Embeddings

```bash
podman-compose exec abstracts-explorer \
  abstracts-explorer create-embeddings --db-path /app/data/abstracts.db
```

### Search Papers

```bash
podman-compose exec abstracts-explorer \
  abstracts-explorer search "graph neural networks"
```

### Open Shell

```bash
podman-compose exec -it abstracts-explorer /bin/bash
```

### Stop Services

```bash
podman-compose down
```

## Troubleshooting

### Port Already in Use

Change the port mapping in `docker-compose.yml`:

```yaml
ports:
  - "8080:5000"  # Use port 8080 instead
```

### Can't Connect to LM Studio

- Verify LM Studio server is running on host
- Check the LLM_BACKEND_URL in docker-compose.yml
- For Linux, try using `network_mode: host`

### Database Errors

Make sure you've downloaded papers first:

```bash
podman-compose exec abstracts-explorer \
  abstracts-explorer download --year 2025
```

## Full Documentation

📖 **[Complete Docker/Podman Guide](docs/docker.md)**

## Services

The compose file includes:

- **abstracts-explorer** (port 5000) - Main application
- **chromadb** (port 8000) - Optional standalone vector database
- **postgres** (port 5432) - Optional PostgreSQL backend

By default, only the main application starts.

## Data Persistence

All data is stored in Docker volumes:

- `abstracts-data` - Paper database
- `abstracts-chroma` - Embeddings

## Support

- 🐛 [Report Issues](https://github.com/thawn/abstracts-explorer/issues)
- 📖 [Full Documentation](README.md)
- 💬 [Discussions](https://github.com/thawn/abstracts-explorer/discussions)

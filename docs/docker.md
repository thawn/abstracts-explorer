# Docker and Podman Setup Guide

This guide explains how to run Abstracts Explorer using containers with Podman (recommended) or Docker.

**Note:** The container images are production-optimized and use pre-built static vendor files (CSS/JS libraries). Node.js is **not required** for production containers - it's only needed for local development if you want to rebuild vendor files.

## Pre-built Images

Pre-built container images are available from:
- **GitHub Container Registry**: `ghcr.io/thawn/abstracts-explorer:latest`
- **Docker Hub** (releases only): `thawn/abstracts-explorer:latest`

```bash
# Pull and run from GitHub Container Registry
podman pull ghcr.io/thawn/abstracts-explorer:latest
podman run -p 5000:5000 ghcr.io/thawn/abstracts-explorer:latest

# Or from Docker Hub (for releases)
docker pull thawn/abstracts-explorer:latest
docker run -p 5000:5000 thawn/abstracts-explorer:latest
```

Available tags:
- `latest` - Latest stable release or main branch
- `main` - Latest main branch build
- `develop` - Latest develop branch build
- `v*.*.*` - Specific version releases (e.g., `v0.1.0`)

## Prerequisites

### Podman (Recommended)

Podman is a daemonless container engine that's more secure and doesn't require root privileges.

**Linux:**
```bash
# Debian/Ubuntu
sudo apt-get install podman podman-compose

# Fedora/RHEL
sudo dnf install podman podman-compose
```

**macOS:**
```bash
brew install podman podman-compose
podman machine init
podman machine start
```

**Windows:** Download from [Podman Desktop](https://podman-desktop.io/)

### Docker (Alternative)

**Linux:** `curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh`

**macOS/Windows:** Install [Docker Desktop](https://www.docker.com/products/docker-desktop)

## Quick Start

### 1. Start Services

```bash
# Podman
podman-compose up -d

# Docker
docker compose up -d
```

### 2. Download Data

```bash
# Podman
podman-compose exec abstracts-explorer \
  abstracts-explorer download --year 2025 --output /app/data/abstracts.db

# Docker: Replace 'podman-compose' with 'docker compose'
```

### 3. Generate Embeddings (Optional)

Requires an LLM backend (like LM Studio) running and accessible.

```bash
podman-compose exec abstracts-explorer \
  abstracts-explorer create-embeddings --db-path /app/data/abstracts.db
```

### 4. Access the Web UI

Open http://localhost:5000 in your browser.

## Configuration

### Environment Variables

Configure via `docker-compose.yml` or mount a custom `.env` file.

**Option 1: Edit docker-compose.yml**

```yaml
services:
  abstracts-explorer:
    environment:
      - LLM_BACKEND_URL=http://host.docker.internal:1234
      - CHAT_MODEL=your-chat-model
```

**Option 2: Mount .env File**

```bash
cp .env.docker .env
# Edit .env with your settings
```

Uncomment in `docker-compose.yml`:
```yaml
volumes:
  - ./.env:/app/.env:ro
```

### Key Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_BACKEND_URL` | LLM backend URL | `http://host.docker.internal:1234` |
| `CHAT_MODEL` | Chat model name | `gemma-3-4b-it-qat` |
| `EMBEDDING_MODEL` | Embedding model name | `text-embedding-qwen3-embedding-4b` |
| `PAPER_DB_PATH` | SQLite database path | `/app/data/abstracts.db` |
| `EMBEDDING_DB_PATH` | Embeddings path | `/app/chroma_db` |

### Connecting to Host LM Studio

**Podman on Linux:**
```yaml
environment:
  - LLM_BACKEND_URL=http://host.containers.internal:1234
```

**Docker (Mac/Windows) or Podman (Mac):**
```yaml
environment:
  - LLM_BACKEND_URL=http://host.docker.internal:1234
```

**Alternative (Linux): Use host network**
```yaml
services:
  abstracts-explorer:
    network_mode: host
    environment:
      - LLM_BACKEND_URL=http://localhost:1234
```

## Services

### Main Application (abstracts-explorer)
- **Port:** 5000
- **Volumes:** `abstracts-data`, `abstracts-chroma`
- **Purpose:** Web UI and CLI tools

### ChromaDB (Optional)
- **Port:** 8000
- **Purpose:** Standalone vector database (app uses embedded by default)
- **Enable:** Uncomment dependency and set `EMBEDDING_DB_PATH=http://chromadb:8000`

### PostgreSQL (Optional)
- **Port:** 5432
- **Purpose:** Alternative to SQLite
- **Enable:** Uncomment dependency and set `DATABASE_URL=postgresql://abstracts:abstracts_password@postgres/abstracts`

## Common Commands

### View Logs
```bash
podman-compose logs -f abstracts-explorer
```

### Execute CLI Commands
```bash
podman-compose exec abstracts-explorer abstracts-explorer search "neural networks"
```

### Interactive Shell
```bash
podman-compose exec -it abstracts-explorer /bin/bash
```

### Stop Services
```bash
podman-compose down

# Remove volumes (deletes data)
podman-compose down -v
```

## Data Persistence

All data is stored in named volumes:
- `abstracts-data` - Paper database
- `abstracts-chroma` - Embeddings
- `chromadb-data` - ChromaDB (if enabled)
- `postgres-data` - PostgreSQL (if enabled)

### Backup
```bash
podman-compose exec abstracts-explorer \
  tar czf /tmp/backup.tar.gz /app/data /app/chroma_db

podman cp abstracts-explorer:/tmp/backup.tar.gz ./backup.tar.gz
```

### Restore
```bash
podman cp ./backup.tar.gz abstracts-explorer:/tmp/
podman-compose exec abstracts-explorer tar xzf /tmp/backup.tar.gz -C /
```

## Troubleshooting

### Container Won't Start
- Check logs: `podman-compose logs abstracts-explorer`
- Verify port 5000 is available: `lsof -i :5000`
- Rebuild: `podman-compose build --no-cache && podman-compose up -d`

### Cannot Connect to LM Studio
- Ensure LM Studio server is running with models loaded
- Verify URL: `podman-compose exec abstracts-explorer curl -v http://host.docker.internal:1234/v1/models`
- For Linux, try `host.containers.internal` or `network_mode: host`

### Permission Errors (Podman)
```bash
podman unshare chown 1000:1000 /path/to/volume
```

### Database Locked
- Use PostgreSQL for multi-user scenarios
- Ensure only one process accesses the database

### Out of Memory
Increase container memory limits:
```yaml
services:
  abstracts-explorer:
    deploy:
      resources:
        limits:
          memory: 4G
```

## Production Deployment

1. **Change default passwords** in `docker-compose.yml`
2. **Use external secrets** for tokens
3. **Enable HTTPS** with a reverse proxy (nginx, traefik)
4. **Set resource limits** for memory and CPU
5. **Configure monitoring** and health checks
6. **Use specific image tags** instead of `latest`

### Example Production Settings
```yaml
services:
  abstracts-explorer:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
    restart: unless-stopped
```

## Further Reading

- [Podman Documentation](https://docs.podman.io/)
- [Docker Documentation](https://docs.docker.com/)
- [Main README](../README.md)
- [Configuration Guide](configuration.md)

## Support

- 🐛 [Report issues](https://github.com/thawn/abstracts-explorer/issues)
- 💬 [Discussions](https://github.com/thawn/abstracts-explorer/discussions)

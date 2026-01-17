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

Available tags (following container best practices):
- `latest` - Latest stable release only (never points to branch builds)
- `main` - Latest main branch build
- `develop` - Latest develop branch build
- `v*.*.*` - Specific version releases (e.g., `v0.1.0`)
- `v*.*` - Major.minor version (e.g., `v0.1`)
- `v*` - Major version (e.g., `v0`)
- `sha-*` - Specific commit SHA for traceability (e.g., `sha-5f8567d`)
- `pr-*` - Pull request builds for testing (e.g., `pr-40`)

## Testing Pull Requests

To test changes from a pull request before they're merged:

1. **Find the PR number** (e.g., PR #40)
2. **Update docker-compose.yml** to use the PR image:

```yaml
services:
  abstracts-explorer:
    image: ghcr.io/thawn/abstracts-explorer:pr-40  # Replace 40 with your PR number
```

3. **Pull and start services**:

```bash
docker compose pull
docker compose up -d
```

4. **Verify the setup**:

```bash
# Check service health
docker compose ps

# View logs
docker compose logs abstracts-explorer

# Access web UI
curl http://localhost:5000/health
```

**Note:** PR images are automatically built and pushed when commits are made to pull requests. They're tagged with `pr-<number>` for easy testing.

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
| `DATABASE_URL` | PostgreSQL connection URL | `postgresql://abstracts:abstracts_password@postgres:5432/abstracts` |
| `EMBEDDING_DB_URL` | ChromaDB HTTP endpoint | `http://chromadb:8000` |
| `LLM_BACKEND_URL` | LLM backend URL | `http://host.docker.internal:1234` |
| `CHAT_MODEL` | Chat model name | `gemma-3-4b-it-qat` |
| `EMBEDDING_MODEL` | Embedding model name | `text-embedding-qwen3-embedding-4b` |
| `COLLECTION_NAME` | ChromaDB collection | `papers` |

**Note:** The setup now uses PostgreSQL and ChromaDB by default. SQLite mode is still supported for local development by setting `PAPER_DB_PATH` instead of `DATABASE_URL`.

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

The Docker Compose setup includes three services that work together:

### Main Application (abstracts-explorer)
- **Port:** 5000 (exposed to host)
- **Volumes:** `abstracts-data`
- **Purpose:** Web UI and CLI tools
- **Image:** `ghcr.io/thawn/abstracts-explorer:latest`

### ChromaDB
- **Port:** 8000 (internal only, not exposed to host)
- **Purpose:** Vector database for semantic search embeddings
- **Health Check:** TCP check on port 8000
- **Data:** Persisted in `chromadb-data` volume

### PostgreSQL
- **Port:** 5432 (internal only, not exposed to host)
- **Purpose:** Relational database for paper metadata
- **Health Check:** `pg_isready` command
- **Data:** Persisted in `postgres-data` volume
- **Credentials:** Set in `docker-compose.yml` (change for production!)

**Security Note:** Database ports (5432, 8000) are **not exposed** to the host system. Only the web UI port (5000) is accessible from outside the container network. All inter-service communication happens via Docker's internal network.

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
- `abstracts-data` - Application data directory
- `chromadb-data` - ChromaDB vector embeddings
- `postgres-data` - PostgreSQL database

### Backup
```bash
# Backup PostgreSQL database
podman-compose exec postgres pg_dump -U abstracts abstracts > backup.sql

# Backup ChromaDB data
podman-compose exec abstracts-explorer \
  tar czf /tmp/chroma-backup.tar.gz /app/chroma_db

podman cp abstracts-chromadb:/chroma/chroma ./chroma-backup
```

### Restore
```bash
# Restore PostgreSQL database
cat backup.sql | podman-compose exec -T postgres psql -U abstracts

# Restore ChromaDB data
podman cp ./chroma-backup abstracts-chromadb:/chroma/chroma
podman-compose restart chromadb
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
- PostgreSQL is now the default for Docker Compose (no locking issues)
- For SQLite mode, ensure only one process accesses the database

### ChromaDB Health Check Fails
- The health check uses TCP port checking (bash built-in)
- If failing, check logs: `podman-compose logs chromadb`
- Verify ChromaDB container started: `podman-compose ps`

### Cannot Access Databases from Host
- Database ports (5432, 8000) are **intentionally not exposed** for security
- Access via application container: `podman-compose exec abstracts-explorer psql`
- For debugging, temporarily add port mappings to `docker-compose.yml`

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
6. **Use specific image tags** instead of `latest` (e.g., `v1.0.0` or `sha-5f8567d` for precise version control)

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

# Docker and Podman Setup Guide

This guide explains how to run Abstracts Explorer using containers with Podman (recommended) or Docker.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start with Podman](#quick-start-with-podman)
- [Quick Start with Docker](#quick-start-with-docker)
- [Configuration](#configuration)
- [Services](#services)
- [Usage Examples](#usage-examples)
- [Data Persistence](#data-persistence)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### For Podman (Recommended)

Podman is a daemonless container engine that's more secure and doesn't require root privileges.

**Linux:**
```bash
# Debian/Ubuntu
sudo apt-get install podman podman-compose

# Fedora/RHEL
sudo dnf install podman podman-compose

# Arch Linux
sudo pacman -S podman podman-compose
```

**macOS:**
```bash
brew install podman podman-compose
podman machine init
podman machine start
```

**Windows:**
Download and install from [Podman Desktop](https://podman-desktop.io/)

### For Docker (Alternative)

**Linux:**
```bash
# Install Docker Engine
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

**macOS/Windows:**
Install [Docker Desktop](https://www.docker.com/products/docker-desktop)

## Quick Start with Podman

### 1. Build and Start Services

```bash
# Build the image and start all services
podman-compose up -d

# Or build explicitly first
podman-compose build
podman-compose up -d
```

### 2. Access the Application

Open your browser and navigate to: http://localhost:5000

### 3. Download Conference Data

```bash
# Download NeurIPS 2025 papers
podman-compose exec abstracts-explorer \
  abstracts-explorer download --year 2025 --output /app/data/abstracts.db

# Or use a specific plugin
podman-compose exec abstracts-explorer \
  abstracts-explorer download --plugin neurips --year 2025
```

### 4. Generate Embeddings (Optional)

**Note:** This requires an LLM backend (like LM Studio) running and accessible.

```bash
podman-compose exec abstracts-explorer \
  abstracts-explorer create-embeddings --db-path /app/data/abstracts.db
```

### 5. Stop Services

```bash
podman-compose down

# Stop and remove volumes (WARNING: deletes all data)
podman-compose down -v
```

## Quick Start with Docker

The commands are nearly identical to Podman, just replace `podman-compose` with `docker-compose`:

```bash
# Build and start
docker-compose up -d

# Download data
docker-compose exec abstracts-explorer \
  abstracts-explorer download --year 2025 --output /app/data/abstracts.db

# Stop
docker-compose down
```

## Configuration

### Environment Variables

You can configure the application using environment variables in `docker-compose.yml` or by mounting a custom `.env` file.

#### Option 1: Edit docker-compose.yml

Edit the `environment` section in `docker-compose.yml`:

```yaml
services:
  abstracts-explorer:
    environment:
      - LLM_BACKEND_URL=http://host.docker.internal:1234
      - CHAT_MODEL=your-chat-model
      - EMBEDDING_MODEL=your-embedding-model
```

#### Option 2: Mount .env File

Create a `.env` file and mount it:

```bash
# Create .env from example
cp .env.example .env
# Edit .env with your settings
nano .env
```

Uncomment this line in `docker-compose.yml`:
```yaml
volumes:
  - ./.env:/app/.env:ro
```

### Key Configuration Options

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_BACKEND_URL` | URL of your LLM backend | `http://host.docker.internal:1234` |
| `CHAT_MODEL` | Chat model name | `gemma-3-4b-it-qat` |
| `EMBEDDING_MODEL` | Embedding model name | `text-embedding-qwen3-embedding-4b` |
| `PAPER_DB_PATH` | Path to SQLite database | `/app/data/abstracts.db` |
| `EMBEDDING_DB_PATH` | Path to embeddings | `/app/chroma_db` |

### Connecting to Host LM Studio

By default, the container is configured to connect to LM Studio running on your host machine.

**Podman:**
```yaml
environment:
  - LLM_BACKEND_URL=http://host.containers.internal:1234
```

**Docker:**
```yaml
environment:
  - LLM_BACKEND_URL=http://host.docker.internal:1234
```

**Alternative: Host Network Mode**

For Linux, you can use host network mode:

```yaml
services:
  abstracts-explorer:
    network_mode: host
    environment:
      - LLM_BACKEND_URL=http://localhost:1234
```

## Services

The `docker-compose.yml` defines three services:

### 1. abstracts-explorer (Main Application)

- **Port:** 5000
- **Purpose:** Web UI and CLI tools
- **Volumes:** 
  - `abstracts-data` - Paper database
  - `abstracts-chroma` - Embeddings database

### 2. chromadb (Optional Vector Database)

- **Port:** 8000
- **Purpose:** Standalone ChromaDB server (alternative to embedded)
- **Volumes:** `chromadb-data`

**Note:** By default, the app uses an embedded ChromaDB. Use this service if you want a separate ChromaDB server.

To use the standalone ChromaDB service, update the configuration:
```yaml
environment:
  - EMBEDDING_DB_PATH=http://chromadb:8000
```

### 3. postgres (Optional Database Backend)

- **Port:** 5432
- **Purpose:** PostgreSQL database (alternative to SQLite)
- **Volumes:** `postgres-data`

To use PostgreSQL instead of SQLite:

1. Uncomment the `postgres` dependency in `docker-compose.yml`
2. Update the configuration:
```yaml
environment:
  - DATABASE_URL=postgresql://abstracts:abstracts_password@postgres/abstracts
```

## Usage Examples

### Running CLI Commands

Execute any abstracts-explorer command inside the container:

```bash
# List available plugins
podman-compose exec abstracts-explorer abstracts-explorer download --list-plugins

# Search papers (after downloading and creating embeddings)
podman-compose exec abstracts-explorer \
  abstracts-explorer search "graph neural networks"

# Start interactive RAG chat
podman-compose exec -it abstracts-explorer \
  abstracts-explorer chat

# Generate clusters
podman-compose exec abstracts-explorer \
  abstracts-explorer cluster-embeddings --n-clusters 8
```

### Accessing Logs

```bash
# View all logs
podman-compose logs

# Follow logs for specific service
podman-compose logs -f abstracts-explorer

# View last 100 lines
podman-compose logs --tail=100 abstracts-explorer
```

### Interactive Shell

```bash
# Open bash shell in container
podman-compose exec -it abstracts-explorer /bin/bash

# Run Python interactively
podman-compose exec -it abstracts-explorer python
```

### Building Custom Image

```bash
# Build with custom tag
podman build -t abstracts-explorer:custom .

# Build with specific target stage
podman build --target python-builder -t abstracts-builder .

# Build without cache
podman build --no-cache -t abstracts-explorer:latest .
```

## Data Persistence

### Named Volumes

Docker Compose creates named volumes for persistent data:

- `abstracts-data` - Paper database and downloads
- `abstracts-chroma` - Embeddings and vector database
- `chromadb-data` - Standalone ChromaDB data
- `postgres-data` - PostgreSQL database

### Inspecting Volumes

**Podman:**
```bash
# List volumes
podman volume ls

# Inspect volume
podman volume inspect abstracts-explorer_abstracts-data

# Remove unused volumes
podman volume prune
```

**Docker:**
```bash
# List volumes
docker volume ls

# Inspect volume
docker volume inspect abstracts-explorer_abstracts-data

# Remove unused volumes
docker volume prune
```

### Backup Data

**Podman:**
```bash
# Backup database
podman-compose exec abstracts-explorer \
  tar czf /tmp/backup.tar.gz /app/data /app/chroma_db

podman cp abstracts-explorer:/tmp/backup.tar.gz ./backup.tar.gz
```

**Docker:**
```bash
# Backup database
docker-compose exec abstracts-explorer \
  tar czf /tmp/backup.tar.gz /app/data /app/chroma_db

docker cp abstracts-explorer:/tmp/backup.tar.gz ./backup.tar.gz
```

### Restore Data

```bash
# Copy backup to container
podman cp ./backup.tar.gz abstracts-explorer:/tmp/

# Extract in container
podman-compose exec abstracts-explorer \
  tar xzf /tmp/backup.tar.gz -C /
```

## Troubleshooting

### Container Won't Start

1. Check logs:
```bash
podman-compose logs abstracts-explorer
```

2. Verify port availability:
```bash
# Check if port 5000 is in use
sudo netstat -tlnp | grep 5000
# Or with lsof
lsof -i :5000
```

3. Rebuild the image:
```bash
podman-compose build --no-cache
podman-compose up -d
```

### Cannot Connect to LM Studio

1. **Check LM Studio is running:**
   - Open LM Studio
   - Go to "Local Server" tab
   - Ensure server is started

2. **Verify the URL:**
   - For Podman on Linux: `http://host.containers.internal:1234`
   - For Docker on Mac/Windows: `http://host.docker.internal:1234`
   - For host network mode: `http://localhost:1234`

3. **Test connection from container:**
```bash
podman-compose exec abstracts-explorer \
  curl -v http://host.docker.internal:1234/v1/models
```

### Permission Errors

If you get permission errors with Podman:

```bash
# Run with proper user mapping
podman-compose run --user $(id -u):$(id -g) abstracts-explorer bash

# Or fix volume permissions
podman unshare chown 1000:1000 /path/to/volume
```

### Database Locked Errors

SQLite database locked errors can occur with multiple connections:

1. Use PostgreSQL for multi-user scenarios
2. Ensure only one process accesses the database
3. Check for stale lock files

### Out of Memory

If the container runs out of memory during embedding generation:

1. **Increase container memory limits:**

```yaml
services:
  abstracts-explorer:
    deploy:
      resources:
        limits:
          memory: 4G
```

2. **Process papers in batches:**
```bash
# Use smaller batch size
podman-compose exec abstracts-explorer \
  abstracts-explorer create-embeddings --batch-size 50
```

### Network Issues

1. **Verify network exists:**
```bash
podman network ls
# Or
docker network ls
```

2. **Check container networking:**
```bash
podman-compose exec abstracts-explorer ping chromadb
```

3. **Reset network:**
```bash
podman-compose down
podman network prune
podman-compose up -d
```

## Advanced Usage

### Using PostgreSQL Backend

1. Uncomment the `postgres` service dependency in `docker-compose.yml`

2. Update environment variables:
```yaml
environment:
  - DATABASE_URL=postgresql://abstracts:abstracts_password@postgres/abstracts
```

3. Start services:
```bash
podman-compose up -d
```

### Using External ChromaDB

To use a standalone ChromaDB service:

1. Configure the embeddings path:
```yaml
environment:
  - EMBEDDING_DB_PATH=http://chromadb:8000
```

2. Ensure ChromaDB service is running:
```bash
podman-compose up -d chromadb
```

### Production Deployment

For production use:

1. **Change default passwords:**
```yaml
environment:
  - POSTGRES_PASSWORD=your_secure_password_here
```

2. **Use external secrets:**
```yaml
secrets:
  llm_token:
    external: true
environment:
  - LLM_BACKEND_AUTH_TOKEN_FILE=/run/secrets/llm_token
```

3. **Enable HTTPS:**
   - Use a reverse proxy (nginx, traefik)
   - Configure SSL certificates
   - Update port mappings

4. **Resource limits:**
```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
    reservations:
      cpus: '1'
      memory: 2G
```

### Custom Build Arguments

Build with custom Python or Node.js versions:

```dockerfile
# In Dockerfile, add build args
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim
```

```bash
# Build with custom version
podman build --build-arg PYTHON_VERSION=3.11 -t abstracts-explorer .
```

## Further Reading

- [Podman Documentation](https://docs.podman.io/)
- [Docker Documentation](https://docs.docker.com/)
- [Abstracts Explorer Main Documentation](README.md)
- [Configuration Guide](docs/configuration.md)

## Support

For issues or questions:
- 🐛 [Report issues](https://github.com/thawn/abstracts-explorer/issues)
- 💬 [Discussions](https://github.com/thawn/abstracts-explorer/discussions)

# Docker and Podman Setup Guide

This guide explains how to run Abstracts Explorer using containers with Podman (recommended) or Docker.

**Note:** The container images are production-optimized and use pre-built static vendor files (CSS/JS libraries). Node.js is **not required** for production containers - it's only needed for local development if you want to rebuild vendor files.

## Available Images

Pre-built container images are available from:
- **GitHub Container Registry**: `ghcr.io/thawn/abstracts-explorer:latest`
- **Docker Hub** (releases only): `thawn/abstracts-explorer:latest`

Available tags (following container best practices):
- `latest` - Latest stable release only (never points to branch builds)
- `main` - Latest main branch build
- `develop` - Latest develop branch build
- `v*.*.*` - Specific version releases (e.g., `v0.1.0`)
- `v*.*` - Major.minor version (e.g., `v0.1`)
- `v*` - Major version (e.g., `v0`)
- `sha-*` - Specific commit SHA for traceability (e.g., `sha-5f8567d`)
- `pr-*` - Pull request builds for testing (e.g., `pr-40`)

## Quick Start

### 1. Create .env File

First create a `.env` file with your [blablador token](https://sdlaml.pages.jsc.fz-juelich.de/ai/guides/blablador_api_access/):

```bash
LLM_BACKEND_AUTH_TOKEN=your_blablador_token_here
```

### 2. Download the compose file

```bash
curl -L https://github.com/thawn/abstracts-explorer/raw/main/docker-compose.yml -o docker-compose.yml
```

> **HTTPS certificates required** — the default `docker-compose.yml` uses nginx with
> your own certificate files.  See [HTTPS / SSL Setup](#https--ssl-setup) below for
> how to place your certificate (Option 1) or use Let's Encrypt (Option 2).

### 3. Start Services

```bash
# Podman
podman-compose up -d

# Docker
docker compose up -d
```

### 4. Download Data

```bash
# Podman
podman-compose exec abstracts-explorer \
  abstracts-explorer download --year 2025 --plugin neurips

# Docker: Replace 'podman-compose' with 'docker compose'
```

### 5. Generate Embeddings (Optional)

```bash
podman-compose exec abstracts-explorer \
  abstracts-explorer create-embeddings
```

### 6. Access the Web UI

Open https://localhost in your browser (HTTP on port 80 is automatically redirected to HTTPS).

## HTTPS / SSL Setup

Both Docker Compose files include an **nginx reverse proxy** that handles SSL termination.
Waitress (the application server) continues to serve plain HTTP on port 5000 inside the
container network while nginx exposes the service securely on port 443.

Choose the approach that matches your situation:

| Approach | Compose file | When to use |
|---|---|---|
| **Existing certificate** | `docker-compose.yml` | You already have a valid certificate (e.g. from your institution or a wildcard cert) |
| **Let's Encrypt** | `docker-compose.letsencrypt.yml` | You need a free, automatically renewed certificate for a public domain |

---

### Option 1: Existing Certificate

> Compose file: `docker-compose.yml`

Use this when you already have a valid SSL certificate (e.g. issued by your institution,
a wildcard cert, or any other CA).

#### Certificate files

Before starting the services place your SSL certificate and private key in a
`certs/` directory next to `docker-compose.yml`:

```
certs/
├── cert.pem   ← your certificate (or full chain)
└── key.pem    ← your private key
```

The files are mounted into the nginx container as read-only at `/etc/nginx/certs/`.

> **Note:** The nginx configuration (`nginx/nginx.conf`) references these paths.  If
> your certificate files have different names, update the `ssl_certificate` and
> `ssl_certificate_key` directives in `nginx/nginx.conf` accordingly.

#### Changing the server name

By default nginx uses `server_name _;` (match any hostname).  To restrict it to a
specific domain, edit `nginx/nginx.conf` and replace `_` with your domain:

```nginx
server_name abstracts.example.com;
```

#### Start the stack

```bash
docker compose up -d
```

---

### Option 2: Let's Encrypt Certificate

> Compose file: `docker-compose.letsencrypt.yml`

Use this when you need a free, automatically renewed certificate from
[Let's Encrypt](https://letsencrypt.org/).

> **Requirements**
> - A **public domain** that points to this server (Let's Encrypt cannot issue
>   certificates for `localhost` or private IP addresses).
> - Ports **80** and **443** must be reachable from the internet so Let's Encrypt can
>   verify domain ownership.

#### Step 1 — Replace the placeholder domain

Edit **both** `docker-compose.letsencrypt.yml` **and** `nginx/nginx.letsencrypt.conf`,
replacing every occurrence of `abstracts.example.com` with your real domain.

#### Step 2 — Obtain the initial certificate

Run Certbot once in standalone mode *before* starting the stack (nginx must not be
running yet so that Certbot can bind to port 80):

```bash
# Docker
docker run --rm \
  -p 80:80 \
  -v letsencrypt-certs:/etc/letsencrypt \
  certbot/certbot certonly --standalone \
  --domain abstracts.example.com \
  --email your@email.com \
  --agree-tos --non-interactive

# Podman
podman run --rm \
  -p 80:80 \
  -v letsencrypt-certs:/etc/letsencrypt \
  certbot/certbot certonly --standalone \
  --domain abstracts.example.com \
  --email your@email.com \
  --agree-tos --non-interactive
```

#### Step 3 — Start the stack

```bash
docker compose -f docker-compose.letsencrypt.yml up -d
```

#### Automatic renewal

The `certbot` service in `docker-compose.letsencrypt.yml` checks for renewal every
12 hours.  Let's Encrypt certificates expire after 90 days; renewal is attempted
automatically when fewer than 30 days remain.

After a successful renewal, nginx must be **reloaded** to activate the new certificate
(certbot and nginx run in separate containers, so this cannot happen automatically).
Run this command once after renewal:

```bash
docker compose -f docker-compose.letsencrypt.yml exec nginx nginx -s reload
```

To automate the reload, add a daily host cron job (replace `docker` with `podman`
if using Podman):

```bash
# Add via: crontab -e
0 3 * * * docker exec abstracts-nginx nginx -s reload
```

To force an immediate renewal:

```bash
docker compose -f docker-compose.letsencrypt.yml exec certbot \
  certbot renew --webroot -w /var/www/certbot --force-renewal
```

---

### HTTP → HTTPS redirect

In both setups port 80 is redirected to HTTPS and port 5000 is **not** exposed to the
host; all traffic must go through nginx on port 443.

### Security hardening

Both nginx configurations include the following security hardening out of the box:

| Setting | Value / Behaviour |
|---|---|
| **TLS protocol** | TLS 1.3 only (TLS 1.2 disabled; see comments in config to re-enable for legacy clients) |
| **TLS 1.2 ciphers** (if re-enabled) | ECDHE + AES-GCM + ChaCha20-Poly1305 only; weak/export ciphers excluded |
| **SSL session tickets** | Disabled (`ssl_session_tickets off`) to preserve forward secrecy |
| **SSL session cache** | Shared 10 MB cache, 1 day timeout |
| **OCSP stapling** | Enabled — reduces handshake latency and supports revocation checking |
| **Server version** | Hidden (`server_tokens off`) |
| **HSTS** | `max-age=63072000; includeSubDomains` (2 years) |
| **X-Content-Type-Options** | `nosniff` |
| **X-Frame-Options** | `SAMEORIGIN` |
| **Referrer-Policy** | `strict-origin-when-cross-origin` |
| **X-XSS-Protection** | `1; mode=block` |
| **X-Powered-By** | Stripped from upstream responses |

> **OCSP stapling note (Option 1 — existing cert):** OCSP stapling requires a certificate
> issued by a public CA and the full certificate chain in `cert.pem`.  If you are using a
> self-signed certificate, remove the `ssl_stapling`, `ssl_stapling_verify`,
> `ssl_trusted_certificate`, `resolver`, and `resolver_timeout` lines from
> `nginx/nginx.conf`.

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
# - If you have a CA-signed certificate omit the -k flag
# - -k skips certificate verification (only use for self-signed/test certificates)
curl -k https://localhost/health
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
| `EMBEDDING_DB` | ChromaDB location (URL or path) | `http://chromadb:8000` |
| `LLM_BACKEND_URL` | LLM backend URL | `http://host.docker.internal:1234` |
| `CHAT_MODEL` | Chat model name | `gemma-3-4b-it-qat` |
| `EMBEDDING_MODEL` | Embedding model name | `text-embedding-qwen3-embedding-4b` |
| `COLLECTION_NAME` | ChromaDB collection | `papers` |
| `GITHUB_TOKEN` | PAT for registry upload/download | *(empty)* |
| `REGISTRY_REPOSITORY` | Default OCI repository for registry commands | *(empty)* |

**Note:** The setup uses PostgreSQL and ChromaDB by default. For local development with SQLite, set `PAPER_DB=abstracts.db` instead of the PostgreSQL URL. For local ChromaDB, set `EMBEDDING_DB=chroma_db` instead of the HTTP URL.

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

The Docker Compose setup includes four services that work together:

### Nginx Reverse Proxy (nginx)
- **Ports:** 80 (HTTP → HTTPS redirect), 443 (HTTPS)
- **Purpose:** SSL termination and reverse proxy to the application
- **Config:** `./nginx/nginx.conf` (mounted read-only)
- **Certs:** `./certs/` directory (mounted read-only)

### Main Application (abstracts-explorer)
- **Port:** 5000 (internal only, not exposed to host — access via nginx)
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

**Security Note:** Database ports (5432, 8000) and the application port (5000) are **not exposed** to the host system. Only the nginx ports (80, 443) are accessible from outside the container network. All inter-service communication happens via Docker's internal network.

## Sharing Data via Registry

The `registry` commands let you exchange pre-built paper databases and embeddings between instances without re-downloading or re-computing them. See the [Registry documentation](registry.md) for the full feature description.

### Setting Up Registry Credentials

Add your GitHub Personal Access Token to your `.env` file:

```bash
# .env
# Required for registry upload/download
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx

# Optional: set a default repository so you don't have to pass -r every time
REGISTRY_REPOSITORY=ghcr.io/thawn/abstracts-data
```

The `docker-compose.yml` already passes both variables into the container:

```yaml
environment:
  - GITHUB_TOKEN=${GITHUB_TOKEN:-}
  - REGISTRY_REPOSITORY=${REGISTRY_REPOSITORY:-}
```

The `${VAR:-}` syntax means the variable is optional — if it is not set in your `.env` file, the container simply starts without it and you can still pass `--token` and `-r` on the command line.

### Download Data from the Registry

To pre-populate a fresh container with data from the registry:

```bash
# Download NeurIPS 2025 data into the running container
podman-compose exec abstracts-explorer \
  abstracts-explorer registry download \
    -r ghcr.io/thawn/abstracts-data \
    --conference neurips --year 2025

# Or download all available data (REGISTRY_REPOSITORY must be set in .env)
podman-compose exec abstracts-explorer \
  abstracts-explorer registry download --conference all --yes
```

### Upload Data to the Registry

After generating embeddings and clustering cache, push the data so other instances can use it:

```bash
# Upload NeurIPS 2025 (GITHUB_TOKEN must be set in .env)
podman-compose exec abstracts-explorer \
  abstracts-explorer registry upload \
    --conference neurips --year 2025

# Upload all conferences at once
podman-compose exec abstracts-explorer \
  abstracts-explorer registry upload --conference all --yes
```

### One-Shot Data Seeding

You can also run a one-off container to seed data before starting the main stack:

```bash
# Seed data, then start the full stack
docker run --rm \
  --env GITHUB_TOKEN=$GITHUB_TOKEN \
  --env REGISTRY_REPOSITORY=ghcr.io/thawn/abstracts-data \
  -v abstracts-data:/app/data \
  ghcr.io/thawn/abstracts-explorer:latest \
  abstracts-explorer registry download --conference all --yes

docker compose up -d
```

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
- Verify ports 80 and 443 are available: `lsof -i :80 -i :443`
- Rebuild: `podman-compose build --no-cache && podman-compose up -d`
- **Existing cert setup:** ensure `./certs/cert.pem` and `./certs/key.pem` exist before starting nginx
- **Existing cert + self-signed cert:** remove the `ssl_stapling*`, `resolver`, and `resolver_timeout` lines from `nginx/nginx.conf` — OCSP stapling is not available for self-signed certificates
- **Let's Encrypt setup:** ensure you ran the Certbot standalone command (Step 2) before starting the stack

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
- The application port (5000) is also internal — use https://localhost instead
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

1. **Change default passwords** in your compose file
2. **Use external secrets** for tokens
3. **HTTPS is enabled by default** via the built-in nginx reverse proxy — see [Option 1](#option-1-existing-certificate) for an existing cert or [Option 2](#option-2-lets-encrypt-certificate) for Let's Encrypt
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

## Starting Automatically with systemd (Podman Quadlets)

The `systemd/` directory contains Podman
[quadlet](https://docs.podman.io/en/latest/markdown/podman-systemd.unit.5.html)
files that integrate the container stack directly with systemd.  Quadlets are
the modern, daemonless alternative to podman-compose: each container, volume,
and network is a native systemd unit, giving you the full `systemctl` and
`journalctl` experience with no compose wrapper needed.

Two TLS variants are provided:

| Variant | Reverse proxy | Certificate source |
|---|---|---|
| **nginx** | nginx | Your own certificate files (`cert.pem` / `key.pem`) |
| **caddy** | Caddy | Automatic Let's Encrypt (recommended for public domains) |

### Architecture

```
Internet ──► :80/:443 (systemd socket, root)
                 │
                 ▼
         systemd-socket-proxyd ──► 127.0.0.1:8080
                                        │
                                        ▼
                               nginx or Caddy (rootless Podman)
                                        │
                                        ▼
                               abstracts-explorer:5000
                                   │           │
                                   ▼           ▼
                              PostgreSQL    ChromaDB
```

Privileged ports 80/443 are held by a system-level systemd socket.
All containers run rootless under your normal user account — no `sysctl`
changes, no Docker daemon, no root containers.

Sensitive values (API tokens, database password) are stored as
[Podman secrets](https://docs.podman.io/en/latest/markdown/podman-secret-create.1.html)
and injected at runtime — they never appear in plain text in unit files.

### Automated install

An install script automates the full setup:

```bash
# nginx variant (existing SSL certificate):
curl -fsSL https://raw.githubusercontent.com/thawn/abstracts-explorer/main/scripts/install-podman.sh \
  | bash -s -- --variant nginx

# Caddy variant (automatic Let's Encrypt):
curl -fsSL https://raw.githubusercontent.com/thawn/abstracts-explorer/main/scripts/install-podman.sh \
  | bash -s -- --variant caddy
```

The script:
1. Downloads all quadlet, configuration, and environment files
2. Installs the system socket units (requires `sudo`)
3. Enables lingering for your user
4. Generates a secure database password and stores it as a Podman secret
5. Prints the remaining manual steps

After the script finishes, you only need to:

1. **Create the LLM backend API token secret:**

   ```bash
   printf '%s' 'YOUR_BLABLADOR_TOKEN' | podman secret create llm-backend-auth-token -
   ```

2. **Edit the environment file** (`~/abstracts-explorer/abstracts-explorer.env`) to
   adjust model names, LLM backend URL, and other settings.

3. **Set up TLS:**
   - **nginx:** place your certificate and key in `~/abstracts-explorer/certs/`

     ```bash
     cp /path/to/cert.pem ~/abstracts-explorer/certs/
     cp /path/to/key.pem  ~/abstracts-explorer/certs/
     ```

   - **Caddy:** edit `~/abstracts-explorer/caddy/Caddyfile` — replace
     `abstracts.example.com` with your domain and `your@email.com` with
     your email address.

4. **Start all services:**

   ```bash
   systemctl --user daemon-reload
   systemctl --user start abstracts-postgres abstracts-chromadb abstracts-explorer
   # nginx variant:
   systemctl --user start abstracts-nginx
   # Caddy variant:
   systemctl --user start abstracts-caddy
   ```

### Configuration

All non-secret settings live in a single environment file:

```
~/abstracts-explorer/abstracts-explorer.env
```

Edit this file to change the LLM backend URL, model names, log level, RAG
parameters, and other options.  Changes take effect after restarting the
abstracts-explorer service:

```bash
systemctl --user restart abstracts-explorer
```

### Managing secrets

| Secret | Required | Purpose |
|---|---|---|
| `postgres-password` | Yes | PostgreSQL database password (auto-generated by install script) |
| `llm-backend-auth-token` | Yes | Blablador or other LLM backend API key |
| `github-token` | No | GitHub token for the OCI registry feature |

Create or update a secret:

```bash
printf '%s' 'NEW_VALUE' | podman secret create --replace SECRET_NAME -
systemctl --user restart abstracts-explorer  # pick up the new value
```

### Checking status and logs

```bash
# Status of all services
systemctl --user status 'abstracts-*'

# Follow logs for a specific container
journalctl --user -u abstracts-explorer -f

# Follow logs for all containers
journalctl --user -u 'abstracts-*' -f
```

### Updating containers

An update script pulls the latest images and restarts all services:

```bash
~/abstracts-explorer/update-podman.sh
```

Or download and run it directly:

```bash
curl -fsSL https://raw.githubusercontent.com/thawn/abstracts-explorer/main/scripts/update-podman.sh | bash
```

## Further Reading

- [Podman Documentation](https://docs.podman.io/)
- [Main README](../README.md)
- [Configuration Guide](configuration.md)

## Support

- 🐛 [Report issues](https://github.com/thawn/abstracts-explorer/issues)
- 💬 [Discussions](https://github.com/thawn/abstracts-explorer/discussions)

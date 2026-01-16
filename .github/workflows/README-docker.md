# Docker Container Workflow

This workflow automatically builds and publishes Docker container images for Abstracts Explorer.

## Triggers

The workflow runs on:
- **Push to main/develop branches** - Builds and pushes to GitHub Container Registry
- **Pull requests** - Builds only (no push) for validation
- **Git tags** (v*.*.*) - Builds and pushes to both GitHub Container Registry and Docker Hub
- **Manual trigger** - Via workflow_dispatch

## Published Images

### GitHub Container Registry (GHCR)
- **Registry**: `ghcr.io`
- **Image**: `ghcr.io/thawn/abstracts-explorer`
- **Authentication**: Automatic via GitHub token
- **Availability**: All builds (branches, PRs, releases)

### Docker Hub
- **Registry**: `docker.io`
- **Image**: `thawn/abstracts-explorer`
- **Authentication**: Requires secrets (see below)
- **Availability**: Release tags only (v*.*.*)

## Image Tags

The workflow creates the following tags following best practices:

| Tag Pattern | When Created | Example | Notes |
|-------------|--------------|---------|-------|
| `latest` | Release tags only | `latest` | Always points to latest release |
| `main` | Main branch | `main` | Latest main branch build |
| `develop` | Develop branch | `develop` | Latest develop branch build |
| `v*.*.*` | Release tags | `v1.0.0` | Full semver version |
| `v*.*` | Release tags | `v1.0` | Major.minor version |
| `v*` | Release tags | `v1` | Major version |
| `sha-*` | All builds | `sha-5f8567d` | Git commit SHA (short) |
| `pr-*` | Pull requests | `pr-123` | Pull request number |

### Tagging Best Practices

This workflow follows container image tagging best practices:

1. **`latest` tag** - Points ONLY to the latest release, never to branch builds
2. **Git tags** - Each git tag (`v*.*.*`) creates corresponding container tags
3. **Git commit SHA** - Every container has a `sha-*` tag for traceability to source code
4. **Branch tags** - Development branches have their own tags for testing

## Cleanup Policy

The workflow automatically cleans up old container images:

1. **Untagged images**: Deleted immediately after new build
2. **Old images (including SHA tags)**: Deleted after 7 days
3. **Protected tags**: Never deleted (latest, main, develop, release versions like v*.*.*)
4. **Minimum kept**: Always keeps at least 10 most recent versions

**Note**: SHA tags (`sha-*`) are cleaned up after 7 days since they're primarily for debugging and development. For long-term reproducibility, use release version tags (`v*.*.*`).

## Required Secrets

### For Docker Hub Publishing

To publish to Docker Hub (for releases), add these repository secrets:

1. **DOCKERHUB_USERNAME**: Your Docker Hub username
2. **DOCKERHUB_TOKEN**: Docker Hub access token

To create a Docker Hub access token:
1. Log in to [Docker Hub](https://hub.docker.com/)
2. Go to Account Settings → Security
3. Click "New Access Token"
4. Name it (e.g., "GitHub Actions")
5. Copy the token and add it to GitHub repository secrets

### GitHub Container Registry

No additional secrets needed - uses automatic `GITHUB_TOKEN`.

## Permissions

The workflow requires these GitHub Actions permissions:
- `contents: read` - Read repository code
- `packages: write` - Push to GitHub Container Registry
- `id-token: write` - Generate attestations

## Multi-architecture Support

Images are built for:
- `linux/amd64` (x86_64)
- `linux/arm64` (ARM64/Apple Silicon)

## Build Features

- **Multi-stage build** - Optimized image size
- **Layer caching** - Faster subsequent builds via GitHub Actions cache
- **SBOM generation** - Software Bill of Materials
- **Provenance attestation** - Build provenance for releases
- **Security scanning** - Via GitHub Security tab

## Using Published Images

### Pull and run from GHCR (latest release)
```bash
podman pull ghcr.io/thawn/abstracts-explorer:latest
podman run -p 5000:5000 -v abstracts-data:/app/data ghcr.io/thawn/abstracts-explorer:latest
```

### Pull and run from Docker Hub (releases)
```bash
docker pull thawn/abstracts-explorer:v1.0.0
docker run -p 5000:5000 -v abstracts-data:/app/data thawn/abstracts-explorer:v1.0.0
```

### Pull specific commit (for debugging or testing)
```bash
# Using git SHA tag for precise version control
docker pull ghcr.io/thawn/abstracts-explorer:sha-5f8567d
docker run -p 5000:5000 -v abstracts-data:/app/data ghcr.io/thawn/abstracts-explorer:sha-5f8567d
```

### Use with docker-compose
```yaml
services:
  abstracts-explorer:
    image: ghcr.io/thawn/abstracts-explorer:latest
    # ... rest of configuration
```

## Troubleshooting

### Docker Hub push fails
- Verify DOCKERHUB_USERNAME and DOCKERHUB_TOKEN secrets are set
- Check Docker Hub token has write permissions
- Ensure Docker Hub repository exists or allows auto-creation

### Build fails
- Check Dockerfile syntax
- Verify all COPY sources exist
- Review build logs in GitHub Actions

### Image not found
- Wait for workflow to complete
- Check if workflow succeeded
- Verify you're using the correct registry and image name
- For Docker Hub, ensure it's a release tag (not a branch)

## Manual Workflow Dispatch

To manually trigger the workflow:
1. Go to Actions tab
2. Select "Build and Push Docker Images"
3. Click "Run workflow"
4. Select branch and run

## Further Reading

- [GitHub Container Registry Documentation](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
- [Docker Hub Documentation](https://docs.docker.com/docker-hub/)
- [Docker Buildx Documentation](https://docs.docker.com/buildx/working-with-buildx/)

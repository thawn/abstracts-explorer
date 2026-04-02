# syntax=docker/dockerfile:1
# Multi-stage Dockerfile for Abstracts Explorer
# This Dockerfile supports both Docker and Podman

# Stage 1: Build Python application
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS python-builder

WORKDIR /app

# Install git for version detection (hatch-vcs needs git to read .git directory)
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Mount the full build context (including .git with complete git history) without
# adding any files to an image layer.  Everything is done in one RUN so that only
# the net result (.venv and src) appears in the layer:
#   1. Copy the full build context to a temp directory.
#   2. Restore all git-tracked files via `git checkout -- .` so hatch-vcs sees a
#      clean working tree (a dirty tree causes it to append a dev marker).
#   3. Delete any stale _version.py so hatch-vcs regenerates it from git tags.
#   4. Run `uv sync`, which triggers hatch-vcs to write the correct version.
#   5. Copy only .venv and src to /app, then remove the temp directory.
RUN --mount=type=bind,target=/build \
    mkdir /tmp/repo && \
    cp -rp /build/. /tmp/repo/ && \
    cd /tmp/repo && \
    git checkout -- . && \  # Restore files excluded by .dockerignore so the working tree is clean
    rm -f src/abstracts_explorer/_version.py && \
    uv sync --frozen --no-dev --extra web && \
    cp -rp .venv /app/.venv && \
    cp -rp src /app/src && \
    cd / && rm -rf /tmp/repo


# Stage 2: Final runtime image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 abstracts && \
    mkdir -p /app/data && \
    chown -R abstracts:abstracts /app

# Copy Python virtual environment from builder
COPY --from=python-builder --chown=abstracts:abstracts /app/.venv /app/.venv
COPY --from=python-builder --chown=abstracts:abstracts /app/src /app/src

# Copy additional necessary files
COPY --chown=abstracts:abstracts README.md LICENSE ./

# Set up Python path and activate virtual environment
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH="/app/src:${PYTHONPATH}" \
    PYTHONUNBUFFERED=1

# Copy default .env configuration
COPY --chown=abstracts:abstracts .env.docker /app/.env

# Switch to non-root user
USER abstracts

# Expose web UI port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Use ENTRYPOINT + CMD pattern to allow passing arguments
ENTRYPOINT ["abstracts-explorer"]
CMD ["web-ui", "--host", "0.0.0.0", "--port", "5000"]

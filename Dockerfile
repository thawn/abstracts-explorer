# Multi-stage Dockerfile for Abstracts Explorer
# This Dockerfile supports both Docker and Podman

# Stage 1: Build Python application
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS python-builder

WORKDIR /app

# Install git for version detection (hatch-vcs needs git to determine the version from git tags)
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Copy the full repository including .git so hatch-vcs can read the git tags.
# .dockerignore excludes some git-tracked files, which would make the working tree appear
# dirty and cause hatch-vcs to generate a dev version. We fix this by running
# `git checkout -- .` to restore all tracked files to HEAD before installing.
# Layers in this builder stage don't matter — the final image only copies what it needs.
COPY . .
RUN git checkout -- .
RUN rm -f src/abstracts_explorer/_version.py
RUN uv sync --frozen --no-dev --extra web


# Stage 2: Final runtime image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    bash-completion \
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

# Enable bash tab completion for the abstracts-explorer command
RUN activate-global-python-argcomplete

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

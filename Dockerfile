# Multi-stage Dockerfile for Abstracts Explorer
# This Dockerfile supports both Docker and Podman

# Stage 1: Build Node.js dependencies
FROM node:20-slim AS node-builder

WORKDIR /app

# Copy package files
COPY package.json package-lock.json ./

# Install Node.js dependencies (production only, skip postinstall)
# Skip postinstall scripts as they try to build vendor files and setup git hooks
# Vendor files (including tailwind.css) are already committed to the repo
RUN npm ci --omit=dev --ignore-scripts

# Copy source files with pre-built vendor files
COPY src/abstracts_explorer/web_ui/static/ src/abstracts_explorer/web_ui/static/
COPY tailwind.config.js ./


# Stage 2: Build Python application
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS python-builder

WORKDIR /app

# Copy Python project files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/

# Install Python dependencies with uv
RUN uv sync --frozen --no-dev


# Stage 3: Final runtime image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    sqlite3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 abstracts && \
    mkdir -p /app/data /app/chroma_db && \
    chown -R abstracts:abstracts /app

# Copy Python virtual environment from builder
COPY --from=python-builder --chown=abstracts:abstracts /app/.venv /app/.venv
COPY --from=python-builder --chown=abstracts:abstracts /app/src /app/src

# Copy Node.js vendor files from builder (overwrite existing)
COPY --from=node-builder --chown=abstracts:abstracts /app/src/abstracts_explorer/web_ui/static/vendor /app/src/abstracts_explorer/web_ui/static/vendor
COPY --from=node-builder --chown=abstracts:abstracts /app/src/abstracts_explorer/web_ui/static/webfonts /app/src/abstracts_explorer/web_ui/static/webfonts

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

# Health check (requires requests library, which is already installed)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Default command: start web UI
CMD ["abstracts-explorer", "web-ui", "--host", "0.0.0.0", "--port", "5000"]

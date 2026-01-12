# Multi-stage Dockerfile for Abstracts Explorer
# This Dockerfile supports both Docker and Podman

# Stage 1: Build Node.js dependencies
FROM node:20-slim AS node-builder

WORKDIR /app

# Copy package files
COPY package.json package-lock.json ./

# Install Node.js dependencies
RUN npm ci --omit=dev

# Copy source files needed for vendor installation
COPY src/abstracts_explorer/web_ui/static/tailwind.input.css src/abstracts_explorer/web_ui/static/
COPY tailwind.config.js ./

# Install vendor dependencies (fonts, CSS libraries, etc.)
RUN npm run install:vendor


# Stage 2: Build Python application
FROM python:3.12-slim AS python-builder

WORKDIR /app

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

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
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 abstracts && \
    mkdir -p /app/data /app/chroma_db && \
    chown -R abstracts:abstracts /app

# Copy Python virtual environment from builder
COPY --from=python-builder --chown=abstracts:abstracts /app/.venv /app/.venv
COPY --from=python-builder --chown=abstracts:abstracts /app/src /app/src

# Copy Node.js vendor files from builder
COPY --from=node-builder --chown=abstracts:abstracts /app/src/abstracts_explorer/web_ui/static/vendor /app/src/abstracts_explorer/web_ui/static/vendor
COPY --from=node-builder --chown=abstracts:abstracts /app/src/abstracts_explorer/web_ui/static/webfonts /app/src/abstracts_explorer/web_ui/static/webfonts

# Copy additional necessary files
COPY --chown=abstracts:abstracts README.md LICENSE ./

# Set up Python path and activate virtual environment
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH="/app/src:${PYTHONPATH}" \
    PYTHONUNBUFFERED=1

# Switch to non-root user
USER abstracts

# Create default .env file (can be overridden by volume mount)
RUN echo "# Abstracts Explorer Configuration\n\
DATA_DIR=/app/data\n\
PAPER_DB_PATH=/app/data/abstracts.db\n\
EMBEDDING_DB_PATH=/app/chroma_db\n\
# LLM Backend (configure based on your setup)\n\
LLM_BACKEND_URL=http://localhost:1234\n\
CHAT_MODEL=gemma-3-4b-it-qat\n\
EMBEDDING_MODEL=text-embedding-qwen3-embedding-4b\n\
" > /app/.env

# Expose web UI port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health', timeout=5)" || exit 1

# Default command: start web UI
CMD ["abstracts-explorer", "web-ui", "--host", "0.0.0.0", "--port", "5000"]

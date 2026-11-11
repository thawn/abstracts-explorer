#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Abstracts Explorer — container update script
# https://abstracts-explorer.readthedocs.io/en/latest/docker.html
#
# Pulls the latest container images and restarts all Abstracts Explorer
# services.  Run this whenever you want to update to a new release.
#
# Usage:
#   ~/abstracts-explorer/update-podman.sh
#   # or via curl:
#   curl -fsSL https://raw.githubusercontent.com/thawn/abstracts-explorer/main/scripts/update-podman.sh | bash
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

info()  { printf '\033[1;34m▸ %s\033[0m\n' "$*"; }
ok()    { printf '\033[1;32m✔ %s\033[0m\n' "$*"; }
warn()  { printf '\033[1;33m⚠ %s\033[0m\n' "$*" >&2; }

if [ -z "${ABSTRACTS_EXPLORER_TAG:-}" ]; then
    ABSTRACTS_EXPLORER_TAG=$(systemctl --user show-environment |
        sed -n 's/^ABSTRACTS_EXPLORER_TAG=//p')
fi
ABSTRACTS_EXPLORER_TAG="${ABSTRACTS_EXPLORER_TAG:-latest}"

if [[ ! "$ABSTRACTS_EXPLORER_TAG" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$ ]]; then
    printf 'Invalid ABSTRACTS_EXPLORER_TAG: %s\n' "$ABSTRACTS_EXPLORER_TAG" >&2
    exit 1
fi

info "Pulling latest container images"
podman pull "ghcr.io/thawn/abstracts-explorer:$ABSTRACTS_EXPLORER_TAG"
podman pull docker.io/postgres:16-alpine
podman pull docker.io/chromadb/chroma:latest

# Pull the reverse-proxy image (whichever variant is active)
if systemctl --user is-active --quiet abstracts-nginx.service 2>/dev/null; then
    podman pull docker.io/nginxinc/nginx-unprivileged:stable-alpine
    PROXY_UNIT="abstracts-nginx"
elif systemctl --user is-active --quiet abstracts-caddy.service 2>/dev/null; then
    podman pull docker.io/caddy:alpine
    PROXY_UNIT="abstracts-caddy"
else
    warn "No active reverse-proxy unit found (nginx or caddy). Skipping proxy pull."
    PROXY_UNIT=""
fi

info "Restarting services"
systemctl --user set-environment "ABSTRACTS_EXPLORER_TAG=$ABSTRACTS_EXPLORER_TAG"
systemctl --user daemon-reload
systemctl --user restart abstracts-postgres abstracts-chromadb abstracts-explorer

if [ -n "$PROXY_UNIT" ]; then
    systemctl --user restart "$PROXY_UNIT"
fi

ok "All services updated and restarted"
echo ""
echo "Check status:  systemctl --user status 'abstracts-*'"
echo "Follow logs:   journalctl --user -u 'abstracts-*' -f"

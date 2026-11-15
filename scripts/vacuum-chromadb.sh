#!/usr/bin/env bash
# Reclaim disk space in the ChromaDB volume used by the Podman/Quadlet deployment.
#
# The application container connects to ChromaDB over HTTP, so
# `migrate_embeddings_compaction.py --vacuum` cannot vacuum the database from
# inside the application container. This helper stops the services, runs
# `chroma vacuum` in a one-off container with the ChromaDB volume mounted, and
# restarts the services afterwards.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: vacuum-chromadb.sh [OPTIONS]

Stop the Abstracts Explorer services and vacuum the ChromaDB data volume.

Options:
  --volume NAME   ChromaDB volume name (default: systemd-abstracts-chromadb-data)
  --path PATH     Chroma persistent directory inside the volume (default: /data)
  --image IMAGE   Container image containing the chroma CLI
                  (default: docker.io/chromadb/chroma:latest)
  --yes           Skip the confirmation prompt
  --dry-run       Print the commands without changing services or data
  -h, --help      Show this help

Environment variable equivalents:
  CHROMA_VOLUME, CHROMA_PATH, CHROMA_IMAGE

Examples:
  ./scripts/vacuum-chromadb.sh
  ./scripts/vacuum-chromadb.sh --yes
  ./scripts/vacuum-chromadb.sh --volume abstracts-chromadb-data --path /data
EOF
}

info() { printf '\033[1;34m▸ %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m✔ %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m⚠ %s\033[0m\n' "$*" >&2; }

VOLUME="${CHROMA_VOLUME:-systemd-abstracts-chromadb-data}"
CHROMA_PATH="${CHROMA_PATH:-/data}"
IMAGE="${CHROMA_IMAGE:-docker.io/chromadb/chroma:latest}"
ASSUME_YES=0
DRY_RUN=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --volume)
            VOLUME="${2:?--volume requires a value}"
            shift 2
            ;;
        --path)
            CHROMA_PATH="${2:?--path requires a value}"
            shift 2
            ;;
        --image)
            IMAGE="${2:?--image requires a value}"
            shift 2
            ;;
        --yes)
            ASSUME_YES=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            warn "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

SERVICES=(abstracts-explorer abstracts-chromadb)

run() {
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '  '
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}

restart_services() {
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '  systemctl --user start %s\n' "${SERVICES[*]}"
    elif ! systemctl --user start "${SERVICES[@]}"; then
        warn "Failed to restart services. Run: systemctl --user start ${SERVICES[*]}"
    fi
}

if [ "$DRY_RUN" -ne 1 ]; then
    command -v podman >/dev/null 2>&1 || {
        warn "podman is required for this helper."
        exit 1
    }
    command -v systemctl >/dev/null 2>&1 || {
        warn "systemctl is required for this helper."
        exit 1
    }
    if ! podman volume inspect "$VOLUME" >/dev/null 2>&1; then
        warn "ChromaDB volume '$VOLUME' was not found."
        warn "List volumes with: podman volume ls"
        warn "Override the name with: --volume NAME"
        exit 1
    fi
    if ! podman run --rm -v "$VOLUME:/data:ro" --entrypoint sh "$IMAGE" \
        -c 'test -f "$1/chroma.sqlite3"' sh "$CHROMA_PATH" >/dev/null 2>&1; then
        warn "No chroma.sqlite3 found at $CHROMA_PATH in volume '$VOLUME'."
        warn "Inspect the volume with:"
        warn "  podman run --rm -v $VOLUME:/data:ro --entrypoint sh $IMAGE -c 'find /data -maxdepth 2 -name chroma.sqlite3 -print'"
        warn "Override the path with: --path PATH"
        exit 1
    fi
fi

if [ "$ASSUME_YES" -ne 1 ]; then
    if [ ! -t 0 ]; then
        warn "Refusing to run non-interactively without --yes."
        exit 1
    fi
    printf 'This will stop %s and vacuum %s in volume %s.\n' \
        "${SERVICES[*]}" "$CHROMA_PATH" "$VOLUME"
    read -r -p "Continue? [y/N] " answer
    case "$answer" in
        y|Y) ;;
        *) echo "Aborted."; exit 1 ;;
    esac
fi

trap restart_services EXIT

info "Stopping ${SERVICES[*]}"
run systemctl --user stop "${SERVICES[@]}"

info "Vacuuming $CHROMA_PATH in volume $VOLUME"
if [ "$DRY_RUN" -eq 1 ]; then
    run podman run --rm --entrypoint chroma \
        -v "$VOLUME:/data" \
        "$IMAGE" \
        vacuum --path "$CHROMA_PATH" --force
else
    if output=$(podman run --rm --entrypoint chroma \
        -v "$VOLUME:/data" \
        "$IMAGE" \
        vacuum --path "$CHROMA_PATH" --force 2>&1); then
        printf '%s\n' "$output"
        if ! grep -q "Vacuum complete" <<<"$output"; then
            warn "ChromaDB vacuum failed."
            exit 1
        fi
    else
        printf '%s\n' "$output"
        warn "ChromaDB vacuum failed."
        exit 1
    fi
fi

ok "ChromaDB vacuum complete"

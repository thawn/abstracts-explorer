#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Abstracts Explorer — migrate existing data to Podman named volumes
# https://abstracts-explorer.readthedocs.io/en/latest/docker.html
#
# Copies data from an existing Docker Compose deployment into Podman named
# volumes used by the quadlet setup.  A timestamped backup of the source data
# is created before any changes are made.
#
# Prerequisites:
#   - The Podman quadlet install script has already been run
#   - Podman is installed and the user has rootless access
#   - The source data directories exist
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/thawn/abstracts-explorer/main/scripts/migrate-to-podman.sh | bash
#
#   Or with a custom source directory:
#   bash migrate-to-podman.sh --source /path/to/old/deployment
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── defaults ──────────────────────────────────────────────────────────────────
SOURCE_DIR="."
BACKUP_DIR=""
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ── helpers ───────────────────────────────────────────────────────────────────
info()  { printf '\033[1;34m▸ %s\033[0m\n' "$*"; }
ok()    { printf '\033[1;32m✔ %s\033[0m\n' "$*"; }
warn()  { printf '\033[1;33m⚠ %s\033[0m\n' "$*" >&2; }
die()   { printf '\033[1;31m✘ %s\033[0m\n' "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
Usage: migrate-to-podman.sh [OPTIONS]
  --source DIR   Source deployment directory (default: current directory)
  --help         Show this help message
EOF
    exit 0
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "'$1' is required but not installed."
}

# ── parse arguments ──────────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
    case "$1" in
        --source) SOURCE_DIR="${2:?--source requires a value}"; shift 2 ;;
        --help)   usage ;;
        *)        die "Unknown option: $1" ;;
    esac
done

SOURCE_DIR=$(cd "$SOURCE_DIR" && pwd)
BACKUP_DIR="$SOURCE_DIR/backup_${TIMESTAMP}"

# ── pre-flight checks ────────────────────────────────────────────────────────
require_cmd podman

info "Migrating data from $SOURCE_DIR to Podman named volumes"

# Check that at least one data source exists
HAS_DATA=false
[ -d "$SOURCE_DIR/data" ]      && HAS_DATA=true
[ -d "$SOURCE_DIR/chroma_db" ] && HAS_DATA=true

if [ "$HAS_DATA" = false ]; then
    die "No data directories found in $SOURCE_DIR (expected data/ or chroma_db/)"
fi

# ── stop running services ────────────────────────────────────────────────────
info "Stopping Abstracts Explorer services (if running)"
systemctl --user stop abstracts-nginx.service 2>/dev/null || true
systemctl --user stop abstracts-caddy.service 2>/dev/null || true
systemctl --user stop abstracts-explorer.service 2>/dev/null || true
systemctl --user stop abstracts-chromadb.service 2>/dev/null || true
systemctl --user stop abstracts-postgres.service 2>/dev/null || true
ok "Services stopped"

# ── create backup ─────────────────────────────────────────────────────────────
info "Creating backup at $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

if [ -d "$SOURCE_DIR/data" ]; then
    cp -a "$SOURCE_DIR/data" "$BACKUP_DIR/data"
    ok "Backed up data/"
fi

if [ -d "$SOURCE_DIR/chroma_db" ]; then
    cp -a "$SOURCE_DIR/chroma_db" "$BACKUP_DIR/chroma_db"
    ok "Backed up chroma_db/"
fi
ok "Backup complete: $BACKUP_DIR"

# ── helper: resolve UID/GID for an image ─────────────────────────────────────
# In rootless Podman the host UID of files inside a volume differs from the
# container-internal UID due to user-namespace UID remapping.  We therefore
# query the container's own view of its UID/GID so that the chown command
# runs inside the target namespace and Podman applies the correct mapping
# automatically.
#
# Resolution order:
#   1. Inspect the existing volume's mountpoint for the owner of its contents
#      (most accurate when the volume was already initialised by the service).
#   2. Run the image briefly with `id -u` / `id -g` to query the runtime UID.
#   3. Fall back to the supplied defaults.
get_uid_gid_for_volume() {
    local volume_name="$1"
    local image="$2"
    local default_uid="${3:-0}"
    local default_gid="${4:-0}"

    # 1. Try to read from an existing, populated volume mountpoint
    local mount_point
    mount_point=$(podman volume inspect "$volume_name" --format '{{ .Mountpoint }}' 2>/dev/null || true)
    if [ -n "$mount_point" ] && [ -d "$mount_point" ]; then
        local first_entry
        first_entry=$(find "$mount_point" -maxdepth 1 -mindepth 1 2>/dev/null | head -1)
        if [ -n "$first_entry" ]; then
            local vol_uid vol_gid
            vol_uid=$(stat -c '%u' "$first_entry" 2>/dev/null || true)
            vol_gid=$(stat -c '%g' "$first_entry" 2>/dev/null || true)
            if [ -n "$vol_uid" ] && [ "$vol_uid" != "0" ]; then
                echo "${vol_uid}:${vol_gid}"
                return
            fi
        fi
    fi

    # 2. Query the running UID directly from the container image
    local run_uid run_gid
    run_uid=$(podman run --rm --quiet "$image" id -u 2>/dev/null || true)
    run_gid=$(podman run --rm --quiet "$image" id -g 2>/dev/null || true)
    if [ -n "$run_uid" ]; then
        echo "${run_uid}:${run_gid:-$run_uid}"
        return
    fi

    # 3. Fall back to defaults
    echo "${default_uid}:${default_gid}"
}

# ── helper: copy data into a named volume ─────────────────────────────────────
copy_to_volume() {
    local src_dir="$1"
    local volume_name="$2"
    local owner="${3:-0:0}"   # "uid:gid" as seen inside the container namespace

    if [ ! -d "$src_dir" ]; then
        warn "Source directory $src_dir does not exist — skipping"
        return
    fi

    info "Copying $src_dir → volume $volume_name (owner ${owner})"

    # Ensure the volume exists
    podman volume inspect "$volume_name" >/dev/null 2>&1 || \
        podman volume create "$volume_name"

    # Copy contents inside a container so that Podman's user-namespace UID
    # remapping is applied correctly — chown uses container-internal IDs.
    podman run --rm \
        -v "$src_dir:/source:ro" \
        -v "${volume_name}:/dest" \
        docker.io/alpine:3.21 \
        sh -c "cp -a /source/. /dest/ && chown -R ${owner} /dest/"

    ok "Copied to volume $volume_name"
}

# ── migrate application data ─────────────────────────────────────────────────
if [ -d "$SOURCE_DIR/data" ]; then
    APP_IMAGE="ghcr.io/thawn/abstracts-explorer:latest"
    APP_OWNER=$(get_uid_gid_for_volume "systemd-abstracts-data" "$APP_IMAGE" 1000 1000)
    info "abstracts-explorer data owner: $APP_OWNER"
    copy_to_volume "$SOURCE_DIR/data" "systemd-abstracts-data" "$APP_OWNER"
fi

# ── migrate ChromaDB data ────────────────────────────────────────────────────
if [ -d "$SOURCE_DIR/chroma_db" ]; then
    CHROMA_IMAGE="docker.io/chromadb/chroma:latest"
    CHROMA_OWNER=$(get_uid_gid_for_volume "systemd-abstracts-chromadb-data" "$CHROMA_IMAGE" 1000 1000)
    info "ChromaDB data owner: $CHROMA_OWNER"
    copy_to_volume "$SOURCE_DIR/chroma_db" "systemd-abstracts-chromadb-data" "$CHROMA_OWNER"
fi

# ── migrate PostgreSQL data (if available) ────────────────────────────────────
# Docker Compose PostgreSQL data is usually in a Docker named volume.
# If the user has extracted it to a local directory, handle it here.
PG_DATA_DIRS=("$SOURCE_DIR/postgres_data" "$SOURCE_DIR/pgdata" "$SOURCE_DIR/postgres")
for pg_dir in "${PG_DATA_DIRS[@]}"; do
    if [ -d "$pg_dir" ]; then
        info "Found PostgreSQL data at $pg_dir"
        PG_IMAGE="docker.io/postgres:16-alpine"
        PG_OWNER=$(get_uid_gid_for_volume "systemd-abstracts-postgres-data" "$PG_IMAGE" 999 999)
        info "PostgreSQL data owner: $PG_OWNER"
        copy_to_volume "$pg_dir" "systemd-abstracts-postgres-data" "$PG_OWNER"
        break
    fi
done

# ── print summary ────────────────────────────────────────────────────────────
echo ""
ok "Migration complete!"
echo ""
echo "Backup of original data: $BACKUP_DIR"
echo ""
echo "Start the services:"
echo "  systemctl --user daemon-reload"
echo "  systemctl --user start abstracts-postgres abstracts-chromadb abstracts-explorer"
echo ""
echo "If using Caddy:  systemctl --user start abstracts-caddy"
echo "If using nginx:  systemctl --user start abstracts-nginx"
echo ""
echo "Check status:  systemctl --user status 'abstracts-*'"

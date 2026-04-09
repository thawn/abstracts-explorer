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
declare -A ORIGINAL_OWNERS  # tracks dirs whose ownership we temporarily changed
# Initialised to empty; populated after SOURCE_DIR is resolved (see below)
CHR_DATA_DIR=""
PG_DATA_DIR=""
PG_TEMP_DIR=""

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

# ── helper: temporarily grant the current user access to a source directory ───
# Docker Compose volumes are often owned by root or a service account, making
# them unreadable by the unprivileged user running this script.  We chown the
# directory to the current user for the duration of the migration and restore
# the original ownership when the script exits (see the EXIT trap below).
ensure_accessible() {
    local dir="$1"
    [ -d "$dir" ] || return 0   # nothing to do if the directory does not exist
    if ! [ -r "$dir" ] || ! [ -x "$dir" ]; then
        info "Source directory $dir is not accessible — requesting elevated access"
        ORIGINAL_OWNERS["$dir"]=$(sudo stat -c '%u:%g' "$dir") || \
            die "Cannot stat $dir. Ensure sudo is available and you have permission to use it."
        sudo chown -R "$(id -u):$(id -g)" "$dir" || \
            die "Cannot access $dir. Run: sudo chown -R $(id -u):$(id -g) $dir"
        ok "Temporarily changed ownership of $dir"
    fi
}

# Restore original ownership of any directories we temporarily chowned, and
# remove any leftover temp directory that held the postgres password.
# Called automatically via the EXIT trap so cleanup always happens regardless
# of whether the script succeeds, fails, or is interrupted.
cleanup() {
    # Remove the password temp dir if it was created
    if [ -n "$PG_TEMP_DIR" ] && [ -d "$PG_TEMP_DIR" ]; then
        rm -rf "$PG_TEMP_DIR"
    fi
    # Restore original ownership of any directories we temporarily chowned
    for dir in "${!ORIGINAL_OWNERS[@]}"; do
        local original="${ORIGINAL_OWNERS[$dir]}"
        info "Restoring ownership of $dir to $original"
        sudo chown -R "$original" "$dir" 2>/dev/null || \
            warn "Could not restore ownership of $dir to $original — please run: sudo chown -R $original $dir"
    done
}

trap cleanup EXIT

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

# ── discover data directories (after SOURCE_DIR is resolved) ─────────────────
for dir in "$SOURCE_DIR/chroma_db" "$SOURCE_DIR/chromadb_data" "$SOURCE_DIR/chroma"; do
    if [ -d "$dir" ]; then
        CHR_DATA_DIR="$dir"
        info "Found ChromaDB data at $dir"
        break
    fi
done
for dir in "$SOURCE_DIR/postgres_data" "$SOURCE_DIR/pgdata" "$SOURCE_DIR/postgres"; do
    if [ -d "$dir" ]; then
        PG_DATA_DIR="$dir"
        info "Found PostgreSQL data at $dir"
        break
    fi
done

# ── pre-flight checks ────────────────────────────────────────────────────────
require_cmd podman

info "Migrating data from $SOURCE_DIR to Podman named volumes"

# Check that at least one data source exists
HAS_DATA=false
[ -d "$SOURCE_DIR/data" ]       && HAS_DATA=true
[ -n "$PG_DATA_DIR" ]           && HAS_DATA=true
[ -n "$CHR_DATA_DIR" ]          && HAS_DATA=true

if [ "$HAS_DATA" = false ]; then
    die "No data directories found in $SOURCE_DIR (expected data/, chroma_db/, or postgres_data/)"
fi

# ── stop running services ────────────────────────────────────────────────────
info "Stopping Abstracts Explorer services (if running)"
systemctl --user stop abstracts-nginx.service 2>/dev/null || true
systemctl --user stop abstracts-caddy.service 2>/dev/null || true
systemctl --user stop abstracts-explorer.service 2>/dev/null || true
systemctl --user stop abstracts-chromadb.service 2>/dev/null || true
systemctl --user stop abstracts-postgres.service 2>/dev/null || true
ok "Services stopped"

# ── ensure source directories are accessible ─────────────────────────────────
# Docker Compose volumes are often owned by root; request elevated access now
# so that both the backup (cp -a) and the Podman bind-mount can read them.
[ -d "$SOURCE_DIR/data" ]  && ensure_accessible "$SOURCE_DIR/data"
[ -n "$CHR_DATA_DIR" ]     && ensure_accessible "$CHR_DATA_DIR"
[ -n "$PG_DATA_DIR" ]      && ensure_accessible "$PG_DATA_DIR"

# ── create backup ─────────────────────────────────────────────────────────────
info "Creating backup at $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

if [ -d "$SOURCE_DIR/data" ]; then
    cp -a "$SOURCE_DIR/data" "$BACKUP_DIR/data"
    ok "Backed up data/"
fi

if [ -n "$PG_DATA_DIR" ]; then
    cp -a "$PG_DATA_DIR" "$BACKUP_DIR/$(basename "$PG_DATA_DIR")"
    ok "Backed up PostgreSQL data from $PG_DATA_DIR"
fi

if [ -n "$CHR_DATA_DIR" ]; then
    cp -a "$CHR_DATA_DIR" "$BACKUP_DIR/$(basename "$CHR_DATA_DIR")"
    ok "Backed up ChromaDB data from $CHR_DATA_DIR"
fi
ok "Backup complete: $BACKUP_DIR"

# ── helper: resolve UID/GID for a volume ─────────────────────────────────────
# In rootless Podman the host UID of files inside a volume differs from the
# container-internal UID due to user-namespace UID remapping.  We therefore
# mount the volume inside an alpine container and stat its contents from
# within that container namespace — the IDs returned are the correct values
# to pass to chown inside copy_to_volume.
#
# Resolution order:
#   1. Mount an existing, populated volume inside an alpine container and stat
#      the owner of its contents (most accurate when already initialised).
#   2. Fall back to the supplied defaults.
get_uid_gid_for_volume() {
    local volume_name="$1"
    local default_uid="${2:-0}"
    local default_gid="${3:-0}"

    # 1. Query ownership from inside the container namespace
    local uid_gid
    uid_gid=$(podman run --rm \
        -v "${volume_name}:/mnt" \
        docker.io/alpine:3.21 \
        sh -c 'f=$(find /mnt -maxdepth 1 -mindepth 1 2>/dev/null | head -1); [ -n "$f" ] && stat -c "%u:%g" "$f"' \
        2>/dev/null || true)
    if [ -n "$uid_gid" ] && [ "$uid_gid" != "0:0" ]; then
        echo "$uid_gid"
        return
    fi

    # 2. Fall back to defaults
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
        sh -c 'cp -a /source/. /dest/ && chown -R "$1" /dest' sh "$owner"

    ok "Copied to volume $volume_name"
}

# ── migrate application data ─────────────────────────────────────────────────
if [ -d "$SOURCE_DIR/data" ]; then
    APP_OWNER=$(get_uid_gid_for_volume "systemd-abstracts-data" 1000 1000)
    info "abstracts-explorer data owner: $APP_OWNER"
    copy_to_volume "$SOURCE_DIR/data" "systemd-abstracts-data" "$APP_OWNER"
fi

# ── migrate ChromaDB data ────────────────────────────────────────────────────
if [ -n "$CHR_DATA_DIR" ]; then
    CHROMA_OWNER=$(get_uid_gid_for_volume "systemd-abstracts-chromadb-data" 1000 1000)
    info "ChromaDB data owner: $CHROMA_OWNER"
    copy_to_volume "$CHR_DATA_DIR" "systemd-abstracts-chromadb-data" "$CHROMA_OWNER"
fi

# ── migrate PostgreSQL data (if available) ────────────────────────────────────
# Docker Compose PostgreSQL data is usually in a Docker named volume.
# If the user has extracted it to a local directory, handle it here.
PG_MIGRATED=false
if [ -n "$PG_DATA_DIR" ]; then
    PG_OWNER=$(get_uid_gid_for_volume "systemd-abstracts-postgres-data" 999 999)
    info "PostgreSQL data owner: $PG_OWNER"
    copy_to_volume "$PG_DATA_DIR" "systemd-abstracts-postgres-data" "$PG_OWNER"
    PG_MIGRATED=true
fi

# ── reset PostgreSQL password to match the Podman secret ─────────────────────
# The Docker Compose deployment uses a hard-coded password; after migrating the
# data directory that password is still stored in pg_shadow.  We start a
# temporary postgres container with trust authentication (via a custom
# hba_file so the original pg_hba.conf is never modified) and run ALTER USER
# to set the password to what is stored in the 'postgres-password' Podman
# secret, which is what abstracts-postgres.container expects.
if [ "$PG_MIGRATED" = "true" ]; then
    if ! podman secret inspect postgres-password >/dev/null 2>&1; then
        warn "Podman secret 'postgres-password' not found — skipping password reset"
        warn "Run install-podman.sh first, then manually reset the PostgreSQL password"
    else
        info "Resetting PostgreSQL user password to match Podman secret"

        # Retrieve the secret value via a helper container
        NEW_PG_PASSWORD=$(podman run --rm \
            --secret postgres-password,type=env,target=POSTGRES_PASSWORD \
            docker.io/alpine:3.21 \
            sh -c 'printf "%s" "$POSTGRES_PASSWORD"' 2>/dev/null) || NEW_PG_PASSWORD=""

        if [ -z "$NEW_PG_PASSWORD" ]; then
            warn "Could not read postgres-password secret — skipping password reset"
        else
            # Write password and a trust-only pg_hba.conf to temp files so they
            # can be bind-mounted into the container without appearing in `ps`.
            # PG_TEMP_DIR is tracked by the EXIT trap (cleanup()) so it is always
            # removed even if the script is interrupted.
            PG_TEMP_DIR=$(mktemp -d)
            printf '%s' "$NEW_PG_PASSWORD" > "$PG_TEMP_DIR/new_password"
            chmod 600 "$PG_TEMP_DIR/new_password"
            cat > "$PG_TEMP_DIR/pg_hba.conf" << 'PGEOF'
# Temporary trust config for password migration — replaced immediately after
local   all             all                                     trust
host    all             all             127.0.0.1/32            trust
host    all             all             ::1/128                 trust
PGEOF
            unset NEW_PG_PASSWORD  # clear from shell memory

            # Remove any leftover container from a previous failed run
            podman rm -f abstracts-postgres-pwmigration >/dev/null 2>&1 || true

            # Start a temporary postgres container with our trust HBA so we can
            # connect without knowing the old password.
            # Passing 'postgres -c hba_file=...' as CMD causes docker-entrypoint.sh
            # to exec postgres directly (existing PGDATA is not re-initialised).
            MIGRATION_CID=$(podman run -d \
                --name abstracts-postgres-pwmigration \
                -v "systemd-abstracts-postgres-data:/var/lib/postgresql/data" \
                -v "$PG_TEMP_DIR/pg_hba.conf:/tmp/pg_hba.conf:ro" \
                -v "$PG_TEMP_DIR/new_password:/tmp/new_password:ro" \
                docker.io/postgres:16-alpine \
                postgres -c hba_file=/tmp/pg_hba.conf 2>/dev/null) || MIGRATION_CID=""

            if [ -z "$MIGRATION_CID" ]; then
                warn "Could not start temporary postgres container — skipping password reset"
            else
                # Wait up to 30 s for postgres to become ready
                PG_READY=false
                PG_WAIT=0
                while [ "$PG_WAIT" -lt 30 ]; do
                    if podman exec "$MIGRATION_CID" pg_isready -U abstracts -q 2>/dev/null; then
                        PG_READY=true
                        break
                    fi
                    sleep 1
                    PG_WAIT=$((PG_WAIT + 1))
                done

                if [ "$PG_READY" = "true" ]; then
                    # Escape any single quotes in the password for SQL string literal
                    # safety (passwords from install-podman.sh are alphanumeric, but
                    # guard against manually created secrets too).
                    if podman exec "$MIGRATION_CID" \
                        sh -c "PW=\$(cat /tmp/new_password)
                               PW_ESC=\$(printf '%s' \"\$PW\" | sed \"s/'/''/g\")
                               psql -U abstracts -c \"ALTER USER abstracts PASSWORD '\$PW_ESC';\""; then
                        ok "PostgreSQL password updated to match Podman secret"
                    else
                        warn "Failed to update PostgreSQL password — you may need to do this manually"
                    fi
                else
                    warn "PostgreSQL did not become ready in time — skipping password reset"
                fi

                podman stop "$MIGRATION_CID" >/dev/null
                podman rm   "$MIGRATION_CID" >/dev/null
            fi

            rm -rf "$PG_TEMP_DIR"
            PG_TEMP_DIR=""  # prevent double-removal by cleanup()
        fi
    fi
fi

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

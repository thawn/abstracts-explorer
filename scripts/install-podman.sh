#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Abstracts Explorer — Podman quadlet installer
# https://abstracts-explorer.readthedocs.io/en/latest/docker.html
#
# Downloads configuration files, generates secrets, and installs systemd units
# so that Abstracts Explorer starts automatically on boot.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/thawn/abstracts-explorer/main/scripts/install-podman.sh | bash -s -- [OPTIONS]
#
# Options:
#   --variant caddy|nginx   TLS variant (default: caddy)
#   --branch  BRANCH        Git branch to download from (default: main)
#   --help                  Show this help message
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── defaults ──────────────────────────────────────────────────────────────────
VARIANT="caddy"
BRANCH="main"
BASE_URL=""  # set after parsing args
DEPLOY_DIR="$HOME/abstracts-explorer"
QUADLET_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/containers/systemd"

# ── helpers ───────────────────────────────────────────────────────────────────
info()  { printf '\033[1;34m▸ %s\033[0m\n' "$*"; }
ok()    { printf '\033[1;32m✔ %s\033[0m\n' "$*"; }
warn()  { printf '\033[1;33m⚠ %s\033[0m\n' "$*" >&2; }
die()   { printf '\033[1;31m✘ %s\033[0m\n' "$*" >&2; exit 1; }

usage() {
    sed -n '/^# Usage:/,/^# ─/{ /^# ─/d; s/^# //; p }' "$0" 2>/dev/null || cat <<'EOF'
Usage: install-podman.sh [OPTIONS]
  --variant caddy|nginx   TLS variant (default: caddy)
  --branch  BRANCH        Git branch to download from (default: main)
  --help                  Show this help message
EOF
    exit 0
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "'$1' is required but not installed."
}

download() {
    # download URL DEST
    curl -fsSL "$1" -o "$2" || die "Failed to download $1"
}

generate_password() {
    # 32-character alphanumeric password.  Read 48 random bytes to ensure
    # enough characters survive the base64 + alphanumeric filter.
    head -c 48 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 32
}

# ── parse arguments ──────────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
    case "$1" in
        --variant) VARIANT="${2:?--variant requires a value}"; shift 2 ;;
        --branch)  BRANCH="${2:?--branch requires a value}";   shift 2 ;;
        --help)    usage ;;
        *)         die "Unknown option: $1" ;;
    esac
done

case "$VARIANT" in
    nginx|caddy) ;;
    *) die "Invalid variant '$VARIANT'. Use 'nginx' or 'caddy'." ;;
esac

BASE_URL="https://raw.githubusercontent.com/thawn/abstracts-explorer/${BRANCH}"

# ── pre-flight checks ────────────────────────────────────────────────────────
require_cmd curl
require_cmd podman
require_cmd systemctl
require_cmd loginctl

info "Installing Abstracts Explorer (variant: $VARIANT, branch: $BRANCH)"

# ── 1. Create directory structure ─────────────────────────────────────────────
info "Creating directory structure under $DEPLOY_DIR"
mkdir -p "$DEPLOY_DIR"
mkdir -p "$QUADLET_DIR"

if [ "$VARIANT" = "nginx" ]; then
    mkdir -p "$DEPLOY_DIR/nginx" "$DEPLOY_DIR/certs"
elif [ "$VARIANT" = "caddy" ]; then
    mkdir -p "$DEPLOY_DIR/caddy"
fi

# ── 2. Download configuration files ──────────────────────────────────────────
info "Downloading configuration files from branch '$BRANCH'"

# Environment file (no secrets — restrict permissions anyway for safety)
download "$BASE_URL/systemd/abstracts-explorer.env" \
         "$DEPLOY_DIR/abstracts-explorer.env"
chmod 600 "$DEPLOY_DIR/abstracts-explorer.env"

# Update script
download "$BASE_URL/scripts/update-podman.sh" \
         "$DEPLOY_DIR/update-podman.sh"
chmod +x "$DEPLOY_DIR/update-podman.sh"

# Shared quadlet files
for f in \
    abstracts-explorer.container \
    abstracts-postgres.container \
    abstracts-chromadb.container \
    abstracts.network \
    abstracts-data.volume \
    abstracts-chromadb-data.volume \
    abstracts-postgres-data.volume; do
    download "$BASE_URL/systemd/user/$f" "$QUADLET_DIR/$f"
done

# Log-retention timer (regular user units, not Quadlet)
SYSTEMD_USER_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$SYSTEMD_USER_DIR"
for f in abstracts-log-cleanup.service abstracts-log-cleanup.timer; do
    download "$BASE_URL/systemd/user/$f" "$SYSTEMD_USER_DIR/$f"
done

# Variant-specific files
if [ "$VARIANT" = "nginx" ]; then
    download "$BASE_URL/systemd/user/nginx/abstracts-nginx.container" \
             "$QUADLET_DIR/abstracts-nginx.container"
    download "$BASE_URL/nginx/nginx.quadlet.conf" \
             "$DEPLOY_DIR/nginx/nginx.quadlet.conf"
elif [ "$VARIANT" = "caddy" ]; then
    download "$BASE_URL/systemd/user/caddy/abstracts-caddy.container" \
             "$QUADLET_DIR/abstracts-caddy.container"
    download "$BASE_URL/systemd/user/caddy/abstracts-caddy-data.volume" \
             "$QUADLET_DIR/abstracts-caddy-data.volume"
    download "$BASE_URL/caddy/Caddyfile" \
             "$DEPLOY_DIR/caddy/Caddyfile"
fi

# System socket units (needs sudo)
info "Installing system socket units (requires sudo)"
TMPDIR_SYSTEM=$(mktemp -d)
download "$BASE_URL/systemd/system/abstracts-web.socket" \
         "$TMPDIR_SYSTEM/abstracts-web.socket"
download "$BASE_URL/systemd/system/abstracts-web-proxy.service" \
         "$TMPDIR_SYSTEM/abstracts-web-proxy.service"

# Set SocketUser to the current user's UID
CURRENT_UID=$(id -u)
sed -i "s/^SocketUser=.*/SocketUser=${CURRENT_UID}/" \
    "$TMPDIR_SYSTEM/abstracts-web.socket"

sudo cp "$TMPDIR_SYSTEM/abstracts-web.socket"       /etc/systemd/system/
sudo cp "$TMPDIR_SYSTEM/abstracts-web-proxy.service" /etc/systemd/system/
rm -rf "$TMPDIR_SYSTEM"

sudo systemctl daemon-reload
sudo systemctl enable --now abstracts-web.socket
ok "System socket units installed and enabled"

# Reload the user systemd manager so Podman Quadlet generates the .service
# units from the .container / .volume / .network files installed above.
# Without this step 'abstracts-explorer.service' (and friends) do not exist yet.
info "Reloading user systemd daemon (Podman Quadlet generator)"
systemctl --user daemon-reload
ok "User systemd daemon reloaded — quadlet service units are now available"

# ── 3. Enable lingering ──────────────────────────────────────────────────────
info "Enabling lingering for user $USER"
loginctl enable-linger "$USER"
ok "Lingering enabled"

# ── 4. Generate database password and store as Podman secret ─────────────────
if podman secret inspect postgres-password >/dev/null 2>&1; then
    ok "Podman secret 'postgres-password' already exists — keeping it"
else
    info "Generating database password"
    DB_PASSWORD=$(generate_password)
    printf '%s' "$DB_PASSWORD" | podman secret create postgres-password -
    ok "Podman secret 'postgres-password' created"
fi

# ── 5. Enable log-retention timer (GDPR: 7 days) ────────────────────────────
info "Enabling log-retention timer (GDPR: 7-day vacuum)"
# The timer runs 'journalctl --user --vacuum-time=7d' daily so container logs
# are automatically deleted after 7 days.
# Reload to ensure the timer unit is visible before enabling it.
systemctl --user daemon-reload
systemctl --user enable --now abstracts-log-cleanup.timer
ok "Log-retention timer enabled"

# ── 6. Prompt for remaining configuration ────────────────────────────────────
echo ""
info "Configuration files are in $DEPLOY_DIR"

if [ "$VARIANT" = "nginx" ]; then
    warn "Place your SSL certificate and key in:"
    echo "  $DEPLOY_DIR/certs/cert.pem"
    echo "  $DEPLOY_DIR/certs/key.pem"
    echo ""
elif [ "$VARIANT" = "caddy" ]; then
    warn "Edit the Caddyfile to set your domain and email:"
    echo "  $DEPLOY_DIR/caddy/Caddyfile"
    echo ""
fi

warn "Create the LLM backend API token secret:"
echo "  printf '%s' 'YOUR_TOKEN' | podman secret create llm-backend-auth-token -"
echo ""
warn "Review and edit the environment file:"
echo "  $DEPLOY_DIR/abstracts-explorer.env"
echo ""

# ── 7. Print next steps ──────────────────────────────────────────────────────
cat <<EOF

$(ok "Installation complete!")

Next steps:
  1. Create the LLM backend API token secret:
       printf '%s' 'YOUR_TOKEN' | podman secret create llm-backend-auth-token -

  2. Edit the environment file:
       nano $DEPLOY_DIR/abstracts-explorer.env
EOF

if [ "$VARIANT" = "nginx" ]; then
    cat <<EOF

  3. Place your SSL certificate and key:
       cp /path/to/cert.pem $DEPLOY_DIR/certs/
       cp /path/to/key.pem  $DEPLOY_DIR/certs/
EOF
elif [ "$VARIANT" = "caddy" ]; then
    cat <<EOF

  3. Edit the Caddyfile (set domain and email):
       nano $DEPLOY_DIR/caddy/Caddyfile
EOF
fi

cat <<EOF

  4. Start all services:
       systemctl --user start abstracts-postgres abstracts-chromadb abstracts-explorer
EOF

if [ "$VARIANT" = "nginx" ]; then
    echo "       systemctl --user start abstracts-nginx"
else
    echo "       systemctl --user start abstracts-caddy"
fi

cat <<EOF

  5. Check status:
       systemctl --user status 'abstracts-*'

  Optional — enable GitHub token (only required for OCI registry uploads, downloads work without):
       printf '%s' 'YOUR_GITHUB_TOKEN' | podman secret create github-token -
       # Then uncomment the Secret=github-token line in:
       #   $QUADLET_DIR/abstracts-explorer.container

Documentation: https://abstracts-explorer.readthedocs.io/en/latest/docker.html
EOF

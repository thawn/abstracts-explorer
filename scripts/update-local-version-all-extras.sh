#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Abstracts Explorer — local version refresh (main branch)
#
# Fast-forwards the local checkout to origin/main (the latest tagged release),
# then regenerates the hatch-vcs version file by reinstalling ONLY the
# abstracts-explorer package.  uv.lock is never modified (`--frozen`).
#
# Usage:
#   scripts/update-local-version.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

info() { printf '\033[1;34m▸ %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m✔ %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m⚠ %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31m✖ %s\033[0m\n' "$*" >&2; exit 1; }

command -v uv >/dev/null 2>&1 || die "uv is not installed (https://astral.sh/uv)"

if ! git diff --quiet -- uv.lock; then
    warn "uv.lock already has uncommitted changes; refusing to continue"
    die "commit or discard them first so we can prove it stays untouched"
fi

info "Fetching tags"
git fetch --tags origin

info "Switching to main and fast-forwarding to origin/main"
git checkout main
git merge --ff-only origin/main

info "Regenerating version (reinstalling abstracts-explorer only, uv.lock frozen)"
uv sync --frozen --all-extras --reinstall-package abstracts-explorer

if ! git diff --quiet -- uv.lock; then
    die "uv.lock was modified unexpectedly"
fi
ok "uv.lock unchanged"

ok "Version now: $(uv run abstracts-explorer --version)"
echo "git describe: $(git describe --tags)"

#!/usr/bin/env bash
# RunCLIImageConfigStudio.sh — start the Rust backend and Vite frontend together.
#
# The Rust backend runs in the background; its stdout/stderr go to
#   Typescript/CLIImageConfigStudio/studio-backend.log  (covered by *.log gitignore)
# The Vite dev server runs in the foreground so you see the URL and HMR output.
# Ctrl+C (or any exit signal) stops both.
#
# Usage (from anywhere in the repo, or outside it):
#   bash Scripts/QuickAliases/RunCLIImageConfigStudio.sh
#
# First run compiles the Rust backend; HEALTH_TIMEOUT is set to 120 s to allow
# for that. Subsequent runs use the Cargo cache and start in seconds.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
STUDIO_DIR="$REPO_ROOT/Typescript/CLIImageConfigStudio"
BACKEND_MANIFEST="$STUDIO_DIR/backend/Cargo.toml"
FRONTEND_DIR="$STUDIO_DIR/frontend"
BACKEND_LOG="$STUDIO_DIR/studio-backend.log"

BACKEND_HOST="127.0.0.1"
BACKEND_PORT=8876
HEALTH_TIMEOUT=120  # seconds; first cargo compile can take ~30–60 s

backend_pid=""
frontend_pid=""

# ── Cleanup ───────────────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "[studio] Shutting down…"
    [[ -n "$frontend_pid" ]] && kill "$frontend_pid" 2>/dev/null || true
    [[ -n "$backend_pid"  ]] && kill "$backend_pid"  2>/dev/null || true
    wait 2>/dev/null || true
    echo "[studio] Done."
}
trap cleanup EXIT INT TERM

# ── Helpers ───────────────────────────────────────────────────────────────────
port_open() {
    # Uses bash built-in /dev/tcp — no external tool required.
    (echo >/dev/tcp/"$1"/"$2") 2>/dev/null
}

wait_for_port() {
    local host="$1" port="$2"
    local i=0 max=$(( HEALTH_TIMEOUT * 2 ))  # 0.5 s steps
    while (( i < max )); do
        port_open "$host" "$port" && return 0
        sleep 0.5
        i=$(( i + 1 ))
    done
    return 1
}

# ── Backend ───────────────────────────────────────────────────────────────────
if port_open "$BACKEND_HOST" "$BACKEND_PORT"; then
    echo "[studio] Port ${BACKEND_PORT} already open — reusing existing backend."
else
    echo "[studio] Starting Rust backend on ${BACKEND_HOST}:${BACKEND_PORT}…"
    echo "[studio] Backend log → ${BACKEND_LOG}"
    cargo run \
        --manifest-path "$BACKEND_MANIFEST" \
        -- \
        --host "$BACKEND_HOST" \
        --port "$BACKEND_PORT" \
        --repo-root "$REPO_ROOT" \
        >"$BACKEND_LOG" 2>&1 &
    backend_pid=$!

    echo "[studio] Waiting for backend (up to ${HEALTH_TIMEOUT}s — includes first compile)…"
    if ! wait_for_port "$BACKEND_HOST" "$BACKEND_PORT"; then
        echo "[studio] ERROR: Backend did not start within ${HEALTH_TIMEOUT}s." >&2
        echo "[studio]        Check: ${BACKEND_LOG}" >&2
        exit 1
    fi
    echo "[studio] Backend ready on ${BACKEND_HOST}:${BACKEND_PORT}."
fi

# ── Frontend ──────────────────────────────────────────────────────────────────
echo "[studio] Starting Vite frontend — open http://127.0.0.1:5173"
echo "[studio] Press Ctrl+C to stop both."
echo ""
cd "$FRONTEND_DIR"
npm run dev &
frontend_pid=$!

wait "$frontend_pid"

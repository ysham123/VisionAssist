#!/usr/bin/env bash
set -euo pipefail

# Start VisionAssist locally on macOS/Linux
# - Creates/uses a local virtualenv in ./venv
# - Installs requirements if needed
# - Starts the Flask server with production-safe defaults
# - Opens the app in your default browser once ready

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_DIR"

PYTHON_BIN="python3"
VENV_DIR="venv"
PORT="5000"
HOST="127.0.0.1"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "[setup] Creating virtual environment in ./$VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  echo "[setup] Upgrading pip"
  "$VENV_DIR/bin/python" -m pip install -U pip
  echo "[setup] Installing dependencies"
  "$VENV_DIR/bin/pip" install -r requirements.txt
fi

export HOST="$HOST"
export PORT="$PORT"
export DEBUG="false"
export ENVIRONMENT="production"
# Adjust CORS_ORIGINS if you will open UI from a different origin (not needed for same-origin)
export CORS_ORIGINS="http://localhost:$PORT,http://127.0.0.1:$PORT"

# If another process is using the port, warn the user
if lsof -i ":$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "[warn] Port $PORT is already in use. The server may fail to bind."
fi

# Start server in background, then open browser once health responds
"$VENV_DIR/bin/python" app.py &
APP_PID=$!

cleanup() {
  echo "\n[stop] Stopping VisionAssist (pid=$APP_PID)"
  kill "$APP_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

# Wait for health endpoint (supports legacy /api/v1/health or /health)
echo "[wait] Waiting for server to become ready on http://$HOST:$PORT ..."
for i in {1..30}; do
  if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1 || curl -fsS "http://$HOST:$PORT/api/v1/health" >/dev/null 2>&1; then
    echo "[ok] Server is ready"
    if command -v open >/dev/null 2>&1; then
      open "http://$HOST:$PORT/" || true
    elif command -v xdg-open >/dev/null 2>&1; then
      xdg-open "http://$HOST:$PORT/" || true
    fi
    break
  fi
  sleep 1
  echo -n "."
done

# Attach to server process
wait "$APP_PID"

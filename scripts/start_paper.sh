#!/bin/bash
# Start V2b paper trading daemon
# Usage: ./scripts/start_paper.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_FILE="/tmp/v2b_paper.pid"
LOG_FILE="$PROJECT_DIR/logs/paper_trading.log"

# Guard: already running?
if [ -f "$PID_FILE" ]; then
    OLD_PID="$(cat "$PID_FILE")"
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "V2b paper trading already running (PID: $OLD_PID)"
        exit 0
    else
        echo "Stale PID file found — removing"
        rm -f "$PID_FILE"
    fi
fi

mkdir -p "$PROJECT_DIR/logs"

cd "$PROJECT_DIR"
nohup .venv/bin/python -m src.scheduler.main >> "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

echo "V2b paper trading started (PID: $(cat "$PID_FILE"))"
echo "Log: $LOG_FILE"

#!/bin/bash
# Stop V2b paper trading daemon
# Usage: ./scripts/stop_paper.sh

PID_FILE="/tmp/v2b_paper.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "No PID file found at $PID_FILE — daemon may not be running"
    exit 0
fi

PID="$(cat "$PID_FILE")"

if kill -0 "$PID" 2>/dev/null; then
    kill "$PID"
    rm -f "$PID_FILE"
    echo "V2b paper trading stopped (PID: $PID)"
else
    echo "Process $PID not found — cleaning up stale PID file"
    rm -f "$PID_FILE"
fi

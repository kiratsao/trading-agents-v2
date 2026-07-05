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

# Guard: systemd daemon already running? Two daemons = every scheduled job
# (and every LINE message) fires twice — refuse to double-start.
if command -v systemctl &>/dev/null && systemctl is-active --quiet trading-agents-v2 2>/dev/null; then
    echo "❌ systemd 的 trading-agents-v2 已在跑 — 不重複啟動第二個 daemon"
    echo "   (要用手動模式請先: sudo systemctl stop trading-agents-v2)"
    exit 1
fi

# Guard: any other scheduler.main process alive (e.g. started by hand)?
if pgrep -f "python.*-m src\.scheduler\.main" >/dev/null 2>&1; then
    echo "❌ 已有 src.scheduler.main process 在跑 — 不重複啟動"
    pgrep -fl "python.*-m src\.scheduler\.main"
    exit 1
fi

mkdir -p "$PROJECT_DIR/logs"

cd "$PROJECT_DIR"
nohup .venv/bin/python -m src.scheduler.main >> "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

echo "V2b paper trading started (PID: $(cat "$PID_FILE"))"
echo "Log: $LOG_FILE"

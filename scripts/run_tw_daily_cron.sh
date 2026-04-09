#!/bin/bash
# 台指期 + 美股每日自動執行腳本
# 建議 crontab: 0 14 * * 1-5
#   (台灣時間 14:00，台指期日盤已收市，美股尚未開盤)
#
# 執行順序：
#   1. 收集當天 15min K 線 (TXF + MXF)
#   2. 台指期 Swing Scaled 策略每日信號 (dry-run 記錄)
#   3. 台指期 Swing Scaled 策略每日執行 (simulation mode)
#   4. 美股動能策略每日 PnL 檢查

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$REPO_DIR/data/logs"

mkdir -p "$LOG_DIR"

cd "$REPO_DIR"

# Activate pyenv if available
if command -v pyenv &>/dev/null; then
    eval "$(pyenv init -)"
fi

echo "======================================" >> "$LOG_DIR/daily_cron.log"
echo "$(date '+%Y-%m-%d %H:%M:%S')  daily cron START" >> "$LOG_DIR/daily_cron.log"

# ── Step 1: Collect 15min K-bars ─────────────────────────────────────────────
echo "$(date '+%H:%M:%S')  [1/4] collect 15min kbars" >> "$LOG_DIR/daily_cron.log"
python scripts/collect_15min.py \
    --product ALL \
    --skip-existing \
    >> "$LOG_DIR/collect_15min.log" 2>&1 || true   # non-fatal: proceed even on API error

# ── Step 2: TW Swing Scaled — dry-run (signal logging only) ──────────────────
echo "$(date '+%H:%M:%S')  [2/4] tw swing dry-run" >> "$LOG_DIR/daily_cron.log"
python scripts/run_tw_live.py \
    --dry-run --mode daily \
    >> "$LOG_DIR/tw_dryrun.log" 2>&1 || true

# ── Step 3: TW Swing Scaled — live execution (simulation) ────────────────────
echo "$(date '+%H:%M:%S')  [3/4] tw swing live" >> "$LOG_DIR/daily_cron.log"
python scripts/run_tw_live.py \
    --mode daily --force \
    >> "$LOG_DIR/tw_swing.log" 2>&1 || true

# ── Step 4: US Equity — daily PnL check ──────────────────────────────────────
echo "$(date '+%H:%M:%S')  [4/4] us equity daily check" >> "$LOG_DIR/daily_cron.log"
python scripts/run_live.py \
    --mode daily \
    >> "$LOG_DIR/us_equity.log" 2>&1 || true

echo "$(date '+%Y-%m-%d %H:%M:%S')  daily cron END" >> "$LOG_DIR/daily_cron.log"
echo "" >> "$LOG_DIR/daily_cron.log"

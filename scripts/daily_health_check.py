"""Daily Health Check — 每天 08:00 自動跑，檢查系統狀態。

Usage: python scripts/daily_health_check.py

異常時透過 LINE 告警。加到 cron：
0 8 * * 1-5 cd ~/trading-agents-v2 && python scripts/daily_health_check.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_DATA = Path("data/MXF_Daily_Clean_2020_to_now.parquet")
_STATE = Path("data/paper_state.json")


def _send_line(msg: str) -> None:
    token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
    uid = os.environ.get("LINE_USER_ID", "")
    if not token or not uid:
        return
    import urllib.request
    payload = json.dumps({"to": uid, "messages": [{"type": "text", "text": msg}]})
    req = urllib.request.Request(
        "https://api.line.me/v2/bot/message/push",
        data=payload.encode(),
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10):
            pass
    except Exception as exc:
        logger.warning("LINE alert failed: %s", exc)


def main():
    issues = []
    today = pd.Timestamp.now(tz="Asia/Taipei").date()

    # 1. Parquet freshness
    if _DATA.exists():
        df = pd.read_parquet(_DATA)
        df.index = pd.to_datetime(df.index)
        latest = df.index[-1].date()
        gap = (today - latest).days
        if gap > 3:
            issues.append(f"Parquet 過期: latest={latest}, gap={gap}天")
        logger.info("Parquet: %d bars, latest=%s, gap=%dd", len(df), latest, gap)
    else:
        issues.append(f"Parquet 不存在: {_DATA}")

    # 2. State sanity
    if _STATE.exists():
        try:
            raw = json.loads(_STATE.read_text())
            s = raw.get("state", {})
            pos = s.get("position", 0)
            eq = s.get("equity", 0)
            entry = s.get("entry_price")
            logger.info("State: position=%d, equity=%.0f, entry=%s",
                        pos, eq, entry)
            if pos > 0 and not entry:
                issues.append(f"State 不一致: position={pos} 但 entry_price=None")
            if pos < 0:
                issues.append(f"State 異常: position={pos} (負數)")
            if eq <= 0:
                issues.append(f"State 異常: equity={eq} (<=0)")
        except Exception as exc:
            issues.append(f"State 讀取失敗: {exc}")
    else:
        logger.info("State file not found (new deployment?)")

    # 3. Report
    if issues:
        msg = "🔴 Daily Health Check 異常:\n" + "\n".join(f"• {i}" for i in issues)
        logger.error(msg)
        _send_line(msg)
        sys.exit(1)
    else:
        logger.info("✅ Daily health check passed")


if __name__ == "__main__":
    main()

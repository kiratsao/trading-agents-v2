"""LINE Messaging API 推播測試腳本。

Usage
-----
    python scripts/test_line_notify.py
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.monitor.notifier import Notifier


def main() -> None:
    n = Notifier()

    if not n._line_configured:
        print("[ERROR] LINE 未設定。請確認 .env 中已設定：")
        print("  LINE_CHANNEL_ACCESS_TOKEN=...")
        print("  LINE_USER_ID=...")
        sys.exit(1)

    now = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=8)))
    ts = now.strftime("%Y-%m-%d %H:%M:%S")

    message = (
        f"🤖 Trading Bot 連線測試\n"
        f"台指期自動交易系統已上線\n"
        f"策略：V2b EMA50/150 Long-Only 2口\n"
        f"狀態：Simulation 模式\n"
        f"時間：{ts}"
    )

    print(f"Sending LINE push to {n._line_user[:8]}… (user_id redacted)")
    print(f"Message:\n{message}\n")

    ok = n.send_line(message, level="INFO")

    if ok:
        print("[OK] LINE 推播成功！請確認手機是否收到訊息。")
    else:
        print("[FAIL] LINE 推播失敗。請確認 Channel Access Token 和 User ID 是否正確。")
        sys.exit(1)


if __name__ == "__main__":
    main()

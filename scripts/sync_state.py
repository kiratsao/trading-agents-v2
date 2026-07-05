"""State/broker 一鍵對帳 CLI — 取代「手改 state JSON」的 SOP。

用法（在 GCP 上、daemon 可以不停）:
    python scripts/sync_state.py            # 只顯示 state vs broker 差異（dry-run）
    python scripts/sync_state.py --apply    # 以 broker 為準寫入 state（先自動備份）

daemon 本身已會在啟動時與每天 14:30 信號前自動對帳（orchestrator.
reconcile_state_with_broker）；這支 CLI 給「想立刻同步/立刻確認」的場景用。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO)
logger = logging.getLogger("sync_state")

_CONFIG = "config/accounts.yaml"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="State/broker 對帳")
    ap.add_argument("--apply", action="store_true", help="以 broker 為準寫入 state")
    ap.add_argument("--account", default="mxf_aggressive", help="帳戶名 (accounts.yaml)")
    ap.add_argument("--config", default=_CONFIG)
    args = ap.parse_args(argv)

    import yaml

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    acc = cfg.get("accounts", {}).get(args.account)
    if acc is None:
        print(f"❌ 帳戶 {args.account} 不在 {args.config}")
        return 2
    product = acc.get("product", "MXF")

    from src.scheduler.main import _build_broker
    from src.scheduler.orchestrator import (
        _query_broker_avg_price,
        _query_live_equity,
        _read_broker_long,
    )
    from src.state.state_manager import StateManager

    state_mgr = StateManager(
        path=f"data/state_{args.account}.json",
        initial_equity=float(acc.get("equity", 350_000)),
    )
    state = state_mgr.load()

    broker = _build_broker(live=True)
    if broker is None:
        print("❌ 無法建立 broker（檢查 SHIOAJI_* 環境變數）")
        return 2

    try:
        actual = _read_broker_long(broker, product)
        if actual is None:
            print("❌ broker 部位讀取失敗")
            return 2
        avg = _query_broker_avg_price(broker, product, actual) if actual > 0 else None
        equity, equity_src = _query_live_equity(broker, state.equity)

        print("━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"帳戶: {args.account} ({product})")
        print(f"{'':14}{'state':>14}{'broker':>14}")
        print(f"{'position':<14}{state.position:>14}{actual:>14}")
        print(
            f"{'entry_price':<14}{state.entry_price or 0:>14,.0f}"
            f"{avg or 0:>14,.0f}"
        )
        print(f"{'equity':<14}{state.equity:>14,.0f}{equity:>14,.0f} ({equity_src})")
        print("━━━━━━━━━━━━━━━━━━━━━━━━")

        in_sync = actual == state.position and (
            actual == 0 or avg is None or abs((state.entry_price or 0) - avg) < 1
        )
        if in_sync:
            print("✅ state 與 broker 一致")

        if not args.apply:
            if not in_sync:
                print("👉 加 --apply 以 broker 為準寫入 state")
            return 0 if in_sync else 1

        # ── apply: broker 為準 ────────────────────────────────────────
        if actual == 0:
            state.position = 0
            state.contracts = 0
            state.entry_price = None
            state.highest_high = None
            state.pyramided = False
        else:
            from src.utils.tw_time import today_taipei

            if state.position == 0:
                state.entry_date = today_taipei().isoformat()
                state.pyramided = False
            state.position = actual
            state.contracts = actual
            if avg is not None and avg > 0:
                state.entry_price = avg
            if state.entry_price is None:
                print("⚠️ broker 均價不可讀且 state 無 entry_price — 未寫入；"
                      "請稍後重試（均價通常幾秒後可讀）")
                return 2
            if state.highest_high is None or state.highest_high < state.entry_price:
                state.highest_high = state.entry_price
        if equity_src == "即時" and equity > 0:
            state.equity = equity
        state_mgr.save(state)
        print(f"✅ 已寫入 state: position={state.position} entry={state.entry_price} "
              f"equity={state.equity:,.0f}")
        return 0
    finally:
        try:
            broker.logout()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())

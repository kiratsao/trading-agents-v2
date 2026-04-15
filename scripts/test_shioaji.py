"""Connection and smoke test for ShioajiAdapter (simulation mode).

Usage
-----
    python scripts/test_shioaji.py

Requires SHIOAJI_API_KEY and SHIOAJI_SECRET_KEY in .env (or environment).
Runs entirely in simulation mode — no real orders are placed.

Tests
-----
1. Login (simulation=True)
2. TXF near-month contract info
3. Snapshot (即時報價)
4. Account margin / equity
5. Open positions
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from tw_futures.executor.shioaji_adapter import ExecutionError, ShioajiAdapter  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_shioaji")

SEP = "=" * 60


def _section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def main() -> None:
    if not os.environ.get("SHIOAJI_API_KEY", "") or not os.environ.get("SHIOAJI_SECRET_KEY", ""):
        print("\n[ERROR] SHIOAJI_API_KEY and SHIOAJI_SECRET_KEY must be set in .env\n")
        sys.exit(1)

    # ── 1. Login ──────────────────────────────────────────────────────
    _section("1. Login (simulation=True)")
    try:
        adapter = ShioajiAdapter(
            api_key=os.environ.get("SHIOAJI_API_KEY", ""),
            secret_key=os.environ.get("SHIOAJI_SECRET_KEY", ""),
            simulation=True,
        )
        print("  OK — logged in successfully")
    except ExecutionError as exc:
        print(f"  FAIL — {exc}")
        sys.exit(1)

    # ── 2. TXF near-month contract ────────────────────────────────────
    _section("2. TXF Near-Month Contract")
    try:
        contract = adapter.get_contract("TXF")
        print(f"  code           : {contract.code}")
        print(f"  name           : {contract.name}")
        print(f"  delivery_month : {contract.delivery_month}")
        print(f"  delivery_date  : {contract.delivery_date}")
        print(f"  multiplier     : {contract.multiplier}")
        print(f"  limit_up       : {contract.limit_up}")
        print(f"  limit_down     : {contract.limit_down}")
        print(f"  reference      : {contract.reference}")
    except ExecutionError as exc:
        print(f"  FAIL — {exc}")

    # Also check MXF
    _section("2b. MXF Near-Month Contract")
    try:
        mxf = adapter.get_contract("MXF")
        print(f"  code           : {mxf.code}")
        print(f"  delivery_month : {mxf.delivery_month}")
        print(f"  delivery_date  : {mxf.delivery_date}")
    except ExecutionError as exc:
        print(f"  FAIL — {exc}")

    # ── 3. Snapshot (即時報價) ─────────────────────────────────────────
    _section("3. TXF Snapshot (即時報價)")
    try:
        snap = adapter.get_snapshots("TXF")
        import datetime

        ts_dt = datetime.datetime.fromtimestamp(snap["ts"] / 1e9, tz=datetime.UTC)
        ts_local = ts_dt.astimezone()
        print(f"  code         : {snap['code']}")
        print(f"  timestamp    : {ts_local.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"  open         : {snap['open']:,.0f}")
        print(f"  high         : {snap['high']:,.0f}")
        print(f"  low          : {snap['low']:,.0f}")
        print(f"  close (last) : {snap['close']:,.0f}")
        print(f"  volume       : {snap['volume']:,}")
        print(f"  total_volume : {snap['total_volume']:,}")
        print(f"  bid          : {snap['buy_price']:,.0f}")
        print(f"  ask          : {snap['sell_price']:,.0f}")
        print(f"  change       : {snap['change_price']:+,.0f}  ({snap['change_rate']:+.2f}%)")
    except ExecutionError as exc:
        print(f"  FAIL — {exc}")

    # ── 4. Account margin / equity ────────────────────────────────────
    _section("4. Account Info (Margin)")
    try:
        acct = adapter.get_account()
        print(f"  equity           : TWD {acct['equity']:>14,.0f}")
        print(f"  margin_used      : TWD {acct['margin_used']:>14,.0f}")
        print(f"  available_margin : TWD {acct['available_margin']:>14,.0f}")
        print(f"  unrealized_pnl   : TWD {acct['unrealized_pnl']:>+14,.0f}")
    except ExecutionError as exc:
        print(f"  FAIL — {exc}")

    # ── 5. Open positions ─────────────────────────────────────────────
    _section("5. Open Positions")
    try:
        positions = adapter.get_positions()
        if not positions:
            print("  (no open positions)")
        else:
            print(f"  {'Code':<12} {'Dir':<6} {'Qty':>5} {'AvgPx':>8} {'LastPx':>8} {'PnL':>10}")
            print("  " + "-" * 56)
            for pos in positions:
                print(
                    f"  {pos['code']:<12} {pos['direction']:<6} {pos['contracts']:>5} "
                    f"{pos['avg_price']:>8,.0f} {pos['last_price']:>8,.0f} "
                    f"{pos['unrealized_pnl']:>+10,.0f}"
                )
    except ExecutionError as exc:
        print(f"  FAIL — {exc}")

    # ── Done ──────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  All tests complete.")
    print(SEP)
    adapter.logout()


if __name__ == "__main__":
    main()

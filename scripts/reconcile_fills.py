"""Pull REAL fills + realized P&L from Shioaji for the 7/22→7/29 round trip.

Run on the GCP VM (needs .env credentials):
    python3 scripts/reconcile_fills.py
    python3 scripts/reconcile_fills.py --begin 2026-07-22 --end 2026-07-29

Prints every profit/loss record (all attributes, schema-agnostic), then the
weighted-average entry/cover fills and total realized P&L, and derives:
  1. book-vs-actual entry diff (book entry 44,957 == the 7/21 NIGHT close —
     the 2026/07/22 盤後 row — i.e. a night-quote artifact, not a real fill;
     the actual 7/22 15:05 window opened at 44,471);
  2. the 7/28-bug increment: (7/28 night open 41,797 − actual exit) × qty × 50
     — what a normal 7/28 15:05 stop-out would have changed;
  3. margin-staleness verdict for the 7/28 14:30 equity read (756,963):
     implied mark = actual_entry − (1,200,938 − 756,963) / (qty × 50);
     ≈ 41,608 (true close) → margin was FRESH; ≫ that → STALE.
Read-only: no orders, no state writes.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TICK = 50.0

# Locked public-data reference values (TAIFEX, verified 2026-07-29):
NIGHT_OPEN_0728 = 41_797.0     # 7/28 15:00 night open (2026/07/29 盤後 row, 202608)
BOOK_ENTRY = 44_957.0          # what state.json recorded (= 7/21 night close)
EQUITY_PRE_ENTRY = 1_200_938.0
EQUITY_1430_0728 = 756_963.0
TRUE_CLOSE_0728 = 41_608.0


def _dump(obj) -> dict:
    for attr in ("dict", "__dict__"):
        try:
            d = getattr(obj, attr)
            return d() if callable(d) else dict(d)
        except Exception:
            continue
    return {"repr": repr(obj)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--begin", default="2026-07-22")
    ap.add_argument("--end", default="2026-07-29")
    args = ap.parse_args(argv)

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    api_key = os.environ.get("SHIOAJI_API_KEY", "")
    secret_key = os.environ.get("SHIOAJI_SECRET_KEY", "")
    if not (api_key and secret_key):
        print("🔴 SHIOAJI_API_KEY / SHIOAJI_SECRET_KEY not set — run on the VM with .env")
        return 1

    from tw_futures.executor.shioaji_adapter import ShioajiAdapter
    adapter = ShioajiAdapter(
        api_key=api_key, secret_key=secret_key, simulation=False,
        cert_path=os.environ.get("SHIOAJI_CERT_PATH") or None,
        cert_password=os.environ.get("SHIOAJI_CERT_PASSWORD") or None,
        person_id=os.environ.get("SHIOAJI_PERSON_ID") or None,
    )
    try:
        api = adapter._api
        acct = getattr(api, "futopt_account", None)
        print(f"account: {getattr(acct, 'account_id', acct)}")
        try:
            pl = api.list_profit_loss(acct, begin=args.begin, end=args.end)
        except TypeError:
            pl = api.list_profit_loss(acct, args.begin, args.end)

        if not pl:
            print("(no profit/loss records in range — check dates/account)")
            return 1

        print(f"\n── raw profit/loss records {args.begin}→{args.end} ──")
        rows = []
        for r in pl:
            d = _dump(r)
            print(d)
            rows.append(d)
            det_id = d.get("id")
            if det_id is not None:
                try:
                    for det in api.list_profit_loss_detail(acct, det_id):
                        print("   detail:", _dump(det))
                except Exception as exc:
                    print(f"   (detail unavailable: {exc})")

        def _f(d, *keys, default=0.0):
            for k in keys:
                if k in d and d[k] not in (None, ""):
                    return float(d[k])
            return default

        qty = sum(_f(d, "quantity", "qty", default=0) for d in rows)
        if qty <= 0:
            print("🔴 cannot aggregate: no quantity fields — read raw above manually")
            return 1
        entry_avg = sum(_f(d, "entry_price", "buy_price") * _f(d, "quantity", "qty")
                        for d in rows) / qty
        cover_avg = sum(_f(d, "cover_price", "sell_price") * _f(d, "quantity", "qty")
                        for d in rows) / qty
        pnl_total = sum(_f(d, "pnl") for d in rows)

        print("\n── aggregate ──")
        print(f"qty={qty:.0f}口  entry_avg={entry_avg:,.1f}  cover_avg={cover_avg:,.1f}  "
              f"realized(broker)={pnl_total:+,.0f}")

        print("\n── derived ──")
        print(f"1) 帳面 entry {BOOK_ENTRY:,.0f} vs 實際 {entry_avg:,.1f} → "
              f"差 {BOOK_ENTRY - entry_avg:+,.1f}pt "
              f"(帳面值 == 7/21夜盤收盤,為夜盤報價汙染)")
        inc = (NIGHT_OPEN_0728 - cover_avg) * qty * TICK
        print(f"2) 7/28 bug 增量: (夜盤開 {NIGHT_OPEN_0728:,.0f} − 實際出場 "
              f"{cover_avg:,.1f}) × {qty:.0f}口 × 50 = {inc:+,.0f} NTD "
              f"({'7/28 準時出場可少虧' if inc > 0 else '實際 7/29 出場反而較優'})")
        implied = entry_avg - (EQUITY_PRE_ENTRY - EQUITY_1430_0728) / (qty * TICK)
        print(f"3) 7/28 14:30 margin 隱含 mark = {implied:,.1f} "
              f"(真值 {TRUE_CLOSE_0728:,.0f}, 偏離 {implied - TRUE_CLOSE_0728:+,.1f}pt → "
              f"{'FRESH' if abs(implied - TRUE_CLOSE_0728) <= 300 else 'STALE'})")
        print(f"4) equity 軌跡核對: {EQUITY_PRE_ENTRY:,.0f} + realized {pnl_total:+,.0f} "
              f"= {EQUITY_PRE_ENTRY + pnl_total:,.0f} (實際終值 650,708 — 差額即"
              f"費用/稅或其他現金流)")
        return 0
    finally:
        try:
            adapter.logout()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())

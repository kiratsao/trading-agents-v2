"""Reconcile the tail of the MXF day-K parquet against TAIFEX 一般 (day) values.

Fixes two things in one verified pass, using the validated TAIFEX day-session
source (the same one the cross-source oracle uses):
  * ADD    — a missing recent trading day (gap, e.g. 7/17→7/20 stalled backfill)
  * CORRECT — a present bar whose close diverges from TAIFEX 一般 by > threshold
             (e.g. a Shioaji night value slipped past the ±500pt validator on a
             low day/night-gap day)

Day-vs-day micro-differences (Shioaji last tick vs TAIFEX official close, a few
points) are left as OK — only genuine divergences are corrected.

Dry-run by default; pass --apply to write (backs up the parquet first, atomic
write, then re-reads to verify). Run with the daemon STOPPED.

Usage:
    python scripts/reconcile_recent_bars.py                     # dry-run, last 10 trading days
    python scripts/reconcile_recent_bars.py --from 2026-07-16   # dry-run from a date
    python scripts/reconcile_recent_bars.py --from 2026-07-16 --apply
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data.daily_updater import _PRIMARY_PARQUET, _last_trading_day
from src.data.tw_holidays import is_trading_day
from src.data.validation import fetch_taifex_day_session_range
from src.utils.tw_time import today_taipei

# Correct a present bar only when it diverges from TAIFEX 一般 by more than this
# (points). Below it is normal Shioaji-vs-TAIFEX day-close basis noise.
CORRECT_THRESHOLD = 10.0
_COLS = ["open", "high", "low", "close", "volume"]


def _trading_days(start: date, end: date) -> list[date]:
    out, d = [], start
    while d <= end:
        if is_trading_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out


def reconcile(parquet_path: Path, start: date, end: date, apply: bool) -> int:
    df = pd.read_parquet(parquet_path)
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df = df.sort_index()

    tx = fetch_taifex_day_session_range(start, end)
    if tx is None or tx.empty:
        print("🔴 TAIFEX 一般 fetch returned nothing for the range — aborting.")
        return 1

    days = _trading_days(start, end)
    print(f"\nRange: {start} → {end}  ({len(days)} trading days)  parquet={parquet_path.name}")
    print(f"{'date':12}{'parquet':>10}{'TAIFEX一般':>12}{'diff':>9}   action")
    print("-" * 55)

    plan: list[tuple[pd.Timestamp, str]] = []  # (ts, action)
    for d in days:
        ts = pd.Timestamp(d)
        t_present = ts in tx.index
        p_present = ts in df.index
        t_close = float(tx.loc[ts, "close"]) if t_present else None
        p_close = float(df.loc[ts, "close"]) if p_present else None

        if not t_present:
            action = "SKIP (no TAIFEX day bar yet)"
        elif not p_present:
            action = "ADD"
            plan.append((ts, "ADD"))
        elif abs(p_close - t_close) > CORRECT_THRESHOLD:
            action = f"CORRECT (Δ{p_close - t_close:+.0f})"
            plan.append((ts, "CORRECT"))
        else:
            action = "ok"

        pc = f"{p_close:,.0f}" if p_close is not None else "—"
        tc = f"{t_close:,.0f}" if t_close is not None else "—"
        diff = f"{p_close - t_close:+.0f}" if (p_close is not None and t_close is not None) else "—"
        print(f"{str(d):12}{pc:>10}{tc:>12}{diff:>9}   {action}")

    n_add = sum(1 for _, a in plan if a == "ADD")
    n_fix = sum(1 for _, a in plan if a == "CORRECT")
    print("-" * 55)
    print(f"Plan: {n_add} ADD, {n_fix} CORRECT, {len(days) - len(plan)} unchanged")

    if not plan:
        print("Nothing to do — parquet tail already matches TAIFEX 一般.")
        return 0
    if not apply:
        print("\nDRY-RUN — re-run with --apply to write (daemon must be stopped).")
        return 0

    # ── apply ────────────────────────────────────────────────────────────────
    backup = parquet_path.with_name(
        f"{parquet_path.stem}.reconcile_backup_{today_taipei():%Y%m%d}.parquet"
    )
    shutil.copy2(parquet_path, backup)
    print(f"\nBacked up → {backup.name}")

    for ts, _action in plan:
        for col in _COLS:
            if col in tx.columns:
                df.loc[ts, col] = tx.loc[ts, col]
    df = df.sort_index()
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    tmp = parquet_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=True)
    tmp.replace(parquet_path)

    # verify
    v = pd.read_parquet(parquet_path)
    v.index = pd.to_datetime(v.index)
    ok = True
    for ts, _action in plan:
        got = float(v.loc[ts, "close"])
        want = float(tx.loc[ts, "close"])
        if abs(got - want) > 0.5:
            ok = False
            print(f"🔴 verify FAILED {ts.date()}: got {got:,.0f} want {want:,.0f}")
    print(
        f"✅ applied {n_add} ADD + {n_fix} CORRECT; parquet latest={v.index[-1].date()} "
        f"close={float(v['close'].iloc[-1]):,.0f}"
        if ok
        else "🔴 verification mismatch — restore from backup and investigate"
    )
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", default=str(_PRIMARY_PARQUET))
    ap.add_argument("--from", dest="frm", default=None,
                    help="start date YYYY-MM-DD (default: 10 trading days back)")
    ap.add_argument("--to", dest="to", default=None,
                    help="end date YYYY-MM-DD (default: last completed trading day)")
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry-run)")
    args = ap.parse_args(argv)

    end = date.fromisoformat(args.to) if args.to else _last_trading_day(today_taipei())
    if args.frm:
        start = date.fromisoformat(args.frm)
    else:
        start = end
        n = 0
        while n < 10:
            start -= timedelta(days=1)
            if is_trading_day(start):
                n += 1

    return reconcile(Path(args.parquet), start, end, args.apply)


if __name__ == "__main__":
    raise SystemExit(main())

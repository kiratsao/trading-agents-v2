"""Reconcile a NARROW, explicit window of the MXF day-K parquet against TAIFEX
一般 (day) — spot-guarded so it can never overwrite a correct value with a worse
one.

Every candidate change is gated on the TWSE spot index (^TWII, day session):
  * ADD (missing day)  — apply only if |TAIFEX一般 − spot| ≤ SPOT_ADD_TOL, else
    SKIP (the re-fetched TAIFEX value looks like a night/anomaly, not the day).
  * CORRECT (present, differs) — apply only if the new value is STRICTLY closer
    to spot than the existing one, else SKIP.
If spot cannot be fetched, the run ABORTS (never guesses).

Lesson: a prior version reconciled a wide default window on a loose "new ≠ old"
threshold and overwrote correct day values with re-fetched night values. The
window is now REQUIRED (--from / --to) and every write must beat spot.

Dry-run by default; --apply writes (backup + atomic + re-read verify). Run with
the daemon STOPPED.

Usage:
    python scripts/reconcile_recent_bars.py --from 2026-07-16 --to 2026-07-20
    python scripts/reconcile_recent_bars.py --from 2026-07-16 --to 2026-07-20 --apply
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data.daily_updater import _PRIMARY_PARQUET
from src.data.tw_holidays import is_trading_day
from src.data.validation import fetch_taifex_day_session_range
from src.utils.tw_time import today_taipei

# A present bar is a CORRECT *candidate* only when it diverges from TAIFEX 一般
# by more than this (points); below it is normal basis noise. The spot-guard
# then decides whether the change actually happens.
CORRECT_THRESHOLD = 10.0
# ADD a missing day only when the re-fetched TAIFEX 一般 is within this of spot
# (a night/anomalous re-fetch would be far from the day-session spot close).
SPOT_ADD_TOL = 400.0
_COLS = ["open", "high", "low", "close", "volume"]


def _trading_days(start: date, end: date) -> list[date]:
    out, d = [], start
    while d <= end:
        if is_trading_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out


def _fetch_spot(start: date, end: date) -> dict | None:
    """TWSE spot index (^TWII) day-session closes, {Timestamp: close}, or None."""
    try:
        import yfinance as yf

        sp = yf.download(
            "^TWII", start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            progress=False, auto_adjust=True,
        )
    except Exception as exc:
        print(f"🔴 spot (^TWII) fetch failed: {exc}")
        return None
    if sp is None or sp.empty:
        return None
    if isinstance(sp.columns, pd.MultiIndex):
        sp.columns = [c[0] for c in sp.columns]
    idx = pd.to_datetime(sp.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    return {ts.normalize(): float(v) for ts, v in zip(idx, sp["Close"])}


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
    spot = _fetch_spot(start, end)
    if not spot:
        print("🔴 spot (^TWII) unavailable — refusing to reconcile without the "
              "spot-guard (install yfinance or run where it is available).")
        return 1

    days = _trading_days(start, end)
    print(f"\nRange: {start} → {end}  ({len(days)} trading days)  parquet={parquet_path.name}")
    print(f"{'date':12}{'parquet':>9}{'TAIFEX一般':>11}{'spot':>9}"
          f"{'|old-sp|':>9}{'|new-sp|':>9}   action")
    print("-" * 74)

    plan: list[tuple[pd.Timestamp, str]] = []
    for d in days:
        ts = pd.Timestamp(d)
        t_close = float(tx.loc[ts, "close"]) if ts in tx.index else None
        p_close = float(df.loc[ts, "close"]) if ts in df.index else None
        s_close = spot.get(ts.normalize())

        old_sp = abs(p_close - s_close) if (p_close is not None and s_close is not None) else None
        new_sp = abs(t_close - s_close) if (t_close is not None and s_close is not None) else None

        if t_close is None:
            action = "SKIP (no TAIFEX day bar)"
        elif s_close is None:
            action = "SKIP (no spot — cannot guard)"
        elif p_close is None:
            # missing day → ADD, guarded on spot proximity
            if new_sp <= SPOT_ADD_TOL:
                action = "ADD"
                plan.append((ts, "ADD"))
            else:
                action = f"SKIP ADD (new {new_sp:.0f}>{SPOT_ADD_TOL:.0f} from spot — not day?)"
        elif abs(p_close - t_close) <= CORRECT_THRESHOLD:
            action = "ok"
        elif new_sp < old_sp:
            action = f"CORRECT (spot: {old_sp:.0f}→{new_sp:.0f})"
            plan.append((ts, "CORRECT"))
        else:
            action = f"SKIP (old closer to spot {old_sp:.0f}≤{new_sp:.0f})"

        def _f(x):
            return f"{x:,.0f}" if x is not None else "—"
        print(f"{str(d):12}{_f(p_close):>9}{_f(t_close):>11}{_f(s_close):>9}"
              f"{_f(old_sp):>9}{_f(new_sp):>9}   {action}")

    n_add = sum(1 for _, a in plan if a == "ADD")
    n_fix = sum(1 for _, a in plan if a == "CORRECT")
    print("-" * 74)
    print(f"Plan: {n_add} ADD, {n_fix} CORRECT, {len(days) - len(plan)} unchanged/skipped")

    if not plan:
        print("Nothing to do.")
        return 0
    if not apply:
        print("\nDRY-RUN — re-run with --apply to write (daemon must be stopped).")
        return 0

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

    v = pd.read_parquet(parquet_path)
    v.index = pd.to_datetime(v.index)
    ok = all(abs(float(v.loc[ts, "close"]) - float(tx.loc[ts, "close"])) < 0.5 for ts, _ in plan)
    print(
        f"✅ applied {n_add} ADD + {n_fix} CORRECT; latest={v.index[-1].date()} "
        f"close={float(v['close'].iloc[-1]):,.0f}"
        if ok else "🔴 verification mismatch — restore from backup and investigate"
    )
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", default=str(_PRIMARY_PARQUET))
    ap.add_argument("--from", dest="frm", required=True,
                    help="start date YYYY-MM-DD (REQUIRED — keep the window narrow)")
    ap.add_argument("--to", dest="to", default=None,
                    help="end date YYYY-MM-DD (default: --from)")
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry-run)")
    args = ap.parse_args(argv)
    start = date.fromisoformat(args.frm)
    end = date.fromisoformat(args.to) if args.to else start
    return reconcile(Path(args.parquet), start, end, args.apply)


if __name__ == "__main__":
    raise SystemExit(main())

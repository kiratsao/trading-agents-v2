"""Dry-run scan for night-session pollution in the MXF day-K parquet.

For each stored bar it compares the parquet close against the night-proof spot
index (^TWII) and the TAIFEX day (一般) / night (盤後) values, and flags any date
whose stored value is not the day value — classifying it NIGHT (matches 盤後) or
ANOMALY. NEVER writes; output is a list for the operator to review, then fix with
scripts/reconcile_recent_bars.py.

Requires the parse to be healthy (TAIFEX 一般 must return day values) — run only
after confirming that, or the reference itself is night and the scan is blind.

Usage:
    python scripts/scan_pollution.py --from 2026-01-01 --to 2026-07-31
    python scripts/scan_pollution.py                       # default: full parquet
"""
from __future__ import annotations

import argparse
import io
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from scripts.init_data import _download_taifex_csv
from src.data.daily_updater import _PRIMARY_PARQUET
from src.data.spot_ref import fetch_spot_range
from src.data.validation import fetch_taifex_day_session_range

TOL = 20.0  # |stored − TAIFEX day| beyond this = stored is not the day value


def _taifex_night(start: date, end: date) -> dict:
    """{Timestamp: 盤後 (night) close} for the nearest monthly contract."""
    out: dict[pd.Timestamp, float] = {}
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        raw = _download_taifex_csv(y, m, "MTX")
        lines = raw.strip().split("\n")
        df = pd.read_csv(io.StringIO("\n".join(x.rstrip(", ") for x in lines)), index_col=False)
        df["d"] = pd.to_datetime(df["交易日期"].astype(str).str.strip(), errors="coerce")
        df = df.dropna(subset=["d"])
        df["e"] = df["到期月份(週別)"].astype(str).str.strip()
        df = df[df["e"].str.match(r"^\d{6}$")]
        df["c"] = pd.to_numeric(df["收盤價"].astype(str).str.replace(",", ""), errors="coerce")
        df["s"] = df["交易時段"].astype(str).str.strip()
        for d, g in df.groupby("d"):
            nm = sorted(g["e"].unique())[0]
            x = g[(g["e"] == nm) & (g["s"] == "盤後")]["c"]
            if len(x):
                out[pd.Timestamp(d)] = float(x.iloc[0])
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def scan(parquet_path: Path, start: date, end: date) -> list[dict]:
    df = pd.read_parquet(parquet_path)
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df = df.sort_index()
    df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
    if df.empty:
        print("no bars in range")
        return []

    day = fetch_taifex_day_session_range(start, end)
    night = _taifex_night(start, end)
    spot = fetch_spot_range(start, end)

    flagged: list[dict] = []
    checked = 0
    for ts in df.index:
        d_close = float(day.loc[ts, "close"]) if (day is not None and ts in day.index) else None
        if d_close is None:
            continue
        checked += 1
        p = float(df.loc[ts, "close"])
        if abs(p - d_close) <= TOL:
            continue  # stored == day value → clean
        n_close = night.get(ts)
        s_close = spot.get(ts.date())
        cls = "NIGHT" if (n_close is not None and abs(p - n_close) <= TOL) else "ANOMALY"
        flagged.append({
            "date": ts.date(), "stored": p, "taifex_day": d_close,
            "taifex_night": n_close, "spot": s_close, "class": cls,
        })

    print(f"\nScanned {checked} bars {start}→{end}; flagged {len(flagged)}:")
    if flagged:
        print(f"{'date':12}{'stored':>9}{'day(一般)':>10}{'night(盤後)':>12}{'spot':>9}   class")
        for f in flagged:
            n = f"{f['taifex_night']:,.0f}" if f["taifex_night"] is not None else "—"
            s = f"{f['spot']:,.0f}" if f["spot"] is not None else "—"
            print(f"{str(f['date']):12}{f['stored']:>9,.0f}{f['taifex_day']:>10,.0f}"
                  f"{n:>12}{s:>9}   {f['class']}")
        print("\n→ review, then fix with: scripts/reconcile_recent_bars.py "
              f"--from {flagged[0]['date']} --to {flagged[-1]['date']}")
    else:
        print("  ✅ no night pollution / anomalies found.")
    return flagged


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", default=str(_PRIMARY_PARQUET))
    ap.add_argument("--from", dest="frm", default=None)
    ap.add_argument("--to", dest="to", default=None)
    args = ap.parse_args(argv)

    pq = Path(args.parquet)
    df = pd.read_parquet(pq)
    df.index = pd.to_datetime(df.index)
    start = date.fromisoformat(args.frm) if args.frm else df.index.min().date()
    end = date.fromisoformat(args.to) if args.to else df.index.max().date()
    scan(pq, start, end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

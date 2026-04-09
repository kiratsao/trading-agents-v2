"""Smoke test: fetch TX daily data for the last 60 days, save to SQLite, print summary.

Usage
-----
    python scripts/test_tw_fetch.py [--product TX|MTX] [--days N]
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

# Make repo root importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from core.data.store import load_ohlcv, save_ohlcv_bulk
from tw_futures.data.fetcher import TaifexFetcher, TaifexFetchError
from tw_futures.data.rollover import (
    get_expiry_date,
    get_front_month,
    get_next_contract,
    should_rollover,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_tw_fetch")

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "db" / "market_data.db"


def main() -> None:
    parser = argparse.ArgumentParser(description="Test TAIFEX data fetch")
    parser.add_argument("--product", default="TX", choices=["TX", "MTX"])
    parser.add_argument("--days", type=int, default=60, help="Number of calendar days to fetch")
    args = parser.parse_args()

    end_date = date.today()
    start_date = end_date - timedelta(days=args.days)

    print("=" * 68)
    print(f"  TAIFEX Data Fetch Test — {args.product}")
    print("=" * 68)
    print(f"  Range : {start_date} → {end_date}  ({args.days} calendar days)")
    print(f"  DB    : {DB_PATH}")
    print()

    # ── 1. Rollover helper smoke test ─────────────────────────────────
    today = date.today()
    front = get_front_month(today)
    expiry = get_expiry_date(int(front[:4]), int(front[4:]))
    next_c = get_next_contract(front)
    rolling = should_rollover(today, front)

    print("── Rollover helpers ─────────────────────────────────────────")
    print(f"  Today               : {today}")
    print(f"  Front-month contract: {front}")
    print(f"  Expiry date         : {expiry}")
    print(f"  Next contract       : {next_c}")
    print(f"  Should rollover now : {rolling}")
    print()

    # ── 2. Fetch daily OHLCV ─────────────────────────────────────────
    print("── Fetching daily OHLCV ─────────────────────────────────────")
    fetcher = TaifexFetcher(request_delay=1.0)

    try:
        df = fetcher.fetch_daily(
            args.product,
            start_date.isoformat(),
            end_date.isoformat(),
        )
    except TaifexFetchError as exc:
        print(f"  ERROR: {exc}")
        sys.exit(1)

    if df.empty:
        print("  No data returned — market may have been closed for the entire range.")
        sys.exit(0)

    print(f"  Rows fetched        : {len(df)}")
    print(f"  Date range          : {df.index.min().date()} → {df.index.max().date()}")
    print(f"  Contracts seen      : {sorted(df['contract'].unique())}")
    print()

    # ── 3. Save to SQLite ─────────────────────────────────────────────
    print("── Saving to SQLite ─────────────────────────────────────────")
    # save_ohlcv_bulk expects {symbol: DataFrame} with UTC DatetimeIndex
    df_store = df.copy()
    df_store.index = df_store.index.tz_convert("UTC")

    try:
        result = save_ohlcv_bulk({args.product: df_store}, db_path=str(DB_PATH))
        saved = result.get(args.product, 0)
        print(f"  Saved {saved} rows to {DB_PATH.name}")
    except Exception as exc:
        print(f"  WARNING: SQLite save failed: {exc}")
        print("  (Continuing — data is still available in memory)")

    # ── 4. Print first 10 rows ────────────────────────────────────────
    print()
    print("── First 10 rows ────────────────────────────────────────────")
    display = df.copy()
    display.index = display.index.strftime("%Y-%m-%d")
    display["volume"] = display["volume"].map("{:,.0f}".format)
    display["oi"] = display["oi"].map(lambda x: f"{x:,.0f}" if pd.notna(x) else "-")
    print(
        display[["open", "high", "low", "close", "volume", "oi", "contract"]].head(10).to_string()
    )
    print()

    # ── 5. Summary statistics ─────────────────────────────────────────
    print("── Summary statistics ───────────────────────────────────────")
    close_series = df["close"].dropna()
    first_close = close_series.iloc[0]
    last_close = close_series.iloc[-1]
    period_return = (last_close / first_close - 1) * 100

    print(f"  Total rows          : {len(df)}")
    print(f"  First close         : {first_close:,.0f}")
    print(f"  Last close          : {last_close:,.0f}")
    print(f"  Period return       : {period_return:+.2f}%")
    print(f"  Avg daily volume    : {df['volume'].mean():,.0f}")
    print(f"  Max daily volume    : {df['volume'].max():,.0f}  ({df['volume'].idxmax().date()})")
    print()

    # ── 6. Verify reload from SQLite ─────────────────────────────────
    print("── Reload verification ──────────────────────────────────────")
    try:
        df_reload = load_ohlcv(
            args.product,
            start_date.isoformat(),
            end_date.isoformat(),
            db_path=str(DB_PATH),
        )
        print(f"  Reloaded {len(df_reload)} rows from SQLite ✓")
    except Exception as exc:
        print(f"  Reload skipped: {exc}")

    print()
    print("=" * 68)
    print("  Done.")
    print("=" * 68)


if __name__ == "__main__":
    main()

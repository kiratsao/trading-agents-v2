"""Parquet 健康檢查。

Usage:
    python scripts/verify_data.py                 # 預設 MXF
    python scripts/verify_data.py --product MXF
    python scripts/verify_data.py --product 2330
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Per-product config: parquet filename + sane close-price band
PRODUCT_CONFIG = {
    "MXF":  {"filename": "MXF_Daily_Clean_2020_to_now.parquet",  "close_range": (5_000, 60_000)},
    "2330": {"filename": "2330_Daily_Clean_2020_to_now.parquet", "close_range": (200, 3_000)},
}


def main():
    parser = argparse.ArgumentParser(description="Verify daily K parquet health.")
    parser.add_argument(
        "--product",
        default="MXF",
        choices=list(PRODUCT_CONFIG.keys()),
        help="Which parquet to check (default: MXF)",
    )
    args = parser.parse_args()

    cfg = PRODUCT_CONFIG[args.product]
    data_path = DATA_DIR / cfg["filename"]
    lo, hi = cfg["close_range"]

    if not data_path.exists():
        print(f"FAIL: {data_path} not found")
        print(f"  Run: python scripts/init_data.py --product {args.product}")
        sys.exit(1)

    df = pd.read_parquet(data_path)
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.sort_index()

    issues = []

    # 1. Latest date — gap measured in trading days (excludes weekends + TAIFEX holidays)
    latest = df.index[-1].date()
    today = pd.Timestamp.now(tz="Asia/Taipei").date()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data.tw_holidays import trading_days_between

    trading_gap = len(trading_days_between(latest, today)) - 1  # exclude latest itself
    calendar_gap = (today - latest).days
    if latest > today:
        issues.append(
            f"FUTURE BAR: latest={latest} is after today={today} — "
            f"a night-session (盤後) row was written as a day K; rebuild with "
            f"scripts/init_data.py"
        )
    if trading_gap > 3:
        issues.append(
            f"STALE: latest={latest}, today={today}, "
            f"gap={trading_gap} trading days ({calendar_gap} calendar)"
        )

    # 2. Weekend bars + holiday bars
    weekend = df[df.index.dayofweek >= 5]
    if len(weekend) > 0:
        issues.append(f"WEEKEND BARS: {len(weekend)} rows on Sat/Sun")

    from src.data.tw_holidays import is_taifex_holiday

    holiday_mask = (df.index.dayofweek < 5) & df.index.map(
        lambda ts: is_taifex_holiday(ts.date())
    )
    if holiday_mask.any():
        dates_str = df.index[holiday_mask].strftime("%Y-%m-%d").tolist()[:5]
        issues.append(
            f"HOLIDAY BARS: {holiday_mask.sum()} rows on TAIFEX holidays ({dates_str})"
        )

    # 3. Duplicate dates
    dups = df.index[df.index.duplicated()]
    if len(dups) > 0:
        issues.append(f"DUPLICATE DATES: {len(dups)} duplicates")

    # 4. Close range
    out_of_range = df[(df["close"] < lo) | (df["close"] > hi)]
    if len(out_of_range) > 0:
        issues.append(
            f"OUT OF RANGE: {len(out_of_range)} rows with close outside {lo:,}-{hi:,}"
        )

    # 5. NaN
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        issues.append(f"NaN: {nan_count} NaN values")

    # Report
    print(f"Data: {data_path.name}")
    print(f"  Bars: {len(df)}")
    print(f"  Range: {df.index[0].date()} → {latest}")
    print(f"  Last close: {df['close'].iloc[-1]:,.0f}")
    print(f"  Gap to today: {trading_gap} trading days ({calendar_gap} calendar)")

    if issues:
        print(f"\n  ISSUES ({len(issues)}):")
        for issue in issues:
            print(f"    - {issue}")
        sys.exit(1)
    else:
        print("\n  ✅ All checks passed")


if __name__ == "__main__":
    main()

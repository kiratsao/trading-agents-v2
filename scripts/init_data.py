"""首次部署或資料損壞時，從 Shioaji 重建完整日K parquet。

Usage: python scripts/init_data.py
"""

from __future__ import annotations

import os
import sys
from datetime import date, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = DATA_DIR / "MXF_Daily_Clean_2020_to_now.parquet"

DAY_OPEN = time(8, 45)
DAY_CLOSE = time(13, 45)
START_DATE = date(2020, 1, 1)


def main():
    api_key = os.environ.get("SHIOAJI_API_KEY", "")
    secret_key = os.environ.get("SHIOAJI_SECRET_KEY", "")
    if not api_key or not secret_key:
        print("ERROR: SHIOAJI_API_KEY / SHIOAJI_SECRET_KEY not set")
        sys.exit(1)

    from tw_futures.executor.shioaji_adapter import ShioajiAdapter

    print("Connecting to Shioaji...")
    adapter = ShioajiAdapter(
        api_key=api_key,
        secret_key=secret_key,
        simulation=False,
        cert_path=os.environ.get("SHIOAJI_CERT_PATH") or None,
        cert_password=os.environ.get("SHIOAJI_CERT_PASSWORD") or None,
        person_id=os.environ.get("SHIOAJI_PERSON_ID") or None,
    )

    today = pd.Timestamp.now(tz="Asia/Taipei").date()
    end_plus = pd.Timestamp(today) + pd.Timedelta(days=1)

    print(f"Fetching MXF kbars {START_DATE} → {today}...")
    contract = adapter.get_contract("MXF")
    kbars = adapter._api.kbars(
        contract,
        start=str(START_DATE),
        end=str(end_plus.date()),
        timeout=60_000,
    )
    adapter.logout()

    if not kbars or len(kbars.ts) == 0:
        print("ERROR: Shioaji returned empty kbars")
        sys.exit(1)

    raw = pd.DataFrame({
        "ts": kbars.ts,
        "open": kbars.Open,
        "high": kbars.High,
        "low": kbars.Low,
        "close": kbars.Close,
        "volume": kbars.Volume,
    })
    raw["ts"] = pd.to_datetime(raw["ts"], unit="ns", utc=True).dt.tz_convert("Asia/Taipei")
    raw = raw.sort_values("ts")

    mask = (raw["ts"].dt.time >= DAY_OPEN) & (raw["ts"].dt.time < DAY_CLOSE)
    day = raw[mask].copy()
    day["date"] = day["ts"].dt.date

    daily = day.groupby("date").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    daily.index = pd.DatetimeIndex(daily.index, name="date")
    daily = daily[daily.index.dayofweek < 5]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(OUT_PATH, index=True)
    print(f"Saved: {OUT_PATH}")
    print(f"  Bars: {len(daily)}")
    print(f"  Range: {daily.index[0].date()} → {daily.index[-1].date()}")
    print(f"  Last close: {daily['close'].iloc[-1]:,.0f}")


if __name__ == "__main__":
    main()

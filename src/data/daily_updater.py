"""每日自動更新 MXF 日K parquet。

14:25 排程執行，確保 14:30 信號計算使用最新資料。

流程
----
1. 讀取現有 parquet 的最後日期
2. 連線 Shioaji 抓取從最後日期+1 到今天的 1-min kbars
3. 聚合為日K（日盤 08:45-13:44）
4. 過濾週末 bar
5. 追加到 parquet（不重複）
6. 成功/失敗都透過 notify_fn 通知
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from datetime import date, time, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_PRIMARY_PARQUET = _DATA_DIR / "MXF_Daily_Clean_2020_to_now.parquet"

_DAY_OPEN = time(8, 45)
_DAY_CLOSE = time(13, 45)


def update(
    parquet_path: Path | None = None,
    notify_fn: Callable[[str], Any] | None = None,
) -> dict:
    """Fetch new daily bars from Shioaji and append to parquet.

    Returns
    -------
    dict with keys: success, bars_added, latest_date, error.
    """
    parquet_path = parquet_path or _PRIMARY_PARQUET
    _notify = notify_fn or (lambda msg: None)

    # 1. Load existing parquet
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.sort_index()
        last_date = df.index[-1].date() if len(df) > 0 else date(2020, 1, 1)
    else:
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        last_date = date(2020, 1, 1)

    # 2. Determine fetch range
    fetch_start = last_date + timedelta(days=1)
    today = _today_taipei()
    if fetch_start > today:
        logger.info("daily_updater: already up-to-date (last=%s)", last_date)
        return {"success": True, "bars_added": 0, "latest_date": str(last_date), "error": None}

    # 3. Fetch kbars from Shioaji
    logger.info("daily_updater: fetching kbars %s → %s", fetch_start, today)
    try:
        new_bars = _fetch_and_aggregate(fetch_start, today)
    except Exception as exc:
        err = f"Shioaji fetch raised: {exc}"
        logger.error("🔴 資料更新失敗: %s", err)
        _notify(f"🔴 資料更新失敗: {err}")
        return {"success": False, "bars_added": 0, "latest_date": str(last_date), "error": err}

    if new_bars is None or new_bars.empty:
        logger.info("daily_updater: no new bars from Shioaji")
        return {"success": True, "bars_added": 0, "latest_date": str(last_date), "error": None}

    # 4. Filter weekends
    new_bars = new_bars[new_bars.index.dayofweek < 5]
    if new_bars.empty:
        return {"success": True, "bars_added": 0, "latest_date": str(last_date), "error": None}

    # 5. Remove duplicates
    existing_dates = set(df.index.normalize())
    new_bars = new_bars[~new_bars.index.normalize().isin(existing_dates)]
    if new_bars.empty:
        return {"success": True, "bars_added": 0, "latest_date": str(last_date), "error": None}

    # 6. Append and save
    df = pd.concat([df, new_bars]).sort_index()
    df.to_parquet(parquet_path, index=True)
    n_new = len(new_bars)
    new_last = df.index[-1].date()
    last_close = float(df["close"].iloc[-1])

    msg = f"✅ 資料更新: +{n_new} bars, 最新: {new_last}, close: {last_close:,.0f}"
    logger.info(msg)
    _notify(msg)

    return {"success": True, "bars_added": n_new, "latest_date": str(new_last), "error": None}


def _fetch_and_aggregate(start: date, end: date) -> pd.DataFrame | None:
    """Connect to Shioaji, fetch 1-min kbars, aggregate to daily OHLCV."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.environ.get("SHIOAJI_API_KEY", "")
    secret_key = os.environ.get("SHIOAJI_SECRET_KEY", "")
    if not api_key or not secret_key:
        raise RuntimeError("SHIOAJI_API_KEY / SHIOAJI_SECRET_KEY not set — check .env")

    from tw_futures.executor.shioaji_adapter import ShioajiAdapter

    adapter = ShioajiAdapter(
        api_key=api_key,
        secret_key=secret_key,
        simulation=False,
        cert_path=os.environ.get("SHIOAJI_CERT_PATH") or None,
        cert_password=os.environ.get("SHIOAJI_CERT_PASSWORD") or None,
        person_id=os.environ.get("SHIOAJI_PERSON_ID") or None,
    )

    try:
        contract = adapter.get_contract("MXF")
        end_plus = pd.Timestamp(end) + pd.Timedelta(days=1)
        kbars = adapter._api.kbars(
            contract,
            start=str(start),
            end=str(end_plus.date()),
            timeout=30_000,
        )
    finally:
        adapter.logout()

    if not kbars or len(kbars.ts) == 0:
        logger.info("daily_updater: Shioaji returned empty kbars")
        return None

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

    # Day session only: 08:45-13:44 (< 13:45 excludes settlement bar)
    mask = (raw["ts"].dt.time >= _DAY_OPEN) & (raw["ts"].dt.time < _DAY_CLOSE)
    day = raw[mask].copy()

    if day.empty:
        return None

    day["date"] = day["ts"].dt.date
    daily = day.groupby("date").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    daily.index = pd.DatetimeIndex(daily.index, name="date")
    return daily


def _today_taipei() -> date:
    return pd.Timestamp.now(tz="Asia/Taipei").date()

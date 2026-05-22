"""每日自動更新 MXF 日K parquet。

14:25 排程執行，只更新到「昨天」(已收盤交易日)。
今天的即時 bar 由 orchestrator._load_data 用 Shioaji snapshot 取得。

這樣 14:25 永遠只處理已完成的交易日，不會有 partial bar 問題。

流程
----
1. 讀取現有 parquet 的最後日期
2. 連線 Shioaji 抓取從最後日期+1 到昨天的 1-min kbars
3. 聚合為日K（日盤 08:45-13:44）
4. 過濾週末 bar
5. 追加到 parquet（不重複）
6. 驗證 latest_date 是否為昨天(交易日)
7. 成功/失敗都透過 notify_fn 通知
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
_SETTLE_CLOSE = time(13, 30)  # settlement day regular session ends 13:30

# Number of recent trading days to re-scan after each update. Anything
# missing inside this window (per TAIFEX calendar) triggers a back-fill
# attempt and, on failure, a LINE alert so the operator can intervene.
# Keep this modest: a wider window risks many Shioaji round trips against
# old dates the operator has already accepted as lost.
_GAP_SCAN_LOOKBACK_DAYS = 10


def update(
    parquet_path: Path | None = None,
    notify_fn: Callable[[str], Any] | None = None,
) -> dict:
    """Fetch new daily bars from Shioaji and append to parquet.

    Only fetches bars up to *yesterday* (last completed trading day).
    Today's bar is handled by orchestrator via Shioaji snapshot at signal time.

    Returns
    -------
    dict with keys: success, bars_added, latest_date, error.
    """
    parquet_path = parquet_path or _PRIMARY_PARQUET
    _notify = notify_fn or (lambda msg: None)

    # 1. Load existing parquet — ensure DatetimeIndex
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        df.index = pd.to_datetime(df.index)
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.sort_index()
        last_date = df.index[-1].date() if len(df) > 0 else date(2020, 1, 1)
    else:
        df = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([], name="date"),
        )
        last_date = date(2020, 1, 1)

    # 2. Determine fetch range — only up to YESTERDAY (completed bars only)
    fetch_start = last_date + timedelta(days=1)
    today = _today_taipei()
    yesterday = _last_trading_day(today)

    if fetch_start > yesterday:
        logger.info("daily_updater: already up-to-date (last=%s)", last_date)
        # Tail-already-current doesn't mean the whole look-back window is
        # whole. A missed cron 5 days ago could still leave that day
        # absent; run the scan unconditionally.
        gaps_filled, still_missing = _detect_and_fill_gaps(
            df, parquet_path, _notify, yesterday,
        )
        return {
            "success": not still_missing,
            "bars_added": 0,
            "gaps_filled": gaps_filled,
            "latest_date": str(last_date),
            "error": None if not still_missing else f"unfilled gaps: {still_missing}",
        }

    # 3. Fetch kbars from Shioaji (only up to yesterday)
    logger.info(
        "daily_updater: fetching kbars %s → %s (today=%s excluded)",
        fetch_start, yesterday, today,
    )
    try:
        new_bars = _fetch_and_aggregate(fetch_start, yesterday)
    except Exception as exc:
        err = f"Shioaji fetch raised: {exc}"
        logger.error("🔴 資料更新失敗: %s", err)
        _notify(f"🔴 資料更新失敗: {err}")
        return {
            "success": False,
            "bars_added": 0,
            "gaps_filled": 0,
            "latest_date": str(last_date),
            "error": err,
        }

    if new_bars is None or new_bars.empty:
        # Validate: if latest_date < yesterday (trading day), this is a problem
        if last_date < yesterday and yesterday.weekday() < 5:
            warn = (
                f"🔴 資料缺口: parquet 最新={last_date}, "
                f"應至少到 {yesterday}, bars_added=0"
            )
            logger.error(warn)
            _notify(warn)
            return {
                "success": False,
                "bars_added": 0,
                "gaps_filled": 0,
                "latest_date": str(last_date),
                "error": warn,
            }
        logger.info("daily_updater: no new bars (last=%s, yesterday=%s)", last_date, yesterday)
        gaps_filled, still_missing = _detect_and_fill_gaps(
            df, parquet_path, _notify, yesterday,
        )
        return {
            "success": not still_missing,
            "bars_added": 0,
            "gaps_filled": gaps_filled,
            "latest_date": str(last_date),
            "error": None if not still_missing else f"unfilled gaps: {still_missing}",
        }

    # 4. Filter weekends + TAIFEX holidays
    new_bars = new_bars[new_bars.index.dayofweek < 5]
    from src.data.tw_holidays import is_taifex_holiday

    holiday_mask = new_bars.index.map(lambda ts: is_taifex_holiday(ts.date()))
    if holiday_mask.any():
        removed = new_bars.index[holiday_mask].strftime("%Y-%m-%d").tolist()
        logger.info("daily_updater: removed %d holiday bars: %s", len(removed), removed)
        new_bars = new_bars[~holiday_mask]
    if new_bars.empty:
        return {
            "success": True,
            "bars_added": 0,
            "gaps_filled": 0,
            "latest_date": str(last_date),
            "error": None,
        }

    # 5. Remove duplicates
    existing_dates = set(df.index.normalize())
    new_bars = new_bars[~new_bars.index.normalize().isin(existing_dates)]
    if new_bars.empty:
        return {
            "success": True,
            "bars_added": 0,
            "gaps_filled": 0,
            "latest_date": str(last_date),
            "error": None,
        }

    # 6. Append and save
    df = pd.concat([df, new_bars]).sort_index()
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df.to_parquet(parquet_path, index=True)
    n_new = len(new_bars)
    new_last = df.index[-1].date()
    last_close = float(df["close"].iloc[-1])

    # 7. Validate: latest_date should be >= yesterday
    if new_last < yesterday and yesterday.weekday() < 5:
        warn = (
            f"⚠️ 資料更新: +{n_new} bars 但仍有缺口 "
            f"(最新={new_last}, 應至少到 {yesterday})"
        )
        logger.warning(warn)
        _notify(warn)
    else:
        msg = (
            f"✅ 資料更新: +{n_new} bars, 最新: {new_last}, "
            f"close: {last_close:,.0f}"
        )
        logger.info(msg)
        _notify(msg)

    # 8. Re-scan the recent look-back window. Even after a successful tail
    # append, an older day inside the window can still be missing (e.g. an
    # earlier 14:25 cron silently produced 0 bars). One unnoticed gap
    # surfaces later as silent backtest divergence -- worth the extra scan.
    gaps_filled, still_missing = _detect_and_fill_gaps(
        df, parquet_path, _notify, yesterday,
    )

    return {
        "success": not still_missing,
        "bars_added": n_new,
        "gaps_filled": gaps_filled,
        "latest_date": str(new_last),
        "error": None if not still_missing else f"unfilled gaps: {still_missing}",
    }


def _last_trading_day(ref: date) -> date:
    """Return the most recent completed trading day before *ref*.

    Skips weekends and TAIFEX holidays.
    """
    from src.data.tw_holidays import last_trading_day_before

    return last_trading_day_before(ref)


def _trading_days_in_window(end: date, n_days: int) -> list[date]:
    """Return up to *n_days* most recent TAIFEX trading days <= *end*
    (inclusive), oldest first. Uses tw_holidays.is_trading_day so weekends
    AND TAIFEX holidays are skipped."""
    from src.data.tw_holidays import is_trading_day

    out: list[date] = []
    cur = end
    while len(out) < n_days and cur >= date(2020, 1, 1):
        if is_trading_day(cur):
            out.append(cur)
        cur -= timedelta(days=1)
    return list(reversed(out))


def _detect_and_fill_gaps(
    df: pd.DataFrame,
    parquet_path: Path,
    notify_fn: Callable[[str], Any],
    yesterday: date,
    *,
    _fetch_override: Callable[[date, date], pd.DataFrame | None] | None = None,
) -> tuple[int, list[date]]:
    """Scan the last N trading days for missing bars and try to back-fill.

    Returns ``(gaps_filled, still_missing)``. The parquet on disk is
    rewritten in-place after every successful back-fill so a later loop
    failure doesn't lose earlier successes. Each remaining gap emits a
    LINE alert with the date so the operator can investigate manually.

    ``_fetch_override`` is a testing seam -- production callers leave it
    None and the real Shioaji fetcher is used.
    """
    window = _trading_days_in_window(yesterday, _GAP_SCAN_LOOKBACK_DAYS)
    if not window:
        return 0, []

    existing = {ts.date() for ts in pd.DatetimeIndex(df.index).normalize()}
    missing = [d for d in window if d not in existing]
    if not missing:
        return 0, []

    logger.info(
        "daily_updater: gap scan found %d missing day(s) in last %d trading days: %s",
        len(missing), _GAP_SCAN_LOOKBACK_DAYS,
        [d.isoformat() for d in missing],
    )

    fetcher = _fetch_override if _fetch_override is not None else _fetch_and_aggregate

    filled = 0
    still_missing: list[date] = []
    for d in missing:
        bar: pd.DataFrame | None
        try:
            bar = fetcher(d, d)
        except Exception as exc:
            logger.warning("daily_updater: back-fill fetch for %s raised: %s", d, exc)
            bar = None

        if bar is None or bar.empty:
            still_missing.append(d)
            warn = f"⚠️ {d.isoformat()} 資料缺失，需手動處理 (自動補回失敗)"
            logger.warning(warn)
            notify_fn(warn)
            continue

        # Filter out weekends + anything already present.
        bar = bar[bar.index.dayofweek < 5]
        if bar.empty:
            still_missing.append(d)
            continue
        existing_dates = {ts.date() for ts in pd.DatetimeIndex(df.index).normalize()}
        bar = bar[~bar.index.normalize().isin(
            pd.DatetimeIndex([pd.Timestamp(x) for x in existing_dates])
        )]
        if bar.empty:
            # Already present after a prior loop iteration -- nothing more
            # to do, and definitely no alert.
            continue

        df = pd.concat([df, bar]).sort_index()
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df.to_parquet(parquet_path, index=True)
        filled += 1
        logger.info("daily_updater: back-filled %s", d)
        notify_fn(f"✅ {d.isoformat()} 資料補回成功")

    return filled, still_missing


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
        raise RuntimeError(
            "SHIOAJI_API_KEY / SHIOAJI_SECRET_KEY not set — check .env"
        )

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
        # `end` is the LAST completed trading day (yesterday at 14:25 cron
        # time). Empirically Shioaji's `end` is INCLUSIVE, so the previous
        # `end_plus = end + 1 day` query pulled today's still-evolving day
        # session bar. At 14:25 the day session has just closed (13:44) but
        # the settlement close hasn't published yet, so today's intra-day
        # last-tick mis-represented the official close by up to ~1,300pt.
        # Operators verifying the API semantics can run the probe script at
        # scripts/probe_shioaji_kbars_end_semantics.py before changing this.
        kbars = adapter._api.kbars(
            contract,
            start=str(start),
            end=str(end),
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
    raw["ts"] = pd.to_datetime(
        raw["ts"], unit="ns", utc=True,
    ).dt.tz_convert("Asia/Taipei")
    raw = raw.sort_values("ts")

    # Day session filter: 08:45 <= t < 13:45 (normal) or < 13:30 (settlement)
    from src.strategy.v2b_engine import _is_settlement_day

    t = raw["ts"].dt.time
    mask = (t >= _DAY_OPEN) & (t < _DAY_CLOSE)
    day = raw[mask].copy()

    # Remove 13:30-13:44 bars on settlement days
    if not day.empty:
        day_dates = day["ts"].dt.normalize()
        is_settle = day_dates.apply(lambda d: _is_settlement_day(d))
        settle_late = is_settle & (day["ts"].dt.time >= _SETTLE_CLOSE)
        day = day[~settle_late]

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
    # Defense-in-depth: even if the upstream `end` argument leaks a future
    # API change or someone re-introduces +1, any bar strictly newer than
    # the requested end (= last completed trading day) MUST be dropped.
    # A 14:25 query that surfaces today's still-evolving bar is the exact
    # bug that contaminated the parquet by ~1,300pt.
    cutoff = pd.Timestamp(end)
    bad = daily.index > cutoff
    if bad.any():
        dropped = [str(d.date()) for d in daily.index[bad]]
        logger.warning(
            "daily_updater: dropping %d bar(s) newer than end=%s: %s",
            int(bad.sum()), end, dropped,
        )
        daily = daily[~bad]
    return daily


def _today_taipei() -> date:
    return pd.Timestamp.now(tz="Asia/Taipei").date()

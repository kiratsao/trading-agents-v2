"""THE single authoritative Shioaji day-session fetch.

Historically four modules each re-implemented "pull the day-session close from
Shioaji kbars" (daily_updater, orchestrator, validation, init_data), each with
its own subtly-wrong filter — so a fix in one left the others broken and a bad
value once overwrote a correct parquet. ALL day-session filtering now lives
here and nowhere else.

Day session: 08:45 ≤ t < 13:45 (settlement days: 08:45 ≤ t < 13:30).
``end`` is INCLUSIVE everywhere — never add a day (the +1 bug pulled today's
still-evolving bar and mis-stated the close by ~1,300pt).
"""

from __future__ import annotations

import logging
from datetime import date, time, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

_DAY_OPEN = time(8, 45)
_DAY_CLOSE = time(13, 45)        # normal day session ends 13:44:59
_SETTLE_CLOSE = time(13, 30)     # settlement day regular session ends 13:30
_VOLUME_FLOOR = 30_000           # a real MXF day session trades well above this
_KBARS_TIMEOUT = 30_000


def fetch_day_session_bar(
    api,
    contract,
    day: date,
    *,
    timeout: int = _KBARS_TIMEOUT,
    volume_floor: int = _VOLUME_FLOOR,
) -> dict | None:
    """Authoritative single-day day-session OHLCV from Shioaji 1-min kbars.

    Returns ``{open, high, low, close, volume}`` or ``None`` when there is no
    day-session data or volume looks too low to be a complete session (a guard
    against partial / wrong fetches silently producing a bad bar).
    """
    try:
        kbars = api.kbars(contract, start=str(day), end=str(day), timeout=timeout)
    except Exception as exc:
        logger.warning("fetch_day_session_bar: kbars failed for %s: %s", day, exc)
        return None

    ts = getattr(kbars, "ts", None)
    if not kbars or ts is None or len(ts) == 0:
        return None

    raw = pd.DataFrame({
        "ts": kbars.ts, "open": kbars.Open, "high": kbars.High,
        "low": kbars.Low, "close": kbars.Close, "volume": kbars.Volume,
    })
    raw["ts"] = pd.to_datetime(raw["ts"], unit="ns", utc=True).dt.tz_convert("Asia/Taipei")
    raw = raw.sort_values("ts")

    from src.strategy.v2b_engine import _is_settlement_day
    close_cut = _SETTLE_CLOSE if _is_settlement_day(pd.Timestamp(day)) else _DAY_CLOSE

    t = raw["ts"].dt.time
    mask = (raw["ts"].dt.date == day) & (t >= _DAY_OPEN) & (t < close_cut)
    sess = raw[mask]
    if sess.empty:
        return None

    bar = {
        "open": float(sess.iloc[0]["open"]),
        "high": float(sess["high"].max()),
        "low": float(sess["low"].min()),
        "close": float(sess.iloc[-1]["close"]),
        "volume": int(sess["volume"].sum()),
    }
    if bar["volume"] <= volume_floor:
        logger.warning(
            "fetch_day_session_bar: %s volume=%d ≤ %d — 疑似非完整日盤，回傳 None",
            day, bar["volume"], volume_floor,
        )
        return None
    return bar


def fetch_day_session_bars(api, contract, start: date, end: date) -> pd.DataFrame:
    """Batch day-by-day fetch over [start, end] INCLUSIVE (never +1).

    Non-trading days are skipped (no pointless API call). Returns a daily
    OHLCV DataFrame (DatetimeIndex named ``date``); empty if nothing valid.
    """
    from src.data.tw_holidays import is_trading_day

    rows, idx = [], []
    d = start
    while d <= end:
        if is_trading_day(d):
            bar = fetch_day_session_bar(api, contract, d)
            if bar is not None:
                rows.append(bar)
                idx.append(pd.Timestamp(d))
        d += timedelta(days=1)

    if not rows:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([], name="date"),
        )
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx, name="date"))


def fetch_via_env(start: date, end: date, product: str = "MXF") -> pd.DataFrame | None:
    """Convenience: build a Shioaji adapter from env vars and fetch [start, end].

    Used by daily_updater / validation as the default production fetcher.
    Raises RuntimeError when credentials are missing so callers degrade cleanly.
    """
    import os

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
        api_key=api_key, secret_key=secret_key, simulation=False,
        cert_path=os.environ.get("SHIOAJI_CERT_PATH") or None,
        cert_password=os.environ.get("SHIOAJI_CERT_PASSWORD") or None,
        person_id=os.environ.get("SHIOAJI_PERSON_ID") or None,
    )
    try:
        contract = adapter.get_contract(product)
        return fetch_day_session_bars(adapter._api, contract, start, end)
    finally:
        adapter.logout()

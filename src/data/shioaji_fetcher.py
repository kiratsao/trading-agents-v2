"""THE single authoritative Shioaji day-session fetch.

Historically four modules each re-implemented "pull the day-session close from
Shioaji kbars" (daily_updater, orchestrator, validation, init_data), each with
its own subtly-wrong filter — so a fix in one left the others broken and a bad
value once overwrote a correct parquet. ALL day-session filtering now lives
here and nowhere else.

Day session: 08:45 ≤ t ≤ 13:45 INCLUSIVE — the official close prints in the
13:45 收盤集合競價 bar (settlement days: 08:45 ≤ t ≤ 13:30). ``end`` is
INCLUSIVE everywhere — never add a day (the +1 bug pulled today's
still-evolving bar and mis-stated the close by ~1,300pt).

Timestamp semantics (2026-07-25 server-side switch): Shioaji kbars ``ts``
changed from real UTC nanoseconds to Taipei-naive nanoseconds, decided by
QUERY time not bar date (same lib, same code, both behaviors observed). A
hardcoded ``utc=True`` double-shifted +8h, so the "day-session window"
silently selected the previous night's 00:45–05:45 tail — every day close
became the night close (the 7/25→8/06 degraded-feed incident). Semantics are
now detected PER RESPONSE from structure alone (see ``_kbars_ts_interpretations``)
— never from a date era or any label.
"""

from __future__ import annotations

import logging
from datetime import date, time, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

_DAY_OPEN = time(8, 45)
_DAY_CLOSE = time(13, 45)        # INCLUSIVE — 13:45 收盤集合競價 bar 是官方 close
_SETTLE_CLOSE = time(13, 30)     # INCLUSIVE — settlement day final auction 13:30
_KBARS_TIMEOUT = 30_000

# Legal TAIFEX trading minutes: day 08:45–13:45, night 15:00–05:00 (both ends
# inclusive). Used by the per-response ts-semantics detector: under the CORRECT
# interpretation 100% of bars fall in these windows; the wrong one puts bars in
# impossible hours (16:45–21:45, 05:01–08:44, 13:46–14:59).
_NIGHT_OPEN = time(15, 0)
_NIGHT_CLOSE = time(5, 0)
_FUTURE_TOL = pd.Timedelta(minutes=2)


def _legal_session_time(t: time) -> bool:
    # Minute truncation: a boundary bar stamped 13:45:30 / 05:00:30 is the
    # closing-auction print, not an illegal minute — second precision would
    # kill the TRUE interpretation and refuse the whole day (availability).
    t = time(t.hour, t.minute)
    return (_DAY_OPEN <= t <= _DAY_CLOSE) or t >= _NIGHT_OPEN or t <= _NIGHT_CLOSE


def _kbars_ts_interpretations(ts_ns, *, now=None) -> list[tuple[str, pd.Series]]:
    """Structurally-valid interpretations of a kbars ``ts`` array.

    Candidates: ``utc-legacy`` (ts is real UTC → +8h to Taipei) and
    ``taipei-naive`` (ts is already Taipei wall-clock). An interpretation is
    killed by (a) any bar in the future (> now+2min, Taipei) or (b) any bar
    outside legal TAIFEX session minutes. Returns the survivors as
    (name, naive-Taipei Series) — the caller decides what 0/1/2 survivors mean.
    """
    if now is None:
        from src.utils.tw_time import now_taipei

        now = pd.Timestamp(now_taipei()).tz_localize(None)
    else:
        now = pd.Timestamp(now)
    limit = now + _FUTURE_TOL

    base = pd.Series(pd.to_datetime(list(ts_ns), unit="ns"))
    candidates = [
        ("utc-legacy", base + pd.Timedelta(hours=8)),
        ("taipei-naive", base),
    ]
    survivors: list[tuple[str, pd.Series]] = []
    for name, ser in candidates:
        if len(ser) and ser.max() > limit:
            continue
        if all(_legal_session_time(t) for t in ser.dt.time):
            survivors.append((name, ser))
    return survivors

# Volume is a WARNING signal only — NEVER a gate. MXFR1 is a rolling near-month
# contract, so a historical query returns the *new* contract's (low) volume for
# that day (e.g. 05/15 → June contract's ~1,222, not May's ~198,312). Gating on
# volume wrongly dropped valid day-session bars. The 08:45–13:45 time filter is
# the real defense: a real day session (even a non-main contract) is > ~1,000,
# while night/anomaly bars are < ~500 and already excluded by the time filter.
_VOLUME_WARN = 1_000


def fetch_day_session_bar(
    api,
    contract,
    day: date,
    *,
    timeout: int = _KBARS_TIMEOUT,
    volume_warn: int = _VOLUME_WARN,
    _now=None,
) -> dict | None:
    """Authoritative single-day day-session OHLCV from Shioaji 1-min kbars.

    Returns ``{open, high, low, close, volume}``, or ``None`` when there is no
    day-session data at all (empty kbars / no bars in 08:45–close) OR when the
    timestamp semantics cannot be structurally determined (fail-loud — a wrong
    guess writes night values as day closes). Volume is NOT a gate — a low
    volume only logs a warning (see ``_VOLUME_WARN``), because rolling-contract
    historical queries legitimately report low volume. ``_now`` is a test seam
    for the future-timestamp check.
    """
    try:
        kbars = api.kbars(contract, start=str(day), end=str(day), timeout=timeout)
    except Exception as exc:
        logger.warning("fetch_day_session_bar: kbars failed for %s: %s", day, exc)
        return None

    ts = getattr(kbars, "ts", None)
    if not kbars or ts is None or len(ts) == 0:
        return None

    variants = _kbars_ts_interpretations(ts, now=_now)
    if not variants:
        logger.error(
            "fetch_day_session_bar: 🔴 %s kbars 時戳語義無法判定 "
            "(兩種解讀皆含未來或非法時段 bar) — 拒用", day,
        )
        return None

    raw = pd.DataFrame({
        "open": kbars.Open, "high": kbars.High,
        "low": kbars.Low, "close": kbars.Close, "volume": kbars.Volume,
    })

    from src.strategy.v2b_engine import _is_settlement_day
    close_cut = _SETTLE_CLOSE if _is_settlement_day(pd.Timestamp(day)) else _DAY_CLOSE

    picks = []
    for name, ts_ser in variants:
        t = ts_ser.dt.time
        picks.append((name, ts_ser,
                      (ts_ser.dt.date == day) & (t >= _DAY_OPEN) & (t <= close_cut)))
    if len(picks) == 2 and bool((picks[0][2] != picks[1][2]).any()):
        # Both interpretations structurally legal but selecting DIFFERENT rows
        # as the day session (e.g. a pure 00:00–05:00 night tail that a +8h
        # shift dresses up as 08:45–13:00 "day" bars) — exactly the bug class
        # this detector exists for. Never guess.
        logger.error(
            "fetch_day_session_bar: 🔴 %s kbars 時戳語義不明確 "
            "(兩解讀選出不同日盤集合) — 拒用", day,
        )
        return None
    semantics, ts_ser, mask = picks[0]
    if len(picks) == 2:
        semantics = "ambiguous-consistent"
    elif semantics == "taipei-naive":
        logger.info(
            "fetch_day_session_bar: %s kbars 時戳判定=台北 naive, 已自動校正 "
            "(2026-07-25 server 語義切換)", day,
        )
    raw["ts"] = ts_ser.values
    sess = raw[mask.values].sort_values("ts")
    if sess.empty:
        return None

    bar = {
        "open": float(sess.iloc[0]["open"]),
        "high": float(sess["high"].max()),
        "low": float(sess["low"].min()),
        "close": float(sess.iloc[-1]["close"]),
        "volume": int(sess["volume"].sum()),
        # Completeness metadata (2026-07-28 lesson: the feed died at 09:0x and
        # the truncated last bar masqueraded as the day close, 43,175 vs the
        # real 41,608 — a HOLD instead of a stop-out). Callers on the LIVE
        # path must check last_ts before treating close as the session close.
        "last_ts": sess.iloc[-1]["ts"].isoformat(),
        "n_bars": int(len(sess)),
        # Largest gap (minutes) between consecutive session bars. A real live
        # day session is ~300 contiguous 1-min bars; a forged session (night
        # tail + stray bars dressed up by a wrong ts interpretation) is sparse
        # or has a large hole before its "13:4x" last bar. Consumed by the P0
        # completeness gate together with n_bars.
        "max_gap_min": (
            float(sess["ts"].diff().dt.total_seconds().max() / 60.0)
            if len(sess) > 1 else 0.0
        ),
        "ts_semantics": semantics,
    }
    if bar["volume"] < volume_warn:
        logger.warning(
            "fetch_day_session_bar: %s volume=%d < %d — 量偏低（可能為滾動合約歷史量），"
            "仍回傳；交叉驗證請以 TAIFEX 為準",
            day, bar["volume"], volume_warn,
        )
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
                # Strip metadata keys — the parquet schema stays OHLCV only.
                rows.append({k: bar[k] for k in ("open", "high", "low", "close", "volume")})
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

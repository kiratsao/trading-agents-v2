"""Single source of truth for parquet freshness / expected-latest-date.

Four call sites (daily_updater, orchestrator, daily_health_check, deep_health)
each used to compute "what's the expected latest parquet bar" independently and
drifted apart. They all delegate here now.

Contract (固定): the parquet persists daily bars **up to and including T-1**
(`previous_trading_day(today)`). The updater never writes today's bar — today's
in-memory bar is added by orchestrator._load_data via snapshot and is not
persisted. The 14:30 update window means T-1 only lands at 14:30 on day T, so
``expected_parquet_latest`` is time-of-day aware: before 14:30 (or a non-trading
day) it allows one extra trading day of slack (expect T-2) to avoid the morning
false-alarm; at/after 14:30 it expects T-1.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

#: The daily update is expected to have completed by this Taipei wall-clock time.
UPDATE_DONE_TIME = time(14, 30)


class DataIntegrityError(Exception):
    """Raised when the parquet cannot be brought to / verified at the expected
    latest trading day. Callers at a process boundary translate this to a
    non-zero exit code; library code never calls sys.exit itself."""


def is_trading_day(d: date) -> bool:
    from src.data.tw_holidays import is_trading_day as _itd

    return _itd(d)


def previous_trading_day(d: date) -> date:
    """Most recent trading day strictly before *d* (holiday + weekend aware)."""
    from src.data.tw_holidays import last_trading_day_before

    return last_trading_day_before(d)


def trading_days_between(start_exclusive: date, end_inclusive: date) -> list[date]:
    """Trading days in ``(start_exclusive, end_inclusive]`` — oldest first."""
    from src.data.tw_holidays import trading_days_between as _tdb

    if end_inclusive <= start_exclusive:
        return []
    return [d for d in _tdb(start_exclusive, end_inclusive) if d > start_exclusive]


def expected_parquet_latest(now: datetime | None = None) -> date:
    """The latest trading-day bar the parquet *should* already contain.

    - trading day AND now ≥ 14:30 → ``previous_trading_day(today)`` (T-1)
    - otherwise (before 14:30, or non-trading day) → T-2 (one extra day slack)
    """
    now = now or _now_taipei()
    today = now.date()
    prev = previous_trading_day(today)
    if is_trading_day(today) and now.time() >= UPDATE_DONE_TIME:
        return prev
    return previous_trading_day(prev)


def check_parquet_freshness(
    parquet_path: str | Path,
    now: datetime | None = None,
) -> tuple[bool, str, date]:
    """Return ``(is_fresh, msg, expected_date)``.

    Reads the date from the parquet **index** (named ``date``) — there is no
    ``ts`` column. Fresh ⇔ ``max(index).date() >= expected_parquet_latest``.
    """
    expected = expected_parquet_latest(now)
    p = Path(parquet_path)
    if not p.exists():
        return False, f"🔴 parquet 不存在: {p}", expected
    try:
        df = pd.read_parquet(p)
    except Exception as exc:
        return False, f"🔴 parquet 無法讀取: {exc}", expected
    if len(df) == 0:
        return False, "🔴 parquet 為空", expected

    latest = pd.DatetimeIndex(df.index).max().date()
    if latest >= expected:
        return True, f"✅ 資料新鮮: latest={latest} ≥ expected={expected}", expected
    behind = len(trading_days_between(latest, expected))
    return (
        False,
        f"⚠️ Parquet 過期: latest={latest} < expected={expected}, 落後 {behind} 交易日",
        expected,
    )


def _now_taipei() -> datetime:
    return pd.Timestamp.now(tz="Asia/Taipei").to_pydatetime()

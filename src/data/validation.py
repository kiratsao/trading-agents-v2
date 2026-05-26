"""Cross-source data validation for the daily-K parquet.

Two independent oracles exist for TAIFEX index futures:

* **Shioaji** — 1-min kbars aggregated to daily (what ``daily_updater`` uses).
* **TAIFEX** — the exchange's official daily download (what ``init_data`` uses).

Comparing one against the other catches the silent-corruption class of bug
(e.g. the off-by-~1,300pt contamination when today's evolving bar leaked in).

All external fetches are injectable so the logic is unit-testable offline;
production callers fall back to the real fetchers, and any reference that
fails to fetch is skipped (never blocks the pipeline).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

import pandas as pd

logger = logging.getLogger(__name__)

WARN_THRESHOLD = 50.0    # points — surface a ⚠️ but keep the bar
ALERT_THRESHOLD = 200.0  # points — 🔴 and refuse to persist

# A reference fetcher takes (start, end) inclusive dates and returns a daily
# OHLCV DataFrame (DatetimeIndex) or None when unavailable.
FetchFn = Callable[[date, date], "pd.DataFrame | None"]


@dataclass
class CloseDiff:
    day: date
    parquet_close: float
    ref_close: float
    diff: float          # absolute difference in points
    source: str          # "shioaji" | "taifex"

    def __str__(self) -> str:
        return (f"{self.day} {self.source}: parquet={self.parquet_close:.0f} "
                f"vs ref={self.ref_close:.0f} (差 {self.diff:.0f})")


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.DatetimeIndex(out.index).normalize()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Default production fetchers (lazy imports avoid circulars / network at import)
# ─────────────────────────────────────────────────────────────────────────────
def _default_shioaji_fetch(start: date, end: date) -> pd.DataFrame | None:
    from src.data.daily_updater import _fetch_and_aggregate
    return _fetch_and_aggregate(start, end)


def _default_taifex_fetch(start: date, end: date) -> pd.DataFrame | None:
    from scripts.init_data import fetch_taifex_month
    frames = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        df = fetch_taifex_month(y, m, product="MTX")
        if df is not None and not df.empty:
            frames.append(df)
        m += 1
        if m > 12:
            m, y = 1, y + 1
    if not frames:
        return None
    allf = _normalize(pd.concat(frames).sort_index())
    mask = (allf.index >= pd.Timestamp(start)) & (allf.index <= pd.Timestamp(end))
    return allf[mask]


# ─────────────────────────────────────────────────────────────────────────────
# B2: validate the latest appended bar against independent oracles
# ─────────────────────────────────────────────────────────────────────────────
def validate_latest_bar(
    day: date,
    close: float,
    *,
    shioaji_fetch: FetchFn | None = None,
    taifex_fetch: FetchFn | None = None,
    warn: float = WARN_THRESHOLD,
    alert: float = ALERT_THRESHOLD,
) -> tuple[str, list[CloseDiff]]:
    """Compare *close* for *day* against each available oracle.

    Returns ``(level, diffs)`` where level is ``"ok"`` / ``"warn"`` / ``"alert"``.
    Each oracle is fetched independently; a fetch failure or missing day is
    skipped (logged), so a dead reference never blocks the update.
    """
    fetchers = {
        "shioaji": shioaji_fetch or _default_shioaji_fetch,
        "taifex": taifex_fetch or _default_taifex_fetch,
    }
    diffs: list[CloseDiff] = []
    for source, fetch in fetchers.items():
        try:
            ref = fetch(day, day)
        except Exception as exc:
            logger.warning("validate_latest_bar: %s fetch failed: %s", source, exc)
            continue
        if ref is None or len(ref) == 0:
            continue
        ref = _normalize(ref)
        ts = pd.Timestamp(day)
        if ts not in ref.index or "close" not in ref.columns:
            continue
        ref_close = float(ref.loc[ts, "close"])
        d = abs(float(close) - ref_close)
        if d > warn:
            diffs.append(CloseDiff(day, float(close), ref_close, d, source))

    level = "ok"
    if any(cd.diff > alert for cd in diffs):
        level = "alert"
    elif diffs:
        level = "warn"
    return level, diffs


# ─────────────────────────────────────────────────────────────────────────────
# B1: cross-validate recent parquet history vs Shioaji, override on divergence
# ─────────────────────────────────────────────────────────────────────────────
def validate_and_override_with_shioaji(
    df: pd.DataFrame,
    *,
    recent_days: int = 20,
    shioaji_fetch: FetchFn | None = None,
    threshold: float = WARN_THRESHOLD,
    log: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, list[CloseDiff]]:
    """For the last *recent_days* parquet bars, compare close vs Shioaji and
    OVERRIDE the parquet row (full OHLCV) with Shioaji's when |diff| > threshold.

    Returns ``(possibly-modified df, list of overridden CloseDiff)``.
    """
    emit = log or logger.warning
    if df is None or len(df) == 0:
        return df, []

    df = _normalize(df).sort_index()
    days = [d.date() for d in df.index[-recent_days:]]
    if not days:
        return df, []

    fetch = shioaji_fetch or _default_shioaji_fetch
    try:
        ref = fetch(days[0], days[-1])
    except Exception as exc:
        emit(f"⚠️ B1 cross-validation skipped — Shioaji fetch failed: {exc}")
        return df, []
    if ref is None or len(ref) == 0:
        emit("⚠️ B1 cross-validation skipped — Shioaji returned no data")
        return df, []

    ref = _normalize(ref)
    overridden: list[CloseDiff] = []
    for d in days:
        ts = pd.Timestamp(d)
        if ts not in ref.index or "close" not in ref.columns:
            continue
        pc = float(df.loc[ts, "close"])
        rc = float(ref.loc[ts, "close"])
        if abs(pc - rc) > threshold:
            cd = CloseDiff(d, pc, rc, abs(pc - rc), "shioaji")
            emit(f"⚠️ {cd} — 用 Shioaji 覆蓋")
            for col in ("open", "high", "low", "close", "volume"):
                if col in ref.columns:
                    df.loc[ts, col] = ref.loc[ts, col]
            overridden.append(cd)

    return df, overridden

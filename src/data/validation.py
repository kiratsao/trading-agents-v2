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
    ref_volume: float = 0.0   # reference bar volume (for day-session sanity)

    def __str__(self) -> str:
        return (f"{self.day} {self.source}: parquet={self.parquet_close:.0f} "
                f"vs ref={self.ref_close:.0f} (差 {self.diff:.0f}, vol={self.ref_volume:.0f})")


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.DatetimeIndex(out.index).normalize()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Default production fetchers (lazy imports avoid circulars / network at import)
# ─────────────────────────────────────────────────────────────────────────────
def _default_shioaji_fetch(start: date, end: date) -> pd.DataFrame | None:
    # Single authoritative day-session fetch (same as daily_updater uses).
    from src.data.shioaji_fetcher import fetch_via_env
    return fetch_via_env(start, end, product="MXF")


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
# Recent-history comparison vs Shioaji — REPORT ONLY (no mutation).
# The override decision + safety valve (volume / >200pt / backup / LINE) lives
# in deep_health_check Round 2 so nothing silently rewrites the parquet.
# ─────────────────────────────────────────────────────────────────────────────
def compare_to_shioaji(
    df: pd.DataFrame,
    *,
    recent_days: int = 20,
    shioaji_fetch: FetchFn | None = None,
    threshold: float = WARN_THRESHOLD,
) -> tuple[list[CloseDiff], pd.DataFrame]:
    """Compare the last *recent_days* parquet closes against Shioaji.

    Returns ``(diffs over threshold, normalized reference df)``. Pure: never
    mutates *df*. Fetch failure / empty reference propagates so the caller
    reports a skip rather than masquerading a dead oracle as "clean".
    """
    if df is None or len(df) == 0:
        return [], pd.DataFrame()

    df = _normalize(df).sort_index()
    days = [d.date() for d in df.index[-recent_days:]]
    if not days:
        return [], pd.DataFrame()

    fetch = shioaji_fetch or _default_shioaji_fetch
    ref = fetch(days[0], days[-1])
    if ref is None or len(ref) == 0:
        raise ValueError("Shioaji returned no reference data")

    ref = _normalize(ref)
    diffs: list[CloseDiff] = []
    for d in days:
        ts = pd.Timestamp(d)
        if ts not in ref.index or "close" not in ref.columns:
            continue
        pc = float(df.loc[ts, "close"])
        rc = float(ref.loc[ts, "close"])
        rv = float(ref.loc[ts, "volume"]) if "volume" in ref.columns else 0.0
        if abs(pc - rc) > threshold:
            diffs.append(CloseDiff(d, pc, rc, abs(pc - rc), "shioaji", rv))
    return diffs, ref

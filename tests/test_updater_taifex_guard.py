"""Guard: on a cross-source validation ALERT the updater substitutes the
validated TAIFEX 一般 (day) value instead of rejecting the bar and leaving a gap.

Shioaji occasionally hands back a night/anomalous bar (the 2026-07-17 back-fill:
44,714 = overnight vs the real 42,697 day close). The validator flags it; rather
than stall the parquet with a gap, ``_rescue_divergent_bar`` swaps in TAIFEX's
day value — the same source the oracle trusts.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from src.data import daily_updater


def _bar(day: str, close: float) -> pd.DataFrame:
    return pd.DataFrame(
        [{"open": close, "high": close, "low": close, "close": close, "volume": 190000}],
        index=pd.DatetimeIndex([pd.Timestamp(day)], name="date"),
    )


def _patch_taifex(monkeypatch, frame):
    monkeypatch.setattr(
        "src.data.validation.fetch_taifex_day_session_range", lambda s, e: frame
    )


def test_rescue_swaps_night_bar_for_taifex_day(monkeypatch):
    _patch_taifex(monkeypatch, _bar("2026-07-17", 42697.0))
    notes: list[str] = []
    out = daily_updater._rescue_divergent_bar(
        date(2026, 7, 17), _bar("2026-07-17", 44714.0), notes.append
    )
    assert out is not None
    assert abs(float(out["close"].iloc[-1]) - 42697.0) < 1
    assert any("改用 TAIFEX" in n for n in notes)


def test_rescue_keeps_bar_that_already_matches_taifex(monkeypatch):
    # Guard-corrected / genuine day bar already equals TAIFEX → keep it, and the
    # alert is just the flaky Shioaji oracle.
    _patch_taifex(monkeypatch, _bar("2026-07-16", 45715.0))
    notes: list[str] = []
    bar = _bar("2026-07-16", 45715.0)
    out = daily_updater._rescue_divergent_bar(date(2026, 7, 16), bar, notes.append)
    assert out is bar
    assert any("已符合 TAIFEX" in n for n in notes)


def test_rescue_returns_none_when_taifex_unavailable(monkeypatch):
    # No TAIFEX day bar → cannot rescue → caller must reject (gap stays).
    _patch_taifex(monkeypatch, None)
    out = daily_updater._rescue_divergent_bar(
        date(2026, 7, 17), _bar("2026-07-17", 44714.0), None
    )
    assert out is None

"""Guard: the updater resolves the true day close from three sources (spot
^TWII, TAIFEX 一般, Shioaji) and never trusts a parse's session label.

It swaps a night bar for the spot-anchored day value, rescues via Shioaji even
when TAIFEX itself is night (a whole-environment mislabel), and fails LOUD
(rejects) only when every source looks like night vs the night-proof spot index.
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


def _setup(monkeypatch, taifex_close, spot):
    tx = _bar("2026-07-17", taifex_close) if taifex_close is not None else None
    monkeypatch.setattr(daily_updater, "_taifex_day_bar", lambda d: tx)
    monkeypatch.setattr("src.data.spot_ref.fetch_spot_close", lambda d, **k: spot)


def test_rescue_picks_taifex_day_via_spot(monkeypatch):
    _setup(monkeypatch, 42697.0, 42671.0)  # TAIFEX=day, spot=day
    notes: list[str] = []
    out = daily_updater._rescue_divergent_bar(
        date(2026, 7, 17), _bar("2026-07-17", 44714.0), notes.append
    )
    assert out is not None and abs(float(out["close"].iloc[-1]) - 42697.0) < 1
    assert any("採 TAIFEX" in n for n in notes)


def test_rescue_via_shioaji_when_taifex_also_night(monkeypatch):
    # Whole-env mislabel: TAIFEX parse ALSO returns night; spot rescues via the
    # Shioaji bar, which is the true day value.
    _setup(monkeypatch, 44714.0, 42671.0)  # TAIFEX=night, spot=day
    day_bar = _bar("2026-07-17", 42697.0)
    notes: list[str] = []
    out = daily_updater._rescue_divergent_bar(date(2026, 7, 17), day_bar, notes.append)
    assert out is day_bar
    assert any("採 Shioaji" in n for n in notes)


def test_rescue_failloud_when_all_sources_night(monkeypatch):
    _setup(monkeypatch, 44714.0, 42671.0)  # TAIFEX=night, spot=day
    notes: list[str] = []
    out = daily_updater._rescue_divergent_bar(
        date(2026, 7, 17), _bar("2026-07-17", 44714.0), notes.append  # Shioaji=night too
    )
    assert out is None
    assert any(n.startswith("🔴") for n in notes)


def test_spot_flags_night_bar(monkeypatch):
    monkeypatch.setattr("src.data.spot_ref.fetch_spot_close", lambda d, **k: 42671.0)
    assert daily_updater._spot_flags_bar(date(2026, 7, 17), 44714.0) is True   # night, far
    assert daily_updater._spot_flags_bar(date(2026, 7, 17), 42697.0) is False  # day, near


def test_spot_flags_degrades_when_spot_unavailable(monkeypatch):
    monkeypatch.setattr("src.data.spot_ref.fetch_spot_close", lambda d, **k: None)
    # No spot → defer to the other oracles, never blanket-reject.
    assert daily_updater._spot_flags_bar(date(2026, 7, 17), 44714.0) is False

"""The single authoritative Shioaji day-session fetcher (logic, offline).

ts are built as real UTC nanoseconds from Asia/Taipei wall-clock strings, so
the fetcher's tz_convert + between-time filtering is exercised exactly as in
production. (The live "05/15 → 41,009" check needs real Shioaji on GCP; here we
prove the filtering returns the day-session last close from synthetic kbars.)
"""

import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.shioaji_fetcher import fetch_day_session_bar, fetch_day_session_bars
from tests.fakes import _FakeKbars


def _kbars(taipei_bars):
    """taipei_bars: list of (wallclock_str, open, high, low, close, volume)."""
    ts = [pd.Timestamp(t, tz="Asia/Taipei").value for t, *_ in taipei_bars]  # UTC ns
    o = [b[1] for b in taipei_bars]
    h = [b[2] for b in taipei_bars]
    low = [b[3] for b in taipei_bars]
    c = [b[4] for b in taipei_bars]
    v = [b[5] for b in taipei_bars]
    return _FakeKbars(ts, o, h, low, c, v)


class _Api:
    def __init__(self, kb):
        self.kb = kb

    def kbars(self, contract, start=None, end=None, timeout=None):
        return self.kb


def test_fetcher_filters_day_session_only_and_returns_day_close():
    api = _Api(_kbars([
        ("2026-05-15 09:00", 41_000, 41_010, 40_990, 41_005, 20_000),
        ("2026-05-15 12:00", 41_005, 41_015, 41_000, 41_008, 20_000),
        ("2026-05-15 13:40", 41_008, 41_012, 41_006, 41_009, 20_000),  # day close
        ("2026-05-15 15:00", 41_400, 41_600, 41_300, 41_500, 99_999),  # night — excluded
        ("2026-05-15 22:00", 41_500, 41_700, 41_400, 41_600, 99_999),  # night — excluded
    ]))
    bar = fetch_day_session_bar(api, None, date(2026, 5, 15))
    assert bar is not None
    assert bar["close"] == 41_009          # last day-session bar, not the night
    assert bar["open"] == 41_000           # first day-session bar
    assert bar["volume"] == 60_000         # 3 day bars only (night not summed)


def test_fetcher_settlement_day_excludes_1330(monkeypatch):
    monkeypatch.setattr("src.strategy.v2b_engine._is_settlement_day", lambda d: True)
    api = _Api(_kbars([
        ("2026-05-20 13:25", 41_000, 41_010, 40_990, 41_009, 40_000),
        ("2026-05-20 13:35", 41_009, 41_300, 41_009, 41_200, 40_000),  # ≥13:30 → excluded
    ]))
    bar = fetch_day_session_bar(api, None, date(2026, 5, 20))
    assert bar is not None
    assert bar["close"] == 41_009          # 13:35 bar dropped on settlement day
    assert bar["volume"] == 40_000


def test_fetcher_low_volume_returns_none():
    api = _Api(_kbars([
        ("2026-05-15 09:00", 41_000, 41_010, 40_990, 41_005, 100),  # < 30,000 floor
    ]))
    assert fetch_day_session_bar(api, None, date(2026, 5, 15)) is None


def test_fetcher_end_not_plus_one():
    # kbars (fake) holds BOTH days; a fetch for 05/15 must not leak 05/16.
    api = _Api(_kbars([
        ("2026-05-15 10:00", 41_000, 41_010, 40_990, 41_009, 40_000),
        ("2026-05-16 10:00", 42_000, 42_010, 41_990, 42_009, 40_000),
    ]))
    bar = fetch_day_session_bar(api, None, date(2026, 5, 15))
    assert bar["close"] == 41_009          # 05/16 excluded by date filter

    df = fetch_day_session_bars(api, None, date(2026, 5, 15), date(2026, 5, 15))
    assert list(df.index) == [pd.Timestamp("2026-05-15")]   # end inclusive, no +1


def test_fetcher_empty_kbars_returns_none():
    assert fetch_day_session_bar(_Api(_kbars([])), None, date(2026, 5, 15)) is None

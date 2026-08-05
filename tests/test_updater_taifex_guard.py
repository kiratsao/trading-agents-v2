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


def _setup(monkeypatch, taifex_close, spot, night_close=None):
    tx = _bar("2026-07-17", taifex_close) if taifex_close is not None else None
    monkeypatch.setattr(daily_updater, "_taifex_day_bar", lambda d: tx)
    monkeypatch.setattr(daily_updater, "_taifex_night_close", lambda d: night_close)
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


class TestNightProvenance:
    """2026-07-22 small-gap blind spot: a night bar sitting INSIDE the spot band.

    stored 44,957 == TAIFEX 盤後 bit-exactly, only 131pt from spot while the
    day value (44,627) was 199pt away — the spot distance preferred the night
    value. Provenance (identity with the 盤後 candidate) must outrank spot."""

    def _setup22(self, monkeypatch, night_close=44_957.0):
        day_bar = _bar("2026-07-22", 44_627.0)
        monkeypatch.setattr(daily_updater, "_taifex_day_bar", lambda d: day_bar)
        monkeypatch.setattr(daily_updater, "_taifex_night_close",
                            lambda d: night_close)
        monkeypatch.setattr("src.data.spot_ref.fetch_spot_close",
                            lambda d, **k: 44_826.0)
        return day_bar

    def test_rescue_provenance_beats_spot(self, monkeypatch):
        self._setup22(monkeypatch)
        notes: list[str] = []
        out = daily_updater._rescue_divergent_bar(
            date(2026, 7, 22), _bar("2026-07-22", 44_957.0), notes.append,
        )
        assert out is not None
        assert abs(float(out["close"].iloc[-1]) - 44_627.0) < 1
        assert any("provenance" in n for n in notes)

    def test_night_provenance_detector(self, monkeypatch):
        self._setup22(monkeypatch)
        assert daily_updater._night_provenance(date(2026, 7, 22), 44_957.0) is True
        assert daily_updater._night_provenance(date(2026, 7, 22), 44_627.0) is False
        # matches neither candidate → provenance ambiguous → spot rules apply
        assert daily_updater._night_provenance(date(2026, 7, 22), 44_800.0) is False

    def test_write_path_catches_in_band_night_bar(self, monkeypatch, tmp_path):
        """End-to-end update(): validator ok + spot gate passes (131 ≤ 500) —
        the provenance trigger must still swap the night bar for the day value."""
        self._setup22(monkeypatch)
        idx = pd.DatetimeIndex([pd.Timestamp("2026-07-21")], name="date")
        seed = pd.DataFrame(
            [{"open": 44_300.0, "high": 44_500.0, "low": 44_100.0,
              "close": 44_379.0, "volume": 150_000}], index=idx,
        )
        pq = tmp_path / "MXF.parquet"
        seed.to_parquet(pq)

        night_bar = _bar("2026-07-22", 44_957.0)
        monkeypatch.setattr(daily_updater, "_fetch_and_aggregate",
                            lambda s, e, notify_fn=None: night_bar)
        monkeypatch.setattr(daily_updater, "_today_taipei",
                            lambda: date(2026, 7, 23))
        monkeypatch.setattr(daily_updater, "_detect_and_fill_gaps",
                            lambda *a, **k: (0, []))

        notes: list[str] = []
        res = daily_updater.update(
            parquet_path=pq, notify_fn=notes.append,
            validate_fn=lambda day, close: ("ok", []),
        )
        assert res["success"] is True
        out = pd.read_parquet(pq)
        assert abs(float(out.loc[pd.Timestamp("2026-07-22"), "close"]) - 44_627.0) < 1
        assert any("provenance" in n for n in notes)


class TestCalendarGuard:
    """2026-07-10 幽靈 bar: the typhoon closure was absent from the holiday
    table, so update() on 7/13 treated 7/10 as a missing trading day and wrote
    the 7/09-night midnight tail as a 7/10 day K (close 46,275) — pushing
    ADX(7/28) 24.82→28.36 across the entry threshold and turning 7/22 into a
    data-induced buy. Every write must now be TAIFEX-confirmed."""

    def test_holiday_table_has_typhoon_closure(self):
        from src.data.tw_holidays import is_trading_day
        assert is_trading_day(date(2026, 7, 10)) is False

    def test_guard_refuses_unconfirmed_date(self, monkeypatch):
        monkeypatch.setattr(daily_updater, "_taifex_confirms_day", lambda d: False)
        notes: list[str] = []
        assert daily_updater._calendar_guard(date(2026, 7, 10), notes.append) is False
        assert any("拒寫幽靈 bar" in n for n in notes)

    def test_guard_refuses_indeterminate_with_retry_note(self, monkeypatch):
        monkeypatch.setattr(daily_updater, "_taifex_confirms_day", lambda d: None)
        notes: list[str] = []
        assert daily_updater._calendar_guard(date(2026, 7, 9), notes.append) is False
        assert any("重試" in n for n in notes)

    @staticmethod
    def _seed(tmp_path, days_closes):
        idx = pd.DatetimeIndex([pd.Timestamp(d) for d, _ in days_closes], name="date")
        seed = pd.DataFrame(
            [{"open": c, "high": c, "low": c, "close": c, "volume": 150_000}
             for _, c in days_closes], index=idx,
        )
        pq = tmp_path / "MXF.parquet"
        seed.to_parquet(pq)
        return seed, pq

    def test_gap_fill_refuses_unconfirmed_day(self, monkeypatch, tmp_path):
        monkeypatch.setattr(daily_updater, "_taifex_confirms_day", lambda d: False)
        seed, pq = self._seed(
            tmp_path, [("2026-07-07", 45_759.0), ("2026-07-08", 45_900.0)])
        notes: list[str] = []
        filled, still = daily_updater._detect_and_fill_gaps(
            seed, pq, notes.append, date(2026, 7, 9),
            _fetch_override=lambda s, e: _bar("2026-07-09", 46_275.0),
        )
        assert filled == 0
        assert date(2026, 7, 9) in still
        out = pd.read_parquet(pq)
        assert pd.Timestamp("2026-07-09") not in out.index
        assert any("拒寫幽靈 bar" in n for n in notes)

    def test_gap_fill_writes_confirmed_day(self, monkeypatch, tmp_path):
        # conftest autouse: _taifex_confirms_day → True for every date
        seed, pq = self._seed(
            tmp_path, [("2026-07-07", 45_759.0), ("2026-07-08", 45_900.0)])
        filled, still = daily_updater._detect_and_fill_gaps(
            seed, pq, lambda m: None, date(2026, 7, 9),
            _fetch_override=lambda s, e: (
                _bar("2026-07-09", 45_681.0) if s == date(2026, 7, 9) else None
            ),
        )
        assert filled == 1
        out = pd.read_parquet(pq)
        assert abs(float(out.loc[pd.Timestamp("2026-07-09"), "close"]) - 45_681.0) < 1

    def test_update_main_path_refuses_unconfirmed_date(self, monkeypatch, tmp_path):
        monkeypatch.setattr(daily_updater, "_taifex_confirms_day", lambda d: False)
        _, pq = self._seed(tmp_path, [("2026-07-08", 45_900.0)])
        monkeypatch.setattr(daily_updater, "_fetch_and_aggregate",
                            lambda s, e, notify_fn=None: _bar("2026-07-09", 46_275.0))
        monkeypatch.setattr(daily_updater, "_today_taipei", lambda: date(2026, 7, 10))
        notes: list[str] = []
        res = daily_updater.update(parquet_path=pq, notify_fn=notes.append,
                                   validate_fn=lambda day, close: ("ok", []))
        assert res["success"] is False
        assert "幽靈" in res["error"]
        out = pd.read_parquet(pq)
        assert pd.Timestamp("2026-07-09") not in out.index

    def test_update_lock_timeout_skips_cycle(self, monkeypatch, tmp_path):
        """A concurrently-held write lock makes update() skip loudly instead of
        interleaving a read-modify-write (daemon vs systemd-timer race)."""
        import fcntl

        pq = tmp_path / "MXF.parquet"
        lock_path = pq.with_suffix(pq.suffix + ".lock")
        monkeypatch.setattr(daily_updater, "_LOCK_WAIT_S", 0.1)
        holder = open(lock_path, "w")
        try:
            fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
            notes: list[str] = []
            res = daily_updater.update(parquet_path=pq, notify_fn=notes.append)
            assert res["success"] is False
            assert "寫入鎖" in res["error"]
            assert any("寫入鎖" in n for n in notes)
        finally:
            holder.close()

    def test_incident_0713_cannot_recur(self, monkeypatch, tmp_path):
        """With 7/10 in the holiday table, the exact 7/13 14:25 run no longer
        even attempts a fetch for 7/10 — already-up-to-date branch, no ghost."""
        _, pq = self._seed(
            tmp_path, [("2026-07-08", 45_900.0), ("2026-07-09", 45_681.0)])

        def _no_fetch(s, e, notify_fn=None):
            raise AssertionError(f"must not fetch {s}→{e} — 7/10 is a holiday")

        monkeypatch.setattr(daily_updater, "_fetch_and_aggregate", _no_fetch)
        monkeypatch.setattr(daily_updater, "_today_taipei", lambda: date(2026, 7, 13))
        monkeypatch.setattr(daily_updater, "_detect_and_fill_gaps",
                            lambda *a, **k: (0, []))
        res = daily_updater.update(parquet_path=pq, notify_fn=lambda m: None,
                                   validate_fn=lambda day, close: ("ok", []))
        assert res["success"] is True
        assert res["latest_date"] == "2026-07-09"

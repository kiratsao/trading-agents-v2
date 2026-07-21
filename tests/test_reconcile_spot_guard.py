"""The reconcile tool must be spot-guarded: it may only overwrite a bar when the
re-fetched TAIFEX 一般 value is STRICTLY closer to the spot index than what is
already there. This is the safety net that stops a bad re-fetch (e.g. a night
value) from clobbering a correct day value — the failure that damaged the parquet
before the guard existed.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

import scripts.reconcile_recent_bars as rc


def test_spot_guard_corrects_skips_and_adds(tmp_path, monkeypatch):
    # Real trading days (is_trading_day is not mocked).
    d_corr = pd.Timestamp("2026-07-16")   # present night value, TAIFEX day closer -> CORRECT
    d_prot = pd.Timestamp("2026-07-15")   # present value already closer to spot -> SKIP
    d_add = pd.Timestamp("2026-07-17")    # missing, TAIFEX near spot -> ADD
    d_add_bad = pd.Timestamp("2026-07-20")  # missing, TAIFEX far from spot -> SKIP ADD

    seed = pd.DataFrame(
        {"open": [45_818.0, 45_650.0], "high": [45_900.0, 45_700.0],
         "low": [45_700.0, 45_600.0], "close": [45_818.0, 45_650.0],
         "volume": [1, 1]},
        index=pd.DatetimeIndex([d_corr, d_prot], name="date"),
    )
    pq = tmp_path / "MXF.parquet"
    seed.to_parquet(pq)

    def _row(close):
        return {"open": close, "high": close, "low": close, "close": close, "volume": 1}

    taifex = pd.DataFrame(
        [_row(45_715.0), _row(45_831.0), _row(42_697.0), _row(48_000.0)],
        index=pd.DatetimeIndex([d_corr, d_prot, d_add, d_add_bad], name="date"),
    )
    spot = {d_corr: 45_625.0, d_prot: 45_632.0, d_add: 42_671.0, d_add_bad: 42_450.0}

    monkeypatch.setattr(rc, "fetch_taifex_day_session_range", lambda s, e: taifex)
    monkeypatch.setattr(rc, "_fetch_spot", lambda s, e: spot)

    rc.reconcile(pq, date(2026, 7, 15), date(2026, 7, 20), apply=True)
    out = pd.read_parquet(pq)

    # CORRECT: night 45,818 (|193| from spot) -> day 45,715 (|90|)
    assert abs(float(out.loc[d_corr, "close"]) - 45_715.0) < 1
    # PROTECTED: old 45,650 (|18|) closer than TAIFEX 45,831 (|199|) -> untouched
    assert abs(float(out.loc[d_prot, "close"]) - 45_650.0) < 1
    # ADD near spot
    assert d_add in out.index and abs(float(out.loc[d_add, "close"]) - 42_697.0) < 1
    # ADD far from spot (looks like night/anomaly) -> refused
    assert d_add_bad not in out.index


def test_spot_guard_aborts_without_spot(tmp_path, monkeypatch):
    seed = pd.DataFrame(
        {"open": [45_818.0], "high": [45_900.0], "low": [45_700.0],
         "close": [45_818.0], "volume": [1]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-07-16")], name="date"),
    )
    pq = tmp_path / "MXF.parquet"
    seed.to_parquet(pq)
    taifex = pd.DataFrame(
        [{"open": 45_715.0, "high": 45_715.0, "low": 45_715.0, "close": 45_715.0, "volume": 1}],
        index=pd.DatetimeIndex([pd.Timestamp("2026-07-16")], name="date"),
    )
    monkeypatch.setattr(rc, "fetch_taifex_day_session_range", lambda s, e: taifex)
    monkeypatch.setattr(rc, "_fetch_spot", lambda s, e: None)  # spot unavailable

    rc_code = rc.reconcile(pq, date(2026, 7, 16), date(2026, 7, 16), apply=True)
    assert rc_code == 1  # aborted
    # parquet untouched (still the night value, nothing guessed)
    assert abs(float(pd.read_parquet(pq).loc[pd.Timestamp("2026-07-16"), "close"]) - 45_818.0) < 1

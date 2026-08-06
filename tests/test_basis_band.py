"""裁示 2026-08-05: spot band 改 max(500, 1.6%×spot)。

四案 regression: 7/31 合法大漲 bar (basis 559) 須過;7/28 截斷仍擋;7/17 大偏離
夜值仍擋;7/22 型 band 內夜值由 provenance 攔。另 _spot_flags_bar 同步改相對 band
(2026-06-02 案例不再誤傷)。全史 0 誤傷複驗見 scratchpad spot_band_study。
"""
from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pandas as pd

from src.data.spot_ref import _BASIS_BAND, basis_band_for
from src.scheduler.orchestrator import _validate_today_bar


def _bar(close: float, last_ts: str) -> dict:
    return {"open": close, "high": close, "low": close, "close": close,
            "volume": 190_000, "last_ts": last_ts}


class TestBandFormula:
    def test_floor_at_low_index(self):
        assert basis_band_for(30_000.0) == _BASIS_BAND
        assert basis_band_for(None) == _BASIS_BAND
        assert basis_band_for(0.0) == _BASIS_BAND

    def test_relative_at_2026_levels(self):
        assert abs(basis_band_for(43_119.8) - 689.9168) < 1e-3
        assert basis_band_for(31_250.0) == 500.0  # 1.6% 恰等於樓地板


class TestRulingRegressions:
    def test_case1_0731_legit_big_move_passes(self):
        """+3,402pt 合法大漲日, basis 559.2 — 舊平坦 500 拒、新 band 689.9 過。"""
        with patch("src.data.spot_ref.fetch_spot_close", return_value=43_119.8):
            meta = _validate_today_bar(
                _bar(43_679.0, "2026-07-31T13:44:00+08:00"),
                date(2026, 7, 31), source="kbars",
            )
        assert meta["spot_ok"] is True
        assert meta["validated"] is True

    def test_case2_0728_truncated_still_blocked(self):
        with patch("src.data.spot_ref.fetch_spot_close", return_value=41_603.0):
            meta = _validate_today_bar(
                _bar(43_175.0, "2026-07-28T09:05:00+08:00"),
                date(2026, 7, 28), source="kbars",
            )
        assert meta["complete"] is False
        assert meta["spot_ok"] is False        # 1,572 > 665.6 — band 也擋
        assert meta["validated"] is False
        assert not meta["usable_for_exit"]

    def test_case3_0717_night_value_still_blocked(self):
        with patch("src.data.spot_ref.fetch_spot_close", return_value=42_671.0):
            meta = _validate_today_bar(
                _bar(44_714.0, "2026-07-17T13:44:00+08:00"),
                date(2026, 7, 17), source="kbars",
            )
        assert meta["spot_ok"] is False        # 2,043 > 682.7
        assert meta["validated"] is False
        assert not meta["usable_for_exit"]

    def test_case4_0722_in_band_night_blocked_by_provenance(self, monkeypatch):
        """band 放寬後 131pt 依然在 band 內 — 攔截層是 provenance,不是 band。"""
        from src.data import daily_updater

        monkeypatch.setattr("src.data.spot_ref.fetch_spot_close",
                            lambda d, **k: 44_826.0)
        monkeypatch.setattr(daily_updater, "_taifex_night_close",
                            lambda d: 44_957.0)
        monkeypatch.setattr(daily_updater, "_taifex_day_bar", lambda d: pd.DataFrame(
            [{"open": 44_627.0, "high": 44_627.0, "low": 44_627.0,
              "close": 44_627.0, "volume": 150_000}],
            index=pd.DatetimeIndex([pd.Timestamp("2026-07-22")], name="date")))
        meta = _validate_today_bar(
            _bar(44_957.0, "2026-07-22T13:44:00+08:00"),
            date(2026, 7, 22), source="kbars",
        )
        assert meta["spot_ok"] is True         # 131 ≤ 717 — band 過
        assert meta["night_hit"] is True       # provenance 攔
        assert meta["validated"] is False
        assert not meta["usable_for_exit"]


class TestSpotFlagsBarRelative:
    def test_legit_big_basis_no_longer_flags(self, monkeypatch):
        """2026-06-02: 46,110 vs spot 45,557 (dev 552.7) — 舊 500 誤傷、新 band 729 過。"""
        from src.data import daily_updater

        monkeypatch.setattr("src.data.spot_ref.fetch_spot_close",
                            lambda d, **k: 45_557.3)
        assert daily_updater._spot_flags_bar(date(2026, 6, 2), 46_110.0) is False

    def test_night_value_still_flags(self, monkeypatch):
        from src.data import daily_updater

        monkeypatch.setattr("src.data.spot_ref.fetch_spot_close",
                            lambda d, **k: 42_671.0)
        assert daily_updater._spot_flags_bar(date(2026, 7, 17), 44_714.0) is True

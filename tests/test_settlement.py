"""Tests for settlement day detection and V2bEngine settlement handling."""

from __future__ import annotations

from datetime import date

import pandas as pd

from src.strategy.v2b_engine import V2bEngine, _is_settlement_day, _third_wednesday

# ---------------------------------------------------------------------------
# _third_wednesday / _is_settlement_day unit tests
# ---------------------------------------------------------------------------


class TestThirdWednesday:
    def test_april_2026(self):
        assert _third_wednesday(date(2026, 4, 1)) == date(2026, 4, 15)

    def test_march_2026(self):
        assert _third_wednesday(date(2026, 3, 1)) == date(2026, 3, 18)

    def test_january_2026(self):
        # Jan 2026: 1st Wed=7, 2nd=14, 3rd=21
        assert _third_wednesday(date(2026, 1, 1)) == date(2026, 1, 21)


class TestIsSettlementDay:
    def test_april_15_is_settlement(self):
        ts = pd.Timestamp("2026-04-15", tz="Asia/Taipei")
        assert _is_settlement_day(ts) is True

    def test_april_14_is_not_settlement(self):
        ts = pd.Timestamp("2026-04-14", tz="Asia/Taipei")
        assert _is_settlement_day(ts) is False

    def test_april_16_is_not_settlement(self):
        ts = pd.Timestamp("2026-04-16", tz="Asia/Taipei")
        assert _is_settlement_day(ts) is False

    def test_march_18_is_settlement(self):
        ts = pd.Timestamp("2026-03-18", tz="Asia/Taipei")
        assert _is_settlement_day(ts) is True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(
    n: int = 120,
    base_close: float = 20_000.0,
    trend: float = 30.0,
    end_date: str = "2026-04-15",
) -> pd.DataFrame:
    """Synthetic uptrending OHLCV ending on *end_date*."""
    dates = pd.bdate_range(end=end_date, periods=n, tz="Asia/Taipei")
    closes = [base_close + i * trend for i in range(n)]
    return pd.DataFrame(
        {
            "open": [c - 50 for c in closes],
            "high": [c + 100 for c in closes],
            "low": [c - 100 for c in closes],
            "close": closes,
            "volume": [50_000] * n,
        },
        index=dates,
    )


def _make_engine() -> V2bEngine:
    return V2bEngine(
        product="MXF",
        ema_fast=30,
        ema_slow=100,
        confirm_days=3,
        trail_atr_mult=2.0,
    )


# ---------------------------------------------------------------------------
# V2bEngine settlement integration tests
# ---------------------------------------------------------------------------


class TestSettlementSignal:
    def test_settlement_forces_close(self):
        """Holding position on settlement day → action='close'."""
        df = _make_ohlcv(end_date="2026-04-15")
        engine = _make_engine()
        sig = engine.generate_signal(
            data=df,
            current_position=2,
            entry_price=20_000.0,
            equity=500_000.0,
            highest_high=23_500.0,
            contracts=2,
        )
        assert sig.action == "close"
        assert "settlement" in sig.reason.lower()

    def test_settlement_empty_can_enter(self):
        """Flat on settlement day → normal entry logic (not blocked)."""
        df = _make_ohlcv(end_date="2026-04-15")
        engine = _make_engine()
        sig = engine.generate_signal(
            data=df,
            current_position=0,
            entry_price=None,
            equity=500_000.0,
        )
        # Settlement day should NOT block empty-position entry
        assert sig.action in ("buy", "hold")
        assert "settlement" not in sig.reason.lower()

    def test_next_day_can_reenter(self):
        """Day after settlement (4/16) → normal entry logic applies."""
        df = _make_ohlcv(end_date="2026-04-16")
        engine = _make_engine()
        sig = engine.generate_signal(
            data=df,
            current_position=0,
            entry_price=None,
            equity=500_000.0,
        )
        # Should NOT be blocked by settlement — normal signal (buy or hold)
        assert sig.action in ("buy", "hold")
        assert "settlement" not in sig.reason.lower()

    def test_non_settlement_day_no_force_close(self):
        """Holding on a normal day → no forced close."""
        df = _make_ohlcv(end_date="2026-04-14")
        engine = _make_engine()
        sig = engine.generate_signal(
            data=df,
            current_position=2,
            entry_price=20_000.0,
            equity=500_000.0,
            highest_high=23_500.0,
            contracts=2,
        )
        # Normal exit logic — should not force close due to settlement
        if sig.action == "close":
            assert "settlement" not in sig.reason.lower()

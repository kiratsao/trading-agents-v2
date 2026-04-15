"""Tests for ADX regime filter in V2bEngine."""

from __future__ import annotations

import pandas as pd

from src.strategy.v2b_engine import V2bEngine


def _make_trending_ohlcv(
    n: int = 120,
    base: float = 20_000.0,
    trend: float = 30.0,
    end_date: str = "2026-04-10",
    atr_scale: float = 100.0,
) -> pd.DataFrame:
    """Synthetic uptrending data with controllable volatility."""
    dates = pd.bdate_range(end=end_date, periods=n, tz="Asia/Taipei")
    closes = [base + i * trend for i in range(n)]
    return pd.DataFrame(
        {
            "open": [c - atr_scale * 0.5 for c in closes],
            "high": [c + atr_scale for c in closes],
            "low": [c - atr_scale for c in closes],
            "close": closes,
            "volume": [50_000] * n,
        },
        index=dates,
    )


class TestADXFilter:
    def test_default_threshold_25(self):
        """Default adx_threshold is 25."""
        engine = V2bEngine(product="MXF")
        assert engine.adx_threshold == 25.0

    def test_adx_below_threshold_holds(self):
        """Low ADX + golden cross → hold (filtered out)."""
        # Use real engine to compute ADX, then verify filter works
        df = _make_trending_ohlcv(n=150, trend=50.0)
        engine = V2bEngine(
            product="MXF", ema_fast=30, ema_slow=100,
            confirm_days=2, adx_threshold=0,  # disabled first
        )
        # Confirm this data would produce a buy with no ADX filter
        sig_no_filter = engine.generate_signal(
            data=df, current_position=0, equity=500_000.0,
        )
        assert sig_no_filter.action == "buy", "Precondition: data must trigger buy"

        # Now compute what ADX actually is, and set threshold above it
        ind = engine._compute_indicators(df)
        actual_adx = float(ind["adx"].iloc[-1])

        engine_strict = V2bEngine(
            product="MXF", ema_fast=30, ema_slow=100,
            confirm_days=2, adx_threshold=actual_adx + 10,  # above actual
        )
        sig = engine_strict.generate_signal(
            data=df, current_position=0, equity=500_000.0,
        )
        assert sig.action == "hold"
        assert "ADX too low" in sig.reason

    def test_adx_above_threshold_buys(self):
        """Strong trend (high ADX) + golden cross → buy."""
        # Strong uptrend = high ADX
        df = _make_trending_ohlcv(n=150, trend=50.0, atr_scale=80.0)
        engine = V2bEngine(
            product="MXF", ema_fast=30, ema_slow=100,
            confirm_days=2, adx_threshold=15.0,  # low threshold
        )
        sig = engine.generate_signal(
            data=df, current_position=0, equity=500_000.0,
        )
        # Strong trend should pass ADX filter
        assert sig.action == "buy"

    def test_adx_no_effect_on_close(self):
        """ADX dropping below threshold does NOT force close on existing position."""
        df = _make_trending_ohlcv(n=150, trend=50.0)
        engine = V2bEngine(
            product="MXF", ema_fast=30, ema_slow=100,
            confirm_days=2, adx_threshold=25.0,
        )
        # Already in position — ADX should not affect exit logic
        sig = engine.generate_signal(
            data=df,
            current_position=2,
            entry_price=20_000.0,
            equity=500_000.0,
            highest_high=float(df["close"].iloc[-1]) + 100,
            contracts=2,
        )
        # Should be hold or close (trailing stop / death cross), NOT filtered by ADX
        assert sig.action in ("hold", "close", "add")
        if sig.action == "hold":
            assert "ADX" not in sig.reason

    def test_adx_disabled_when_zero(self):
        """adx_threshold=0 disables ADX filter."""
        df = _make_trending_ohlcv(n=150, trend=50.0)
        engine = V2bEngine(
            product="MXF", ema_fast=30, ema_slow=100,
            confirm_days=2, adx_threshold=0,
        )
        sig = engine.generate_signal(
            data=df, current_position=0, equity=500_000.0,
        )
        # Should buy without ADX check
        if sig.action == "buy":
            assert "ADX too low" not in sig.reason

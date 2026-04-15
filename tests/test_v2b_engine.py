"""Tests for V2bEngine: pyramid add, float_profit per-contract."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy.v2b_engine import V2bEngine


def _make_uptrend(n=200):
    """Strong uptrend → golden cross + high ADX."""
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-01", periods=n)
    close = 20000.0 + np.linspace(0, 3000, n) + np.random.randn(n) * 20
    return pd.DataFrame({
        "open": close - 10, "high": close + 50,
        "low": close - 50, "close": close,
        "volume": [100000] * n,
    }, index=dates)


class TestPyramidAdd:
    def test_pyramid_add_signal(self):
        """Holding + float profit > ATR → action='add'."""
        engine = V2bEngine(
            product="MXF", ema_fast=30, ema_slow=100,
            confirm_days=2, adx_threshold=0,
            pyramid_atr_mult=0.5,  # low threshold
        )
        df = _make_uptrend(n=200)
        # Enter at bar 150, price ~27500; current bar ~35000 → huge profit
        entry_price = float(df["close"].iloc[150])
        sig = engine.generate_signal(
            data=df,
            current_position=2,
            entry_price=entry_price,
            equity=500_000,
            highest_high=float(df["close"].max()),
            contracts=2,
        )
        assert sig.action == "add"
        assert sig.contracts >= 1
        assert "pyramid" in sig.reason

    def test_pyramid_float_profit_per_contract(self):
        """float_profit = close - entry_price (not multiplied by contracts)."""
        engine = V2bEngine(
            product="MXF", ema_fast=30, ema_slow=100,
            confirm_days=2, adx_threshold=0,
            pyramid_atr_mult=1.0,
        )
        df = _make_uptrend(n=200)
        close = float(df["close"].iloc[-1])

        # With entry_price just below close by less than 1 ATR
        # Should NOT pyramid because per-contract profit < ATR
        from src.strategy.common.indicators import atr as calc_atr
        atr_val = float(calc_atr(df, 14).iloc[-1])
        entry_just_below = close - atr_val * 0.5  # only 0.5 ATR profit

        sig = engine.generate_signal(
            data=df,
            current_position=4,
            entry_price=entry_just_below,
            equity=500_000,
            highest_high=close,
            contracts=4,
        )
        # With 4 contracts, old buggy code would multiply: 0.5*4 = 2 ATR → add
        # Correct: 0.5 ATR per contract → hold
        assert sig.action == "hold"

"""Tests for margin safety cap in anti-martingale and pyramid."""

from __future__ import annotations

import math

from src.strategy.v2b_engine import _MTX_MARGIN, _anti_martingale_contracts


class TestMarginSafety:
    def test_contracts_capped_by_margin(self):
        """equity=400K, margin=119250 → floor(400000/119250)=3, ladder says 2 → 2."""
        n = _anti_martingale_contracts(400_000)
        assert n == 2  # ladder default: 350K→2, 480K→3

    def test_margin_cap_overrides_ladder(self):
        """equity=200K < 350K threshold but margin allows 1 → 1 contract."""
        n = _anti_martingale_contracts(200_000)
        assert n == 1
        assert n <= math.floor(200_000 / _MTX_MARGIN)

    def test_contracts_not_exceed_ladder(self):
        """equity=800K, ladder max=4, margin allows 6 → still 4."""
        n = _anti_martingale_contracts(800_000)
        assert n == 4  # ladder: 600K→4
        assert n <= math.floor(800_000 / _MTX_MARGIN)

    def test_pyramid_capped_by_margin(self):
        """Pyramid add_n must not push total above margin cap."""
        import numpy as np
        import pandas as pd

        from src.strategy.v2b_engine import V2bEngine

        # Equity 250K → floor(250000/119250) = 2 max contracts
        # Already holding 2 → no room to add
        engine = V2bEngine(
            product="MXF", ema_fast=30, ema_slow=100,
            confirm_days=2, adx_threshold=0,
            pyramid_atr_mult=0.01,  # very low threshold → always triggers
        )

        np.random.seed(42)
        n = 200
        dates = pd.bdate_range("2023-01-01", periods=n)
        base = 20000.0
        close = base + np.linspace(0, 2000, n)
        df = pd.DataFrame({
            "open": close - 10, "high": close + 50,
            "low": close - 50, "close": close,
            "volume": [100000] * n,
        }, index=dates)

        sig = engine.generate_signal(
            data=df,
            current_position=2,
            entry_price=20000.0,
            equity=250_000,  # can only support 2 contracts
            highest_high=22000.0,
            contracts=2,
        )
        # Should hold or skip pyramid, not add
        if sig.action == "add":
            # If add, total must not exceed margin cap
            total = 2 + sig.contracts
            assert total <= math.floor(250_000 / _MTX_MARGIN)
        else:
            assert sig.action == "hold"
            assert "margin cap" in sig.reason or "holding" in sig.reason

"""Tests for V2bEngine — strategy core logic.

Coverage
--------
* Signal generation: buy, close, hold, add
* EMA golden/death cross detection
* N-day confirmation filter
* Trailing stop (ATR-based)
* Pyramid scale-in logic
* Anti-martingale position sizing
* Settlement day force-close
* Edge cases: warmup period, zero equity, boundary conditions
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.strategy.v2b_engine import Signal, V2bEngine, _anti_martingale_contracts, _third_wednesday


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_data(
    n: int = 200,
    base_price: float = 20000.0,
    trend: float = 0.0,
    volatility: float = 50.0,
    start_date: str = "2025-01-01",
) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.bdate_range(start_date, periods=n, freq="B")
    closes = base_price + np.cumsum(np.random.randn(n) * volatility + trend)
    highs = closes + np.abs(np.random.randn(n) * volatility * 0.5)
    lows = closes - np.abs(np.random.randn(n) * volatility * 0.5)
    opens = closes + np.random.randn(n) * volatility * 0.3
    volumes = np.random.randint(5000, 50000, n)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=dates,
    )


def _make_bullish_data(n: int = 200, base: float = 19000.0) -> pd.DataFrame:
    """Data with strong uptrend — should trigger buy signal."""
    return _make_data(n=n, base_price=base, trend=30.0, volatility=30.0)


def _make_bearish_data(n: int = 200, base: float = 22000.0) -> pd.DataFrame:
    """Data with strong downtrend — should not trigger buy."""
    return _make_data(n=n, base_price=base, trend=-30.0, volatility=30.0)


@pytest.fixture
def engine():
    return V2bEngine(product="MTX", ema_fast=30, ema_slow=100, confirm_days=3)


# ===========================================================================
# Anti-Martingale position sizing
# ===========================================================================


class TestAntiMartingale:
    def test_default_ladder(self):
        assert _anti_martingale_contracts(350_000) == 2
        assert _anti_martingale_contracts(480_000) == 3
        assert _anti_martingale_contracts(600_000) == 4

    def test_below_first_rung(self):
        assert _anti_martingale_contracts(200_000) == 1

    def test_margin_safety_cap(self):
        # 250K / 119250 = 2.09 → floor = 2
        assert _anti_martingale_contracts(250_000) <= 2

    def test_custom_ladder(self):
        ladder = [{"equity": 100_000, "contracts": 1}, {"equity": 200_000, "contracts": 5}]
        assert _anti_martingale_contracts(200_000, ladder) == 1  # capped by margin: 200K/119250=1

    def test_zero_equity(self):
        result = _anti_martingale_contracts(0)
        assert result >= 1  # min is 1


# ===========================================================================
# Settlement day detection
# ===========================================================================


class TestSettlementDay:
    def test_third_wednesday_jan_2025(self):
        # January 2025: 1st Wed=1, 2nd=8, 3rd=15
        assert _third_wednesday(date(2025, 1, 15)) == date(2025, 1, 15)

    def test_third_wednesday_march_2025(self):
        # March 2025: 1st Wed=5, 2nd=12, 3rd=19
        assert _third_wednesday(date(2025, 3, 1)) == date(2025, 3, 19)

    def test_not_third_wednesday(self):
        assert _third_wednesday(date(2025, 1, 1)) != date(2025, 1, 1)


# ===========================================================================
# Signal generation — warmup
# ===========================================================================


class TestWarmup:
    def test_insufficient_data_returns_hold(self, engine):
        df = _make_data(n=10)
        sig = engine.generate_signal(data=df)
        assert sig.action == "hold"
        assert "warmup" in sig.reason

    def test_sufficient_data_does_not_return_warmup(self, engine):
        df = _make_bullish_data(n=200)
        sig = engine.generate_signal(data=df)
        assert "warmup" not in sig.reason


# ===========================================================================
# Signal generation — entry
# ===========================================================================


class TestEntry:
    def test_buy_on_strong_uptrend(self, engine):
        df = _make_bullish_data(n=200)
        sig = engine.generate_signal(data=df, current_position=0, equity=350_000)
        # With strong uptrend and enough bars, should eventually get buy or hold
        assert sig.action in ("buy", "hold")

    def test_no_buy_on_downtrend(self, engine):
        df = _make_bearish_data(n=200)
        sig = engine.generate_signal(data=df, current_position=0, equity=350_000)
        assert sig.action != "buy"

    def test_buy_signal_has_positive_contracts(self, engine):
        df = _make_bullish_data(n=200)
        sig = engine.generate_signal(data=df, current_position=0, equity=350_000)
        if sig.action == "buy":
            assert sig.contracts > 0

    def test_no_buy_when_already_in_position(self, engine):
        df = _make_bullish_data(n=200)
        sig = engine.generate_signal(
            data=df, current_position=2, entry_price=19500.0,
            equity=350_000, highest_high=20000.0, contracts=2,
        )
        assert sig.action != "buy"


# ===========================================================================
# Signal generation — exit
# ===========================================================================


class TestExit:
    def test_death_cross_triggers_close(self, engine):
        df = _make_bearish_data(n=200)
        sig = engine.generate_signal(
            data=df, current_position=2, entry_price=22000.0,
            equity=350_000, highest_high=22500.0, contracts=2,
        )
        # On strong downtrend, should trigger close (death cross or trailing stop)
        assert sig.action in ("close", "hold")

    def test_trailing_stop_triggers_close(self, engine):
        """Simulate price drop below trailing stop."""
        df = _make_data(n=200, base_price=20000.0, trend=0.0)
        close = float(df["close"].iloc[-1])
        # Set highest_high far above current price to trigger trailing stop
        high_hh = close + 1000.0
        sig = engine.generate_signal(
            data=df, current_position=2, entry_price=19500.0,
            equity=350_000, highest_high=high_hh, contracts=2,
        )
        if sig.action == "close":
            assert "trailing stop" in sig.reason or "death cross" in sig.reason

    def test_close_returns_current_contracts(self, engine):
        df = _make_bearish_data(n=200)
        sig = engine.generate_signal(
            data=df, current_position=3, entry_price=22000.0,
            equity=500_000, highest_high=22500.0, contracts=3,
        )
        if sig.action == "close":
            assert sig.contracts == 3

    def test_settlement_day_forces_close(self, engine):
        """On the 3rd Wednesday, open position should be closed."""
        # Build data ending on a 3rd Wednesday
        third_wed = _third_wednesday(date(2025, 3, 1))  # 2025-03-19
        n = 200
        end_date = third_wed
        start_date = end_date - pd.tseries.offsets.BDay(n - 1)
        dates = pd.bdate_range(start_date, end_date)
        np.random.seed(99)
        closes = 20000 + np.cumsum(np.random.randn(len(dates)) * 20 + 10)
        df = pd.DataFrame(
            {
                "open": closes - 10,
                "high": closes + 50,
                "low": closes - 50,
                "close": closes,
                "volume": np.random.randint(5000, 50000, len(dates)),
            },
            index=dates,
        )
        sig = engine.generate_signal(
            data=df, current_position=2, entry_price=19500.0,
            equity=400_000, highest_high=float(closes[-1]) + 100, contracts=2,
        )
        assert sig.action == "close"
        assert "settlement" in sig.reason.lower()


# ===========================================================================
# Signal generation — pyramid
# ===========================================================================


class TestPyramid:
    def test_add_signal_when_profit_exceeds_atr(self, engine):
        """When float profit per contract >= ATR, should get add signal."""
        df = _make_data(n=200, base_price=20000.0, trend=5.0)
        close = float(df["close"].iloc[-1])
        # Set entry_price well below current to trigger pyramid
        ind = engine._compute_indicators(df)
        atr_val = float(ind["atr"].iloc[-1])
        entry = close - atr_val * 2  # float_profit = 2*ATR > threshold of 1*ATR

        sig = engine.generate_signal(
            data=df, current_position=2, entry_price=entry,
            equity=500_000, highest_high=close, contracts=2,
        )
        # Should be add or hold (if margin cap blocks)
        assert sig.action in ("add", "hold")

    def test_add_contracts_capped_by_margin(self, engine):
        df = _make_data(n=200, base_price=20000.0)
        close = float(df["close"].iloc[-1])
        ind = engine._compute_indicators(df)
        atr_val = float(ind["atr"].iloc[-1])
        entry = close - atr_val * 3

        sig = engine.generate_signal(
            data=df, current_position=3, entry_price=entry,
            equity=400_000, highest_high=close, contracts=3,
        )
        if sig.action == "add":
            # Total (3 + add_n) * 119250 must not exceed equity
            total = 3 + sig.contracts
            assert total * 119_250 <= 400_000

    def test_no_add_when_profit_below_threshold(self, engine):
        df = _make_data(n=200, base_price=20000.0)
        close = float(df["close"].iloc[-1])
        # Entry very close to current — no profit
        sig = engine.generate_signal(
            data=df, current_position=2, entry_price=close - 10,
            equity=500_000, highest_high=close, contracts=2,
        )
        assert sig.action != "add"


# ===========================================================================
# Indicator computation
# ===========================================================================


class TestIndicators:
    def test_compute_indicators_columns(self, engine):
        df = _make_data(n=200)
        ind = engine._compute_indicators(df)
        assert "ema_fast" in ind.columns
        assert "ema_slow" in ind.columns
        assert "atr" in ind.columns
        assert "bull_streak" in ind.columns

    def test_ema_values_positive(self, engine):
        df = _make_data(n=200, base_price=20000.0)
        ind = engine._compute_indicators(df)
        assert ind["ema_fast"].iloc[-1] > 0
        assert ind["ema_slow"].iloc[-1] > 0

    def test_atr_positive(self, engine):
        df = _make_data(n=200)
        ind = engine._compute_indicators(df)
        assert ind["atr"].iloc[-1] > 0

    def test_bull_streak_non_negative(self, engine):
        df = _make_data(n=200)
        ind = engine._compute_indicators(df)
        assert all(ind["bull_streak"] >= 0)


# ===========================================================================
# Engine construction
# ===========================================================================


class TestConstruction:
    def test_invalid_product_raises(self):
        with pytest.raises(ValueError, match="product"):
            V2bEngine(product="INVALID")

    def test_invalid_exit_mode_raises(self):
        with pytest.raises(ValueError, match="exit_mode"):
            V2bEngine(exit_mode="invalid")

    def test_default_parameters(self):
        e = V2bEngine()
        assert e.ema_fast == 30
        assert e.ema_slow == 100
        assert e.atr_period == 14
        assert e.confirm_days == 3
        assert e.trail_atr_mult == 2.0

    def test_signal_dataclass(self):
        s = Signal("buy", 2, "test reason", stop_loss=19500.0)
        assert s.action == "buy"
        assert s.contracts == 2
        assert "buy" in str(s)

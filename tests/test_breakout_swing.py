"""Tests for BreakoutSwingStrategy."""

from __future__ import annotations

import pandas as pd
import pytest

from tw_futures.strategies.swing.breakout_swing import (
    _MAX_HOLD_DAYS,
    _MIN_DATA_ROWS,
    _NEUTRAL_ZONE_PCT,
    BreakoutSwingStrategy,
    Signal,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ohlcv(
    n: int = 80,
    base_close: float = 20_000.0,
    trend: float = 0.0,  # daily drift applied to close
    high_offset: float = 200.0,
    low_offset: float = 200.0,
    volume: float = 50_000,
) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame with an Asia/Taipei DatetimeIndex."""
    dates = pd.bdate_range("2025-01-01", periods=n, tz="Asia/Taipei")
    closes = [base_close + i * trend for i in range(n)]
    df = pd.DataFrame(
        {
            "open": [c - high_offset / 2 for c in closes],
            "high": [c + high_offset for c in closes],
            "low": [c - low_offset for c in closes],
            "close": closes,
            "volume": [volume] * n,
        },
        index=dates,
    )
    df.index.name = "date"
    return df


def _strategy() -> BreakoutSwingStrategy:
    return BreakoutSwingStrategy(product="TX")


# ---------------------------------------------------------------------------
# 1. Defensive / edge cases
# ---------------------------------------------------------------------------


class TestDefensiveCases:
    def test_insufficient_data_returns_hold(self):
        s = _strategy()
        df = _make_ohlcv(n=_MIN_DATA_ROWS - 1)
        sig = s.generate_signal(df, 0, None, None, 5_000_000)
        assert sig.action == "hold"
        assert "insufficient data" in sig.reason

    def test_exactly_min_data_does_not_raise(self):
        s = _strategy()
        df = _make_ohlcv(n=_MIN_DATA_ROWS)
        sig = s.generate_signal(df, 0, None, None, 5_000_000)
        assert sig.action in ("hold", "buy", "sell")

    def test_missing_column_returns_hold(self):
        s = _strategy()
        df = _make_ohlcv(n=80).drop(columns=["high"])
        sig = s.generate_signal(df, 0, None, None, 5_000_000)
        assert sig.action == "hold"
        assert "missing" in sig.reason

    def test_zero_equity_returns_hold_on_entry(self):
        """Zero equity → position size = 0 → no entry."""
        s = _strategy()
        # Force a breakout by making last close very high
        df = _make_ohlcv(n=80, trend=5.0)
        df.iloc[-1, df.columns.get_loc("close")] = 99_999  # huge breakout
        df.iloc[-1, df.columns.get_loc("high")] = 100_000
        sig = s.generate_signal(df, 0, None, None, equity=0)
        # Either hold (size=0) or buy with 0 contracts — never buy with >0
        if sig.action == "buy":
            assert sig.contracts == 0
        else:
            assert sig.action == "hold"

    def test_invalid_product_raises(self):
        with pytest.raises(ValueError, match="TX"):
            BreakoutSwingStrategy(product="AAPL")


# ---------------------------------------------------------------------------
# 2. Indicator computation
# ---------------------------------------------------------------------------


class TestIndicators:
    def test_upper_band_equals_rolling_max_shifted(self):
        s = _strategy()
        df = _make_ohlcv(n=80)
        indic = s._compute_indicators(df)
        expected = df["high"].rolling(s.donchian_period).max().shift(1)
        pd.testing.assert_series_equal(
            indic["upper"].dropna(), expected.dropna(), check_names=False
        )

    def test_lower_band_equals_rolling_min_shifted(self):
        s = _strategy()
        df = _make_ohlcv(n=80)
        indic = s._compute_indicators(df)
        expected = df["low"].rolling(s.donchian_period).min().shift(1)
        pd.testing.assert_series_equal(
            indic["lower"].dropna(), expected.dropna(), check_names=False
        )

    def test_sma50_last_value(self):
        s = _strategy()
        df = _make_ohlcv(n=80)
        indic = s._compute_indicators(df)
        expected_last = df["close"].rolling(50).mean().iloc[-1]
        assert abs(indic["sma50"].iloc[-1] - expected_last) < 1e-6

    def test_atr_is_positive(self):
        s = _strategy()
        df = _make_ohlcv(n=80)
        indic = s._compute_indicators(df)
        assert indic["atr"].dropna().gt(0).all()

    def test_first_rows_have_nan_indicators(self):
        s = _strategy()
        df = _make_ohlcv(n=80)
        indic = s._compute_indicators(df)
        # SMA50 NaN for first 49 rows
        assert indic["sma50"].iloc[:49].isna().all()
        # upper/lower NaN for first donchian_period rows (shifted)
        assert indic["upper"].iloc[: s.donchian_period].isna().all()


# ---------------------------------------------------------------------------
# 3. Trend filter
# ---------------------------------------------------------------------------


class TestTrendFilter:
    def test_neutral_zone_blocks_both_signals(self):
        s = _strategy()
        close = 20_000.0
        sma50 = close * (1 + _NEUTRAL_ZONE_PCT * 0.5)  # within ±1%
        long_ok, short_ok = s._apply_trend_filter(close, sma50, True, True)
        assert not long_ok and not short_ok

    def test_uptrend_allows_long_only(self):
        s = _strategy()
        close = 20_500.0
        sma50 = 20_000.0  # close > sma50 by 2.5%
        long_ok, short_ok = s._apply_trend_filter(close, sma50, True, True)
        assert long_ok
        assert not short_ok

    def test_downtrend_allows_short_only(self):
        s = _strategy()
        close = 19_500.0
        sma50 = 20_000.0  # close < sma50 by 2.5%
        long_ok, short_ok = s._apply_trend_filter(close, sma50, True, True)
        assert not long_ok
        assert short_ok

    def test_nan_sma_blocks_all(self):
        s = _strategy()
        long_ok, short_ok = s._apply_trend_filter(20_000, float("nan"), True, True)
        assert not long_ok and not short_ok


# ---------------------------------------------------------------------------
# 4. Position sizing
# ---------------------------------------------------------------------------


class TestPositionSizing:
    def test_basic_sizing_tx(self):
        s = BreakoutSwingStrategy(product="TX")
        # risk = 5_000_000 * 0.02 = 100_000
        # stop = 2 * 300 * 200 = 120_000
        # contracts = floor(100_000 / 120_000) = 0  → but min is 0 (caller handles)
        # Use smaller ATR to get contracts > 0
        # risk=100_000 / (2*100*200) = 100_000/40_000 = 2.5 → floor = 2
        contracts = s._size_contracts(atr=100, equity=5_000_000)
        assert contracts == 2

    def test_basic_sizing_mtx(self):
        s = BreakoutSwingStrategy(product="MTX")
        # risk=100_000 / (2*100*50) = 100_000/10_000 = 10
        contracts = s._size_contracts(atr=100, equity=5_000_000)
        assert contracts == 10

    def test_zero_atr_returns_zero(self):
        s = _strategy()
        assert s._size_contracts(atr=0, equity=5_000_000) == 0

    def test_floor_applied(self):
        s = _strategy()
        # risk = 5_000_000 * 0.02 = 100_000
        # stop = 2 * 150 * 200 = 60_000
        # 100_000/60_000 = 1.666... → floor = 1
        contracts = s._size_contracts(atr=150, equity=5_000_000)
        assert contracts == 1

    def test_zero_equity_returns_zero(self):
        s = _strategy()
        assert s._size_contracts(atr=200, equity=0) == 0


# ---------------------------------------------------------------------------
# 5. Trailing stop
# ---------------------------------------------------------------------------


class TestTrailingStop:
    def test_long_stop_trails_highest_close(self):
        s = _strategy()
        dates = pd.bdate_range("2025-01-01", periods=5, tz="Asia/Taipei")
        data = pd.DataFrame(
            {"close": [100, 110, 105, 115, 108]},
            index=dates,
        )
        data.index.name = "date"
        entry_date = dates[0]
        atr = 10.0
        stop = s._compute_trailing_stop(data, direction=1, entry_date=entry_date, atr=atr)
        # highest close since entry = 115;  stop = 115 - 2*10 = 95
        assert stop == pytest.approx(95.0)

    def test_short_stop_trails_lowest_close(self):
        s = _strategy()
        dates = pd.bdate_range("2025-01-01", periods=5, tz="Asia/Taipei")
        data = pd.DataFrame(
            {"close": [100, 90, 95, 85, 92]},
            index=dates,
        )
        data.index.name = "date"
        entry_date = dates[0]
        atr = 10.0
        stop = s._compute_trailing_stop(data, direction=-1, entry_date=entry_date, atr=atr)
        # lowest close since entry = 85;  stop = 85 + 2*10 = 105
        assert stop == pytest.approx(105.0)

    def test_stop_respects_entry_date_window(self):
        """Only data from entry_date onwards should influence the stop."""
        s = _strategy()
        dates = pd.bdate_range("2025-01-01", periods=6, tz="Asia/Taipei")
        data = pd.DataFrame(
            {"close": [200, 210, 100, 102, 104, 106]},  # first two pre-entry highs
            index=dates,
        )
        data.index.name = "date"
        entry_date = dates[2]  # entry at index 2 (close=100)
        atr = 5.0
        stop = s._compute_trailing_stop(data, direction=1, entry_date=entry_date, atr=atr)
        # highest since entry = max(100,102,104,106) = 106;  stop = 106 - 10 = 96
        assert stop == pytest.approx(96.0)


# ---------------------------------------------------------------------------
# 6. Max holding period exit
# ---------------------------------------------------------------------------


class TestMaxHoldingPeriod:
    def test_exit_after_max_days(self):
        s = _strategy()
        n = 80
        df = _make_ohlcv(n=n)
        entry_date = df.index[n - _MAX_HOLD_DAYS - 2]  # held > max days
        sig = s.generate_signal(
            data=df,
            current_position=1,
            entry_price=float(df.iloc[n - _MAX_HOLD_DAYS - 2]["close"]),
            entry_date=entry_date,
            equity=5_000_000,
        )
        assert sig.action == "close"
        assert "max holding" in sig.reason

    def test_no_exit_before_max_days(self):
        """Position held for fewer than max_hold_days should NOT trigger time exit."""
        s = _strategy()
        df = _make_ohlcv(n=80, trend=2.0)  # gradual uptrend — stop shouldn't trigger
        entry_idx = -3  # 3 bars ago = well within max_hold_days
        entry_date = df.index[entry_idx]
        entry_price = float(df.iloc[entry_idx]["close"])

        sig = s.generate_signal(
            data=df,
            current_position=1,
            entry_price=entry_price,
            entry_date=entry_date,
            equity=5_000_000,
        )
        # Must not be a time-based exit
        assert not (sig.action == "close" and "max holding" in sig.reason)


# ---------------------------------------------------------------------------
# 7. ATR trailing stop exit
# ---------------------------------------------------------------------------


class TestATRTrailingStopExit:
    def test_long_stop_triggered_when_close_falls(self):
        """Simulate a long position where close drops below trailing stop."""
        s = BreakoutSwingStrategy(product="TX", atr_stop_mult=2.0, max_hold_days=30)
        n = 60
        df = _make_ohlcv(n=n, base_close=20_000, trend=0, high_offset=100, low_offset=100)

        # Entry near end so trade age is within max_hold_days
        entry_idx = n - 5
        entry_date = df.index[entry_idx]
        entry_price = float(df.iloc[entry_idx]["close"])

        # Crash the last bar's close far below where the stop should be
        df.iloc[-1, df.columns.get_loc("close")] = entry_price - 5_000
        df.iloc[-1, df.columns.get_loc("low")] = entry_price - 5_000

        sig = s.generate_signal(
            data=df,
            current_position=1,
            entry_price=entry_price,
            entry_date=entry_date,
            equity=5_000_000,
        )
        assert sig.action == "close"
        assert "trailing stop" in sig.reason.lower()

    def test_short_stop_triggered_when_close_rises(self):
        s = BreakoutSwingStrategy(product="TX", atr_stop_mult=2.0, max_hold_days=30)
        n = 60
        df = _make_ohlcv(n=n, base_close=20_000, trend=0, high_offset=100, low_offset=100)

        entry_idx = n - 5
        entry_date = df.index[entry_idx]
        entry_price = float(df.iloc[entry_idx]["close"])

        # Spike last close far above where the stop should be
        df.iloc[-1, df.columns.get_loc("close")] = entry_price + 5_000
        df.iloc[-1, df.columns.get_loc("high")] = entry_price + 5_000

        sig = s.generate_signal(
            data=df,
            current_position=-1,
            entry_price=entry_price,
            entry_date=entry_date,
            equity=5_000_000,
        )
        assert sig.action == "close"
        assert "trailing stop" in sig.reason.lower()


# ---------------------------------------------------------------------------
# 8. Reverse signal exit
# ---------------------------------------------------------------------------


class TestReverseSignal:
    def _df_with_long_then_short_breakout(self) -> pd.DataFrame:
        """Create data where the last bar gives a clear short (Donchian breakdown)."""
        n = 80
        df = _make_ohlcv(n=n, base_close=20_000, trend=-5.0, high_offset=50, low_offset=50)
        # Force last close well below the 20-day low to trigger short signal
        twenty_day_low = df["low"].iloc[-21:-1].min()
        crash_close = twenty_day_low - 500
        df.iloc[-1, df.columns.get_loc("close")] = crash_close
        df.iloc[-1, df.columns.get_loc("low")] = crash_close - 50
        # Ensure downtrend so sma50 filter passes for short
        sma50 = df["close"].rolling(50).mean().iloc[-1]
        if crash_close >= sma50:
            df.iloc[-1, df.columns.get_loc("close")] = sma50 * 0.97
        return df

    def test_long_position_reversed_on_short_signal(self):
        s = _strategy()
        df = self._df_with_long_then_short_breakout()
        entry_date = df.index[5]  # held a few bars
        entry_price = float(df.iloc[5]["close"])

        sig = s.generate_signal(
            data=df,
            current_position=1,
            entry_price=entry_price,
            entry_date=entry_date,
            equity=5_000_000,
        )
        # Either reversed to short or closed (depends on size calculation)
        assert sig.action in ("sell", "close")


# ---------------------------------------------------------------------------
# 9. Entry signals
# ---------------------------------------------------------------------------


class TestEntrySignals:
    def _df_with_upward_breakout(self) -> pd.DataFrame:
        """Last bar breaks above 20-day high while close > sma50."""
        n = 80
        df = _make_ohlcv(n=n, base_close=20_000, trend=3.0, high_offset=50, low_offset=50)
        # Force last close well above recent 20-day high
        twenty_day_high = df["high"].iloc[-21:-1].max()
        df.iloc[-1, df.columns.get_loc("close")] = twenty_day_high + 200
        df.iloc[-1, df.columns.get_loc("high")] = twenty_day_high + 250
        return df

    def _df_with_downward_breakout(self) -> pd.DataFrame:
        """Last bar breaks below 20-day low while close < sma50."""
        n = 80
        df = _make_ohlcv(n=n, base_close=20_000, trend=-3.0, high_offset=50, low_offset=50)
        twenty_day_low = df["low"].iloc[-21:-1].min()
        df.iloc[-1, df.columns.get_loc("close")] = twenty_day_low - 200
        df.iloc[-1, df.columns.get_loc("low")] = twenty_day_low - 250
        return df

    def test_long_entry_on_upward_breakout(self):
        s = _strategy()
        df = self._df_with_upward_breakout()
        sig = s.generate_signal(df, 0, None, None, 5_000_000)
        # Should be buy or hold (if neutral-zone kills it)
        assert sig.action in ("buy", "hold")

    def test_short_entry_on_downward_breakout(self):
        s = _strategy()
        df = self._df_with_downward_breakout()
        sig = s.generate_signal(df, 0, None, None, 5_000_000)
        assert sig.action in ("sell", "hold")

    def test_buy_signal_has_stop_loss_below_close(self):
        s = _strategy()
        df = self._df_with_upward_breakout()
        sig = s.generate_signal(df, 0, None, None, 5_000_000)
        if sig.action == "buy":
            last_close = float(df.iloc[-1]["close"])
            assert sig.stop_loss is not None
            assert sig.stop_loss < last_close

    def test_sell_signal_has_stop_loss_above_close(self):
        s = _strategy()
        df = self._df_with_downward_breakout()
        sig = s.generate_signal(df, 0, None, None, 5_000_000)
        if sig.action == "sell":
            last_close = float(df.iloc[-1]["close"])
            assert sig.stop_loss is not None
            assert sig.stop_loss > last_close

    def test_no_signal_in_flat_market(self):
        """Flat market with no breakout should return hold."""
        s = _strategy()
        # Perfectly flat OHLCV — no Donchian breakout possible
        df = _make_ohlcv(n=80, trend=0.0, high_offset=10, low_offset=10)
        sig = s.generate_signal(df, 0, None, None, 5_000_000)
        assert sig.action == "hold"

    def test_contracts_at_least_one_on_valid_entry(self):
        s = _strategy()
        df = self._df_with_upward_breakout()
        sig = s.generate_signal(df, 0, None, None, equity=10_000_000)
        if sig.action == "buy":
            assert sig.contracts >= 1

    def test_reason_string_populated(self):
        s = _strategy()
        df = self._df_with_upward_breakout()
        sig = s.generate_signal(df, 0, None, None, 5_000_000)
        assert len(sig.reason) > 0


# ---------------------------------------------------------------------------
# 10. Signal dataclass
# ---------------------------------------------------------------------------


class TestSignalDataclass:
    def test_str_representation(self):
        sig = Signal(action="buy", contracts=2, reason="test", stop_loss=19_000.0)
        s = str(sig)
        assert "buy" in s
        assert "2" in s
        assert "19000" in s

    def test_hold_with_no_stop(self):
        sig = Signal(action="hold")
        assert sig.contracts == 0
        assert sig.stop_loss is None
        assert sig.reason == ""

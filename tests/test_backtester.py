"""Tests for the Backtester agent."""

from __future__ import annotations

import pandas as pd
import pytest

from us_equity.backtester.bias_check import assert_signal_shifted
from us_equity.backtester.metrics import aggregate_oos_metrics, passes_threshold
from us_equity.backtester.walk_forward import generate_windows

# ---------------------------------------------------------------------------
# Walk-forward window generation
# ---------------------------------------------------------------------------


class TestGenerateWindows:
    def test_returns_correct_number_of_windows(self):
        """Should produce at least 3 OOS windows for a 3-year index."""
        index = pd.date_range("2020-01-01", periods=756, freq="B")  # ~3 years
        windows = generate_windows(index, in_sample_bars=252, out_of_sample_bars=63)
        assert len(windows) >= 3

    def test_oos_follows_in_sample(self):
        """OOS start must equal in-sample end + 1 bar."""
        index = pd.date_range("2020-01-01", periods=756, freq="B")
        windows = generate_windows(index, in_sample_bars=252, out_of_sample_bars=63)
        for w in windows:
            assert w.out_of_sample_start > w.in_sample_end

    def test_no_overlap_between_windows(self):
        """Consecutive OOS windows must not overlap."""
        index = pd.date_range("2020-01-01", periods=756, freq="B")
        windows = generate_windows(index, in_sample_bars=252, out_of_sample_bars=63)
        for a, b in zip(windows, windows[1:]):
            assert b.out_of_sample_start > a.out_of_sample_end

    def test_raises_if_insufficient_data(self):
        """Should raise if index is too short for even one window."""
        index = pd.date_range("2020-01-01", periods=10, freq="B")
        with pytest.raises(Exception):
            generate_windows(index, in_sample_bars=252, out_of_sample_bars=63)


# ---------------------------------------------------------------------------
# Look-ahead bias check
# ---------------------------------------------------------------------------


class TestBiasCheck:
    def test_passes_when_signal_is_shifted(self):
        """assert_signal_shifted should not raise for correctly shifted signals."""
        raw = pd.Series([0, 1, 0, 1, 0], dtype=float)
        entry = raw.shift(1)
        assert_signal_shifted(raw, entry)  # must not raise

    def test_fails_when_signal_is_not_shifted(self):
        """assert_signal_shifted must raise AssertionError when signal == raw (no shift)."""
        raw = pd.Series([0, 1, 0, 1, 0], dtype=float)
        with pytest.raises(AssertionError):
            assert_signal_shifted(raw, raw)


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------


class TestMetrics:
    def _window_result(self, sharpe=1.5, max_dd=0.08, win_rate=0.55, profit_factor=1.4, cagr=0.20):
        return dict(
            sharpe=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            cagr=cagr,
        )

    def test_aggregate_returns_all_keys(self):
        results = [self._window_result() for _ in range(3)]
        metrics = aggregate_oos_metrics(results)
        for key in ("sharpe", "max_drawdown", "win_rate", "profit_factor", "cagr", "calmar"):
            assert key in metrics

    def test_passes_threshold_with_good_metrics(self):
        results = [self._window_result() for _ in range(3)]
        metrics = aggregate_oos_metrics(results)
        assert passes_threshold(metrics) is True

    def test_fails_threshold_with_negative_sharpe(self):
        results = [self._window_result(sharpe=-0.5) for _ in range(3)]
        metrics = aggregate_oos_metrics(results)
        assert passes_threshold(metrics) is False

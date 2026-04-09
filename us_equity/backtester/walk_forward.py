"""Walk-Forward Validation — out-of-sample robustness testing.

Slides a rolling (train_months + test_months) window across the full data
range, running Backtester.run() on each *test* period with the full historical
data dict available for strategy warmup.  All test periods are chained into a
single OOS equity curve so overall OOS performance can be evaluated.

For parameter-free strategies like Dual Momentum, the train window is used
purely as a timing reference (ensuring there is sufficient prior history before
each test period begins) rather than for parameter optimisation.

Window generation example (train=12, test=3, 2023-01-01 → 2025-12-31):

    Window 1  train: 2023-01-01→2023-12-31  test: 2024-01-01→2024-03-31
    Window 2  train: 2023-04-01→2024-03-31  test: 2024-04-01→2024-06-30
    ...
    Window 8  train: 2024-10-01→2025-09-30  test: 2025-10-01→2025-12-31
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd
from pandas.tseries.offsets import DateOffset

from core.risk.slippage import DynamicSlippage
from us_equity.backtester.backtester import Backtester, SignalGenerator

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardResult:
    """All outputs from a walk-forward validation run.

    Attributes
    ----------
    windows:
        One dict per test window.  Keys:
        ``window``, ``train_start``, ``train_end``, ``test_start``,
        ``test_end``, ``metrics``, ``equity_curve``.
    oos_equity_curve:
        Chained equity curve across all test periods.  Each window's curve
        starts where the previous one ended, so the series represents a
        continuously compounded OOS portfolio.
    oos_metrics:
        Aggregate performance over the full OOS period.  Keys:
        ``Total Return (%)``, ``CAGR (%)``, ``Sharpe Ratio``,
        ``Max Drawdown (%)``.
    consistency_ratio:
        Fraction of test windows that produced a positive return.
        Range [0.0, 1.0].  A ratio < 0.5 suggests the strategy is not
        reliably profitable across market regimes.
    """

    windows: list[dict] = field(default_factory=list)
    oos_equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    oos_metrics: dict = field(default_factory=dict)
    consistency_ratio: float = 0.0


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class WalkForwardValidator:
    """Rolling walk-forward validator for monthly-rebalance strategies.

    Parameters
    ----------
    commission_rate:
        One-way transaction cost passed through to :class:`Backtester`
        (default 0.001 = 0.1 %).

    Example
    -------
    >>> wfv = WalkForwardValidator()
    >>> result = wfv.run(strategy, data, "2023-01-01", "2025-12-31")
    >>> print(result.consistency_ratio)
    """

    def __init__(
        self,
        commission_rate: float = 0.001,
        slippage_model: DynamicSlippage | None = None,
    ) -> None:
        self.commission_rate = commission_rate
        self.slippage_model = slippage_model

    def run(
        self,
        strategy: SignalGenerator,
        data: dict[str, pd.DataFrame],
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        train_months: int = 12,
        test_months: int = 3,
        initial_capital: float = 1_000_000,
    ) -> WalkForwardResult:
        """Run rolling walk-forward validation and return a :class:`WalkForwardResult`.

        Parameters
        ----------
        strategy:
            Any object implementing ``generate_signals(data, rebalance_date)``.
        data:
            Full historical data dict.  Each test window receives this entire
            dict so the strategy has its full look-back history available.
        start_date:
            First day of the *train* window of window 1 (inclusive).
        end_date:
            Last day of the final *test* window (inclusive).
        train_months:
            Length of each training window in calendar months.
        test_months:
            Length of each out-of-sample test window in calendar months.
            The rolling step is also ``test_months``.
        initial_capital:
            Starting portfolio value in USD for the chained OOS curve.

        Returns
        -------
        WalkForwardResult
        """
        windows_spec = _generate_windows(start_date, end_date, train_months, test_months)

        if not windows_spec:
            logger.warning(
                "No walk-forward windows generated for [%s → %s] with train=%d, test=%d months.",
                start_date,
                end_date,
                train_months,
                test_months,
            )
            return WalkForwardResult()

        logger.info(
            "Walk-Forward: %d windows | train=%d mo | test=%d mo | OOS period: %s → %s",
            len(windows_spec),
            train_months,
            test_months,
            windows_spec[0][2].date(),
            windows_spec[-1][3].date(),
        )

        backtester = Backtester(
            commission_rate=self.commission_rate,
            slippage_model=self.slippage_model,
        )
        window_results: list[dict] = []
        oos_pieces: list[pd.Series] = []
        running_nav = float(initial_capital)

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows_spec, start=1):
            logger.info(
                "Window %2d/%d | train: %s→%s | test: %s→%s",
                i,
                len(windows_spec),
                train_start.date(),
                train_end.date(),
                test_start.date(),
                test_end.date(),
            )

            # Reset stateful strategies between windows so held-state from a
            # previous test period does not leak into the next one.
            if hasattr(strategy, "reset") and callable(strategy.reset):
                strategy.reset()

            # Warmup: step stateful strategies through the training period so
            # their held-portfolio state reflects end-of-training holdings.
            # This prevents quarterly strategies from sitting in SHY for the
            # first N non-quarter months of each test window.
            if hasattr(strategy, "warmup") and callable(strategy.warmup):
                logger.debug(
                    "Window %d: warming up strategy over training period [%s → %s].",
                    i,
                    train_start.date(),
                    train_end.date(),
                )
                strategy.warmup(data, train_start, train_end)

            try:
                result = backtester.run(
                    strategy=strategy,
                    data=data,
                    start_date=test_start,
                    end_date=test_end,
                    initial_capital=running_nav,
                )
            except Exception as exc:
                logger.error("Window %d backtester.run() failed: %s — skipping window.", i, exc)
                continue

            ec = result.equity_curve
            if ec.empty:
                logger.warning("Window %d produced an empty equity curve — skipping.", i)
                continue

            running_nav = float(ec.iloc[-1])
            oos_pieces.append(ec)

            window_results.append(
                {
                    "window": i,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "metrics": result.metrics,
                    "equity_curve": ec,
                }
            )

        if not oos_pieces:
            logger.error("All walk-forward windows failed or were empty.")
            return WalkForwardResult(windows=window_results)

        # ----------------------------------------------------------------
        # Chain all OOS test-period equity curves
        # ----------------------------------------------------------------
        oos_equity_curve = pd.concat(oos_pieces)
        oos_equity_curve.index.name = "date"

        # ----------------------------------------------------------------
        # Aggregate OOS metrics over the full chained curve
        # ----------------------------------------------------------------
        spy_df = data.get("SPY")
        oos_metrics = _compute_oos_metrics(oos_equity_curve, spy_df, initial_capital)

        # ----------------------------------------------------------------
        # Consistency ratio: fraction of windows with positive test return
        # ----------------------------------------------------------------
        profitable = sum(1 for w in window_results if w["metrics"].get("Total Return (%)", 0.0) > 0)
        consistency_ratio = profitable / len(window_results)

        logger.info(
            "Walk-Forward complete | OOS Total Return=%.2f%% | CAGR=%.2f%% | "
            "Sharpe=%.3f | MaxDD=%.2f%% | Consistency=%d/%d (%.0f%%)",
            oos_metrics.get("Total Return (%)", float("nan")),
            oos_metrics.get("CAGR (%)", float("nan")),
            oos_metrics.get("Sharpe Ratio", float("nan")),
            oos_metrics.get("Max Drawdown (%)", float("nan")),
            profitable,
            len(window_results),
            consistency_ratio * 100,
        )

        return WalkForwardResult(
            windows=window_results,
            oos_equity_curve=oos_equity_curve,
            oos_metrics=oos_metrics,
            consistency_ratio=consistency_ratio,
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _generate_windows(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    train_months: int,
    test_months: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Generate (train_start, train_end, test_start, test_end) tuples.

    The step size equals *test_months* so test periods tile without overlap.
    The final window's test_end is capped at *end_date*.
    """
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    windows: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    train_start = start

    while True:
        train_end = train_start + DateOffset(months=train_months) - pd.Timedelta(days=1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + DateOffset(months=test_months) - pd.Timedelta(days=1)

        if test_start > end:
            break

        # Cap the last window at end_date
        if test_end > end:
            test_end = end

        windows.append(
            (
                train_start,
                pd.Timestamp(train_end).normalize(),
                pd.Timestamp(test_start).normalize(),
                pd.Timestamp(test_end).normalize(),
            )
        )

        train_start = train_start + DateOffset(months=test_months)

    return windows


def _compute_oos_metrics(
    equity_curve: pd.Series,
    spy_df: pd.DataFrame | None,
    initial_capital: float,
) -> dict:
    """Compute aggregate metrics over the full chained OOS equity curve."""
    if len(equity_curve) < 2:
        return {}

    start = equity_curve.index[0]
    end = equity_curve.index[-1]
    years = max((end - start).days / 365.25, 1e-9)

    final_value = float(equity_curve.iloc[-1])
    total_return = (final_value / initial_capital - 1.0) * 100.0
    cagr = ((final_value / initial_capital) ** (1.0 / years) - 1.0) * 100.0

    daily_rets = equity_curve.pct_change().dropna()
    sharpe = (daily_rets.mean() / daily_rets.std()) * (252.0**0.5) if daily_rets.std() > 0 else 0.0

    rolling_max = equity_curve.cummax()
    max_dd = float(abs((equity_curve / rolling_max - 1.0).min()) * 100.0)

    # SPY buy-and-hold over the same OOS window
    benchmark_return: float = float("nan")
    if spy_df is not None and not spy_df.empty and "close" in spy_df.columns:
        spy_close = spy_df["close"].copy()
        idx = spy_close.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        spy_close.index = idx.normalize()
        spy_in_range = spy_close.loc[start:end].dropna()
        if len(spy_in_range) >= 2:
            benchmark_return = (spy_in_range.iloc[-1] / spy_in_range.iloc[0] - 1.0) * 100.0

    return {
        "Total Return (%)": round(total_return, 2),
        "CAGR (%)": round(cagr, 2),
        "Sharpe Ratio": round(sharpe, 3),
        "Max Drawdown (%)": round(max_dd, 2),
        "Benchmark Return (%)": round(benchmark_return, 2),
    }


# ---------------------------------------------------------------------------
# Public helpers (used by tests and external callers)
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardWindow:
    """A single walk-forward window with in-sample and OOS date ranges.

    Attributes
    ----------
    in_sample_start, in_sample_end :
        Inclusive date range for the training period.
    out_of_sample_start, out_of_sample_end :
        Inclusive date range for the test period.
    """

    in_sample_start: pd.Timestamp
    in_sample_end: pd.Timestamp
    out_of_sample_start: pd.Timestamp
    out_of_sample_end: pd.Timestamp


def generate_windows(
    index: pd.DatetimeIndex,
    in_sample_bars: int,
    out_of_sample_bars: int,
) -> list[WalkForwardWindow]:
    """Generate non-overlapping walk-forward windows from a DatetimeIndex.

    Parameters
    ----------
    index :
        Full price/return index (must have at least in_sample_bars +
        out_of_sample_bars entries).
    in_sample_bars :
        Number of bars in each training window.
    out_of_sample_bars :
        Number of bars in each OOS (test) window.

    Returns
    -------
    list[WalkForwardWindow]
        At least one window.  Raises ``ValueError`` if the index is too
        short to form even one window.
    """
    min_bars = in_sample_bars + out_of_sample_bars
    if len(index) < min_bars:
        raise ValueError(
            f"Index has {len(index)} bars but at least {min_bars} are required "
            f"(in_sample_bars={in_sample_bars} + out_of_sample_bars={out_of_sample_bars})."
        )

    windows: list[WalkForwardWindow] = []
    start = 0
    while start + min_bars <= len(index):
        is_start = index[start]
        is_end = index[start + in_sample_bars - 1]
        oos_start = index[start + in_sample_bars]
        oos_end_idx = min(start + in_sample_bars + out_of_sample_bars - 1, len(index) - 1)
        oos_end = index[oos_end_idx]
        windows.append(WalkForwardWindow(is_start, is_end, oos_start, oos_end))
        start += out_of_sample_bars

    return windows

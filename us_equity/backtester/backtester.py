"""Backtester — event-driven monthly-rebalance portfolio simulation.

Simulates a long-only, monthly-rebalancing portfolio driven by any strategy
that implements ``generate_signals(data, rebalance_date) -> dict[str, float]``.

Simulation mechanics
--------------------
* **Rebalance timing** : signals are computed using close prices on the *last
  trading day of each calendar month* (``rebalance_date``).  The new target
  weights take effect from the *next* trading day; the transaction cost is
  deducted from the portfolio value at the close of the rebalance day.
* **Daily P&L** : ``portfolio_value *= 1 + Σ(weight_i × r_i)`` where
  ``r_i = close_i_t / close_i_{t-1} − 1`` for each held symbol.
* **Transaction cost** : one-way ``commission_rate`` (default 0.1 %) applied
  to gross turnover:
  ``cost = portfolio_value × Σ|w_new_i − w_old_i| × commission_rate``
* **Forward-fill** : price gaps within the backtest window are filled forward
  so no position ever has a NaN return.
* **Look-ahead guard** : ``bias_check.check_no_lookahead`` is called before
  every ``generate_signals`` invocation; a detected violation raises
  ``ValueError`` immediately.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import pandas as pd

from core.risk.slippage import DynamicSlippage
from us_equity.backtester.bias_check import check_no_lookahead

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class SignalGenerator(Protocol):
    """Structural type for any monthly-rebalance strategy."""

    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
        rebalance_date: pd.Timestamp,
    ) -> dict[str, float]:
        """Return ``{symbol: weight}`` with weights summing to 1.0."""
        ...


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """All outputs from a completed backtest run.

    Attributes
    ----------
    equity_curve:
        Portfolio value at the close of every trading day in the backtest
        window.  Index is a UTC DatetimeIndex; name is ``"portfolio_value"``.
    trades:
        One entry per rebalance event.  Keys per entry:
        ``date``, ``weights``, ``turnover``, ``cost_usd``, ``portfolio_value``.
    metrics:
        Performance summary.  Keys:
        ``Total Return (%)``, ``CAGR (%)``, ``Sharpe Ratio``,
        ``Max Drawdown (%)``, ``Calmar Ratio``, ``Win Rate (%)``,
        ``Benchmark Return (%)``.
    """

    equity_curve: pd.Series
    trades: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    # Populated when DynamicSlippage is used; None for fixed-commission runs.
    slippage_detail: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class Backtester:
    """Event-driven monthly-rebalance backtester.

    Parameters
    ----------
    db_path:
        Path to the SQLite market-data cache (used by DB-polling methods).
    commission_rate:
        One-way transaction cost as a decimal fraction (default ``0.001``
        = 0.1 %, representing slippage + commission for US large-caps).

    Example
    -------
    >>> bt = Backtester()
    >>> result = bt.run(strategy, data, "2023-01-01", "2025-12-31")
    >>> print(result.metrics)
    """

    def __init__(
        self,
        db_path: str = "data/db/market_data.db",
        commission_rate: float = 0.001,
        slippage_model: DynamicSlippage | None = None,
    ) -> None:
        self.db_path = db_path
        self.commission_rate = commission_rate
        # When set, overrides commission_rate with per-symbol dynamic cost.
        self.slippage_model = slippage_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        strategy: SignalGenerator,
        data: dict[str, pd.DataFrame],
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        initial_capital: float = 1_000_000,
    ) -> BacktestResult:
        """Run a full backtest and return a :class:`BacktestResult`.

        The *data* dict may contain historical bars that precede *start_date*;
        those extra bars serve as look-back warmup for the strategy and are
        not included in the reported equity curve.

        Parameters
        ----------
        strategy:
            Any object implementing ``generate_signals(data, rebalance_date)
            -> dict[str, float]``.
        data:
            ``{symbol: OHLCV DataFrame}`` covering at least *start_date* to
            *end_date*.  **SPY** must be present for benchmark calculation.
        start_date:
            First day of the reported equity curve (inclusive).
        end_date:
            Last day of the reported equity curve (inclusive).
        initial_capital:
            Starting portfolio value in USD.

        Returns
        -------
        BacktestResult
            Contains ``equity_curve``, ``trades``, and ``metrics``.

        Raises
        ------
        ValueError
            If *data* contains no usable price bars in the specified window,
            or if a look-ahead bias violation is detected.
        """
        start = _to_utc(start_date)
        end = _to_utc(end_date)

        # Build aligned price matrix (backtest window only; strategy receives
        # full historical data for its own look-back calculations).
        prices = _build_price_matrix(data, start, end)

        if prices.empty:
            raise ValueError(
                f"No price data in [{start.date()} → {end.date()}]. "
                "Ensure data covers the backtest window."
            )

        trading_days = prices.index.tolist()
        rebalance_set = set(_get_rebalance_dates(prices.index))

        logger.info(
            "Backtester.run: %d trading days | %d rebalance dates [%s → %s] | initial_capital=$%s",
            len(trading_days),
            len(rebalance_set),
            trading_days[0].date(),
            trading_days[-1].date(),
            f"{initial_capital:,.0f}",
        )

        # ----------------------------------------------------------------
        # Build volume matrix (only needed for DynamicSlippage)
        # ----------------------------------------------------------------
        volumes: pd.DataFrame | None = None
        if self.slippage_model is not None:
            volumes = _build_volume_matrix(data, start, end)

        # ----------------------------------------------------------------
        # Event-driven simulation
        # ----------------------------------------------------------------
        portfolio_value: float = float(initial_capital)
        current_weights: dict[str, float] = {}  # weights entering each day
        equity_data: list[tuple[pd.Timestamp, float]] = []
        trades: list[dict] = []
        slippage_detail: list[dict] = []

        # Accumulators for avg_slippage_bps metric
        _total_slip_cost: float = 0.0
        _total_trade_val: float = 0.0

        for i, today in enumerate(trading_days):
            # ---- Apply today's return with weights set at previous close ----
            if i > 0 and current_weights:
                prev = trading_days[i - 1]
                port_ret = _portfolio_return(prices, today, prev, current_weights)
                portfolio_value *= 1.0 + port_ret

            # ---- Rebalance at today's close if applicable ----
            if today in rebalance_set:
                # Pass only data up to and including today — no future rows.
                snapshot = {sym: df.loc[df.index <= today] for sym, df in data.items()}
                check_no_lookahead(snapshot, today)  # invariant — should never raise

                try:
                    new_weights = strategy.generate_signals(snapshot, today)
                except Exception as exc:
                    logger.error(
                        "generate_signals raised on %s: %s — "
                        "holding current weights for this period.",
                        today.date(),
                        exc,
                    )
                    equity_data.append((today, portfolio_value))
                    continue

                turnover = _turnover(current_weights, new_weights)

                if self.slippage_model is not None and volumes is not None:
                    cost, sym_bps = _dynamic_cost(
                        self.slippage_model,
                        current_weights,
                        new_weights,
                        portfolio_value,
                        prices,
                        volumes,
                        today,
                    )
                    trade_val = portfolio_value * turnover
                    _total_slip_cost += cost
                    _total_trade_val += trade_val
                    slippage_detail.append(
                        {
                            "date": today,
                            "per_symbol_bps": sym_bps,
                            "total_cost_usd": round(cost, 2),
                            "trade_value_usd": round(trade_val, 2),
                        }
                    )
                else:
                    cost = portfolio_value * turnover * self.commission_rate
                    _total_trade_val += portfolio_value * turnover
                    _total_slip_cost += cost

                portfolio_value -= cost

                logger.debug(
                    "Rebalance %s | turnover=%.2f%% | cost=$%.2f | NAV=$%.2f | holdings=%s",
                    today.date(),
                    turnover * 100,
                    cost,
                    portfolio_value,
                    {s: f"{w:.1%}" for s, w in new_weights.items()},
                )

                trades.append(
                    {
                        "date": today,
                        "weights": new_weights.copy(),
                        "turnover": round(turnover, 6),
                        "cost_usd": round(cost, 2),
                        "portfolio_value": round(portfolio_value, 2),
                    }
                )
                current_weights = new_weights

            equity_data.append((today, portfolio_value))

        # ----------------------------------------------------------------
        # Assemble result
        # ----------------------------------------------------------------
        equity_curve = pd.Series(
            {ts: val for ts, val in equity_data},
            name="portfolio_value",
        )
        equity_curve.index.name = "date"

        spy_df = data.get("SPY")
        metrics = _compute_metrics(equity_curve, spy_df, initial_capital)

        # Avg slippage bps across all rebalances
        if _total_trade_val > 0:
            metrics["Avg Slippage (bps)"] = round(_total_slip_cost / _total_trade_val * 10_000, 2)

        logger.info(
            "Backtest complete | Total Return=%.2f%% | CAGR=%.2f%% | "
            "Sharpe=%.3f | MaxDD=%.2f%% | Calmar=%.3f",
            metrics.get("Total Return (%)", float("nan")),
            metrics.get("CAGR (%)", float("nan")),
            metrics.get("Sharpe Ratio", float("nan")),
            metrics.get("Max Drawdown (%)", float("nan")),
            metrics.get("Calmar Ratio", float("nan")),
        )

        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            metrics=metrics,
            slippage_detail=slippage_detail,
        )

    # ------------------------------------------------------------------
    # DB-polling methods (agent pipeline — not used by run())
    # ------------------------------------------------------------------

    def validate(self, strategy_id: str) -> dict:
        """Run walk-forward validation for a single strategy. Returns metrics dict."""
        raise NotImplementedError

    def _load_candidate(self, strategy_id: str) -> dict:
        """Fetch strategy params from strategy_candidates table."""
        raise NotImplementedError

    def _persist_results(self, strategy_id: str, metrics: dict, verdict: str) -> None:
        """Write to backtest_results and update strategy_candidates.status."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _to_utc(dt: str | pd.Timestamp) -> pd.Timestamp:
    """Parse *dt* as a UTC Timestamp regardless of input format."""
    ts = pd.Timestamp(dt)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _build_price_matrix(
    data: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Build a wide close-price DataFrame aligned to UTC midnight.

    * Rows    : trading days in [start, end] (union across all symbols)
    * Columns : one per symbol
    * Missing values are forward-filled within each column after alignment.
    """
    frames: dict[str, pd.Series] = {}
    for sym, df in data.items():
        if df.empty or "close" not in df.columns:
            continue
        close = df["close"].copy()
        idx = close.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        close.index = idx.normalize()  # align to midnight
        close = close.sort_index()
        # Only keep rows within the backtest window
        close = close.loc[start:end]
        if not close.empty:
            frames[sym] = close

    if not frames:
        return pd.DataFrame()

    prices = pd.DataFrame(frames)
    prices = prices.sort_index()
    prices = prices.ffill()  # fill any intra-range gaps (e.g. staggered holidays)
    return prices


def _get_rebalance_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Return the last trading day of each calendar month present in *index*."""
    last_of_month: dict[tuple[int, int], pd.Timestamp] = {}
    for ts in index:
        key = (ts.year, ts.month)
        if key not in last_of_month or ts > last_of_month[key]:
            last_of_month[key] = ts
    return sorted(last_of_month.values())


def _portfolio_return(
    prices: pd.DataFrame,
    today: pd.Timestamp,
    prev: pd.Timestamp,
    weights: dict[str, float],
) -> float:
    """Compute the weighted portfolio return from *prev* close to *today* close.

    Symbols absent from *prices* or with NaN/zero prices are skipped; their
    weight effectively becomes uninvested cash (return = 0) for that day.
    """
    port_ret = 0.0
    for sym, weight in weights.items():
        if sym not in prices.columns:
            continue
        p_prev = prices.at[prev, sym]
        p_today = prices.at[today, sym]
        if pd.isna(p_prev) or pd.isna(p_today) or p_prev <= 0:
            continue
        port_ret += weight * (p_today / p_prev - 1.0)
    return port_ret


def _turnover(
    old_weights: dict[str, float],
    new_weights: dict[str, float],
) -> float:
    """Compute gross portfolio turnover (sum of absolute weight changes)."""
    all_symbols = set(old_weights) | set(new_weights)
    return sum(abs(new_weights.get(s, 0.0) - old_weights.get(s, 0.0)) for s in all_symbols)


def _compute_metrics(
    equity_curve: pd.Series,
    spy_df: pd.DataFrame | None,
    initial_capital: float,
) -> dict:
    """Compute the standard set of backtest performance metrics."""
    if len(equity_curve) < 2:
        logger.warning("Equity curve has fewer than 2 points — metrics unavailable.")
        return {}

    start = equity_curve.index[0]
    end = equity_curve.index[-1]
    years = max((end - start).days / 365.25, 1e-9)

    final_value = equity_curve.iloc[-1]

    # ---- Returns ----
    total_return = (final_value / initial_capital - 1.0) * 100.0
    cagr = ((final_value / initial_capital) ** (1.0 / years) - 1.0) * 100.0

    # ---- Sharpe (annualised, rf = 0) ----
    daily_rets = equity_curve.pct_change().dropna()
    if daily_rets.std() > 0:
        sharpe = (daily_rets.mean() / daily_rets.std()) * (252.0**0.5)
    else:
        sharpe = 0.0

    # ---- Max Drawdown ----
    rolling_max = equity_curve.cummax()
    drawdowns = equity_curve / rolling_max - 1.0
    max_dd = abs(drawdowns.min()) * 100.0

    # ---- Calmar ----
    calmar = cagr / max_dd if max_dd > 0 else float("inf")

    # ---- Monthly Win Rate ----
    monthly = equity_curve.resample("ME").last()
    monthly_rets = monthly.pct_change().dropna()
    win_rate = (monthly_rets > 0).mean() * 100.0 if len(monthly_rets) > 0 else float("nan")

    # ---- Benchmark (SPY buy-and-hold) ----
    benchmark_return: float = float("nan")
    if spy_df is not None and not spy_df.empty and "close" in spy_df.columns:
        spy_close = spy_df["close"].copy()
        idx = spy_close.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        spy_close.index = idx.normalize()
        spy_close = spy_close.sort_index()
        spy_in_range = spy_close.loc[start:end].dropna()
        if len(spy_in_range) >= 2:
            benchmark_return = (spy_in_range.iloc[-1] / spy_in_range.iloc[0] - 1.0) * 100.0

    return {
        "Total Return (%)": round(total_return, 2),
        "CAGR (%)": round(cagr, 2),
        "Sharpe Ratio": round(sharpe, 3),
        "Max Drawdown (%)": round(max_dd, 2),
        "Calmar Ratio": round(calmar, 3),
        "Win Rate (%)": round(win_rate, 1),
        "Benchmark Return (%)": round(benchmark_return, 2),
    }


def _build_volume_matrix(
    data: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Build a wide share-volume DataFrame aligned to UTC midnight.

    Mirrors :func:`_build_price_matrix` but uses the ``volume`` column.
    Symbols without a ``volume`` column are silently excluded.
    """
    frames: dict[str, pd.Series] = {}
    for sym, df in data.items():
        if df.empty or "volume" not in df.columns:
            continue
        vol = df["volume"].copy().astype(float)
        idx = vol.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        vol.index = idx.normalize()
        vol = vol.sort_index().loc[start:end]
        if not vol.empty:
            frames[sym] = vol

    if not frames:
        return pd.DataFrame()

    volumes = pd.DataFrame(frames).sort_index()
    # Forward-fill: use previous day's volume when a day is missing
    volumes = volumes.ffill()
    return volumes


def _dynamic_cost(
    model: DynamicSlippage,
    old_weights: dict[str, float],
    new_weights: dict[str, float],
    portfolio_value: float,
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    today: pd.Timestamp,
) -> tuple[float, dict[str, float]]:
    """Compute rebalance cost using DynamicSlippage and return (total_cost, per_sym_bps).

    Only symbols with a non-zero weight change are costed.
    Falls back to BASE_BPS for symbols missing from the volume or price matrices.
    """
    all_syms = set(old_weights) | set(new_weights)
    total_cost = 0.0
    sym_bps: dict[str, float] = {}

    for sym in all_syms:
        delta_w = abs(new_weights.get(sym, 0.0) - old_weights.get(sym, 0.0))
        if delta_w < 1e-9:
            continue

        trade_val = delta_w * portfolio_value

        # Price: use today's close from the aligned price matrix
        price = float(prices.at[today, sym]) if sym in prices.columns else 0.0
        # Volume: today's share volume from the aligned volume matrix
        volume = float(volumes.at[today, sym]) if sym in volumes.columns else 0.0

        bps = model.estimate(sym, trade_val, volume, price)
        cost = trade_val * bps / 10_000.0

        total_cost += cost
        sym_bps[sym] = round(bps, 3)

    return total_cost, sym_bps


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backtester Agent")
    parser.add_argument("--strategy-id", type=str, help="Validate a specific strategy ID")
    args = parser.parse_args()
    logger.info("Backtester starting (strategy_id=%s)", args.strategy_id)

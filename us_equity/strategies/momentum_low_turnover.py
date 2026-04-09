"""Momentum Low-Turnover Strategy — quarterly rebalance with holding buffer.

Design goals
------------
- **Lower transaction costs** by rebalancing quarterly instead of monthly and
  using a holding buffer to avoid churning positions unnecessarily.
- **Simpler signal** — pure 12-1 momentum only (mean-reversion excluded
  because its short look-back drives excessive monthly turnover).

Strategy logic
--------------
1. **Rebalance trigger**: only on the last trading day of March, June,
   September, and December (``rebalance_date.month in {3, 6, 9, 12}``).
   On all other months, the existing weights are returned unchanged.

2. **Universe**: SP500_TOP50 (same as Dual Momentum).

3. **Absolute momentum gate**: same as Dual Momentum — symbol 12-1 momentum
   must exceed SPY's.

4. **Holding buffer (anti-churn rule)**:
   After filtering through the SPY gate, rank the passing symbols by 12-1
   momentum (descending).

   - **Keep** any currently-held stock that is still within the top 15
     of the filtered & ranked list.
   - **Fill** empty slots (new positions or slots where the prior holder
     dropped out of the top 15) from the top 10 of the current ranking.
   - If fewer than 15 candidates pass the SPY gate, take all of them.

5. **Portfolio size**: up to 15 stocks at equal weight (1/15 ≈ 6.67% each).
   Remaining slots (if < 15 candidates pass) go to SHY.

6. **State**: the strategy keeps track of ``_current_holdings`` between
   rebalances.  Call :meth:`reset` between walk-forward windows to clear
   this state.

Minimum data requirement: 252 trading days (momentum window).
"""

from __future__ import annotations

import logging
from typing import Final

import pandas as pd

from .dual_momentum import SP500_TOP50

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BENCHMARK: Final[str] = "SPY"
_SAFE_HAVEN: Final[str] = "SHY"

_LOOKBACK_MOM_LONG: Final[int] = 252  # 12 months
_LOOKBACK_MOM_SHORT: Final[int] = 21  #  1 month  (skipped)
_MIN_BARS: Final[int] = 252

_QUARTER_MONTHS: Final[frozenset[int]] = frozenset({3, 6, 9, 12})

_TOP_N: Final[int] = 15  # max portfolio size
_BUFFER_KEEP: Final[int] = 15  # keep current stock if still in top-N
_BUFFER_ADD: Final[int] = 10  # fill new slots from top-N
_WEIGHT_PER_SLOT: Final[float] = 1.0 / _TOP_N


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class MomentumLowTurnoverStrategy:
    """Quarterly momentum strategy with holding buffer to minimise turnover.

    Parameters
    ----------
    universe:
        Equity symbols to consider.  Defaults to :data:`SP500_TOP50`.
    top_n:
        Maximum number of stocks to hold.  Defaults to 15.

    Example
    -------
    >>> strategy = MomentumLowTurnoverStrategy()
    >>> weights = strategy.generate_signals(data, rebalance_date)
    >>> assert abs(sum(weights.values()) - 1.0) < 1e-9
    """

    def __init__(
        self,
        universe: list[str] | None = None,
        top_n: int = _TOP_N,
    ) -> None:
        self.universe: list[str] = universe if universe is not None else SP500_TOP50
        self.top_n: int = top_n
        self._weight_per_slot: float = 1.0 / top_n
        self._current_holdings: set[str] = set()
        self._current_weights: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear held-state between walk-forward windows."""
        self._current_holdings = set()
        self._current_weights = {}
        logger.debug("MomentumLowTurnoverStrategy state reset.")

    def warmup(
        self,
        data: dict[str, pd.DataFrame],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> None:
        """Step through historical rebalance dates to prime the held-portfolio state.

        Called by :class:`~agents.backtester.walk_forward.WalkForwardValidator`
        before each test window so that ``_current_holdings`` reflects what the
        strategy would naturally hold at the end of the training period — rather
        than starting cold (which forces non-quarter months to sit in SHY).

        Parameters
        ----------
        data:
            Full historical data dict (same dict passed to generate_signals).
        start_date, end_date:
            Inclusive date range of the training window.
        """
        # Build a list of the last calendar day of each month in the range,
        # then snap to the nearest available trading day using the SPY index.
        spy_index = _get_trading_days(data, start_date, end_date)
        if spy_index is None or spy_index.empty:
            logger.debug("Warmup: no trading days found in [%s, %s].", start_date, end_date)
            return

        # Group by year-month, pick the last trading day of each month
        by_month: dict[tuple[int, int], pd.Timestamp] = {}
        for dt in spy_index:
            key = (dt.year, dt.month)
            if key not in by_month or dt > by_month[key]:
                by_month[key] = dt
        rebalance_dates = sorted(by_month.values())

        logger.debug(
            "Warmup: stepping through %d rebalance dates [%s → %s].",
            len(rebalance_dates),
            rebalance_dates[0].date() if rebalance_dates else "N/A",
            rebalance_dates[-1].date() if rebalance_dates else "N/A",
        )
        for rd in rebalance_dates:
            self.generate_signals(data, rd)  # side-effect: updates _current_weights

    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
        rebalance_date: pd.Timestamp,
    ) -> dict[str, float]:
        """Compute target portfolio weights for *rebalance_date*.

        On non-quarter-end months the existing weights are returned as-is
        (no transaction cost incurred).  On quarter-end months a full
        momentum ranking is run with the holding buffer.

        Parameters
        ----------
        data:
            ``{symbol: OHLCV DataFrame}`` with DatetimeIndex.  Must include
            **SPY** and **SHY** in addition to the equity universe.
        rebalance_date:
            Last trading day of the month.

        Returns
        -------
        dict[str, float]
            ``{symbol: weight}`` with weights summing to 1.0.
        """
        if _BENCHMARK not in data:
            raise KeyError(f"Benchmark '{_BENCHMARK}' missing from data.")
        if _SAFE_HAVEN not in data:
            raise KeyError(f"Safe-haven '{_SAFE_HAVEN}' missing from data.")

        # Non-quarter months: hold current portfolio unchanged
        if rebalance_date.month not in _QUARTER_MONTHS:
            if self._current_weights:
                logger.debug(
                    "%s: non-quarter month — holding current portfolio.",
                    rebalance_date.date(),
                )
                return dict(self._current_weights)
            # No prior portfolio (start of backtest) — stay in SHY
            logger.info(
                "%s: non-quarter month, no prior portfolio — 100%% SHY.",
                rebalance_date.date(),
            )
            return {_SAFE_HAVEN: 1.0}

        # ---- Quarter-end rebalance ----
        logger.info("Quarter-end rebalance on %s", rebalance_date.date())

        # Benchmark absolute-momentum threshold
        spy_mom = self._compute_momentum(data[_BENCHMARK], rebalance_date, label=_BENCHMARK)
        if spy_mom is None:
            logger.warning(
                "Insufficient SPY data on %s — returning 100%% SHY.", rebalance_date.date()
            )
            weights = {_SAFE_HAVEN: 1.0}
            self._current_weights = weights
            self._current_holdings = set()
            return weights

        # Compute 12-1 momentum for each symbol
        mom_scores: dict[str, float] = {}
        skipped: list[str] = []

        for sym in self.universe:
            if sym not in data:
                skipped.append(sym)
                continue
            mom = self._compute_momentum(data[sym], rebalance_date, label=sym)
            if mom is None:
                skipped.append(sym)
                continue
            mom_scores[sym] = mom

        if skipped:
            logger.debug(
                "%d symbol(s) skipped on %s: %s",
                len(skipped),
                rebalance_date.date(),
                ", ".join(skipped),
            )

        # Absolute momentum filter: must beat SPY
        candidates = {s: m for s, m in mom_scores.items() if m > spy_mom}

        logger.debug(
            "%s: %d symbols pass absolute-momentum filter | %d fail | SPY=%.4f",
            rebalance_date.date(),
            len(candidates),
            len(mom_scores) - len(candidates),
            spy_mom,
        )

        if not candidates:
            logger.info("%s: 0 candidates pass SPY gate — 100%% SHY.", rebalance_date.date())
            weights = {_SAFE_HAVEN: 1.0}
            self._current_weights = weights
            self._current_holdings = set()
            return weights

        # Rank by momentum descending
        ranked: list[str] = sorted(candidates, key=lambda s: candidates[s], reverse=True)

        # Apply holding buffer
        selected = self._select_with_buffer(ranked)

        n_selected = len(selected)
        n_shy_slots = self.top_n - n_selected
        shy_weight = round(n_shy_slots * self._weight_per_slot, 10)

        logger.info(
            "Rebalance %s — selected %d equity slot(s): %s | SHY slots: %d (%.0f%%)",
            rebalance_date.date(),
            n_selected,
            selected,
            n_shy_slots,
            shy_weight * 100,
        )

        weights: dict[str, float] = {s: self._weight_per_slot for s in selected}
        if shy_weight > 0:
            weights[_SAFE_HAVEN] = shy_weight

        total = sum(weights.values())
        if abs(total - 1.0) > 1e-9:
            logger.error(
                "Weight sum %.10f ≠ 1.0 on %s — this is a bug.",
                total,
                rebalance_date.date(),
            )

        self._current_holdings = set(selected)
        self._current_weights = dict(weights)
        return weights

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _select_with_buffer(self, ranked: list[str]) -> list[str]:
        """Apply the holding-buffer rule to produce the final selection.

        Rules
        -----
        - Any current holding that is still within the top ``_BUFFER_KEEP``
          of the momentum-ranked, SPY-filtered list is **kept**.
        - Empty slots (including slots vacated by holdings that fell out of
          top ``_BUFFER_KEEP``) are filled from the top ``_BUFFER_ADD`` of
          the ranked list.
        - Total portfolio size is capped at ``self.top_n``.

        Parameters
        ----------
        ranked:
            Symbols ranked by momentum descending, already filtered by the
            SPY absolute-momentum gate.
        """
        keep_zone = set(ranked[:_BUFFER_KEEP])
        add_zone = ranked[:_BUFFER_ADD]

        # Keep current holdings that are still in keep_zone
        kept: list[str] = [s for s in self._current_holdings if s in keep_zone]

        # Add new positions from add_zone (avoid duplicates with kept)
        kept_set = set(kept)
        additions: list[str] = [s for s in add_zone if s not in kept_set]

        # Combine: kept + additions, cap at top_n
        selected = kept + additions
        selected = selected[: self.top_n]

        logger.debug(
            "Buffer: kept=%d %s | new=%d %s | final=%d",
            len(kept),
            kept,
            len(additions),
            additions[:5],
            len(selected),
        )
        return selected

    @staticmethod
    def _compute_momentum(
        df: pd.DataFrame,
        as_of: pd.Timestamp,
        label: str = "",
    ) -> float | None:
        """12-1 momentum for a single symbol, or None if insufficient data."""
        close = df["close"].sort_index()
        close = close.loc[close.index <= as_of]
        if close.empty or len(close) < _MIN_BARS:
            if not close.empty and label:
                logger.warning(
                    "%s: only %d bars on %s (need ≥ %d).",
                    label,
                    len(close),
                    as_of.date(),
                    _MIN_BARS,
                )
            return None
        p_12m = close.iloc[-_LOOKBACK_MOM_LONG]
        p_1m = close.iloc[-_LOOKBACK_MOM_SHORT]
        if p_12m <= 0:
            return None
        return float(p_1m / p_12m - 1.0)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _get_trading_days(
    data: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DatetimeIndex | None:
    """Return a sorted DatetimeIndex of trading days in [start, end].

    Uses SPY's index if available, otherwise the first available symbol.
    """
    ref_df = data.get("SPY")
    if ref_df is None:
        ref_df = next(iter(data.values()), None)
    if ref_df is None:
        return None
    idx = ref_df.index.normalize()
    # Ensure both the index and the boundary timestamps share the same tz
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    start_ts = (
        pd.Timestamp(start).tz_localize("UTC")
        if pd.Timestamp(start).tz is None
        else pd.Timestamp(start)
    )
    end_ts = (
        pd.Timestamp(end).tz_localize("UTC") if pd.Timestamp(end).tz is None else pd.Timestamp(end)
    )
    mask = (idx >= start_ts) & (idx <= end_ts)
    return idx[mask].sort_values()

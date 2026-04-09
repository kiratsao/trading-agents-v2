"""Multi-Factor Strategy — Momentum × Quality × Mean-Reversion.

Combines three cross-sectional factors into a composite score, then applies
an absolute-momentum gate (same SPY threshold as Dual Momentum) before
selecting the top-10 stocks by composite rank.

Factors (equal-weighted 1/3 each)
-----------------------------------
1. **Momentum** (12-1 month return)
   ``close[-21] / close[-252] − 1``
   Higher → better.  Directional persistence signal.

2. **Quality** (low-volatility proxy)
   Annualised daily-return std over the past 60 trading days.
   Lower → better.  High-quality / low-risk stocks.

3. **Mean Reversion** (5-day return)
   ``close[-1] / close[-5] − 1``
   Lower (more negative) → better.  Short-term oversold bounce.

Ranking
-------
Each factor is converted to a cross-sectional percentile rank in [0, 1]
across the valid universe on each rebalance date, then averaged:

    composite = (mom_rank + quality_rank + reversion_rank) / 3

Only symbols with a 12-1 momentum exceeding SPY's (absolute filter) enter
the final ranking.  The top-10 by composite score are held at equal weight
(10 % each); remaining slots go to SHY.

Minimum data requirement: 252 trading days (driven by the momentum window).
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
_LOOKBACK_VOL: Final[int] = 60  # quality window
_LOOKBACK_REV: Final[int] = 5  # mean-reversion window

_MIN_BARS: Final[int] = 252  # minimum bars required for any symbol
_TOP_N: Final[int] = 10
_WEIGHT_PER_SLOT: Final[float] = 1.0 / _TOP_N


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class MultiFactorStrategy:
    """Monthly multi-factor strategy combining momentum, quality and mean-reversion.

    Parameters
    ----------
    universe:
        List of equity symbols to consider.  Defaults to :data:`SP500_TOP50`.
    top_n:
        Maximum number of stocks to hold.  Defaults to 10.

    Example
    -------
    >>> strategy = MultiFactorStrategy()
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
        rebalance_date: pd.Timestamp,
    ) -> dict[str, float]:
        """Compute target portfolio weights for *rebalance_date*.

        Parameters
        ----------
        data:
            ``{symbol: OHLCV DataFrame}`` with DatetimeIndex.  Must include
            **SPY** and **SHY** in addition to the equity universe.
        rebalance_date:
            Last trading day of the month.  Only rows ``≤ rebalance_date``
            are used internally to prevent look-ahead bias.

        Returns
        -------
        dict[str, float]
            ``{symbol: weight}`` with weights summing to 1.0.

        Raises
        ------
        KeyError
            If SPY or SHY is absent from *data*.
        """
        if _BENCHMARK not in data:
            raise KeyError(f"Benchmark '{_BENCHMARK}' missing from data.")
        if _SAFE_HAVEN not in data:
            raise KeyError(f"Safe-haven '{_SAFE_HAVEN}' missing from data.")

        # ---- Benchmark absolute-momentum threshold ----
        spy_mom = self._compute_momentum(data[_BENCHMARK], rebalance_date, label=_BENCHMARK)
        if spy_mom is None:
            logger.warning(
                "Insufficient SPY data on %s — returning 100%% SHY.", rebalance_date.date()
            )
            return {_SAFE_HAVEN: 1.0}

        # ---- Compute all three factors for each symbol ----
        mom_raw: dict[str, float] = {}
        vol_raw: dict[str, float] = {}
        rev_raw: dict[str, float] = {}
        skipped: list[str] = []

        for sym in self.universe:
            if sym not in data:
                skipped.append(sym)
                continue

            close = self._get_close(data[sym], rebalance_date)
            if close is None or len(close) < _MIN_BARS:
                if close is not None:
                    logger.warning(
                        "%s: only %d bars on %s (need ≥ %d) — skipping.",
                        sym,
                        len(close),
                        rebalance_date.date(),
                        _MIN_BARS,
                    )
                skipped.append(sym)
                continue

            # Factor 1: Momentum (12-1)
            p_12m = close.iloc[-_LOOKBACK_MOM_LONG]
            p_1m = close.iloc[-_LOOKBACK_MOM_SHORT]
            if p_12m <= 0:
                skipped.append(sym)
                continue
            mom_raw[sym] = float(p_1m / p_12m - 1.0)

            # Factor 2: Quality — annualised daily-return volatility (60 days)
            # close guaranteed ≥ 252 bars, so 60-bar window is always available
            daily_rets = close.iloc[-(_LOOKBACK_VOL + 1) :].pct_change().dropna()
            vol_raw[sym] = float(daily_rets.std() * (252**0.5))

            # Factor 3: Mean Reversion — 5-day return
            rev_raw[sym] = float(close.iloc[-1] / close.iloc[-(_LOOKBACK_REV + 1)] - 1.0)

        if skipped:
            logger.debug(
                "%d symbol(s) skipped on %s: %s",
                len(skipped),
                rebalance_date.date(),
                ", ".join(skipped),
            )

        # Symbols with all three factors valid
        valid = set(mom_raw) & set(vol_raw) & set(rev_raw)
        if not valid:
            logger.warning("No valid symbols on %s — returning 100%% SHY.", rebalance_date.date())
            return {_SAFE_HAVEN: 1.0}

        # ---- Cross-sectional percentile ranking ----
        mom_rank = _percentile_rank({s: mom_raw[s] for s in valid}, ascending=True)
        # Low vol → high quality rank
        vol_rank = _percentile_rank({s: vol_raw[s] for s in valid}, ascending=False)
        # Low 5-day return (oversold) → high reversion rank
        rev_rank = _percentile_rank({s: rev_raw[s] for s in valid}, ascending=False)

        composite: dict[str, float] = {
            s: (mom_rank[s] + vol_rank[s] + rev_rank[s]) / 3.0 for s in valid
        }

        # ---- Absolute momentum filter ----
        candidates = {s: composite[s] for s in valid if mom_raw[s] > spy_mom}

        n_pass = len(candidates)
        n_fail = len(valid) - n_pass
        logger.debug(
            "%s: %d symbols pass absolute-momentum filter | %d fail | SPY mom=%.4f",
            rebalance_date.date(),
            n_pass,
            n_fail,
            spy_mom,
        )

        # ---- Select top-N by composite score ----
        ranked = sorted(candidates, key=lambda s: candidates[s], reverse=True)
        selected = ranked[: self.top_n]

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

        # ---- Build weight dict ----
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

        return weights

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_momentum(
        df: pd.DataFrame,
        as_of: pd.Timestamp,
        label: str = "",
    ) -> float | None:
        """12-1 momentum for a single symbol, or None if insufficient data."""
        close = MultiFactorStrategy._get_close(df, as_of)
        if close is None or len(close) < _MIN_BARS:
            if close is not None and label:
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

    @staticmethod
    def _get_close(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series | None:
        """Return the close series up to and including *as_of*, sorted."""
        close = df["close"].sort_index()
        close = close.loc[close.index <= as_of]
        return close if not close.empty else None


# ---------------------------------------------------------------------------
# Module-level ranking helper
# ---------------------------------------------------------------------------


def _percentile_rank(scores: dict[str, float], ascending: bool) -> dict[str, float]:
    """Convert raw factor scores to cross-sectional percentile ranks in [0, 1].

    Parameters
    ----------
    scores:
        ``{symbol: raw_factor_value}``
    ascending:
        If True, higher raw value → higher rank (rank 1.0).
        If False, lower raw value → higher rank (rank 1.0).

    Returns
    -------
    dict[str, float]
        ``{symbol: rank}`` with ranks in [0.0, 1.0].
        Single-symbol universe always gets rank 0.5.
    """
    symbols = list(scores.keys())
    n = len(symbols)
    if n == 0:
        return {}
    if n == 1:
        return {symbols[0]: 0.5}

    # Sort so that the "best" symbol ends up last (index n-1, rank 1.0)
    sorted_syms = sorted(symbols, key=lambda s: scores[s], reverse=not ascending)
    return {sym: i / (n - 1) for i, sym in enumerate(sorted_syms)}

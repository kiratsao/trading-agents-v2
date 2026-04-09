"""Dual Momentum strategy (Gary Antonacci).

Combines:
- **Relative Momentum**: rank stocks by 12-1 momentum (12-month return
  excluding the most recent month), keep the top 10.
- **Absolute Momentum**: only hold a stock if its 12-1 momentum exceeds
  that of SPY over the same window.  Slots that fail the absolute filter
  are filled with SHY (iShares 1-3 Year Treasury Bond ETF).

Universe: fixed list of the 50 largest S&P 500 constituents by market cap
(as of strategy initialisation).  SPY and SHY are always required in the
data dict.

Rebalance frequency: monthly (caller is responsible for invoking
``generate_signals`` on the last trading day of each month).

Reference
---------
Antonacci, G. (2014). *Dual Momentum Investing*. McGraw-Hill.
"""

from __future__ import annotations

import logging
from typing import Final

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

SP500_TOP50: Final[list[str]] = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "BRK.B",
    "JPM",
    "V",
    "UNH",
    "XOM",
    "JNJ",
    "WMT",
    "MA",
    "PG",
    "AVGO",
    "HD",
    "CVX",
    "MRK",
    "ABBV",
    "KO",
    "PEP",
    "COST",
    "LLY",
    "ADBE",
    "CRM",
    "MCD",
    "CSCO",
    "TMO",
    "ACN",
    "NFLX",
    "AMD",
    "LIN",
    "ABT",
    "DHR",
    "TXN",
    "CMCSA",
    "NEE",
    "PM",
    "ORCL",
    "HON",
    "UNP",
    "INTC",
    "LOW",
    "AMGN",
    "IBM",
    "GE",
    "CAT",
    "BA",
]

# Benchmark (absolute-momentum threshold) and safe-haven asset
_BENCHMARK: Final[str] = "SPY"
_SAFE_HAVEN: Final[str] = "SHY"

# Lookback windows (trading days)
_LOOKBACK_LONG: Final[int] = 252  # ≈ 12 months
_LOOKBACK_SHORT: Final[int] = 21  # ≈  1 month  (skipped / excluded)
_MIN_BARS: Final[int] = 252  # minimum history required

# Portfolio construction
_TOP_N: Final[int] = 10
_WEIGHT_PER_SLOT: Final[float] = 1.0 / _TOP_N  # 0.10


class DualMomentumStrategy:
    """Monthly Dual Momentum strategy for US equities.

    Parameters
    ----------
    universe:
        Equity symbols to consider.  Defaults to :data:`SP500_TOP50`.
    top_n:
        Number of stocks to hold at most.  Defaults to 10.

    Examples
    --------
    >>> strategy = DualMomentumStrategy()
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
            Mapping of ``symbol → OHLCV DataFrame`` with a DatetimeIndex.
            Each DataFrame must contain at least a ``close`` column.
            **SPY** and **SHY** must be present in addition to the equity
            universe.
        rebalance_date:
            The last trading day of the month for which weights are computed.
            Only rows whose index is ≤ *rebalance_date* are used, preventing
            any look-ahead bias.

        Returns
        -------
        dict[str, float]
            ``{symbol: weight}`` where weights sum to 1.0.  Symbols absent
            from the output should be treated as weight 0.

        Raises
        ------
        KeyError
            If **SPY** or **SHY** is missing from *data*.
        """
        if _BENCHMARK not in data:
            raise KeyError(f"Benchmark '{_BENCHMARK}' missing from data dict.")
        if _SAFE_HAVEN not in data:
            raise KeyError(f"Safe-haven '{_SAFE_HAVEN}' missing from data dict.")

        # ----------------------------------------------------------------
        # Compute 12-1 momentum for each symbol and for SPY
        # ----------------------------------------------------------------
        spy_mom = self._momentum_12_1(data[_BENCHMARK], rebalance_date, label=_BENCHMARK)
        if spy_mom is None:
            logger.warning(
                "Insufficient data for %s on %s — returning 100%% SHY.",
                _BENCHMARK,
                rebalance_date.date(),
            )
            return {_SAFE_HAVEN: 1.0}

        logger.debug("SPY 12-1 momentum on %s: %.4f", rebalance_date.date(), spy_mom)

        candidates: list[tuple[str, float]] = []  # (symbol, momentum)
        skipped: list[str] = []

        for symbol in self.universe:
            if symbol not in data:
                logger.warning("Symbol %s not found in data — skipping.", symbol)
                skipped.append(symbol)
                continue

            mom = self._momentum_12_1(data[symbol], rebalance_date, label=symbol)
            if mom is None:
                skipped.append(symbol)
                continue

            # Absolute momentum filter: must beat SPY
            if mom > spy_mom:
                candidates.append((symbol, mom))
            else:
                logger.debug(
                    "%s momentum %.4f ≤ SPY momentum %.4f — excluded.",
                    symbol,
                    mom,
                    spy_mom,
                )

        if skipped:
            logger.info(
                "%d symbol(s) skipped due to insufficient data: %s",
                len(skipped),
                ", ".join(skipped),
            )

        # ----------------------------------------------------------------
        # Rank by relative momentum, keep top_n
        # ----------------------------------------------------------------
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected = [sym for sym, _ in candidates[: self.top_n]]

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

        # ----------------------------------------------------------------
        # Build weight dict
        # ----------------------------------------------------------------
        weights: dict[str, float] = {}
        for sym in selected:
            weights[sym] = self._weight_per_slot

        if shy_weight > 0:
            weights[_SAFE_HAVEN] = shy_weight

        # Sanity check — should always hold
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

    def _momentum_12_1(
        self,
        df: pd.DataFrame,
        as_of: pd.Timestamp,
        label: str = "",
    ) -> float | None:
        """Compute 12-1 momentum for a single symbol.

        The return is defined as::

            close[as_of - 21 bars] / close[as_of - 252 bars] - 1

        Parameters
        ----------
        df:
            OHLCV DataFrame with DatetimeIndex, must contain ``close``.
        as_of:
            Reference date (inclusive upper bound).
        label:
            Symbol name used in log messages.

        Returns
        -------
        float | None
            Momentum value, or ``None`` if history is insufficient.
        """
        close = self._get_close(df, as_of)
        if close is None:
            return None

        n = len(close)
        if n < _MIN_BARS:
            logger.warning(
                "%s: only %d bars available on %s (need ≥ %d) — skipping.",
                label,
                n,
                as_of.date(),
                _MIN_BARS,
            )
            return None

        price_12m_ago = close.iloc[-_LOOKBACK_LONG]  # ≈ 12 months ago
        price_1m_ago = close.iloc[-_LOOKBACK_SHORT]  # ≈  1 month ago  (skip)

        if price_12m_ago <= 0:
            logger.warning("%s: price 12 months ago is zero or negative — skipping.", label)
            return None

        return float(price_1m_ago / price_12m_ago - 1.0)

    @staticmethod
    def _get_close(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series | None:
        """Return the ``close`` series up to and including *as_of*.

        Returns ``None`` if the DataFrame is empty after filtering.
        """
        close = df["close"].sort_index()
        close = close.loc[close.index <= as_of]
        if close.empty:
            return None
        return close

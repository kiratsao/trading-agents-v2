"""Slippage and market impact models.

DynamicSlippage
---------------
Models realistic execution cost as a function of *participation rate* — the
fraction of a day's dollar volume that your trade represents.  Larger orders
relative to available liquidity incur proportionally more price impact.

Formula
~~~~~~~
    participation = trade_value / (daily_volume × daily_close)
    slippage_bps  = BASE_BPS + IMPACT_FACTOR × participation
    slippage_bps  = min(slippage_bps, CAP_BPS)

Default constants:
    BASE_BPS      = 2.0   ← bid-ask half-spread for US large-cap equities
    IMPACT_FACTOR = 10.0  ← empirical market-impact coefficient
    CAP_BPS       = 50.0  ← hard cap to prevent extreme-liquidity-crunch outliers

Example: A $100 k trade in a stock with $500 M daily dollar volume
    participation ≈ 0.02 %  →  impact ≈ 0.002 bps  →  total ≈ 2 bps  (≈ 0.02 %)

Comparison with flat 0.05 % (5 bps):
    The dynamic model produces *lower* cost for large-cap liquid names and
    *higher* cost for illiquid or micro-cap names.  For SP500 top-50 the
    majority of trades land well below 5 bps.
"""

from __future__ import annotations

import logging
from typing import Final

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_BPS: Final[float] = 2.0
IMPACT_FACTOR: Final[float] = 10.0
CAP_BPS: Final[float] = 50.0

_MIN_DOLLAR_VOLUME: Final[float] = 1.0  # guard against zero-division


# ---------------------------------------------------------------------------
# DynamicSlippage
# ---------------------------------------------------------------------------


class DynamicSlippage:
    """Participation-rate-based slippage estimator.

    Parameters
    ----------
    base_bps:
        Fixed component representing the typical bid-ask half-spread
        (default 2 bps).
    impact_factor:
        Coefficient for the variable market-impact term (default 10.0).
    cap_bps:
        Maximum slippage in basis points (default 50 bps = 0.5 %).

    Example
    -------
    >>> model = DynamicSlippage()
    >>> bps = model.estimate("AAPL", trade_value=500_000,
    ...                      daily_volume=80_000_000, daily_close=180.0)
    >>> print(f"{bps:.2f} bps")
    """

    def __init__(
        self,
        base_bps: float = BASE_BPS,
        impact_factor: float = IMPACT_FACTOR,
        cap_bps: float = CAP_BPS,
    ) -> None:
        self.base_bps = base_bps
        self.impact_factor = impact_factor
        self.cap_bps = cap_bps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(
        self,
        symbol: str,
        trade_value: float,
        daily_volume: float,
        daily_close: float,
    ) -> float:
        """Estimate one-way slippage in basis points for a single order.

        Parameters
        ----------
        symbol:
            Ticker (used only for logging).
        trade_value:
            Absolute dollar value of the trade (buy or sell).
        daily_volume:
            Number of shares traded on the day (from the OHLCV ``volume``
            column).
        daily_close:
            Closing price on the day (used to convert share volume to
            dollar volume).

        Returns
        -------
        float
            Estimated slippage in basis points, capped at :attr:`cap_bps`.
        """
        if trade_value <= 0:
            return 0.0

        dollar_volume = max(daily_volume * daily_close, _MIN_DOLLAR_VOLUME)
        participation = trade_value / dollar_volume
        bps = self.base_bps + self.impact_factor * participation
        capped = min(bps, self.cap_bps)

        if capped < bps:
            logger.debug(
                "%s: slippage capped at %.1f bps (raw=%.2f bps, trade=$%.0f, dvol=$%.0f)",
                symbol,
                self.cap_bps,
                bps,
                trade_value,
                dollar_volume,
            )

        return capped

    def estimate_portfolio(
        self,
        target_weights: dict[str, float],
        capital: float,
        volume_data: dict[str, float],
        price_data: dict[str, float],
    ) -> dict[str, float]:
        """Estimate per-symbol slippage costs for entering a target portfolio.

        Treats the full target weight × capital as the trade size for each
        symbol (i.e., a buy from zero).  Symbols missing from *volume_data*
        or *price_data* receive a fallback cost using :attr:`base_bps` only.

        Parameters
        ----------
        target_weights:
            ``{symbol: weight}`` — the portfolio to enter.
        capital:
            Total portfolio value in USD.
        volume_data:
            ``{symbol: daily_volume}`` — share count for the trading day.
        price_data:
            ``{symbol: daily_close}`` — closing price for the trading day.

        Returns
        -------
        dict[str, float]
            ``{symbol: estimated_slippage_cost_usd}`` for each symbol with
            a non-zero weight.
        """
        costs: dict[str, float] = {}
        for sym, weight in target_weights.items():
            if weight <= 0:
                continue
            trade_value = weight * capital
            vol = volume_data.get(sym, 0.0)
            price = price_data.get(sym, 0.0)

            if vol > 0 and price > 0:
                bps = self.estimate(sym, trade_value, vol, price)
            else:
                # No liquidity data — fall back to base spread only
                bps = self.base_bps
                logger.debug("%s: no volume/price data — using base_bps=%.1f", sym, self.base_bps)

            costs[sym] = trade_value * bps / 10_000.0

        return costs


# ---------------------------------------------------------------------------
# Legacy module-level functions (kept for backward compatibility)
# ---------------------------------------------------------------------------


def fixed_slippage(price: float, bps: float = 10.0) -> float:
    """Return execution price after fixed basis-point slippage (buy side)."""
    return price * (1.0 + bps / 10_000.0)


def linear_impact(
    price: float,
    qty: int,
    avg_daily_volume: int,
    impact_factor: float = 0.1,
) -> float:
    """Square-root market impact model.  Returns estimated fill price."""
    import math

    if avg_daily_volume <= 0:
        raise ValueError("avg_daily_volume must be positive.")
    participation = abs(qty) / avg_daily_volume
    return price * (1.0 + impact_factor * math.sqrt(participation))

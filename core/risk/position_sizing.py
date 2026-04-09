"""Position sizing models."""

from __future__ import annotations

import math


def fixed_fraction(equity: float, risk_pct: float, stop_distance: float) -> float:
    """Risk a fixed percentage of equity per trade.

    Parameters
    ----------
    equity:
        Total portfolio value.
    risk_pct:
        Fraction of equity to risk (e.g. 0.01 = 1 %).
    stop_distance:
        Fractional distance to the stop-loss (e.g. 0.05 = 5 %).

    Returns
    -------
    float
        Dollar amount to allocate to the position.
    """
    if stop_distance <= 0:
        raise ValueError("stop_distance must be positive.")
    return (equity * risk_pct) / stop_distance


def volatility_target(
    equity: float,
    target_vol: float,
    atr: float,
    price: float,
) -> int:
    """Size position so portfolio volatility contribution equals *target_vol*.

    Uses ATR as a proxy for daily price volatility.

    Parameters
    ----------
    equity:
        Total portfolio value.
    target_vol:
        Target annualised volatility contribution as a fraction (e.g. 0.15).
    atr:
        Average True Range of the instrument (in price units).
    price:
        Current price of the instrument.

    Returns
    -------
    int
        Number of shares (rounded down to whole shares, minimum 1).
    """
    if price <= 0 or atr <= 0:
        raise ValueError("price and atr must be positive.")
    daily_vol_per_share = atr / price
    target_daily_vol = target_vol / math.sqrt(252)
    dollar_position = (equity * target_daily_vol) / daily_vol_per_share
    return max(1, int(dollar_position / price))


def half_kelly(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    equity: float,
) -> float:
    """Half-Kelly criterion position size in dollars.

    Parameters
    ----------
    win_rate:
        Probability of a winning trade (0 < win_rate < 1).
    avg_win:
        Average gain per winning trade as a multiple of the risk unit.
    avg_loss:
        Average loss per losing trade as a multiple of the risk unit.
    equity:
        Total portfolio value.

    Returns
    -------
    float
        Dollar position size.  May be ≤ 0 for strategies with negative edge.
    """
    if avg_loss <= 0:
        raise ValueError("avg_loss must be positive.")
    odds = avg_win / avg_loss
    full_kelly = win_rate - (1 - win_rate) / odds
    return equity * (full_kelly / 2)

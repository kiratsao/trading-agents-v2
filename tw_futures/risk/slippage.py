"""Futures slippage model — tick-based cost estimation for TAIFEX products.

Skeleton.  Implement tick-size aware slippage that accounts for TAIFEX
market depth and time-of-day liquidity profiles.
"""

from __future__ import annotations

import logging
from typing import Final

logger = logging.getLogger(__name__)

# TAIFEX product tick values (TWD per tick)
_TICK_VALUE_TX: Final[float] = 200.0  # 1 index point = TWD 200
_TICK_VALUE_MTX: Final[float] = 50.0  # 1 index point = TWD 50
_DEFAULT_SLIPPAGE_TICKS: Final[float] = 1.0  # 1 tick default


def tick_slippage(
    product: str,
    contracts: int,
    ticks: float = _DEFAULT_SLIPPAGE_TICKS,
) -> float:
    """Estimate round-trip slippage cost in TWD.

    Parameters
    ----------
    product :
        TAIFEX product code ("TX" | "MTX").
    contracts :
        Number of contracts (absolute value used).
    ticks :
        Expected slippage in ticks per side.

    Returns
    -------
    float
        Total estimated slippage cost in TWD (round-trip).
    """
    raise NotImplementedError

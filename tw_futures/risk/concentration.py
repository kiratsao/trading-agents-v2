"""Futures position concentration limits — maximum open contracts per product.

Skeleton.  Enforces TAIFEX position limits and internal risk limits.
"""

from __future__ import annotations

import logging
from typing import Final

logger = logging.getLogger(__name__)

# TAIFEX speculative position limits (single-side)
_TAIFEX_LIMIT_TX: Final[int] = 100  # TAIEX futures (TX)
_TAIFEX_LIMIT_MTX: Final[int] = 400  # Mini TAIEX futures (MTX)


class FuturesConcentrationGuard:
    """Enforce per-product contract limits for TAIFEX futures.

    Parameters
    ----------
    max_tx :
        Maximum open TX contracts (net, per side).
    max_mtx :
        Maximum open MTX contracts (net, per side).
    """

    def __init__(
        self,
        max_tx: int = _TAIFEX_LIMIT_TX,
        max_mtx: int = _TAIFEX_LIMIT_MTX,
    ) -> None:
        self.max_tx = max_tx
        self.max_mtx = max_mtx

    def check(self, positions: dict[str, int]) -> dict[str, str]:
        """Check each product's open contracts against its limit.

        Parameters
        ----------
        positions :
            ``{product_code: net_contracts}`` e.g. ``{"TX": 5, "MTX": -10}``.

        Returns
        -------
        dict
            ``{product_code: "ok" | "warning" | "breach"}``
        """
        raise NotImplementedError

    def max_order_size(self, product: str, current_net: int) -> int:
        """Return the maximum additional contracts that can be placed.

        Parameters
        ----------
        product :
            Product code (e.g. "TX", "MTX").
        current_net :
            Current net open contracts (positive = long, negative = short).
        """
        raise NotImplementedError

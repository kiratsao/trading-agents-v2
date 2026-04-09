"""Futures order manager — manages TAIFEX order lifecycle via ShioajiAdapter.

Skeleton.  Implement sell-before-buy for position reversal and handle
the unique characteristics of futures (no fractional contracts, margin calls).
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tw_futures.executor.shioaji_adapter import ShioajiAdapter

logger = logging.getLogger(__name__)


class FuturesOrderStatus(str, Enum):
    """Lifecycle states for a futures order."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    ERROR = "error"


class FuturesOrderManager:
    """Manage order submission and tracking for TAIFEX futures.

    Key differences from US equity OrderManager
    -------------------------------------------
    - Orders are in contracts (integer), not shares
    - Position reversal requires explicit close + open (no netting by default)
    - Margin must be verified before each order

    Parameters
    ----------
    db_path :
        SQLite path for order persistence.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path

    def rebalance(
        self,
        target_positions: dict[str, int],
        current_positions: dict[str, int],
        adapter: ShioajiAdapter,
    ) -> list[dict[str, Any]]:
        """Rebalance to *target_positions* by submitting the minimum set of orders.

        Parameters
        ----------
        target_positions :
            ``{product_code: target_net_contracts}``
        current_positions :
            ``{product_code: current_net_contracts}``
        adapter :
            Live Shioaji adapter for order submission.

        Returns
        -------
        list[dict]
            Execution results, one dict per order.
        """
        raise NotImplementedError

    def submit_order(
        self,
        product: str,
        action: str,
        quantity: int,
        order_type: str = "market",
        price: float | None = None,
        adapter: ShioajiAdapter | None = None,
    ) -> dict[str, Any]:
        """Submit a single futures order.

        Parameters
        ----------
        product :
            TAIFEX product code (e.g. "TXFJ4", "MXFJ4").
        action :
            "buy" or "sell".
        quantity :
            Number of contracts (must be >= 1).
        order_type :
            "market" | "limit".
        price :
            Required for limit orders.
        adapter :
            Shioaji adapter instance.

        Returns
        -------
        dict
            Order result with status and fill details.
        """
        raise NotImplementedError

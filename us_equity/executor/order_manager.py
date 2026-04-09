"""Order management: state machine + portfolio rebalancing.

The :class:`OrderManager` has two responsibilities:

1. **State machine** — tracks order lifecycle transitions
   (PENDING → SUBMITTED → FILLED / CANCELLED / ERROR) in the DB.
2. **Rebalance** — given target weights and current positions, computes the
   minimum set of trades to reach the target, submits SELLs first (to free
   cash) then BUYs, and optionally runs the full risk pipeline before trading.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from us_equity.executor.alpaca_adapter import AlpacaAdapter

logger = logging.getLogger(__name__)

# Ignore delta values below this dollar threshold (avoids hairline rebalances)
_MIN_TRADE_VALUE: float = 1.0
# Ignore fractional qty below this threshold (Alpaca minimum is ~0.001 shares)
_MIN_QTY: float = 1e-4


# ---------------------------------------------------------------------------
# Order state machine
# ---------------------------------------------------------------------------


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    ERROR = "error"


# ---------------------------------------------------------------------------
# OrderManager
# ---------------------------------------------------------------------------


class OrderManager:
    """Tracks order states and orchestrates portfolio rebalances.

    Parameters
    ----------
    db_path : str
        SQLite path (used for future persistence; state-machine methods are
        functional stubs until the DB layer is wired in ``executor.py``).
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    # ------------------------------------------------------------------
    # State machine (DB-backed stubs)
    # ------------------------------------------------------------------

    def update_status(
        self,
        order_id: str,
        status: OrderStatus,
        fill: dict | None = None,
    ) -> None:
        """Persist a status transition for *order_id*."""
        logger.debug("update_status: order=%s status=%s fill=%s", order_id, status, fill)

    def get_pending_orders(self) -> list[dict]:
        """Return all orders in PENDING or SUBMITTED state from the DB."""
        return []

    def mark_error(self, order_id: str, error: str) -> None:
        """Record an error against *order_id*."""
        logger.error("Order %s error: %s", order_id, error)

    # ------------------------------------------------------------------
    # Portfolio rebalance
    # ------------------------------------------------------------------

    def rebalance(
        self,
        target_weights: dict[str, float],
        current_positions: dict[str, dict],
        equity: float,
        adapter: AlpacaAdapter,
        risk_manager: object | None = None,
        daily_returns: pd.Series | None = None,
    ) -> list[dict]:
        """Rebalance the portfolio to *target_weights*.

        Execution order: all SELL orders first (freeing cash), then BUY orders.
        This prevents cash-shortfall errors when rotating out of one holding
        and into another.

        Parameters
        ----------
        target_weights :
            ``{symbol: weight}`` summing to ~1.0 (output of the strategy /
            risk pipeline).
        current_positions :
            Live positions from ``adapter.get_positions()``.
            Each value must contain ``market_value`` and ``current_price``
            (or ``qty``).
        equity :
            Current portfolio NAV in USD (from ``adapter.get_account()``).
        adapter :
            Broker adapter used to submit orders.
        risk_manager :
            Optional :class:`~agents.risk_manager.risk_manager.RiskManager`.
            When provided, ``validate_trade()`` is called before computing
            deltas; the returned (possibly adjusted) weights are used.
        daily_returns :
            Recent portfolio daily returns passed to
            ``risk_manager.validate_trade()`` for the Kill Switch check.

        Returns
        -------
        list[dict]
            One record per submitted order (including error records for failed
            orders).  Each dict contains at minimum:
            ``{symbol, side, status}`` plus broker-filled fields when available.
        """
        if equity <= 0:
            logger.error("rebalance: equity=%.2f ≤ 0, aborting.", equity)
            return []

        # ---- Step 1: run risk pipeline (optional) -----------------------
        if risk_manager is not None:
            try:
                target_weights = risk_manager.validate_trade(
                    target_weights,
                    equity,
                    daily_returns if daily_returns is not None else pd.Series(dtype=float),
                )
                logger.info(
                    "rebalance: risk pipeline approved weights for %d symbol(s).",
                    len(target_weights),
                )
            except Exception as exc:
                logger.error("rebalance: risk pipeline rejected trade — %s", exc)
                raise

        # ---- Step 2: compute per-symbol quantity deltas -----------------
        sells: list[tuple[str, float]] = []  # (symbol, abs_qty_to_sell)
        buys: list[tuple[str, float]] = []  # (symbol, abs_qty_to_buy)

        all_symbols = set(target_weights) | set(current_positions)

        for symbol in all_symbols:
            target_weight = target_weights.get(symbol, 0.0)
            target_value = target_weight * equity
            current_value = current_positions.get(symbol, {}).get("market_value", 0.0)
            delta_value = target_value - current_value

            if abs(delta_value) < _MIN_TRADE_VALUE:
                logger.debug(
                    "%s: |delta_value|=%.2f below threshold, skipping.", symbol, delta_value
                )
                continue

            price = self._get_price(symbol, current_positions, adapter)
            if price <= 0:
                logger.warning("%s: could not determine current price — skipping this leg.", symbol)
                continue

            delta_qty = delta_value / price

            if delta_qty < -_MIN_QTY:
                sells.append((symbol, abs(delta_qty)))
            elif delta_qty > _MIN_QTY:
                buys.append((symbol, delta_qty))

        logger.info(
            "rebalance: %d sell(s), %d buy(s) | equity=$%.2f",
            len(sells),
            len(buys),
            equity,
        )

        # ---- Step 3: SELLs first, then BUYs ----------------------------
        from us_equity.executor.alpaca_adapter import ExecutionError

        results: list[dict] = []

        for symbol, qty in sells:
            try:
                conf = adapter.submit_order(symbol, qty, "sell")
                conf["computed_delta_qty"] = -round(qty, 6)
                logger.info(
                    "SELL %-6s qty=%10.4f | id=%-36s status=%s  filled=%.4f @ %.4f",
                    symbol,
                    qty,
                    conf["order_id"],
                    conf["status"],
                    conf["filled_qty"],
                    conf["filled_avg_price"],
                )
                results.append(conf)
            except ExecutionError as exc:
                logger.error("SELL %s failed: %s", symbol, exc)
                results.append(
                    {
                        "symbol": symbol,
                        "side": "sell",
                        "qty": qty,
                        "status": "error",
                        "error": str(exc),
                    }
                )

        for symbol, qty in buys:
            try:
                conf = adapter.submit_order(symbol, qty, "buy")
                conf["computed_delta_qty"] = round(qty, 6)
                logger.info(
                    "BUY  %-6s qty=%10.4f | id=%-36s status=%s  filled=%.4f @ %.4f",
                    symbol,
                    qty,
                    conf["order_id"],
                    conf["status"],
                    conf["filled_qty"],
                    conf["filled_avg_price"],
                )
                results.append(conf)
            except ExecutionError as exc:
                logger.error("BUY %s failed: %s", symbol, exc)
                results.append(
                    {
                        "symbol": symbol,
                        "side": "buy",
                        "qty": qty,
                        "status": "error",
                        "error": str(exc),
                    }
                )

        ok = sum(1 for r in results if r.get("status") != "error")
        err = len(results) - ok
        logger.info(
            "rebalance complete: %d order(s) submitted (%d sell, %d buy) | %d ok, %d error(s).",
            len(results),
            len(sells),
            len(buys),
            ok,
            err,
        )
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_price(
        self,
        symbol: str,
        current_positions: dict[str, dict],
        adapter: AlpacaAdapter,
    ) -> float:
        """Return the best available price for *symbol*.

        Priority:
        1. ``current_positions[symbol]["current_price"]``
        2. ``market_value / qty`` derived from the position
        3. ``adapter.get_latest_price(symbol)`` for new positions
        """
        pos = current_positions.get(symbol)
        if pos:
            price = float(pos.get("current_price", 0) or 0)
            if price > 0:
                return price
            mv = float(pos.get("market_value", 0) or 0)
            qty = float(pos.get("qty", 0) or 0)
            if mv > 0 and qty > 0:
                return mv / qty

        # Symbol not yet in portfolio — query broker for latest price
        try:
            return adapter.get_latest_price(symbol)
        except Exception as exc:
            logger.warning("_get_price(%s): adapter failed — %s", symbol, exc)
            return 0.0

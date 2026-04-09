"""Executor orchestrator — routes approved orders to broker APIs."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class Executor:
    """Polls approved_orders and submits them to the appropriate broker."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def run(self) -> None:
        """Main loop: poll approved_orders → route → update trades/positions."""
        raise NotImplementedError

    def execute_order(self, order: dict) -> dict:
        """Submit a single order to the correct broker. Returns fill dict."""
        raise NotImplementedError

    def _is_trading_halted(self) -> bool:
        """Check system_state.trading_halted flag in DB."""
        raise NotImplementedError

    def _select_broker(self, symbol: str) -> object:
        """Return the appropriate BrokerAdapter for the symbol's market."""
        raise NotImplementedError

    def _record_fill(self, order: dict, fill: dict) -> None:
        """Write to trades table and update positions table."""
        raise NotImplementedError


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Executor Agent")
    parser.add_argument("--order-id", type=str, help="Execute a specific order ID")
    args = parser.parse_args()
    logger.info("Executor starting (order_id=%s)", args.order_id)

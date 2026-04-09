"""Fugle (富果) broker adapter for Taiwan stocks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OrderResult:
    order_id: str
    status: str
    filled_qty: int
    filled_price: float | None


@dataclass
class Position:
    symbol: str
    qty: int
    avg_cost: float


@dataclass
class AccountInfo:
    equity: float
    cash: float
    margin_used: float


class FugleAdapter:
    """Wraps the fugle-trade SDK. Taiwan lot size = 1000 shares per lot (一張)."""

    LOT_SIZE = 1000

    def __init__(self, api_key: str, secret_key: str) -> None:
        self.api_key = api_key
        self.secret_key = secret_key
        self._client = None  # fugle_trade client, set in connect()

    def connect(self) -> None:
        raise NotImplementedError

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        order_type: str = "ROD",
        price: float | None = None,
    ) -> OrderResult:
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError

    def get_positions(self) -> list[Position]:
        raise NotImplementedError

    def get_account(self) -> AccountInfo:
        raise NotImplementedError

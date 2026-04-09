"""Alpaca Paper/Live Trading adapter using alpaca-py SDK.

Usage
-----
    adapter = AlpacaAdapter(api_key=..., secret_key=..., paper=True)
    account   = adapter.get_account()
    positions = adapter.get_positions()
    conf      = adapter.submit_order("AAPL", qty=10.0, side="buy")
"""

from __future__ import annotations

import logging
import time
from typing import Any

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

logger = logging.getLogger(__name__)

# HTTP status codes that are worth retrying (server-side / rate-limit).
# 4xx client errors (except 429) are non-retryable and surface immediately.
_RETRYABLE_HTTP_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

# Smallest fractional-share qty Alpaca will accept
_MIN_ORDER_QTY: float = 1e-6


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class ExecutionError(RuntimeError):
    """Raised when a broker operation fails after all retry attempts.

    Attributes
    ----------
    original : Exception | None
        The underlying exception that caused the failure.
    """

    def __init__(self, message: str, original: Exception | None = None) -> None:
        super().__init__(message)
        self.original = original


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class AlpacaAdapter:
    """Alpaca Paper/Live Trading adapter (alpaca-py SDK).

    All public methods wrap the underlying SDK calls with exponential back-off
    retry (up to *max_retries* attempts).  Non-retryable API errors (4xx except
    429) are surfaced immediately as :class:`ExecutionError`.

    Parameters
    ----------
    api_key :
        Alpaca API key ID.
    secret_key :
        Alpaca secret key.
    paper :
        ``True`` → paper-trading endpoint (default).
        ``False`` → live-trading endpoint.
    max_retries :
        Total attempts per API call (default 3 → back-off delays 1 s, 2 s).
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self._max_retries = max_retries
        self._client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
        )
        # Optional data client for latest-price queries (non-fatal if unavailable)
        self._data_client = self._build_data_client(api_key, secret_key)

    # ------------------------------------------------------------------
    # Legacy connect() — kept for BrokerAdapter protocol compatibility
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """No-op.  TradingClient connects lazily on the first request."""

    # ------------------------------------------------------------------
    # Account & positions
    # ------------------------------------------------------------------

    def get_account(self) -> dict:
        """Return a live account snapshot.

        Returns
        -------
        dict
            ``{equity, cash, buying_power, portfolio_value}`` — all float USD.
        """
        acct = self._with_retry(self._client.get_account)
        return {
            "equity": float(acct.equity or 0),
            "cash": float(acct.cash or 0),
            "buying_power": float(acct.buying_power or 0),
            "portfolio_value": float(acct.portfolio_value or 0),
        }

    def get_positions(self) -> dict[str, dict]:
        """Return all open positions keyed by symbol.

        Returns
        -------
        dict[str, dict]
            ``{symbol: {qty, market_value, avg_entry, unrealized_pnl,
            current_price}}``
        """
        raw = self._with_retry(self._client.get_all_positions)
        result: dict[str, dict] = {}
        for pos in raw:
            result[pos.symbol] = {
                "qty": float(pos.qty or 0),
                "market_value": float(pos.market_value or 0),
                "avg_entry": float(pos.avg_entry_price or 0),
                "unrealized_pnl": float(pos.unrealized_pl or 0),
                "current_price": float(pos.current_price or 0),
            }
        return result

    def get_latest_price(self, symbol: str) -> float:
        """Return the most recent trade price for *symbol*.

        Called by :class:`~agents.executor.order_manager.OrderManager` when a
        symbol is not in the current portfolio (no live position available).

        Raises
        ------
        ExecutionError
            If the data client is unavailable or the query fails.
        """
        if self._data_client is None:
            raise ExecutionError(
                f"Data client unavailable; cannot fetch latest price for {symbol}."
            )
        try:
            from alpaca.data.requests import StockLatestTradeRequest

            req = StockLatestTradeRequest(symbol_or_symbols=[symbol])
            trades = self._with_retry(self._data_client.get_stock_latest_trade, req)
            trade = trades.get(symbol)
            if trade is None:
                raise ExecutionError(f"No latest trade data returned for {symbol}.")
            return float(trade.price)
        except ExecutionError:
            raise
        except Exception as exc:
            raise ExecutionError(f"Failed to get latest price for {symbol}: {exc}", exc) from exc

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        limit_price: float | None = None,
    ) -> dict:
        """Submit a single order.

        Parameters
        ----------
        symbol :
            Ticker (e.g. ``"AAPL"``).
        qty :
            Fractional share quantity.  Must be ≥ 1e-6.
        side :
            ``"buy"`` or ``"sell"``.
        order_type :
            ``"market"`` (default) or ``"limit"``.
        limit_price :
            Required when *order_type* is ``"limit"``.

        Returns
        -------
        dict
            ``{order_id, client_order_id, symbol, side, status,
            filled_qty, filled_avg_price, order_type}``

        Raises
        ------
        ExecutionError
            On invalid parameters or API failure after all retries.
        """
        if qty < _MIN_ORDER_QTY:
            raise ExecutionError(
                f"submit_order: qty={qty:.8f} is below minimum {_MIN_ORDER_QTY}. "
                "Caller should skip sub-minimum orders."
            )

        alpaca_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        if order_type == "market":
            request: MarketOrderRequest | LimitOrderRequest = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=alpaca_side,
                time_in_force=TimeInForce.DAY,
            )
        elif order_type == "limit":
            if limit_price is None:
                raise ExecutionError("limit_price must be set for limit orders.")
            request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=alpaca_side,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price,
            )
        else:
            raise ExecutionError(f"Unsupported order_type: {order_type!r}")

        order = self._with_retry(self._client.submit_order, request)

        price_str = "MKT" if order_type == "market" else f"LMT {limit_price}"
        logger.info(
            "Order submitted: %s %s %.6f shares @ %s | id=%s status=%s",
            side.upper(),
            symbol,
            qty,
            price_str,
            order.id,
            order.status,
        )
        return self._order_to_dict(order)

    def close_position(self, symbol: str) -> dict:
        """Flatten a single position with a market sell.

        Returns
        -------
        dict
            Order confirmation (same shape as :meth:`submit_order`).
        """
        order = self._with_retry(self._client.close_position, symbol)
        logger.info("close_position: %s | order_id=%s status=%s", symbol, order.id, order.status)
        return self._order_to_dict(order)

    def close_all_positions(self) -> list[dict]:
        """Flatten *all* open positions (Kill Switch emergency use case).

        Also cancels any pending open orders before liquidating.

        Returns
        -------
        list[dict]
            One order confirmation per position closed.
        """
        logger.warning(
            "close_all_positions() called — liquidating entire portfolio. "
            "This is a Kill Switch emergency operation."
        )
        responses = self._with_retry(self._client.close_all_positions, cancel_orders=True)
        results: list[dict] = []
        for resp in responses or []:
            try:
                results.append(self._order_to_dict(resp))
            except Exception:
                # Individual items may be error responses for illiquid symbols
                results.append({"raw": str(resp), "status": "error"})

        logger.info("close_all_positions: %d liquidation order(s) submitted.", len(results))
        return results

    # ------------------------------------------------------------------
    # Legacy interface (BrokerAdapter protocol)
    # ------------------------------------------------------------------

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        order_type: str = "market",
        price: float | None = None,
    ) -> dict:
        """Legacy alias — delegates to :meth:`submit_order`."""
        return self.submit_order(symbol, float(qty), side, order_type, price)

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order by ID.

        Returns
        -------
        bool
            ``True`` if cancelled; ``False`` if the cancel failed.
        """
        try:
            self._with_retry(self._client.cancel_order_by_id, order_id)
            logger.info("cancel_order: %s cancelled.", order_id)
            return True
        except ExecutionError as exc:
            logger.warning("cancel_order %s failed: %s", order_id, exc)
            return False

    # ------------------------------------------------------------------
    # Retry helper
    # ------------------------------------------------------------------

    def _with_retry(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Call *fn(*args, **kwargs)* with exponential back-off on transient errors.

        Back-off delays: 1 s, 2 s  (2^0, 2^1).

        Raises
        ------
        ExecutionError
            After all retries are exhausted, or immediately on a non-retryable
            HTTP 4xx error.
        """
        last_exc: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                return fn(*args, **kwargs)

            except APIError as exc:
                sc = exc.status_code
                # Fail fast on non-retryable client errors (400, 403, 404, 422 …)
                if sc is not None and sc not in _RETRYABLE_HTTP_CODES:
                    raise ExecutionError(
                        f"Alpaca API non-retryable error [{sc}]: {exc}", exc
                    ) from exc
                last_exc = exc
                logger.warning(
                    "Alpaca API error (attempt %d/%d, http=%s): %s",
                    attempt,
                    self._max_retries,
                    sc,
                    exc,
                )

            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Unexpected error calling %s (attempt %d/%d): %s",
                    getattr(fn, "__name__", str(fn)),
                    attempt,
                    self._max_retries,
                    exc,
                )

            if attempt < self._max_retries:
                sleep_secs = 2 ** (attempt - 1)  # 1 s then 2 s
                logger.debug("Retrying in %d s ...", sleep_secs)
                time.sleep(sleep_secs)

        raise ExecutionError(
            f"Alpaca API failed after {self._max_retries} attempt(s): {last_exc}",
            last_exc,
        ) from last_exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _order_to_dict(order: Any) -> dict:
        """Normalise an alpaca-py Order object into a plain dict."""
        return {
            "order_id": str(getattr(order, "id", "") or ""),
            "client_order_id": str(getattr(order, "client_order_id", "") or ""),
            "symbol": str(getattr(order, "symbol", "") or ""),
            "side": str(getattr(order, "side", "") or ""),
            "status": str(getattr(order, "status", "") or ""),
            "filled_qty": float(getattr(order, "filled_qty", 0) or 0),
            "filled_avg_price": float(getattr(order, "filled_avg_price", 0) or 0),
            "order_type": str(getattr(order, "order_type", getattr(order, "type", "")) or ""),
        }

    @staticmethod
    def _build_data_client(api_key: str, secret_key: str) -> Any:
        """Construct StockHistoricalDataClient; silently return None on failure."""
        try:
            from alpaca.data import StockHistoricalDataClient

            return StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
        except Exception as exc:  # pragma: no cover
            logger.debug("Could not build StockHistoricalDataClient: %s", exc)
            return None

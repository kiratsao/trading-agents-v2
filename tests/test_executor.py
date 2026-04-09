"""Tests for the Executor agent.

Coverage
--------
* AlpacaAdapter — interface contract, paper-mode default, retry logic,
  error handling, order/position normalisation.
* OrderManager — state machine stubs, sell-before-buy ordering, delta
  computation, risk-pipeline integration, zero-equity guard, error recovery.
* Reconciler — drift calculation, status classification, equity-zero guard.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tw_futures.executor.shioaji_adapter import ShioajiAdapter
from us_equity.executor.alpaca_adapter import AlpacaAdapter, ExecutionError
from us_equity.executor.order_manager import OrderManager, OrderStatus
from us_equity.executor.reconciler import Reconciler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_order(
    order_id: str = "ord-1",
    symbol: str = "AAPL",
    side: str = "buy",
    status: str = "accepted",
    filled_qty: float = 0.0,
    filled_avg_price: float = 0.0,
) -> MagicMock:
    """Return a MagicMock that quacks like an alpaca-py Order."""
    o = MagicMock()
    o.id = order_id
    o.client_order_id = f"client-{order_id}"
    o.symbol = symbol
    o.side = side
    o.status = status
    o.filled_qty = filled_qty
    o.filled_avg_price = filled_avg_price
    o.order_type = "market"
    o.type = "market"
    return o


def _make_adapter(positions: dict[str, dict] | None = None) -> MagicMock:
    """Return a mock AlpacaAdapter with sensible defaults."""
    adapter = MagicMock(spec=AlpacaAdapter)

    # submit_order returns a confirmation dict (side recorded from call args)
    def _submit(symbol, qty, side, order_type="market", limit_price=None):
        return {
            "order_id": f"ord-{symbol}-{side}",
            "client_order_id": "",
            "symbol": symbol,
            "side": side,
            "status": "accepted",
            "filled_qty": 0.0,
            "filled_avg_price": 0.0,
            "order_type": order_type,
        }

    adapter.submit_order.side_effect = _submit

    # get_positions
    adapter.get_positions.return_value = positions or {}

    # get_latest_price — default 100.0 for any symbol
    adapter.get_latest_price.return_value = 100.0

    return adapter


# ===========================================================================
# AlpacaAdapter — construction & interface
# ===========================================================================


class TestAlpacaAdapterConstruction:
    """AlpacaAdapter constructs correctly; TradingClient is patched away."""

    def test_paper_mode_default_true(self):
        with (
            patch("us_equity.executor.alpaca_adapter.TradingClient"),
            patch("us_equity.executor.alpaca_adapter.AlpacaAdapter._build_data_client"),
        ):
            adapter = AlpacaAdapter(api_key="k", secret_key="s")
        assert adapter.paper is True

    def test_paper_mode_explicit_false(self):
        with (
            patch("us_equity.executor.alpaca_adapter.TradingClient"),
            patch("us_equity.executor.alpaca_adapter.AlpacaAdapter._build_data_client"),
        ):
            adapter = AlpacaAdapter(api_key="k", secret_key="s", paper=False)
        assert adapter.paper is False

    def test_has_required_methods(self):
        expected = (
            "connect",
            "get_account",
            "get_positions",
            "get_latest_price",
            "submit_order",
            "close_position",
            "close_all_positions",
            "place_order",
            "cancel_order",
        )
        for method in expected:
            assert hasattr(AlpacaAdapter, method), f"Missing method: {method}"

    def test_connect_is_noop(self):
        with (
            patch("us_equity.executor.alpaca_adapter.TradingClient"),
            patch("us_equity.executor.alpaca_adapter.AlpacaAdapter._build_data_client"),
        ):
            adapter = AlpacaAdapter(api_key="k", secret_key="s")
        adapter.connect()  # must not raise


class TestAlpacaAdapterGetAccount:
    def test_normalises_account_fields(self):
        mock_acct = MagicMock()
        mock_acct.equity = "123456.78"
        mock_acct.cash = "50000.00"
        mock_acct.buying_power = "100000.00"
        mock_acct.portfolio_value = "123456.78"

        with (
            patch("us_equity.executor.alpaca_adapter.TradingClient") as MockTC,
            patch("us_equity.executor.alpaca_adapter.AlpacaAdapter._build_data_client"),
        ):
            MockTC.return_value.get_account.return_value = mock_acct
            adapter = AlpacaAdapter(api_key="k", secret_key="s")
            result = adapter.get_account()

        assert result["equity"] == pytest.approx(123456.78)
        assert result["cash"] == pytest.approx(50000.0)
        assert result["buying_power"] == pytest.approx(100000.0)


class TestAlpacaAdapterGetPositions:
    def test_normalises_position_fields(self):
        pos = MagicMock()
        pos.symbol = "NVDA"
        pos.qty = "5.0"
        pos.market_value = "2500.00"
        pos.avg_entry_price = "480.00"
        pos.unrealized_pl = "100.00"
        pos.current_price = "500.00"

        with (
            patch("us_equity.executor.alpaca_adapter.TradingClient") as MockTC,
            patch("us_equity.executor.alpaca_adapter.AlpacaAdapter._build_data_client"),
        ):
            MockTC.return_value.get_all_positions.return_value = [pos]
            adapter = AlpacaAdapter(api_key="k", secret_key="s")
            positions = adapter.get_positions()

        assert "NVDA" in positions
        p = positions["NVDA"]
        assert p["qty"] == pytest.approx(5.0)
        assert p["market_value"] == pytest.approx(2500.0)
        assert p["avg_entry"] == pytest.approx(480.0)
        assert p["unrealized_pnl"] == pytest.approx(100.0)
        assert p["current_price"] == pytest.approx(500.0)

    def test_empty_positions(self):
        with (
            patch("us_equity.executor.alpaca_adapter.TradingClient") as MockTC,
            patch("us_equity.executor.alpaca_adapter.AlpacaAdapter._build_data_client"),
        ):
            MockTC.return_value.get_all_positions.return_value = []
            adapter = AlpacaAdapter(api_key="k", secret_key="s")
            assert adapter.get_positions() == {}


class TestAlpacaAdapterSubmitOrder:
    def test_market_buy_returns_confirmation(self):
        order = _make_order("ord-99", "AAPL", "buy", "accepted")
        with (
            patch("us_equity.executor.alpaca_adapter.TradingClient") as MockTC,
            patch("us_equity.executor.alpaca_adapter.AlpacaAdapter._build_data_client"),
        ):
            MockTC.return_value.submit_order.return_value = order
            adapter = AlpacaAdapter(api_key="k", secret_key="s")
            result = adapter.submit_order("AAPL", 10.0, "buy")

        assert result["order_id"] == "ord-99"
        assert result["symbol"] == "AAPL"
        assert result["side"] == "buy"
        assert result["status"] == "accepted"

    def test_sub_minimum_qty_raises_execution_error(self):
        with (
            patch("us_equity.executor.alpaca_adapter.TradingClient"),
            patch("us_equity.executor.alpaca_adapter.AlpacaAdapter._build_data_client"),
        ):
            adapter = AlpacaAdapter(api_key="k", secret_key="s")
        with pytest.raises(ExecutionError, match="below minimum"):
            adapter.submit_order("AAPL", 0.0, "buy")

    def test_limit_order_without_price_raises(self):
        with (
            patch("us_equity.executor.alpaca_adapter.TradingClient"),
            patch("us_equity.executor.alpaca_adapter.AlpacaAdapter._build_data_client"),
        ):
            adapter = AlpacaAdapter(api_key="k", secret_key="s")
        with pytest.raises(ExecutionError, match="limit_price"):
            adapter.submit_order("AAPL", 1.0, "buy", order_type="limit")

    def test_unknown_order_type_raises(self):
        with (
            patch("us_equity.executor.alpaca_adapter.TradingClient"),
            patch("us_equity.executor.alpaca_adapter.AlpacaAdapter._build_data_client"),
        ):
            adapter = AlpacaAdapter(api_key="k", secret_key="s")
        with pytest.raises(ExecutionError, match="Unsupported order_type"):
            adapter.submit_order("AAPL", 1.0, "buy", order_type="stop")


class TestAlpacaAdapterRetry:
    """Verify retry logic and ExecutionError surfaces correctly."""

    def test_retries_on_500_then_succeeds(self):
        from alpaca.common.exceptions import APIError

        order = _make_order()
        call_count = 0

        def flaky(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                exc = APIError("{}")
                exc._http_error = MagicMock()
                exc._http_error.response.status_code = 500
                raise exc
            return order

        with (
            patch("us_equity.executor.alpaca_adapter.TradingClient") as MockTC,
            patch("us_equity.executor.alpaca_adapter.AlpacaAdapter._build_data_client"),
            patch("us_equity.executor.alpaca_adapter.time.sleep"),
        ):
            MockTC.return_value.submit_order.side_effect = flaky
            adapter = AlpacaAdapter(api_key="k", secret_key="s", max_retries=3)
            result = adapter.submit_order("AAPL", 1.0, "buy")

        assert result["order_id"] == order.id
        assert call_count == 3

    def test_non_retryable_400_raises_immediately(self):
        from alpaca.common.exceptions import APIError

        call_count = 0

        def always_400(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            exc = APIError("{}")
            exc._http_error = MagicMock()
            exc._http_error.response.status_code = 400
            raise exc

        with (
            patch("us_equity.executor.alpaca_adapter.TradingClient") as MockTC,
            patch("us_equity.executor.alpaca_adapter.AlpacaAdapter._build_data_client"),
            patch("us_equity.executor.alpaca_adapter.time.sleep"),
        ):
            MockTC.return_value.submit_order.side_effect = always_400
            adapter = AlpacaAdapter(api_key="k", secret_key="s", max_retries=3)
            with pytest.raises(ExecutionError, match="non-retryable"):
                adapter.submit_order("AAPL", 1.0, "buy")

        # Should fail on the very first attempt without retrying
        assert call_count == 1

    def test_exhausted_retries_raises_execution_error(self):
        from alpaca.common.exceptions import APIError

        def always_503(*_args, **_kwargs):
            exc = APIError("{}")
            exc._http_error = MagicMock()
            exc._http_error.response.status_code = 503
            raise exc

        with (
            patch("us_equity.executor.alpaca_adapter.TradingClient") as MockTC,
            patch("us_equity.executor.alpaca_adapter.AlpacaAdapter._build_data_client"),
            patch("us_equity.executor.alpaca_adapter.time.sleep"),
        ):
            MockTC.return_value.submit_order.side_effect = always_503
            adapter = AlpacaAdapter(api_key="k", secret_key="s", max_retries=3)
            with pytest.raises(ExecutionError, match="3 attempt"):
                adapter.submit_order("AAPL", 1.0, "buy")


class TestAlpacaAdapterClosePositions:
    def test_close_position_returns_dict(self):
        order = _make_order("ord-close", "MSFT", "sell")
        with (
            patch("us_equity.executor.alpaca_adapter.TradingClient") as MockTC,
            patch("us_equity.executor.alpaca_adapter.AlpacaAdapter._build_data_client"),
        ):
            MockTC.return_value.close_position.return_value = order
            adapter = AlpacaAdapter(api_key="k", secret_key="s")
            result = adapter.close_position("MSFT")
        assert result["symbol"] == "MSFT"
        assert result["order_id"] == "ord-close"

    def test_close_all_positions_returns_list(self):
        orders = [_make_order(f"ord-{i}", f"SYM{i}", "sell") for i in range(3)]
        with (
            patch("us_equity.executor.alpaca_adapter.TradingClient") as MockTC,
            patch("us_equity.executor.alpaca_adapter.AlpacaAdapter._build_data_client"),
        ):
            MockTC.return_value.close_all_positions.return_value = orders
            adapter = AlpacaAdapter(api_key="k", secret_key="s")
            results = adapter.close_all_positions()
        assert len(results) == 3
        symbols = {r["symbol"] for r in results}
        assert symbols == {"SYM0", "SYM1", "SYM2"}

    def test_close_all_empty_portfolio(self):
        with (
            patch("us_equity.executor.alpaca_adapter.TradingClient") as MockTC,
            patch("us_equity.executor.alpaca_adapter.AlpacaAdapter._build_data_client"),
        ):
            MockTC.return_value.close_all_positions.return_value = []
            adapter = AlpacaAdapter(api_key="k", secret_key="s")
            results = adapter.close_all_positions()
        assert results == []


# ===========================================================================
# OrderManager — state machine stubs (backward-compat)
# ===========================================================================


class TestOrderManagerStateMachine:
    def test_get_pending_orders_returns_list(self, tmp_path):
        mgr = OrderManager(db_path=str(tmp_path / "test.db"))
        assert isinstance(mgr.get_pending_orders(), list)

    def test_update_status_does_not_raise(self, tmp_path):
        mgr = OrderManager(db_path=str(tmp_path / "test.db"))
        mgr.update_status("ord-1", OrderStatus.FILLED, fill={"price": 150.0, "qty": 10})

    def test_mark_error_does_not_raise(self, tmp_path):
        mgr = OrderManager(db_path=str(tmp_path / "test.db"))
        mgr.mark_error("ord-1", error="Insufficient funds")


# ===========================================================================
# OrderManager.rebalance — core logic
# ===========================================================================


class TestOrderManagerRebalanceSellBeforeBuy:
    """Verify that sells are always submitted before buys."""

    def test_sells_submitted_before_buys(self, tmp_path):
        """All SELL calls must precede any BUY call in submit_order invocations."""
        adapter = _make_adapter(
            positions={
                "AAPL": {"market_value": 20_000.0, "current_price": 200.0, "qty": 100},
                "MSFT": {"market_value": 10_000.0, "current_price": 100.0, "qty": 100},
            }
        )
        # Target: drop AAPL from 20 % to 5 %, increase MSFT from 10 % to 25 %
        target = {"AAPL": 0.05, "MSFT": 0.25, "NVDA": 0.10}
        mgr = OrderManager(db_path=str(tmp_path / "test.db"))
        mgr.rebalance(target, adapter.get_positions(), equity=100_000.0, adapter=adapter)

        sides = [c.args[2] for c in adapter.submit_order.call_args_list]
        # Find the index of the first buy
        first_buy_idx = next((i for i, s in enumerate(sides) if s == "buy"), len(sides))
        # All sells must come before the first buy
        for i, s in enumerate(sides):
            if s == "sell":
                assert i < first_buy_idx, (
                    f"SELL at position {i} comes after a BUY at position {first_buy_idx}"
                )

    def test_only_sells_when_reducing_all_positions(self, tmp_path):
        adapter = _make_adapter(
            positions={
                "AAPL": {"market_value": 50_000.0, "current_price": 500.0, "qty": 100},
                "MSFT": {"market_value": 50_000.0, "current_price": 250.0, "qty": 200},
            }
        )
        # Liquidate to all-cash (SHY)
        target = {"SHY": 1.0}
        mgr = OrderManager(db_path=str(tmp_path / "test.db"))
        # SHY not in current_positions, so get_latest_price will be called
        adapter.get_latest_price.return_value = 85.0  # SHY price
        results = mgr.rebalance(target, adapter.get_positions(), equity=100_000.0, adapter=adapter)

        sides = [r["side"] for r in results]
        assert "sell" in sides
        # BUY for SHY should follow all sells
        sell_indices = [i for i, s in enumerate(sides) if s == "sell"]
        buy_indices = [i for i, s in enumerate(sides) if s == "buy"]
        if buy_indices:
            assert max(sell_indices) < min(buy_indices)

    def test_only_buys_when_deploying_new_capital(self, tmp_path):
        """Starting from all-cash: only BUY orders, no SELLs."""
        adapter = _make_adapter(positions={})  # empty portfolio
        adapter.get_latest_price.return_value = 150.0

        target = {"AAPL": 0.5, "MSFT": 0.5}
        mgr = OrderManager(db_path=str(tmp_path / "test.db"))
        results = mgr.rebalance(target, {}, equity=100_000.0, adapter=adapter)

        sides = [r["side"] for r in results]
        assert all(s == "buy" for s in sides)
        assert len(sides) == 2


class TestOrderManagerRebalanceDeltaComputation:
    """Verify correct delta calculations and skipping logic."""

    def test_no_trade_when_already_at_target(self, tmp_path):
        """No orders if current allocation exactly matches target."""
        equity = 100_000.0
        positions = {
            "AAPL": {"market_value": 10_000.0, "current_price": 100.0, "qty": 100},
        }
        target = {"AAPL": 0.10}  # exactly 10 % of 100k = 10k → no trade
        adapter = _make_adapter(positions=positions)
        mgr = OrderManager(db_path=str(tmp_path / "test.db"))
        results = mgr.rebalance(target, positions, equity=equity, adapter=adapter)
        adapter.submit_order.assert_not_called()
        assert results == []

    def test_sells_symbol_not_in_target(self, tmp_path):
        """Symbol in current holdings but absent from target → full sell."""
        positions = {
            "AAPL": {"market_value": 50_000.0, "current_price": 500.0, "qty": 100},
        }
        target = {}  # sell everything
        adapter = _make_adapter(positions=positions)
        mgr = OrderManager(db_path=str(tmp_path / "test.db"))
        results = mgr.rebalance(target, positions, equity=100_000.0, adapter=adapter)

        assert len(results) == 1
        assert results[0]["side"] == "sell"
        assert results[0]["symbol"] == "AAPL"

    def test_zero_equity_returns_empty(self, tmp_path):
        adapter = _make_adapter()
        mgr = OrderManager(db_path=str(tmp_path / "test.db"))
        results = mgr.rebalance({"AAPL": 1.0}, {}, equity=0.0, adapter=adapter)
        assert results == []
        adapter.submit_order.assert_not_called()

    def test_skips_symbol_with_no_price(self, tmp_path):
        """Symbol with zero current_price and failing get_latest_price is skipped."""
        positions = {
            "AAPL": {"market_value": 0.0, "current_price": 0.0, "qty": 0},
        }
        adapter = _make_adapter(positions=positions)
        adapter.get_latest_price.side_effect = ExecutionError("no price")
        target = {"AAPL": 0.5}
        mgr = OrderManager(db_path=str(tmp_path / "test.db"))
        results = mgr.rebalance(target, positions, equity=10_000.0, adapter=adapter)
        adapter.submit_order.assert_not_called()
        assert results == []


class TestOrderManagerRebalanceErrorHandling:
    def test_sell_failure_recorded_as_error_result(self, tmp_path):
        """A failing SELL is recorded in results; execution continues for remaining orders."""
        positions = {
            "AAPL": {"market_value": 30_000.0, "current_price": 300.0, "qty": 100},
            "MSFT": {"market_value": 30_000.0, "current_price": 150.0, "qty": 200},
        }
        adapter = _make_adapter(positions=positions)

        def _submit(symbol, qty, side, **_kw):
            if symbol == "AAPL" and side == "sell":
                raise ExecutionError("AAPL sell rejected")
            return {
                "order_id": f"ord-{symbol}",
                "client_order_id": "",
                "symbol": symbol,
                "side": side,
                "status": "accepted",
                "filled_qty": 0.0,
                "filled_avg_price": 0.0,
                "order_type": "market",
            }

        adapter.submit_order.side_effect = _submit
        target = {"MSFT": 0.50}  # sell AAPL (will fail), keep MSFT
        mgr = OrderManager(db_path=str(tmp_path / "test.db"))
        results = mgr.rebalance(target, positions, equity=100_000.0, adapter=adapter)

        error_results = [r for r in results if r.get("status") == "error"]
        assert len(error_results) == 1
        assert error_results[0]["symbol"] == "AAPL"
        assert "AAPL sell rejected" in error_results[0]["error"]

    def test_buy_failure_does_not_prevent_prior_sells(self, tmp_path):
        """A failed BUY does not undo the already-submitted SELLs."""
        positions = {
            "AAPL": {"market_value": 50_000.0, "current_price": 500.0, "qty": 100},
        }
        adapter = _make_adapter(positions=positions)
        adapter.get_latest_price.return_value = 200.0  # price for NVDA

        def _submit(symbol, qty, side, **_kw):
            if symbol == "NVDA" and side == "buy":
                raise ExecutionError("NVDA buy rejected")
            return {
                "order_id": f"ord-{symbol}",
                "client_order_id": "",
                "symbol": symbol,
                "side": side,
                "status": "accepted",
                "filled_qty": 0.0,
                "filled_avg_price": 0.0,
                "order_type": "market",
            }

        adapter.submit_order.side_effect = _submit
        target = {"NVDA": 0.30}  # sell AAPL, buy NVDA (NVDA buy will fail)
        mgr = OrderManager(db_path=str(tmp_path / "test.db"))
        results = mgr.rebalance(target, positions, equity=100_000.0, adapter=adapter)

        sides = [r["side"] for r in results]
        assert "sell" in sides  # AAPL sell completed
        error_results = [r for r in results if r.get("status") == "error"]
        assert any(r["symbol"] == "NVDA" for r in error_results)


class TestOrderManagerRebalanceRiskPipeline:
    def test_risk_manager_validate_trade_called(self, tmp_path):
        """rebalance() must call risk_manager.validate_trade() when provided."""
        risk_manager = MagicMock()
        risk_manager.validate_trade.return_value = {"AAPL": 0.50}

        adapter = _make_adapter(positions={})
        adapter.get_latest_price.return_value = 200.0

        mgr = OrderManager(db_path=str(tmp_path / "test.db"))
        mgr.rebalance(
            {"AAPL": 0.50},
            {},
            equity=10_000.0,
            adapter=adapter,
            risk_manager=risk_manager,
        )
        risk_manager.validate_trade.assert_called_once()

    def test_risk_manager_rejection_propagates(self, tmp_path):
        """If validate_trade raises, rebalance propagates the exception."""
        from core.risk.kill_switch import TradingHaltedError

        risk_manager = MagicMock()
        risk_manager.validate_trade.side_effect = TradingHaltedError(MagicMock())

        adapter = _make_adapter()
        mgr = OrderManager(db_path=str(tmp_path / "test.db"))
        with pytest.raises(TradingHaltedError):
            mgr.rebalance(
                {"AAPL": 1.0}, {}, equity=10_000.0, adapter=adapter, risk_manager=risk_manager
            )
        adapter.submit_order.assert_not_called()


# ===========================================================================
# Reconciler.check — drift report
# ===========================================================================


class TestReconcilerCheck:
    def test_exact_match_is_ok(self):
        reconciler = Reconciler(db_path=":memory:")
        expected = {"AAPL": 0.10, "MSFT": 0.10}
        actual = {
            "AAPL": {"market_value": 10_000.0},
            "MSFT": {"market_value": 10_000.0},
        }
        report = reconciler.check(expected, actual, equity=100_000.0)
        assert report["AAPL"]["status"] == "ok"
        assert report["MSFT"]["status"] == "ok"
        assert report["AAPL"]["drift_pct"] == pytest.approx(0.0)

    def test_drift_above_threshold_is_warning(self):
        reconciler = Reconciler(db_path=":memory:")
        # Expected 10 %, actual 15 % → +5 pp drift
        expected = {"AAPL": 0.10}
        actual = {"AAPL": {"market_value": 15_000.0}}
        report = reconciler.check(expected, actual, equity=100_000.0)
        assert report["AAPL"]["status"] == "warning"
        assert report["AAPL"]["drift_pct"] == pytest.approx(5.0)

    def test_drift_below_threshold_is_ok(self):
        reconciler = Reconciler(db_path=":memory:")
        # Expected 10 %, actual 11 % → +1 pp — below 2 pp threshold
        expected = {"AAPL": 0.10}
        actual = {"AAPL": {"market_value": 11_000.0}}
        report = reconciler.check(expected, actual, equity=100_000.0)
        assert report["AAPL"]["status"] == "ok"
        assert report["AAPL"]["drift_pct"] == pytest.approx(1.0)

    def test_missing_symbol_flagged(self):
        """Symbol in expected but absent from actual positions → 'missing'."""
        reconciler = Reconciler(db_path=":memory:")
        expected = {"AAPL": 0.20}
        actual = {}  # no positions
        report = reconciler.check(expected, actual, equity=100_000.0)
        assert report["AAPL"]["status"] == "missing"
        assert report["AAPL"]["actual_weight"] == pytest.approx(0.0)
        assert report["AAPL"]["drift_pct"] == pytest.approx(-20.0)

    def test_unexpected_position_flagged(self):
        """Symbol in actual but not in expected → 'unexpected'."""
        reconciler = Reconciler(db_path=":memory:")
        expected = {}
        actual = {"AAPL": {"market_value": 5_000.0}}
        report = reconciler.check(expected, actual, equity=100_000.0)
        assert report["AAPL"]["status"] == "unexpected"
        assert report["AAPL"]["expected_weight"] == pytest.approx(0.0)
        assert report["AAPL"]["drift_pct"] == pytest.approx(5.0)

    def test_negative_drift_under_weight(self):
        """actual < expected → negative drift_pct."""
        reconciler = Reconciler(db_path=":memory:")
        # Expected 30 %, actual 20 % → −10 pp
        expected = {"AAPL": 0.30}
        actual = {"AAPL": {"market_value": 20_000.0}}
        report = reconciler.check(expected, actual, equity=100_000.0)
        assert report["AAPL"]["drift_pct"] == pytest.approx(-10.0)

    def test_multiple_symbols_mixed_statuses(self):
        reconciler = Reconciler(db_path=":memory:")
        expected = {"AAPL": 0.10, "MSFT": 0.20, "NVDA": 0.05}
        actual = {
            "AAPL": {"market_value": 10_000.0},  # exact → ok
            "MSFT": {"market_value": 24_000.0},  # +4 pp → warning
            # NVDA missing from actual
            "GOOG": {"market_value": 3_000.0},  # not in expected → unexpected
        }
        report = reconciler.check(expected, actual, equity=100_000.0)
        assert report["AAPL"]["status"] == "ok"
        assert report["MSFT"]["status"] == "warning"
        assert report["NVDA"]["status"] == "missing"
        assert report["GOOG"]["status"] == "unexpected"

    def test_zero_equity_returns_empty(self):
        reconciler = Reconciler(db_path=":memory:")
        report = reconciler.check({"AAPL": 0.5}, {"AAPL": {"market_value": 5000}}, equity=0.0)
        assert report == {}

    def test_report_keys_present(self):
        reconciler = Reconciler(db_path=":memory:")
        expected = {"AAPL": 0.10}
        actual = {"AAPL": {"market_value": 10_000.0}}
        report = reconciler.check(expected, actual, equity=100_000.0)
        assert set(report["AAPL"].keys()) >= {
            "expected_weight",
            "actual_weight",
            "drift_pct",
            "status",
        }


# ===========================================================================
# Reconciler.reconcile — legacy backward-compat
# ===========================================================================


class TestReconcilerLegacy:
    def test_reconcile_returns_list(self, tmp_path):
        reconciler = Reconciler(db_path=str(tmp_path / "test.db"))
        result = reconciler.reconcile([{"symbol": "AAPL", "qty": 100, "avg_cost": 180.0}])
        assert isinstance(result, list)

    def test_reconcile_discrepancy_returns_list(self, tmp_path):
        reconciler = Reconciler(db_path=str(tmp_path / "test.db"))
        result = reconciler.reconcile([{"symbol": "2330.TW", "qty": 2000, "avg_cost": 580.0}])
        assert isinstance(result, list)


# ===========================================================================
# Legacy adapter interface tests (backward-compat)
# ===========================================================================


class TestShioajiAdapterInterface:
    def test_has_required_methods(self):
        for method in ("connect", "place_order", "cancel_order", "get_positions", "get_account"):
            assert hasattr(ShioajiAdapter, method), f"Missing method: {method}"

    def test_lot_size_is_1000(self):
        assert ShioajiAdapter.LOT_SIZE == 1000

"""Tests for orchestrator run_daily: buy, close, hold, add paths."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.scheduler.orchestrator import V2bOrchestrator
from src.state.state_manager import StateManager, TradingState
from src.strategy.v2b_engine import Signal, V2bEngine


def _make_data(n=200):
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-01", periods=n)
    close = 20000.0 + np.linspace(0, 3000, n) + np.random.randn(n) * 20
    return pd.DataFrame({
        "open": close - 10, "high": close + 50,
        "low": close - 50, "close": close,
        "volume": [100000] * n,
    }, index=dates)


def _make_orch(state, signal_override=None):
    """Build orchestrator with mocked state_mgr and strategy."""
    strategy = V2bEngine(product="MXF", ema_fast=30, ema_slow=100,
                         confirm_days=2, adx_threshold=25)
    if signal_override:
        strategy.generate_signal = lambda *a, **kw: signal_override

    state_mgr = MagicMock(spec=StateManager)
    state_mgr.load.return_value = state

    orch = V2bOrchestrator(
        strategy=strategy,
        state_mgr=state_mgr,
        notify_fn=MagicMock(),
        execution_timing="next_open",
        live=False,
    )
    return orch, state_mgr


class TestRunDailyBuy:
    def test_buy_calls_broker(self):
        """Empty position + buy signal → broker.place_order('MXF', 'Buy', N)."""
        state = TradingState(position=0, equity=500_000)
        sig = Signal("buy", 2, "golden cross")
        orch, _ = _make_orch(state, sig)
        broker = MagicMock()
        broker.place_order.return_value = {"order_id": "1", "fill_price": 21000.0}
        broker.get_positions.return_value = [{"contracts": 2}]

        df = _make_data()
        with patch.object(orch, "_load_data", return_value=df):
            result = orch.run_daily(broker=broker)

        assert result["action"] == "buy"
        broker.place_order.assert_called_once_with("MXF", "Buy", 2)


class TestRunDailyClose:
    def test_close_calls_broker(self):
        """Holding + close signal → broker.place_order('MXF', 'Sell', N)."""
        state = TradingState(position=2, entry_price=20000.0, contracts=2, equity=500_000)
        sig = Signal("close", 2, "death cross")
        orch, _ = _make_orch(state, sig)
        broker = MagicMock()
        broker.place_order.return_value = {"order_id": "2", "fill_price": 21000.0}
        broker.get_positions.return_value = []

        df = _make_data()
        with patch.object(orch, "_load_data", return_value=df):
            result = orch.run_daily(broker=broker)

        assert result["action"] == "close"
        broker.place_order.assert_called_once_with("MXF", "Sell", 2)
        assert "pnl_twd" in result


class TestRunDailyHold:
    def test_hold_no_broker_call(self):
        """Holding + hold signal → broker NOT called."""
        state = TradingState(position=2, entry_price=20000.0, contracts=2,
                             equity=500_000, highest_high=21000.0)
        sig = Signal("hold", 2, "holding position")
        orch, _ = _make_orch(state, sig)
        broker = MagicMock()

        df = _make_data()
        with patch.object(orch, "_load_data", return_value=df):
            result = orch.run_daily(broker=broker)

        assert result["action"] == "hold"
        broker.place_order.assert_not_called()


class TestRunDailyAdd:
    def test_add_calls_broker(self):
        """Holding + add signal → broker.place_order('MXF', 'Buy', add_n)."""
        state = TradingState(position=2, entry_price=20000.0, contracts=2, equity=500_000)
        sig = Signal("add", 1, "pyramid: float_profit=500, add=1")
        orch, _ = _make_orch(state, sig)
        broker = MagicMock()
        broker.place_order.return_value = {"order_id": "3", "fill_price": 21000.0}
        broker.get_positions.return_value = [{"contracts": 3}]

        df = _make_data()
        with patch.object(orch, "_load_data", return_value=df):
            result = orch.run_daily(broker=broker)

        assert result["action"] == "add"
        broker.place_order.assert_called_once_with("MXF", "Buy", 1)


class TestEquityAutoUpdate:
    """Successful live-equity reads must write through to state on disk;
    failures must leave the cached value alone."""

    def _build_night_orch(self, state, signal_override):
        strategy = V2bEngine(
            product="MXF", ema_fast=30, ema_slow=100,
            confirm_days=2, adx_threshold=25,
        )
        strategy.generate_signal = lambda *a, **kw: signal_override
        state_mgr = MagicMock(spec=StateManager)
        state_mgr.load.return_value = state
        orch = V2bOrchestrator(
            strategy=strategy, state_mgr=state_mgr,
            notify_fn=MagicMock(),
            execution_timing="night_open", live=False,
        )
        return orch, state_mgr

    def test_equity_auto_update_on_signal(self):
        """Broker returns live equity > 0 → state.equity updated AND
        state_mgr.save called with the new value."""
        state = TradingState(position=0, equity=500_000.0)
        sig = Signal("hold", 0, "no entry")
        orch, state_mgr = self._build_night_orch(state, sig)

        broker = MagicMock()
        broker.get_account.return_value = {"equity": 612_345.0}

        df = _make_data()
        with patch.object(orch, "_load_data", return_value=df):
            orch.run_signal(broker=broker)

        broker.get_account.assert_called()
        assert state.equity == 612_345.0, (
            "live equity must be persisted into state.equity"
        )
        saved_equities = [
            call.args[0].equity for call in state_mgr.save.call_args_list
        ]
        assert 612_345.0 in saved_equities

    def test_equity_auto_update_fallback(self):
        """Broker.get_account raises → state.equity is NOT overwritten
        (the previous cached value is preserved)."""
        state = TradingState(position=0, equity=480_000.0)
        sig = Signal("hold", 0, "no entry")
        orch, state_mgr = self._build_night_orch(state, sig)

        broker = MagicMock()
        broker.get_account.side_effect = ConnectionError("api down")

        df = _make_data()
        with patch.object(orch, "_load_data", return_value=df):
            orch.run_signal(broker=broker)

        broker.get_account.assert_called()
        assert state.equity == 480_000.0, (
            "broker failure must not overwrite cached equity"
        )

    def test_equity_auto_update_on_execution(self):
        """run_execution also persists live equity (covers the
        15:05 entry + post-fill verify combined path)."""
        state = TradingState(
            position=0, equity=500_000.0,
            pending_action=None, pending_contracts=0,
        )
        sig = Signal("hold", 0, "no entry")
        orch, state_mgr = self._build_night_orch(state, sig)

        broker = MagicMock()
        broker.get_account.return_value = {"equity": 730_000.0}

        orch.run_execution(broker=broker, exec_price=21000.0)

        broker.get_account.assert_called()
        assert state.equity == 730_000.0
        saved_equities = [
            call.args[0].equity for call in state_mgr.save.call_args_list
        ]
        assert 730_000.0 in saved_equities

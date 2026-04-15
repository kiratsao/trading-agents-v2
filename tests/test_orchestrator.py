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

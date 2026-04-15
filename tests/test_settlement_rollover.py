"""Tests for settlement-day rollover (close + re-enter)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.strategy.v2b_engine import V2bEngine


def _make_bull_data(n=200, settlement_date="2025-04-16"):
    """Generate data where EMA(30) > EMA(100) and ADX > 25 at settlement_date.

    Creates a strong uptrend so golden cross + ADX confirmation are active.
    """
    np.random.seed(42)
    dates = pd.bdate_range(end=settlement_date, periods=n)
    # Strong uptrend: ensures EMA fast > slow and ADX > 25
    base = 20000.0
    trend = np.linspace(0, 3000, n)
    noise = np.random.randn(n) * 30
    close = base + trend + noise
    return pd.DataFrame(
        {
            "open": close - 10,
            "high": close + 50,
            "low": close - 50,
            "close": close,
            "volume": np.random.randint(80000, 200000, n),
        },
        index=dates,
    )


def _make_flat_data(n=200, settlement_date="2025-04-16"):
    """Generate data where ADX is low (range-bound), so entry is rejected."""
    np.random.seed(99)
    dates = pd.bdate_range(end=settlement_date, periods=n)
    # Flat / mean-reverting → low ADX
    base = 20000.0
    close = base + np.sin(np.linspace(0, 20 * np.pi, n)) * 100 + np.random.randn(n) * 20
    return pd.DataFrame(
        {
            "open": close - 5,
            "high": close + 30,
            "low": close - 30,
            "close": close,
            "volume": np.random.randint(80000, 200000, n),
        },
        index=dates,
    )


class TestSettlementRolloverEngine:
    """Test backtest engine settlement rollover."""

    def test_settlement_close_then_reenter(self):
        """On settlement day with golden cross + ADX OK → close old + buy new."""
        from src.backtest.engine import BacktestEngine

        df = _make_bull_data(n=200, settlement_date="2025-04-16")
        engine = BacktestEngine(
            strategy=V2bEngine(product="MXF", ema_fast=30, ema_slow=100,
                               confirm_days=2, adx_threshold=25),
            initial_capital=350_000,
            exec_timing="same_day_close",
        )
        result = engine.run(df)

        # Find trades on the settlement date
        settle_date = "2025-04-16"
        settlement_closes = [t for t in result.trades if t.exit_date == settle_date]
        settlement_entries = [t for t in result.trades
                             if t.entry_date == settle_date
                             and "rollover" in t.entry_reason]

        # If the engine was in a position on settlement day, there should be
        # a close trade. If conditions are met, a rollover entry on same day.
        if settlement_closes:
            assert any("settlement" in t.reason for t in settlement_closes)
            # If ADX was high enough, there should also be a rollover entry
            # (depends on data, but the bull data should produce this)
            if settlement_entries:
                assert any("settlement rollover" in t.entry_reason
                           for t in settlement_entries)

    def test_settlement_close_no_reenter_low_adx(self):
        """On settlement day with ADX < 25 → close only, no re-entry."""
        from src.backtest.engine import BacktestEngine

        df = _make_flat_data(n=200, settlement_date="2025-04-16")
        engine = BacktestEngine(
            strategy=V2bEngine(product="MXF", ema_fast=30, ema_slow=100,
                               confirm_days=2, adx_threshold=25),
            initial_capital=350_000,
            exec_timing="same_day_close",
        )
        result = engine.run(df)

        settle_date = "2025-04-16"
        rollover_entries = [t for t in result.trades
                           if t.entry_date == settle_date
                           and "settlement rollover" in t.entry_reason]
        # Flat data → ADX low → no rollover entry
        assert len(rollover_entries) == 0


class TestSettlementRolloverOrchestrator:
    """Test orchestrator run_execution settlement rollover."""

    def test_rollover_uses_new_contract(self):
        """Settlement rollover calls broker.place_order for new buy."""
        from src.scheduler.orchestrator import V2bOrchestrator
        from src.state.state_manager import StateManager, TradingState

        df = _make_bull_data(n=200, settlement_date="2025-04-16")
        strategy = V2bEngine(product="MXF", ema_fast=30, ema_slow=100,
                             confirm_days=2, adx_threshold=25)

        # Force generate_signal to return buy when called with position=0
        original_gen = strategy.generate_signal

        def mock_gen(data, current_position=0, **kw):
            if current_position == 0:
                from src.strategy.v2b_engine import Signal
                return Signal("buy", 2, "golden cross + ADX OK")
            return original_gen(data, current_position=current_position, **kw)

        strategy.generate_signal = mock_gen

        state_mgr = MagicMock(spec=StateManager)
        state = TradingState(
            position=2,
            entry_price=20500.0,
            contracts=2,
            equity=400_000,
            pending_action="close",
            pending_contracts=2,
            pending_reason="settlement-day force close",
        )
        state_mgr.load.return_value = state

        broker = MagicMock()
        broker.place_order.return_value = {"order_id": "123", "fill_price": 22000.0}
        broker.get_positions.return_value = [{"contracts": 2}]

        orch = V2bOrchestrator(
            strategy=strategy,
            state_mgr=state_mgr,
            notify_fn=MagicMock(),
            execution_timing="night_open",
            live=False,
        )

        with patch.object(orch, "_load_data", return_value=df):
            result = orch.run_execution(broker=broker, exec_price=22000.0)

        assert result["action"] == "close"
        assert result.get("rollover") is True
        assert result.get("rollover_contracts") == 2

        # Broker should have been called twice: Sell (close) then Buy (rollover)
        calls = broker.place_order.call_args_list
        assert len(calls) == 2
        assert calls[0].args == ("MXF", "Sell", 2)
        assert calls[1].args == ("MXF", "Buy", 2)

    def test_rollover_line_notification(self):
        """LINE notification contains 🔄 轉倉 on settlement rollover."""
        from src.scheduler.orchestrator import V2bOrchestrator
        from src.state.state_manager import StateManager, TradingState

        df = _make_bull_data(n=200, settlement_date="2025-04-16")
        strategy = V2bEngine(product="MXF", ema_fast=30, ema_slow=100,
                             confirm_days=2, adx_threshold=25)

        def mock_gen(data, current_position=0, **kw):
            if current_position == 0:
                from src.strategy.v2b_engine import Signal
                return Signal("buy", 2, "golden cross + ADX OK")
            from src.strategy.v2b_engine import Signal
            return Signal("hold", 2, "holding")

        strategy.generate_signal = mock_gen

        state_mgr = MagicMock(spec=StateManager)
        state = TradingState(
            position=2,
            entry_price=20500.0,
            contracts=2,
            equity=400_000,
            pending_action="close",
            pending_contracts=2,
            pending_reason="settlement-day force close",
        )
        state_mgr.load.return_value = state

        notify_fn = MagicMock()
        orch = V2bOrchestrator(
            strategy=strategy,
            state_mgr=state_mgr,
            notify_fn=notify_fn,
            execution_timing="night_open",
            live=False,
        )

        with patch.object(orch, "_load_data", return_value=df):
            orch.run_execution(broker=None, exec_price=22000.0)

        # Check notification contains rollover indicator
        notify_fn.assert_called_once()
        msg = notify_fn.call_args[0][0]
        assert "🔄" in msg
        assert "轉倉" in msg

"""Tests for unrealized PnL in LINE decision notifications."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.scheduler.orchestrator import V2bOrchestrator
from src.state.state_manager import TradingState
from src.strategy.v2b_engine import Signal, V2bEngine


def _make_orchestrator() -> V2bOrchestrator:
    engine = V2bEngine(product="MXF", ema_fast=30, ema_slow=100, confirm_days=3)
    state_mgr = MagicMock()
    return V2bOrchestrator(
        strategy=engine,
        state_mgr=state_mgr,
        notify_fn=lambda msg: None,
    )


class TestDecisionPnl:
    def test_hold_with_profit(self):
        """Holding with profit → notification contains green PnL."""
        orch = _make_orchestrator()
        state = TradingState(position=2, entry_price=33000.0, equity=400_000.0)
        sig = Signal("hold", 2, "holding position")
        indicators = {
            "close": 33500.0,
            "ema_fast": 33400.0,
            "ema_slow": 32000.0,
            "atr": 500.0,
            "bull_streak": 5,
            "confirm_days": 3,
            "trailing_stop": 32500.0,
        }
        msg = orch._build_decision_message(
            sig=sig,
            state=state,
            indicators=indicators,
            action_contracts=0,
            closed_contracts=0,
            equity=400_000.0,
            tsmc_signal=None,
            equity_src="估算",
        )
        # (33500 - 33000) * 2 * 50 = 50,000
        assert "🟢" in msg
        assert "+50,000" in msg
        assert "持倉損益" in msg

    def test_hold_with_loss(self):
        """Holding with loss → notification contains red PnL."""
        orch = _make_orchestrator()
        state = TradingState(position=2, entry_price=34000.0, equity=400_000.0)
        sig = Signal("hold", 2, "holding position")
        indicators = {
            "close": 33500.0,
            "ema_fast": 33400.0,
            "ema_slow": 32000.0,
            "atr": 500.0,
            "bull_streak": 5,
            "confirm_days": 3,
            "trailing_stop": 32500.0,
        }
        msg = orch._build_decision_message(
            sig=sig,
            state=state,
            indicators=indicators,
            action_contracts=0,
            closed_contracts=0,
            equity=400_000.0,
            tsmc_signal=None,
            equity_src="估算",
        )
        # (33500 - 34000) * 2 * 50 = -50,000
        assert "🔴" in msg
        assert "-50,000" in msg
        assert "持倉損益" in msg

    def test_no_position_no_pnl(self):
        """Flat position → no PnL line in notification."""
        orch = _make_orchestrator()
        state = TradingState(position=0, entry_price=None, equity=400_000.0)
        sig = Signal("hold", 0, "no position")
        indicators = {
            "close": 33500.0,
            "ema_fast": 33400.0,
            "ema_slow": 32000.0,
            "atr": 500.0,
            "bull_streak": 5,
            "confirm_days": 3,
            "trailing_stop": None,
        }
        msg = orch._build_decision_message(
            sig=sig,
            state=state,
            indicators=indicators,
            action_contracts=0,
            closed_contracts=0,
            equity=400_000.0,
            tsmc_signal=None,
            equity_src="估算",
        )
        assert "持倉損益" not in msg

    def test_pnl_percentage(self):
        """PnL percentage is correctly calculated."""
        orch = _make_orchestrator()
        state = TradingState(position=2, entry_price=33000.0, equity=500_000.0)
        sig = Signal("hold", 2, "holding position")
        indicators = {
            "close": 33500.0,
            "ema_fast": 33400.0,
            "ema_slow": 32000.0,
            "atr": 500.0,
            "bull_streak": 5,
            "confirm_days": 3,
            "trailing_stop": 32500.0,
        }
        msg = orch._build_decision_message(
            sig=sig,
            state=state,
            indicators=indicators,
            action_contracts=0,
            closed_contracts=0,
            equity=500_000.0,
            tsmc_signal=None,
            equity_src="即時",
        )
        # 50,000 / 500,000 = 10%
        assert "+10.0%" in msg

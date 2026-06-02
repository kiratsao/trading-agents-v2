"""Tests for orchestrator run_daily: buy, close, hold, add paths."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.scheduler.orchestrator import V2bOrchestrator, _sync_position_from_broker
from src.state.state_manager import StateManager, TradingState
from src.strategy.v2b_engine import Signal, V2bEngine

_NOSLEEP = lambda *_a, **_k: None  # noqa: E731 — injected to skip the 2s/3s waits


class TestSyncPositionFromBroker:
    """Unit coverage for the broker-truth position sync (2026-06-01 fix)."""

    def test_returns_broker_actual_on_match(self):
        broker = MagicMock()
        broker.get_positions.return_value = [
            {"code": "MXFF6", "direction": "Buy", "contracts": 5},
        ]
        notify = MagicMock()
        assert _sync_position_from_broker(
            broker, "MXF", 5, notify, sleep=_NOSLEEP,
        ) == 5
        notify.assert_not_called()  # exact match → no alert

    def test_ioc_partial_returns_broker_truth_not_request(self):
        """Requested total 45 but broker only holds 20 (IOC under-filled) →
        return 20 (truth) + alert. This is the 2026-06-01 scenario."""
        broker = MagicMock()
        broker.get_positions.return_value = [
            {"code": "MXFF6", "direction": "Buy", "contracts": 20},
        ]
        notify = MagicMock()
        actual = _sync_position_from_broker(broker, "MXF", 45, notify, sleep=_NOSLEEP)
        assert actual == 20
        assert any("broker 為準" in c.args[0] for c in notify.call_args_list)

    def test_retry_takes_later_read_when_fill_report_lags(self):
        """First read is below expected (fill report lag), second read catches
        up → take the later (higher) read."""
        broker = MagicMock()
        broker.get_positions.side_effect = [
            [{"code": "MXFF6", "direction": "Buy", "contracts": 0}],   # lagging
            [{"code": "MXFF6", "direction": "Buy", "contracts": 3}],   # settled
        ]
        notify = MagicMock()
        assert _sync_position_from_broker(
            broker, "MXF", 3, notify, sleep=_NOSLEEP,
        ) == 3
        assert broker.get_positions.call_count == 2

    def test_broker_unreadable_returns_none_and_alerts(self):
        broker = MagicMock()
        broker.get_positions.side_effect = ConnectionError("api down")
        notify = MagicMock()
        assert _sync_position_from_broker(
            broker, "MXF", 5, notify, sleep=_NOSLEEP,
        ) is None
        assert any("部位同步失敗" in c.args[0] for c in notify.call_args_list)

    def test_excludes_non_mxf_and_shorts(self):
        broker = MagicMock()
        broker.get_positions.return_value = [
            {"code": "MXFF6", "direction": "Buy", "contracts": 7},
            {"code": "TXFF6", "direction": "Buy", "contracts": 3},   # other product
            {"code": "MXFF6", "direction": "Sell", "contracts": 2},  # short leg
        ]
        notify = MagicMock()
        assert _sync_position_from_broker(
            broker, "MXF", 7, notify, sleep=_NOSLEEP,
        ) == 7


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


class TestAddEntryPriceReconcile:
    """Anti-Martingale add must refresh state.entry_price -- broker
    weighted avg preferred, local weighted avg fallback. Without this
    fix the LINE PnL kept using the pre-add average and overstated
    float profit by (broker_avg - old_avg) × new_position."""

    def test_add_uses_broker_avg_price(self):
        """Holding 13口 @ 40,532; add 2口 fills at 42,415; broker reports
        the blended avg at 40,783 → state.entry_price = 40,783 (not the
        old 40,532, not the new fill 42,415)."""
        state = TradingState(
            position=13, entry_price=40_532.0, contracts=13,
            equity=2_000_000.0,
        )
        sig = Signal("add", 2, "pyramid")
        orch, _ = _make_orch(state, sig)
        broker = MagicMock()
        broker.place_order.return_value = {
            "order_id": "ADD-1", "fill_price": 42_415.0,
        }
        # Broker reports the post-fill blended average.
        broker.get_positions.return_value = [
            {"code": "MXFE5", "direction": "Buy", "contracts": 15,
             "avg_price": 40_783.0},
        ]

        df = _make_data()
        with patch.object(orch, "_load_data", return_value=df):
            result = orch.run_daily(broker=broker)

        assert result["action"] == "add"
        assert state.position == 15
        assert state.entry_price == 40_783.0, (
            "state.entry_price must follow the broker's blended average"
        )
        assert result.get("entry_price_source") == "broker"

    def test_add_local_weighted_avg_when_broker_avg_unreadable(self):
        """Position fill IS confirmed by the broker (15 lots) but the broker's
        avg_price is unreadable (0.0) → entry falls back to the local weighted
        avg: (13 × 40,532 + 2 × 42,415) / 15 = 40,783.0.

        Post-fix: the position count always follows broker truth; only the
        cost-basis falls back. (Was: a get_positions *exception* still bumped
        to 15 — that optimistic bump is exactly the 2026-06-01 bug and is now
        forbidden; see test_add_aborts_when_broker_unreadable.)"""
        state = TradingState(
            position=13, entry_price=40_532.0, contracts=13,
            equity=2_000_000.0,
        )
        sig = Signal("add", 2, "pyramid")
        orch, _ = _make_orch(state, sig)
        broker = MagicMock()
        broker.place_order.return_value = {
            "order_id": "ADD-2", "fill_price": 42_415.0,
        }
        # Broker confirms the 15-lot fill but reports no usable avg_price.
        broker.get_positions.return_value = [
            {"code": "MXFE5", "direction": "Buy", "contracts": 15,
             "avg_price": 0.0},
        ]

        df = _make_data()
        with patch("time.sleep"), patch.object(orch, "_load_data", return_value=df):
            result = orch.run_daily(broker=broker)

        assert state.position == 15
        expected = (13 * 40_532.0 + 2 * 42_415.0) / 15
        assert abs(state.entry_price - expected) < 0.01, (
            f"local weighted avg expected {expected:.2f}, got {state.entry_price}"
        )
        assert result.get("entry_price_source") == "local"

    def test_add_not_filled_keeps_position_no_runaway(self):
        """2026-06-01 REGRESSION: broker still shows the pre-add 13 lots after
        the add order (IOC did not fill). The local state MUST stay at 13 — the
        old code blindly did state.position += add_n → 15 while the broker held
        13, the exact runaway that pushed live state to 45 vs broker 20."""
        state = TradingState(
            position=13, entry_price=40_532.0, contracts=13,
            equity=2_000_000.0,
        )
        sig = Signal("add", 2, "pyramid")
        orch, _ = _make_orch(state, sig)
        broker = MagicMock()
        broker.place_order.return_value = {
            "order_id": "ADD-3", "fill_price": 42_415.0,
        }
        # Broker shows the add did NOT fill — still 13 lots.
        broker.get_positions.return_value = [
            {"code": "MXFE5", "direction": "Buy", "contracts": 13,
             "avg_price": 40_532.0},
        ]

        df = _make_data()
        with patch("time.sleep"), patch.object(orch, "_load_data", return_value=df):
            result = orch.run_daily(broker=broker)

        assert state.position == 13, "must NOT bump when broker confirms no fill"
        assert state.entry_price == 40_532.0, "cost basis unchanged on no-fill"
        assert result.get("add_contracts") == 0
        alerts = [c.args[0] for c in orch.notify_fn.call_args_list]
        assert any("加碼未成交" in m for m in alerts), alerts

    def test_add_aborts_when_broker_unreadable(self):
        """Broker.get_positions raises on both reads → cannot confirm the fill →
        keep local state unchanged + loud alert (never optimistically bump)."""
        state = TradingState(
            position=13, entry_price=40_532.0, contracts=13,
            equity=2_000_000.0,
        )
        sig = Signal("add", 2, "pyramid")
        orch, _ = _make_orch(state, sig)
        broker = MagicMock()
        broker.place_order.return_value = {
            "order_id": "ADD-4", "fill_price": 42_415.0,
        }
        broker.get_positions.side_effect = ConnectionError("api down")

        df = _make_data()
        with patch("time.sleep"), patch.object(orch, "_load_data", return_value=df):
            result = orch.run_daily(broker=broker)

        assert state.position == 13, "broker unreadable → state must not change"
        assert result.get("add_contracts") == 0
        alerts = [c.args[0] for c in orch.notify_fn.call_args_list]
        assert any("部位同步失敗" in m for m in alerts), alerts

    def test_add_via_run_execution_uses_broker_avg(self):
        """run_execution (night_open phase 2) must apply the same
        reconcile -- the bug was reported on a live night-session add."""
        state = TradingState(
            position=13, entry_price=40_532.0, contracts=13,
            equity=2_000_000.0,
            pending_action="add", pending_contracts=2,
        )
        # Strategy.generate_signal won't be called from run_execution,
        # so the signal_override is irrelevant; pass a placeholder.
        orch, _ = _make_orch(state, Signal("hold", 0, ""))
        broker = MagicMock()
        broker.place_order.return_value = {
            "order_id": "ADD-N", "fill_price": 42_415.0,
        }
        broker.get_positions.return_value = [
            {"code": "MXFE5", "direction": "Buy", "contracts": 15,
             "avg_price": 40_783.0},
        ]

        orch.run_execution(broker=broker, exec_price=42_415.0)

        assert state.position == 15
        assert state.entry_price == 40_783.0

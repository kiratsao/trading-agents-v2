"""Tests for the 2026-07 robustness batch.

Covers: notify dedup, stale-pending guard, broker auto-reconcile, pre-sell
naked-short guard, holiday-adjusted settlement day, state corruption
restore, and the trading-day job guard.
"""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.notify.line import DedupNotifier
from src.scheduler.orchestrator import V2bOrchestrator
from src.state.state_manager import StateCorruptionError, StateManager, TradingState
from src.strategy.v2b_engine import V2bEngine, _is_settlement_day
from src.utils.tw_time import today_taipei

_NOSLEEP = lambda *_a, **_k: None  # noqa: E731


def _make_data(n=200):
    dates = pd.bdate_range("2025-06-01", periods=n)
    base = np.linspace(20000, 22000, n)
    return pd.DataFrame(
        {"open": base, "high": base + 100, "low": base - 100,
         "close": base, "volume": [100_000] * n},
        index=dates,
    )


def _make_orch(state, sig=None, notify=None):
    strategy = V2bEngine(product="MXF", ema_fast=30, ema_slow=100,
                         confirm_days=2, adx_threshold=25)
    if sig is not None:
        strategy.generate_signal = lambda *a, **kw: sig
    state_mgr = MagicMock(spec=StateManager)
    state_mgr.load.return_value = state
    orch = V2bOrchestrator(
        strategy=strategy,
        state_mgr=state_mgr,
        notify_fn=notify or MagicMock(),
        execution_timing="night_open",
        live=False,
    )
    return orch, state_mgr


# ─────────────────────────────────────────────────────────────────────────
# DedupNotifier
# ─────────────────────────────────────────────────────────────────────────
class TestDedupNotifier:
    def test_alert_deduped_within_ttl(self, tmp_path):
        sent = []
        n = DedupNotifier(sent.append, journal_path=tmp_path / "j.json")
        n("⚠️ 2026-07-01 日K缺失，自動補回失敗 — 需手動處理")
        n("⚠️ 2026-07-01 日K缺失，自動補回失敗 — 需手動處理")
        assert len(sent) == 1, "identical alert within TTL must be suppressed"

    def test_cross_instance_dedup_via_shared_journal(self, tmp_path):
        """daemon(14:25) 與 systemd timer(14:30) 是兩個 process — 共用
        journal 檔必須讓第二個 process 的相同告警被抑制。"""
        journal = tmp_path / "j.json"
        sent_a, sent_b = [], []
        DedupNotifier(sent_a.append, journal_path=journal)("🔴 資料更新失敗: X")
        DedupNotifier(sent_b.append, journal_path=journal)("🔴 資料更新失敗: X")
        assert len(sent_a) == 1 and len(sent_b) == 0

    def test_non_alert_messages_never_deduped(self, tmp_path):
        sent = []
        n = DedupNotifier(sent.append, journal_path=tmp_path / "j.json")
        n("✅ 資料更新: +1 bars")
        n("✅ 資料更新: +1 bars")
        n("⚙️ state 已自動同步 broker: 2口 → 空倉")
        n("⚙️ state 已自動同步 broker: 2口 → 空倉")
        assert len(sent) == 4

    def test_different_alerts_not_suppressed(self, tmp_path):
        sent = []
        n = DedupNotifier(sent.append, journal_path=tmp_path / "j.json")
        n("⚠️ 2026-07-01 日K缺失，自動補回失敗 — 需手動處理")
        n("⚠️ 2026-07-02 日K缺失，自動補回失敗 — 需手動處理")
        assert len(sent) == 2

    def test_corrupt_journal_never_blocks_send(self, tmp_path):
        journal = tmp_path / "j.json"
        journal.write_text("{not json")
        sent = []
        DedupNotifier(sent.append, journal_path=journal)("🔴 alert")
        assert sent == ["🔴 alert"]


# ─────────────────────────────────────────────────────────────────────────
# Stale-pending guard (run_execution)
# ─────────────────────────────────────────────────────────────────────────
class TestStalePendingGuard:
    def test_yesterday_pending_discarded(self):
        state = TradingState(
            position=0, equity=500_000,
            pending_action="buy", pending_contracts=2,
            pending_signal_date="2026-01-05",  # not today
        )
        notify = MagicMock()
        orch, state_mgr = _make_orch(state, notify=notify)
        broker = MagicMock()

        result = orch.run_execution(broker=broker, exec_price=21000.0)

        assert result["action"] == "stale_pending_discarded"
        broker.place_order.assert_not_called()
        assert state.pending_action is None
        alerts = [c.args[0] for c in notify.call_args_list]
        assert any("過期 pending" in m for m in alerts)

    def test_today_pending_executes(self):
        state = TradingState(
            position=0, equity=500_000,
            pending_action="buy", pending_contracts=2,
            pending_signal_date=today_taipei().isoformat(),
        )
        orch, _ = _make_orch(state)
        broker = MagicMock()
        broker.place_order.return_value = {"order_id": "1", "fill_price": 21000.0}
        broker.get_positions.return_value = [
            {"code": "MXFG6", "direction": "Buy", "contracts": 2, "avg_price": 21000.0},
        ]
        with patch("time.sleep"):
            result = orch.run_execution(broker=broker, exec_price=21000.0)
        assert result["action"] == "buy"
        broker.place_order.assert_called_once_with("MXF", "Buy", 2)


# ─────────────────────────────────────────────────────────────────────────
# Broker auto-reconcile
# ─────────────────────────────────────────────────────────────────────────
class TestReconcileStateWithBroker:
    def test_manual_close_adopted(self):
        """App 手動平倉後（broker=0, state=2）→ state 自動歸零。"""
        state = TradingState(position=2, entry_price=21000.0, contracts=2,
                             highest_high=21500.0, equity=500_000)
        notify = MagicMock()
        orch, state_mgr = _make_orch(state, notify=notify)
        broker = MagicMock()
        broker.get_positions.return_value = []

        changed = orch.reconcile_state_with_broker(broker, sleep=_NOSLEEP)

        assert changed is True
        assert state.position == 0 and state.entry_price is None
        state_mgr.save.assert_called_once()
        assert any("自動同步" in c.args[0] for c in notify.call_args_list)

    def test_transient_zero_read_does_not_wipe(self):
        """第一次讀到 0、第二次讀回 2 → 視為讀取不穩定，state 不動。"""
        state = TradingState(position=2, entry_price=21000.0, contracts=2,
                             equity=500_000)
        notify = MagicMock()
        orch, state_mgr = _make_orch(state, notify=notify)
        broker = MagicMock()
        broker.get_positions.side_effect = [
            [],
            [{"code": "MXFG6", "direction": "Buy", "contracts": 2,
              "avg_price": 21000.0}],
        ]

        changed = orch.reconcile_state_with_broker(broker, sleep=_NOSLEEP)

        assert changed is False
        assert state.position == 2
        state_mgr.save.assert_not_called()

    def test_manual_open_adopted_with_broker_avg(self):
        """App 手動建倉 2 口（state 空倉）→ 採 broker 均價入 state。"""
        state = TradingState(position=0, equity=500_000)
        orch, state_mgr = _make_orch(state)
        broker = MagicMock()
        broker.get_positions.return_value = [
            {"code": "MXFG6", "direction": "Buy", "contracts": 2,
             "avg_price": 21_234.0},
        ]

        changed = orch.reconcile_state_with_broker(broker, sleep=_NOSLEEP)

        assert changed is True
        assert state.position == 2
        assert state.entry_price == 21_234.0
        assert state.highest_high == 21_234.0

    def test_matching_state_untouched(self):
        state = TradingState(position=2, entry_price=21000.0, contracts=2,
                             equity=500_000)
        orch, state_mgr = _make_orch(state)
        broker = MagicMock()
        broker.get_positions.return_value = [
            {"code": "MXFG6", "direction": "Buy", "contracts": 2,
             "avg_price": 21000.0},
        ]
        assert orch.reconcile_state_with_broker(broker, sleep=_NOSLEEP) is False
        state_mgr.save.assert_not_called()

    def test_unreadable_broker_leaves_state(self):
        state = TradingState(position=2, entry_price=21000.0, contracts=2,
                             equity=500_000)
        notify = MagicMock()
        orch, state_mgr = _make_orch(state, notify=notify)
        broker = MagicMock()
        broker.get_positions.side_effect = ConnectionError("down")

        assert orch.reconcile_state_with_broker(broker, sleep=_NOSLEEP) is False
        assert state.position == 2
        assert any("對帳失敗" in c.args[0] for c in notify.call_args_list)


# ─────────────────────────────────────────────────────────────────────────
# Pre-sell naked-short guard
# ─────────────────────────────────────────────────────────────────────────
class TestPreSellGuard:
    def test_close_skipped_when_broker_flat(self):
        """結算日 15:05：舊倉已被現金結算（broker 空）→ 絕不下 Sell
        （get_contract 會解析到次月合約 = 裸空單）。"""
        state = TradingState(
            position=2, entry_price=21000.0, contracts=2, equity=500_000,
            pending_action="close", pending_contracts=2,
            pending_signal_date=today_taipei().isoformat(),
            pending_reason="settlement-day force close",
        )
        notify = MagicMock()
        orch, state_mgr = _make_orch(state, notify=notify)
        broker = MagicMock()
        broker.get_positions.return_value = []

        result = orch.run_execution(broker=broker, exec_price=21000.0)

        assert result["action"] == "close_already_flat"
        broker.place_order.assert_not_called()
        assert state.position == 0 and state.pending_action is None

    def test_partial_position_sells_broker_qty(self):
        """broker 只剩 1 口（state 記 2）→ 只平 1 口。"""
        state = TradingState(
            position=2, entry_price=21000.0, contracts=2, equity=500_000,
            pending_action="close", pending_contracts=2,
            pending_signal_date=today_taipei().isoformat(),
            pending_reason="trailing stop",
        )
        orch, _ = _make_orch(state)
        broker = MagicMock()
        broker.place_order.return_value = {"order_id": "1", "status": "Filled"}
        broker.get_positions.side_effect = [
            [{"code": "MXFG6", "direction": "Buy", "contracts": 1,
              "avg_price": 21000.0}],
            [],
        ]
        with patch("time.sleep"):
            orch.run_execution(broker=broker, exec_price=21500.0)
        broker.place_order.assert_called_once_with("MXF", "Sell", 1)


# ─────────────────────────────────────────────────────────────────────────
# Holiday-adjusted settlement day
# ─────────────────────────────────────────────────────────────────────────
class TestHolidayAdjustedSettlement:
    def test_2026_february_defers_past_cny(self):
        """2026-02-18（第三個週三）在春節休市 — 結算順延到下一交易日。"""
        from src.data.tw_holidays import settlement_day_of_month

        s = settlement_day_of_month(date(2026, 2, 1))
        assert s > date(2026, 2, 18)
        from src.data.tw_holidays import is_trading_day

        assert is_trading_day(s)
        # engine follows suit
        assert _is_settlement_day(pd.Timestamp("2026-02-18")) is False
        assert _is_settlement_day(pd.Timestamp(s)) is True

    def test_normal_month_unchanged(self):
        """2026-04-15 是正常的第三個週三 → 判斷不變。"""
        assert _is_settlement_day(pd.Timestamp("2026-04-15")) is True
        assert _is_settlement_day(pd.Timestamp("2026-04-14")) is False


# ─────────────────────────────────────────────────────────────────────────
# State corruption → backup restore / refuse to trade
# ─────────────────────────────────────────────────────────────────────────
class TestStateCorruptionRecovery:
    def test_corrupt_state_restores_latest_backup(self, tmp_path):
        state_path = tmp_path / "state_x.json"
        backup = tmp_path / "state_x_backup_2026-07-01.json"
        backup.write_text(json.dumps(
            {"state": {"position": 3, "equity": 700_000.0}, "trades": []}
        ))
        state_path.write_text("{corrupt json")

        with patch("src.state.state_manager._BACKUP_DIR", str(tmp_path)):
            mgr = StateManager(path=str(state_path))
            state = mgr.load()

        assert state.position == 3
        assert state.equity == 700_000.0

    def test_corrupt_state_without_backup_raises(self, tmp_path):
        state_path = tmp_path / "state_y.json"
        state_path.write_text("{corrupt json")
        with patch("src.state.state_manager._BACKUP_DIR", str(tmp_path)):
            mgr = StateManager(path=str(state_path))
            try:
                mgr.load()
                raise AssertionError("must raise, never trade on default flat state")
            except StateCorruptionError:
                pass

    def test_missing_file_still_returns_initial_equity(self, tmp_path):
        mgr = StateManager(path=str(tmp_path / "new.json"), initial_equity=880_000.0)
        assert mgr.load().equity == 880_000.0


# ─────────────────────────────────────────────────────────────────────────
# Trading-day job guard
# ─────────────────────────────────────────────────────────────────────────
class TestTradingDayGuard:
    def test_holiday_detected(self):
        from src.scheduler.main import _is_tw_trading_day

        with patch("src.utils.tw_time.now_taipei") as m:
            import datetime as dt

            from src.utils.tw_time import TAIPEI

            m.return_value = dt.datetime(2026, 2, 18, 14, 30, tzinfo=TAIPEI)  # 春節
            assert _is_tw_trading_day() is False
            m.return_value = dt.datetime(2026, 7, 3, 14, 30, tzinfo=TAIPEI)  # 週五
            assert _is_tw_trading_day() is True

"""裁示 2026-08-05: 15:05 execution sizing 重檢 (只降不升) + pyramid 加碼路徑
補 margin_buffer (_max_total_contracts)。

Regressions 指定三案: pending 3 → 重檢降 2 + ⚠️ 全文;equity/ATR 不變 no-op;
降到 0 → 跳過進場、不下單。"""
from __future__ import annotations

import math
from unittest.mock import MagicMock

from src.scheduler.orchestrator import V2bOrchestrator
from src.state.state_manager import StateManager, TradingState
from src.strategy.v2b_engine import V2bEngine
from src.utils.tw_time import now_taipei
from tests.fakes import write_synthetic_parquet


def _engine() -> V2bEngine:
    return V2bEngine(product="MXF", ema_fast=30, ema_slow=100, confirm_days=2,
                     adx_threshold=25, trail_atr_mult=2.0,
                     risk_cap_pct=0.55, margin_buffer_atr=1.0,
                     # margin 縮小到不咬 (合成資料 ATR≈16, equity 由 cap 反推
                     # 到數千元量級) — 讓測試單獨隔離 risk-cap 這一層。
                     margin_per_contract=1_000, max_contracts=20)


def _orch(tmp_path, state: TradingState):
    pq = tmp_path / "MXF.parquet"
    write_synthetic_parquet(pq)
    eng = _engine()
    state_mgr = MagicMock(spec=StateManager)
    state_mgr.load.return_value = state
    notify = MagicMock()
    orch = V2bOrchestrator(strategy=eng, state_mgr=state_mgr, notify_fn=notify,
                           execution_timing="night_open", live=False,
                           data_path=pq)
    df = orch._load_data()
    atr_v = float(eng._compute_indicators(df).iloc[-1]["atr"])
    return orch, notify, atr_v


def _pending_buy(equity: float) -> TradingState:
    return TradingState(
        position=0, contracts=0, equity=equity,
        pending_action="buy", pending_contracts=3,
        pending_signal_date=now_taipei().strftime("%Y-%m-%d"),
        pending_reason="golden cross test",
    )


def _notes(notify) -> list[str]:
    return [c.args[0] for c in notify.call_args_list]


class TestExecutionRecheck:
    def test_pending3_rechecked_down_to_2_with_alert(self, tmp_path):
        """equity 於 14:30→15:05 間下滑 → cap 只容 2口: min(3,2)=2 + ⚠️ 全文。"""
        probe_state = TradingState(position=0, contracts=0, equity=1.0)
        _, _, atr_v = _orch(tmp_path, probe_state)
        risk_per = 2.0 * atr_v * 50.0
        equity = 2.4 * risk_per / 0.55            # floor(0.55E/risk_per) == 2
        state = _pending_buy(equity)
        orch, notify, _ = _orch(tmp_path, state)

        res = orch.run_execution(broker=None, exec_price=20_999.0)

        assert res["contracts"] == 2
        assert state.position == 2 and state.contracts == 2
        msgs = [n for n in _notes(notify) if "15:05 重檢降口數" in n]
        assert len(msgs) == 1
        assert "進場): 3→2口" in msgs[0] and "risk-cap" in msgs[0]
        assert f"{equity:,.0f}" in msgs[0]        # 15:05 淨值入訊息

    def test_unchanged_equity_atr_is_noop(self, tmp_path):
        probe_state = TradingState(position=0, contracts=0, equity=1.0)
        _, _, atr_v = _orch(tmp_path, probe_state)
        equity = 20.0 * (2.0 * atr_v * 50.0) / 0.55   # cap 遠大於 3
        state = _pending_buy(equity)
        orch, notify, _ = _orch(tmp_path, state)

        res = orch.run_execution(broker=None, exec_price=20_999.0)

        assert res["contracts"] == 3
        assert state.position == 3
        assert not any("重檢" in n for n in _notes(notify))

    def test_recheck_to_zero_skips_entry(self, tmp_path):
        probe_state = TradingState(position=0, contracts=0, equity=1.0)
        _, _, atr_v = _orch(tmp_path, probe_state)
        equity = 0.4 * (2.0 * atr_v * 50.0) / 0.55    # floor == 0
        state = _pending_buy(equity)
        orch, notify, _ = _orch(tmp_path, state)

        res = orch.run_execution(broker=None, exec_price=20_999.0)

        assert state.position == 0 and state.contracts == 0
        assert res.get("filled") == 0
        assert any("15:05 重檢 0口可進 — 跳過進場" in n for n in _notes(notify))
        assert state.pending_action is None           # pending 已清


class TestMaxTotalContracts:
    """§4(a): pyramid 加碼路徑補 margin_buffer。"""

    def test_buffer_binds_on_total(self):
        eng = V2bEngine(product="MXF", ema_fast=30, ema_slow=100,
                        margin_per_contract=131_500, margin_buffer_atr=1.0,
                        max_contracts=20)
        # denom = 131,500 + 1×1,660×50 = 214,506 → floor(640,000/214,506) = 2
        # (無 buffer 時 floor(640,000/131,500) = 4)
        assert eng._max_total_contracts(640_000.0, 1_660.13) == 2

    def test_no_buffer_bit_identical_to_plain_margin_floor(self):
        eng = V2bEngine(product="MXF", ema_fast=30, ema_slow=100,
                        margin_per_contract=131_500)
        expected = max(1, math.floor(640_000 / 131_500))
        assert eng._max_total_contracts(640_000.0, 1_660.13) == expected

    def test_risk_cap_binds_total(self):
        eng = V2bEngine(product="MXF", ema_fast=30, ema_slow=100,
                        margin_per_contract=131_500, risk_cap_pct=0.55)
        # floor(0.55×640,000 / (2×1,660.13×50)) = floor(2.12) = 2 < margin floor 4
        assert eng._max_total_contracts(640_000.0, 1_660.13) == 2


def test_track_pnl_aborts_on_zero_equity(monkeypatch, capsys):
    """§9: equity=0 fallback 不得進分帳 (偽報全員 −100%)。"""
    from scripts import pnl_tracker

    monkeypatch.setattr(pnl_tracker, "_load_config",
                        lambda: {"investors": [{"name": "Kira", "capital": 400_000}]})
    monkeypatch.setattr(pnl_tracker, "get_equity", lambda: (0.0, "估算"))
    assert pnl_tracker.track_pnl() is None
    assert "跳過分帳" in capsys.readouterr().out

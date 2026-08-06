"""裁示 2026-08-05: 15:05 execution sizing 重檢 (只降不升) + pyramid 加碼路徑
補 margin_buffer (_max_total_contracts)。

重檢用 14:30 持久化的 state.pending_atr — 同一顆日 ATR、下單前零行情抓取
(verifier R1/R2 修正)。Regressions 指定三案: pending 3 → 重檢降 2 + ⚠️ 全文;
equity/ATR 不變 no-op;降到 0 → 跳過進場、不下單、LINE 顯示 SKIP 非 BUY 0。"""
from __future__ import annotations

import math
from unittest.mock import MagicMock

from src.scheduler.orchestrator import V2bOrchestrator
from src.state.state_manager import StateManager, TradingState
from src.strategy.v2b_engine import V2bEngine
from src.utils.tw_time import now_taipei

_ATR = 1_660.13                       # 2026-08-04 production 值
_RISK_PER = 2.0 * _ATR * 50.0         # 166,013 /口


def _engine() -> V2bEngine:
    return V2bEngine(product="MXF", ema_fast=30, ema_slow=100, confirm_days=2,
                     adx_threshold=25, trail_atr_mult=2.0,
                     risk_cap_pct=0.55, margin_buffer_atr=1.0,
                     # margin 縮小到不咬 — 隔離 risk-cap 這一層
                     margin_per_contract=1_000, max_contracts=20)


def _orch(state: TradingState):
    state_mgr = MagicMock(spec=StateManager)
    state_mgr.load.return_value = state
    notify = MagicMock()
    orch = V2bOrchestrator(strategy=_engine(), state_mgr=state_mgr,
                           notify_fn=notify, execution_timing="night_open",
                           live=False)
    return orch, notify


def _pending_buy(equity: float, atr: float | None = _ATR) -> TradingState:
    return TradingState(
        position=0, contracts=0, equity=equity,
        pending_action="buy", pending_contracts=3,
        pending_signal_date=now_taipei().strftime("%Y-%m-%d"),
        pending_reason="golden cross test", pending_atr=atr,
    )


def _notes(notify) -> list[str]:
    return [c.args[0] for c in notify.call_args_list]


class TestExecutionRecheck:
    def test_pending3_rechecked_down_to_2_with_alert(self):
        """equity 於 14:30→15:05 間下滑至 750K → cap 0.55 只容 2口。"""
        state = _pending_buy(equity=750_000.0)   # floor(412,500/166,013) == 2
        orch, notify = _orch(state)

        res = orch.run_execution(broker=None, exec_price=43_000.0)

        assert res["contracts"] == 2
        assert state.position == 2 and state.contracts == 2
        msgs = [n for n in _notes(notify) if "15:05 重檢降口數" in n]
        assert len(msgs) == 1
        assert "進場): 3→2口" in msgs[0] and "risk-cap" in msgs[0]
        assert "750,000" in msgs[0]              # 15:05 淨值入訊息

    def test_unchanged_equity_atr_is_noop(self):
        state = _pending_buy(equity=1_000_000.0)  # cap floor 3.31 → 3 == pending
        orch, notify = _orch(state)

        res = orch.run_execution(broker=None, exec_price=43_000.0)

        assert res["contracts"] == 3
        assert state.position == 3
        assert not any("重檢" in n for n in _notes(notify))

    def test_recheck_to_zero_skips_entry(self):
        state = _pending_buy(equity=250_000.0)    # floor(137,500/166,013) == 0
        orch, notify = _orch(state)

        res = orch.run_execution(broker=None, exec_price=43_000.0)

        assert state.position == 0 and state.contracts == 0
        assert res.get("filled") == 0
        assert res.get("skipped_by_recheck") is True
        assert any("15:05 重檢 0口可進 — 跳過進場" in n for n in _notes(notify))
        # LINE 執行通知須顯示 SKIP,不得出現「BUY 0×MXF」偽執行 (verifier R4)
        assert any("SKIP（15:05 重檢 0口，未下單）" in n for n in _notes(notify))
        assert not any("BUY 0×MXF" in n for n in _notes(notify))
        assert state.pending_action is None       # pending 已清
        assert state.pending_atr is None

    def test_missing_pending_atr_degrades_to_noop(self):
        """舊 state 檔無 pending_atr (升級首日) → 沿用 14:30 口數,不抓行情。"""
        state = _pending_buy(equity=250_000.0, atr=None)
        orch, notify = _orch(state)

        res = orch.run_execution(broker=None, exec_price=43_000.0)

        assert res["contracts"] == 3
        assert state.position == 3
        assert not any("重檢" in n for n in _notes(notify))

    def test_gate_math_error_degrades_to_noop(self, monkeypatch):
        """gate 內任何例外 → no-op 沿用 14:30 口數,不 abort run_execution
        (verifier R3: try/except 須涵蓋 gate 計算本身)。"""
        state = _pending_buy(equity=1_000_000.0)
        orch, notify = _orch(state)

        def _boom(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr(orch.strategy, "_risk_capped_contracts", _boom)
        res = orch.run_execution(broker=None, exec_price=43_000.0)

        assert res["contracts"] == 3              # 降級沿用 14:30 口數
        assert state.position == 3
        assert state.pending_action is None       # 流程走完,pending 已清


class TestMaxTotalContracts:
    """§4(a): pyramid 加碼路徑補 margin_buffer。"""

    def test_buffer_binds_on_total(self):
        eng = V2bEngine(product="MXF", ema_fast=30, ema_slow=100,
                        margin_per_contract=131_500, margin_buffer_atr=1.0,
                        max_contracts=20)
        # denom = 131,500 + 1×1,660.13×50 = 214,506 → floor(640,000/214,506) = 2
        # (無 buffer 時 floor(640,000/131,500) = 4)
        assert eng._max_total_contracts(640_000.0, _ATR) == 2

    def test_no_buffer_bit_identical_to_plain_margin_floor(self):
        eng = V2bEngine(product="MXF", ema_fast=30, ema_slow=100,
                        margin_per_contract=131_500)
        expected = max(1, math.floor(640_000 / 131_500))
        assert eng._max_total_contracts(640_000.0, _ATR) == expected

    def test_risk_cap_binds_total(self):
        eng = V2bEngine(product="MXF", ema_fast=30, ema_slow=100,
                        margin_per_contract=131_500, risk_cap_pct=0.55)
        # floor(0.55×640,000 / 166,013) = 2 < margin floor 4
        assert eng._max_total_contracts(640_000.0, _ATR) == 2


def test_track_pnl_aborts_on_zero_equity(monkeypatch, capsys):
    """§9: equity=0 fallback 不得進分帳 (偽報全員 −100%)。"""
    from scripts import pnl_tracker

    monkeypatch.setattr(pnl_tracker, "_load_config",
                        lambda: {"investors": [{"name": "Kira", "capital": 400_000}]})
    monkeypatch.setattr(pnl_tracker, "get_equity", lambda: (0.0, "估算"))
    assert pnl_tracker.track_pnl() is None
    assert "跳過分帳" in capsys.readouterr().out

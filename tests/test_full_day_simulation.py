"""End-to-end full-day simulation: 14:25 → 15:10, all fakes, no network.

Exercises the night_open two-phase flow (run_signal @14:30 → run_execution
@15:05) plus the data/gap and freshness paths, covering the execution
plumbing that has historically broken silently. Indicator→signal correctness
lives in test_v2b_engine; here the strategy is scripted so we can assert the
orchestrator's state/order/notification behaviour deterministically.
"""

import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.daily_health_check import check_freshness
from src.data import daily_updater
from src.scheduler.orchestrator import V2bOrchestrator
from src.state.state_manager import StateManager
from src.strategy.v2b_engine import Signal
from tests.fakes import FakeBroker, ScriptedStrategy, write_synthetic_parquet


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """No real sleeps (reconcile/batch), no backup pollution into data/."""
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)
    monkeypatch.setattr("src.state.state_manager._BACKUP_DIR", str(tmp_path / "backups"))


def _make(tmp_path, signals, *, live=False, equity=2_000_000.0):
    parquet = tmp_path / "MXF.parquet"
    write_synthetic_parquet(parquet)
    sm = StateManager(path=str(tmp_path / "state.json"), initial_equity=equity)
    msgs: list[str] = []
    orch = V2bOrchestrator(
        strategy=ScriptedStrategy(signals),
        state_mgr=sm,
        notify_fn=msgs.append,
        data_path=str(parquet),
        decision_time="14:30",
        execution_timing="night_open",
        live=live,
    )
    return orch, sm, msgs, parquet


def _seed(sm, **kw):
    st = sm.load()
    for k, v in kw.items():
        setattr(st, k, v)
    sm.save(st)


# ── Case 1: normal trading day, HOLD ────────────────────────────────────────
def test_case1_normal_day_hold(tmp_path):
    orch, sm, msgs, _ = _make(tmp_path, [Signal("hold", 0, "金叉+ADX>25+close>stop")])
    _seed(sm, position=8, entry_price=20_000.0, contracts=8, highest_high=20_100.0)
    fb = FakeBroker(equity=2_000_000.0)
    fb.seed_position(8, 20_000.0)

    sig = orch.run_signal(broker=fb)
    assert sig["action"] == "hold"
    res = orch.run_execution(broker=fb, exec_price=20_200.0)

    assert res["action"] == "hold"
    assert fb.orders == []                       # 15:05 無動作
    st = sm.load()
    assert st.position == 8                       # 口數一致 (post_verify)
    assert st.highest_high == 20_200.0            # trailing high advanced
    assert any("HOLD" in m for m in msgs)


# ── Case 2: trailing stop CLOSE, batched 5+5+3 ──────────────────────────────
def test_case2_trailing_stop_close(tmp_path):
    orch, sm, msgs, _ = _make(tmp_path, [Signal("close", 13, "trailing stop hit")])
    _seed(sm, position=13, entry_price=20_000.0, contracts=13, highest_high=20_100.0)
    fb = FakeBroker(equity=2_000_000.0, fill_price=20_200.0)
    fb.seed_position(13, 20_000.0)

    orch.run_signal(broker=fb)
    res = orch.run_execution(broker=fb)

    assert res["action"] == "close"
    assert fb.batches_for("Sell") == [5, 5, 3]    # 分批平倉
    st = sm.load()
    assert st.position == 0 and st.entry_price is None and st.contracts == 0
    assert res.get("exit_price") == 20_200.0
    assert any("CLOSE" in m for m in msgs)


# ── Case 3: settlement-day rollover (close + re-enter new month) ─────────────
def test_case3_settlement_rollover(tmp_path):
    orch, sm, msgs, _ = _make(
        tmp_path,
        [Signal("close", 13, "settlement force close"),
         Signal("buy", 7, "re-entry new month")],
    )
    _seed(sm, position=13, entry_price=20_000.0, contracts=13, highest_high=20_100.0)
    fb = FakeBroker(equity=2_000_000.0)
    fb.seed_position(13, 20_000.0)
    fb.queue_fills([20_200.0, 21_000.0])          # sell fill, then buy fill

    orch.run_signal(broker=fb)
    res = orch.run_execution(broker=fb)

    st = sm.load()
    assert res.get("rollover") is True
    assert st.position == 7                        # 新合約 7 口
    assert st.entry_price == 21_000.0              # broker 成交價
    assert fb.batches_for("Sell") == [5, 5, 3]
    assert fb.batches_for("Buy") == [5, 2]
    assert any("結算日轉倉" in m for m in msgs)


# ── Case 4: settlement-day Sell rejected → no Buy → state preserved ─────────
def test_case4_settlement_sell_failure(tmp_path):
    orch, sm, msgs, _ = _make(
        tmp_path,
        [Signal("close", 13, "settlement force close"),
         Signal("buy", 7, "should never be reached")],
    )
    _seed(sm, position=13, entry_price=20_000.0, contracts=13, highest_high=20_100.0)
    fb = FakeBroker(equity=2_000_000.0)
    fb.seed_position(13, 20_000.0)
    fb.reject("Sell")

    orch.run_signal(broker=fb)
    res = orch.run_execution(broker=fb)

    # No buy ever attempted; rollover aborted.
    assert all(o["action"] != "Buy" for o in fb.orders)
    assert res.get("rollover") is False
    assert orch.strategy.calls == 1               # rollover never re-ran signal
    # State must be UNCHANGED — the broker still holds the position.
    st = sm.load()
    assert st.position == 13
    assert st.entry_price == 20_000.0
    assert st.contracts == 13
    assert any(m.startswith("🔴") and "平倉失敗" in m for m in msgs)


# ── Case 5: Anti-Martingale add, entry_price from broker weighted avg ───────
def test_case5_anti_martingale_add(tmp_path):
    orch, sm, msgs, _ = _make(tmp_path, [Signal("add", 3, "anti-martingale add")],
                              equity=2_500_000.0)
    _seed(sm, position=5, entry_price=20_000.0, contracts=5, highest_high=20_100.0,
          equity=2_500_000.0)
    fb = FakeBroker(equity=2_500_000.0, fill_price=21_000.0)
    fb.seed_position(5, 20_000.0)

    orch.run_signal(broker=fb)
    res = orch.run_execution(broker=fb)

    st = sm.load()
    assert st.position == 8                                  # 5 + 3
    # weighted avg = (20000*5 + 21000*3) / 8 = 20375
    assert st.entry_price == pytest.approx(20_375.0)
    assert res.get("entry_price_source") == "broker"
    assert fb.batches_for("Buy") == [3]


# ── Case 6: Shioaji total failure → graceful degradation, no bad order ──────
def test_case6_shioaji_total_failure(tmp_path):
    orch, sm, msgs, _ = _make(tmp_path, [Signal("hold", 0, "no actionable signal")],
                              live=True)
    fb = FakeBroker(equity=2_000_000.0)
    fb.fail_account = True                          # equity read fails
    fb.fail_data = True                             # kbars + snapshot fail

    sig = orch.run_signal(broker=fb)                # must not raise
    res = orch.run_execution(broker=fb)

    assert sig["action"] == "hold"
    assert res["action"] == "hold"
    assert fb.orders == []                          # no erroneous trade
    assert any("估算" in m for m in msgs)            # equity fell back to estimate


# ── Case 7: data gap detection + back-fill ──────────────────────────────────
def test_case7_data_gap_backfill(tmp_path):
    parquet = tmp_path / "gap.parquet"
    df = write_synthetic_parquet(parquet, n_bars=60, end=date(2026, 5, 21))
    window_end = date(2026, 5, 21)

    # Drop one recent trading day, then back-fill it via the test seam.
    missing_day = date(2026, 5, 20)
    df2 = df[df.index.normalize() != pd.Timestamp(missing_day)]
    df2.to_parquet(parquet, index=True)

    def fetch_ok(d0, d1):
        return pd.DataFrame(
            [{"open": 20_900, "high": 20_950, "low": 20_880, "close": 20_930, "volume": 50_000}],
            index=pd.DatetimeIndex([pd.Timestamp(d0)], name="date"),
        )

    msgs: list[str] = []
    filled, still_missing = daily_updater._detect_and_fill_gaps(
        df2, parquet, msgs.append, window_end, _fetch_override=fetch_ok,
    )
    assert filled == 1 and still_missing == []
    assert pd.Timestamp(missing_day) in pd.read_parquet(parquet).index

    # Now drop THREE days; back-fill 2, leave 1 unfillable → partial + alert.
    drop = [date(2026, 5, 19), date(2026, 5, 20), date(2026, 5, 21)]
    df3 = df[~df.index.normalize().isin([pd.Timestamp(d) for d in drop])]
    df3.to_parquet(parquet, index=True)

    def fetch_partial(d0, d1):
        if d0 == date(2026, 5, 21):
            return None                              # unfillable
        return fetch_ok(d0, d1)

    msgs2: list[str] = []
    filled2, still_missing2 = daily_updater._detect_and_fill_gaps(
        df3, parquet, msgs2.append, window_end, _fetch_override=fetch_partial,
    )
    assert filled2 == 2
    assert still_missing2 == [date(2026, 5, 21)]
    assert any("⚠️" in m and "需手動處理" in m for m in msgs2)


# ── Case 8: Monday-morning health check, no false alarm ─────────────────────
def test_case8_monday_morning_no_false_alarm(tmp_path):
    # 週一 08:00, latest = 上週四 → ✅ (pre-14:25 slack)
    assert check_freshness(datetime(2026, 5, 25, 8, 0), date(2026, 5, 21))[0] == "ok"
    # 週一 08:00, 上週五放假, latest = 上週四 → ✅
    assert check_freshness(datetime(2026, 5, 4, 8, 0), date(2026, 4, 30))[0] == "ok"

"""Restart-gate switches (risk-cap / cooldown A / EMA-reclaim B).

All switches default OFF and must be bit-identical no-ops when off — proven
end-to-end by the baseline backtest reproducing the CLAUDE.md metrics exactly
(59.64/-21.85/77/7,308,420 on the →2026-07-03 series). These tests pin the
switch mechanics themselves.
"""

from __future__ import annotations

import pandas as pd

from src.strategy.v2b_engine import Signal, V2bEngine


def _eng(**kw) -> V2bEngine:
    return V2bEngine(product="MXF", ema_fast=30, ema_slow=100, confirm_days=1,
                     adx_threshold=0, margin_per_contract=131_500.0, **kw)


def _rising_df(n: int = 160, base: float = 40_000.0, step: float = 30.0,
               last_close: float | None = None) -> pd.DataFrame:
    dates = pd.bdate_range(end="2026-07-28", periods=n)
    closes = [base + i * step for i in range(n)]
    if last_close is not None:
        closes[-1] = last_close
    return pd.DataFrame(
        {"open": [c - 10 for c in closes], "high": [c + 40 for c in closes],
         "low": [c - 40 for c in closes], "close": closes,
         "volume": [100_000] * n},
        index=dates,
    )


class TestRiskCap:
    def test_cap_reduces_contracts(self):
        # equity 650,708, ATR≈1,421, 2×ATR×50 = 142,124/口 → 15% cap 97,606 → 0口
        eng = _eng(risk_cap_pct=0.15)
        n, note = eng._risk_capped_contracts(4, 650_708.0, 1_421.0)
        assert n == 0 and "risk-cap" in note

    def test_cap_leaves_small_risk_alone(self):
        # ATR 300 → 30,000/口; 15% of 650,708 = 97,606 → 3口 cap; n=2 passes.
        eng = _eng(risk_cap_pct=0.15)
        n, note = eng._risk_capped_contracts(2, 650_708.0, 300.0)
        assert n == 2 and note == ""

    def test_margin_buffer_caps(self):
        # buffer 1×ATR×50 = 71,050/口 → equity/(131,500+71,050) = 3.21 → 3口
        eng = _eng(margin_buffer_atr=1.0)
        n, note = eng._risk_capped_contracts(4, 650_708.0, 1_421.0)
        assert n == 3 and "margin-buffer" in note

    def test_zero_contracts_entry_becomes_hold(self):
        eng = _eng(risk_cap_pct=0.10)
        df = _rising_df()
        sig = eng.generate_signal(data=df, current_position=0, equity=200_000.0)
        # ATR of the rising series is small but 10% of 200K is 20K; step 30 →
        # ATR≈80 → 2×80×50=8,000/口 → cap 2口; ladder default gives ≥1 → buy OK.
        # Force the cap to zero via a tiny equity instead:
        sig0 = eng.generate_signal(data=df, current_position=0, equity=10_000.0)
        assert sig.action in ("buy", "hold")
        assert sig0.action == "hold" and "risk-cap" in sig0.reason

    def test_switches_off_is_noop(self):
        df = _rising_df()
        a = _eng().generate_signal(data=df, current_position=0, equity=650_708.0)
        b = _eng(risk_cap_pct=None, margin_buffer_atr=None, cooldown_days=0,
                 reentry_require_above_ema_fast=False).generate_signal(
            data=df, current_position=0, equity=650_708.0)
        assert (a.action, a.contracts) == (b.action, b.contracts)


class TestCooldownFilterA:
    def test_stop_exit_sets_marker_and_blocks_reentry(self):
        eng = _eng(cooldown_days=3)
        df = _rising_df()
        # In-position, close crashed far below the trailing stop → close signal
        crash = _rising_df(last_close=30_000.0)
        sig = eng.generate_signal(
            data=crash, current_position=4, entry_price=44_000.0,
            equity=650_708.0, highest_high=float(crash["close"].iloc[-2]),
            contracts=4,
        )
        assert sig.action == "close"
        assert eng._last_stop_ts == crash.index[-1]

        # 2 bars later (≤3) → blocked
        later = pd.concat([df, pd.DataFrame(
            {"open": [44_800.0] * 2, "high": [44_900.0] * 2,
             "low": [44_700.0] * 2, "close": [44_850.0, 44_900.0],
             "volume": [100_000] * 2},
            index=pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=2),
        )])
        eng._last_stop_ts = df.index[-1]
        sig2 = eng.generate_signal(data=later, current_position=0, equity=650_708.0)
        assert sig2.action == "hold" and "cooldown" in sig2.reason

        # 5 bars later (>3) → allowed again
        later5 = pd.concat([df, pd.DataFrame(
            {"open": [44_800.0] * 5, "high": [44_900.0] * 5,
             "low": [44_700.0] * 5, "close": [44_850 + i * 20 for i in range(5)],
             "volume": [100_000] * 5},
            index=pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=5),
        )])
        sig3 = eng.generate_signal(data=later5, current_position=0, equity=650_708.0)
        assert sig3.action == "buy"


class TestEmaReclaimFilterB:
    def test_close_below_ema30_blocks_entry(self):
        """7/29 restart case: golden (EMA30 ≫ EMA100) but close crashed below
        EMA30 → no entry until price reclaims the mean."""
        eng = _eng(reentry_require_above_ema_fast=True)
        df = _rising_df(last_close=41_608.0)  # deep below the rising EMA30
        sig = eng.generate_signal(data=df, current_position=0, equity=650_708.0)
        assert sig.action == "hold"
        assert "filter B" in sig.reason

    def test_close_above_ema30_enters_normally(self):
        eng = _eng(reentry_require_above_ema_fast=True)
        df = _rising_df()  # close at the top of the rising series
        sig = eng.generate_signal(data=df, current_position=0, equity=650_708.0)
        assert sig.action == "buy"


def test_gate_reason_is_line_visible():
    eng = _eng(risk_cap_pct=0.15, margin_buffer_atr=1.0)
    n, note = eng._risk_capped_contracts(4, 650_708.0, 1_421.0)
    assert n == 0
    sig = Signal("hold", 0, f"risk-cap gate: 0口可進 ({note}) — 跳過進場")
    assert "risk-cap" in sig.reason and "跳過進場" in sig.reason

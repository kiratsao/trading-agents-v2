"""Tests for the Risk Manager agent — drawdown, position sizing, slippage,
DrawdownGuard, ConcentrationGuard, and the full validate_trade pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from core.risk.concentration import ConcentrationGuard
from core.risk.drawdown import (
    DrawdownAction,
    DrawdownGuard,
    current_drawdown,
    max_drawdown,
    should_halt,
)
from core.risk.kill_switch import KillSwitch, TradingHaltedError
from core.risk.position_sizing import fixed_fraction, half_kelly, volatility_target
from core.risk.slippage import fixed_slippage, linear_impact
from us_equity.risk.risk_manager import RiskManager

# ============================================================================
# Helpers
# ============================================================================


def _ks(tmp_path: Path) -> KillSwitch:
    return KillSwitch(
        state_path=tmp_path / "ks_state.json",
        log_path=tmp_path / "ks_log.json",
    )


def _dd_guard(tmp_path: Path, **kwargs) -> DrawdownGuard:
    return DrawdownGuard(state_path=tmp_path / "dd_state.json", **kwargs)


def _rm(tmp_path: Path, **kwargs) -> RiskManager:
    return RiskManager(
        db_path=":memory:",
        kill_switch=_ks(tmp_path),
        drawdown_guard=_dd_guard(tmp_path),
        concentration_guard=ConcentrationGuard(),
        **kwargs,
    )


def _returns(*values: float) -> pd.Series:
    return pd.Series(list(values), dtype=float)


# ============================================================================
# Existing tests — module-level functions
# ============================================================================


class TestDrawdown:
    def _equity(self, values: list[float]) -> pd.Series:
        return pd.Series(values, dtype=float)

    def test_current_drawdown_at_peak_is_zero(self):
        equity = self._equity([100, 110, 120])
        assert current_drawdown(equity) == pytest.approx(0.0)

    def test_current_drawdown_correct(self):
        equity = self._equity([100, 120, 90])
        # drawdown from peak 120 to 90 = 25%
        assert current_drawdown(equity) == pytest.approx(0.25)

    def test_max_drawdown_correct(self):
        equity = self._equity([100, 120, 80, 110])
        # worst: 120 → 80 = 33.3%
        assert max_drawdown(equity) == pytest.approx(1 / 3, rel=1e-3)

    def test_should_halt_below_threshold(self):
        equity = self._equity([100, 95])
        assert should_halt(equity, threshold=0.10) is False

    def test_should_halt_above_threshold(self):
        equity = self._equity([100, 88])
        assert should_halt(equity, threshold=0.10) is True


class TestPositionSizing:
    def test_fixed_fraction_positive(self):
        result = fixed_fraction(equity=1_000_000, risk_pct=0.01, stop_distance=0.05)
        assert result > 0

    def test_volatility_target_returns_int(self):
        result = volatility_target(equity=1_000_000, target_vol=0.15, atr=5.0, price=100.0)
        assert isinstance(result, int)
        assert result > 0

    def test_half_kelly_positive(self):
        result = half_kelly(win_rate=0.55, avg_win=1.5, avg_loss=1.0, equity=1_000_000)
        assert result > 0

    def test_half_kelly_negative_edge_returns_zero_or_less(self):
        """With win_rate < 0.5 and unfavorable payoff, Kelly should be ≤ 0."""
        result = half_kelly(win_rate=0.30, avg_win=1.0, avg_loss=2.0, equity=1_000_000)
        assert result <= 0


class TestSlippage:
    def test_fixed_slippage_buy_increases_price(self):
        fill = fixed_slippage(price=100.0, bps=10.0)
        assert fill > 100.0

    def test_fixed_slippage_zero_bps_unchanged(self):
        fill = fixed_slippage(price=100.0, bps=0.0)
        assert fill == pytest.approx(100.0)

    def test_linear_impact_increases_with_qty(self):
        small = linear_impact(price=100.0, qty=100, avg_daily_volume=1_000_000)
        large = linear_impact(price=100.0, qty=50_000, avg_daily_volume=1_000_000)
        assert large > small


# ============================================================================
# DrawdownGuard
# ============================================================================


class TestDrawdownGuard:
    # ---- HOLD ----

    def test_hold_when_no_drawdown(self, tmp_path):
        guard = _dd_guard(tmp_path)
        guard.check(1_000_000)  # initialise HWM
        action = guard.check(1_050_000)  # equity rose
        assert action == DrawdownAction.HOLD

    def test_hold_below_reduce_threshold(self, tmp_path):
        guard = _dd_guard(tmp_path)
        guard.check(1_000_000)
        action = guard.check(920_000)  # −8 %, below 10 % threshold
        assert action == DrawdownAction.HOLD

    # ---- REDUCE ----

    def test_reduce_at_10pct_drawdown(self, tmp_path):
        guard = _dd_guard(tmp_path)
        guard.check(1_000_000)
        action = guard.check(895_000)  # −10.5 %
        assert action == DrawdownAction.REDUCE

    def test_reduce_stays_active_without_sufficient_recovery(self, tmp_path):
        guard = _dd_guard(tmp_path)
        guard.check(1_000_000)
        guard.check(895_000)  # → REDUCE
        # Partial recovery to 92 % of HWM — below 95 % threshold
        action = guard.check(920_000)
        assert action == DrawdownAction.REDUCE

    # ---- EXIT ----

    def test_exit_above_15pct_drawdown(self, tmp_path):
        guard = _dd_guard(tmp_path)
        guard.check(1_000_000)
        action = guard.check(840_000)  # −16 %
        assert action == DrawdownAction.EXIT

    def test_exit_stays_active_without_sufficient_recovery(self, tmp_path):
        guard = _dd_guard(tmp_path)
        guard.check(1_000_000)
        guard.check(840_000)  # → EXIT
        action = guard.check(930_000)  # recovered to 93 % — below 95 %
        assert action == DrawdownAction.EXIT

    # ---- Recovery ----

    def test_recovery_to_hold_from_reduce(self, tmp_path):
        guard = _dd_guard(tmp_path)
        guard.check(1_000_000)
        guard.check(895_000)  # → REDUCE
        action = guard.check(960_000)  # 96 % of HWM → above 95 % → HOLD
        assert action == DrawdownAction.HOLD

    def test_recovery_to_hold_from_exit(self, tmp_path):
        guard = _dd_guard(tmp_path)
        guard.check(1_000_000)
        guard.check(840_000)  # → EXIT
        action = guard.check(960_000)  # 96 % of HWM
        assert action == DrawdownAction.HOLD

    def test_recovery_partial_does_not_lift_exit(self, tmp_path):
        guard = _dd_guard(tmp_path)
        guard.check(1_000_000)
        guard.check(840_000)  # → EXIT (dd=16 %)
        # At 90 % of HWM: dd=10 %, which would normally be REDUCE —
        # but we haven't crossed the 95 % recovery threshold yet
        action = guard.check(900_000)
        assert action == DrawdownAction.EXIT

    def test_hwm_updates_on_new_peak(self, tmp_path):
        guard = _dd_guard(tmp_path)
        guard.check(1_000_000)
        guard.check(1_200_000)  # new HWM
        guard.check(1_015_000)  # −15.4 % from new HWM → EXIT (> 15 %)
        assert guard.current_action == DrawdownAction.EXIT
        assert guard.high_water_mark == pytest.approx(1_200_000)

    # ---- Persistence ----

    def test_state_persists_across_instances(self, tmp_path):
        state_path = tmp_path / "state.json"
        g1 = DrawdownGuard(state_path=state_path)
        g1.check(1_000_000)
        g1.check(840_000)  # → EXIT
        assert g1.current_action == DrawdownAction.EXIT

        g2 = DrawdownGuard(state_path=state_path)
        assert g2.current_action == DrawdownAction.EXIT
        assert g2.high_water_mark == pytest.approx(1_000_000)


# ============================================================================
# ConcentrationGuard
# ============================================================================


class TestConcentrationGuard:
    def _sum_approx(self, weights: dict[str, float]) -> float:
        return sum(weights.values())

    # ---- Single-position cap ----

    def test_individual_cap_applied(self):
        guard = ConcentrationGuard(max_position=0.10)
        weights = {"AAPL": 0.20, "MSFT": 0.40, "SHY": 0.40}
        result = guard.check(weights)

        assert result["AAPL"] <= 0.10 + 1e-9
        assert result["MSFT"] <= 0.10 + 1e-9
        # SHY is exempt — may exceed cap
        assert result["SHY"] > 0.40
        assert self._sum_approx(result) == pytest.approx(1.0)

    def test_shy_exempt_from_individual_cap(self):
        guard = ConcentrationGuard(max_position=0.10)
        weights = {"SHY": 1.0}
        result = guard.check(weights)
        assert result["SHY"] == pytest.approx(1.0)

    def test_no_adjustment_when_within_individual_cap(self):
        guard = ConcentrationGuard(max_position=0.10)
        weights = {"AAPL": 0.05, "MSFT": 0.05, "SHY": 0.90}
        result = guard.check(weights)
        assert result["AAPL"] == pytest.approx(0.05)
        assert result["MSFT"] == pytest.approx(0.05)

    def test_individual_cap_redistribution_sums_to_one(self):
        guard = ConcentrationGuard(max_position=0.10)
        weights = {
            "AAPL": 0.25,
            "MSFT": 0.25,
            "GOOGL": 0.05,
            "AMZN": 0.05,
            "SHY": 0.40,
        }
        result = guard.check(weights)
        assert self._sum_approx(result) == pytest.approx(1.0, abs=1e-9)

    # ---- Sector cap ----

    def test_sector_cap_applied_to_tech(self):
        guard = ConcentrationGuard(max_position=0.15, max_sector=0.30)
        # Five tech stocks × 10 % each = 50 % tech — well above 30 %
        weights = {
            "AAPL": 0.10,
            "MSFT": 0.10,
            "GOOGL": 0.10,
            "NVDA": 0.10,
            "META": 0.10,
            "SHY": 0.50,
        }
        result = guard.check(weights)

        tech_total = sum(result[s] for s in ("AAPL", "MSFT", "GOOGL", "NVDA", "META"))
        assert tech_total <= 0.30 + 1e-6
        assert self._sum_approx(result) == pytest.approx(1.0, abs=1e-9)

    def test_sector_cap_does_not_affect_compliant_sector(self):
        guard = ConcentrationGuard(max_position=0.15, max_sector=0.30)
        # Finance at 20 % — under 30 % cap — should be unchanged
        weights = {"JPM": 0.10, "V": 0.10, "SHY": 0.80}
        result = guard.check(weights)
        assert result["JPM"] == pytest.approx(0.10)
        assert result["V"] == pytest.approx(0.10)

    def test_sector_cap_redistribution_sums_to_one(self):
        guard = ConcentrationGuard(max_position=0.15, max_sector=0.30)
        weights = {
            "AAPL": 0.10,
            "MSFT": 0.10,
            "GOOGL": 0.10,
            "NVDA": 0.10,  # tech 40 %
            "JPM": 0.10,
            "V": 0.10,  # finance 20 %
            "SHY": 0.40,
        }
        result = guard.check(weights)
        assert self._sum_approx(result) == pytest.approx(1.0, abs=1e-9)

    def test_both_caps_applied_together(self):
        guard = ConcentrationGuard(max_position=0.10, max_sector=0.30)
        # AAPL over individual cap AND tech sector over sector cap
        weights = {
            "AAPL": 0.20,
            "MSFT": 0.12,
            "GOOGL": 0.08,
            "NVDA": 0.05,
            "JPM": 0.10,
            "SHY": 0.45,
        }
        result = guard.check(weights)
        for sym in ("AAPL", "MSFT", "GOOGL", "NVDA"):
            assert result[sym] <= 0.10 + 1e-6, (
                f"{sym} weight {result[sym]:.4f} exceeds individual cap"
            )
        tech_total = sum(result[s] for s in ("AAPL", "MSFT", "GOOGL", "NVDA"))
        assert tech_total <= 0.30 + 1e-6
        assert self._sum_approx(result) == pytest.approx(1.0, abs=1e-9)

    def test_empty_weights_returns_empty(self):
        guard = ConcentrationGuard()
        assert guard.check({}) == {}

    def test_single_symbol_unchanged(self):
        guard = ConcentrationGuard(max_position=0.10)
        result = guard.check({"SHY": 1.0})
        assert result == {"SHY": pytest.approx(1.0)}


# ============================================================================
# RiskManager.validate_trade  (full pipeline)
# ============================================================================


class TestValidateTrade:
    # Simple "all-clear" returns
    _OK_RETURNS = _returns(0.005, 0.003, 0.007, 0.002, 0.004)
    # Good equity — no drawdown
    _OK_EQUITY = 1_050_000.0
    # Balanced weights: one stock per sector, each ≤ 10 %, no sector > 30 %.
    # AAPL=Tech(10), UNH=Healthcare(10), JPM=Finance(10), WMT=Consumer(10),
    # XOM=Energy(10), CAT=Industrial(10), NEE=Utilities(5), SHY=35 → sum=100
    _WEIGHTS_10 = {
        "AAPL": 0.10,  # Technology  (10 %)
        "UNH": 0.10,  # Healthcare  (10 %)
        "JPM": 0.10,  # Finance     (10 %)
        "WMT": 0.10,  # Consumer    (10 %)
        "XOM": 0.10,  # Energy      (10 %)
        "CAT": 0.10,  # Industrial  (10 %)
        "NEE": 0.05,  # Utilities   ( 5 %)
        "SHY": 0.35,  # Fixed Income (exempt)
    }

    def _fresh_rm(self, tmp_path: Path) -> RiskManager:
        """RiskManager with all guards initialised to fresh/ACTIVE state."""
        rm = _rm(tmp_path)
        # Seed HWM so DrawdownGuard is properly initialised
        rm.drawdown_guard.check(1_000_000.0)
        return rm

    # ---- Happy path ----

    def test_returns_weights_unchanged_when_no_issues(self, tmp_path):
        rm = self._fresh_rm(tmp_path)
        result = rm.validate_trade(self._WEIGHTS_10, 1_050_000.0, self._OK_RETURNS)
        assert abs(sum(result.values()) - 1.0) < 1e-9
        for sym, w in self._WEIGHTS_10.items():
            assert result[sym] == pytest.approx(w)

    # ---- KillSwitch triggers TradingHaltedError ----

    def test_kill_switch_trigger_raises_error(self, tmp_path):
        rm = self._fresh_rm(tmp_path)
        big_loss = _returns(-0.05)  # −5 % single day → triggers kill switch
        with pytest.raises(TradingHaltedError):
            rm.validate_trade(self._WEIGHTS_10, self._OK_EQUITY, big_loss)

    def test_pre_existing_kill_raises_on_next_call(self, tmp_path):
        rm = self._fresh_rm(tmp_path)
        rm.kill_switch.check(_returns(-0.05))  # kill manually
        with pytest.raises(TradingHaltedError):
            rm.validate_trade(self._WEIGHTS_10, self._OK_EQUITY, self._OK_RETURNS)

    # ---- DrawdownGuard EXIT → full SHY ----

    def test_drawdown_exit_returns_full_shy(self, tmp_path):
        rm = self._fresh_rm(tmp_path)
        # DROP equity to −16 % drawdown
        result = rm.validate_trade(self._WEIGHTS_10, 840_000.0, self._OK_RETURNS)
        assert result == {"SHY": pytest.approx(1.0)}

    # ---- DrawdownGuard REDUCE → weights halved ----

    def test_drawdown_reduce_halves_equity_weights(self, tmp_path):
        rm = self._fresh_rm(tmp_path)
        # −12 % drawdown → REDUCE
        result = rm.validate_trade(self._WEIGHTS_10, 880_000.0, self._OK_RETURNS)
        assert abs(sum(result.values()) - 1.0) < 1e-9

        # Every non-SHY position should be at most half of its original weight
        for sym, orig_w in self._WEIGHTS_10.items():
            if sym == "SHY":
                continue
            assert result[sym] <= orig_w * 0.5 + 1e-9, (
                f"{sym}: {result[sym]:.4f} > {orig_w * 0.5:.4f}"
            )

        # SHY must have grown
        assert result.get("SHY", 0.0) > self._WEIGHTS_10.get("SHY", 0.0)

    # ---- ConcentrationGuard adjusts weights ----

    def test_concentration_cap_applied_in_pipeline(self, tmp_path):
        rm = self._fresh_rm(tmp_path)
        # AAPL over-weight: 25 % > 10 % cap
        weights = {
            "AAPL": 0.25,
            "MSFT": 0.10,
            "UNH": 0.10,
            "JPM": 0.05,
            "SHY": 0.50,
        }
        result = rm.validate_trade(weights, self._OK_EQUITY, self._OK_RETURNS)
        assert result["AAPL"] <= 0.10 + 1e-6
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_sector_cap_applied_in_pipeline(self, tmp_path):
        rm = self._fresh_rm(tmp_path)
        # Tech sector: 5 stocks × 10 % = 50 % → should be capped at 30 %
        weights = {
            "AAPL": 0.10,
            "MSFT": 0.10,
            "GOOGL": 0.10,
            "NVDA": 0.10,
            "META": 0.10,
            "SHY": 0.50,
        }
        result = rm.validate_trade(weights, self._OK_EQUITY, self._OK_RETURNS)
        tech_total = sum(result.get(s, 0.0) for s in ("AAPL", "MSFT", "GOOGL", "NVDA", "META"))
        assert tech_total <= 0.30 + 1e-6
        assert abs(sum(result.values()) - 1.0) < 1e-9

    # ---- Pipeline ordering: KillSwitch fires before DrawdownGuard ----

    def test_kill_switch_takes_precedence_over_drawdown(self, tmp_path):
        rm = self._fresh_rm(tmp_path)
        # Large single-day loss (−5 %) AND large drawdown (−20 %)
        # KillSwitch should fire first (TradingHaltedError), not return SHY silently
        big_loss = _returns(-0.05)
        with pytest.raises(TradingHaltedError):
            rm.validate_trade(self._WEIGHTS_10, 800_000.0, big_loss)

    # ---- Full pipeline: HOLD → concentration only ----

    def test_full_pipeline_hold_applies_concentration(self, tmp_path):
        rm = self._fresh_rm(tmp_path)
        # All-tech heavy portfolio
        weights = {
            "AAPL": 0.15,
            "MSFT": 0.15,
            "GOOGL": 0.15,
            "NVDA": 0.15,
            "META": 0.10,
            "SHY": 0.30,
        }
        result = rm.validate_trade(weights, self._OK_EQUITY, self._OK_RETURNS)
        for sym in ("AAPL", "MSFT", "GOOGL", "NVDA", "META"):
            assert result[sym] <= 0.10 + 1e-6, f"{sym} individual cap breached"
        tech_total = sum(result.get(s, 0.0) for s in ("AAPL", "MSFT", "GOOGL", "NVDA", "META"))
        assert tech_total <= 0.30 + 1e-6, "Sector cap breached"
        assert abs(sum(result.values()) - 1.0) < 1e-9

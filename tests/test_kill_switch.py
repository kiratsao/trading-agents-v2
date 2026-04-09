"""Tests for KillSwitch and RiskManager.pre_trade_check()."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from core.risk.kill_switch import (
    SINGLE_DAY_LOSS_THRESHOLD,
    WEEKLY_LOSS_THRESHOLD,
    KillSwitch,
    KillSwitchEvent,
    KillSwitchState,
    TradingHaltedError,
)
from us_equity.risk.risk_manager import RiskManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ks(tmp_path: Path) -> KillSwitch:
    """Return a fresh KillSwitch that writes to the pytest temp directory."""
    return KillSwitch(
        state_path=tmp_path / "state.json",
        log_path=tmp_path / "log.json",
    )


def _returns(*values: float) -> pd.Series:
    return pd.Series(list(values), dtype=float)


# ---------------------------------------------------------------------------
# Trigger: single-day loss
# ---------------------------------------------------------------------------


class TestSingleDayLoss:
    def test_triggers_when_loss_exceeds_threshold(self, tmp_path):
        ks = _ks(tmp_path)
        big_loss = -(SINGLE_DAY_LOSS_THRESHOLD + 0.01)  # −4 %
        event = ks.check(_returns(0.01, 0.02, big_loss))

        assert event is not None
        assert "single_day_loss" in event.pnl_snapshot["trigger"]
        assert ks.is_active() is False
        assert ks._state == KillSwitchState.KILLED

    def test_does_not_trigger_at_threshold_boundary(self, tmp_path):
        ks = _ks(tmp_path)
        # Exactly −3 % should NOT trigger (condition is strict <)
        event = ks.check(_returns(-SINGLE_DAY_LOSS_THRESHOLD))
        assert event is None
        assert ks.is_active() is True

    def test_does_not_trigger_below_threshold(self, tmp_path):
        ks = _ks(tmp_path)
        event = ks.check(_returns(0.01, -0.02))  # −2 % loss, under threshold
        assert event is None
        assert ks.is_active() is True


# ---------------------------------------------------------------------------
# Trigger: consecutive losses
# ---------------------------------------------------------------------------


class TestConsecutiveLoss:
    def test_triggers_after_five_consecutive_losses(self, tmp_path):
        ks = _ks(tmp_path)
        # Five consecutive small losses — none breaches the 3 % single-day limit
        returns = _returns(-0.005, -0.006, -0.007, -0.008, -0.009)
        event = ks.check(returns)

        assert event is not None
        assert event.pnl_snapshot["trigger"] == "consecutive_loss"
        assert ks.is_active() is False

    def test_does_not_trigger_with_one_positive_day(self, tmp_path):
        ks = _ks(tmp_path)
        # Four losses then a small positive — consecutive streak broken
        returns = _returns(-0.005, -0.006, -0.007, -0.008, 0.001)
        event = ks.check(returns)
        assert event is None
        assert ks.is_active() is True

    def test_uses_only_the_last_n_days(self, tmp_path):
        ks = _ks(tmp_path)
        # Six values: the first is positive (breaks prior streak).
        # The last five are all negative → should trigger.
        returns = _returns(0.01, -0.005, -0.006, -0.007, -0.008, -0.009)
        event = ks.check(returns)
        assert event is not None
        assert event.pnl_snapshot["trigger"] == "consecutive_loss"


# ---------------------------------------------------------------------------
# Trigger: weekly loss
# ---------------------------------------------------------------------------


class TestWeeklyLoss:
    def test_triggers_when_weekly_loss_exceeds_threshold(self, tmp_path):
        ks = _ks(tmp_path)
        # Mix of positive and negative days so consecutive-loss doesn't fire,
        # but cumulative return is below −5 %.
        # +0.5 %, −2.5 %, +0.5 %, −2.5 %, −1.5 %  →  cumulative ≈ −5.4 %
        returns = _returns(0.005, -0.025, 0.005, -0.025, -0.015)
        cum = float((1 + returns).prod() - 1)
        assert cum < -WEEKLY_LOSS_THRESHOLD  # sanity check

        event = ks.check(returns)
        assert event is not None
        assert event.pnl_snapshot["trigger"] == "weekly_loss"
        assert ks.is_active() is False

    def test_does_not_trigger_when_weekly_loss_within_threshold(self, tmp_path):
        ks = _ks(tmp_path)
        # Mix of positive and negative days; cumulative ~−1.5 % — under the 5 % threshold
        # and not all-negative so consecutive-loss doesn't fire either.
        returns = _returns(-0.005, 0.001, -0.005, 0.001, -0.005)
        event = ks.check(returns)
        assert event is None


# ---------------------------------------------------------------------------
# Priority: single-day loss takes precedence over consecutive / weekly
# ---------------------------------------------------------------------------


class TestTriggerPriority:
    def test_single_day_reported_when_all_conditions_met(self, tmp_path):
        ks = _ks(tmp_path)
        # All five days negative, cumulative > 5 %, last day also > 3 %
        returns = _returns(-0.01, -0.01, -0.01, -0.01, -0.04)
        event = ks.check(returns)
        assert event is not None
        assert event.pnl_snapshot["trigger"] == "single_day_loss"


# ---------------------------------------------------------------------------
# Already KILLED: check is a no-op
# ---------------------------------------------------------------------------


class TestAlreadyKilled:
    def test_check_returns_existing_event_when_killed(self, tmp_path):
        ks = _ks(tmp_path)
        ks.check(_returns(-0.05))  # trigger
        first_event = ks._last_event
        assert first_event is not None

        # Subsequent check should return the same event without re-evaluating
        second_event = ks.check(_returns(0.10))  # very positive — would not trigger
        assert second_event is first_event


# ---------------------------------------------------------------------------
# Reset: success and failure
# ---------------------------------------------------------------------------


class TestReset:
    def _kill(self, ks: KillSwitch) -> None:
        ks.check(_returns(-0.05))
        assert ks.is_active() is False

    def test_reset_succeeds_with_correct_code(self, tmp_path):
        ks = _ks(tmp_path)
        self._kill(ks)

        code = datetime.now(tz=UTC).strftime("%Y-%m-%d") + "-RESET"
        result = ks.reset(code)

        assert result is True
        assert ks.is_active() is True
        assert ks._state == KillSwitchState.ACTIVE
        assert ks._last_event is None

    def test_reset_fails_with_wrong_code(self, tmp_path):
        ks = _ks(tmp_path)
        self._kill(ks)

        result = ks.reset("wrong-code")

        assert result is False
        assert ks.is_active() is False  # still KILLED

    def test_reset_fails_with_old_date_code(self, tmp_path):
        ks = _ks(tmp_path)
        self._kill(ks)

        result = ks.reset("2020-01-01-RESET")

        assert result is False
        assert ks.is_active() is False

    def test_reset_on_active_switch_returns_false(self, tmp_path):
        ks = _ks(tmp_path)  # starts ACTIVE
        code = datetime.now(tz=UTC).strftime("%Y-%m-%d") + "-RESET"
        result = ks.reset(code)
        assert result is False


# ---------------------------------------------------------------------------
# Persistence: KILLED state survives process restart
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_killed_state_persists_across_instances(self, tmp_path):
        state_path = tmp_path / "state.json"
        log_path = tmp_path / "log.json"

        # First instance: trigger kill switch
        ks1 = KillSwitch(state_path=state_path, log_path=log_path)
        ks1.check(_returns(-0.05))
        assert ks1.is_active() is False

        # Second instance: reads state from disk
        ks2 = KillSwitch(state_path=state_path, log_path=log_path)
        assert ks2.is_active() is False
        assert ks2._state == KillSwitchState.KILLED
        assert ks2._last_event is not None

    def test_active_state_is_default_when_no_file(self, tmp_path):
        ks = KillSwitch(
            state_path=tmp_path / "nonexistent.json",
            log_path=tmp_path / "log.json",
        )
        assert ks.is_active() is True

    def test_state_file_written_on_trigger(self, tmp_path):
        state_path = tmp_path / "state.json"
        ks = KillSwitch(state_path=state_path, log_path=tmp_path / "log.json")
        ks.check(_returns(-0.05))

        assert state_path.exists()
        raw = json.loads(state_path.read_text())
        assert raw["state"] == KillSwitchState.KILLED.value
        assert "last_event" in raw

    def test_log_file_appends_entries(self, tmp_path):
        log_path = tmp_path / "log.json"
        ks = KillSwitch(state_path=tmp_path / "state.json", log_path=log_path)

        # Trigger
        ks.check(_returns(-0.05))
        # Reset
        code = datetime.now(tz=UTC).strftime("%Y-%m-%d") + "-RESET"
        ks.reset(code)
        # Trigger again
        ks.check(_returns(-0.05))

        entries = json.loads(log_path.read_text())
        assert len(entries) == 3
        actions = [e["action"] for e in entries]
        assert actions == ["KILLED", "RESET", "KILLED"]

    def test_reset_clears_state_file(self, tmp_path):
        state_path = tmp_path / "state.json"
        ks = KillSwitch(state_path=state_path, log_path=tmp_path / "log.json")
        ks.check(_returns(-0.05))

        code = datetime.now(tz=UTC).strftime("%Y-%m-%d") + "-RESET"
        ks.reset(code)

        raw = json.loads(state_path.read_text())
        assert raw["state"] == KillSwitchState.ACTIVE.value
        assert "last_event" not in raw


# ---------------------------------------------------------------------------
# Event dataclass
# ---------------------------------------------------------------------------


class TestKillSwitchEvent:
    def test_to_dict_and_from_dict_roundtrip(self):
        now = datetime.now(tz=UTC)
        event = KillSwitchEvent(
            timestamp=now,
            reason="test reason",
            pnl_snapshot={"trigger": "single_day_loss", "daily_return": -0.04},
        )
        d = event.to_dict()
        restored = KillSwitchEvent.from_dict(d)

        assert restored.reason == event.reason
        assert restored.pnl_snapshot == event.pnl_snapshot
        # Timestamps are equal modulo microsecond rounding in isoformat
        assert abs((restored.timestamp - event.timestamp).total_seconds()) < 1e-3


# ---------------------------------------------------------------------------
# RiskManager integration
# ---------------------------------------------------------------------------


class TestRiskManagerIntegration:
    def test_pre_trade_check_raises_when_killed(self, tmp_path):
        ks = _ks(tmp_path)
        rm = RiskManager(db_path=":memory:", kill_switch=ks)

        big_loss = _returns(-0.05)
        with pytest.raises(TradingHaltedError) as exc_info:
            rm.pre_trade_check(daily_returns=big_loss)

        assert exc_info.value.event is not None
        assert "HALTED" in str(exc_info.value)

    def test_pre_trade_check_passes_when_active(self, tmp_path):
        ks = _ks(tmp_path)
        rm = RiskManager(db_path=":memory:", kill_switch=ks)

        # Small positive returns — no trigger
        rm.pre_trade_check(daily_returns=_returns(0.01, 0.02, 0.005))
        assert ks.is_active() is True

    def test_pre_trade_check_raises_when_already_killed_no_new_data(self, tmp_path):
        ks = _ks(tmp_path)
        # Manually trigger via check
        ks.check(_returns(-0.05))
        rm = RiskManager(db_path=":memory:", kill_switch=ks)

        with pytest.raises(TradingHaltedError):
            rm.pre_trade_check()  # no new data — should still raise

    def test_pre_trade_check_passes_after_reset(self, tmp_path):
        ks = _ks(tmp_path)
        rm = RiskManager(db_path=":memory:", kill_switch=ks)

        # Trigger
        with pytest.raises(TradingHaltedError):
            rm.pre_trade_check(daily_returns=_returns(-0.05))

        # Reset
        code = datetime.now(tz=UTC).strftime("%Y-%m-%d") + "-RESET"
        ks.reset(code)

        # Should pass now
        rm.pre_trade_check(daily_returns=_returns(0.01))
        assert ks.is_active() is True

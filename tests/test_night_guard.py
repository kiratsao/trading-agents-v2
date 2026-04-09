"""Tests for NightGuard — night session risk management.

Coverage
--------
* Guard 1: trailing stop (night low < entry - ATR×mult)
* Guard 2: reversal (night close < night open - ATR×mult)
* Guard 3: drawdown % (night close < entry × (1 - pct))
* Guard enable/disable via enabled_guards frozenset
* Edge cases: no position, None entry_price, zero ATR
"""

from __future__ import annotations

import pytest

from src.risk.night_guard import GuardResult, NightGuard, NightSession


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def guard():
    return NightGuard(guard1_atr_mult=2.0, guard2_atr_mult=2.0, guard3_pct=0.05)


@pytest.fixture
def safe_session():
    """Night session where price stays near entry — no guard should trigger."""
    return NightSession(open_price=20000.0, high=20100.0, low=19900.0, close=20050.0)


@pytest.fixture
def crash_session():
    """Night session with a significant drop."""
    return NightSession(open_price=20000.0, high=20050.0, low=19200.0, close=19300.0)


# ===========================================================================
# No position — should always return safe
# ===========================================================================


class TestNoPosition:
    def test_zero_position_returns_safe(self, guard, safe_session):
        result = guard.check(position=0, entry_price=None, atr=200.0, session=safe_session)
        assert result.should_close is False

    def test_negative_position_returns_safe(self, guard, safe_session):
        result = guard.check(position=-1, entry_price=20000.0, atr=200.0, session=safe_session)
        assert result.should_close is False

    def test_none_entry_returns_safe(self, guard, safe_session):
        result = guard.check(position=2, entry_price=None, atr=200.0, session=safe_session)
        assert result.should_close is False


# ===========================================================================
# Guard 1: Trailing stop
# ===========================================================================


class TestGuard1:
    def test_triggers_when_low_below_stop(self, guard):
        # entry=20000, ATR=200, stop=20000-2*200=19600
        # night low=19500 < 19600 → trigger
        session = NightSession(open_price=20000.0, high=20050.0, low=19500.0, close=19700.0)
        result = guard.check(position=2, entry_price=20000.0, atr=200.0, session=session)
        assert result.should_close is True
        assert "guard1" in result.reason

    def test_safe_when_low_above_stop(self, guard, safe_session):
        # entry=20000, ATR=200, stop=19600
        # night low=19900 > 19600 → safe
        result = guard.check(position=2, entry_price=20000.0, atr=200.0, session=safe_session)
        # May still be safe (guard1 won't trigger, others might not either)
        if result.should_close:
            assert "guard1" not in result.reason or result.reason == ""

    def test_exactly_at_stop_is_safe(self, guard):
        # stop = 20000 - 2*200 = 19600, low = 19600 → NOT < stop, so safe
        session = NightSession(open_price=20000.0, high=20050.0, low=19600.0, close=19800.0)
        result = guard.check(position=2, entry_price=20000.0, atr=200.0, session=session)
        # guard1 should not trigger (low == stop, not < stop)
        if result.should_close:
            assert "guard1" not in result.reason

    def test_disabled_guard1(self):
        guard = NightGuard(guard1_atr_mult=2.0, guard2_atr_mult=None, guard3_pct=None,
                          enabled_guards=frozenset({"guard2"}))
        session = NightSession(open_price=20000.0, high=20050.0, low=19000.0, close=19700.0)
        result = guard.check(position=2, entry_price=20000.0, atr=200.0, session=session)
        # guard1 disabled, guard2 disabled (mult=None), guard3 disabled
        assert result.should_close is False


# ===========================================================================
# Guard 2: Reversal
# ===========================================================================


class TestGuard2:
    def test_triggers_on_strong_reversal(self):
        # Use only guard2 enabled to isolate it
        guard = NightGuard(guard1_atr_mult=None, guard2_atr_mult=2.0, guard3_pct=None)
        # night_open=20000, ATR=200, reversal_stop=20000-2*200=19600
        # night_close=19500 < 19600 → trigger
        session = NightSession(open_price=20000.0, high=20100.0, low=19400.0, close=19500.0)
        result = guard.check(position=2, entry_price=20500.0, atr=200.0, session=session)
        assert result.should_close is True
        assert "guard2" in result.reason

    def test_safe_on_mild_decline(self, guard):
        # night_close=19800 > 19600 → safe for guard2
        session = NightSession(open_price=20000.0, high=20100.0, low=19750.0, close=19800.0)
        result = guard.check(position=2, entry_price=19500.0, atr=200.0, session=session)
        if result.should_close:
            assert "guard2" not in result.reason


# ===========================================================================
# Guard 3: Drawdown percentage
# ===========================================================================


class TestGuard3:
    def test_triggers_on_large_drawdown(self):
        # Use only guard3 enabled to isolate it
        guard = NightGuard(guard1_atr_mult=None, guard2_atr_mult=None, guard3_pct=0.05)
        # entry=20000, pct=0.05, drawdown_stop=20000*0.95=19000
        # night_close=18900 < 19000 → trigger
        session = NightSession(open_price=19500.0, high=19600.0, low=18800.0, close=18900.0)
        result = guard.check(position=2, entry_price=20000.0, atr=200.0, session=session)
        assert result.should_close is True
        assert "guard3" in result.reason

    def test_safe_within_drawdown_limit(self, guard):
        # night_close=19200 > 19000 → safe for guard3
        session = NightSession(open_price=19500.0, high=19600.0, low=19100.0, close=19200.0)
        # But guard1 might trigger: entry=20000, stop=20000-400=19600, low=19100 < 19600
        result = guard.check(position=2, entry_price=20000.0, atr=200.0, session=session)
        if result.should_close:
            assert "guard3" not in result.reason

    def test_guard3_disabled_by_default(self):
        guard = NightGuard(guard3_pct=None)
        session = NightSession(open_price=20000.0, high=20100.0, low=18000.0, close=18500.0)
        result = guard.check(position=2, entry_price=20000.0, atr=200.0, session=session)
        if result.should_close:
            assert "guard3" not in result.reason


# ===========================================================================
# Guard enable/disable
# ===========================================================================


class TestGuardEnabling:
    def test_only_guard2_enabled(self):
        guard = NightGuard(
            guard1_atr_mult=2.0, guard2_atr_mult=2.0, guard3_pct=0.05,
            enabled_guards=frozenset({"guard2"}),
        )
        # This would trigger guard1 (low=19000 < stop=19600) but guard1 is disabled
        session = NightSession(open_price=20000.0, high=20100.0, low=19000.0, close=19800.0)
        result = guard.check(position=2, entry_price=20000.0, atr=200.0, session=session)
        # guard1 disabled, guard2 safe (close=19800 > 19600), guard3 disabled
        assert result.should_close is False

    def test_all_guards_disabled_returns_safe(self):
        guard = NightGuard(
            guard1_atr_mult=2.0, guard2_atr_mult=2.0, guard3_pct=0.05,
            enabled_guards=frozenset(),
        )
        session = NightSession(open_price=20000.0, high=20100.0, low=18000.0, close=18500.0)
        result = guard.check(position=2, entry_price=20000.0, atr=200.0, session=session)
        assert result.should_close is False


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_zero_atr(self, guard, safe_session):
        # ATR=0 → stop = entry - 0 = entry; low=19900 < 20000 → trigger guard1
        result = guard.check(position=2, entry_price=20000.0, atr=0.0, session=safe_session)
        assert result.should_close is True

    def test_very_large_atr_keeps_safe(self, guard, crash_session):
        # ATR=10000 → stop = 20000 - 20000 = 0; low=19200 > 0 → safe for guard1
        result = guard.check(position=2, entry_price=20000.0, atr=10000.0, session=crash_session)
        assert result.should_close is False

    def test_guard_result_dataclass(self):
        r = GuardResult(should_close=True, reason="test")
        assert r.should_close is True
        assert r.reason == "test"

    def test_night_session_frozen(self):
        s = NightSession(open_price=100.0, high=110.0, low=90.0, close=105.0)
        with pytest.raises(AttributeError):
            s.open_price = 200.0

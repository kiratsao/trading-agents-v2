"""P0 regressions for the 2026-07-28 incident: live today-bar dual validation.

The Shioaji kbars feed died at 09:0x and the truncated last bar (43,175) was
treated as the day close; 43,175 > trailing stop 42,673 → silent HOLD while the
real close was 41,608 (−1,065 through the stop). Five open contracts rode the
crash one extra day. These tests pin the fixes:

  * completeness — last kbar must fall in 13:40–13:45 (settlement 13:25–13:30)
  * spot gate    — |close − ^TWII| ≤ 500 (night-proof truth)
  * exit fail-SAFE  — invalid bar + position → proxy (spot+median basis)
    evaluation; a clear stop break SELLs, never a silent HOLD
  * entry fail-CLOSED — invalid bar → buy/add cancelled
  * ambiguity / no spot → 🔴 human alert, never silence
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd

from src.scheduler.orchestrator import V2bOrchestrator, _validate_today_bar
from src.state.state_manager import StateManager, TradingState
from src.strategy.v2b_engine import Signal, V2bEngine
from src.utils.tw_time import today_taipei
from tests.fakes import write_synthetic_parquet

_D728 = date(2026, 7, 28)
_SPOT_728 = 41_603.0


def _bar(close: float, last_ts: str | None = "2026-07-28T13:44:00+08:00") -> dict:
    b = {"open": 44_000.0, "high": 44_500.0, "low": 41_500.0,
         "close": close, "volume": 190_000}
    if last_ts is not None:
        b["last_ts"] = last_ts
    return b


def _df(n: int = 30, close: float = 45_000.0) -> pd.DataFrame:
    dates = pd.bdate_range(end="2026-07-27", periods=n)
    return pd.DataFrame(
        {"open": close - 10, "high": close + 60, "low": close - 60,
         "close": [close] * n, "volume": [100_000] * n},
        index=dates,
    )


def _orch(state: TradingState, sig: Signal, live: bool = True):
    strategy = V2bEngine(product="MXF", ema_fast=30, ema_slow=100,
                         confirm_days=2, adx_threshold=25)
    strategy.generate_signal = lambda *a, **kw: sig
    state_mgr = MagicMock(spec=StateManager)
    state_mgr.load.return_value = state
    notify = MagicMock()
    orch = V2bOrchestrator(
        strategy=strategy, state_mgr=state_mgr, notify_fn=notify,
        execution_timing="night_open", live=live,
    )
    return orch, state_mgr, notify


def _notes(notify) -> list[str]:
    return [c.args[0] for c in notify.call_args_list]


# ── _validate_today_bar unit coverage ───────────────────────────────────────
class TestValidateTodayBar:
    def test_complete_kbars_near_spot_is_validated(self):
        with patch("src.data.spot_ref.fetch_spot_close", return_value=_SPOT_728):
            meta = _validate_today_bar(_bar(41_608.0), _D728, source="kbars")
        assert meta["complete"] and meta["spot_ok"] and meta["validated"]
        assert meta["usable_for_exit"]

    def test_truncated_0905_is_rejected(self):
        """The exact 7/28 failure: last kbar 09:05 → close must not be trusted."""
        with patch("src.data.spot_ref.fetch_spot_close", return_value=_SPOT_728):
            meta = _validate_today_bar(
                _bar(43_175.0, last_ts="2026-07-28T09:05:00+08:00"),
                _D728, source="kbars",
            )
        assert not meta["complete"] and not meta["validated"]
        assert not meta["usable_for_exit"]
        assert "截斷" in meta["reject"] and "09:05" in meta["reject"]

    def test_spot_gate_blocks_728_value_even_if_complete(self):
        """|43,175 − 41,603| = 1,572 > 500 → rejected regardless of kbar time."""
        with patch("src.data.spot_ref.fetch_spot_close", return_value=_SPOT_728):
            meta = _validate_today_bar(_bar(43_175.0), _D728, source="kbars")
        assert meta["complete"] and meta["spot_ok"] is False
        assert not meta["validated"] and not meta["usable_for_exit"]

    def test_snapshot_passing_spot_is_exit_only(self):
        with patch("src.data.spot_ref.fetch_spot_close", return_value=_SPOT_728):
            meta = _validate_today_bar(
                _bar(41_650.0, last_ts=None), _D728, source="snapshot",
            )
        assert not meta["validated"]          # never a 台指收盤
        assert meta["usable_for_exit"]        # good enough to evaluate exits

    def test_snapshot_failing_spot_is_unusable(self):
        """7/27-style snapshot on a degraded feed (43,906 vs spot ~41.6k)."""
        with patch("src.data.spot_ref.fetch_spot_close", return_value=_SPOT_728):
            meta = _validate_today_bar(
                _bar(43_906.0, last_ts=None), _D728, source="snapshot",
            )
        assert not meta["validated"] and not meta["usable_for_exit"]

    def test_complete_kbars_without_spot_is_exit_only(self):
        with patch("src.data.spot_ref.fetch_spot_close", return_value=None):
            meta = _validate_today_bar(_bar(41_608.0), _D728, source="kbars")
        assert not meta["validated"]          # spot gate could not run
        assert meta["usable_for_exit"]        # complete kbars still exit-worthy


# ── proxy exit evaluation ───────────────────────────────────────────────────
class TestProxyExitCheck:
    def _spots_for(self, df: pd.DataFrame, basis: float = 5.0) -> dict:
        return {ts.date(): float(df.loc[ts, "close"]) - basis for ts in df.index}

    def test_clear_break_sells_728_numbers(self):
        """spot 41,603 + basis ≈ +5 → proxy 41,608 < stop 42,673 − 150 → sell."""
        state = TradingState(position=5, contracts=5, entry_price=44_957.0,
                             highest_high=45_020.0, equity=1_200_938.0)
        orch, _, _ = _orch(state, Signal("hold", 5, "holding position"))
        df = _df()
        with (
            patch("src.data.spot_ref.fetch_spot_close", return_value=_SPOT_728),
            patch("src.data.spot_ref.fetch_spot_range",
                  return_value=self._spots_for(df)),
            patch("src.scheduler.orchestrator.today_taipei", return_value=_D728),
        ):
            verdict, detail = orch._proxy_exit_check(df, state, {"atr": 1_173.5})
        assert verdict == "sell"
        assert "明確跌穿" in detail

    def test_inside_buffer_is_ambiguous(self):
        state = TradingState(position=5, contracts=5, entry_price=44_957.0,
                             highest_high=45_020.0)
        orch, _, _ = _orch(state, Signal("hold", 5, "holding position"))
        df = _df()
        # spot chosen so proxy lands ~68pt above the 42,673 stop (< 150 buffer)
        with (
            patch("src.data.spot_ref.fetch_spot_close", return_value=42_600.0),
            patch("src.data.spot_ref.fetch_spot_range",
                  return_value=self._spots_for(df)),
            patch("src.scheduler.orchestrator.today_taipei", return_value=_D728),
        ):
            verdict, _ = orch._proxy_exit_check(df, state, {"atr": 1_173.5})
        assert verdict == "ambiguous"

    def test_no_spot_is_ambiguous(self):
        state = TradingState(position=5, contracts=5, highest_high=45_020.0)
        orch, _, _ = _orch(state, Signal("hold", 5, "holding position"))
        with (
            patch("src.data.spot_ref.fetch_spot_close", return_value=None),
            patch("src.scheduler.orchestrator.today_taipei", return_value=_D728),
        ):
            verdict, detail = orch._proxy_exit_check(_df(), state, {"atr": 1_173.5})
        assert verdict == "ambiguous"
        assert "spot" in detail


# ── run_signal end-to-end: the pending intent must be the SAFE action ───────
class TestRunSignalPolicy:
    _META_INVALID = {
        "source": "kbars", "validated": False, "usable_for_exit": False,
        "reject": "kbars截斷(末根 09:05)", "desc": "kbars(末根 09:05) 🔴",
        "last_kbar": "09:05", "spot_dev": None,
    }

    def _run(self, orch, df):
        with (
            patch.object(orch, "_load_data", return_value=df),
            patch.object(orch, "_check_data_freshness", return_value=(True, "")),
            patch("src.scheduler.orchestrator.today_taipei", return_value=_D728),
        ):
            return orch.run_signal(broker=None)

    def test_truncated_bar_with_position_never_silent_holds(self):
        """THE 7/28 regression: truncated bar + proxy clearly below stop →
        pending close, not a silent HOLD."""
        state = TradingState(position=5, contracts=5, entry_price=44_957.0,
                             highest_high=45_020.0, equity=1_200_938.0)
        orch, _, notify = _orch(state, Signal("hold", 5, "holding position"))
        orch.today_bar_meta = dict(self._META_INVALID)
        with patch.object(
            orch, "_proxy_exit_check",
            return_value=("sell", "proxy=41,608 vs stop 42,673 → 明確跌穿"),
        ):
            res = self._run(orch, _df())
        assert res["action"] == "close"
        assert state.pending_action == "close"
        assert "proxy trailing stop" in state.pending_reason

    def test_ambiguous_proxy_alerts_human_loudly(self):
        state = TradingState(position=5, contracts=5, entry_price=44_957.0,
                             highest_high=45_020.0)
        orch, _, notify = _orch(state, Signal("hold", 5, "holding position"))
        orch.today_bar_meta = dict(self._META_INVALID)
        with patch.object(
            orch, "_proxy_exit_check", return_value=("ambiguous", "spot 不可得"),
        ):
            res = self._run(orch, _df())
        assert res["action"] == "hold"
        assert any("需人工確認持倉" in n for n in _notes(notify))

    def test_fail_closed_blocks_entry_on_invalid_bar(self):
        state = TradingState(position=0, equity=650_708.0)
        orch, _, notify = _orch(state, Signal("buy", 4, "golden cross"))
        orch.today_bar_meta = dict(self._META_INVALID)
        res = self._run(orch, _df())
        assert res["action"] == "hold"
        assert state.pending_action == "hold"
        assert any("進場封鎖" in n for n in _notes(notify))

    def test_validated_bar_keeps_normal_flow(self):
        state = TradingState(position=0, equity=650_708.0)
        orch, _, notify = _orch(state, Signal("buy", 4, "golden cross"))
        orch.today_bar_meta = {
            "source": "kbars", "validated": True, "usable_for_exit": True,
            "reject": None, "desc": "kbars(末根 13:44) spot誤差+5pt ✅",
            "last_kbar": "13:44", "spot_dev": 5.0,
        }
        res = self._run(orch, _df())
        assert res["action"] == "buy"
        assert state.pending_action == "buy"
        assert not any("進場封鎖" in n for n in _notes(notify))


# ── _load_data: an invalid bar must never enter the decision df ─────────────
class TestLoadDataRejection:
    def _orch_with_parquet(self, tmp_path, live=True):
        pq = tmp_path / "MXF.parquet"
        write_synthetic_parquet(pq, n_bars=120, end=date(2026, 7, 27))
        strategy = V2bEngine(product="MXF", ema_fast=30, ema_slow=100)
        orch = V2bOrchestrator(
            strategy=strategy, state_mgr=MagicMock(spec=StateManager),
            notify_fn=MagicMock(), live=live, data_path=pq,
        )
        return orch

    def test_invalid_bar_not_appended(self, tmp_path):
        orch = self._orch_with_parquet(tmp_path)
        bad = {"date": "2026-07-28", "open": 44_000.0, "high": 44_500.0,
               "low": 43_000.0, "close": 43_175.0, "volume": 50_000,
               "_meta": {"source": "kbars", "validated": False,
                          "usable_for_exit": False, "reject": "kbars截斷",
                          "desc": "kbars(末根 09:05) 🔴"}}
        with patch("src.scheduler.orchestrator._fetch_today_bar_shioaji",
                   return_value=bad):
            df = orch._load_data(broker=None)
        assert pd.Timestamp("2026-07-28") not in df.index
        assert orch.today_bar_meta["validated"] is False

    def test_validated_bar_appended(self, tmp_path):
        orch = self._orch_with_parquet(tmp_path)
        good = {"date": "2026-07-28", "open": 41_700.0, "high": 41_900.0,
                "low": 41_400.0, "close": 41_608.0, "volume": 190_000,
                "_meta": {"source": "kbars", "validated": True,
                           "usable_for_exit": True, "reject": None,
                           "desc": "kbars(末根 13:44) ✅"}}
        with patch("src.scheduler.orchestrator._fetch_today_bar_shioaji",
                   return_value=good):
            df = orch._load_data(broker=None)
        assert pd.Timestamp("2026-07-28") in df.index
        assert float(df.loc[pd.Timestamp("2026-07-28"), "close"]) == 41_608.0


# ── equity sanity + LINE labeling + entry_date ──────────────────────────────
class TestEquitySanity:
    def test_stale_margin_flagged_and_conservative_used(self):
        """7/28 numbers: Δclose −1,692 × 5口 × 50 → expected 777,938; a broker
        read of 1,169,688 implies a +1,567pt-stale mark → flag + use lower."""
        state = TradingState(position=5, contracts=5, equity=1_200_938.0)
        orch, _, notify = _orch(state, Signal("hold", 5, "holding"))
        orch.today_bar_meta = {"validated": True}
        df = pd.DataFrame(
            {"close": [43_300.0, 41_608.0]},
            index=pd.to_datetime(["2026-07-27", "2026-07-28"]),
        )
        eq, note = orch._sanity_check_equity(
            1_169_688.0, "即時", 1_200_938.0, state, df,
        )
        assert eq == 777_938.0
        assert note is not None and "stale" in note
        assert any("stale" in n for n in _notes(notify))

    def test_plausible_read_passes_through(self):
        state = TradingState(position=5, contracts=5, equity=1_200_938.0)
        orch, _, _ = _orch(state, Signal("hold", 5, "holding"))
        orch.today_bar_meta = {"validated": True}
        df = pd.DataFrame(
            {"close": [43_300.0, 41_608.0]},
            index=pd.to_datetime(["2026-07-27", "2026-07-28"]),
        )
        eq, note = orch._sanity_check_equity(
            780_000.0, "即時", 1_200_938.0, state, df,
        )
        assert eq == 780_000.0 and note is None


class TestLineLabels:
    def test_unvalidated_bar_labeled_as_reference_not_close(self):
        state = TradingState(position=0)
        orch, _, _ = _orch(state, Signal("hold", 0, "x"))
        msg = orch._build_decision_message(
            sig=Signal("hold", 0, "x"), state=state,
            indicators={"close": 43_175.0}, action_contracts=0,
            closed_contracts=0, equity=650_708.0, tsmc_signal=None,
            bar_meta={"validated": False, "desc": "kbars(末根 09:05) 🔴"},
        )
        assert "參考價(未驗證): 43,175" in msg
        assert "台指收盤" not in msg
        assert "Bar來源: kbars(末根 09:05) 🔴" in msg

    def test_validated_bar_keeps_official_close_label(self):
        state = TradingState(position=0)
        orch, _, _ = _orch(state, Signal("hold", 0, "x"))
        msg = orch._build_decision_message(
            sig=Signal("hold", 0, "x"), state=state,
            indicators={"close": 41_608.0}, action_contracts=0,
            closed_contracts=0, equity=650_708.0, tsmc_signal=None,
            bar_meta={"validated": True, "desc": "kbars(末根 13:44) ✅"},
        )
        assert "台指收盤: 41,608" in msg


class TestEntryDateWritten:
    def test_buy_writes_entry_date_and_close_clears_it(self):
        """7/22 bug: the buy path never wrote state.entry_date (it still showed
        the 6/15 entry). Paper-mode run_daily buy must stamp today."""
        state = TradingState(position=0, equity=650_708.0)
        orch, _, _ = _orch(state, Signal("buy", 2, "golden cross"), live=False)
        with patch.object(orch, "_load_data", return_value=_df()):
            orch.run_daily(broker=None)
        assert state.position == 2
        assert state.entry_date == today_taipei().isoformat()

        sell_state = TradingState(position=2, contracts=2, entry_price=45_000.0,
                                  entry_date="2026-07-22", equity=650_708.0)
        orch2, _, _ = _orch(sell_state, Signal("close", 2, "trailing stop"),
                            live=False)
        with patch.object(orch2, "_load_data", return_value=_df()):
            orch2.run_daily(broker=None)
        assert sell_state.position == 0
        assert sell_state.entry_date is None


class TestProvenanceGateLive:
    """7/22-class small-gap night bar on the LIVE path: close bit-equals
    today's TAIFEX 盤後 row (= last night) while sitting INSIDE the spot band —
    completeness and spot both pass, provenance must still reject (entries
    fail-closed, exit falls to the proxy path)."""

    _D722 = date(2026, 7, 22)

    @staticmethod
    def _day_frame(close: float) -> pd.DataFrame:
        return pd.DataFrame(
            [{"open": close, "high": close, "low": close, "close": close,
              "volume": 150_000}],
            index=pd.DatetimeIndex([pd.Timestamp("2026-07-22")], name="date"),
        )

    def _setup(self, monkeypatch):
        from src.data import daily_updater

        monkeypatch.setattr("src.data.spot_ref.fetch_spot_close",
                            lambda d, **k: 44_826.0)
        monkeypatch.setattr(daily_updater, "_taifex_night_close",
                            lambda d: 44_957.0)
        monkeypatch.setattr(daily_updater, "_taifex_day_bar",
                            lambda d: self._day_frame(44_627.0))

    def test_night_value_rejected_despite_spot_pass(self, monkeypatch):
        self._setup(monkeypatch)
        bar = _bar(44_957.0, last_ts="2026-07-22T13:44:00+08:00")
        meta = _validate_today_bar(bar, self._D722, source="kbars")
        assert meta["night_hit"] is True
        assert meta["validated"] is False
        assert meta["usable_for_exit"] is False
        assert "provenance" in meta["reject"]
        assert "盤後值" in meta["desc"]

    def test_day_value_passes(self, monkeypatch):
        self._setup(monkeypatch)
        bar = _bar(44_627.0, last_ts="2026-07-22T13:44:00+08:00")
        meta = _validate_today_bar(bar, self._D722, source="kbars")
        assert meta["night_hit"] is False
        assert meta["validated"] is True

    def test_provenance_unavailable_degrades_open(self, monkeypatch):
        # conftest nulls _taifex_night_close/_taifex_day_bar → provenance
        # degrades False; behavior identical to pre-gate (spot+completeness).
        monkeypatch.setattr("src.data.spot_ref.fetch_spot_close",
                            lambda d, **k: 44_826.0)
        bar = _bar(44_957.0, last_ts="2026-07-22T13:44:00+08:00")
        meta = _validate_today_bar(bar, self._D722, source="kbars")
        assert meta["night_hit"] is False
        assert meta["validated"] is True

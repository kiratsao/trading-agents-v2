"""V2b 排程整合 Orchestrator。

執行模式
--------
next_open (預設)：
    14:30 一次完成：信號計算 → 下單 → LINE 通知

night_open：
    14:30 run_signal()  → 信號計算 → 暫存 pending intent → LINE 決策通知
    15:05 run_execution() → 讀取 pending → 夜盤下單 → LINE 執行通知
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.signals.fetcher import fetch_prices
from src.signals.tsmc_tracker import TsmcSignal, compute_signal
from src.state.state_manager import StateManager, TradingState
from src.strategy.v2b_engine import Signal, V2bEngine
from src.utils.tw_time import now_taipei, today_taipei

logger = logging.getLogger(__name__)

_REAL_DATA = Path("data/MXF_Daily_Clean_2020_to_now.parquet")
# Fallback paths in priority order
_REAL_DATA_FALLBACKS = [
    Path("data/MXF_Daily_Clean_2020_to_now.parquet"),
    Path.home() / "trading-agents-v2" / "data" / "MXF_Daily_Clean_2020_to_now.parquet",
]
_TAIEX_WEIGHT_LEVEL = 22_000

# ── Live today-bar validation (2026-07-28 incident) ─────────────────────────
# A truncated kbars feed (last bar 09:0x) reported close=43,175 vs the real
# 41,608 day close; 43,175 > trailing stop 42,673 → silent HOLD instead of a
# stop-out. Every live decision bar must now pass (a) kbar completeness and
# (b) the night-proof spot gate before it may be treated as the day close.
_PROXY_AMBIG_BUFFER = 150.0  # pts — |proxy − stop| below this = ambiguous → human
_BASIS_MIN_PAIRS = 5         # min (fut close − spot) samples to trust the basis
_BASIS_LOOKBACK_BARS = 25    # parquet bars scanned for the rolling basis
_EQUITY_SANITY_PTS = 600.0   # implied-mark deviation beyond this = stale margin


class V2bOrchestrator:
    """Daily run orchestrator for V2b strategy.

    Parameters
    ----------
    strategy :
        V2bEngine instance.
    state_mgr :
        StateManager for persistence.
    notify_fn :
        Callable(str) for LINE/Slack notifications.
    enable_tsmc_signal :
        Whether to fetch overnight TSM ADR/SOX signal.
    data_path :
        Path to daily OHLCV parquet.
    decision_time :
        HH:MM string shown in the notification header (default "14:30").
    execution_timing :
        "next_open" or "night_open".
    """

    def __init__(
        self,
        strategy: V2bEngine,
        state_mgr: StateManager,
        notify_fn: Any = None,
        enable_tsmc_signal: bool = False,
        data_path: str | Path = _REAL_DATA,
        decision_time: str = "14:30",
        execution_timing: str = "next_open",
        live: bool = False,
    ) -> None:
        if execution_timing not in ("next_open", "night_open"):
            raise ValueError(
                f"execution_timing must be 'next_open' or 'night_open', got {execution_timing!r}"
            )
        self.strategy = strategy
        self.state_mgr = state_mgr
        self.notify_fn = notify_fn or (lambda msg: None)
        self.enable_tsmc_signal = enable_tsmc_signal
        self.data_path = Path(data_path)
        self.decision_time = decision_time
        self.execution_timing = execution_timing
        self.live = live

    # ------------------------------------------------------------------
    # Public: broker reconcile (state 自動同步)
    # ------------------------------------------------------------------

    def reconcile_state_with_broker(self, broker, *, sleep=None) -> bool:
        """Adopt the broker's actual position when local state disagrees.

        Called at daemon startup and before every 14:30 signal so manual App
        operations (open/close outside the daemon) and exchange-side events
        (cash settlement) no longer require hand-editing the state JSON.
        Broker book is the source of truth. Returns True when state changed.
        """
        if broker is None:
            return False
        import time as _time

        _sleep = sleep or _time.sleep
        state = self.state_mgr.load()
        product = self.strategy.product
        actual = _read_broker_long(broker, product)
        if actual is None:
            self.notify_fn(
                f"⚠️ state/broker 對帳失敗: 無法讀取 broker 部位 "
                f"(state={state.position}口)，state 未變更"
            )
            return False
        if actual == state.position:
            logger.info(
                "reconcile: state matches broker (%d口) — no change", actual
            )
            return False

        if actual == 0 and state.position > 0:
            # Guard against a transient empty read wiping a real position:
            # require a second confirming read before adopting 0.
            _sleep(3)
            confirm = _read_broker_long(broker, product)
            if confirm != 0:
                self.notify_fn(
                    f"⚠️ broker 部位讀取不穩定（0 → {confirm}），state 未變更，"
                    f"請人工核對永豐 App"
                )
                return False

        old_pos, old_entry = state.position, state.entry_price
        if actual == 0:
            state.position = 0
            state.contracts = 0
            state.entry_price = None
            state.highest_high = None
            state.pyramided = False
            desc = f"{old_pos}口 → 空倉（App 手動平倉或已結算）"
        else:
            avg = _query_broker_avg_price(broker, product, actual)
            if avg is None and state.position == 0:
                # Fresh manual position but no readable cost basis — cannot
                # seed entry_price/trailing stop safely. Leave state alone
                # and ask for a manual sync (scripts/sync_state.py).
                self.notify_fn(
                    f"⚠️ broker 有 {actual}口 但 state 空倉，且無法讀取均價 — "
                    f"state 未變更，請跑 scripts/sync_state.py 手動同步"
                )
                return False
            state.position = actual
            state.contracts = actual
            if avg is not None and avg > 0:
                state.entry_price = avg
            if state.highest_high is None and state.entry_price is not None:
                state.highest_high = state.entry_price
            if old_pos == 0:
                state.entry_date = today_taipei().isoformat()
                state.pyramided = False
            entry_str = (
                f"{state.entry_price:,.0f}" if state.entry_price else "未知"
            )
            desc = f"{old_pos}口 → {actual}口 @ {entry_str}"

        self.state_mgr.save(state)
        msg = (
            f"⚙️ state 已自動同步 broker: {desc}\n"
            f"(原 state: {old_pos}口 @ {old_entry or 0:,.0f}；"
            f"若此同步有誤請立即人工核對)"
        )
        logger.warning(msg)
        # ⚙️ prefix is not alert-class → never deduped; every actual state
        # change stays visible even if identical to a recent one.
        self.notify_fn(msg)
        return True

    # ------------------------------------------------------------------
    # Public: single-phase (next_open)
    # ------------------------------------------------------------------

    def run_daily(self, broker=None) -> dict:
        """Run one full daily cycle.

        Returns a summary dict with action, reason, and optional tsmc info.
        """
        tsmc_signal = self._fetch_tsmc_signal() if self.enable_tsmc_signal else None
        df = self._load_data(broker=broker)
        if df is None or len(df) < 1:
            logger.error("No market data available.")
            return {"action": "error", "reason": "no data"}

        state = self.state_mgr.load()
        equity, equity_src = _query_live_equity(broker, state.equity)
        display_ind = self._compute_display_indicators(df, state)

        sig = self.strategy.generate_signal(
            data=df,
            current_position=state.position,
            entry_price=state.entry_price,
            equity=equity,
            highest_high=state.highest_high,
            contracts=state.contracts,
            tsmc_signal=tsmc_signal,
        )

        # Live today-bar policy (entry fail-closed / exit fail-safe) — same
        # rule as run_signal so same_day_close mode is equally protected.
        sig, _bar_lines = self._apply_today_bar_policy(sig, state, df, display_ind)

        result: dict = {
            "action": sig.action,
            "contracts": sig.contracts,
            "reason": sig.reason,
        }
        if tsmc_signal:
            result["tsmc"] = str(tsmc_signal)

        _action_contracts = 0
        _closed_contracts = 0
        COST_PER_SIDE, TICK_VALUE = _load_execution_constants()

        if sig.action == "buy" and state.position == 0:
            filled_n = sig.contracts
            if broker is not None:
                order = broker.place_order("MXF", "Buy", sig.contracts)
                result["order_id"] = order.get("order_id")
                exec_price = order.get("fill_price", float(df["close"].iloc[-1]))
                # Source of truth = broker actual, NOT the requested quantity.
                actual = _sync_position_from_broker(
                    broker, "MXF", sig.contracts, self.notify_fn,
                )
                filled_n = actual if actual is not None else 0
            else:
                exec_price = float(df["close"].iloc[-1])
            if filled_n > 0:
                state.equity -= COST_PER_SIDE * filled_n
                state.position = filled_n
                state.entry_price = exec_price
                state.contracts = filled_n
                state.highest_high = exec_price
                state.entry_date = today_taipei().isoformat()
                result["entry_price"] = exec_price
                _action_contracts = filled_n
            else:
                # IOC did not fill (or broker unreadable) → stay flat, no phantom
                # entry. _sync_position_from_broker already alerted on a None read.
                msg = f"🔴 進場未成交: Buy {sig.contracts}口 IOC 未成交，維持空倉"
                logger.warning(msg)
                self.notify_fn(msg)
                result["filled"] = 0

        elif sig.action in ("close", "sell") and state.position > 0:
            closed_n = state.position
            is_settlement = "settlement" in sig.reason
            sell_ok = True
            already_flat = False
            if broker is not None:
                # Pre-sell guard — see run_execution: never sell what the
                # broker no longer holds (settled / manually closed).
                broker_actual = _read_broker_long(broker, "MXF")
                if broker_actual == 0:
                    msg = (
                        f"⚠️ 平倉時 broker 已無多單（手動平倉或已結算）— 不下單，"
                        f"state {closed_n}口 已同步為空倉"
                    )
                    logger.warning(msg)
                    self.notify_fn(msg)
                    state.position = 0
                    state.entry_price = None
                    state.contracts = 0
                    state.highest_high = None
                    state.pyramided = False
                    state.entry_date = None
                    sell_ok = False
                    already_flat = True
                    result["already_flat"] = True
                elif broker_actual is not None and 0 < broker_actual < closed_n:
                    self.notify_fn(
                        f"⚠️ broker 只持有 {broker_actual}口（state {closed_n}口）— "
                        f"只平 {broker_actual}口"
                    )
                    closed_n = broker_actual
                if not already_flat:
                    try:
                        order = broker.place_order("MXF", "Sell", closed_n)
                    except Exception as exc:
                        logger.error("Sell order failed: %s", exc)
                        order = {"order_id": "FAILED", "status": "Failed"}
                    result["order_id"] = order.get("order_id")
                    sell_status = order.get("status", "")
                    if sell_status in ("Failed", "Cancelled", "Inactive"):
                        sell_ok = False
                        msg = f"🔴 平倉失敗: Sell {closed_n}口 status={sell_status}"
                        logger.error(msg)
                        self.notify_fn(msg)
                    else:
                        exec_price = order.get("fill_price", float(df["close"].iloc[-1]))
                        _reconcile_position(broker, 0, self.notify_fn)
            else:
                exec_price = float(df["close"].iloc[-1])

            if sell_ok:
                pnl_pts = exec_price - (state.entry_price or 0.0)
                round_trip = COST_PER_SIDE * 2
                pnl_twd = pnl_pts * closed_n * TICK_VALUE - round_trip * closed_n
                state.equity += pnl_twd
                result["exit_price"] = exec_price
                result["pnl_twd"] = pnl_twd
                _closed_contracts = closed_n
                # Reset position state — only when the close actually filled.
                # A rejected Sell means the broker still holds the position;
                # zeroing here would desync state and risk a double position.
                state.position = 0
                state.entry_price = None
                state.contracts = 0
                state.highest_high = None
                state.pyramided = False
                state.entry_date = None

            # Settlement rollover: re-check entry immediately
            # Only proceed if sell was successful
            if is_settlement and sell_ok:
                re_sig = self.strategy.generate_signal(
                    data=df,
                    current_position=0,
                    entry_price=None,
                    equity=state.equity,
                    highest_high=None,
                    contracts=0,
                    tsmc_signal=tsmc_signal,
                )
                logger.info(
                    "Settlement rollover: equity=%.0f → signal=%s contracts=%d reason=%s",
                    state.equity, re_sig.action, re_sig.contracts, re_sig.reason,
                )
                if re_sig.action == "buy":
                    buy_n = re_sig.contracts
                    if broker is not None:
                        try:
                            buy_order = broker.place_order("MXF", "Buy", buy_n)
                        except Exception as exc:
                            logger.error("Rollover buy failed: %s", exc)
                            buy_order = {"order_id": "FAILED", "status": "Failed"}
                        buy_status = buy_order.get("status", "")
                        if buy_status in ("Failed", "Cancelled", "Inactive"):
                            msg = f"🔴 結算日轉倉買單失敗: Buy {buy_n}口 status={buy_status}"
                            logger.error(msg)
                            self.notify_fn(msg)
                            result["rollover"] = False
                            result["rollover_reason"] = f"buy order {buy_status}"
                        else:
                            buy_price = buy_order.get("fill_price", exec_price)
                            # Broker actual is source of truth (rollover from flat).
                            actual = _sync_position_from_broker(
                                broker, "MXF", buy_n, self.notify_fn,
                            )
                            filled_n = actual if actual is not None else 0
                            if filled_n > 0:
                                state.equity -= COST_PER_SIDE * filled_n
                                state.position = filled_n
                                state.entry_price = buy_price
                                state.contracts = filled_n
                                state.highest_high = buy_price
                                state.entry_date = today_taipei().isoformat()
                                _action_contracts = filled_n
                                result["rollover"] = True
                                result["rollover_contracts"] = filled_n
                            else:
                                msg = (
                                    f"🔴 結算日轉倉買單未成交: Buy {buy_n}口 IOC 未成交，"
                                    f"維持空倉"
                                )
                                logger.warning(msg)
                                self.notify_fn(msg)
                                result["rollover"] = False
                                result["rollover_reason"] = "rollover buy IOC 未成交"
                    else:
                        buy_price = exec_price
                        state.equity -= COST_PER_SIDE * buy_n
                        state.position = buy_n
                        state.entry_price = buy_price
                        state.contracts = buy_n
                        state.highest_high = buy_price
                        state.entry_date = today_taipei().isoformat()
                        _action_contracts = buy_n
                        result["rollover"] = True
                        result["rollover_contracts"] = buy_n
                else:
                    result["rollover"] = False
                    result["rollover_reason"] = re_sig.reason
            elif is_settlement and not sell_ok:
                result["rollover"] = False
                result["rollover_reason"] = "sell order failed — rollover aborted"

        elif sig.action == "add" and state.position > 0:
            add_n = sig.contracts
            old_n = state.position
            if broker is not None:
                order = broker.place_order("MXF", "Buy", add_n)
                result["order_id"] = order.get("order_id")
                exec_price = order.get("fill_price", float(df["close"].iloc[-1]))
                # Source of truth = broker actual. The 2026-06-01 bug was
                # state.position += add_n regardless of fill: a 15-lot IOC that
                # never filled pushed local 30→45 while broker stayed 20.
                actual = _sync_position_from_broker(
                    broker, "MXF", old_n + add_n, self.notify_fn,
                )
                filled = max(0, actual - old_n) if actual is not None else 0
                if filled > 0:
                    # Recompute cost basis with the ACTUAL filled qty BEFORE
                    # mutating state.position (helper reads old_n = state.position).
                    new_entry, entry_src = _reconcile_add_entry_price(
                        broker, "MXF", state, filled, exec_price,
                    )
                    state.equity -= COST_PER_SIDE * filled
                    state.position = actual
                    state.contracts = actual
                    state.pyramided = True
                    result["add_contracts"] = filled
                    result["entry_price"] = new_entry
                    result["entry_price_source"] = entry_src
                    _action_contracts = filled
                else:
                    # IOC add did not fill (or broker unreadable) → keep old_n.
                    msg = (
                        f"🔴 加碼未成交: Buy {add_n}口 IOC 未成交，維持 {old_n}口"
                    )
                    logger.warning(msg)
                    self.notify_fn(msg)
                    result["add_contracts"] = 0
            else:
                exec_price = float(df["close"].iloc[-1])
                new_entry, entry_src = _reconcile_add_entry_price(
                    broker, "MXF", state, add_n, exec_price,
                )
                state.equity -= COST_PER_SIDE * add_n
                state.position += add_n
                state.contracts = state.position
                state.pyramided = True
                result["add_contracts"] = add_n
                result["entry_price"] = new_entry
                result["entry_price_source"] = entry_src
                _action_contracts = add_n

        elif sig.action == "hold" and state.position > 0:
            # Update trailing stop / highest_high
            curr_close = float(df["close"].iloc[-1])
            if state.highest_high is None or curr_close > state.highest_high:
                state.highest_high = curr_close

        self.state_mgr.save(state)

        msg = self._build_decision_message(
            sig=sig,
            state=state,
            indicators=display_ind,
            action_contracts=_action_contracts,
            closed_contracts=_closed_contracts,
            equity=equity,
            equity_src=equity_src,
            tsmc_signal=tsmc_signal,
            data_date=df.index[-1].date(),
            bar_meta=getattr(self, "today_bar_meta", None),
        )
        self.notify_fn(msg)
        return result

    # ------------------------------------------------------------------
    # Public: two-phase (night_open) — Phase 1
    # ------------------------------------------------------------------

    def run_signal(self, broker=None) -> dict:
        """14:30 phase: compute signal, save pending intent, send decision notification.

        Used when execution_timing="night_open".  Does NOT place orders.
        Returns signal dict with action/contracts/reason.
        """
        tsmc_signal = self._fetch_tsmc_signal() if self.enable_tsmc_signal else None
        df = self._load_data(broker=broker)
        if df is None or len(df) < 1:
            logger.error("No market data available.")
            return {"action": "error", "reason": "no data"}

        # Data freshness guard — delegates to the single source of truth.
        is_fresh, fresh_msg = self._check_data_freshness()
        if not is_fresh:
            self.notify_fn(fresh_msg)
            logger.warning("Data freshness check: %s", fresh_msg)

        # Loudly flag a decision that is about to be made on a stale bar
        # (previously a Shioaji outage silently degraded to T-1 data with
        # only a log line — the LINE message looked current but wasn't).
        last_bar_date = df.index[-1].date()
        data_stale = self._alert_if_stale(last_bar_date)

        # Broker reconcile BEFORE the signal so manual App operations and
        # exchange-side cash settlement are absorbed automatically instead
        # of requiring the stop-daemon / edit-JSON / restart manual SOP.
        if self.live and broker is not None:
            self.reconcile_state_with_broker(broker)

        state = self.state_mgr.load()
        # Capture the previous persisted equity BEFORE the live refresh — the
        # staleness check needs it as its baseline.
        prev_equity = float(state.equity or 0.0)
        # Cache the latest live equity into state when broker can serve it.
        equity, equity_src = _persist_live_equity(broker, state, self.state_mgr)
        # Plausibility-check the margin read (stale-mark guard, 7/28 follow-up).
        equity, _equity_note = self._sanity_check_equity(
            equity, equity_src, prev_equity, state, df,
        )

        # Advance the trailing high-water mark from today's close BEFORE the
        # signal is computed. Backtest trails on close (backtest/engine.py:258);
        # in night_open mode this used to never persist (run_signal didn't touch
        # it; run_execution only did when exec_price was passed, which prod does
        # not), so highest_high froze for the whole hold and the trailing stop
        # never ratcheted up. Persist it here every trading day.
        if state.position > 0:
            today_close = float(df["close"].iloc[-1])
            if state.highest_high is None or today_close > state.highest_high:
                state.highest_high = today_close

        display_ind = self._compute_display_indicators(df, state)

        sig = self.strategy.generate_signal(
            data=df,
            current_position=state.position,
            entry_price=state.entry_price,
            equity=equity,
            highest_high=state.highest_high,
            contracts=state.contracts,
            tsmc_signal=tsmc_signal,
        )

        # Live today-bar policy: entry fail-closed / exit fail-safe on an
        # unvalidated bar. May override sig (cancel buy/add, or force close
        # via the spot+basis proxy) — the pending intent below stores the
        # POST-policy signal so 15:05 executes the safe action.
        sig, _bar_lines = self._apply_today_bar_policy(sig, state, df, display_ind)

        # Save pending intent
        today_str = now_taipei().strftime("%Y-%m-%d")
        state.pending_action = sig.action
        state.pending_contracts = sig.contracts
        state.pending_signal_date = today_str
        state.pending_reason = sig.reason
        # The 14:30 gate's ATR, for the 15:05 min-only re-check — stored so the
        # re-check never re-fetches market data on the pre-order critical path.
        try:
            state.pending_atr = float(display_ind.get("atr") or 0.0) or None
        except (TypeError, ValueError):
            state.pending_atr = None
        self.state_mgr.save(state)

        msg = self._build_decision_message(
            sig=sig,
            state=state,
            indicators=display_ind,
            action_contracts=sig.contracts,
            closed_contracts=state.contracts,
            equity=equity,
            equity_src=equity_src,
            tsmc_signal=tsmc_signal,
            # run_signal hasn't executed yet → state.position is pre-add; show
            # the post-add total so "加碼至 N口" reflects current + add.
            target_position=state.position + sig.contracts if sig.action == "add" else None,
            data_date=last_bar_date,
            data_stale=data_stale,
            bar_meta=getattr(self, "today_bar_meta", None),
        )
        self.notify_fn(msg)

        return {
            "action": sig.action,
            "contracts": sig.contracts,
            "reason": sig.reason,
        }

    # ------------------------------------------------------------------
    # Public: two-phase (night_open) — Phase 2
    # ------------------------------------------------------------------

    def run_execution(
        self,
        broker=None,
        exec_price: float | None = None,
    ) -> dict:
        """15:05 phase: execute pending intent from run_signal() at night session price.

        Parameters
        ----------
        broker :
            Broker adapter (Shioaji/Fugle).  If None and self.live=True, a
            ShioajiAdapter is created automatically from env vars.
            If None and self.live=False, paper-trade simulation.
        exec_price :
            Night session execution price (15:00 bar open).
            If None, falls back to last daily close.
        """
        # Auto-create live broker if not supplied
        if broker is None and self.live:
            import os

            try:
                from dotenv import load_dotenv

                load_dotenv()
            except ImportError:
                pass
            api_key = os.environ.get("SHIOAJI_API_KEY", "")
            secret_key = os.environ.get("SHIOAJI_SECRET_KEY", "")
            if api_key and secret_key:
                from tw_futures.executor.shioaji_adapter import ShioajiAdapter

                broker = ShioajiAdapter(
                    api_key=api_key,
                    secret_key=secret_key,
                    simulation=not self.live,
                    cert_path=os.environ.get("SHIOAJI_CERT_PATH") or None,
                    cert_password=os.environ.get("SHIOAJI_CERT_PASSWORD") or None,
                    person_id=os.environ.get("SHIOAJI_PERSON_ID") or None,
                )
                logger.info("run_execution: live broker created (simulation=%s)", not self.live)
            else:
                logger.error("run_execution: live=True but SHIOAJI credentials missing")

        state = self.state_mgr.load()
        # Cache the latest live equity into state when broker can serve it.
        # Runs BEFORE any pending-action branch so even a "hold" cycle
        # refreshes the cached equity snapshot.
        _persist_live_equity(broker, state, self.state_mgr)

        # Stale-pending guard: a pending intent is only valid on the day its
        # signal was computed. If run_signal errored today (no data / crash)
        # yesterday's intent would otherwise survive and execute at today's
        # price — discard it loudly instead.
        if state.pending_action and state.pending_action != "hold":
            today_str = now_taipei().strftime("%Y-%m-%d")
            if state.pending_signal_date != today_str:
                msg = (
                    f"🔴 過期 pending 已丟棄: {state.pending_action} "
                    f"{state.pending_contracts}口 "
                    f"(signal_date={state.pending_signal_date}, today={today_str})"
                    f" — 不執行，等下一次 14:30 信號"
                )
                logger.error(msg)
                self.notify_fn(msg)
                state.pending_action = None
                state.pending_signal_date = None
                state.pending_contracts = 0
                state.pending_reason = None
                self.state_mgr.save(state)
                return {"action": "stale_pending_discarded"}

        # No pending action or explicit hold
        if (not state.pending_action) or state.pending_action == "hold":
            # Still update highest_high if in position
            if state.position > 0 and exec_price is not None:
                if state.highest_high is None or exec_price > state.highest_high:
                    state.highest_high = exec_price
                    self.state_mgr.save(state)
            # Clear pending
            state.pending_action = None
            state.pending_signal_date = None
            state.pending_contracts = 0
            state.pending_reason = None
            self.state_mgr.save(state)
            return {"action": "hold"}

        # Resolve execution price
        if exec_price is None:
            df = self._load_data(broker=broker)
            exec_price = float(df["close"].iloc[-1]) if (df is not None and len(df) > 0) else 0.0

        result: dict = {
            "action": state.pending_action,
            "contracts": state.pending_contracts,
            "exec_price": exec_price,
            "execution_timing": "night_open",
        }

        COST_PER_SIDE, TICK_VALUE = _load_execution_constants()

        if state.pending_action == "buy" and state.position == 0:
            # 15:05 sizing re-check (裁示 2026-08-05): 只降不升.
            n_exec = self._execution_sizing_recheck(state, state.pending_contracts)
            result["contracts"] = n_exec
            filled_n = n_exec
            if n_exec <= 0:
                msg = (
                    f"⚠️ 15:05 重檢 0口可進 — 跳過進場 "
                    f"(14:30 pending {state.pending_contracts}口)"
                )
                logger.warning(msg)
                self.notify_fn(msg)
                result["filled"] = 0
                result["skipped_by_recheck"] = True
            elif broker is not None:
                order = broker.place_order("MXF", "Buy", n_exec)
                result["order_id"] = order.get("order_id")
                exec_price = order.get("fill_price", exec_price)
                # Source of truth = broker actual, NOT the requested quantity.
                actual = _sync_position_from_broker(
                    broker, "MXF", n_exec, self.notify_fn,
                )
                filled_n = actual if actual is not None else 0
            if filled_n > 0:
                state.equity -= COST_PER_SIDE * filled_n
                state.position = filled_n
                state.entry_price = exec_price
                state.contracts = filled_n
                state.highest_high = exec_price
                state.entry_date = today_taipei().isoformat()
            elif n_exec > 0:
                msg = (
                    f"🔴 進場未成交: Buy {n_exec}口 IOC 未成交，維持空倉"
                )
                logger.warning(msg)
                self.notify_fn(msg)
                result["filled"] = 0

        elif state.pending_action == "close" and state.position > 0:
            closed_n = state.position
            is_settlement = "settlement" in (state.pending_reason or "")
            sell_ok = True
            if broker is not None:
                # Pre-sell guard: never sell more than the broker actually
                # holds. On settlement day the expired contract may already be
                # cash-settled (broker flat) and get_contract() would resolve
                # the Sell to the NEXT month — opening a naked short. A manual
                # App close leaves the same trap.
                broker_actual = _read_broker_long(broker, "MXF")
                if broker_actual == 0:
                    msg = (
                        f"⚠️ 平倉時 broker 已無多單（手動平倉或已結算）— 不下單，"
                        f"state {closed_n}口 已同步為空倉"
                    )
                    logger.warning(msg)
                    self.notify_fn(msg)
                    state.position = 0
                    state.entry_price = None
                    state.contracts = 0
                    state.highest_high = None
                    state.pyramided = False
                    state.entry_date = None
                    state.pending_action = None
                    state.pending_signal_date = None
                    state.pending_contracts = 0
                    state.pending_reason = None
                    self.state_mgr.save(state)
                    _persist_live_equity(broker, state, self.state_mgr)
                    result["action"] = "close_already_flat"
                    return result
                if broker_actual is not None and 0 < broker_actual < closed_n:
                    self.notify_fn(
                        f"⚠️ broker 只持有 {broker_actual}口（state {closed_n}口）— "
                        f"只平 {broker_actual}口"
                    )
                    closed_n = broker_actual
                try:
                    order = broker.place_order("MXF", "Sell", closed_n)
                except Exception as exc:
                    logger.error("Sell order failed: %s", exc)
                    order = {"order_id": "FAILED", "status": "Failed"}
                result["order_id"] = order.get("order_id")
                sell_status = order.get("status", "")
                if sell_status in ("Failed", "Cancelled", "Inactive"):
                    sell_ok = False
                    msg = f"🔴 結算日平倉失敗: Sell {closed_n}口 status={sell_status}"
                    logger.error(msg)
                    self.notify_fn(msg)
                else:
                    exec_price = order.get("fill_price", exec_price)
                    _reconcile_position(broker, 0, self.notify_fn)

            if sell_ok:
                pnl_pts = exec_price - (state.entry_price or 0.0)
                round_trip = COST_PER_SIDE * 2
                pnl_twd = pnl_pts * closed_n * TICK_VALUE - round_trip * closed_n
                state.equity += pnl_twd
                result["exit_price"] = exec_price
                result["pnl_twd"] = pnl_twd
                # Only flatten local state when the close actually filled. A
                # rejected Sell means the broker still holds the position —
                # zeroing here would desync state and risk a double position
                # (the system would think it's flat and buy again).
                state.position = 0
                state.entry_price = None
                state.contracts = 0
                state.highest_high = None
                state.pyramided = False
                state.entry_date = None

            # ── Settlement rollover: re-check entry immediately ───
            # Only proceed if sell was successful (not rejected by exchange)
            if is_settlement and sell_ok:
                df = self._load_data(broker=broker)
                if df is not None and len(df) > 0:
                    re_sig = self.strategy.generate_signal(
                        data=df,
                        current_position=0,
                        entry_price=None,
                        equity=state.equity,
                        highest_high=None,
                        contracts=0,
                        )
                    logger.info(
                        "Settlement rollover: equity=%.0f → signal=%s contracts=%d reason=%s",
                        state.equity, re_sig.action, re_sig.contracts, re_sig.reason,
                    )
                    if re_sig.action == "buy":
                        buy_n = re_sig.contracts
                        if broker is not None:
                            try:
                                buy_order = broker.place_order("MXF", "Buy", buy_n)
                            except Exception as exc:
                                logger.error("Rollover buy failed: %s", exc)
                                buy_order = {"order_id": "FAILED", "status": "Failed"}
                            buy_status = buy_order.get("status", "")
                            if buy_status in ("Failed", "Cancelled", "Inactive"):
                                msg = (
                                    f"🔴 結算日轉倉買單失敗: Buy {buy_n}口 "
                                    f"status={buy_status}"
                                )
                                logger.error(msg)
                                self.notify_fn(msg)
                                result["rollover"] = False
                                result["rollover_reason"] = f"buy order {buy_status}"
                            else:
                                buy_price = buy_order.get("fill_price", exec_price)
                                # Broker actual is source of truth (rollover from flat).
                                actual = _sync_position_from_broker(
                                    broker, "MXF", buy_n, self.notify_fn,
                                )
                                filled_n = actual if actual is not None else 0
                                if filled_n > 0:
                                    state.equity -= COST_PER_SIDE * filled_n
                                    state.position = filled_n
                                    state.entry_price = buy_price
                                    state.contracts = filled_n
                                    state.highest_high = buy_price
                                    state.entry_date = today_taipei().isoformat()
                                    result["rollover"] = True
                                    result["rollover_contracts"] = filled_n
                                    result["rollover_price"] = buy_price
                                else:
                                    msg = (
                                        f"🔴 結算日轉倉買單未成交: Buy {buy_n}口 "
                                        f"IOC 未成交，維持空倉"
                                    )
                                    logger.warning(msg)
                                    self.notify_fn(msg)
                                    result["rollover"] = False
                                    result["rollover_reason"] = "rollover buy IOC 未成交"
                        else:
                            buy_price = exec_price
                            state.equity -= COST_PER_SIDE * buy_n
                            state.position = buy_n
                            state.entry_price = buy_price
                            state.contracts = buy_n
                            state.highest_high = buy_price
                            state.entry_date = today_taipei().isoformat()
                            result["rollover"] = True
                            result["rollover_contracts"] = buy_n
                            result["rollover_price"] = buy_price
                    else:
                        result["rollover"] = False
                        result["rollover_reason"] = re_sig.reason
            elif is_settlement and not sell_ok:
                result["rollover"] = False
                result["rollover_reason"] = "sell order failed — rollover aborted"

        elif state.pending_action == "add" and state.position > 0:
            # 15:05 sizing re-check (裁示 2026-08-05): 只降不升, total-based.
            add_n = self._execution_sizing_recheck(
                state, state.pending_contracts, add_mode=True,
            )
            result["contracts"] = add_n
            old_n = state.position
            if add_n <= 0:
                msg = (
                    f"⚠️ 15:05 重檢 0口可加 — 跳過加碼 "
                    f"(14:30 pending {state.pending_contracts}口, 持倉 {old_n}口)"
                )
                logger.warning(msg)
                self.notify_fn(msg)
                result["add_contracts"] = 0
                result["skipped_by_recheck"] = True
            elif broker is not None:
                order = broker.place_order("MXF", "Buy", add_n)
                result["order_id"] = order.get("order_id")
                exec_price = order.get("fill_price", exec_price)
                # Source of truth = broker actual (the 2026-06-01 runaway fix).
                actual = _sync_position_from_broker(
                    broker, "MXF", old_n + add_n, self.notify_fn,
                )
                filled = max(0, actual - old_n) if actual is not None else 0
                if filled > 0:
                    # Recompute cost basis with the ACTUAL filled qty BEFORE
                    # mutating state.position (helper reads old_n = state.position).
                    new_entry, entry_src = _reconcile_add_entry_price(
                        broker, "MXF", state, filled, exec_price,
                    )
                    state.equity -= COST_PER_SIDE * filled
                    state.position = actual
                    state.contracts = actual
                    state.pyramided = True
                    result["add_contracts"] = filled
                    result["entry_price"] = new_entry
                    result["entry_price_source"] = entry_src
                else:
                    msg = f"🔴 加碼未成交: Buy {add_n}口 IOC 未成交，維持 {old_n}口"
                    logger.warning(msg)
                    self.notify_fn(msg)
                    result["add_contracts"] = 0
            else:
                new_entry, entry_src = _reconcile_add_entry_price(
                    broker, "MXF", state, add_n, exec_price,
                )
                state.equity -= COST_PER_SIDE * add_n
                state.position += add_n
                state.contracts = state.position
                state.pyramided = True
                result["add_contracts"] = add_n
                result["entry_price"] = new_entry
                result["entry_price_source"] = entry_src

        # Clear pending
        state.pending_action = None
        state.pending_signal_date = None
        state.pending_contracts = 0
        state.pending_reason = None
        state.pending_atr = None
        self.state_mgr.save(state)

        # Post-execution verify (15:10 hook): the order has filled and
        # _reconcile_position has run. Re-read live equity now so
        # state.equity reflects the post-fill margin balance rather
        # than the cost+pnl bookkeeping estimate.
        _persist_live_equity(broker, state, self.state_mgr)

        # LINE execution notification
        exec_time = now_taipei().strftime("%H:%M")
        action = result["action"]
        if result.get("skipped_by_recheck"):
            # No order was placed — a "BUY 0×MXF" line would read as a fill.
            action_desc = "SKIP（15:05 重檢 0口，未下單）"
        elif action == "buy":
            action_desc = f"BUY {result['contracts']}×MXF @ {exec_price:.0f}"
        elif action == "close":
            pnl = result.get("pnl_twd", 0)
            closed_n = result.get("contracts", 0)
            if result.get("rollover"):
                roll_n = result.get("rollover_contracts", 0)
                roll_px = result.get("rollover_price", exec_price)
                action_desc = (
                    f"🔄 結算日轉倉: CLOSE {closed_n}口 + BUY {roll_n}口 "
                    f"@ {roll_px:.0f}  PnL={pnl:+,.0f} NTD"
                )
            elif result.get("rollover") is False:
                reason = result.get("rollover_reason", "")
                action_desc = (
                    f"CLOSE {closed_n}×MXF @ {exec_price:.0f}  PnL={pnl:+,.0f} NTD\n"
                    f"結算日平倉，暫不進場（{reason}）"
                )
            else:
                action_desc = f"CLOSE {closed_n}×MXF @ {exec_price:.0f}  PnL={pnl:+.0f} NTD"
        elif action == "add":
            add_n = result.get("add_contracts", result.get("contracts", 0))
            action_desc = f"ADD {add_n}×MXF @ {exec_price:.0f}（加碼）"
        else:
            action_desc = action.upper()

        msg = f"\n━━━━━━━━━━━━\n動作: {action_desc}\n時間: {exec_time} (夜盤)\n━━━━━━━━━━━━"
        self.notify_fn(msg)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _alert_if_stale(self, last_bar_date) -> bool:
        """Live mode: loudly flag a decision made without today's bar.

        Returns True when today is a trading day but the latest bar is older —
        i.e. the Shioaji today-bar fetch failed and we silently degraded to
        T-1. On a settlement day this is escalated: the force-close trigger
        keys off the bar's date, so stale data can silently skip settlement
        handling entirely.
        """
        from src.strategy.v2b_engine import _is_settlement_day
        from src.utils.freshness import is_trading_day

        today = today_taipei()
        if not (self.live and is_trading_day(today) and last_bar_date != today):
            return False
        if _is_settlement_day(pd.Timestamp(today)):
            self.notify_fn(
                f"🔴 結算日但未取得今日({today})日盤資料，決策基於 {last_bar_date} — "
                f"結算平倉判斷可能失效，請人工確認部位"
            )
        else:
            self.notify_fn(
                f"⚠️ 未取得今日({today})日盤資料，決策基於 {last_bar_date} 收盤"
                f"（無資料、時戳語義無法判定或全遭驗證拒用 — 詳見 log 的"
                f" fetch_day_session_bar 判定行）"
            )
        return True

    def _apply_today_bar_policy(
        self, sig, state: TradingState, df: pd.DataFrame, indicators: dict,
    ):
        """Asymmetric fail-mode policy for an unvalidated live today-bar.

        * Entry fail-CLOSED — never open/scale a position on an unvalidated
          bar: buy/add is cancelled with a 🔴 alert.
        * Exit fail-SAFE — never silently HOLD through a stop: when the bar
          was unusable the trailing stop is evaluated on a spot+median-basis
          proxy; a clear break SELLs as normal, ambiguity/no-spot alerts 🔴
          for the human (2026-07-28: the truncated 43,175 held straight
          through the 42,673 stop while the market closed at 41,608).

        Returns (possibly-overridden sig, extra LINE lines). No-op when not
        live, when the meta seam is absent (tests patch _load_data), when
        today isn't a trading day, or when the bar is fully validated.
        """
        meta = getattr(self, "today_bar_meta", None)
        if not self.live or meta is None:
            return sig, []
        from src.utils.freshness import is_trading_day

        today = today_taipei()
        if not is_trading_day(today):
            return sig, []
        lines = [f"Bar來源: {meta.get('desc', '?')}"]
        if meta.get("validated"):
            return sig, lines

        if sig.action in ("buy", "add"):
            reason = (
                f"🔒 fail-closed: {sig.action} {sig.contracts}口 取消 — "
                f"當日bar未驗證 ({meta.get('reject')})"
            )
            self.notify_fn(f"🔴 進場封鎖(fail-closed): {reason}")
            logger.error(reason)
            return Signal("hold", 0, reason), lines

        if state.position > 0 and not meta.get("usable_for_exit"):
            verdict, detail = self._proxy_exit_check(df, state, indicators)
            lines.append(f"Proxy評估: {detail}")
            if verdict == "sell":
                reason = f"proxy trailing stop: {detail} (bar未驗證,以spot+basis代理評估)"
                logger.warning("today-bar policy: forcing close — %s", reason)
                return Signal("close", state.contracts or state.position, reason), lines
            if verdict == "hold":
                self.notify_fn(
                    f"⚠️ {today} 當日bar未驗證({meta.get('reject')})，"
                    f"proxy 未跌穿 stop（{detail}）— 維持持倉"
                )
            else:  # ambiguous
                self.notify_fn(
                    f"🔴 {today} 當日bar未驗證({meta.get('reject')}) 且 trailing "
                    f"未能評估（{detail}）— 需人工確認持倉！"
                )
        return sig, lines

    def _proxy_exit_check(
        self, df: pd.DataFrame, state: TradingState, indicators: dict,
    ) -> tuple[str, str]:
        """Evaluate the trailing stop via spot + rolling median basis.

        Returns (verdict, detail): verdict ∈ {"sell", "hold", "ambiguous"}.
        proxy_close = today's ^TWII spot + median(fut_close − spot) over the
        recent parquet bars. A break beyond ±_PROXY_AMBIG_BUFFER of the stop
        is decisive; inside the band (or missing inputs) → ambiguous → human.
        """
        import statistics

        try:
            from src.data.spot_ref import fetch_spot_close, fetch_spot_range

            spot_today = fetch_spot_close(today_taipei())
        except Exception as exc:
            return "ambiguous", f"spot(^TWII) 取得失敗: {exc}"
        if spot_today is None:
            return "ambiguous", "spot(^TWII) 不可得"

        tail = df.tail(_BASIS_LOOKBACK_BARS)
        try:
            spots = fetch_spot_range(tail.index[0].date(), tail.index[-1].date())
        except Exception:
            spots = {}
        pairs = [
            float(tail.loc[ts, "close"]) - spots[ts.date()]
            for ts in tail.index if ts.date() in spots
        ]
        if len(pairs) < _BASIS_MIN_PAIRS:
            return "ambiguous", f"basis 樣本不足 ({len(pairs)}<{_BASIS_MIN_PAIRS})"
        basis = statistics.median(pairs)
        proxy = spot_today + basis

        atr = indicators.get("atr") or 0.0
        if atr <= 0:
            try:
                atr = float(self.strategy._compute_indicators(df).iloc[-1]["atr"])
            except Exception:
                atr = 0.0
        hh = state.highest_high or state.entry_price
        if hh is None or atr <= 0:
            return "ambiguous", f"無 highest_high/ATR (hh={hh}, atr={atr:.0f})"
        stop = hh - self.strategy.trail_atr_mult * atr

        detail = (
            f"proxy={proxy:,.0f} (spot {spot_today:,.0f} + basis {basis:+,.0f}) "
            f"vs stop {stop:,.0f}"
        )
        if proxy < stop - _PROXY_AMBIG_BUFFER:
            return "sell", detail + " → 明確跌穿"
        if proxy > stop + _PROXY_AMBIG_BUFFER:
            return "hold", detail + " → 明確在上"
        return "ambiguous", detail + f" → 差距 < {_PROXY_AMBIG_BUFFER:.0f}pt"

    def _sanity_check_equity(
        self,
        equity: float,
        equity_src: str,
        prev_equity: float,
        state: TradingState,
        df: pd.DataFrame,
    ) -> tuple[float, str | None]:
        """Plausibility-check the broker margin equity read (7/28 follow-up).

        With a position on and a VALIDATED today close, the equity move since
        the last persisted read should roughly equal Δclose × contracts × 50.
        An implied-mark deviation beyond _EQUITY_SANITY_PTS points flags a
        stale margin read; sizing then uses the conservative (lower) value.
        Approximation note: prev_equity embeds the previous session's mark,
        so the 600pt tolerance absorbs normal overnight/basis noise.
        """
        meta = getattr(self, "today_bar_meta", None)
        tick = 50.0
        if (
            not self.live or equity_src != "即時" or state.position <= 0
            or state.contracts <= 0 or prev_equity <= 0
            or meta is None or not meta.get("validated") or len(df) < 2
        ):
            return equity, None
        today_close = float(df["close"].iloc[-1])
        prev_close = float(df["close"].iloc[-2])
        expected = prev_equity + (today_close - prev_close) * state.contracts * tick
        dev_pts = (equity - expected) / (state.contracts * tick)
        if abs(dev_pts) <= _EQUITY_SANITY_PTS:
            return equity, None
        conservative = min(equity, expected)
        note = (
            f"⚠️ margin equity 疑 stale: broker={equity:,.0f} vs "
            f"估算={expected:,.0f}（隱含 mark 偏離 {dev_pts:+.0f}pt）— "
            f"sizing 改用保守值 {conservative:,.0f}"
        )
        logger.warning(note)
        self.notify_fn(note)
        return conservative, note

    def _check_data_freshness(self) -> tuple[bool, str]:
        """Delegate to the single source of truth (src.utils.freshness).

        Checks the persisted parquet on disk (not the in-memory df, which has
        today's snapshot bar appended). Returns ``(is_fresh, msg)``.
        """
        from src.utils.freshness import check_parquet_freshness

        is_fresh, msg, _ = check_parquet_freshness(self.data_path)
        return is_fresh, msg

    def _compute_display_indicators(
        self,
        df: pd.DataFrame,
        state: TradingState,
    ) -> dict:
        """Extract EMA, ATR, bull_streak, trailing_stop for notification."""
        try:
            ind = self.strategy._compute_indicators(df)
            latest = ind.iloc[-1]
            close = float(latest["close"])
            ema_fast = float(latest["ema_fast"])
            ema_slow = float(latest["ema_slow"])
            atr_v = float(latest["atr"])
            cross = (ind["ema_fast"] > ind["ema_slow"]).astype(int)
            bull_streak = (
                int(cross.iloc[-self.strategy.confirm_days :].sum())
                if len(cross) >= self.strategy.confirm_days
                else 0
            )
            # Use bull_streak from indicators if available
            if "bull_streak" in ind.columns:
                bull_streak = int(latest["bull_streak"])
            trailing_stop: float | None = None
            if state.position > 0 and state.highest_high is not None:
                trailing_stop = state.highest_high - self.strategy.trail_atr_mult * atr_v
            return {
                "close": close,
                "ema_fast": ema_fast,
                "ema_slow": ema_slow,
                "atr": atr_v,
                "bull_streak": bull_streak,
                "confirm_days": self.strategy.confirm_days,
                "trailing_stop": trailing_stop,
            }
        except Exception as exc:
            logger.warning("_compute_display_indicators failed: %s", exc)
            return {}

    def _execution_sizing_recheck(
        self, state, n_pending: int, *, add_mode: bool = False,
    ) -> int:
        """15:05 只降不升 sizing re-check (裁示 2026-08-05).

        The pending count was gated at 14:30; equity may have moved by 15:05
        (the snapshot is refreshed by ``_persist_live_equity`` before this
        runs). Recompute the gate with the fresh equity and the SAME daily ATR
        (``state.pending_atr``, persisted at 14:30 — no market-data fetch here)
        and take ``min`` — never raise the count. ``add_mode`` gates the TOTAL
        position via ``_max_total_contracts`` (pyramid semantics). Degrades to
        a no-op (returns the pending count) on any failure — the 14:30 gate
        already vetted that count.
        """
        if n_pending <= 0:
            return n_pending
        try:
            # SAME daily ATR as the 14:30 gate, persisted with the pending
            # intent — the re-check must NEVER touch the market-data stack on
            # the pre-order critical path (no kbars/TAIFEX/spot fetches, no
            # 15:05 night-tick ATR contamination). Missing → no-op: the 14:30
            # gate already vetted this count.
            atr_v = float(state.pending_atr or 0.0)
            if atr_v <= 0:
                return n_pending
            if add_mode:
                max_total = self.strategy._max_total_contracts(state.equity, atr_v)
                n_re = max(0, max_total - state.position)
                note = f"total上限 {max_total}口 − 持倉 {state.position}口"
            else:
                n_re, note = self.strategy._risk_capped_contracts(
                    n_pending, state.equity, atr_v,
                )
            n_new = min(n_pending, n_re)
        except Exception as exc:
            logger.warning("15:05 重檢不可用 (%s) — 沿用 14:30 口數", exc)
            return n_pending
        if n_new < n_pending:
            label = "加碼" if add_mode else "進場"
            self.notify_fn(
                f"⚠️ 15:05 重檢降口數({label}): {n_pending}→{n_new}口 — "
                f"{note or 'risk-cap'}; 15:05 淨值 {state.equity:,.0f}"
            )
        return n_new

    def _build_decision_message(
        self,
        sig,
        state: TradingState,
        indicators: dict,
        action_contracts: int,
        closed_contracts: int,
        equity: float,
        tsmc_signal: TsmcSignal | None,
        equity_src: str = "估算",
        target_position: int | None = None,
        data_date=None,
        data_stale: bool = False,
        bar_meta: dict | None = None,
    ) -> str:
        """Build the rich LINE decision notification."""
        action = sig.action
        close = indicators.get("close", 0.0)
        ema_f = indicators.get("ema_fast", 0.0)
        ema_s = indicators.get("ema_slow", 0.0)
        atr = indicators.get("atr", 0.0)
        bull_streak = indicators.get("bull_streak", 0)
        confirm_days = indicators.get("confirm_days", self.strategy.confirm_days)
        trailing_stop = indicators.get("trailing_stop")

        if action == "buy":
            action_line = f"BUY {action_contracts}口"
        elif action == "add":
            # target = post-add total. run_signal passes it (state.position is
            # still pre-add there); run_daily leaves it None (already mutated).
            add_target = target_position if target_position is not None else state.position
            action_line = f"ADD {action_contracts}口（加碼至 {add_target}口）"
        elif action in ("close", "sell"):
            action_line = f"CLOSE {closed_contracts}口"
        elif state.position > 0:
            action_line = f"HOLD（維持 {state.position}口）"
        else:
            action_line = "HOLD（空倉）"

        streak_ok = bull_streak >= confirm_days
        streak_icon = "✅" if streak_ok else "⏳"
        streak_line = f"Bull Streak: {bull_streak}/{confirm_days} 日 {streak_icon}"

        if trailing_stop is not None:
            stop_line = f"Trailing Stop: {trailing_stop:,.0f}"
        else:
            stop_line = "Trailing Stop: —"

        tsmc_line = f"TSMC信號: {tsmc_signal}" if tsmc_signal else ""

        sep = "━━━━━━━━━━━━"
        now = now_taipei().strftime("%Y-%m-%d")

        # Unrealized PnL line (only when holding)
        pnl_line = ""
        if state.position > 0 and state.entry_price and close > 0:
            tick_val = 50.0  # MXF tick value
            unrealized = (close - state.entry_price) * state.position * tick_val
            unrealized_pct = (unrealized / equity * 100) if equity > 0 else 0.0
            pnl_icon = "🟢" if unrealized >= 0 else "🔴"
            pnl_line = (
                f"持倉損益: {pnl_icon} {unrealized:+,.0f} NTD "
                f"({unrealized_pct:+.1f}%)"
            )

        # Label the close with its bar date so a stale close is visibly stale
        # instead of masquerading as today's. The 「台指收盤」 label is reserved
        # for a VALIDATED day-session close — an unvalidated value is shown as
        # 參考價 so a wrong bar can never masquerade as the official close.
        if bar_meta is not None and not bar_meta.get("validated"):
            close_line = f"參考價(未驗證): {close:,.0f}"
        else:
            close_line = f"台指收盤: {close:,.0f}"
        if data_date is not None:
            close_line += f" ({data_date})"

        lines = [
            sep,
            f"📊 激進帳戶 {now} 決策  {self.decision_time}",
        ]
        if data_stale:
            lines.append(f"⚠️ 資料非今日（{data_date}），判斷可能過時")
        lines += [
            f"動作: {action_line}",
            close_line,
        ]
        if bar_meta is not None:
            lines.append(f"Bar來源: {bar_meta.get('desc', '?')}")
        if pnl_line:
            lines.append(pnl_line)
        lines += [
            f"EMA{self.strategy.ema_fast}/{self.strategy.ema_slow}: "
            f"{ema_f:,.0f}/{ema_s:,.0f}",
            f"ATR: {atr:.0f}",
            streak_line,
            stop_line,
        ]
        if tsmc_line:
            lines.append(tsmc_line)
        mode_str = "激進(LIVE)" if self.live else "激進(SIMULATION)"
        entry_str = (
            f" @ {state.entry_price:,.0f}" if state.entry_price else ""
        )
        lines += [
            sep,
            f"原因: {sig.reason}",
            f"淨值: {equity:,.0f} NTD ({equity_src})",
            f"持倉: {state.position}口{entry_str}  帳戶: {mode_str}",
            sep,
        ]
        return "\n".join(lines)

    def _fetch_tsmc_signal(self) -> TsmcSignal | None:
        try:
            prices = fetch_prices()
            if prices is None:
                return None
            return compute_signal(
                tsm_adr_change_pct=prices.tsm_change_pct,
                sox_change_pct=prices.sox_change_pct,
                taiex_level=_TAIEX_WEIGHT_LEVEL,
            )
        except Exception as exc:
            logger.warning("TSMC signal fetch failed — proceeding without bias: %s", exc)
            return None

    def _load_data(self, broker=None) -> pd.DataFrame | None:
        """Load daily OHLCV data.

        Parameters
        ----------
        broker :
            Optional ShioajiAdapter to reuse for fetching today's bar.
            Avoids creating a second connection that steals the session.

        Strategy:
          1. Load historical data from parquet (baseline).
          2. Try to append today's bar from Shioaji API.
          3. Log the data source and latest bar date.
          4. Fallback to parquet only if Shioaji is unavailable.
        """
        # Resolve data path — try primary, then fallbacks
        data_path = self.data_path
        if not data_path.exists():
            for fb in _REAL_DATA_FALLBACKS:
                if fb.exists():
                    data_path = fb
                    logger.info("_load_data: using fallback parquet path: %s", data_path)
                    break
            else:
                logger.error("Data file not found: %s (tried fallbacks too)", self.data_path)
                return None

        df = pd.read_parquet(data_path)

        # Normalise index to timezone-naive date
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.sort_index()

        # Try to fetch today's bar from Shioaji (live mode only; skip in simulation to avoid
        # multiple rapid connections that cause segfaults in the Shioaji C extension).
        # Pass broker so we reuse the existing connection instead of creating a second one.
        today_bar = _fetch_today_bar_shioaji(simulation=False, broker=broker) if self.live else None
        meta = today_bar.pop("_meta", None) if isinstance(today_bar, dict) else None
        if today_bar is not None:
            today_ts = pd.Timestamp(today_bar["date"])
            if today_ts not in df.index:
                # An INVALID bar (truncated feed / spot-gate failure) must never
                # enter the decision df — a wrong close silently defeats the
                # trailing stop (2026-07-28: 43,175 vs real 41,608 → HOLD).
                # meta None = legacy caller/test seam → keep old behavior.
                if meta is None or meta.get("usable_for_exit"):
                    row = pd.DataFrame(
                        [
                            {
                                "open": today_bar["open"],
                                "high": today_bar["high"],
                                "low": today_bar["low"],
                                "close": today_bar["close"],
                                "volume": today_bar.get("volume", 0),
                            }
                        ],
                        index=[today_ts],
                    )
                    df = pd.concat([df, row])
                    df = df.sort_index()
                    logger.info(
                        "_load_data: appended today's bar  date=%s  close=%.0f  (%s)",
                        today_ts.date(), today_bar["close"],
                        meta["desc"] if meta else "legacy",
                    )
                else:
                    logger.error(
                        "_load_data: today's bar REJECTED — %s  close=%.0f  (%s)",
                        meta.get("reject"), today_bar["close"], meta.get("desc"),
                    )
            else:
                # Bar already persisted by the (validated) updater path.
                meta = {
                    "source": "parquet", "validated": True, "usable_for_exit": True,
                    "reject": None, "desc": "parquet已含今日bar ✅",
                    "last_kbar": None, "spot_dev": None,
                }
                logger.info(
                    "_load_data: Shioaji bar already in parquet  date=%s",
                    today_ts.date(),
                )
        else:
            if self.live:
                meta = {
                    "source": None, "validated": False, "usable_for_exit": False,
                    "reject": "無今日bar(Shioaji不可用)", "desc": "無今日bar 🔴",
                    "last_kbar": None, "spot_dev": None,
                }
            logger.info(
                "_load_data: Shioaji unavailable — using parquet only  latest=%s",
                df.index[-1].date() if len(df) > 0 else "N/A",
            )

        # Live decision paths read this to apply the fail-closed / fail-safe
        # policy; None = non-live or legacy (patched) seam → policy skipped.
        self.today_bar_meta = meta if self.live else None
        return df


def _validate_today_bar(bar: dict, today, source: str) -> dict:
    """Dual validation of a live today-bar (kbars AND snapshot paths).

    (a) Completeness — kbars only: the LAST session kbar must fall in
        13:40–13:45 INCLUSIVE (settlement day: 13:25–13:30 inclusive — the
        official close prints in the 13:45/13:30 closing-auction bar). A feed
        that died mid-morning yields a truncated "close" that must never be
        treated as the day close.
    (b) Spot gate — both paths: |close − ^TWII spot| ≤ max(500, 1.6%×spot)
        (``basis_band_for``; the spot index has no night session, so it is
        night-proof truth).

    Returns a meta dict:
      validated       — passed BOTH checks → may be labeled 台指收盤, entries OK
      usable_for_exit — good enough to evaluate exits on (validated; or complete
                        kbars with spot unavailable; or snapshot passing spot)
      reject / desc   — human-readable status for alerts and the LINE message
    """
    from datetime import time as _dtime

    from src.strategy.v2b_engine import _is_settlement_day

    close = float(bar["close"])
    last_kbar = None
    complete = False
    if source == "kbars":
        try:
            ts = pd.Timestamp(bar.get("last_ts"))
            last_kbar = ts.strftime("%H:%M")
            t = ts.time()
            if _is_settlement_day(pd.Timestamp(today)):
                complete = _dtime(13, 25) <= t <= _dtime(13, 30)
            else:
                complete = _dtime(13, 40) <= t <= _dtime(13, 45)
        except (TypeError, ValueError):
            complete = False

    spot = spot_dev = None
    spot_ok: bool | None = None
    try:
        from src.data.spot_ref import basis_band_for, fetch_spot_close

        spot = fetch_spot_close(today)
        if spot is not None:
            spot_dev = close - spot
            # 裁示 2026-08-05: relative band max(500, 1.6%×spot) — the flat 500
            # rejected legitimate big-move day bars at 2026 index levels
            # (7/31: legit +3,402pt day, basis 559).
            spot_ok = abs(spot_dev) <= basis_band_for(spot)
    except Exception as exc:
        logger.warning("_validate_today_bar: spot gate unavailable (%s)", exc)

    # (c) Provenance gate — the 7/22-class small-gap night bar: a close that
    #     bit-equals today's TAIFEX 盤後 row (= last night's session, published
    #     by ~14:00) is a night value even when it sits INSIDE the spot band
    #     (7/22: night 44,957 only 131pt from spot). Degrades to False when
    #     TAIFEX hasn't published today's rows yet or the fetch fails.
    night_hit = False
    try:
        from src.data.daily_updater import _night_provenance

        night_hit = bool(_night_provenance(today, close))
    except Exception as exc:
        logger.warning("_validate_today_bar: provenance gate unavailable (%s)", exc)

    validated = bool(complete and spot_ok and not night_hit)
    usable_for_exit = (not night_hit) and (
        validated
        or (complete and spot_ok is None)
        or (source == "snapshot" and spot_ok is True)
    )

    reject = None
    if not validated:
        parts = []
        if night_hit:
            parts.append("close==TAIFEX盤後值(provenance)")
        if source == "kbars" and not complete:
            parts.append(f"kbars截斷(末根 {last_kbar or '?'})")
        if source == "snapshot":
            parts.append("snapshot非日盤kbar")
        if spot_ok is False:
            parts.append(f"spot偏離 {spot_dev:+.0f}pt")
        elif spot_ok is None:
            parts.append("spot不可得")
        reject = "、".join(parts) or "未驗證"

    spot_str = f"spot誤差{spot_dev:+.0f}pt" if spot_dev is not None else "spot n/a"
    if night_hit:
        spot_str += " ⛔盤後值"
    icon = "✅" if validated else ("⚠️" if usable_for_exit else "🔴")
    desc = f"{source}(末根 {last_kbar}) {spot_str} {icon}" if source == "kbars" else \
           f"{source} {spot_str} {icon}"

    return {
        "source": source, "last_kbar": last_kbar, "complete": complete,
        "spot": spot, "spot_dev": spot_dev, "spot_ok": spot_ok,
        "night_hit": night_hit,
        "validated": validated, "usable_for_exit": usable_for_exit,
        "reject": reject, "desc": desc,
    }


def _fetch_today_bar_shioaji(simulation: bool = True, broker=None) -> dict | None:
    """Fetch today's day-session daily bar from Shioaji.

    Delegates the kbars fetch + day-session filtering + aggregation to the one
    authoritative fetcher (``src.data.shioaji_fetcher.fetch_day_session_bar``);
    falls back to a snapshot (last-traded price) only if that returns nothing.

    Uses an existing *broker* (ShioajiAdapter) when provided so that a
    second login does not steal the caller's session.

    Returns a dict with open/high/low/close/volume/date, or None on failure.
    """
    import os

    # Reuse existing broker connection if available
    owns_adapter = broker is None
    if owns_adapter:
        api_key = os.environ.get("SHIOAJI_API_KEY", "")
        secret_key = os.environ.get("SHIOAJI_SECRET_KEY", "")
        if not api_key or not secret_key:
            return None

        from tw_futures.executor.shioaji_adapter import ShioajiAdapter

        try:
            broker = ShioajiAdapter(
                api_key=api_key,
                secret_key=secret_key,
                simulation=simulation,
                cert_path=os.environ.get("SHIOAJI_CERT_PATH") or None,
                cert_password=os.environ.get("SHIOAJI_CERT_PASSWORD") or None,
                person_id=os.environ.get("SHIOAJI_PERSON_ID") or None,
            )
        except Exception as exc:
            logger.debug("_fetch_today_bar_shioaji: broker creation failed: %s", exc)
            return None

    # Taipei calendar date — the VM's local date can lag a day (UTC) between
    # 00:00–08:00 Taipei, which would make Shioaji return the wrong session.
    today = today_taipei()

    # --- Primary: authoritative day-session bar (single source of truth) ---
    try:
        from src.data.shioaji_fetcher import fetch_day_session_bar

        contract = broker.get_contract("MXF")
        bar = fetch_day_session_bar(broker._api, contract, today)
        if bar is not None:
            meta = _validate_today_bar(bar, today, source="kbars")
            logger.info(
                "_fetch_today_bar_shioaji: kbars close=%.0f  %s",
                bar["close"], meta["desc"],
            )
            if owns_adapter:
                broker.logout()
            return {"date": str(today), **bar, "_meta": meta}
        logger.debug("_fetch_today_bar_shioaji: no day-session bar for %s", today)
    except Exception as exc:
        logger.warning("_fetch_today_bar_shioaji: kbars failed: %s — trying snapshot", exc)

    # --- Fallback: snapshot (last traded price — may differ from 13:44 close) ---
    try:
        snap = broker.get_snapshots("MXF")
        close = float(snap["close"])
        logger.warning(
            "_fetch_today_bar_shioaji: using SNAPSHOT fallback  close=%.0f "
            "(⚠️ may not be 13:44 day-session close)",
            close,
        )
        result = {
            "date": str(today),
            "open": float(snap.get("open", close)),
            "high": float(snap.get("high", close)),
            "low": float(snap.get("low", close)),
            "close": close,
            "volume": int(snap.get("total_volume", 0)),
            "_source": "snapshot",  # flag for callers to detect fallback
        }
        result["_meta"] = _validate_today_bar(result, today, source="snapshot")
        if owns_adapter:
            broker.logout()
        return result
    except Exception as snap_exc:
        logger.debug("_fetch_today_bar_shioaji: snapshot also failed: %s", snap_exc)

    if owns_adapter:
        broker.logout()
    return None


def _query_live_equity(broker, fallback_equity: float) -> tuple[float, str]:
    """Query real-time equity from broker API.

    Returns (equity_value, source_label).
    Uses broker.get_account() → margin.equity (balance + unrealized PnL).
    Falls back to state estimate if broker unavailable or returns 0.
    """
    if broker is None:
        return fallback_equity, "估算"

    try:
        acct = broker.get_account()
        equity = float(acct.get("equity", 0))
        if equity > 0:
            logger.info("Live equity from broker: %.0f", equity)
            return equity, "即時"
    except Exception as exc:
        logger.warning("get_account() failed: %s — using state estimate", exc)

    return fallback_equity, "估算"


def _query_broker_avg_price(broker, product: str, expected_contracts: int) -> float | None:
    """Read the broker's weighted-average entry price for *product* after
    a fill. Returns the value when the position size matches the expected
    contract count (sanity check). Returns None on any failure / mismatch
    so callers can fall back to a local weighted-average computation.

    Used by the Anti-Martingale add path: after pyramiding, the broker's
    book carries the true blended cost basis. Without writing it back
    into state.entry_price, the LINE PnL line keeps using the pre-add
    average and overstates float profit by (broker_avg - old_avg) × N.
    """
    if broker is None:
        return None
    try:
        positions = broker.get_positions()
    except Exception as exc:
        logger.warning("get_positions() failed during add reconcile: %s", exc)
        return None

    # MXF contract codes look like "MXFE5"; match by prefix so the futures
    # roll doesn't break the lookup.
    candidates = [
        p for p in positions
        if str(p.get("code", "")).startswith(product)
        and p.get("direction", "Buy") == "Buy"
    ]
    if not candidates:
        return None
    # Sum across multiple position rows if Shioaji split them (shouldn't
    # happen for a single long, but be defensive).
    total_contracts = sum(int(p.get("contracts", p.get("quantity", 0))) for p in candidates)
    if total_contracts != expected_contracts:
        logger.warning(
            "broker avg-price lookup: position size mismatch (expected=%d, broker=%d)",
            expected_contracts, total_contracts,
        )
        return None
    # Weighted average of the candidate rows (only one in practice).
    total_qty = 0
    total_value = 0.0
    for p in candidates:
        q = int(p.get("contracts", p.get("quantity", 0)))
        px = float(p.get("avg_price", 0))
        if q <= 0 or px <= 0:
            return None
        total_qty += q
        total_value += px * q
    if total_qty <= 0:
        return None
    return total_value / total_qty


def _reconcile_add_entry_price(
    broker,
    product: str,
    state: TradingState,
    add_n: int,
    fill_price: float,
) -> tuple[float, str]:
    """Recompute the post-add weighted entry price for *state* and write
    it back. Prefers the broker's reported average (source-of-truth) and
    falls back to a local (old_avg * old_n + fill * add_n) / new_n when
    the broker read fails.

    Returns (new_entry_price, source_label) for logging / notifications.
    Must be called BEFORE state.position is mutated for the add.
    """
    old_n = state.position
    new_n = old_n + add_n

    broker_avg = _query_broker_avg_price(broker, product, new_n) if broker is not None else None
    if broker_avg is not None and broker_avg > 0:
        old_entry = state.entry_price
        state.entry_price = broker_avg
        logger.info(
            "add reconcile: state.entry_price updated from broker avg = %.2f (was %s, +%d口)",
            broker_avg, f"{old_entry:.2f}" if old_entry else "None", add_n,
        )
        return broker_avg, "broker"

    if state.entry_price is None or state.entry_price <= 0 or old_n <= 0:
        # First-fill edge case (shouldn't happen on an add path, but
        # defensive): fall back to the fill price.
        state.entry_price = fill_price
        return fill_price, "fill"

    local_avg = (state.entry_price * old_n + fill_price * add_n) / new_n
    logger.info(
        "add reconcile: broker avg unavailable, local weighted avg = %.2f "
        "(old=%.2f × %d, fill=%.2f × %d)",
        local_avg, state.entry_price, old_n, fill_price, add_n,
    )
    state.entry_price = local_avg
    return local_avg, "local"


def _persist_live_equity(broker, state: TradingState, state_mgr) -> tuple[float, str]:
    """Read live equity AND cache it in state when the read succeeds.

    Stale value is preserved when the broker is unavailable or returns 0
    -- callers see (state.equity, "估算") in that case. Used by every
    entry point that already holds a broker handle (run_signal,
    run_execution start, post-execution verify, daily_health_check) so
    the on-disk state.equity is always the most recent successful real-
    time snapshot rather than the strategy's bookkeeping estimate.
    """
    equity, src = _query_live_equity(broker, state.equity)
    if src == "即時" and equity > 0:
        state.equity = equity
        state_mgr.save(state)
        logger.info("state.equity updated from live broker read: %.0f", equity)
    return equity, src


def _read_broker_long(broker, product: str) -> int | None:
    """Broker's actual total LONG contracts for *product*, or None when the
    read fails. Filters direction=Buy and matches contract code by prefix so
    other products / short legs never pollute the count (the old
    ``_reconcile_position`` summed everything, shorts included)."""
    try:
        positions = broker.get_positions()
    except Exception as exc:
        logger.warning("broker position read failed: %s", exc)
        return None
    total = 0
    for p in positions:
        if p.get("direction", "Buy") != "Buy":
            continue
        code = str(p.get("code", ""))
        if code and not code.startswith(product):
            continue
        total += int(p.get("contracts", p.get("quantity", 0)))
    return total


def _reconcile_position(broker, expected_contracts: int, notify_fn, product: str = "MXF") -> None:
    """Wait 2s then verify broker position matches expected contracts.

    Non-blocking: logs confirmation or sends LINE alert on mismatch. Used by the
    Sell/close paths (expected=0) where the order status already gates state;
    the Buy/add/rollover paths use ``_sync_position_from_broker`` instead.
    """
    import time as _time

    _time.sleep(2)
    actual = _read_broker_long(broker, product)
    if actual is None:
        logger.warning("Position reconciliation failed: broker unreadable")
        return
    if actual == expected_contracts:
        logger.info(
            "Position reconciliation OK: expected=%d, actual=%d",
            expected_contracts, actual,
        )
    else:
        msg = f"⚠️ 持倉不一致: 預期 {expected_contracts} 口, 實際 {actual} 口"
        logger.warning(msg)
        notify_fn(msg)


def _sync_position_from_broker(
    broker,
    product: str,
    expected_total: int,
    notify_fn,
    *,
    sleep=None,
) -> int | None:
    """Read the broker's ACTUAL total long position after an order and return
    it as the source of truth — never trust the requested quantity.

    This is the fix for the 2026-06-01 runaway: a 15-lot IOC market add never
    filled, yet the local state was blindly bumped (state.position += add_n) to
    45 lots while the broker held 20. An IOC/MKT order in the night session
    routinely fills partially or not at all; the broker book is the only
    authority for "what actually filled".

    Timing (per ops decision): wait 2s for the fill report, read; if the read
    is BELOW expected (the fill report can lag), retry ONCE after 3s and take
    the later read. Never optimistically inflate.

    Returns the broker's actual long contracts for *product* (int), or None when
    the broker read failed on both attempts. On None the caller MUST leave local
    state unchanged (conservative: never bump on an unverified order); a loud
    LINE alert is sent here so the operator can reconcile manually.
    """
    import time as _time

    _sleep = sleep or _time.sleep

    def _read() -> int | None:
        return _read_broker_long(broker, product)

    _sleep(2)
    actual = _read()
    if actual is not None and actual < expected_total:
        # Fill report may lag — give it one more read before trusting a low value.
        _sleep(3)
        retry = _read()
        if retry is not None:
            actual = retry

    if actual is None:
        msg = (
            f"🔴 部位同步失敗: broker 部位讀取失敗，state 未更新 "
            f"(預期 {expected_total}口)，請立即手動核對永豐部位"
        )
        logger.error(msg)
        notify_fn(msg)
        return None

    if actual != expected_total:
        logger.warning(
            "position sync: expected=%d but broker=%d (IOC not fully filled) — "
            "state will follow broker truth", expected_total, actual,
        )
        notify_fn(
            f"⚠️ 部位以 broker 為準: 預期 {expected_total}口, 實際 broker={actual}口"
            f"（IOC 未全部成交，已採實際值）"
        )
    else:
        logger.info("position sync OK: broker=%d (matches expected)", actual)
    return actual


def _load_execution_constants() -> tuple[float, float]:
    """Return (COST_PER_SIDE, TICK_VALUE) from backtest engine."""
    try:
        from src.backtest.engine import COST_PER_SIDE, TICK_VALUE

        return COST_PER_SIDE, TICK_VALUE
    except ImportError:
        return 160.0, 50.0

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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.signals.fetcher import fetch_prices
from src.signals.tsmc_tracker import TsmcSignal, compute_signal
from src.state.state_manager import StateManager, TradingState
from src.strategy.v2b_engine import V2bEngine

logger = logging.getLogger(__name__)

_REAL_DATA = Path("data/MXF_Daily_Clean_2020_to_now.parquet")
# Fallback paths in priority order
_REAL_DATA_FALLBACKS = [
    Path("data/MXF_Daily_Clean_2020_to_now.parquet"),
    Path.home() / "trading-agents-v2" / "data" / "MXF_Daily_Clean_2020_to_now.parquet",
]
_TAIEX_WEIGHT_LEVEL = 22_000


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
    # Public: single-phase (next_open)
    # ------------------------------------------------------------------

    def run_daily(self, broker=None) -> dict:
        """Run one full daily cycle.

        Returns a summary dict with action, reason, and optional tsmc info.
        """
        tsmc_signal = self._fetch_tsmc_signal() if self.enable_tsmc_signal else None
        df = self._load_data()
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
            if broker is not None:
                order = broker.place_order("MXF", "Buy", sig.contracts)
                result["order_id"] = order.get("order_id")
                exec_price = order.get("fill_price", float(df["close"].iloc[-1]))
                _reconcile_position(broker, sig.contracts, self.notify_fn)
            else:
                exec_price = float(df["close"].iloc[-1])
            state.equity -= COST_PER_SIDE * sig.contracts
            state.position = sig.contracts
            state.entry_price = exec_price
            state.contracts = sig.contracts
            state.highest_high = exec_price
            result["entry_price"] = exec_price
            _action_contracts = sig.contracts

        elif sig.action in ("close", "sell") and state.position > 0:
            closed_n = state.position
            is_settlement = "settlement" in sig.reason
            if broker is not None:
                order = broker.place_order("MXF", "Sell", closed_n)
                result["order_id"] = order.get("order_id")
                exec_price = order.get("fill_price", float(df["close"].iloc[-1]))
                _reconcile_position(broker, 0, self.notify_fn)
            else:
                exec_price = float(df["close"].iloc[-1])
            pnl_pts = exec_price - (state.entry_price or 0.0)
            round_trip = COST_PER_SIDE * 2
            pnl_twd = pnl_pts * closed_n * TICK_VALUE - round_trip * closed_n
            state.equity += pnl_twd
            result["exit_price"] = exec_price
            result["pnl_twd"] = pnl_twd
            _closed_contracts = closed_n
            # Reset position state
            state.position = 0
            state.entry_price = None
            state.contracts = 0
            state.highest_high = None
            state.pyramided = False

            # Settlement rollover: re-check entry immediately
            if is_settlement:
                re_sig = self.strategy.generate_signal(
                    data=df,
                    current_position=0,
                    entry_price=None,
                    equity=state.equity,
                    highest_high=None,
                    contracts=0,
                    tsmc_signal=tsmc_signal,
                )
                if re_sig.action == "buy":
                    buy_n = re_sig.contracts
                    if broker is not None:
                        buy_order = broker.place_order("MXF", "Buy", buy_n)
                        buy_price = buy_order.get("fill_price", exec_price)
                        _reconcile_position(broker, buy_n, self.notify_fn)
                    else:
                        buy_price = exec_price
                    state.equity -= COST_PER_SIDE * buy_n
                    state.position = buy_n
                    state.entry_price = buy_price
                    state.contracts = buy_n
                    state.highest_high = buy_price
                    _action_contracts = buy_n
                    result["rollover"] = True
                    result["rollover_contracts"] = buy_n
                else:
                    result["rollover"] = False
                    result["rollover_reason"] = re_sig.reason

        elif sig.action == "add" and state.position > 0:
            add_n = sig.contracts
            if broker is not None:
                order = broker.place_order("MXF", "Buy", add_n)
                result["order_id"] = order.get("order_id")
                exec_price = order.get("fill_price", float(df["close"].iloc[-1]))
                _reconcile_position(broker, state.position + add_n, self.notify_fn)
            else:
                exec_price = float(df["close"].iloc[-1])
            state.equity -= COST_PER_SIDE * add_n
            state.position += add_n
            state.contracts = state.position
            state.pyramided = True
            result["add_contracts"] = add_n
            result["entry_price"] = exec_price
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
        df = self._load_data()
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

        # Save pending intent
        tz_cst = timezone(timedelta(hours=8))
        today_str = datetime.now(tz=tz_cst).strftime("%Y-%m-%d")
        state.pending_action = sig.action
        state.pending_contracts = sig.contracts
        state.pending_signal_date = today_str
        state.pending_reason = sig.reason
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
            df = self._load_data()
            exec_price = float(df["close"].iloc[-1]) if (df is not None and len(df) > 0) else 0.0

        result: dict = {
            "action": state.pending_action,
            "contracts": state.pending_contracts,
            "exec_price": exec_price,
            "execution_timing": "night_open",
        }

        COST_PER_SIDE, TICK_VALUE = _load_execution_constants()

        if state.pending_action == "buy" and state.position == 0:
            if broker is not None:
                order = broker.place_order("MXF", "Buy", state.pending_contracts)
                result["order_id"] = order.get("order_id")
                exec_price = order.get("fill_price", exec_price)
                _reconcile_position(broker, state.pending_contracts, self.notify_fn)
            state.equity -= COST_PER_SIDE * state.pending_contracts
            state.position = state.pending_contracts
            state.entry_price = exec_price
            state.contracts = state.pending_contracts
            state.highest_high = exec_price

        elif state.pending_action == "close" and state.position > 0:
            closed_n = state.position
            is_settlement = "settlement" in (state.pending_reason or "")
            if broker is not None:
                order = broker.place_order("MXF", "Sell", closed_n)
                result["order_id"] = order.get("order_id")
                exec_price = order.get("fill_price", exec_price)
                _reconcile_position(broker, 0, self.notify_fn)
            pnl_pts = exec_price - (state.entry_price or 0.0)
            round_trip = COST_PER_SIDE * 2
            pnl_twd = pnl_pts * closed_n * TICK_VALUE - round_trip * closed_n
            state.equity += pnl_twd
            result["exit_price"] = exec_price
            result["pnl_twd"] = pnl_twd
            state.position = 0
            state.entry_price = None
            state.contracts = 0
            state.highest_high = None
            state.pyramided = False

            # ── Settlement rollover: re-check entry immediately ───
            if is_settlement:
                df = self._load_data()
                if df is not None and len(df) > 0:
                    re_sig = self.strategy.generate_signal(
                        data=df,
                        current_position=0,
                        entry_price=None,
                        equity=state.equity,
                        highest_high=None,
                        contracts=0,
                        )
                    if re_sig.action == "buy":
                        buy_n = re_sig.contracts
                        if broker is not None:
                            buy_order = broker.place_order("MXF", "Buy", buy_n)
                            buy_price = buy_order.get("fill_price", exec_price)
                            _reconcile_position(broker, buy_n, self.notify_fn)
                        else:
                            buy_price = exec_price
                        state.equity -= COST_PER_SIDE * buy_n
                        state.position = buy_n
                        state.entry_price = buy_price
                        state.contracts = buy_n
                        state.highest_high = buy_price
                        result["rollover"] = True
                        result["rollover_contracts"] = buy_n
                        result["rollover_price"] = buy_price
                    else:
                        result["rollover"] = False
                        result["rollover_reason"] = re_sig.reason

        elif state.pending_action == "add" and state.position > 0:
            add_n = state.pending_contracts
            if broker is not None:
                order = broker.place_order("MXF", "Buy", add_n)
                result["order_id"] = order.get("order_id")
                exec_price = order.get("fill_price", exec_price)
                _reconcile_position(broker, state.position + add_n, self.notify_fn)
            state.equity -= COST_PER_SIDE * add_n
            state.position += add_n
            state.contracts = state.position
            state.pyramided = True
            result["add_contracts"] = add_n

        # Clear pending
        state.pending_action = None
        state.pending_signal_date = None
        state.pending_contracts = 0
        state.pending_reason = None
        self.state_mgr.save(state)

        # LINE execution notification
        tz_cst = timezone(timedelta(hours=8))
        exec_time = datetime.now(tz=tz_cst).strftime("%H:%M")
        action = result["action"]
        if action == "buy":
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
            action_line = f"ADD {action_contracts}口（加碼至 {state.position}口）"
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
        tz_cst = timezone(timedelta(hours=8))
        now = datetime.now(tz=tz_cst).strftime("%Y-%m-%d")

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

        lines = [
            sep,
            f"📊 激進帳戶 {now} 決策  {self.decision_time}",
            f"動作: {action_line}",
            f"台指收盤: {close:,.0f}",
        ]
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

    def _load_data(self) -> pd.DataFrame | None:
        """Load daily OHLCV data.

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
        # multiple rapid connections that cause segfaults in the Shioaji C extension)
        today_bar = _fetch_today_bar_shioaji(simulation=False) if self.live else None
        if today_bar is not None:
            today_ts = pd.Timestamp(today_bar["date"])
            if today_ts not in df.index:
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
                    "_load_data: appended today's bar from Shioaji  date=%s  close=%.0f",
                    today_ts.date(),
                    today_bar["close"],
                )
            else:
                logger.info(
                    "_load_data: Shioaji bar already in parquet  date=%s",
                    today_ts.date(),
                )
        else:
            logger.info(
                "_load_data: Shioaji unavailable — using parquet only  latest=%s",
                df.index[-1].date() if len(df) > 0 else "N/A",
            )

        return df


def _fetch_today_bar_shioaji(simulation: bool = True) -> dict | None:
    """Try to fetch today's day-session daily bar from Shioaji API.

    Uses ShioajiAdapter (which correctly loads contracts) then aggregates
    1-minute kbars for the day session (08:45–13:45 Asia/Taipei) into a
    single OHLCV bar.

    Returns a dict with open/high/low/close/volume/date, or None on failure.
    """
    try:
        import os
        from datetime import date, time

        import pandas as pd

        api_key = os.environ.get("SHIOAJI_API_KEY", "")
        secret_key = os.environ.get("SHIOAJI_SECRET_KEY", "")
        if not api_key or not secret_key:
            return None

        from tw_futures.executor.shioaji_adapter import ShioajiAdapter

        adapter = ShioajiAdapter(
            api_key=api_key,
            secret_key=secret_key,
            simulation=simulation,
            cert_path=os.environ.get("SHIOAJI_CERT_PATH") or None,
            cert_password=os.environ.get("SHIOAJI_CERT_PASSWORD") or None,
            person_id=os.environ.get("SHIOAJI_PERSON_ID") or None,
        )

        today = date.today()
        # Use tomorrow as end so Shioaji includes all of today's bars
        tomorrow = pd.Timestamp(today) + pd.Timedelta(days=1)

        contract = adapter.get_contract("MXF")
        kbars = adapter._api.kbars(
            contract,
            start=str(today),
            end=str(tomorrow.date()),
            timeout=15_000,
        )
        adapter.logout()

        if not kbars or len(kbars.ts) == 0:
            logger.debug("_fetch_today_bar_shioaji: empty kbars for %s", today)
            return None

        df = pd.DataFrame(
            {
                "ts": kbars.ts,
                "open": kbars.Open,
                "high": kbars.High,
                "low": kbars.Low,
                "close": kbars.Close,
                "volume": kbars.Volume,
            }
        )
        df["ts"] = pd.to_datetime(df["ts"], unit="ns", utc=True).dt.tz_convert("Asia/Taipei")

        # Sort by timestamp to ensure first/last are chronologically correct
        df = df.sort_values("ts")

        # Keep only day-session bars: 08:45–13:44 (< 13:45 excludes settlement bar)
        day_open = time(8, 45)
        day_close = time(13, 45)
        today_mask = (
            (df["ts"].dt.date == today)
            & (df["ts"].dt.time >= day_open)
            & (df["ts"].dt.time < day_close)
        )
        day_df = df[today_mask]

        if day_df.empty:
            logger.debug(
                "_fetch_today_bar_shioaji: no day-session bars for %s (got %d total bars)",
                today,
                len(df),
            )
            return None

        return {
            "date": str(today),
            "open": float(day_df.iloc[0]["open"]),
            "high": float(day_df["high"].max()),
            "low": float(day_df["low"].min()),
            "close": float(day_df.iloc[-1]["close"]),
            "volume": int(day_df["volume"].sum()),
        }
    except Exception as exc:
        logger.debug("_fetch_today_bar_shioaji failed: %s", exc)
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


def _reconcile_position(broker, expected_contracts: int, notify_fn) -> None:
    """Wait 2s then verify broker position matches expected contracts.

    Non-blocking: logs confirmation or sends LINE alert on mismatch.
    """
    import time as _time

    _time.sleep(2)
    try:
        positions = broker.get_positions()
        actual = sum(p.get("contracts", p.get("quantity", 0)) for p in positions)
        if actual == expected_contracts:
            logger.info(
                "Position reconciliation OK: expected=%d, actual=%d",
                expected_contracts, actual,
            )
        else:
            msg = f"⚠️ 持倉不一致: 預期 {expected_contracts} 口, 實際 {actual} 口"
            logger.warning(msg)
            notify_fn(msg)
    except Exception as exc:
        logger.warning("Position reconciliation failed: %s", exc)


def _load_execution_constants() -> tuple[float, float]:
    """Return (COST_PER_SIDE, TICK_VALUE) from backtest engine."""
    try:
        from src.backtest.engine import COST_PER_SIDE, TICK_VALUE

        return COST_PER_SIDE, TICK_VALUE
    except ImportError:
        return 160.0, 50.0
